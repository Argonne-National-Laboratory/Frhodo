"""Smurf coarse global stage: Sensitivity Multistart Rate Fitting.

A domain-specific global-stage optimizer. Each iteration attributes the
per-experiment residual across reactions using the time-resolved
multiplier sensitivity (a small weighted least-squares per experiment),
fits each reaction's per-experiment log-multiplier cloud across the
campaign's temperature spread to an Arrhenius-basis shape model, and
takes a Levenberg-Marquardt-damped step scored by the true objective.
It reaches the optimum's neighborhood in a handful of sensitivity
sweeps and hands off to the local stage; it is not run to convergence.

The step direction is built from the objective's own per-shock residual
and adaptive weights, and each reaction's cloud contribution is weighted
by how much that shock's loss moves the campaign aggregate — so the
step descends the true aggregate, not a per-experiment proxy.
"""
import numpy as np
import cantera as ct
from scipy.stats import qmc

from frhodo.optimize.cost.aggregation import coverage_weights
from frhodo._vendor.opendsm.adaptive_loss import adaptive_weights
from frhodo.optimize.cost.fit_fcn import (
    calculate_residuals,
    pool_worker_mech,
    update_mech_coef_opt,
)
from frhodo.optimize.screening import _scaled_residual
from frhodo.optimize.shock_prep import shock_sigma_bar
from frhodo.simulation.shock.sensitivity import (
    compute_sensitivity,
    sensitivity_observable,
)



_RU = float(ct.gas_constant)
# Ridge on the per-experiment multiplier solve, relative to the Gram
# trace; holds collinear (unidentifiable) reaction pairs at the
# minimum-norm split.
_RIDGE = 1e-2
_DAMPING = (1.0, 0.5, 0.25, 0.1, 0.03, 0.01)


def _linearize_shock(mech, reactor_state, scale, var_dict, coef_opt, x,
                     shock, target_idx):
    """``(loss, (A, resid, w, T2, P))`` for one shock from the
    objective's own residual, or ``(loss, None)`` when it cannot be
    linearized.

    ``A[i, k] = chain_i · ∂ln(obs_i)/∂ln(k_k)``; solving ``resid ≈ A·δ``
    gives the per-reaction log-multipliers that flatten this experiment.
    Parameter-explicit so it runs identically on the master or inside a
    pool worker (with the worker's mechanism).
    """
    out = calculate_residuals(mech, (var_dict, coef_opt, x, shock))
    loss = float(out.get("loss", np.inf))
    resid = np.asarray(out["resid"], dtype=float)
    # Return the user/profile weight; the solve re-adapts the Barron
    # factor on top of it (IRLS) to match the objective's per-point
    # weight (user/profile x adaptive Barron).
    agg = np.asarray(out["aggregate_weights"], dtype=float)
    barron = np.asarray(out["weights"], dtype=float)
    user_w = np.where(barron > 1e-300, agg / np.where(barron > 1e-300, barron, 1.0), 0.0)
    obs_sim_interp = np.asarray(out["obs_sim_interp"], dtype=float)
    if resid.size < 2 or not np.all(np.isfinite(resid)):
        return loss, None

    shift = (float(out.get("t_unc") or 0.0)
             + float(getattr(shock, "opt_time_offset", 0.0) or 0.0))
    observable, species_idx = sensitivity_observable(shock, mech.gas)
    t_sens, S = compute_sensitivity(
        mech, reactor_state, shock, observable=observable,
        species_idx=species_idx, method="forward_sens", n_workers=1,
    )
    t_exp = shock.exp_data_trim[:, 0]
    obs_exp = shock.exp_data_trim[:, 1]
    window = (t_exp >= t_sens[0] + shift) & (t_exp <= t_sens[-1] + shift)
    t_win = t_exp[window]
    if t_win.size != resid.size:
        return loss, None

    sigma = shock_sigma_bar(shock, scale)
    _resid, _mask, chain = _scaled_residual(
        obs_exp[window], obs_sim_interp, scale,
        getattr(shock, "bisymlog", None), sigma,
    )
    S_exp = np.column_stack([
        np.interp(t_win - shift, t_sens, S[:, j]) for j in target_idx
    ])
    a_matrix = chain[:, None] * S_exp
    lin = (a_matrix, resid, user_w, float(shock.T_reactor),
           float(shock.P_reactor))

    return loss, lin


def _pool_smurf_linearization(args):
    """Pool-worker task: one shock's linearization on the worker mech."""
    var_dict, coef_opt, x, shock, target_idx, reactor_state, scale = args
    try:
        result = _linearize_shock(
            pool_worker_mech(), reactor_state, scale, var_dict, coef_opt,
            x, shock, target_idx)
    except Exception:
        result = (np.inf, None)

    return result


def _campaign_weights(fit, n):
    """User × coverage × frozen-uniqueness weights across experiments."""
    user_w = fit._user_weights
    cov = np.ones(n)
    mode = fit.cost_settings.experiment_weighting
    if fit._exp_balance is not None and fit._exp_balance.size == n:
        cov = fit._exp_balance
        if fit._cov_features is not None and n >= 4:
            cov = cov * coverage_weights(fit._cov_features, user_w)
    elif (mode in ("coverage", "uniqueness")
          and fit._cov_features is not None and n >= 4):
        cov = coverage_weights(fit._cov_features, user_w)

    return user_w * cov


def _aggregate_shock_sensitivity(fit, losses_std):
    """Per-shock ``dA/d(raw loss)`` by finite difference through the
    objective's own aggregation — how much each shock's loss moves the
    campaign aggregate. Non-negative."""
    n = losses_std.size
    sigma_totals = np.array([
        float(getattr(s2, "sigma_total", 1.0) or 1.0) for s2 in fit.shocks2run
    ])
    losses_raw = losses_std * sigma_totals
    w_agg = _campaign_weights(fit, n)
    a0 = fit._legacy_residual_aggregate(losses_raw, w_agg)
    d_agg = np.zeros(n)
    for si in range(n):
        eps = 1e-3 * (abs(losses_raw[si]) + 1e-9)
        perturbed = losses_raw.copy()
        perturbed[si] += eps
        d_agg[si] = (fit._legacy_residual_aggregate(perturbed, w_agg) - a0) / eps
    d_agg = np.maximum(d_agg, 0.0)

    return d_agg


def _solve_multipliers(a_matrix, resid, user_w, loss_c, loss_alpha):
    """Ridged IRLS for one experiment's per-reaction log-multipliers.

    Reweights by user/profile x adaptive-Barron at each step, with the
    Barron factor recomputed on the current residual, so the step
    tracks the objective's residual-adaptive weighting. Returns
    ``(delta, weights)`` with the converged per-point weights.
    """
    n = a_matrix.shape[1]
    delta = np.zeros(n)
    w = np.array(user_w, dtype=float)
    for _ in range(3):
        loss_weights, _c, _alpha = adaptive_weights(
            resid - a_matrix @ delta, weights=user_w,
            C_scalar=loss_c, alpha=loss_alpha)
        w = user_w * loss_weights
        sw = np.sqrt(w)
        a_w = a_matrix * sw[:, None]
        r_w = resid * sw
        gram = a_w.T @ a_w
        lam = _RIDGE * (np.trace(gram) / max(gram.shape[0], 1) + 1e-300)
        delta = np.linalg.solve(gram + lam * np.eye(n), a_w.T @ r_w)
    result = (delta, w)

    return result


def _fit_shape(temperatures, pressures, deltas, weights):
    """Fit ``δln k = c0 + c1·lnT − c2/(Ru·T) + c3·lnP`` to the
    per-experiment multiplier cloud, weighted; returns the four
    coefficients. The lnP term lets the step move falloff (Troe/Plog)
    pressure dependence; it fits to ~0 for pressure-independent rates.
    """
    temperatures = np.asarray(temperatures)
    pressures = np.asarray(pressures)
    basis = np.column_stack([
        np.ones(temperatures.size), np.log(temperatures),
        -1.0 / (_RU * temperatures), np.log(pressures),
    ])
    sw = np.sqrt(np.maximum(weights, 0.0) + 1e-12)
    coef, *_ = np.linalg.lstsq(basis * sw[:, None], deltas * sw, rcond=None)

    return coef


def smurf_coarse(obj_fcn, fit, x0, bnds, options, log=None):
    """Run the Smurf coarse stage from ``x0`` in scaler space.

    ``obj_fcn(s)`` is the tracked scalar objective (line-search scored);
    ``fit`` is the cost function (residual/sensitivity internals). An
    iteration-maximum stop is the stage's TOTAL evaluation budget,
    shared across all multistart descents (the same contract as every
    other algorithm's max-eval stop); each descent also stops on its
    own when a damped step fails to improve the objective.
    ``multistart_count`` sets the number of descents; starts beyond
    the budget are skipped.

    Returns the driver-shaped result dict for the global stage.
    """
    lb = np.asarray(bnds[0], dtype=float)
    ub = np.asarray(bnds[1], dtype=float)
    target_idx = [rc["rxnIdx"] for rc in fit.rxn_coef_opt]
    blocks, offset = [], 0
    for rc in fit.rxn_coef_opt:
        n_pts = len(rc["T"])
        blocks.append((offset, offset + n_pts))
        offset += n_pts
    t_grids = [np.asarray(rc["T"], dtype=float) for rc in fit.rxn_coef_opt]
    p_grids = [np.asarray(rc["P"], dtype=float) for rc in fit.rxn_coef_opt]

    if options["stop_criteria_type"] == "Iteration Maximum":
        budget = max(int(options["stop_criteria_val"]), 2)
    else:
        budget = None
    # Per-descent safety cap; descents normally self-terminate on a
    # damping failure well before this.
    max_outer = 50
    counter = {"nfev": 0}

    def _budget_left():
        return budget is None or counter["nfev"] < budget

    loss_c = fit.cost_settings.loss_c
    loss_alpha = fit.cost_settings.loss_alpha
    use_pool = (
        getattr(fit, "multiprocessing", False)
        and getattr(fit, "pool", None) is not None
    )

    def _sweep(var_dict, x):
        """Per-shock linearization across the campaign; pooled when a
        worker pool is available (the sweep is the stage's dominant
        serial cost otherwise)."""
        if use_pool:
            args = [
                (var_dict, fit.coef_opt, x, shock, target_idx,
                 fit.reactor_state, fit.cost_settings.scale)
                for shock in fit.shocks2run
            ]
            results = fit.pool.map(_pool_smurf_linearization, args)
        else:
            results = [
                _linearize_shock(
                    fit.mech, fit.reactor_state, fit.cost_settings.scale,
                    var_dict, fit.coef_opt, x, shock, target_idx)
                for shock in fit.shocks2run
            ]

        return results

    def _descend(s_start):
        """One Gauss-Newton coarse descent from a start point."""
        s = np.clip(np.asarray(s_start, dtype=float), lb, ub)
        f_cur = obj_fcn(s)
        counter["nfev"] += 1
        for _outer in range(max_outer):
            if not _budget_left():
                break
            x = fit.fit_all_coeffs(np.exp(np.clip(s, lb, ub) + fit.x0))
            if x is None:
                break
            var_dict = fit._build_var_dict()
            losses = np.full(len(fit.shocks2run), np.nan)
            rows = []
            for si, (loss, lin) in enumerate(_sweep(var_dict, x)):
                losses[si] = loss
                if lin is None:
                    continue
                a_matrix, resid, user_w, temperature, pressure = lin
                delta, w = _solve_multipliers(
                    a_matrix, resid, user_w, loss_c, loss_alpha)
                leverage = np.sqrt((w[:, None] * a_matrix**2).sum(axis=0))
                rows.append((si, temperature, pressure, delta, leverage))
            if not rows:
                break

            if np.all(np.isfinite(losses)):
                d_agg = _aggregate_shock_sensitivity(fit, losses)
                d_agg = d_agg / max(d_agg.max(), 1e-300)
            else:
                d_agg = np.ones(len(fit.shocks2run))

            step = np.zeros(lb.size)
            for k, (a, b) in enumerate(blocks):
                temperatures = np.array([r[1] for r in rows])
                pressures = np.array([r[2] for r in rows])
                deltas = np.array([r[3][k] for r in rows])
                weights = np.array([r[4][k] * d_agg[r[0]] for r in rows])
                coef = _fit_shape(temperatures, pressures, deltas, weights)
                t_grid = t_grids[k]
                p_grid = p_grids[k]
                model = (coef[0] + coef[1] * np.log(t_grid)
                         - coef[2] / (_RU * t_grid) + coef[3] * np.log(p_grid))
                step[a:b] = model

            accepted = False
            for damp in _DAMPING:
                if not _budget_left():
                    break
                trial = np.clip(s + damp * step, lb, ub)
                f_trial = obj_fcn(trial)
                counter["nfev"] += 1
                if f_trial < f_cur:
                    s, f_cur, accepted = trial, f_trial, True
                    break
            if not accepted:
                break
        outcome = (np.clip(s, lb, ub), f_cur)

        return outcome

    # Multistart: the incumbent plus Sobol perturbations around it; the
    # local GN descent alone finds only the incumbent's basin, so the
    # starts are what let Smurf compete with a global sampler.
    x0 = np.clip(np.asarray(x0, dtype=float), lb, ub)
    n_starts = max(1, int(options.get("multistart_count", 16) or 16))
    starts = [x0]
    if n_starts > 1:
        seed = int(options.get("random_seed", 0) or 0)
        # Draw the next power of two and slice: Sobol balance properties
        # only hold for 2^m points.
        n_extra = n_starts - 1
        m = max(1, int(np.ceil(np.log2(n_extra))))
        sobol = qmc.Sobol(d=lb.size, scramble=True, seed=seed)
        unit = sobol.random_base2(m=m)[:n_extra]
        span = ub - lb
        for u in unit:
            starts.append(np.clip(x0 + (2.0 * u - 1.0) * 0.5 * span, lb, ub))

    best_s, best_f = x0, np.inf
    for s_start in starts:
        if not _budget_left() and np.isfinite(best_f):
            break
        cand_s, cand_f = _descend(s_start)
        if cand_f < best_f:
            best_s, best_f = cand_s, cand_f

    x_coef = fit.fit_all_coeffs(np.exp(best_s + fit.x0))
    if x_coef is not None:
        update_mech_coef_opt(fit.mech, fit.coef_opt, x_coef)
    obj_val, x_final, shock_output = fit(best_s, optimizing=False)
    result = {
        "x": x_final,
        "s": best_s,
        "shock": shock_output,
        "fval": obj_val,
        "nfev": counter["nfev"],
        "success": True,
        "message": "Smurf coarse stage complete",
        "time": 0.0,
    }

    return result
