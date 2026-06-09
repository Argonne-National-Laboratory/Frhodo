# This file is part of Frhodo. Copyright © 2020, UChicago Argonne, LLC
# and licensed under BSD-3-Clause. See License.txt in the top-level
# directory for license and copyright information.
"""Cost-function machinery: residual evaluation, weighting, loss scaling.

The :class:`CostFunction` instance is what the algorithms in
:mod:`frhodo.optimize.algorithms` call once per iteration; it owns
the worker pool, the parameter unpacking, and the loss-shape choice
(residual / Bayesian / adaptive).
"""
import io, contextlib
import numpy as np
import nlopt
from scipy.optimize import minimize_scalar, brentq
from scipy.interpolate import CubicSpline
from scipy.special import expit
from scipy import stats
from copy import deepcopy

from timeit import default_timer as timer

from frhodo.simulation.mechanism.mech_fcns import ChemicalMechanism
from frhodo.simulation.shock.incident_shock_reactor import run_incident_shock
from frhodo.simulation.shock.zero_d_reactor import run_zero_d
from frhodo.common.units import OoM
from frhodo.simulation.shock.state import zero_d_mode_from_label
from frhodo.optimize._worker_context import MechBuildPayload, WorkerContext
from frhodo._vendor.opendsm.adaptive_loss import adaptive_weights
from frhodo._vendor.opendsm.stats_basic import weighted_quantile
from frhodo.simulation.mechanism.fit_coeffs import fit_coeffs
from frhodo.optimize.cost.bayesian import CheKiPEUQ_Frhodo_interface
from frhodo.optimize.cost.settings import CostSettings
from frhodo.optimize.time_shift_model import regularized_shifts



# Per-worker mechanism for the spawn-pool dispatch path.
# Populated by initialize_parallel_worker; read only by _pool_calculate_residuals.
_pool_worker_ctx: WorkerContext | None = None


def initialize_parallel_worker(payload: MechBuildPayload):
    """Build a per-worker :class:`ChemicalMechanism` from the spawn payload.

    Invoked by ``mp.Pool(initializer=...)`` once per worker. Cantera
    chatters to stdout/stderr during ``set_mechanism``; both streams
    are redirected to avoid garbling the parent's log.

    Sets ``NUMBA_CACHE_DIR`` to a per-worker temp directory before any
    ``@njit(cache=True)`` function is called. Numba's atomic-rename
    cache write is not robust across concurrent multiprocessing
    workers on Windows — two workers can race on the same target
    filename and one sees ``WinError None / file not found`` during
    the rename. Per-worker cache dirs eliminate the shared target.
    """
    import os
    import tempfile

    os.environ["NUMBA_CACHE_DIR"] = tempfile.mkdtemp(prefix="frhodo_numba_")

    global _pool_worker_ctx
    mech = ChemicalMechanism()

    with contextlib.redirect_stderr(io.StringIO()):
        with contextlib.redirect_stdout(io.StringIO()):
            mech.set_mechanism(payload.reset_mech, payload.thermo_coeffs)

    # ``payload`` is the worker's only copy (pickled across the spawn
    # boundary); assigning directly avoids the deepcopy cost.
    mech.coeffs = payload.coeffs
    mech.coeffs_bnds = payload.coeffs_bnds
    mech.rate_bnds = payload.rate_bnds
    _pool_worker_ctx = WorkerContext(mech=mech)


def _pool_calculate_residuals(args):
    return calculate_residuals(_pool_worker_ctx.mech, args)


def _pool_fit_coeffs(args):
    return fit_coeffs(*args, _pool_worker_ctx.mech)


def _log_ratio(obs_exp, obs_sim):
    ratio = np.where(obs_exp >= obs_sim, obs_exp / obs_sim, obs_sim / obs_exp)

    return np.log10(np.abs(ratio))


_WARM_START_HALF_WIDTH = 0.2
_WARM_START_EDGE_TOL = 0.05


def _narrow_bounds(full_bounds, cached):
    """Narrow ``full_bounds`` to ``±_WARM_START_HALF_WIDTH`` around ``cached``.

    Both ``full_bounds`` and ``cached`` must be in the same units. Returns
    ``full_bounds`` unchanged when ``cached`` is ``None`` or outside the
    bounds.
    """
    if cached is None:
        return full_bounds
    lo, hi = float(full_bounds[0]), float(full_bounds[1])
    if not (lo <= cached <= hi):
        return full_bounds
    half_width = _WARM_START_HALF_WIDTH * (hi - lo)
    narrowed_lo = max(lo, cached - half_width)
    narrowed_hi = min(hi, cached + half_width)

    return np.array([narrowed_lo, narrowed_hi])


def _hit_edge(solution, bounds):
    """Return True when ``solution`` sits within ``_WARM_START_EDGE_TOL`` of either bound."""
    lo, hi = float(bounds[0]), float(bounds[1])
    width = hi - lo
    if width <= 0:
        return False

    return (solution - lo) / width < _WARM_START_EDGE_TOL or (hi - solution) / width < _WARM_START_EDGE_TOL


T_UNC_PENALTY_FRACTION = 2.0


def _solve_t_unc(
    *,
    f_interp,
    t_exp,
    obs_exp,
    weights,
    t_offset,
    loss_c,
    loss_alpha,
    full_bounds,
    warm_start_t_unc=None,
    penalty_fraction=T_UNC_PENALTY_FRACTION,
):
    """Brentq root-find on ``dL/dτ = 0`` over ``τ ∈ full_bounds``.

    ``L_fit(τ) = Σ wᵢ_agg · soft_w(tᵢ, τ) · rᵢ²`` where ``soft_w`` is a
    pair of logistic sigmoids around the sim-domain edges. As τ slides
    the window past an experiment point, that point's contribution
    smoothly fades from 1 to 0 over the smoothing scale ε. With ε
    small relative to data spacing, ``soft_w`` is essentially the hard
    mask except in a tiny transition zone — but ``L_fit`` and its
    derivative are C¹ in τ, so brentq applies at any bracket width.

    ``Penalty(τ) = α₁·S(τ) + α₂·S(τ)²`` with ``S(τ) = Σ wᵢ_agg · (1 −
    soft_w)`` — aggregate weight of points the soft mask is excluding.
    ``α₁ + α₂`` are scaled so a full drop costs ``penalty_fraction ×
    L_fit(0)``. Disincentivizes large shifts that ignore weighted
    points without forbidding them when the fit gain is larger than
    the penalty.

    For narrow brackets where no point transitions through the window
    edge (the optimizer's typical case), ``soft_w ≡ 1``, ``S ≡ 0``,
    penalty is inert, and behavior matches the previous intersection-
    window brentq path.

    ``agg_weights = user_weights · adaptive_loss_weights`` are frozen
    at the seed τ. Points outside the seed soft-window keep their user
    weight (adaptive factor 1).

    Linear residuals only — caller must guard on ``scale``.

    Returns τ in the same units as ``full_bounds`` or ``None`` when
    every bracket has same-sign derivative endpoints with no recoverable
    boundary minimum.
    """
    full_lo, full_hi = float(full_bounds[0]), float(full_bounds[1])
    if full_hi <= full_lo:
        return None

    sim_lo, sim_hi = float(f_interp.x[0]), float(f_interp.x[-1])

    seed_tau = warm_start_t_unc if warm_start_t_unc is not None else 0.0
    seed_tau = float(np.clip(seed_tau, full_lo, full_hi))
    seed_shift = t_offset + seed_tau

    # ε = half the median exp-grid spacing — soft mask transitions over
    # roughly one data point. Floor avoids divide-by-zero on degenerate
    # grids.
    dt = np.diff(t_exp)
    eps = max(0.5 * float(np.median(dt)) if dt.size else 0.0, 1e-12)

    def soft_w_components(shift):
        u = (t_exp - sim_lo - shift) / eps
        v = (sim_hi + shift - t_exp) / eps
        sig_lo = expit(u)
        sig_hi = expit(v)

        return sig_lo, sig_hi, sig_lo * sig_hi

    # Freeze adaptive_weights at the seed τ — points with non-trivial
    # soft weight at the seed contribute to the loss-shape estimate.
    sig_lo_s, sig_hi_s, sw_seed = soft_w_components(seed_shift)
    valid_seed = sw_seed > 1e-3
    loss_weights = np.ones_like(weights, dtype=float)
    if valid_seed.sum() >= 2:
        resid_seed = obs_exp[valid_seed] - f_interp(t_exp[valid_seed] - seed_shift)
        seed_w = (weights * sw_seed)[valid_seed]
        lw_seed, _, _ = adaptive_weights(
            resid_seed, weights=seed_w, C_scalar=loss_c, alpha=loss_alpha,
        )
        loss_weights[valid_seed] = lw_seed
    agg_weights = weights * loss_weights
    total_weight = float(np.sum(agg_weights))

    def L_fit_at(tau):
        shift = t_offset + tau
        _, _, sw = soft_w_components(shift)
        r = obs_exp - f_interp(t_exp - shift)

        return float(np.sum(agg_weights * sw * r * r))

    L0_fit = L_fit_at(0.0)
    if not np.isfinite(L0_fit) or L0_fit <= 0:
        L0_fit = 1.0

    if total_weight > 0 and penalty_fraction > 0:
        target = penalty_fraction * L0_fit
        alpha_L1 = 0.5 * target / total_weight
        alpha_L2 = 0.5 * target / (total_weight * total_weight)
    else:
        alpha_L1 = 0.0
        alpha_L2 = 0.0

    def L_total_at(tau):
        shift = t_offset + tau
        _, _, sw = soft_w_components(shift)
        r = obs_exp - f_interp(t_exp - shift)
        fit = float(np.sum(agg_weights * sw * r * r))
        S = float(np.sum(agg_weights * (1.0 - sw)))

        return fit + alpha_L1 * S + alpha_L2 * S * S

    f_deriv = f_interp.derivative()

    def dL_dtau(tau):
        shift = t_offset + tau
        sig_lo, sig_hi, sw = soft_w_components(shift)
        # dsoft_w/dτ = (σ_lo·σ_hi/ε)·(σ_lo − σ_hi)
        dsw = (sw / eps) * (sig_lo - sig_hi)

        r = obs_exp - f_interp(t_exp - shift)
        d_obs = f_deriv(t_exp - shift)
        # d(sw·r²)/dτ = dsw·r² + 2·sw·r·obs_sim'(t−shift)
        d_L_fit = float(np.sum(agg_weights * (dsw * r * r + 2.0 * sw * r * d_obs)))

        # S(τ) = Σ w_agg·(1 − sw); dS/dτ = −Σ w_agg·dsw
        S = float(np.sum(agg_weights * (1.0 - sw)))
        d_S = -float(np.sum(agg_weights * dsw))
        d_Penalty = (alpha_L1 + 2.0 * alpha_L2 * S) * d_S

        return d_L_fit + d_Penalty

    # L_total can be multi-modal — the penalty creates a barrier around
    # the "drop everything" region, but the unpenalized fit may dip again
    # at large τ where only the plateau is in window. Grid-scan dL/dτ for
    # every sign change, brentq each, and return the argmin of L_total
    # over {bounds, warm-start, all roots}.
    n_grid = int(np.clip((full_hi - full_lo) / eps * 4, 11, 100))
    grid = np.linspace(full_lo, full_hi, n_grid)
    dL_vals = np.array([dL_dtau(t) for t in grid])

    candidates = [full_lo, full_hi]
    if warm_start_t_unc is not None and full_lo <= seed_tau <= full_hi:
        candidates.append(seed_tau)

    for i in range(n_grid - 1):
        if dL_vals[i] * dL_vals[i + 1] < 0:
            try:
                root = float(brentq(dL_dtau, grid[i], grid[i + 1],
                                    xtol=1e-9, rtol=1e-6))
                candidates.append(root)
            except (ValueError, RuntimeError):
                continue

    L_vals = [L_total_at(c) for c in candidates]
    finite = [(L, c) for L, c in zip(L_vals, candidates) if np.isfinite(L)]
    if not finite:
        return None

    return float(min(finite)[1])


def rescale_loss_fcn(x, loss, x_outlier=None, weights=[]):
    """Linearly map ``loss`` into the ``x`` value range.

    Used to bring an adaptive-loss output back to residual-magnitude
    units so it is comparable across stages. Outlier rows beyond
    ``x_outlier`` are trimmed for the rescaling bounds but the full
    ``loss`` is returned. Weights, when supplied, drive a weighted
    min/max for the bound computation.
    """
    x = x.copy()
    weights = weights.copy()

    if x_outlier is not None:
        trimmed_indices = np.argwhere(abs(x) < x_outlier)
        x = x[trimmed_indices]
        loss_trimmed = loss[trimmed_indices]
        weights = weights[trimmed_indices]
    else:
        loss_trimmed = loss

    if len(weights) == len(x):
        x_q1, x_q3 = weighted_quantile(x, np.array([0.0, 1.0]), weights=weights)
        loss_q1, loss_q3 = weighted_quantile(
            loss_trimmed, np.array([0.0, 1.0]), weights=weights
        )

    else:
        x_q1, x_q3 = x.min(), x.max()
        loss_q1, loss_q3 = loss_trimmed.min(), loss_trimmed.max()

    if (
        x_q1 != x_q3 and loss_q1 != loss_q3
    ):  # prevent divide by zero if values end up the same
        loss_scaled = (x_q3 - x_q1) / (loss_q3 - loss_q1) * (loss - loss_q1) + x_q1

    else:
        loss_scaled = loss

    return loss_scaled


def update_mech_coef_opt(mech, coef_opt, x):
    """Push optimizer-space coefficients ``x`` back into ``mech``.

    Compares each coefficient to the stored value and skips the
    Cantera ``modify_reaction`` call when nothing changed. After the
    last update, a single ``modify_reactions`` flush commits all
    changes in one pass.

    Raises:
        ValueError: When the optimizer hands back a non-positive
            pre-exponential factor — that means the fit kernel
            escaped its bounds and the upstream Troe-upgrade path
            needs investigation.
    """
    mech_changed = False
    for i, c in enumerate(coef_opt):
        rxnIdx, coefName, coeffs_key = c.rxn_idx, c.coef_name, c.coeffs_key
        if coefName == "pre_exponential_factor" and not x[i] > 0:
            raise ValueError(
                f"Non-positive pre_exponential_factor for R{rxnIdx + 1} "
                f"({coeffs_key}): A={x[i]!r}. The fit kernel produced an "
                f"out-of-bounds A; check that pressure-dependent rxns are "
                f"routed through the Troe upgrade path before optimization."
            )
        if mech.coeffs[rxnIdx][coeffs_key][coefName] != x[i]:
            if type(mech.coeffs[rxnIdx][coeffs_key]) is tuple:
                mech.coeffs[rxnIdx][coeffs_key] = list(mech.coeffs[rxnIdx][coeffs_key])
            mech_changed = True
            mech.coeffs[rxnIdx][coeffs_key][coefName] = x[i]

    if mech_changed:
        mech.modify_reactions(mech.coeffs)


def _aggregate_ode_errors(per_shock: list[str]) -> str | None:
    """Combine multiple per-shock ODE summaries into one log annotation.

    Single failure → that summary verbatim.
    Multiple failures → count + union of suggested-reaction indices,
    so the user sees the breadth of the failure without N copies of the
    same multi-paragraph block.
    """
    if not per_shock:
        return None
    if len(per_shock) == 1:
        return per_shock[0]

    rxn_set: set[str] = set()
    for entry in per_shock:
        if "rxns " in entry:
            tail = entry.split("rxns ", 1)[1]
            for token in tail.split(","):
                token = token.strip()
                if token:
                    rxn_set.add(token)
    n = len(per_shock)
    if rxn_set:
        rxns = ",".join(sorted(rxn_set, key=lambda v: int(v) if v.isdigit() else 0))

        return f"ODE: {n} shocks failed; rxns {rxns}"

    return f"ODE: {n} shocks failed"


def _summarize_ode_failure(verbose) -> str | None:
    """One-line ODE failure summary for the iteration log.

    Reactor backends pack the full multi-paragraph failure description
    into ``verbose["message"]``. During optimization we want a compact
    inline annotation — e.g. ``"ODE: Temperature is invalid; rxns
    2,27,45"`` — rather than the multi-line block. Returns ``None``
    when the sim succeeded.
    """
    if verbose is None or verbose.get("success"):
        return None
    msg = verbose.get("message", "")
    if isinstance(msg, list):
        msg = " ".join(str(m) for m in msg)
    if not msg:
        return None
    parts = [p.strip() for p in msg.replace("ODE Error:", "").split("\n") if p.strip()]
    head = parts[0] if parts else ""
    rxn_part = next(
        (p.split(":", 1)[1].strip() for p in parts if p.startswith("Suggested Reactions")),
        None,
    )
    if rxn_part:
        return f"ODE: {head}; rxns {rxn_part}"

    return f"ODE: {head}"


def _degenerate_trace_output(shock, ind_var: np.ndarray, obs_sim: np.ndarray,
                             coef_opt, var: dict, *,
                             ode_error: str | None = None) -> dict:
    """Per-shock output signaling an undefined objective.

    Extreme parameter perturbations can collapse the simulation to
    fewer than two timesteps, leaving nothing to interpolate. The cost
    function has no defined value at such a point; we signal that with
    ``loss = np.inf`` and let the optimizer's wrapper decide how to
    handle (most algorithms reject inf and pick a different point).
    Other dict slots get finite placeholders so downstream
    bookkeeping (stat_plot, KDE) doesn't crash on shape checks.
    """
    one = np.array([1.0])
    constant_value = float(obs_sim[0, 0]) if obs_sim.size else 0.0

    return {
        "wsse": np.inf,
        "resid": np.array([0.0]),
        "resid_outlier": 0.0,
        "loss": np.inf,
        "weights": one.copy(),
        "aggregate_weights": one.copy(),
        "obs_sim_interp": np.array([constant_value]),
        "obs_exp": np.array([0.0]),
        "obs_bounds": [],
        "shock": shock,
        "independent_var": ind_var,
        "observable": obs_sim,
        "t_unc": 0.0,
        "loss_alpha": float(var.get("loss_alpha", 2.0)) if not isinstance(
            var.get("loss_alpha"), str
        ) else 2.0,
        "KDE": np.column_stack(([0.0], [1.0])),
        "ode_error": ode_error,
    }


def calculate_residuals(mech, args_list):
    """Run one shock simulation and compute the per-point residual stats.

    Pool-worker entrypoint. Builds the reactor according to
    ``args_list``, runs it to ``t_end``, time-aligns against the
    experiment data, and returns the residual array plus the loss
    statistics the parent process needs to aggregate.

    Returns:
        Dict carrying ``"resid"``, ``"weights"``, ``"obs_sim"``,
        ``"ind_var"``, ``"obs_exp"``, ``"t_offset"``, ``"density"``,
        and bookkeeping for the live plot. Returns the
        ``_degenerate_trace_output`` penalty dict when the reactor
        produced fewer than 2 timesteps.
    """
    def resid_func(
        t_offset,
        t_adjust,
        f_interp,
        t_exp,
        obs_exp,
        weights,
        obs_bounds=[],
        loss_alpha=2,
        loss_c=1,
        loss_penalty=True,
        scale="Linear",
        bisymlog=None,
        DoF=1,
        opt_type="Residual",
        verbose=False,
    ):
        shift = t_offset + t_adjust
        t_lo = max(f_interp.x[0] + shift, t_exp[0])
        t_hi = min(f_interp.x[-1] + shift, t_exp[-1])
        exp_bounds = np.where((t_exp >= t_lo) & (t_exp <= t_hi))[0]
        t_exp, obs_exp, weights = (
            t_exp[exp_bounds],
            obs_exp[exp_bounds],
            weights[exp_bounds],
        )
        if opt_type == "Bayesian":
            obs_bounds = obs_bounds[exp_bounds]

        obs_sim_interp = f_interp(t_exp - shift)

        if scale == "Linear":
            resid = np.subtract(obs_exp, obs_sim_interp)

        elif scale == "Log":
            ind = np.argwhere((obs_exp > 0.0) & (obs_sim_interp > 0.0))
            exp_bounds = exp_bounds[ind]
            weights = weights[ind].flatten()

            resid = (
                np.log10(obs_exp[ind]) - np.log10(obs_sim_interp[ind])
            ).flatten()
            if verbose and opt_type == "Bayesian":
                obs_exp = np.log10(obs_exp[ind]).squeeze()
                obs_sim_interp = np.log10(obs_sim_interp[ind]).squeeze()
                obs_bounds = np.log10(obs_bounds[ind]).squeeze()

        elif scale == "AbsoluteLog":
            ind = np.argwhere((obs_exp != 0.0) & (obs_sim_interp != 0.0))
            exp_bounds = exp_bounds[ind]
            weights = weights[ind].flatten()

            resid = _log_ratio(obs_exp[ind], obs_sim_interp[ind]).flatten()
            if verbose and opt_type == "Bayesian":
                obs_exp = np.log10(np.abs(obs_exp[ind])).squeeze()
                obs_sim_interp = np.log10(np.abs(obs_sim_interp[ind])).squeeze()
                obs_bounds = np.log10(np.abs(obs_bounds[ind])).squeeze()

        elif scale == "Bisymlog":
            obs_exp_bisymlog = bisymlog.transform(obs_exp)
            obs_sim_interp_bisymlog = bisymlog.transform(obs_sim_interp)
            resid = np.subtract(obs_exp_bisymlog, obs_sim_interp_bisymlog)
            if verbose and opt_type == "Bayesian":
                obs_exp = obs_exp_bisymlog
                obs_sim_interp = obs_sim_interp_bisymlog
                obs_bounds = bisymlog.transform(obs_bounds)  # THIS NEEDS TO BE CHECKED

        loss_weights, C, alpha = adaptive_weights(
            resid, weights=weights, C_scalar=loss_c, alpha=loss_alpha
        )
        agg_weights = weights * loss_weights

        # Bessel-style weighted RMSE: aggregate weights (user × adaptive)
        # applied to the SSE, normalized by effective DoF.
        wsse = (agg_weights * resid**2).sum()
        agg_w_sum = agg_weights.sum()
        eff_dof = agg_w_sum - DoF
        if eff_dof <= 0:
            eff_dof = agg_w_sum
        if eff_dof > 0:
            loss_scalar = float(np.sqrt(wsse / eff_dof))
        else:
            loss_scalar = 0.0

        if verbose:
            output = {
                "wsse": wsse,
                "resid": resid,
                "resid_outlier": C,
                "loss": loss_scalar,
                "weights": loss_weights,
                "aggregate_weights": agg_weights,
                "obs_sim_interp": obs_sim_interp,
                "obs_exp": obs_exp,
                "obs_bounds": obs_bounds,
            }

            return output

        else:  # needs to return single value for optimization
            return loss_scalar

    def calc_density(x, data, dim=1):
        data = np.asarray(data, dtype=float)
        x = np.asarray(x, dtype=float)
        stdev = np.std(data)
        if stdev == 0:  # constant residuals -> singular KDE covariance
            return np.zeros_like(x)

        [q1, q3] = weighted_quantile(data, np.array([0.25, 0.75]))
        iqr = q3 - q1  # interquartile range
        A = (
            np.min([stdev, iqr / 1.34]) / stdev
        )  # bandwidth is multiplied by std of sample
        bw = 0.9 * A * len(data) ** (-1.0 / (dim + 4))

        try:
            density = stats.gaussian_kde(data, bw_method=bw)(x)
        except (np.linalg.LinAlgError, ValueError):
            density = np.zeros_like(x)

        return density

    if len(args_list) == 5:
        var, coef_opt, x, shock, fixed_t_unc = args_list
    else:
        var, coef_opt, x, shock = args_list
        fixed_t_unc = None

    update_mech_coef_opt(mech, coef_opt, x)

    T_reac, P_reac, mix = shock.T_reactor, shock.P_reactor, shock.thermo_mix

    SIM_kwargs = {
        "u_reac": shock.u2,
        "rho1": shock.rho1,
        "observable": shock.observable,
        "t_lab_save": None,
        "sim_int_f": var["sim_interp_factor"],
        "ODE_solver": var["ode_solver"],
        "rtol": var["ode_rtol"],
        "atol": var["ode_atol"],
    }

    if "0d Reactor" in var["name"]:
        SIM_kwargs["solve_energy"] = var["solve_energy"]
        SIM_kwargs["frozen_comp"] = var["frozen_comp"]

    if var["name"] == "Incident Shock Reactor":
        SIM, verbose = run_incident_shock(
            mech, var["t_end"], T_reac, P_reac, mix, **SIM_kwargs
        )
    elif "0d Reactor" in var["name"]:
        mode = zero_d_mode_from_label(var["name"])
        SIM, verbose = run_zero_d(
            mech, mode, var["t_end"], T_reac, P_reac, mix, **SIM_kwargs
        )
    else:
        raise ValueError(f"unknown reactor: {var['name']!r}")
    ind_var, obs_sim = SIM.independent_var[:, None], SIM.observable[:, None]
    if ind_var.size < 2:
        return _degenerate_trace_output(
            shock, ind_var, obs_sim, coef_opt, var,
            ode_error=_summarize_ode_failure(verbose),
        )
    f_interp = CubicSpline(ind_var.flatten(), obs_sim.flatten())

    weights = shock.weights_trim
    obs_exp = shock.exp_data_trim
    obs_bounds = []
    if var["obj_fcn_type"] == "Bayesian":
        obs_bounds = shock.abs_uncertainties_trim

    t_unc_max = float(np.max(np.abs(var["t_unc"])))
    if fixed_t_unc is not None:
        t_unc = fixed_t_unc
    elif t_unc_max < 1e-12:
        t_unc = 0
    else:
        t_unc_OoM = np.mean(OoM(var["t_unc"]))
        time_adj_func = lambda t_adjust: resid_func(
            shock.opt_time_offset,
            t_adjust * 10**t_unc_OoM,
            f_interp,
            obs_exp[:, 0],
            obs_exp[:, 1],
            weights,
            obs_bounds,
            scale=var["scale"],
            bisymlog=getattr(shock, "bisymlog", None),
            DoF=len(coef_opt),
            opt_type=var["obj_fcn_type"],
        )

        full_bounds = var["t_unc"] / 10**t_unc_OoM
        cached_t_unc = getattr(shock, "last_t_unc", None)

        t_unc = None
        if var["scale"] == "Linear":
            t_unc = _solve_t_unc(
                f_interp=f_interp,
                t_exp=obs_exp[:, 0],
                obs_exp=obs_exp[:, 1],
                weights=weights,
                t_offset=shock.opt_time_offset,
                loss_c=var["loss_c"],
                loss_alpha=2,
                full_bounds=var["t_unc"],
                warm_start_t_unc=cached_t_unc,
            )

        if t_unc is None:
            cached_scaled = (
                cached_t_unc / 10**t_unc_OoM if cached_t_unc is not None else None
            )
            bounds = _narrow_bounds(full_bounds, cached_scaled)
            res = minimize_scalar(
                time_adj_func,
                bounds=bounds,
                method="bounded",
                options={"xatol": 1e-3},
            )
            if _hit_edge(res.x, bounds) and bounds is not full_bounds:
                res = minimize_scalar(
                    time_adj_func,
                    bounds=full_bounds,
                    method="bounded",
                    options={"xatol": 1e-3},
                )
            t_unc = res.x * 10**t_unc_OoM

    # calculate loss shape function (alpha) if it is set to adaptive
    loss_alpha = var["loss_alpha"]
    if loss_alpha == 3.0:
        loss_alpha_fcn = lambda alpha: resid_func(
            shock.opt_time_offset,
            t_unc,
            f_interp,
            obs_exp[:, 0],
            obs_exp[:, 1],
            weights,
            obs_bounds,
            loss_alpha=alpha,
            loss_c=var["loss_c"],
            loss_penalty=True,
            scale=var["scale"],
            bisymlog=getattr(shock, "bisymlog", None),
            DoF=len(coef_opt),
            opt_type=var["obj_fcn_type"],
        )

        full_alpha_bounds = np.array([-100.0, 2.0])
        alpha_bounds = _narrow_bounds(
            full_alpha_bounds, getattr(shock, "last_loss_alpha", None),
        )
        res = minimize_scalar(
            loss_alpha_fcn,
            bounds=alpha_bounds,
            method="bounded",
            options={"xatol": 1e-3},
        )
        if _hit_edge(res.x, alpha_bounds) and alpha_bounds is not full_alpha_bounds:
            res = minimize_scalar(
                loss_alpha_fcn,
                bounds=full_alpha_bounds,
                method="bounded",
                options={"xatol": 1e-3},
            )
        loss_alpha = res.x

    if var["obj_fcn_type"] == "Residual":
        loss_penalty = True
    else:
        loss_penalty = False

    output = resid_func(
        shock.opt_time_offset,
        t_unc,
        f_interp,
        obs_exp[:, 0],
        obs_exp[:, 1],
        weights,
        obs_bounds,
        loss_alpha=loss_alpha,
        loss_c=var["loss_c"],
        loss_penalty=loss_penalty,
        scale=var["scale"],
        bisymlog=getattr(shock, "bisymlog", None),
        DoF=len(coef_opt),
        opt_type=var["obj_fcn_type"],
        verbose=True,
    )

    output["shock"] = shock
    output["independent_var"] = ind_var
    output["observable"] = obs_sim
    output["t_unc"] = t_unc
    output["loss_alpha"] = loss_alpha
    output["ode_error"] = None

    plot_stats = True
    if plot_stats:
        x = np.linspace(output["resid"].min(), output["resid"].max(), 300)
        density = calc_density(x, output["resid"], dim=1)  # kernel density estimation
        output["KDE"] = np.column_stack((x, density))

    return output


# Using optimization vs least squares curve fit because y_ranges change
# if time_offset != 0.
class CostFunction:
    """Cost-function callable used by the optimization loop.

    Built from an :class:`~frhodo.optimize.residual.OptimizeRunInputs`
    bundle plus per-run callbacks. The optimization loop constructs one
    of these and calls it once per iteration; ``__call__`` returns the
    aggregate loss across all shocks.

    Attributes:
        x0: Initial scaled rates baseline; the optimizer's argument
            ``s`` is added to this to get the absolute scaled rates.
        opt_type: Stage label (``"global"`` / ``"local"``); used in
            log lines and to switch behavior inside Bayesian mode.
        i: Iteration counter the optimization loop reads to format
            progress lines.
    """

    def __init__(
        self,
        inputs: "OptimizeRunInputs",
        *,
        pool=None,
        display_shock_provider=None,
        log_callback=None,
        progress_callback=None,
    ):
        self.mech = inputs.mech
        self.shocks2run = inputs.shocks2run
        self.coef_opt = inputs.coef_opt
        self.rxn_coef_opt = inputs.rxn_coef_opt
        self.rxn_rate_opt = inputs.rxn_rate_opt
        self.x0 = inputs.rxn_rate_opt["x0"]
        self.reactor_state = inputs.reactor_state
        self.t_unc = (-inputs.time_unc, inputs.time_unc)
        self.opt_type = "local"
        self.dist = inputs.dist
        self.cost_settings = inputs.cost_settings
        self._display_shock_provider = display_shock_provider
        self.pool = pool
        self.multiprocessing = inputs.multiprocessing if pool is not None else False
        self._log = log_callback or (lambda msg: None)
        self._progress = progress_callback or (lambda update: None)
        self.i = 0
        self.__abort = False
        self._last_aggregate_loss_alpha: float | None = None
        self.random_t_uncertainty = inputs.random_t_uncertainty
        self._shift_model_info: dict | None = None
        self._shift_model_logged = False

        if inputs.cost_settings.obj_fcn_type == "Bayesian":
            self.CheKiPEUQ_Frhodo_interface = CheKiPEUQ_Frhodo_interface(
                bayes_dist_type=inputs.cost_settings.bayes_dist_type,
                coef_opt=inputs.coef_opt,
                rxn_coef_opt=inputs.rxn_coef_opt,
                rxn_rate_opt=inputs.rxn_rate_opt,
                bayes_unc_sigma=inputs.cost_settings.bayes_unc_sigma,
            )

    def _build_var_dict(self) -> dict:
        """Per-call settings bundle the residual workers consume."""
        var_dict = self.reactor_state.model_dump()
        var_dict["t_unc"] = self.t_unc
        var_dict.update(self.cost_settings.model_dump())
        var_dict["random_t_uncertainty"] = self.random_t_uncertainty

        return var_dict

    def _dispatch(self, x, var_dict, fixed_shifts=None):
        """Run ``calculate_residuals`` for every shock (pool or serial).

        ``fixed_shifts`` supplies a per-shock time shift; when ``None`` each
        shock solves its own free-optimal shift.
        """
        shocks = self.shocks2run
        if fixed_shifts is None:
            args = [(var_dict, self.coef_opt, x, s) for s in shocks]
        else:
            args = [
                (var_dict, self.coef_opt, x, s, float(fixed_shifts[i]))
                for i, s in enumerate(shocks)
            ]

        if self.multiprocessing and len(shocks) > 1:
            outputs = list(self.pool.map(_pool_calculate_residuals, args))
        else:
            outputs = [calculate_residuals(self.mech, a) for a in args]

        return outputs

    def _maybe_log_shift_model(self):
        """Log the fitted parametric time-shift model once per run."""
        info = self._shift_model_info
        if self._shift_model_logged or not info or info.get("model") is None:
            return

        self._shift_model_logged = True
        names = info["feature_names"]
        active = [n for n, c in zip(names, info["coefficients"]) if abs(c) > 0]
        feature_summary = ", ".join(active) or "none"
        self._log(
            f"Parametric t-uncertainty: elastic-net penalty={info['penalty']:.3g}, "
            f"l1_ratio={info['l1_ratio']:.2g}; active features: {feature_summary}"
        )

    def warmup_workers(self, n_workers: int, initial_scalers: np.ndarray) -> None:
        """Force per-worker numba JIT before the optimizer starts.

        Without this, iterations 1..N pay 1-5s/kernel × N kernels of JIT
        compilation lazily as workers hit each ``@njit`` function for the
        first time. Dispatching one full ``calculate_residuals`` per
        worker on a representative shock fans the JIT cost across all
        workers in parallel — wall time becomes ~max(per-worker compile)
        instead of the sum across the first several iterations.

        ``initial_scalers`` is the optimizer's starting point in scaler
        space; it gets passed through the same ``fit_all_coeffs`` path
        the cost function uses on every iteration, so the workers see
        physical coefficient values instead of raw optimizer offsets.
        """
        if self.pool is None or not self.shocks2run:
            return
        log_opt_rates = initial_scalers + self.x0
        x = self.fit_all_coeffs(np.exp(log_opt_rates))
        if x is None:
            return
        warmup_args = (
            self._build_var_dict(), self.coef_opt, x, self.shocks2run[0],
        )
        self.pool.map(_pool_calculate_residuals, [warmup_args] * n_workers)

    def __call__(self, s, optimizing=True):
        def append_output(output_dict, calc_resid_output):
            for key in calc_resid_output:
                if key not in output_dict:
                    output_dict[key] = []

                output_dict[key].append(calc_resid_output[key])

            return output_dict

        if self.__abort:
            self._log("\nOptimization aborted")
            raise Exception("Optimization terminated by user")

        log_opt_rates = s + self.x0
        x = self.fit_all_coeffs(np.exp(log_opt_rates))
        if x is None:
            return np.inf

        output_dict = {}

        var_dict = self._build_var_dict()

        display_ind_var = None
        display_observable = None
        display_t_offset = None
        active_display_shock = (
            self._display_shock_provider()
            if self._display_shock_provider is not None
            else None
        )

        t_unc_bound = float(self.t_unc[1])
        parametric = (
            not self.random_t_uncertainty
            and t_unc_bound > 1e-12
            and len(self.shocks2run) >= 1
        )
        if parametric:
            probe = self._dispatch(x, var_dict)
            t_star = np.array(
                [(o.get("t_unc") or 0.0) for o in probe], dtype=float
            )
            conditions = [
                (s.T_reactor, s.P_reactor, s.thermo_mix) for s in self.shocks2run
            ]
            shifts, self._shift_model_info = regularized_shifts(
                conditions, t_star, t_unc_bound
            )
            self._maybe_log_shift_model()
            calc_resid_outputs = self._dispatch(x, var_dict, fixed_shifts=shifts)
        else:
            calc_resid_outputs = self._dispatch(x, var_dict)

        for calc_resid_output, shock in zip(calc_resid_outputs, self.shocks2run):
            append_output(output_dict, calc_resid_output)
            shock.last_t_unc = calc_resid_output.get("t_unc")
            shock.last_loss_alpha = calc_resid_output.get("loss_alpha")
            if shock is active_display_shock:
                display_ind_var = calc_resid_output["independent_var"]
                display_observable = calc_resid_output["observable"]
                display_t_offset = shock.opt_time_offset + (
                    shock.last_t_unc or 0.0
                )

        loss_resid = np.array(output_dict["loss"])
        exp_loss_alpha = np.array(output_dict["loss_alpha"])

        loss_alpha = self.cost_settings.loss_alpha
        if loss_alpha == 3.0:
            if np.size(loss_resid) <= 2:  # optimizing only a few experiments, use SSE
                loss_alpha = 2.0

            else:  # alpha is based on residual loss function, not great, but it's super slow otherwise
                loss_alpha_fcn = lambda alpha: self.calculate_obj_fcn(
                    x,
                    loss_resid,
                    alpha,
                    log_opt_rates,
                    output_dict,
                    obj_fcn_type="Residual",
                )

                full_alpha_bounds = np.array([-100.0, 2.0])
                alpha_bounds = _narrow_bounds(
                    full_alpha_bounds, self._last_aggregate_loss_alpha,
                )
                res = minimize_scalar(
                    loss_alpha_fcn, bounds=alpha_bounds, method="bounded",
                )
                if _hit_edge(res.x, alpha_bounds) and alpha_bounds is not full_alpha_bounds:
                    res = minimize_scalar(
                        loss_alpha_fcn, bounds=full_alpha_bounds, method="bounded",
                    )
                loss_alpha = res.x
                self._last_aggregate_loss_alpha = loss_alpha

        # testing loss alphas
        # print([loss_alpha, *exp_loss_alpha])
        obj_fcn = self.calculate_obj_fcn(
            x,
            loss_resid,
            loss_alpha,
            log_opt_rates,
            output_dict,
            obj_fcn_type=self.cost_settings.obj_fcn_type,
        )

        # For updating
        self.i += 1
        if not optimizing or self.i % 1 == 0:  # 5 == 0: # updates plot every 5
            if obj_fcn == 0 and self.cost_settings.obj_fcn_type != "Bayesian":
                obj_fcn = np.inf

            stat_plot = {
                "shocks2run": self.shocks2run,
                "resid": output_dict["resid"],
                "resid_outlier": self.loss_outlier,
                "weights": output_dict["weights"],
            }

            if "KDE" in output_dict:
                stat_plot["KDE"] = output_dict["KDE"]
                allResid = np.concatenate(output_dict["resid"], axis=0)

                stat_plot["fit_result"] = fitres = self.dist.fit(allResid)
                stat_plot["QQ"] = []
                for resid in stat_plot["resid"]:
                    QQ = stats.probplot(
                        resid, sparams=fitres, dist=self.dist, fit=False
                    )
                    QQ = np.array(QQ).T
                    stat_plot["QQ"].append(QQ)

            ode_errors = [
                e for e in output_dict.get("ode_error", []) if e
            ]
            ode_error = _aggregate_ode_errors(ode_errors)
            update = {
                "type": self.opt_type,
                "i": self.i,
                "obj_fcn": obj_fcn,
                "stat_plot": stat_plot,
                "s": s,
                "x": x,
                "coef_opt": self.coef_opt,
                "ind_var": display_ind_var,
                "observable": display_observable,
                "display_t_offset": display_t_offset,
                "ode_error": ode_error,
            }

            self._progress(update)

        if optimizing:
            return obj_fcn
        else:
            return obj_fcn, x, output_dict["shock"]

    def calculate_obj_fcn(
        self,
        x,
        loss_resid,
        alpha,
        log_opt_rates,
        output_dict,
        obj_fcn_type="Residual",
        loss_outlier=0,
    ):
        """Aggregate per-experiment losses into the optimizer's scalar objective.

        Args:
            x: Fitted coefficients passed through to the Bayesian
                evaluator; unused for residual objectives.
            loss_resid: Per-experiment residual scalars.
            alpha: Adaptive-loss shape parameter, refined in place.
            log_opt_rates: Log-scaled rates for Bayesian priors.
            output_dict: Aggregated per-shock outputs (used to pull
                Bayesian weights).
            obj_fcn_type: ``"Residual"`` or ``"Bayesian"``.
            loss_outlier: Outlier mask threshold for the residual
                aggregator.

        Returns:
            Scalar objective. Lower is better.
        """
        C = self.cost_settings.loss_c
        self.loss_outlier = loss_outlier

        # If any shock returned an inf loss (degenerate sim), the
        # objective is undefined at this point — return inf so the
        # optimizer can reject it. Aggregating finite + inf via
        # downstream weighting would produce nonsense for the Bayesian
        # path (CheKiPEUQ can't ingest inf bounds).
        if np.any(~np.isfinite(np.asarray(loss_resid, dtype=float))):
            return np.inf

        if np.size(loss_resid) == 1:  # optimizing single experiment
            loss_outlier = 0
            loss_exp = loss_resid
        else:  # optimizing multiple experiments
            loss_min = loss_resid.min()
            exp_loss_weights, C, alpha = adaptive_weights(
                loss_resid - loss_min, C_scalar=C, alpha=alpha
            )
            loss_exp = exp_loss_weights * (loss_resid - loss_min) ** 2

        self.loss_outlier = loss_outlier

        if obj_fcn_type == "Residual":
            if np.size(loss_resid) == 1:  # optimizing single experiment
                obj_fcn = loss_exp[0]
            else:
                loss_exp = loss_exp - loss_exp.min() + loss_min
                # obj_fcn = np.median(loss_exp)
                obj_fcn = np.average(loss_exp)

        elif obj_fcn_type == "Bayesian":
            if np.size(loss_resid) == 1:  # optimizing single experiment
                Bayesian_weights = np.array(
                    output_dict["aggregate_weights"], dtype=object
                ).flatten()
            else:
                loss_exp = rescale_loss_fcn(loss_resid, loss_exp)
                aggregate_weights = np.array(
                    output_dict["aggregate_weights"], dtype=object
                )
                exp_loss_weights, C, alpha = adaptive_weights(
                    loss_resid, C_scalar=C, alpha=alpha
                )

                # SSE = penalized_loss_fcn(loss_resid, mu=loss_min, use_penalty=False)
                # SSE = rescale_loss_fcn(loss_resid, SSE)
                # exp_loss_weights = loss_exp/SSE # comparison is between selected loss fcn and SSE (L2 loss)

                Bayesian_weights = np.concatenate(
                    aggregate_weights.T * exp_loss_weights, axis=0
                ).flatten()

            # need to normalize weight values between iterations
            Bayesian_weights = Bayesian_weights / Bayesian_weights.sum()

            obj_fcn = self.CheKiPEUQ_Frhodo_interface.evaluate(
                log_opt_rates=log_opt_rates,
                x=x,
                output_dict=output_dict,
                bayesian_weights=Bayesian_weights,
                iteration_num=self.i,
            )

        return obj_fcn

    def fit_all_coeffs(self, all_rates):
        """Convert optimizer-space rates into per-reaction Cantera coefficients.

        Walks ``rxn_coef_opt`` and calls
        :func:`~frhodo.simulation.mechanism.fit_coeffs.fit_coeffs` for
        each reaction with its slice of ``all_rates``. When a worker
        pool is available and more than one reaction needs fitting, the
        per-reaction calls are dispatched across the pool.

        Returns:
            Flat coefficient vector with each reaction's coefficients
            concatenated, or ``None`` if any per-reaction fit failed.
        """
        args_per_rxn = []
        i = 0
        for rxn_coef in self.rxn_coef_opt:
            T_len = len(rxn_coef["T"])
            args_per_rxn.append((
                all_rates[i : i + T_len],
                rxn_coef["T"],
                rxn_coef["P"],
                rxn_coef["X"],
                rxn_coef["rxnIdx"],
                rxn_coef["key"],
                rxn_coef["coefName"],
                rxn_coef["is_falloff_limit"],
                [rxn_coef["coef_bnds"]["lower"], rxn_coef["coef_bnds"]["upper"]],
            ))
            i += T_len

        if not args_per_rxn:
            return np.array([])

        use_pool = (
            self.multiprocessing
            and self.pool is not None
            and len(args_per_rxn) > 1
        )
        if use_pool:
            per_rxn_coeffs = self.pool.map(_pool_fit_coeffs, args_per_rxn)
        else:
            per_rxn_coeffs = [fit_coeffs(*args, self.mech) for args in args_per_rxn]

        if any(c is None for c in per_rxn_coeffs):
            return None

        return np.concatenate(per_rxn_coeffs)
