# This file is part of Frhodo. Copyright © 2020, UChicago Argonne, LLC
# and licensed under BSD-3-Clause. See License.txt in the top-level
# directory for license and copyright information.
"""Cost-function machinery: residual evaluation, weighting, loss scaling.

The :class:`CostFunction` instance is what the algorithms in
:mod:`frhodo.optimize.algorithms` call once per iteration; it owns
the worker pool, the parameter unpacking, and the loss-shape choice
(residual / Bayesian / adaptive).
"""
import contextlib
import io
import numpy as np
import nlopt
from scipy.optimize import minimize_scalar, brentq
from scipy.interpolate import CubicSpline
from scipy.special import expit
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
from frhodo.optimize.cost.aggregation import (
    LOSS_C_FLOOR_K,
    coverage_weights,
    model_error_floor,
    solve_m_location,
)
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

    Numba cache safety is handled by the staged warmup in
    :meth:`CostFunction.warmup_workers` — one worker compiles and
    writes the on-disk kernel cache alone before the rest fan out as
    readers. (A per-worker ``NUMBA_CACHE_DIR`` cannot work here: spawn
    workers import this module — fixing numba's cache locators — before
    any pool initializer runs.)
    """

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

# Wall-clock bound on shipping the all-shocks "current" sim traces to the
# GUI overlay; "start"/"best" ship as deltas and ignore this.
OVERLAY_TRACE_INTERVAL = 0.2


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
    penalty_fraction=T_UNC_PENALTY_FRACTION,
):
    """Brentq root-find on ``dL/dτ = 0`` over ``τ ∈ full_bounds``.

    ``L_fit(τ) = Σ wᵢ_agg · soft_w(tᵢ, τ) · rᵢ²`` where ``soft_w`` is a
    pair of logistic sigmoids around the sim-domain edges. As τ slides
    the window past an experiment point, that point's contribution
    smoothly fades from 1 to 0 over the smoothing scale ε, so ``L_fit``
    and its derivative are C¹ in τ and brentq applies at any bracket
    width.

    ``Penalty(τ) = α₁·S(τ) + α₂·S(τ)²`` with ``S(τ) = Σ wᵢ_agg · (1 −
    soft_w)`` — aggregate weight of points the soft mask is excluding.
    ``α₁ + α₂`` are scaled so a full drop costs ``penalty_fraction ×
    L_fit(0)``. Disincentivizes large shifts that ignore weighted
    points without forbidding them when the fit gain is larger than
    the penalty.

    ``agg_weights = user_weights · adaptive_loss_weights`` are frozen
    at a deterministic anchor: phase 1 anchors at τ = 0 and coarse-scans
    the grid for the best τ; phase 2 re-freezes the weights there. The
    result depends only on the current simulation, never on evaluation
    history. Points outside the anchor soft-window keep their user
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

    # ε = half the median exp-grid spacing — soft mask transitions over
    # roughly one data point. Floor avoids divide-by-zero on degenerate
    # grids.
    dt = np.diff(t_exp)
    if dt.size:
        eps = max(0.5 * float(np.median(dt)), 1e-12)
    else:
        eps = 1e-12

    def soft_w_components(shift):
        u = (t_exp - sim_lo - shift) / eps
        v = (sim_hi + shift - t_exp) / eps
        sig_lo = expit(u)
        sig_hi = expit(v)
        sw = sig_lo * sig_hi

        return sig_lo, sig_hi, sw

    f_deriv = f_interp.derivative()

    def build_objectives(anchor_tau):
        """Loss closures with adaptive weights frozen at the anchor τ."""
        anchor_shift = t_offset + anchor_tau
        _, _, sw_anchor = soft_w_components(anchor_shift)
        valid = sw_anchor > 1e-3
        loss_weights = np.ones_like(weights, dtype=float)
        if valid.sum() >= 2:
            resid_anchor = obs_exp[valid] - f_interp(t_exp[valid] - anchor_shift)
            w_anchor = (weights * sw_anchor)[valid]
            lw_valid, _, _ = adaptive_weights(
                resid_anchor, weights=w_anchor, C_scalar=loss_c, alpha=loss_alpha,
            )
            loss_weights[valid] = lw_valid
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

        return L_total_at, dL_dtau

    # L_total can be multi-modal — the penalty creates a barrier around
    # the "drop everything" region, but the unpenalized fit may dip again
    # at large τ where only the plateau is in window. Grid-scan dL/dτ for
    # every sign change, brentq each, and return the argmin of L_total
    # over {bounds, anchor, all roots}.
    n_grid = int(np.clip((full_hi - full_lo) / eps * 4, 11, 100))
    grid = np.linspace(full_lo, full_hi, n_grid)

    # Phase 1: τ=0-anchored coarse scan picks the deterministic anchor.
    L_total_phase1, _ = build_objectives(0.0)
    phase1_vals = np.array([L_total_phase1(t) for t in grid])
    anchor_tau = float(grid[int(np.argmin(phase1_vals))])

    # Phase 2: weights re-frozen at the anchor drive the actual solve.
    L_total_at, dL_dtau = build_objectives(anchor_tau)
    dL_vals = np.array([dL_dtau(t) for t in grid])

    candidates = [full_lo, full_hi, anchor_tau]
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
        def rxn_sort_key(v):
            if v.isdigit():
                return int(v)

            return 0

        rxns = ",".join(sorted(rxn_set, key=rxn_sort_key))

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
    if parts:
        head = parts[0]
    else:
        head = ""
    rxn_part = next(
        (p.split(":", 1)[1].strip() for p in parts if p.startswith("Suggested Reactions")),
        None,
    )
    if rxn_part:
        return f"ODE: {head}; rxns {rxn_part}"

    return f"ODE: {head}"


def _finite_loss_alpha(var: dict) -> float:
    raw = var.get("loss_alpha", 2.0)
    if isinstance(raw, str):
        return 2.0

    return float(raw)


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
    if obs_sim.size:
        constant_value = float(obs_sim[0, 0])
    else:
        constant_value = 0.0

    output = {
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
        "loss_alpha": _finite_loss_alpha(var),
        "ode_error": ode_error,
    }

    return output


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
        sigma_bar=1.0,
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

        else:
            raise ValueError(f"unknown residual scale {scale!r}")

        # Standardize by the shock's data-derived noise scale so losses
        # are dimensionless and cross-shock comparable.
        resid = resid / sigma_bar

        # An overflowed sim trace can interpolate to non-finite values;
        # the objective is undefined there, so signal inf for the
        # optimizer to reject.
        if not np.all(np.isfinite(resid)):
            if verbose:
                output = {
                    "wsse": np.inf,
                    "resid": np.array([0.0]),
                    "resid_outlier": 0.0,
                    "loss": np.inf,
                    "weights": np.array([1.0]),
                    "aggregate_weights": np.array([1.0]),
                    "obs_sim_interp": np.array([0.0]),
                    "obs_exp": np.array([0.0]),
                    "obs_bounds": [],
                }

                return output

            return np.inf

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
        degenerate = _degenerate_trace_output(
            shock, ind_var, obs_sim, coef_opt, var,
            ode_error=_summarize_ode_failure(verbose),
        )

        return degenerate
    f_interp = CubicSpline(ind_var.flatten(), obs_sim.flatten())

    weights = shock.weights_trim
    obs_exp = shock.exp_data_trim
    obs_bounds = []
    if var["obj_fcn_type"] == "Bayesian":
        obs_bounds = shock.abs_uncertainties_trim

    s_bar = float(getattr(shock, "sigma_total", 1.0) or 1.0)

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
            sigma_bar=s_bar,
            opt_type=var["obj_fcn_type"],
        )

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
            )

        if t_unc is None:
            res = minimize_scalar(
                time_adj_func,
                bounds=var["t_unc"] / 10**t_unc_OoM,
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
            sigma_bar=s_bar,
            opt_type=var["obj_fcn_type"],
        )

        res = minimize_scalar(
            loss_alpha_fcn,
            bounds=np.array([-100.0, 2.0]),
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
        sigma_bar=s_bar,
        opt_type=var["obj_fcn_type"],
        verbose=True,
    )

    output["shock"] = shock
    output["independent_var"] = ind_var
    output["observable"] = obs_sim
    output["t_unc"] = t_unc
    output["loss_alpha"] = loss_alpha
    output["ode_error"] = None

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
        self.cost_settings = inputs.cost_settings
        self._user_weights = np.array([
            float(getattr(s2, "scalar_weight", 1.0) or 1.0)
            for s2 in inputs.shocks2run
        ])
        if inputs.shocks2run:
            self._cov_features = np.column_stack([
                [1000.0 / float(s2.T_reactor) for s2 in inputs.shocks2run],
                [np.log10(float(s2.P_reactor)) for s2 in inputs.shocks2run],
            ])
        else:
            self._cov_features = None
        self._mloc_diag = None
        self._model_error_floor = 0.0
        self._start_loss_raw = None
        self._last_t_star = None
        self._t_unc_mode = "fixed"
        self._last_agg_weights = None
        self._overlay_start_traces = None
        self._overlay_best_obj = np.inf
        self._last_overlay_emit = 0.0
        self._display_shock_provider = display_shock_provider
        self.pool = pool
        if pool is not None:
            self.multiprocessing = inputs.multiprocessing
        else:
            self.multiprocessing = False
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

        Dispatches one full ``calculate_residuals`` per worker on a
        representative shock so every worker compiles its ``@njit``
        kernels up front, in parallel — wall time is ~max(per-worker
        compile) and the optimizer's iterations run on warm kernels.

        ``initial_scalers`` is the optimizer's starting point in scaler
        space; it gets passed through the same ``fit_all_coeffs`` path
        the cost function uses on every iteration, so the workers see
        physical coefficient values.

        The warmup is staged so the ``@njit(cache=True)`` on-disk kernel
        cache is always written by a single worker at a time (numba's
        cache rename is not atomic under concurrent Windows writers):
        every per-reaction fit task runs alone first — covering the
        pooled ``fit_all_coeffs`` path for each reaction type — then one
        residual task compiles that path's kernels, and only then does
        the full fan-out run, finding a warm cache and reading only.
        """
        if self.pool is None or not self.shocks2run:
            return
        log_opt_rates = initial_scalers + self.x0
        all_rates = np.exp(log_opt_rates)
        for fit_args in self._build_fit_args(all_rates):
            self.pool.map(_pool_fit_coeffs, [fit_args])
        x = self.fit_all_coeffs(all_rates)
        if x is None:
            return
        warmup_args = (
            self._build_var_dict(), self.coef_opt, x, self.shocks2run[0],
        )
        self.pool.map(_pool_calculate_residuals, [warmup_args])
        if n_workers > 1:
            self.pool.map(_pool_calculate_residuals, [warmup_args] * n_workers)

    def calibrate_error_floor(self, initial_scalers: np.ndarray) -> None:
        """Set each shock's standardizing scale to its total expected error.

        One probe evaluation at the optimizer start point measures each
        shock's achieved residual scale r_s; the campaign floor E is a
        low quantile of the excess of r_s over measurement noise, and
        every shock standardizes by σ_total = sqrt(σ̄² + E²) thereafter.
        Measurement noise alone over-amplifies the cleanest shocks —
        irreducible model and numerical error dominates their tiny σ̄ —
        manufacturing an upper loss tail on campaigns the model fits
        well. Runs once, before the optimizer starts, so the objective
        remains a pure function of x.
        """
        if not self.shocks2run:
            return
        log_opt_rates = initial_scalers + self.x0
        all_rates = np.exp(log_opt_rates)
        x = self.fit_all_coeffs(all_rates)
        if x is None:
            self._log(
                "Model-error floor skipped: start-point coefficient fit failed"
            )

            return
        outputs = self._dispatch(x, self._build_var_dict())
        sigma_bars = np.array([
            float(getattr(s2, "sigma_bar", 1.0) or 1.0) for s2 in self.shocks2run
        ])
        # The probe's losses are standardized by whatever scale each
        # shock currently carries; multiply by that same scale — not
        # σ̄ — so recalibration at a stage boundary sees true raw
        # residual scales.
        sigma_div = np.array([
            float(getattr(s2, "sigma_total", 1.0) or 1.0) for s2 in self.shocks2run
        ])
        losses = np.array([float(o["loss"]) for o in outputs])
        resid_scales = losses * sigma_div
        if self._start_loss_raw is None:
            self._start_loss_raw = resid_scales
        floor = model_error_floor(resid_scales, sigma_bars)
        self._model_error_floor = floor
        for shock, s_bar in zip(self.shocks2run, sigma_bars):
            shock.sigma_total = float(np.sqrt(s_bar * s_bar + floor * floor))
        self._log(f"Model-error floor (campaign): {floor:.4g}")

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
        if self._display_shock_provider is not None:
            active_display_shock = self._display_shock_provider()
        else:
            active_display_shock = None

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
            self._last_t_star = t_star
            self._t_unc_mode = "parametric"
            calc_resid_outputs = self._dispatch(x, var_dict, fixed_shifts=shifts)
        else:
            self._last_t_star = None
            if self.random_t_uncertainty and t_unc_bound > 1e-12:
                self._t_unc_mode = "independent"
            else:
                self._t_unc_mode = "fixed"
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

        if self.cost_settings.obj_fcn_type == "Bayesian":
            # Multiply by the same scale resid_func divided by, so the
            # frozen Bayesian flow sees unstandardized losses exactly.
            sigma_totals = np.array([
                float(getattr(s2, "sigma_total", 1.0) or 1.0)
                for s2 in self.shocks2run
            ])
            obj_fcn = self._bayesian_obj_fcn(
                x, loss_resid * sigma_totals, log_opt_rates, output_dict,
            )
        else:
            obj_fcn = self._residual_obj_fcn(loss_resid, output_dict)

        # For updating
        self.i += 1
        if not optimizing or self.i % 1 == 0:  # 5 == 0: # updates plot every 5
            if obj_fcn == 0 and self.cost_settings.obj_fcn_type != "Bayesian":
                obj_fcn = np.inf

            stat_plot = {
                "shocks2run": self.shocks2run,
                "per_shock": self._mloc_diag,
            }

            sim_traces = self._overlay_sim_traces(output_dict, obj_fcn)

            ode_errors = [
                e for e in output_dict.get("ode_error", []) if e
            ]
            ode_error = _aggregate_ode_errors(ode_errors)
            update = {
                "type": self.opt_type,
                "i": self.i,
                "obj_fcn": obj_fcn,
                "stat_plot": stat_plot,
                "sim_traces": sim_traces,
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

        shocks_out = output_dict["shock"]

        return obj_fcn, x, shocks_out

    def _collect_shock_traces(self, output_dict) -> list:
        """One flattened sim trace per shock, in shocks2run order, each
        carrying its own display time offset."""
        iv_list = output_dict.get("independent_var", [])
        obs_list = output_dict.get("observable", [])
        traces = []
        for i, shock in enumerate(self.shocks2run):
            t = np.asarray(iv_list[i], dtype=float).reshape(-1)
            obs = np.asarray(obs_list[i], dtype=float).reshape(-1)
            t_offset = float(shock.opt_time_offset) + float(
                getattr(shock, "last_t_unc", 0.0) or 0.0
            )
            num = int(getattr(shock, "num", i + 1) or (i + 1))
            traces.append({"num": num, "t": t, "obs": obs,
                           "t_offset": t_offset})

        return traces

    def _overlay_sim_traces(self, output_dict, obj_fcn) -> dict:
        """Per-shock sim traces for the signal-plot overlay, shipped as
        deltas: ``start`` once at the first evaluation, ``best`` when the
        incumbent improves, ``current`` no more than every
        ``OVERLAY_TRACE_INTERVAL`` seconds. Absent keys mean the GUI
        keeps its cached copy.
        """
        if "observable" not in output_dict or "independent_var" not in output_dict:
            return {}

        need_start = self._overlay_start_traces is None
        improved = obj_fcn < self._overlay_best_obj
        now = timer()
        need_current = now - self._last_overlay_emit > OVERLAY_TRACE_INTERVAL
        if not (need_start or improved or need_current):
            return {}

        traces = self._collect_shock_traces(output_dict)
        sim_traces = {}
        if need_start:
            self._overlay_start_traces = traces
            sim_traces["start"] = traces
        if improved:
            self._overlay_best_obj = obj_fcn
            sim_traces["best"] = traces
        if need_current:
            self._last_overlay_emit = now
            sim_traces["current"] = traces

        return sim_traces

    def _residual_obj_fcn(self, loss_resid, output_dict):
        """Aggregate per-shock losses into the Residual objective.

        The objective is the legacy adaptive aggregation on
        unstandardized losses, averaged under user × coverage weights.
        The M-location solve runs on the standardized losses purely as
        the per-shock report (z, IRLS weights) carried in the progress
        payload — it never feeds the optimizer.
        """
        losses_std = np.asarray(loss_resid, dtype=float)
        if not np.all(np.isfinite(losses_std)):
            self._mloc_diag = None

            return np.inf

        n = losses_std.size
        user_w = self._user_weights
        sigma_totals = np.array([
            float(getattr(s2, "sigma_total", 1.0) or 1.0)
            for s2 in self.shocks2run
        ])
        losses_raw = losses_std * sigma_totals

        cov = np.ones(n)
        if (
            self.cost_settings.coverage_weighting
            and self._cov_features is not None
            and n >= 4
        ):
            cov = coverage_weights(self._cov_features, user_w)

        obj_fcn = self._legacy_residual_aggregate(losses_raw, user_w * cov)

        n_coef = len(self.coef_opt)
        eff_dofs = []
        for agg_w, shock in zip(output_dict["aggregate_weights"], self.shocks2run):
            dof = max(float(np.sum(agg_w)) - n_coef, 1.0)
            tau = float(getattr(shock, "corr_length", 1.0) or 1.0)
            eff_dofs.append(dof / max(tau, 1.0))
        c_floor = LOSS_C_FLOOR_K / np.sqrt(2.0 * float(np.median(eff_dofs)))

        if self._start_loss_raw is None:
            start_raw = np.full(n, np.nan)
        else:
            start_raw = self._start_loss_raw

        trim_w = self._last_agg_weights
        if trim_w is None or np.size(trim_w) != n:
            trim_w = np.ones(n)
        t_unc_applied = np.array([
            float(v or 0.0) for v in output_dict["t_unc"]
        ])
        if self._last_t_star is None or np.size(self._last_t_star) != n:
            t_unc_star = t_unc_applied
        else:
            t_unc_star = np.asarray(self._last_t_star, dtype=float)

        mloc = solve_m_location(losses_std, user_w * cov, c_floor=c_floor)
        self._mloc_diag = {
            "loss": losses_std,
            "loss_raw": losses_raw,
            "loss_raw_start": start_raw,
            "z": mloc.z,
            "irls_weights": mloc.irls_weights,
            "coverage": cov,
            "user": user_w,
            "trim_weights": np.asarray(trim_w, dtype=float),
            "t_unc": t_unc_applied,
            "t_unc_star": t_unc_star,
            "t_unc_mode": self._t_unc_mode,
            "t_unc_bounds": np.asarray(self.t_unc, dtype=float),
            "t_offset_base": np.array([
                float(getattr(s2, "opt_time_offset", 0.0) or 0.0)
                for s2 in self.shocks2run
            ]),
            "mu": mloc.mu,
            "alpha": mloc.alpha,
            "c": mloc.c,
            "c_floor": c_floor,
            "sigma_bar": np.array([
                float(getattr(s2, "sigma_bar", 1.0) or 1.0)
                for s2 in self.shocks2run
            ]),
            "sigma_total": sigma_totals,
            "model_error_floor": self._model_error_floor,
            "T": np.array([float(s2.T_reactor) for s2 in self.shocks2run]),
            "P": np.array([float(s2.P_reactor) for s2 in self.shocks2run]),
        }

        return obj_fcn

    def _legacy_residual_aggregate(self, losses, weights):
        """Adaptive experiment-level aggregation of raw per-shock losses.

        Per-shock losses are shifted to the campaign minimum, reweighted
        by the adaptive loss, and averaged under the supplied weights.
        The aggregate loss shape solves on full bounds every call, so
        the objective stays a pure function of x.
        """
        if losses.size == 1:
            self._last_agg_weights = np.ones(1)

            return float(losses[0])

        loss_alpha = self.cost_settings.loss_alpha
        if loss_alpha == 3.0:
            if losses.size <= 2:
                loss_alpha = 2.0
            else:
                res = minimize_scalar(
                    lambda a: self._aggregate_at(losses, weights, a),
                    bounds=(-100.0, 2.0), method="bounded",
                )
                loss_alpha = float(res.x)

        return self._aggregate_at(losses, weights, loss_alpha)

    def _aggregate_at(self, losses, weights, alpha):
        """Legacy aggregate at a fixed loss shape: weighted average of
        adaptively-reweighted squared excursions above the campaign
        minimum, re-anchored to the loss scale.

        Stashes the adaptive per-shock weights; the last call in
        ``_legacy_residual_aggregate`` is at the solved shape, so the
        stash always reflects the returned objective."""
        loss_min = losses.min()
        exp_w, _C, _alpha = adaptive_weights(
            losses - loss_min, C_scalar=self.cost_settings.loss_c, alpha=alpha,
        )
        self._last_agg_weights = exp_w
        loss_exp = exp_w * (losses - loss_min) ** 2
        loss_exp = loss_exp - loss_exp.min() + loss_min
        value = float(np.average(loss_exp, weights=weights))

        return value

    def _bayesian_obj_fcn(self, x, loss_resid, log_opt_rates, output_dict):
        """Frozen legacy flow for Bayesian mode, pending its disposition.

        Consumes unstandardized losses and the legacy aggregation so
        Bayesian-mode behavior matches its golden pins exactly.
        """
        loss_alpha = self.cost_settings.loss_alpha
        if loss_alpha == 3.0:
            if np.size(loss_resid) <= 2:  # optimizing only a few experiments, use SSE
                loss_alpha = 2.0

            else:  # alpha based on the legacy residual aggregation
                loss_alpha_fcn = lambda alpha: self._legacy_obj_fcn(
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

        result = self._legacy_obj_fcn(
            x,
            loss_resid,
            loss_alpha,
            log_opt_rates,
            output_dict,
            obj_fcn_type="Bayesian",
        )

        return result

    def _legacy_obj_fcn(
        self,
        x,
        loss_resid,
        alpha,
        log_opt_rates,
        output_dict,
        obj_fcn_type="Residual",
        loss_outlier=0,
    ):
        """Legacy experiment-level aggregation, retained for Bayesian mode.

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

        else:
            raise ValueError(f"unknown objective type {obj_fcn_type!r}")

        return obj_fcn

    def _build_fit_args(self, all_rates):
        """Per-reaction ``fit_coeffs`` argument tuples for ``all_rates``."""
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

        return args_per_rxn

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
        args_per_rxn = self._build_fit_args(all_rates)

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
