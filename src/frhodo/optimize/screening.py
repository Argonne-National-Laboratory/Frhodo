"""Pre-fit reaction screening: importance, leverage, and identifiability.

Scores are built from normalized trace sensitivities
``S[i, j] = ∂ln(obs_i)/∂ln(k_j)`` (one matrix per shock, from
:func:`frhodo.simulation.shock.sensitivity.compute_sensitivity`) and the
start-point weighted residuals. Two per-reaction scores:

* importance — weighted mean of ``|S|`` over the trace: how strongly the
  reaction can move the observable where the fit window has weight.
* leverage — |first-order objective slope| ``|∂L_s/∂ln k_j|`` under the
  shock's standardized residual metric: how much the reaction can fix
  the current misfit.

Screening assumes every reaction can move — no user-set bounds are
required. The suggested set keeps reactions whose best-case improvement
under an assumed factor-``ASSUMED_MOVABILITY`` rate change exceeds a
fraction of the start loss. The identifiability report (SVD of the
stacked per-shock slope matrix) states how many rate *directions* the
campaign constrains.
"""
import numpy as np
from pydantic import BaseModel, ConfigDict
from scipy.interpolate import CubicSpline

from frhodo.optimize.cost.aggregation import coverage_weights
from frhodo.optimize.cost.fit_fcn import _log_ratio
from frhodo.optimize.shock_prep import shock_sigma_bar, trim_shocks
from frhodo.simulation.shock.incident_shock_reactor import run_incident_shock
from frhodo.simulation.shock.sensitivity import (
    compute_sensitivity,
    sensitivity_observable,
)
from frhodo.simulation.shock.state import zero_d_mode_from_label
from frhodo.simulation.shock.zero_d_reactor import run_zero_d



SUGGEST_LOSS_FRACTION = 0.01
SPECTRAL_GAP_SIGNIFICANCE = 3.0
# Movability every reaction is assumed to have when judging whether its
# best-case improvement is material (a factor-2 rate change).
ASSUMED_MOVABILITY = 2.0


class ScreeningResult(BaseModel):
    """Per-reaction screening scores and campaign identifiability."""
    rxn_indices: list[int]
    importance: list[float]
    importance_max: list[float]
    leverage: list[float]
    suggested: list[bool]
    shock_nums: list[int]
    slopes: list[list[float]]
    footprints: list[list[float]]
    spectral_gap_rank: int | None
    loss_start: float
    skipped_shocks: list[int]
    singular_values: list[float]
    effective_rank: float

    model_config = ConfigDict(frozen=True)

    def ranking(self) -> list[int]:
        """Reaction indices sorted by descending leverage."""
        order = np.argsort(self.leverage)[::-1]
        ranked = [self.rxn_indices[i] for i in order]

        return ranked


def shock_scores(S: np.ndarray, weights: np.ndarray, resid: np.ndarray,
                 chain: np.ndarray | None = None):
    """Per-reaction scores for one shock.

    Args:
        S: ``(n_pts, n_rxns)`` normalized sensitivities
            ``∂ln(obs)/∂ln(k)`` on the residual grid.
        weights: ``(n_pts,)`` non-negative residual weights.
        resid: ``(n_pts,)`` standardized residuals at the start point.
        chain: ``(n_pts,)`` factor ``d(standardized transformed obs)/
            d ln(obs)`` converting ``S`` into residual-metric slopes;
            identity when the residual is already in ``ln obs`` units.

    Returns:
        ``(importance, slope, loss, footprint)``: importance
        ``(n_rxns,)`` from the raw ``|S|``; ``slope`` the signed
        first-order objective slope ``∂L_s/∂ln k`` per reaction
        (leverage is its magnitude); the shock's weighted mean-square
        loss; and ``footprint`` the residual-free capability row —
        weight-averaged ``|chain·S|`` in the standardized metric,
        feeding the uniqueness balance and the exclusion mask.
    """
    S = np.asarray(S, dtype=float)
    w = np.asarray(weights, dtype=float)
    r = np.asarray(resid, dtype=float)
    w_sum = w.sum()
    if w_sum <= 0 or S.size == 0:
        if S.ndim == 2:
            n_rxns = S.shape[1]
        else:
            n_rxns = 0
        importance = np.zeros(n_rxns)
        slope = np.zeros(n_rxns)
        loss = 0.0
        footprint = np.zeros(n_rxns)

        return importance, slope, loss, footprint

    if chain is None:
        S_metric = S
    else:
        S_metric = np.asarray(chain, dtype=float)[:, None] * S

    importance = np.abs(S).T @ w / w_sum
    # L_s = Σ w r² / Σ w with r = (exp_T − sim_T)/σ:
    # ∂L_s/∂ln k_j = −2 Σ w r · (∂sim_T/∂ln obs)·S / Σ w.
    slope = S_metric.T @ (w * r) * (-2.0) / w_sum
    loss = float(w @ r**2 / w_sum)
    footprint = np.abs(S_metric).T @ w / w_sum

    return importance, slope, loss, footprint


def campaign_aggregate(per_shock: list[np.ndarray],
                       shock_weights: np.ndarray) -> np.ndarray:
    """Weighted mean of per-shock score vectors."""
    stacked = np.stack(per_shock)
    W = np.asarray(shock_weights, dtype=float)

    return np.average(stacked, axis=0, weights=W)


def effective_rank(G: np.ndarray, shock_weights: np.ndarray):
    """Entropy-based effective rank of the stacked slope matrix.

    Args:
        G: ``(n_shocks, n_rxns)`` signed per-shock objective slopes.
        shock_weights: per-shock campaign weights.

    Returns:
        ``(erank, singular_values)``; ``erank = exp(H(σ/Σσ))`` counts
        the rate directions the campaign constrains.
    """
    W = np.asarray(shock_weights, dtype=float)
    G_w = np.asarray(G, dtype=float) * np.sqrt(W)[:, None]
    svals = np.linalg.svd(G_w, compute_uv=False)
    total = svals.sum()
    if total <= 0:
        return 0.0, svals

    p = svals / total
    p = p[p > 0]
    erank = float(np.exp(-(p * np.log(p)).sum()))

    return erank, svals


def spectral_gap_rank(scores: np.ndarray,
                      significance: float = SPECTRAL_GAP_SIGNIFICANCE):
    """Rank of the kink in the sorted log-score spectrum, or ``None``.

    The cut lands at the largest gap between consecutive sorted
    ``log10`` scores, and only counts when that gap exceeds
    ``significance ×`` the median inter-rank gap — a smooth spectrum has
    no kink and returns ``None``.
    """
    s = np.asarray(scores, dtype=float)
    s = np.sort(s[s > 0])[::-1]
    if s.size < 3:
        return None
    gaps = np.diff(np.log10(s)) * -1.0
    median_gap = np.median(gaps)
    largest = int(np.argmax(gaps))
    if median_gap <= 0 or gaps[largest] < significance * median_gap:
        return None

    return largest + 1


def suggested_set(leverage: np.ndarray, loss_start: float,
                  fraction: float = SUGGEST_LOSS_FRACTION) -> np.ndarray:
    """Reactions whose best-case improvement is material, assuming every
    reaction can move by a factor of ``ASSUMED_MOVABILITY``.

    Keeps reaction ``j`` when ``leverage[j] × ln(ASSUMED_MOVABILITY) >
    fraction × loss_start`` — an absolute test against the data, not a
    relative ranking, so a campaign that no reaction can improve selects
    nothing.
    """
    lev = np.asarray(leverage, dtype=float)
    if loss_start <= 0:
        return np.zeros(lev.size, dtype=bool)

    return lev * np.log(ASSUMED_MOVABILITY) > fraction * loss_start


def uniqueness_weights(slopes: np.ndarray,
                       clip=(0.2, 5.0)) -> np.ndarray:
    """Information-balancing experiment weights from slope collinearity.

    Each experiment's soft duplicate-count ``r_i = Σ_j cos²(g_i, g_j)``
    measures how many experiments constrain the same direction; the
    weight is ``1/r_i``, normalized to mean 1 and clipped. Identical
    experiments share their weight (k copies get ~1/k each); an
    experiment probing a direction nobody else touches keeps full
    weight regardless of where it sits in (T, P). Zero-influence rows
    get weight 1 before clipping — exclusion is a separate decision.
    """
    G = np.asarray(slopes, dtype=float)
    norms = np.linalg.norm(G, axis=1)
    ok = norms > 0
    unit = np.zeros_like(G)
    unit[ok] = G[ok] / norms[ok, None]
    cos2 = (unit @ unit.T) ** 2
    r = cos2.sum(axis=1)
    weights = np.ones(G.shape[0])
    weights[ok] = 1.0 / r[ok]
    weights = weights / weights.mean()
    weights = np.clip(weights, clip[0] * weights.mean(),
                      clip[1] * weights.mean())

    return weights


def experiment_influence(slopes: np.ndarray) -> np.ndarray:
    """Per-experiment total optimizable signal: the L1 norm of its
    objective-slope row."""
    influence = np.abs(np.asarray(slopes, dtype=float)).sum(axis=1)

    return influence


INFLUENCE_KEEP_FRACTION = 0.001


def influential_mask(slopes: np.ndarray,
                     fraction: float = INFLUENCE_KEEP_FRACTION) -> np.ndarray:
    """Experiments worth simulating: influence above ``fraction`` of the
    leader's. A near-zero-influence experiment cannot move the optimum —
    its loss is constant in x — so excluding it from evaluations is pure
    compute savings. The cutoff is orders of magnitude below any live
    experiment: weak or noise-inflated shocks must survive it.
    """
    influence = experiment_influence(slopes)
    if influence.size:
        top = influence.max()
    else:
        top = 0.0
    if top <= 0:
        return np.ones(influence.size, dtype=bool)

    return influence > fraction * top


# Fit-observable main names -> sensitivity observables. Heat Release
# Rate has no sensitivity backend and cannot be screened.
def _run_start_sim(mech, reactor_state, shock):
    """One reactor solve at the current mechanism; ``None`` when the
    trajectory is too short to interpolate."""
    kwargs = {
        "u_reac": shock.u2,
        "rho1": shock.rho1,
        "observable": shock.observable,
        "t_lab_save": None,
        "sim_int_f": reactor_state.sim_interp_factor,
        "ODE_solver": reactor_state.ode_solver,
        "rtol": reactor_state.ode_rtol,
        "atol": reactor_state.ode_atol,
    }
    name = reactor_state.name
    if name == "Incident Shock Reactor":
        SIM, _ = run_incident_shock(
            mech, reactor_state.t_end, shock.T_reactor, shock.P_reactor,
            shock.thermo_mix, **kwargs,
        )
    elif "0d Reactor" in name:
        kwargs["solve_energy"] = reactor_state.solve_energy
        kwargs["frozen_comp"] = reactor_state.frozen_comp
        mode = zero_d_mode_from_label(name)
        SIM, _ = run_zero_d(
            mech, mode, reactor_state.t_end, shock.T_reactor,
            shock.P_reactor, shock.thermo_mix, **kwargs,
        )
    else:
        raise ValueError(f"unknown reactor: {name!r}")
    if SIM.independent_var.size < 2:
        return None

    return SIM.independent_var, SIM.observable


def _scaled_residual(obs_exp, obs_sim, scale, bisymlog, sigma):
    """Standardized residual, valid-point mask, and the chain factor
    ``d(standardized transformed obs)/d ln(obs)`` at the simulation."""
    if scale == "Linear":
        mask = np.ones(obs_exp.size, dtype=bool)
        resid = (obs_exp - obs_sim) / sigma
        chain = obs_sim / sigma
    elif scale == "Log":
        mask = (obs_exp > 0.0) & (obs_sim > 0.0)
        resid = (np.log10(obs_exp[mask]) - np.log10(obs_sim[mask])) / sigma
        chain = np.full(resid.size, 1.0 / (np.log(10.0) * sigma))
    elif scale == "AbsoluteLog":
        mask = (obs_exp != 0.0) & (obs_sim != 0.0)
        resid = _log_ratio(obs_exp[mask], obs_sim[mask]) / sigma
        chain = np.full(resid.size, 1.0 / (np.log(10.0) * sigma))
    elif scale == "Bisymlog":
        mask = np.ones(obs_exp.size, dtype=bool)
        resid = (bisymlog.transform(obs_exp)
                 - bisymlog.transform(obs_sim)) / sigma
        h = 1e-4
        chain = (bisymlog.transform(obs_sim * np.exp(h))
                 - bisymlog.transform(obs_sim)) / (h * sigma)
    else:
        raise ValueError(f"unknown residual scale {scale!r}")

    return resid, mask, chain


def _sim_cache_key(mech, reactor_state, shock, observable, species_idx,
                   method) -> tuple:
    """Everything the trajectory + sensitivity solves depend on."""
    mix = getattr(shock, "thermo_mix", None) or {}
    key = (
        int(getattr(shock, "num", 0) or 0),
        int(getattr(mech, "coeffs_version", 0)),
        float(shock.T_reactor), float(shock.P_reactor),
        tuple(sorted((str(k), float(v)) for k, v in dict(mix).items())),
        repr(reactor_state), observable, species_idx, method,
    )

    return key


def screen_campaign(mech, shocks2run, reactor_state, cost_settings, *,
                    method="auto", n_workers=1,
                    suggest_fraction=SUGGEST_LOSS_FRACTION,
                    sim_cache: dict | None = None) -> ScreeningResult:
    """Screen every reaction against the campaign at the start mechanism.

    Two solves per shock (one trajectory, one sensitivity), outside the
    optimization loop. ``shocks2run`` must carry ``exp_data`` and
    ``normalized_weights``; trimming and noise scales are prepared here
    exactly as the optimizer does. ``sim_cache`` holds per-shock solve
    results keyed on the mechanism version and shock conditions, so a
    re-screen after include-toggles only solves new shocks; entries
    keyed to other mechanism versions are evicted on insert.
    """
    trim_shocks(shocks2run, cost_settings)

    importance_rows = []
    slope_rows = []
    footprint_rows = []
    losses = []
    user_w = []
    features = []
    skipped = []
    shock_nums = []
    for shock in shocks2run:
        num = int(getattr(shock, "num", 0) or 0)
        observable, species_idx = sensitivity_observable(shock, mech.gas)
        cache_key = None
        cached = None
        if sim_cache is not None:
            cache_key = _sim_cache_key(
                mech, reactor_state, shock, observable, species_idx, method,
            )
            cached = sim_cache.get(cache_key)

        if cached is not None:
            t_sim, obs_sim, t_sens, S = cached
        else:
            sim = _run_start_sim(mech, reactor_state, shock)
            if sim is None:
                skipped.append(num)
                continue
            t_sim, obs_sim = sim
            t_sens = None
            S = None

        f_interp = CubicSpline(t_sim, obs_sim)

        shift = float(getattr(shock, "opt_time_offset", 0.0) or 0.0)
        t_exp = shock.exp_data_trim[:, 0]
        obs_exp = shock.exp_data_trim[:, 1]
        weights = shock.weights_trim
        window = (t_exp >= t_sim[0] + shift) & (t_exp <= t_sim[-1] + shift)
        if window.sum() < 2:
            skipped.append(num)
            continue
        t_win = t_exp[window]
        obs_sim_interp = f_interp(t_win - shift)

        sigma = shock_sigma_bar(shock, cost_settings.scale)
        resid, mask, chain = _scaled_residual(
            obs_exp[window], obs_sim_interp, cost_settings.scale,
            getattr(shock, "bisymlog", None), sigma,
        )
        w = weights[window][mask]

        if S is None:
            t_sens, S = compute_sensitivity(
                mech, reactor_state, shock,
                observable=observable, species_idx=species_idx,
                method=method, n_workers=n_workers,
            )
            if sim_cache is not None:
                version = cache_key[1]
                stale = [k for k in sim_cache if k[1] != version]
                for k in stale:
                    del sim_cache[k]
                sim_cache[cache_key] = (t_sim, obs_sim, t_sens, S)

        t_eval = t_win[mask] - shift
        S_exp = np.column_stack([
            np.interp(t_eval, t_sens, S[:, j]) for j in range(S.shape[1])
        ])

        importance_s, slope_s, loss_s, footprint_s = shock_scores(
            S_exp, w, resid, chain,
        )
        importance_rows.append(importance_s)
        slope_rows.append(slope_s)
        footprint_rows.append(footprint_s)
        losses.append(loss_s)
        shock_nums.append(num)
        user_w.append(float(getattr(shock, "scalar_weight", 1.0) or 1.0))
        features.append([1000.0 / float(shock.T_reactor),
                         np.log10(float(shock.P_reactor))])

    if not importance_rows:
        raise ValueError("screening produced no usable shocks")

    user_w = np.asarray(user_w)
    W = user_w.copy()
    # Screening aggregates under the geometric proxy even in
    # uniqueness mode: uniqueness needs the slopes screening is
    # about to produce.
    if (cost_settings.experiment_weighting != "none"
            and len(importance_rows) >= 4):
        W = user_w * coverage_weights(np.asarray(features), user_w)

    importance = campaign_aggregate(importance_rows, W)
    importance_max = np.max(np.stack(importance_rows), axis=0)
    leverage = campaign_aggregate([np.abs(s) for s in slope_rows], W)
    loss_start = float(np.average(losses, weights=W))
    erank, svals = effective_rank(np.stack(slope_rows), W)

    suggested = suggested_set(leverage, loss_start, suggest_fraction)
    gap = spectral_gap_rank(leverage)

    result = ScreeningResult(
        rxn_indices=list(range(mech.gas.n_reactions)),
        importance=importance.tolist(),
        importance_max=importance_max.tolist(),
        leverage=leverage.tolist(),
        suggested=suggested.tolist(),
        shock_nums=shock_nums,
        slopes=[s.tolist() for s in slope_rows],
        footprints=[f.tolist() for f in footprint_rows],
        spectral_gap_rank=gap,
        loss_start=loss_start,
        singular_values=svals.tolist(),
        effective_rank=erank,
        skipped_shocks=skipped,
    )

    return result
