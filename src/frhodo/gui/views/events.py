"""Typed per-iteration event for the optimization-outcome views.

``build_view_context`` snapshots everything static about a run (labels,
bounds, reaction equations, the Arrhenius T grid) at worker
construction; ``compute_arrhenius_lines`` evaluates the incumbent
ln k(T) on the optimizer thread, where the mechanism's coefficients are
coherent with the evaluation that produced the update; and
``IterationEvent.from_update`` assembles the typed event the view
widgets consume. Views read only this model — never optimizer
internals.
"""
import numpy as np
from pydantic import BaseModel, ConfigDict



ARRHENIUS_T_POINTS = 25
ARRHENIUS_P_POINTS = 10


class ViewContext(BaseModel):
    """Static per-run inputs the views need beside each update."""
    param_labels: list[str]
    param_rxn: list[int]
    lower_bounds: list[float]
    upper_bounds: list[float]
    rxn_indices: list[int]
    rxn_equations: list[str]
    rxn_is_pressure_dependent: list[bool]
    rxn_band_halfwidth: list[float]
    T_grid: list[float]
    P_grid: list[float]
    P_reference: float
    ln_k_initial: list[list[list[float]]]

    model_config = ConfigDict(frozen=True)


class IterationEvent(BaseModel):
    """One optimizer iteration, reshaped for display."""
    i: int
    stage: str
    obj_fcn: float
    is_best: bool
    shock_num: list[int]
    T: list[float]
    P: list[float]
    loss_obj: list[float]
    loss_obj_start: list[float]
    z: list[float]
    irls_weights: list[float]
    coverage: list[float]
    user_weights: list[float]
    trim_weights: list[float]
    t_unc: list[float]
    t_unc_star: list[float]
    t_unc_mode: str
    t_unc_bounds: list[float]
    t_offset_base: list[float]
    scalers: list[float]
    ln_k: list[list[list[float]]]

    model_config = ConfigDict(frozen=True)

    @classmethod
    def from_update(cls, update: dict, is_best: bool) -> "IterationEvent | None":
        """Build the event from a raw progress update; ``None`` when the
        update lacks the per-shock diagnostics (degenerate iterations)."""
        stat = update.get("stat_plot") or {}
        diag = stat.get("per_shock")
        views = update.get("views") or {}
        if not diag:
            return None

        loss_raw = np.asarray(diag["loss_raw"], dtype=float)
        start_raw = np.asarray(
            diag.get("loss_raw_start", np.full_like(loss_raw, np.nan)),
            dtype=float,
        )
        shocks = stat.get("shocks2run") or []
        shock_num = [_shock_num(s, j) for j, s in enumerate(shocks)]
        n = loss_raw.size
        ones = np.ones(n)
        t_unc = np.asarray(diag.get("t_unc", np.zeros(n)), dtype=float)
        event = cls(
            i=int(update.get("i", 0)),
            stage=str(update.get("type", "")),
            obj_fcn=float(update.get("obj_fcn", np.nan)),
            is_best=is_best,
            shock_num=shock_num,
            T=np.asarray(diag["T"], dtype=float).tolist(),
            P=np.asarray(diag["P"], dtype=float).tolist(),
            loss_obj=loss_raw.tolist(),
            loss_obj_start=start_raw.tolist(),
            z=np.asarray(diag["z"], dtype=float).tolist(),
            irls_weights=np.asarray(diag["irls_weights"], dtype=float).tolist(),
            coverage=np.asarray(diag["coverage"], dtype=float).tolist(),
            user_weights=np.asarray(diag.get("user", ones), dtype=float).tolist(),
            trim_weights=np.asarray(diag.get("trim_weights", ones),
                                    dtype=float).tolist(),
            t_unc=t_unc.tolist(),
            t_unc_star=np.asarray(diag.get("t_unc_star", t_unc),
                                  dtype=float).tolist(),
            t_unc_mode=str(diag.get("t_unc_mode", "fixed")),
            t_unc_bounds=np.asarray(diag.get("t_unc_bounds", [0.0, 0.0]),
                                    dtype=float).tolist(),
            t_offset_base=np.asarray(diag.get("t_offset_base", np.zeros(n)),
                                     dtype=float).tolist(),
            scalers=np.asarray(update.get("s", []), dtype=float).tolist(),
            ln_k=views.get("ln_k", []),
        )

        return event


def _shock_num(shock, fallback: int) -> int:
    try:
        return int(shock["num"])
    except (TypeError, KeyError, ValueError):
        return fallback + 1


def _ln_k_surface(mech, rxn_indices, T_grid, P_grid, mix) -> list[list[list[float]]]:
    """ln k per reaction on the (P, T) display grid."""
    out = np.empty((len(rxn_indices), len(P_grid), len(T_grid)))
    for p, P in enumerate(P_grid):
        for j, T in enumerate(T_grid):
            mech.gas.TPX = float(T), float(P), mix
            k = mech.gas.forward_rate_constants
            for r, idx in enumerate(rxn_indices):
                out[r, p, j] = np.log(max(float(k[idx]), 1e-300))

    return out.tolist()


def _campaign_reference_state(shocks2run):
    """Display grids and reference state over the campaign conditions.

    The pressure grid spans the campaign with log-spaced headroom so the
    Arrhenius view's pressure box can interpolate within it.
    """
    T = np.array([float(s.T_reactor) for s in shocks2run])
    P = np.array([float(s.P_reactor) for s in shocks2run])
    P_med = float(np.median(P))
    mix = shocks2run[0].thermo_mix
    T_grid = np.linspace(T.min(), T.max(), ARRHENIUS_T_POINTS)
    P_grid = np.geomspace(0.5 * P.min(), 2.0 * P.max(), ARRHENIUS_P_POINTS)

    return T_grid, P_grid, P_med, mix


def build_view_context(mech, rxn_coef_opt, rxn_rate_opt, shocks2run) -> ViewContext:
    """Snapshot static view inputs; call at worker construction, while
    the mechanism still holds the start coefficients."""
    lb = np.asarray(rxn_rate_opt["bnds"]["lower"], dtype=float)
    ub = np.asarray(rxn_rate_opt["bnds"]["upper"], dtype=float)

    param_labels = []
    param_rxn = []
    rxn_indices = []
    band = {}
    pos = 0
    for rxn_coef in rxn_coef_opt:
        idx = int(rxn_coef["rxnIdx"])
        if idx not in rxn_indices:
            rxn_indices.append(idx)
        n_anchor = len(rxn_coef["T"])
        for T_anchor in rxn_coef["T"]:
            param_labels.append(f"R{idx + 1} @ {float(T_anchor):.0f} K")
            param_rxn.append(idx)
        halfwidths = np.abs(np.stack([lb[pos:pos + n_anchor],
                                      ub[pos:pos + n_anchor]]))
        band[idx] = max(float(halfwidths.max()), band.get(idx, 0.0))
        pos += n_anchor

    T_grid, P_grid, P_med, mix = _campaign_reference_state(shocks2run)
    equations = []
    pressure_dependent = []
    with mech.exclusive():
        for idx in rxn_indices:
            rxn = mech.gas.reaction(idx)
            equations.append(str(rxn.equation))
            pressure_dependent.append("(+M)" in rxn.equation
                                      or "PLOG" in type(rxn.rate).__name__.upper()
                                      or "Falloff" in type(rxn.rate).__name__
                                      or "Troe" in type(rxn.rate).__name__
                                      or "Chebyshev" in type(rxn.rate).__name__)
        ln_k0 = _ln_k_surface(mech, rxn_indices, T_grid, P_grid, mix)

    context = ViewContext(
        param_labels=param_labels,
        param_rxn=param_rxn,
        lower_bounds=lb.tolist(),
        upper_bounds=ub.tolist(),
        rxn_indices=rxn_indices,
        rxn_equations=equations,
        rxn_is_pressure_dependent=pressure_dependent,
        rxn_band_halfwidth=[band[idx] for idx in rxn_indices],
        T_grid=T_grid.tolist(),
        P_grid=P_grid.tolist(),
        P_reference=P_med,
        ln_k_initial=ln_k0,
    )

    return context


def compute_arrhenius_lines(mech, context: ViewContext, shocks2run) -> dict:
    """Incumbent ln k surfaces for the update being emitted.

    Runs on the optimizer thread inside the progress callback, where the
    mechanism's coefficients match the evaluation that produced the
    update.
    """
    mix = shocks2run[0].thermo_mix
    ln_k = _ln_k_surface(
        mech, context.rxn_indices,
        np.asarray(context.T_grid), np.asarray(context.P_grid), mix,
    )
    payload = {"ln_k": ln_k}

    return payload
