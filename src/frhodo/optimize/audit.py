"""Post-fit audit: bound saturation and flat directions at the optimum.

Runs once after the last optimization stage, while the evaluation pool
is still open. Answers two questions per coefficient slot: did the
optimizer press against its uncertainty bound, and does the objective
even respond to this direction at the optimum.
"""
import numpy as np



AT_BOUND_FRACTION = 0.01   # within this fraction of the span of a bound
FLAT_STEP_FRACTION = 0.01  # one-sided probe step, fraction of slot span
FLAT_REL_TOL = 1e-4        # |dObj|/obj below this = unconstrained
_LOG_MAX_ITEMS = 6


def slot_labels(rxn_coef_opt) -> list:
    """Human labels per scaler slot: reaction number + anchor T."""
    labels = []
    for rxn_coef in rxn_coef_opt:
        num = rxn_coef["rxnIdx"] + 1
        for T in rxn_coef["T"]:
            labels.append(f"R{num} @ {float(T):.0f} K")

    return labels


def slot_rxn_indices(rxn_coef_opt) -> list:
    """0-based reaction index per scaler slot."""
    indices = []
    for rxn_coef in rxn_coef_opt:
        for _ in rxn_coef["T"]:
            indices.append(int(rxn_coef["rxnIdx"]))

    return indices


def bound_utilization(s, lower, upper) -> np.ndarray:
    """Per-slot fraction of the available room used, side-aware.

    ``s`` is the ln-rate displacement from nominal; a positive slot is
    measured against its upper bound, a negative one against its lower.
    Slots with a degenerate (zero-width) side report 0.
    """
    s = np.asarray(s, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    util = np.zeros(s.size)
    pos = (s > 0) & (upper > 0)
    util[pos] = s[pos] / upper[pos]
    neg = (s < 0) & (lower < 0)
    util[neg] = s[neg] / lower[neg]

    return util


def _shortlist(items) -> str:
    shown = ", ".join(items[:_LOG_MAX_ITEMS])
    extra = len(items) - _LOG_MAX_ITEMS
    if extra > 0:
        shown = f"{shown}, +{extra} more"

    return shown


def audit_optimum(res: dict, inputs, fit_fun, log) -> None:
    """Attach ``audit`` to the winning stage dict and log a summary.

    Probes cost one quiet objective evaluation per slot plus a baseline
    and a final restoring evaluation; each probe steps into the
    interior so bounds are never violated. The audit lives inside the
    stage dict because run consumers iterate the stage keys.
    """
    stage = res.get("local") or res.get("global")
    if not stage or "s" not in stage:
        return

    s = np.asarray(stage["s"], dtype=float)
    lower = np.asarray(inputs.rxn_rate_opt["bnds"]["lower"], dtype=float)
    upper = np.asarray(inputs.rxn_rate_opt["bnds"]["upper"], dtype=float)
    span = upper - lower
    labels = slot_labels(inputs.rxn_coef_opt)
    rxn_indices = slot_rxn_indices(inputs.rxn_coef_opt)

    at_bounds = []
    for i in range(s.size):
        if span[i] <= 0:
            continue

        if s[i] - lower[i] <= AT_BOUND_FRACTION * span[i]:
            at_bounds.append((labels[i], "lower"))
        elif upper[i] - s[i] <= AT_BOUND_FRACTION * span[i]:
            at_bounds.append((labels[i], "upper"))

    f0 = float(fit_fun(s, quiet=True))
    flat = []
    if np.isfinite(f0) and f0 != 0:
        for i in range(s.size):
            if span[i] <= 0:
                continue

            step = FLAT_STEP_FRACTION * span[i]
            if upper[i] - s[i] < s[i] - lower[i]:
                step = -step
            probe = s.copy()
            probe[i] += step
            f_i = float(fit_fun(probe, quiet=True))
            if abs(f_i - f0) / abs(f0) < FLAT_REL_TOL:
                flat.append(labels[i])
        # Probes mutate shared state (mech coefficients, per-shock
        # solved offsets); a final evaluation at the optimum restores it.
        fit_fun(s, quiet=True)

    utilization = bound_utilization(s, lower, upper)
    rxn_utilization = {}
    for idx, u in zip(rxn_indices, utilization):
        rxn_utilization[idx] = max(rxn_utilization.get(idx, 0.0), float(u))

    stage["audit"] = {
        "at_bounds": at_bounds,
        "flat": flat,
        "utilization": utilization,
        "labels": labels,
        "rxn_utilization": rxn_utilization,
    }

    if at_bounds:
        items = [f"{label} ({side})" for label, side in at_bounds]
        log(f"At bounds: {len(at_bounds)} coefficient(s) — {_shortlist(items)}")
    if flat:
        log(
            f"Unconstrained at the optimum: {len(flat)} direction(s) — "
            f"{_shortlist(flat)}"
        )
    if not at_bounds and not flat:
        log("Post-fit audit: no coefficients at bounds, none unconstrained")
