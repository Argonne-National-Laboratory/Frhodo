"""Troe falloff fitting: NN-seeded multistart + AUGLAG-SBPLX polish.

Public entry point is the ``Troe`` class. ``multistart_nn`` and
``falloff_parameters`` are exposed for direct use.

Pipeline (``Troe.fit()``):
  1. ``multistart_nn``: NN predicts K candidate Troe parameter sets
     with per-candidate confidence; the top-confidence subset (cumulative
     softmax >= threshold, at least k_min) is refined by scipy TRF with
     analytical Jacobian ``ln_Troe_jac``. Returns the best refined
     candidate.
  2. ``falloff_parameters`` (AUGLAG-SBPLX): polish with Fcent in
     [Fcent_min, 1.0] and high-T rate-cap constraints.

Coordinates: lna form internally (ln_A in slots 1, 4 to match
``ln_Troe``); capture form (A in slots 1, 4) at API boundaries.
"""
from __future__ import annotations

import time
from copy import deepcopy

import nlopt
import numpy as np
from scipy.optimize import least_squares

from frhodo.common.units import OoM, Bisymlog
from frhodo.simulation.mechanism.coef_helpers import (
    arrhenius_coefNames,
    set_arrhenius_bnds,
    min_pos_system_value,
)
from frhodo.simulation.mechanism.troe_kernels import (
    Arrhenius_constraint_kernel,
    Fcent_calc,
    Fcent_constraint_kernel,
    Ru,
    ln_Troe,
    ln_Troe_jac,
    ln_k_max,
    max_ln_val,
    objective_l2_kernel,
    set_x_from_opt_kernel,
)
from frhodo.simulation.mechanism.troe_nn import (
    K_CANDIDATES,
    get_model,
    normalized_to_capture_np,
)
from frhodo._vendor.opendsm.adaptive_loss import adaptive_weights



__all__ = [
    "Troe",
    "falloff_parameters",
    "multistart_nn",
    "default_Troe_coefNames",
    "troe_all_bnds",
]


falloff_coefNames = ["A", "T3", "T1", "T2"]
default_Troe_coefNames = [
    f"{coefName}_{suffix}"
    for suffix in ["0", "inf"]
    for coefName in arrhenius_coefNames
] + falloff_coefNames

troe_all_bnds = {
    "A":  {"-": [-1e2, 1.0],  "+": [-1e2, 1.0]},
    "T3": {"-": [-1e8, -1e2], "+": [1.0, 1e9]},
    "T1": {"-": [-1e8, -1e2], "+": [1.0, 1e9]},
    "T2": {"-": [-1e4, 1e6],  "+": [-1e4, 1e6]},
}

bisymlog_C = 1 / (np.exp(1) - 1)
bisymlog_base = float(np.exp(1))


# ---------------------------------------------------------------------------
# Multistart: NN seeds + scipy TRF refinement
# ---------------------------------------------------------------------------

_NN_CONF_THRESHOLD = 0.80
_NN_K_MIN = 2

_TRF_XTOL = 1e-8
_TRF_FTOL = 1e-8
_TRF_GTOL = 1e-8
_TRF_MAX_NFEV = 500


# Static physical bounds in lna form (ln_Troe-native coordinates).
# ``_data_aware_bounds`` narrows ``Ea`` further when the data ``T`` range
# would let ln_k_0 / ln_k_inf hit ``ln_Troe``'s numerical clamps at the
# corner of the static box.
_BOUNDS_LOW = np.array([
    -5e8,   # Ea_0 [J/kmol]
    -50.0,  # ln A_0
    -15.0,  # n_0
    -5e8,   # Ea_inf
    -50.0,  # ln A_inf
    -15.0,  # n_inf
    0.0,    # A_Fcent
    1.0,    # T3 [K]
    1.0,    # T1 [K]
    0.0,    # T2 [K]
])
_BOUNDS_HIGH = np.array([
    5e8,
    80.0,
    10.0,
    5e8,
    80.0,
    10.0,
    1.0,
    1e6,
    1e6,
    1e5,
])

_LN_K_BUDGET_FRAC = 0.95

# Skip the AUGLAG-SBPLX polish when the multistart's rate-space log-RMS
# is below this and the fitted parameters are physically valid. The
# threshold is the worst-case rate error the caller will accept per fit.
# 0.10 ≈ 10% rate-constant error. Set to 0.0 to disable.
_POLISH_SKIP_THRESH = 0.10
# Skip the multistart-TRF refinement when the NN's predicted_rms for the
# top-conf seed is below this AND that seed's measured ln_Troe residual
# confirms it. The unpolished seed is returned; the downstream
# AUGLAG-skip gate still fires AUGLAG if its quality is poor. Set 0.0
# to disable.
_TRF_SKIP_THRESH = 0.10
_PHYS_FCENT_MIN = 1e-6


def _physically_valid(x_capture, T_grid):
    """True iff Fcent ∈ [_PHYS_FCENT_MIN, 1] at the data T extrema and
    ln_k_inf at T_max stays below ``ln_k_max``."""
    A_Fc, T3, T1, T2 = x_capture[6:10]
    T_check = np.array([float(T_grid.min()), float(T_grid.max())])
    Fcent = Fcent_calc(T_check, A_Fc, T3, T1, T2)
    if Fcent.min() < _PHYS_FCENT_MIN or Fcent.max() > 1.0:
        return False
    Ea_inf = x_capture[3]
    A_inf = x_capture[4]
    n_inf = x_capture[5]
    if A_inf <= 0:
        return False
    ln_k_inf_max = np.log(A_inf) + n_inf * np.log(T_check[1]) - Ea_inf / (Ru * T_check[1])

    return bool(ln_k_inf_max <= ln_k_max)


def _data_aware_bounds(T):
    """Static bounds with ``Ea`` tightened so the worst-case
    ``ln_k_0 / ln_k_inf`` over the data grid stays inside
    ``[-max_ln_val, max_ln_val]``.

    Budget: ``|ln_A| + |n| · ln(T_max) + |Ea| / (R · T_min) ≤
    _LN_K_BUDGET_FRAC · max_ln_val``. ``ln_A`` and ``n`` keep their
    static box; any remaining budget bounds ``Ea``. A floor of
    ``Ru * T_min`` keeps the bound physically meaningful even for very
    cold data grids.
    """
    T_min = float(np.asarray(T).min())
    T_max = float(np.asarray(T).max())
    ln_A_box = max(abs(_BOUNDS_LOW[1]), abs(_BOUNDS_HIGH[1]))
    n_box = max(abs(_BOUNDS_LOW[2]), abs(_BOUNDS_HIGH[2]))
    budget = _LN_K_BUDGET_FRAC * max_ln_val
    ea_term_budget = budget - ln_A_box - n_box * np.log(T_max)
    ea_dynamic = max(ea_term_budget, 1.0) * Ru * T_min
    ea_static = float(_BOUNDS_HIGH[0])
    ea_bound = min(ea_static, ea_dynamic)

    low = _BOUNDS_LOW.copy()
    high = _BOUNDS_HIGH.copy()
    low[0] = -ea_bound
    low[3] = -ea_bound
    high[0] = ea_bound
    high[3] = ea_bound

    return low, high


def _capture_to_lna(x_capture):
    """[Ea, A, n, Ea, A, n, A_Fc, T3, T1, T2] -> [Ea, ln_A, n, ...]."""
    p = np.asarray(x_capture, dtype=np.float64).copy()
    p[1] = np.log(max(p[1], 1e-300))
    p[4] = np.log(max(p[4], 1e-300))

    return p


def _lna_to_capture(x_lna):
    """Inverse of _capture_to_lna."""
    p = np.asarray(x_lna, dtype=np.float64).copy()
    p[1] = np.exp(p[1])
    p[4] = np.exp(p[4])

    return p


def _log_rms(x_lna, T, M, ln_k_target):
    """Rate-space log-RMS at x_lna."""
    diff = (ln_Troe(T, M, *x_lna) - ln_k_target).flatten()

    return float(np.sqrt(np.mean(diff * diff)))


def _trf_residual(x, T, M, ln_k):
    return (ln_Troe(T, M, *x) - ln_k).flatten()


def _trf_jac(x, T, M, ln_k):
    _, J = ln_Troe_jac(T, M, *x)

    return J.reshape(-1, 10)


def _trf_from_start(x0_lna, T, M, ln_k, bounds):
    """scipy TRF with analytical Jacobian; returns (x_final_lna, log-RMS, nfev)."""
    low, high = bounds
    x0 = np.clip(x0_lna, low + 1e-3, high - 1e-3)
    result = least_squares(
        _trf_residual, x0, args=(T, M, ln_k), jac=_trf_jac,
        method="trf", bounds=(low, high),
        xtol=_TRF_XTOL, ftol=_TRF_FTOL, gtol=_TRF_GTOL,
        max_nfev=_TRF_MAX_NFEV,
    )

    return result.x, _log_rms(result.x, T, M, ln_k), int(result.nfev)


def _select_n_by_confidence(conf_logits, threshold, k_min):
    """Smallest N where cumulative softmax over top-N >= threshold (>= k_min)."""
    shifted = conf_logits - conf_logits.max()
    probs = np.exp(shifted) / np.exp(shifted).sum()
    sorted_probs = np.sort(probs)[::-1]
    cum = 0.0
    n = 0
    for p in sorted_probs:
        cum += p
        n += 1
        if cum >= threshold:
            break

    return max(n, k_min)


def multistart_nn(T, M, ln_k):
    """NN-seeded multistart Troe fit on the joint 10-parameter form.

    Args:
        T: Temperature grid, shape ``(n_P, n_T)``, units K.
        M: Third-body concentration grid in Cantera units.
        ln_k: Natural log of the rate constants, shape matching ``T``.

    Returns:
        Dict with keys ``"x"`` (10-vector of capture-form params),
        ``"fval"`` (rate-space log-RMS), ``"nfev"`` (total function
        evals), ``"elapsed"`` (seconds), ``"k_refined"`` (number of
        seeds carried into the refinement stage).
    """
    t0 = time.perf_counter()
    model, stats = get_model()

    T_flat = T.flatten().astype(np.float64)
    M_flat = M.flatten().astype(np.float64)
    ln_k_flat = ln_k.flatten().astype(np.float64)
    n_pts = T_flat.size

    logT = (np.log(np.clip(T_flat, 1e-3, None)) - stats["logT_mean"]) / stats["logT_std"]
    logM = (np.log(np.clip(M_flat, 1e-30, None)) - stats["logM_mean"]) / stats["logM_std"]
    lnkn = (ln_k_flat - stats["lnk_mean"]) / stats["lnk_std"]
    feats = np.stack([logT, logM, lnkn], axis=-1).astype(np.float64)[None, ...]
    mask = np.ones((1, n_pts), dtype=bool)

    preds_norm, _log_sigmas, predicted_rms, conf_logits = model.forward(feats, mask)
    preds_norm = preds_norm[0]
    conf_logits = conf_logits[0]
    predicted_rms = predicted_rms[0]
    seeds_capture = normalized_to_capture_np(preds_norm)

    order = np.argsort(-conf_logits)
    top_k = int(order[0])

    if _TRF_SKIP_THRESH > 0 and predicted_rms[top_k] < _TRF_SKIP_THRESH:
        top_capture = seeds_capture[top_k]
        if _physically_valid(top_capture, T):
            top_lna = _capture_to_lna(top_capture)
            top_rms = _log_rms(top_lna, T, M, ln_k)
            if top_rms < _TRF_SKIP_THRESH:
                return {
                    "x": top_capture,
                    "fval": top_rms,
                    "nfev": 0,
                    "elapsed": time.perf_counter() - t0,
                    "k_refined": 0,
                }

    n_to_refine = _select_n_by_confidence(
        conf_logits, _NN_CONF_THRESHOLD, _NN_K_MIN,
    )
    bounds = _data_aware_bounds(T)

    best_x = None
    best_rms = float("inf")
    total_nfev = 0
    for k in order[:n_to_refine]:
        x0_lna = _capture_to_lna(seeds_capture[k])
        x_fit, rms, nfev = _trf_from_start(x0_lna, T, M, ln_k, bounds)
        total_nfev += nfev
        if rms < best_rms:
            best_rms = rms
            best_x = x_fit

    elapsed = time.perf_counter() - t0

    return {
        "x": _lna_to_capture(best_x),
        "fval": best_rms,
        "nfev": total_nfev,
        "elapsed": elapsed,
        "k_refined": int(n_to_refine),
    }


# ---------------------------------------------------------------------------
# Constrained polish: AUGLAG wrapping SBPLX with Fcent + rate-cap constraints
# ---------------------------------------------------------------------------

class falloff_parameters:
    """AUGLAG-wrapped constrained polish of Troe parameters.

    Wraps an outer Augmented Lagrangian around an inner SBPLX
    optimizer to enforce Fcent in ``[Fcent_min, 1]`` and an Arrhenius
    rate cap over the data temperature range.

    Attributes:
        T, M, ln_k: Data grid arrays (see :func:`multistart_nn`).
        x0: Starting capture-form parameters.
        algo: Algorithm options dict — supplies ``algorithm`` for the
            sub-optimizer, ``max_eval``, ``xtol_rel``, ``ftol_rel``,
            ``initial_step``, ``loss_fcn_param``, and ``is_P_limit``
            flags marking whether either Arrhenius limb is frozen.
        Fcent_idx: Indices of the Fcent parameters allowed to vary.
            Equals ``[6, 7, 8, 9]`` for a well-formed seed; collapses
            to ``[9]`` when the seed's ``T3``/``T1`` are both below
            10 K (treated as a Lindemann-limit degeneracy).
        alter_idx: Composite list of Arrhenius + Fcent indices the
            optimizer is allowed to touch.
    """

    Fcent_min = 1e-6
    Tmin = 273
    Tmax = 6000

    def __init__(self, T, M, ln_k, x0, algo_options):
        self.T = T
        self.M = M
        self.ln_k = ln_k
        self.x0 = np.array(x0, dtype=np.float64).copy()
        self.s = np.ones_like(self.x0)

        self.bisymlog = Bisymlog(C=bisymlog_C, scaling_factor=2.0, base=np.exp(1))

        self.algo = algo_options
        self.loss_alpha = self.algo["loss_fcn_param"][0]
        self.loss_scale = self.algo["loss_fcn_param"][1]

        if (self.x0[-3:-1] < 10).all() or np.isnan(self.x0).any():
            self.x0[-4:-1] = [1.0, 1.0e-30, 1.0e-30]
            self.Fcent_idx = [9]
        else:
            self.Fcent_idx = [6, 7, 8, 9]

        if all(self.algo["is_P_limit"]):
            self.alter_idx = self.Fcent_idx
        elif self.algo["is_P_limit"][0]:
            self.alter_idx = [3, 4, 5, *self.Fcent_idx]
        elif self.algo["is_P_limit"][1]:
            self.alter_idx = [0, 1, 2, *self.Fcent_idx]
        else:
            self.alter_idx = [0, 1, 2, 3, 4, 5, *self.Fcent_idx]

        self._alter_idx_arr = np.asarray(self.alter_idx, dtype=np.int64)

        self.x = deepcopy(self.x0)
        # set_x_from_opt cache: AUGLAG-SBPLX calls set_x_from_opt three
        # times per inner query (objective + two constraints) with the
        # same x_fit. Memoize on the bytes of the input so the two
        # follow-on calls are O(1).
        self._sxfo_cache_key: bytes | None = None
        self._sxfo_cache_val: np.ndarray | None = None

    def x_bnds(self, x0):
        bnds = []
        for n, coef in enumerate(["A", "T3", "T1", "T2"]):
            if x0[n] < 0:
                bnds.append(troe_all_bnds[coef]["-"])
            elif x0[n] > 0:
                bnds.append(troe_all_bnds[coef]["+"])
            else:
                bnds.append([np.nan, np.nan])

        return np.array(bnds).T

    def fit(self):
        """Run the constrained AUGLAG polish.

        Returns:
            ``{"x": (10,) capture-form params, "fval": loss at
            optimum, "nfev": function evaluations consumed}``.
        """
        self.p0 = self.x0
        self.p0[-3:] = self.convert_Fcent(self.p0[-3:], "base2opt")

        p_bnds = set_arrhenius_bnds(self.p0[0:3], arrhenius_coefNames)
        p_bnds = np.concatenate(
            (p_bnds, set_arrhenius_bnds(self.p0[3:6], arrhenius_coefNames)),
            axis=1,
        )
        p_bnds[:, 1] = np.log(p_bnds[:, 1])
        p_bnds[:, 4] = np.log(p_bnds[:, 4])

        Fcent_bnds = self.x_bnds(self.p0[-4:])
        Fcent_bnds[-3:] = self.convert_Fcent(Fcent_bnds[-3:])

        self.p_bnds = np.concatenate((p_bnds, Fcent_bnds), axis=1)

        if len(self.p_bnds) > 0:
            self.p0 = np.clip(self.p0, self.p_bnds[0, :], self.p_bnds[1, :])

        self.p0 = self.p0[self.alter_idx]
        self.p_bnds = self.p_bnds[:, self.alter_idx]
        self.s = np.ones_like(self.p0)

        p0_opt = np.zeros_like(self.p0)
        self.s = self.calc_s(p0_opt)
        bnds = (self.p_bnds - self.p0) / self.s

        opt = nlopt.opt(nlopt.AUGLAG, len(self.p0))
        opt.set_min_objective(self.objective)
        opt.add_inequality_constraint(self.Fcent_constraint, 0.0)
        opt.add_inequality_constraint(self.Arrhenius_constraint, 1e-8)
        opt.set_maxeval(self.algo["max_eval"])
        opt.set_xtol_rel(self.algo["xtol_rel"])
        opt.set_ftol_rel(self.algo["ftol_rel"])
        opt.set_lower_bounds(bnds[0])
        opt.set_upper_bounds(bnds[1])
        opt.set_initial_step(self.algo["initial_step"])

        sub_opt = nlopt.opt(self.algo["algorithm"], len(self.p0))
        sub_opt.set_initial_step(self.algo["initial_step"])
        sub_opt.set_xtol_rel(self.algo["xtol_rel"])
        sub_opt.set_ftol_rel(self.algo["ftol_rel"])
        opt.set_local_optimizer(sub_opt)

        x_fit = opt.optimize(p0_opt)
        x_fit = self.set_x_from_opt(x_fit)

        x_fit[1] = np.exp(x_fit[1])
        x_fit[4] = np.exp(x_fit[4])

        return {"x": x_fit, "fval": opt.last_optimum_value(), "nfev": opt.get_numevals()}

    def set_x_from_opt(self, x):
        x_arr = np.asarray(x, dtype=np.float64)
        key = x_arr.tobytes()
        if key == self._sxfo_cache_key:
            return self._sxfo_cache_val.copy()

        result = set_x_from_opt_kernel(
            x_arr, self._alter_idx_arr, self.p0, self.s, self.x,
            bisymlog_C, bisymlog_base,
        )

        self._sxfo_cache_key = key
        self._sxfo_cache_val = result

        return result.copy()

    def convert_Fcent(self, x, conv_type="base2opt"):
        y = np.array(x)

        flatten = False
        if y.ndim == 1:
            y = y[np.newaxis, :]
            flatten = True

        if conv_type == "base2opt":
            y = self.bisymlog.transform(y)
        else:
            y = self.bisymlog.invTransform(y)

        if flatten:
            y = y.flatten()
        else:
            y = np.sort(y, axis=0)

        return y

    def objective(self, x_fit, grad=np.array([])):
        x = self.set_x_from_opt(x_fit)
        if self.loss_alpha == 2.0:
            return float(objective_l2_kernel(self.T, self.M, self.ln_k, x))

        resid = (ln_Troe(self.T, self.M, *x) - self.ln_k).flatten()
        loss_weights, _, _ = adaptive_weights(
            resid, C_scalar=self.loss_scale, alpha=self.loss_alpha,
        )

        return float(np.sum(loss_weights * (resid**2)))

    def calc_s(self, x_fit):
        """Per-slot ``1/|grad|`` scaling from the analytical Jacobian.

        Gradient of ``sum_i w_i · r_i^2`` w.r.t. the optimizer's scaled
        coordinates at ``x_fit``. The chain through
        :meth:`set_x_from_opt` is identity for Arrhenius and ``A_Fc``
        slots and ``invTransform_derivative`` for the bisymlog-coded
        Fcent T-slots (7, 8, 9). Adaptive weights are held constant —
        scale estimation only needs sensitivity, not the exact
        weight-chain derivative. With ``loss_alpha == 2`` (the default)
        weights collapse to 1 and we skip the call entirely.
        """
        x_full = self.set_x_from_opt(x_fit)
        ln_k_pred, J_full = ln_Troe_jac(self.T, self.M, *x_full)
        resid = (ln_k_pred - self.ln_k).flatten()
        if self.loss_alpha == 2.0:
            weighted_resid = resid
        else:
            loss_weights, _, _ = adaptive_weights(
                resid, C_scalar=self.loss_scale, alpha=self.loss_alpha,
            )
            weighted_resid = loss_weights * resid
        grad_full = 2.0 * weighted_resid @ J_full.reshape(-1, 10)

        grad = np.empty(len(self.alter_idx))
        for k, slot in enumerate(self.alter_idx):
            chain = self.s[k]
            if slot in (7, 8, 9):
                opt_val = x_fit[k] * self.s[k] + self.p0[k]
                chain *= float(self.bisymlog.invTransform_derivative(opt_val))
            grad[k] = grad_full[slot] * chain

        y = np.abs(grad)
        y = np.where(np.isnan(y), 0.0, y)
        if (y < min_pos_system_value).all():
            y = np.ones_like(y) * 1e-14
        else:
            y[y < min_pos_system_value] = 10 ** (
                OoM(np.min(y[y >= min_pos_system_value])) - 1
            )
        scale = 1.0 / y

        return scale

    def Fcent_constraint(self, x_fit, grad=np.array([])):
        x = self.set_x_from_opt(x_fit)

        return float(Fcent_constraint_kernel(x, self.Tmin, self.Tmax, self.Fcent_min))

    def Arrhenius_constraint(self, x_fit, grad=np.array([])):
        x = self.set_x_from_opt(x_fit)

        return float(Arrhenius_constraint_kernel(x, self.T[-1], ln_k_max))


# ---------------------------------------------------------------------------
# Public API: Troe class
# ---------------------------------------------------------------------------

class Troe:
    """Driver for fitting Troe-form pressure-dependent rates.

    Sequences the NN-seeded multistart (:func:`multistart_nn`) with
    the AUGLAG polish (:class:`falloff_parameters`) and exposes a
    minimal coefficient-subset interface for the orchestrator.

    Args:
        rates: Rate constants ``k(T, P)`` at the sample points.
        T, P: Sample temperatures [K] and pressures [Pa], parallel to
            ``rates``. Must form a meshgrid: identical ``T`` vector
            for every pressure row.
        M: ``M(T, P) -> third-body concentration``; supplied by the
            caller so it can use the live mechanism.
        x0: Optional starting capture-form parameters. Padded with
            sensible defaults when shorter than 10.
        coefNames: Subset of the 10-parameter capture form to fit.
            Defaults to the full set.
        bnds: Optional per-parameter ``[lower, upper]`` bounds matching
            ``coefNames``; ``x0`` is clipped into them.
        is_falloff_limit: Boolean mask marking parameters that are
            already known limits (skipped by the optimizer).
    """

    def __init__(
        self,
        rates,
        T,
        P,
        M,
        x0=[],
        coefNames=default_Troe_coefNames,
        bnds=[],
        is_falloff_limit=None,
    ):
        self.k = rates
        self.ln_k = np.log(rates)
        self.T = T
        self.P = P
        self.M = M

        if is_falloff_limit is None:
            is_falloff_limit = np.zeros(len(default_Troe_coefNames), dtype=bool)

        self.x0 = list(x0)
        if len(self.x0) != 10:
            self.x0[6:] = [1.0, 1e-30, 1e-30, 1500]
        self.x0 = np.array(self.x0)
        self.x = deepcopy(self.x0)
        self.coefNames = np.array(coefNames)

        idx = []
        for n, coefName in enumerate(default_Troe_coefNames):
            if coefName in coefNames:
                idx.append(n)

        self.p0 = self.x0[idx]

        self.bnds = np.array(bnds)
        if len(self.bnds) > 0:
            self.p0 = np.clip(self.p0, self.bnds[0, :], self.bnds[1, :])

        self.alter_idx = {
            "low_rate": [],
            "high_rate": [],
            "pre_exponential_factor": [],
            "all": [],
        }
        for n, coefName in enumerate(default_Troe_coefNames):
            if coefName in coefNames:
                self.alter_idx["all"].append(n)
                if coefName.rsplit("_", 1)[0] in arrhenius_coefNames:
                    if "_0" == coefName[-2:]:
                        self.alter_idx["low_rate"].append(n)
                    elif "_inf" == coefName[-4:]:
                        self.alter_idx["high_rate"].append(n)

                    if "pre_exponential_factor" in coefName:
                        self.alter_idx["pre_exponential_factor"].append(n)

        idx = [-1]
        is_P_limit = [False, False]
        for i, arrhenius_type in enumerate(["low_rate", "high_rate"]):
            x_idx = np.array(self.alter_idx[arrhenius_type])
            idx = np.arange(idx[-1], idx[-1] + len(x_idx)) + 1

            if len(idx) < 3 or any(is_falloff_limit[idx]):
                is_P_limit[i] = True

        self.algorithm_options = {
            "algorithm": nlopt.LN_SBPLX,
            "xtol_rel": 1e-6,
            "ftol_rel": 1e-6,
            "initial_step": 1e-3,
            "max_eval": 10000,
            "loss_fcn_param": [2, 1],
            "is_P_limit": is_P_limit,
        }

    def fit(self):
        """NN multistart + AUGLAG polish; return the fitted coefficients.

        Returns an ``np.ndarray`` matching ``coefNames`` (linear-space
        ``A``, not ``ln A``).
        """
        idx_A = self.alter_idx["pre_exponential_factor"]
        idx_all = self.alter_idx["all"]

        T_2d, M_2d, ln_k_2d = self._reshape_to_meshgrid()

        ms = multistart_nn(T_2d, M_2d, ln_k_2d)
        x = ms["x"]

        if (
            _POLISH_SKIP_THRESH > 0
            and ms["fval"] < _POLISH_SKIP_THRESH
            and _physically_valid(x, T_2d)
        ):
            coeffs = x[idx_all]

            return coeffs

        x0 = deepcopy(x)
        x0[idx_A] = np.log(x0[idx_A])

        falloff = falloff_parameters(T_2d, M_2d, ln_k_2d, x0, self.algorithm_options)
        res = falloff.fit()
        x = res["x"]

        coeffs = x[idx_all]

        return coeffs

    def _reshape_to_meshgrid(self):
        """Build (n_P, n_T) T/M/ln_k arrays from the 1D inputs. n_T must be
        shared across all n_P pressure rows."""
        T_vals = np.asarray(self.T, dtype=np.float64).flatten()
        P_vals = np.asarray(self.P, dtype=np.float64).flatten()
        ln_k_vals = np.asarray(self.ln_k, dtype=np.float64).flatten()

        P_unique = np.unique(P_vals)
        T_unique = np.unique(T_vals)
        n_P = len(P_unique)
        n_T = len(T_unique)
        if n_P * n_T != T_vals.size:
            raise ValueError(
                f"Troe.fit() expects an (n_P, n_T) meshgrid with shared T per "
                f"P row; got {T_vals.size} points but unique(P)={n_P}, "
                f"unique(T)={n_T} (product {n_P*n_T})"
            )

        T_2d = np.tile(T_unique, (n_P, 1))
        P_2d = np.tile(P_unique[:, None], (1, n_T))
        M_2d = np.empty((n_P, n_T), dtype=np.float64)
        for i in range(n_P):
            M_2d[i, :] = self.M(T_2d[i, :], P_2d[i, :])

        ln_k_2d = np.empty((n_P, n_T), dtype=np.float64)
        for i, p in enumerate(P_unique):
            row_mask = P_vals == p
            row_T = T_vals[row_mask]
            row_lnk = ln_k_vals[row_mask]
            order = np.argsort(row_T)
            row_T_sorted = row_T[order]
            row_lnk_sorted = row_lnk[order]
            if not np.allclose(row_T_sorted, T_unique):
                raise ValueError(
                    f"P row {p} has T values {row_T_sorted} that don't match "
                    f"the global T grid {T_unique}; meshgrid requires shared T."
                )
            ln_k_2d[i, :] = row_lnk_sorted

        return T_2d, M_2d, ln_k_2d
