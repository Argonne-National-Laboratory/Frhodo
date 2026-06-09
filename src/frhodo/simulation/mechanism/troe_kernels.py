# This file is part of Frhodo. Copyright © 2020, UChicago Argonne, LLC
# and licensed under BSD-3-Clause. See License.txt in the top-level
# directory for license and copyright information.

"""Numba-JIT kernels for the Troe falloff parameterization.

Provides the forward ``ln_Troe`` and analytical Jacobian ``ln_Troe_jac``
in lna-form coordinates (slots 1 and 4 are ``ln_A``, not ``A``), along
with the Arrhenius and Fcent building blocks. These are pulled out of
``fit_coeffs`` so both ``fit_coeffs`` and ``troe_multistart`` can import
them without a cycle.
"""
import cantera as ct
import numpy as np
from numba import jit

from frhodo._vendor.opendsm.bisymlog import bisymlog_inverse
from frhodo.simulation.mechanism.coef_helpers import (
    min_pos_system_value,
    max_pos_system_value,
)



Ru = ct.gas_constant

min_ln_val = np.log(min_pos_system_value)
max_ln_val = np.log(max_pos_system_value)
min_log_val = np.log10(min_pos_system_value)
max_log_val = np.log10(max_pos_system_value)
ln_k_max = np.log(1e60)


@jit(nopython=True, nogil=True, error_model="numpy", cache=True)
def ln_arrhenius_k(T, Ea, ln_A, n):
    return ln_A + n * np.log(T) - Ea / (Ru * T)


@jit(nopython=True, nogil=True, error_model="numpy", cache=True)
def exp_safe(num, den):
    x = num / den
    res = np.zeros_like(x)
    for i, x in enumerate(num / den):
        if x >= min_ln_val and x <= max_ln_val:
            res[i] = np.exp(x)
        elif x > max_ln_val:
            res[i] = max_pos_system_value

    return res


@jit(nopython=True, nogil=True, cache=True)
def Fcent_calc(T, A, T3, T1, T2):
    exp_T3 = exp_safe(-T, T3)
    exp_T1 = exp_safe(-T, T1)
    exp_T2 = exp_safe(-T2, T)

    Fcent = (1 - A) * exp_T3 + A * exp_T1 + exp_T2

    return Fcent


@jit(nopython=True, nogil=True, error_model="numpy", cache=True)
def ln_Troe_jac(T, M, Ea_0, ln_A_0, n_0, Ea_inf, ln_A_inf, n_inf,
                A_Fc, T3, T1, T2):
    """Forward + analytical Jacobian of ``ln_Troe`` w.r.t. the 10 lna-form params.

    The Jacobian assumes the smooth (un-clamped) function; the caller
    must keep params in the unclamped region for the derivative to
    match finite differences.

    Returns:
        ``(ln_k, jac)`` with shapes ``(n_P, n_T)`` and
        ``(n_P, n_T, 10)``. The last axis of ``jac`` follows the
        argument order ``[Ea_0, ln_A_0, n_0, Ea_inf, ln_A_inf,
        n_inf, A_Fc, T3, T1, T2]``.
    """
    LN10 = 2.302585092994046

    ln_T = np.log(T)
    inv_RT = 1.0 / (Ru * T)

    ln_k_0 = ln_A_0 + n_0 * ln_T - Ea_0 * inv_RT
    ln_k_inf = ln_A_inf + n_inf * ln_T - Ea_inf * inv_RT

    p_diff = ln_k_0 - ln_k_inf
    P_r = np.exp(p_diff) * M
    one_plus_Pr = 1.0 + P_r
    alpha = 1.0 / one_plus_Pr

    log_Pr = np.log10(P_r)

    T_1d = T[0, :]
    exp_T3 = np.exp(-T_1d / T3)
    exp_T1 = np.exp(-T_1d / T1)
    exp_T2 = np.exp(-T2 / T_1d)
    Fcent_1d = (1.0 - A_Fc) * exp_T3 + A_Fc * exp_T1 + exp_T2

    g_1d = np.log(Fcent_1d)
    h_1d = g_1d / LN10
    C_1d = -0.4 - 0.67 * h_1d
    N_1d = 0.75 - 1.27 * h_1d

    n_P = T.shape[0]
    n_T = T.shape[1]
    g = np.empty((n_P, n_T))
    Fcent = np.empty((n_P, n_T))
    C = np.empty((n_P, n_T))
    N = np.empty((n_P, n_T))
    for i in range(n_P):
        g[i, :] = g_1d
        Fcent[i, :] = Fcent_1d
        C[i, :] = C_1d
        N[i, :] = N_1d

    u = log_Pr + C
    v = N - 0.14 * u
    f1 = u / v

    G = 1.0 + f1 * f1
    ln_F = g / G

    ln_k = ln_k_0 + np.log(M) - np.log(one_plus_Pr) + ln_F

    # beta: chain coefficient from a unit change in p to ln_F. Holding Fcent
    # fixed and varying only p: df1/dp = N / (v^2 * ln10), so
    # d(ln_F)/dp = -2*g*f1*N / (G^2 * v^2 * ln10).
    beta = -2.0 * g * f1 * N / (G * G * v * v * LN10)
    aplusb = alpha + beta

    jac = np.empty((n_P, n_T, 10))
    jac[:, :, 0] = -aplusb * inv_RT
    jac[:, :, 1] = aplusb
    jac[:, :, 2] = aplusb * ln_T
    jac[:, :, 3] = (aplusb - 1.0) * inv_RT
    jac[:, :, 4] = 1.0 - aplusb
    jac[:, :, 5] = (1.0 - aplusb) * ln_T

    # d ln_F / d Fcent = 1/(Fcent*G) - 2*g*f1/(G^2 * v^2) * (-0.67*v + 1.1762*u) / (Fcent*ln10).
    # 1.1762 = 1.27 - 0.14 * 0.67 is the coefficient on u in dv/dFcent.
    K = 1.0 / (Fcent * G) - 2.0 * g * f1 / (G * G * v * v) * (
        -0.67 * v + 1.1762 * u
    ) / (Fcent * LN10)

    dFc_dA_Fc_1d = -exp_T3 + exp_T1
    dFc_dT3_1d = (1.0 - A_Fc) * exp_T3 * T_1d / (T3 * T3)
    dFc_dT1_1d = A_Fc * exp_T1 * T_1d / (T1 * T1)
    dFc_dT2_1d = -exp_T2 / T_1d

    dFc_dA_Fc = np.empty((n_P, n_T))
    dFc_dT3 = np.empty((n_P, n_T))
    dFc_dT1 = np.empty((n_P, n_T))
    dFc_dT2 = np.empty((n_P, n_T))
    for i in range(n_P):
        dFc_dA_Fc[i, :] = dFc_dA_Fc_1d
        dFc_dT3[i, :] = dFc_dT3_1d
        dFc_dT1[i, :] = dFc_dT1_1d
        dFc_dT2[i, :] = dFc_dT2_1d

    jac[:, :, 6] = K * dFc_dA_Fc
    jac[:, :, 7] = K * dFc_dT3
    jac[:, :, 8] = K * dFc_dT1
    jac[:, :, 9] = K * dFc_dT2

    return ln_k, jac


@jit(nopython=True, nogil=True, error_model="numpy", cache=True)
def ln_Troe(T, M, *x):
    """``ln k(T, M)`` for the Troe falloff form in lna-form coordinates.

    Args:
        T, M: ``(n_P, n_T)`` arrays.
        x: 10-vector ``(Ea_0, ln_A_0, n_0, Ea_inf, ln_A_inf, n_inf,
            A_Fc, T3, T1, T2)`` — same ordering as :func:`ln_Troe_jac`.

    Returns:
        ``ln k`` of shape ``(n_P, n_T)``.
    """
    Ea_0, ln_A_0, n_0 = x[:3]
    Ea_inf, ln_A_inf, n_inf = x[3:6]
    Fcent_coeffs = x[-4:]

    ln_k_0 = ln_arrhenius_k(T, Ea_0, ln_A_0, n_0)
    ln_k_inf = ln_arrhenius_k(T, Ea_inf, ln_A_inf, n_inf)

    for idx in np.argwhere(ln_k_0 < min_ln_val):
        ln_k_0[idx[0], idx[1]] = min_ln_val

    for idx in np.argwhere(ln_k_0 > max_ln_val):
        ln_k_0[idx[0], idx[1]] = max_ln_val

    for idx in np.argwhere(ln_k_inf < min_ln_val):
        ln_k_inf[idx[0], idx[1]] = min_ln_val

    for idx in np.argwhere(ln_k_inf > max_ln_val):
        ln_k_inf[idx[0], idx[1]] = max_ln_val

    k_0, k_inf = np.exp(ln_k_0), np.exp(ln_k_inf)

    Fcent = Fcent_calc(T[0, :], *Fcent_coeffs)
    for idx in np.argwhere(Fcent <= 0.0):
        Fcent[idx] = min_pos_system_value

    P_r = k_0 / k_inf * M
    for idx in np.argwhere(P_r <= 0.0):
        P_r[idx] = min_pos_system_value

    log_P_r = np.log10(P_r)
    log_Fcent = np.log10(Fcent)
    C = -0.4 - 0.67 * log_Fcent
    N = 0.75 - 1.27 * log_Fcent
    f1 = (log_P_r + C) / (N - 0.14 * (log_P_r + C))
    ln_F = np.log(Fcent) / (1 + f1**2)

    log_interior = k_inf * P_r / (1 + P_r)
    for idx in np.argwhere(log_interior <= 0.0):
        log_interior[idx] = min_pos_system_value

    ln_k_calc = np.log(log_interior) + ln_F

    return ln_k_calc


# ---------------------------------------------------------------------------
# Fused-callback kernels: the AUGLAG-SBPLX polish calls objective + two
# constraints per inner iteration. The kernels below replace the Python
# wrappers' numpy glue with single-pass njit math, cutting the
# Python<->Numba boundary cost to one crossing per callback.
# ---------------------------------------------------------------------------


@jit(nopython=True, cache=True)
def set_x_from_opt_kernel(x_fit_opt, alter_idx, p0, s, x_state, bisymlog_C,
                          bisymlog_base):
    """Reconstruct the full 10-vec from optimizer-space ``x_fit_opt``.

    Args:
        x_fit_opt: Optimizer-space coordinates, one per altered slot.
        alter_idx: ``int64`` array of slot indices the optimizer touches.
        p0, s: Per-altered-slot offset and scale.
        x_state: 10-vec holding the constant values for non-altered
            slots (in base-space).
        bisymlog_C, bisymlog_base: Bisymlog inverse-transform constants
            used to convert slots 7..9 from opt-space back to base-space.

    Returns:
        10-vec in lna-form, ready for ``ln_Troe`` / ``Fcent_calc``.
    """
    result = x_state.copy()
    n_alter = len(alter_idx)
    for i in range(n_alter):
        slot = alter_idx[i]
        result[slot] = x_fit_opt[i] * s[i] + p0[i]

    for j in range(7, 10):
        y = result[j]
        if np.isfinite(y):
            if y >= 0.0:
                result[j] = bisymlog_C * (bisymlog_base ** y - 1.0)
            else:
                result[j] = -bisymlog_C * (bisymlog_base ** (-y) - 1.0)

    for j in range(7, 9):
        v = result[j]
        if v >= 0.0:
            if v < 10.0:
                result[j] = 1e-30
            elif v > 1e8:
                result[j] = 1e30
        elif np.isnan(v):
            result[j] = 1e-30

    return result


@jit(nopython=True, cache=True)
def objective_l2_kernel(T, M, ln_k, x):
    """Sum of squared residuals between ``ln_Troe(x)`` and ``ln_k``."""
    ln_k_pred = ln_Troe(
        T, M,
        x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9],
    )
    diff = ln_k_pred - ln_k

    return np.sum(diff * diff)


@jit(nopython=True, cache=True)
def Fcent_constraint_kernel(x, Tmin, Tmax, Fcent_min):
    """Worst-case Fcent violation at the data temperature extrema."""
    T_eval = np.empty(2)
    T_eval[0] = Tmin
    T_eval[1] = Tmax
    Fcent = Fcent_calc(T_eval, x[6], x[7], x[8], x[9])

    worst_lo = Fcent_min - Fcent[0]
    worst_hi = Fcent[0] - 1.0
    for k in range(1, 2):
        v_lo = Fcent_min - Fcent[k]
        if v_lo > worst_lo:
            worst_lo = v_lo
        v_hi = Fcent[k] - 1.0
        if v_hi > worst_hi:
            worst_hi = v_hi

    if worst_lo > worst_hi:
        return worst_lo * 1e8

    return worst_hi * 1e8


@jit(nopython=True, cache=True)
def Arrhenius_constraint_kernel(x, T_row, ln_k_max_val):
    """Worst-case Arrhenius rate-cap violation over ``T_row``."""
    Ea_0 = x[0]
    ln_A_0 = x[1]
    n_0 = x[2]
    Ea_inf = x[3]
    ln_A_inf = x[4]
    n_inf = x[5]
    ln_k_0 = ln_A_0 + n_0 * np.log(T_row) - Ea_0 / (Ru * T_row)
    ln_k_inf = ln_A_inf + n_inf * np.log(T_row) - Ea_inf / (Ru * T_row)
    worst = ln_k_0[0]
    for k in range(len(T_row)):
        if ln_k_0[k] > worst:
            worst = ln_k_0[k]
        if ln_k_inf[k] > worst:
            worst = ln_k_inf[k]

    return worst - ln_k_max_val
