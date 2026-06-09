"""Incident-shock reactor — scipy and SUNDIALS backends + dispatcher.

Goldsmith / Speth shock-tube governing equations. State vector
``[z, A, rho, v, T, t_shock, Y_1..Y_K]``; integration is in lab time;
derivatives are pre-scaled by ``rho*A/(rho1*A1)``.

Pieces in this file:

1. :func:`_shock_derivatives` — the equations themselves, written
   once. Both backends call it.
2. :class:`IncidentShockReactor` — single reactor class with a
   ``backend`` parameter selecting between SUNDIALS CVODES (via the
   in-tree binding) and ``scipy.solve_ivp`` (BDF/Radau/LSODA). Same
   state vector, validation, and side-channel ``failure_reason``;
   SUNDIALS additionally uses the analytical :func:`_shock_jacobian`.
3. :func:`run_incident_shock` — public entry point that dispatches
   between backends based on the ``ODE_solver`` option and collects
   the trajectory into a :class:`ReactorOutput`.

Goldsmith / Speth ODE derivation:

    Copyright (c) 2016 Raymond L. Speth — MIT License (see LICENSE.txt).
"""
from dataclasses import dataclass

import cantera as ct
import numba
import numpy as np
from scipy.integrate import solve_ivp

from frhodo.common.errors import FailureReason
from frhodo.simulation.mechanism.mech_fcns import check_rxn_rates, list2ct_mixture
from frhodo.simulation.numerics.sundials import CVodeIntegrator
from frhodo.simulation.shock.reactor_output import ReactorOutput


Ru = ct.gas_constant
_RU = float(ct.gas_constant)


# Default integrator tolerances per backend. scipy methods need a looser
# pair; CVODES (SUNDIALS) wants tighter to match output smoothness — its
# accepted-step error budget is less conservative for this stiff system.
SCIPY_DEFAULT_TOLS  = {"rtol": 1e-4, "atol": 1e-7}
CVODES_DEFAULT_TOLS = {"rtol": 1e-6, "atol": 1e-10}


# State vector layout: [z, A, ρ, v, T, t_shock, Y_0, …, Y_{K-1}]
I_Z = 0
I_A = 1
I_RHO = 2
I_V = 3
I_T = 4
I_TS = 5
N_SHOCK = 6


@dataclass(frozen=True)
class _Geometry:
    """Shock-tube geometry / pre-shock reference frame."""
    rho1: float
    A1: float
    L: float
    As: float
    area_change: bool


@numba.njit(cache=True, fastmath=False)
def _shock_derivatives_kernel(
    z: float, A: float, v: float, T: float, rho: float,
    cp: float, Wmix: float,
    hk: np.ndarray, wdot: np.ndarray, Wk: np.ndarray,
    rho1: float, A1: float, L: float, As: float, area_change: bool,
) -> np.ndarray:
    """Pure-math kernel for :func:`_shock_derivatives`; JIT-compiled."""
    if area_change:
        n = 0.5  # Mirels exponent
        xi = z / L
        if xi < 1e-10:
            xi = 1e-10
        dA_dt = v * As * n / L * xi ** (n - 1.0) / (1.0 - xi ** n) ** 2.0
    else:
        dA_dt = 0.0

    beta = v * v * (1.0 / (cp * T) - Wmix / (_RU * T))
    species_sum = 0.0
    h_dot_wdot = 0.0
    K = Wk.size
    for i in range(K):
        species_sum += (hk[i] / (cp * T) - Wmix) * wdot[i]
        h_dot_wdot += hk[i] * wdot[i]
    drho_dt = (species_sum - rho * beta / A * dA_dt) / (1.0 + beta)
    dv_dt = -v * (drho_dt / rho + dA_dt / A)
    dT_dt = -(h_dot_wdot / rho + v * dv_dt) / cp

    scale = rho * A / (rho1 * A1)

    ydot = np.empty(6 + K)
    ydot[0] = v * scale
    ydot[1] = dA_dt * scale
    ydot[2] = drho_dt * scale
    ydot[3] = dv_dt * scale
    ydot[4] = dT_dt * scale
    ydot[5] = scale
    inv_rho_scale = scale / rho
    for i in range(K):
        ydot[6 + i] = wdot[i] * Wk[i] * inv_rho_scale

    return ydot


def _shock_derivatives(
    gas: ct.Solution,
    z: float,
    A: float,
    v: float,
    Wk: np.ndarray,
    *,
    rho1: float,
    A1: float,
    L: float,
    As: float,
    area_change: bool,
) -> np.ndarray:
    """Goldsmith / Speth incident-shock derivatives at the current gas state.

    Reads thermo state from ``gas`` (assumes the caller has already set
    composition + (T, ρ)) and dispatches to the JIT-compiled
    :func:`_shock_derivatives_kernel`.

    Args:
        Wk: Per-species molecular weight array, cached by the caller to
            avoid per-step allocation.

    Returns:
        Pre-scaled derivative vector
        ``[dz, dA, drho, dv, dT, dt_shock, dY_1..dY_K] * rho*A/(rho1*A1)``
        of length ``6 + n_species``.
    """
    return _shock_derivatives_kernel(
        z, A, v, gas.T, gas.density,
        gas.cp_mass, gas.mean_molecular_weight,
        np.ascontiguousarray(gas.partial_molar_enthalpies, dtype=np.float64),
        np.ascontiguousarray(gas.net_production_rates, dtype=np.float64),
        Wk,
        rho1, A1, L, As, area_change,
    )


@numba.njit(cache=True, fastmath=False)
def _shock_jacobian_kernel(
    z: float, A: float, rho: float, v: float, T: float,
    Y: np.ndarray, cp: float, W: float, dcp_dT: float,
    hk: np.ndarray, cp_k_mole: np.ndarray, wdot: np.ndarray,
    DC: np.ndarray, DT: np.ndarray, DP: np.ndarray,
    Wk: np.ndarray,
    rho1: float, A1: float, L: float, As: float, area_change: bool,
) -> np.ndarray:
    """Pure-math kernel for :func:`_shock_jacobian`; JIT-compiled."""
    K = Wk.size
    n = N_SHOCK + K

    cp_k_mass = cp_k_mole / Wk

    dcp_dY = cp_k_mass
    dW_dY = -W * W / Wk

    Y_over_W = Y / Wk
    rho_over_W = rho / Wk
    DP_RuT = DP * (_RU * T)
    dwdot_dT = DT + DP * (rho * _RU / W)
    dwdot_drho = DC @ Y_over_W + DP * (_RU * T / W)
    dwdot_dY = (DC + DP_RuT.reshape(K, 1)) * rho_over_W.reshape(1, K)

    scale = rho * A / (rho1 * A1)
    dscale_drho = A / (rho1 * A1)
    dscale_dA = rho / (rho1 * A1)

    beta = v * v * (1.0 / (cp * T) - W / (_RU * T))
    one_plus_beta = 1.0 + beta

    phi_h = 0.0
    phi_n = 0.0
    for i in range(K):
        phi_h += hk[i] * wdot[i]
        phi_n += wdot[i]
    phi = phi_h / (cp * T) - W * phi_n

    if area_change:
        n_mirels = 0.5
        xi = z / L
        if xi < 1e-10:
            xi = 1e-10
        one_minus_xi_n = 1.0 - xi ** n_mirels
        dA_dt = (
            v * As * n_mirels / L
            * xi ** (n_mirels - 1.0) / one_minus_xi_n ** 2
        )
        ddA_dt_dv = dA_dt / v
        ddA_dt_dz = (
            (v * As * n_mirels / (L * L))
            * xi ** (n_mirels - 2.0)
            * ((n_mirels - 1.0) * one_minus_xi_n + 2.0 * n_mirels * xi ** n_mirels)
            / one_minus_xi_n ** 3
        )
    else:
        dA_dt = 0.0
        ddA_dt_dv = 0.0
        ddA_dt_dz = 0.0

    g_z = v
    g_A = dA_dt
    g_rho_chem = phi / one_plus_beta
    if area_change:
        g_rho_geom = -rho * beta * dA_dt / (A * one_plus_beta)
    else:
        g_rho_geom = 0.0
    g_rho = g_rho_chem + g_rho_geom
    g_v = -v * (g_rho / rho + dA_dt / A)
    psi = 0.0
    for i in range(K):
        psi += wdot[i] * hk[i]
    psi /= rho
    g_T = -(psi + v * g_v) / cp
    g_ts = 1.0

    dbeta_dv = 2.0 * beta / v
    dbeta_dT = -v * v * (
        dcp_dT / (cp * cp * T) + 1.0 / (cp * T * T) - W / (_RU * T * T)
    )
    dbeta_dY = v * v * (-dcp_dY / (cp * cp * T) + W * W / (Wk * _RU * T))

    dphi_h_dT = 0.0
    for i in range(K):
        dphi_h_dT += cp_k_mole[i] * wdot[i] + hk[i] * dwdot_dT[i]
    dphi_h_drho = 0.0
    for i in range(K):
        dphi_h_drho += hk[i] * dwdot_drho[i]
    dphi_h_dY = hk @ dwdot_dY
    dphi_n_dT = 0.0
    for i in range(K):
        dphi_n_dT += dwdot_dT[i]
    dphi_n_drho = 0.0
    for i in range(K):
        dphi_n_drho += dwdot_drho[i]
    dphi_n_dY = np.zeros(K)
    for j in range(K):
        s = 0.0
        for i in range(K):
            s += dwdot_dY[i, j]
        dphi_n_dY[j] = s

    dphi_dT = (
        dphi_h_dT / (cp * T)
        - phi_h * (dcp_dT / cp + 1.0 / T) / (cp * T)
        - W * dphi_n_dT
    )
    dphi_drho = dphi_h_drho / (cp * T) - W * dphi_n_drho
    dphi_dY = (
        dphi_h_dY / (cp * T)
        - phi_h * dcp_dY / (cp * cp * T)
        - dW_dY * phi_n
        - W * dphi_n_dY
    )

    dgrho_chem_dT = dphi_dT / one_plus_beta - g_rho_chem * dbeta_dT / one_plus_beta
    dgrho_chem_drho = dphi_drho / one_plus_beta
    dgrho_chem_dv = -g_rho_chem * dbeta_dv / one_plus_beta
    dgrho_chem_dY = (
        dphi_dY / one_plus_beta - g_rho_chem * dbeta_dY / one_plus_beta
    )

    if area_change:
        mu = beta / one_plus_beta
        dmu_dv = dbeta_dv / (one_plus_beta * one_plus_beta)
        dmu_dT = dbeta_dT / (one_plus_beta * one_plus_beta)
        dmu_dY = dbeta_dY / (one_plus_beta * one_plus_beta)
        dgrho_geom_dz = -rho * mu / A * ddA_dt_dz
        dgrho_geom_dA = rho * mu * dA_dt / (A * A)
        dgrho_geom_drho = -mu * dA_dt / A
        dgrho_geom_dv = -rho * dA_dt / A * dmu_dv - rho * mu / A * (dA_dt / v)
        dgrho_geom_dT = -rho * dA_dt / A * dmu_dT
        dgrho_geom_dY = -rho * dA_dt / A * dmu_dY
    else:
        dgrho_geom_dz = 0.0
        dgrho_geom_dA = 0.0
        dgrho_geom_drho = 0.0
        dgrho_geom_dv = 0.0
        dgrho_geom_dT = 0.0
        dgrho_geom_dY = np.zeros(K)

    dgrho_dz = dgrho_geom_dz
    dgrho_dA = dgrho_geom_dA
    dgrho_drho = dgrho_chem_drho + dgrho_geom_drho
    dgrho_dv = dgrho_chem_dv + dgrho_geom_dv
    dgrho_dT = dgrho_chem_dT + dgrho_geom_dT
    dgrho_dY = dgrho_chem_dY + dgrho_geom_dY

    Q = g_rho / rho + dA_dt / A
    if area_change:
        dQ_dz = dgrho_dz / rho + ddA_dt_dz / A
        dQ_dv = dgrho_dv / rho + ddA_dt_dv / A
    else:
        dQ_dz = dgrho_dz / rho
        dQ_dv = dgrho_dv / rho
    dQ_dA = dgrho_dA / rho - dA_dt / (A * A)
    dQ_drho = dgrho_drho / rho - g_rho / (rho * rho)
    dQ_dT = dgrho_dT / rho
    dQ_dY = dgrho_dY / rho

    dgv_dz = -v * dQ_dz
    dgv_dA = -v * dQ_dA
    dgv_drho = -v * dQ_drho
    dgv_dv = -Q - v * dQ_dv
    dgv_dT = -v * dQ_dT
    dgv_dY = -v * dQ_dY

    h_dot_dwdot_drho = 0.0
    for i in range(K):
        h_dot_dwdot_drho += hk[i] * dwdot_drho[i]
    dpsi_dT = dphi_h_dT / rho
    dpsi_drho = h_dot_dwdot_drho / rho - phi_h / (rho * rho)
    dpsi_dY = (hk @ dwdot_dY) / rho

    dgT_dz = -v * dgv_dz / cp
    dgT_dA = -v * dgv_dA / cp
    dgT_drho = -(dpsi_drho + v * dgv_drho) / cp
    dgT_dv = -(g_v + v * dgv_dv) / cp
    dgT_dT = -dpsi_dT / cp - v * dgv_dT / cp - g_T * dcp_dT / cp
    dgT_dY = -(dpsi_dY + v * dgv_dY) / cp - g_T * dcp_dY / cp

    dgdy = np.zeros((n, n))

    dgdy[I_Z, I_V] = 1.0
    if area_change:
        dgdy[I_A, I_Z] = ddA_dt_dz
        dgdy[I_A, I_V] = ddA_dt_dv

    dgdy[I_RHO, I_Z] = dgrho_dz
    dgdy[I_RHO, I_A] = dgrho_dA
    dgdy[I_RHO, I_RHO] = dgrho_drho
    dgdy[I_RHO, I_V] = dgrho_dv
    dgdy[I_RHO, I_T] = dgrho_dT
    for j in range(K):
        dgdy[I_RHO, N_SHOCK + j] = dgrho_dY[j]

    dgdy[I_V, I_Z] = dgv_dz
    dgdy[I_V, I_A] = dgv_dA
    dgdy[I_V, I_RHO] = dgv_drho
    dgdy[I_V, I_V] = dgv_dv
    dgdy[I_V, I_T] = dgv_dT
    for j in range(K):
        dgdy[I_V, N_SHOCK + j] = dgv_dY[j]

    dgdy[I_T, I_Z] = dgT_dz
    dgdy[I_T, I_A] = dgT_dA
    dgdy[I_T, I_RHO] = dgT_drho
    dgdy[I_T, I_V] = dgT_dv
    dgdy[I_T, I_T] = dgT_dT
    for j in range(K):
        dgdy[I_T, N_SHOCK + j] = dgT_dY[j]

    inv_rho = 1.0 / rho
    for i in range(K):
        Wk_over_rho_i = Wk[i] * inv_rho
        dgdy[N_SHOCK + i, I_T] = dwdot_dT[i] * Wk_over_rho_i
        dgdy[N_SHOCK + i, I_RHO] = (
            dwdot_drho[i] * Wk_over_rho_i
            - wdot[i] * Wk_over_rho_i * inv_rho
        )
        for j in range(K):
            dgdy[N_SHOCK + i, N_SHOCK + j] = dwdot_dY[i, j] * Wk_over_rho_i

    g_vec = np.empty(n)
    g_vec[I_Z] = g_z
    g_vec[I_A] = g_A
    g_vec[I_RHO] = g_rho
    g_vec[I_V] = g_v
    g_vec[I_T] = g_T
    g_vec[I_TS] = g_ts
    for j in range(K):
        g_vec[N_SHOCK + j] = wdot[j] * Wk[j] * inv_rho

    J = scale * dgdy
    for i in range(n):
        J[i, I_A] += dscale_dA * g_vec[i]
        J[i, I_RHO] += dscale_drho * g_vec[i]

    return J


def _shock_jacobian(
    gas: ct.Solution, y: np.ndarray, geometry: _Geometry, Wk: np.ndarray,
) -> np.ndarray:
    """Analytical ``∂f/∂y`` for ``_shock_derivatives`` at state ``y``.

    Mutates the gas thermo state to ``(T, ρ, Y)`` from ``y`` and leaves
    it there on return.

    Returns:
        Dense Jacobian of shape ``(n_state, n_state)`` where
        ``n_state = 6 + n_species``.
    """
    z = float(y[I_Z])
    A = float(y[I_A])
    rho = float(y[I_RHO])
    v = float(y[I_V])
    T = float(y[I_T])
    Y = np.ascontiguousarray(y[N_SHOCK:], dtype=np.float64)

    gas.set_unnormalized_mass_fractions(Y)
    gas.TD = T, rho

    cp = float(gas.cp_mass)
    W = float(gas.mean_molecular_weight)
    hk = np.ascontiguousarray(gas.partial_molar_enthalpies, dtype=np.float64)
    cp_k_mole = np.ascontiguousarray(gas.partial_molar_cp, dtype=np.float64)
    wdot = np.ascontiguousarray(gas.net_production_rates, dtype=np.float64)
    DC = np.ascontiguousarray(gas.net_production_rates_ddCi, dtype=np.float64)
    DT = np.ascontiguousarray(gas.net_production_rates_ddT, dtype=np.float64)
    DP = np.ascontiguousarray(gas.net_production_rates_ddP, dtype=np.float64)

    # FD step for dcp_dT — Cantera does not expose ∂cp/∂T directly,
    # so perturb T, read cp, restore. Kept in Python because it mutates
    # the gas state.
    eps_T = T * 1e-7
    gas.TD = T + eps_T, rho
    cp_plus = float(gas.cp_mass)
    gas.set_unnormalized_mass_fractions(Y)
    gas.TD = T, rho
    dcp_dT = (cp_plus - cp) / eps_T

    return _shock_jacobian_kernel(
        z, A, rho, v, T, Y, cp, W, dcp_dT,
        hk, cp_k_mole, wdot, DC, DT, DP, Wk,
        geometry.rho1, geometry.A1, geometry.L, geometry.As,
        geometry.area_change,
    )


@numba.njit(cache=True, fastmath=False)
def _shock_param_rhs_gradient_kernel(
    A: float, rho: float, v: float, T: float,
    cp: float, W: float,
    hk: np.ndarray, nu: np.ndarray, qj: np.ndarray, Wk: np.ndarray,
    rho1: float, A1: float,
) -> np.ndarray:
    """Pure-math kernel for :func:`_shock_param_rhs_gradient`; JIT-compiled."""
    K = Wk.size
    n_rxns = qj.size
    n_state = N_SHOCK + K

    scale = rho * A / (rho1 * A1)
    beta = v * v * (1.0 / (cp * T) - W / (_RU * T))
    inv_one_plus_beta = 1.0 / (1.0 + beta)
    inv_rho = 1.0 / rho
    inv_cp = 1.0 / cp

    species_factor = np.empty(K)
    for i in range(K):
        species_factor[i] = hk[i] / (cp * T) - W

    species_factor_nu = species_factor @ nu
    h_dot_nu = hk @ nu

    dgdp = np.zeros((n_state, n_rxns))
    for j in range(n_rxns):
        dgrho = species_factor_nu[j] * qj[j] * inv_one_plus_beta
        dgv = -v * dgrho * inv_rho
        dgT = -(h_dot_nu[j] * qj[j] * inv_rho + v * dgv) * inv_cp
        dgdp[I_RHO, j] = dgrho
        dgdp[I_V, j] = dgv
        dgdp[I_T, j] = dgT

    for i in range(K):
        for j in range(n_rxns):
            dgdp[N_SHOCK + i, j] = nu[i, j] * qj[j] * Wk[i] * inv_rho

    return scale * dgdp


def _shock_param_rhs_gradient(
    gas: ct.Solution, y: np.ndarray, geometry: _Geometry, Wk: np.ndarray,
) -> np.ndarray:
    """Analytical ``∂f/∂p_j`` at state ``y`` for all reactions.

    ``p_j`` is the multiplier on reaction ``j``'s rate constant.

    Returns:
        Parameter-gradient matrix of shape ``(n_state, n_rxns)``.
    """
    A = float(y[I_A])
    rho = float(y[I_RHO])
    v = float(y[I_V])
    T = float(y[I_T])
    Y = np.ascontiguousarray(y[N_SHOCK:], dtype=np.float64)

    gas.set_unnormalized_mass_fractions(Y)
    gas.TD = T, rho

    cp = float(gas.cp_mass)
    W = float(gas.mean_molecular_weight)
    hk = np.ascontiguousarray(gas.partial_molar_enthalpies, dtype=np.float64)
    nu = np.ascontiguousarray(
        np.asarray(gas.product_stoich_coeffs, dtype=np.float64)
        - np.asarray(gas.reactant_stoich_coeffs, dtype=np.float64),
    )
    qj = np.ascontiguousarray(gas.net_rates_of_progress, dtype=np.float64)

    return _shock_param_rhs_gradient_kernel(
        A, rho, v, T, cp, W, hk, nu, qj, Wk,
        geometry.rho1, geometry.A1,
    )


def _shock_atol_vector(K: int, atol_scalar: float) -> np.ndarray:
    """Per-component absolute tolerance vector for the shock state.

    Species mass fractions ride in 1e-12 territory for trace radicals;
    the gross shock state ``[z, A, ρ, v, T, t_shock]`` rides in O(1)-O(2000).
    Scaling species atol by 1e-2 below the scalar keeps the species
    error budget meaningful when the scalar is dialled in for the
    gross state.

    Used for the forward state, the adjoint ``λ``, and forward-sens
    columns ``S_k`` — all live in the same state space.
    """
    atol = np.full(N_SHOCK + K, float(atol_scalar))
    atol[N_SHOCK:] = float(atol_scalar) * 1e-2

    return atol


class IncidentShockReactor:
    """Goldsmith / Speth incident-shock reactor, scipy or SUNDIALS.

    ``backend="sundials"`` drives our SUNDIALS CVODES binding through
    :class:`CVodeIntegrator`; advance via :meth:`step_to`. Default.

    ``backend="scipy"`` drives ``scipy.solve_ivp`` with ``method`` (one
    of ``"BDF"``, ``"Radau"``, ``"LSODA"``); the dense solution is
    cached for in-window evaluation via :meth:`dense_eval`.

    Validates ``T`` and ``ρ`` in the RHS; on invalid state writes
    :attr:`failure_reason` before raising so the catch site can wrap
    a typed :class:`IntegrationError`.
    """

    def __init__(
        self,
        gas: ct.Solution,
        *,
        rho1: float,
        u_reac: float,
        A1: float = 0.2,
        As: float = 0.2,
        L: float = 0.1,
        area_change: bool = False,
        backend: str = "sundials",
        rtol: float | None = None,
        atol: float | None = None,
        method: str = "BDF",
        max_steps: int = 200_000,
    ) -> None:
        if backend not in ("sundials", "scipy"):
            raise ValueError(
                f"backend must be 'sundials' or 'scipy', got {backend!r}"
            )
        self._gas = gas
        self._geom = _Geometry(
            rho1=float(rho1), A1=float(A1), L=float(L), As=float(As),
            area_change=bool(area_change),
        )
        self._u_reac = float(u_reac)
        self._Wk = np.asarray(gas.molecular_weights, dtype=float).copy()
        self._initial_y = np.hstack(
            (0.0, float(A1), gas.density, float(u_reac), gas.T, 0.0, gas.Y),
        )
        self._y = self._initial_y.copy()
        self._time = 0.0
        self._failure_reason: FailureReason | None = None
        self._backend = backend

        defaults = (CVODES_DEFAULT_TOLS if backend == "sundials"
                    else SCIPY_DEFAULT_TOLS)
        self._rtol = float(defaults["rtol"] if rtol is None else rtol)
        self._atol = float(defaults["atol"] if atol is None else atol)

        if backend == "sundials":
            self._sol = None
            self._method = None
            atol_vec = _shock_atol_vector(self._Wk.size, self._atol)
            self._integ = CVodeIntegrator(
                self._initial_y.size,
                rhs=self._rhs_callback,
                jac=self._jac_callback,
                rtol=self._rtol,
                atol=atol_vec,
                max_steps=int(max_steps),
            )
            self._integ.reinit(0.0, self._initial_y)
        else:
            self._sol = None  # OdeResult after first advance()
            self._method = method
            self._integ = None

    @property
    def gas(self) -> ct.Solution:
        return self._gas

    @property
    def time(self) -> float:
        return self._time

    @property
    def failure_reason(self) -> FailureReason | None:
        return self._failure_reason

    @property
    def initial_state(self) -> np.ndarray:
        return self._initial_y

    @property
    def backend(self) -> str:
        return self._backend

    def get_state(self) -> np.ndarray:
        return self._y

    # ─── shared RHS / Jacobian (used by both backends) ───────────────

    def _validate_state(self, T: float, rho: float) -> None:
        """Check ``T`` and ``ρ`` are finite and positive.

        Raises:
            RuntimeError: If ``T`` or ``ρ`` is non-finite or non-positive.
                ``failure_reason`` is set on the instance to the matching
                :class:`FailureReason` before the raise.
        """
        if not np.isfinite(T) or T <= 0.0:
            self._failure_reason = FailureReason.TEMPERATURE_INVALID
            raise RuntimeError("ODE Error: Temperature is invalid")
        if not np.isfinite(rho) or rho <= 0.0:
            self._failure_reason = FailureReason.DENSITY_INVALID
            raise RuntimeError("ODE Error: Density is invalid")

    def _rhs(self, t, y):
        """``solve_ivp``-style RHS — validates, syncs gas, returns ydot."""
        T = float(y[I_T])
        rho = float(y[I_RHO])
        self._validate_state(T, rho)
        self._gas.set_unnormalized_mass_fractions(y[N_SHOCK:])
        self._gas.TD = T, rho

        return _shock_derivatives(
            self._gas, float(y[I_Z]), float(y[I_A]), float(y[I_V]), self._Wk,
            rho1=self._geom.rho1, A1=self._geom.A1, L=self._geom.L,
            As=self._geom.As, area_change=self._geom.area_change,
        )

    def _rhs_callback(self, t: float, y: np.ndarray, ydot: np.ndarray) -> None:
        ydot[:] = self._rhs(t, y)

    def _jac_callback(
        self, t: float, y: np.ndarray, fy: np.ndarray, J: np.ndarray,
    ) -> None:
        J[:] = _shock_jacobian(self._gas, y, self._geom, self._Wk)

    # ─── SUNDIALS advance ────────────────────────────────────────────

    def step_to(self, t: float) -> tuple[float, np.ndarray]:
        """Integrate forward to ``t`` (SUNDIALS normal mode).

        Returns:
            ``(t_reached, y)`` — the actual integration time
            SUNDIALS returned (may be slightly under ``t`` on
            warnings) and the full state vector at that time.

        Raises:
            RuntimeError: If the reactor was not constructed with
                ``backend="sundials"``.
        """
        if self._backend != "sundials":
            raise RuntimeError("step_to() requires backend='sundials'")
        t_reached, y_out = self._integ.step_to(float(t))
        self._time = t_reached
        self._y = y_out

        return t_reached, y_out

    def num_steps(self) -> int:
        if self._backend != "sundials":
            raise RuntimeError("num_steps() requires backend='sundials'")
        return self._integ.num_steps()

    def stats(self) -> dict:
        """CVODES diagnostic counters from the last solve.

        See :meth:`CVodeIntegrator.stats` for the field list.

        Raises:
            RuntimeError: If the reactor was not constructed with
                ``backend="sundials"``.
        """
        if self._backend != "sundials":
            raise RuntimeError("stats() requires backend='sundials'")
        return self._integ.stats()

    # ─── scipy advance + dense eval ──────────────────────────────────

    def advance(self, t: float) -> None:
        """Integrate from t=0 up to (at least) ``t``.

        Caches the dense solution so subsequent in-range advances are
        O(eval).

        Raises:
            RuntimeError: If the reactor was not constructed with
                ``backend="scipy"``, or if ``solve_ivp`` reports failure.
        """
        if self._backend != "scipy":
            raise RuntimeError("advance() requires backend='scipy'")
        if self._sol is None or t > self._sol.t[-1]:
            sol = solve_ivp(
                self._rhs, [0.0, float(t)], self._initial_y,
                method=self._method, dense_output=True,
                rtol=self._rtol, atol=self._atol,
            )
            if not sol.success:
                raise RuntimeError(sol.message)
            self._sol = sol

        if t <= 0.0:
            self._y = self._initial_y.copy()
        else:
            self._y = self._sol.sol(t).flatten()
        self._time = float(t)

    def integration_grid(self) -> np.ndarray:
        """Solver-chosen integration time points after :meth:`advance`."""
        if self._backend != "scipy":
            raise RuntimeError("integration_grid() requires backend='scipy'")
        if self._sol is None:
            raise RuntimeError("reactor has not been advanced")
        return self._sol.t

    def dense_eval(self, times) -> np.ndarray:
        """Evaluate the cached dense solution at ``times``.

        Returns:
            Array of shape ``(len(times), n_state)``.
        """
        if self._backend != "scipy":
            raise RuntimeError("dense_eval() requires backend='scipy'")
        if self._sol is None:
            raise RuntimeError("reactor has not been advanced")
        return self._sol.sol(np.asarray(times)).T


def _integrate_scipy(gas, t_end, var, rtol, atol):
    """scipy ``solve_ivp`` backend, driven through :class:`IncidentShockReactor`."""
    reactor = IncidentShockReactor(
        gas,
        rho1=var["rho1"], u_reac=var["u_reac"],
        A1=var["A1"], As=var["As"], L=var["L"],
        area_change=var["area_change"],
        backend="scipy", method=var["ODE_solver"],
        rtol=rtol, atol=atol,
    )

    try:
        reactor.advance(t_end)
    except Exception as e:
        reason = reactor.failure_reason or FailureReason.SOLVER_FAILURE
        return [], False, str(e), reason

    if var["t_lab_save"] is None:
        t_out = reactor.integration_grid()
    else:
        extra = np.atleast_1d(var["t_lab_save"])
        t_out = np.unique(np.sort(np.concatenate([reactor.integration_grid(), extra])))

    states = reactor.dense_eval(t_out)
    trajectory = list(zip(t_out.tolist(), states))

    return trajectory, True, "", None


def _integrate_cvodes(gas, t_end, var, rtol, atol):
    """SUNDIALS CVODES backend, driven through :class:`IncidentShockReactor`."""
    reactor = IncidentShockReactor(
        gas,
        rho1=var["rho1"], u_reac=var["u_reac"],
        A1=var["A1"], As=var["As"], L=var["L"],
        area_change=var["area_change"],
        backend="sundials",
        rtol=rtol, atol=atol,
    )

    grid = np.geomspace(t_end * 1e-4, t_end, int(var["n_output"]))
    if var["t_lab_save"] is not None:
        extra = np.atleast_1d(var["t_lab_save"])
        extra = extra[(extra > 0.0) & (extra <= t_end)]
        grid = np.unique(np.sort(np.concatenate([grid, extra])))

    trajectory = [(0.0, reactor.initial_state.copy())]
    try:
        for t in grid:
            t_reached, y_out = reactor.step_to(t)
            trajectory.append((t_reached, y_out.copy()))
    except Exception as e:
        reason = reactor.failure_reason or FailureReason.SOLVER_FAILURE
        return trajectory, False, str(e), reason

    return trajectory, True, "", None


def _incident_shock_reactor(gas, details, t_end, **kwargs):
    if "u_reac" not in kwargs or "rho1" not in kwargs:
        details["success"] = False
        details["message"] = "velocity and rho1 not specified\n"
        return None, details

    var = {
        "sim_int_f": 1,
        "observable": {"main": "Density Gradient", "sub": 0},
        "A1": 0.2,
        "As": 0.2,
        "L": 0.1,
        "t_lab_save": None,
        "ODE_solver": "CVODES",
        "area_change": False,
        "n_output": 100,
    }
    var.update(kwargs)

    backend = var["ODE_solver"].upper()
    defaults = CVODES_DEFAULT_TOLS if backend == "CVODES" else SCIPY_DEFAULT_TOLS
    rtol = var.get("rtol", defaults["rtol"])
    atol = var.get("atol", defaults["atol"])

    if backend == "CVODES":
        trajectory, sol_success, sol_message, failure_reason = _integrate_cvodes(
            gas, t_end, var, rtol, atol
        )
    else:
        trajectory, sol_success, sol_message, failure_reason = _integrate_scipy(
            gas, t_end, var, rtol, atol
        )

    details["success"] = sol_success
    if not sol_success:
        explanation = "\nCheck for: Fast rates or bad thermo data"
        flagged = check_rxn_rates(gas)
        if len(flagged) > 0:
            explanation += "\nSuggested Reactions: " + ", ".join(
                str(x) for x in flagged
            )
        details["message"] = "\nODE Error: {:s}\n{:s}\n".format(sol_message, explanation)
        details["failure_reason"] = failure_reason

    states = ct.SolutionArray(
        gas,
        extra=["t", "t_shock", "z", "A", "vel",
               "drhodz_tot", "drhodz", "perc_drhodz"],
    )
    if details["success"]:
        for t, y in trajectory:
            z, A, rho, v, T, t_shock = y[0:6]
            states.append(
                TDY=(T, rho, y[6:]),
                t=t, t_shock=t_shock, z=z, A=A, vel=v,
                drhodz_tot=np.nan, drhodz=np.nan, perc_drhodz=np.nan,
            )
    else:
        states.append(
            TDY=(gas.T, gas.density, gas.Y),
            t=0.0, t_shock=0.0, z=0.0,
            A=var["A1"], vel=var["u_reac"],
            drhodz_tot=np.nan, drhodz=np.nan, perc_drhodz=np.nan,
        )

    ind_var = "t_lab"
    reactor_vars = [
        "t_lab", "t_shock", "z", "A", "vel",
        "T", "P", "h_tot", "h", "s_tot", "s", "rho",
        "drhodz_tot", "drhodz", "perc_drhodz", "perc_abs_drhodz",
        "Y", "X", "conc", "wdot", "wdotfor", "wdotrev",
        "HRR_tot", "HRR", "delta_h", "delta_s",
        "eq_con", "rate_con", "rate_con_rev",
        "net_ROP", "for_ROP", "rev_ROP",
    ]

    num = {
        "reac": np.sum(gas.reactant_stoich_coeffs, axis=0),
        "prod": np.sum(gas.product_stoich_coeffs, axis=0),
        "rxns": gas.n_reactions,
    }

    SIM = ReactorOutput(num, states, reactor_vars)
    SIM.finalize(details["success"], ind_var, var["observable"], units="CGS")
    return SIM, details


def run_incident_shock(mech, t_end, T_reac, P_reac, mix, **kwargs):
    """Run an incident shock reactor.

    Holds ``mech.exclusive()`` for the call's duration.

    Returns:
        ``(ReactorOutput | None, details_dict)``. The output is ``None``
        if ``mech.set_TPX`` rejects the post-shock state; ``details_dict``
        always carries at least ``success`` and ``message``.
    """
    if isinstance(mix, list):
        mix = list2ct_mixture(mix)

    with mech.exclusive():
        mech_out = mech.set_TPX(T_reac, P_reac, mix)
        if not mech_out["success"]:
            return None, mech_out

        return _incident_shock_reactor(
            mech.gas, {"success": False, "message": []}, t_end, **kwargs
        )
