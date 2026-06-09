"""Sensitivity-analysis observables: math, registry, and projection.

One module owns the chain-rule math for every observable Frhodo's
sensitivity pipeline supports (``drhodz_tot``, ``T``, ``P``, ``Y``,
``X``, ``conc``). Each :class:`Observable` exposes two callables:

* ``terminal_fn(gas, y, geometry, Wk, n_rxns, species_idx) ->
  (g, direct, dg_dy)``. ``g`` is the scalar observable value (used
  for normalization), ``dg_dy`` is the ``∂g/∂y`` row vector consumed
  by the SUNDIALS adjoint backward IC and the forward-sensitivity
  projection, and ``direct`` is the explicit ``∂g/∂p`` term (non-zero
  only for ``drhodz_tot``).

* ``native_project_fn(species_idx, sens_state_list, snapshot_list,
  idx, gas) -> sens_array``. Chain-rules Cantera-normalized state
  sensitivities ``s = ∂ln(state)/∂ln(k)`` into the observable's
  normalized sensitivity. Used by the Cantera-native path only.

Both callables encode the same six observables, but the SUNDIALS
path operates on raw state at one ``(t_m, y_m)`` while native
operates on a trajectory of normalized log-derivatives. Putting them
in one module ensures the two formulations stay in sync.

The density-gradient kernel lives here too: ``_drhodz_formula`` and
``_drhodz_per_rxn_formula``. Trajectory-shaped (``drhodz`` /
``drhodz_per_rxn``) and single-state (``drhodz_at_state`` /
``drhodz_per_rxn_at_state``) wrappers all call the kernel — the
trajectory variants are used by the reactor-output display registry,
the single-state variants by SUNDIALS callbacks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import cantera as ct
import numpy as np

from frhodo.simulation.shock.incident_shock_reactor import (
    I_A, I_RHO, I_T, I_V, I_Z, N_SHOCK,
    _Geometry, _shock_jacobian,
)


Ru = ct.gas_constant


ObservableName = Literal["drhodz_tot", "HRR_tot", "T", "P", "Y", "X", "conc"]


# ────────────────────────────── kernels ──────────────────────────────

def _drhodz_formula(v, cp, T, W, species_term, area_change_term=0.0):
    """``drhodz = (species_term − area_change_term) / (v · (1 + β))``.

    Where ``β = v²·(1/(cp·T) − W/(Ru·T))``. Inputs may be scalars
    (single-state call) or broadcast-compatible arrays (trajectory
    call); the formula is shape-agnostic.
    """
    beta = v * v * (1.0 / (cp * T) - W / (Ru * T))

    return (species_term - area_change_term) / (v * (1.0 + beta))


def _area_change_term(states, L, As, A1):
    """``ρ · β · (dA/dt) / A`` along a trajectory, for the Mirels-area-change case."""
    n = 0.5
    z = states.z
    A = states.A
    rho = states.density
    vel = states.vel
    T = states.T
    cp = states.cp_mass
    Wmix = states.mean_molecular_weight

    beta = vel ** 2 * (1.0 / (cp * T) - Wmix / (Ru * T))
    xi = np.maximum(z / L, 1e-10)
    dAdt = vel * As * n / L * xi ** (n - 1.0) / (1.0 - xi ** n) ** 2.0

    return rho * beta / A * dAdt


# ────────────────────── drhodz wrappers (trajectory) ──────────────────

def drhodz(states, L=0.1, As=0.2, A1=0.2, area_change=False):
    """Total density gradient along a ``ct.SolutionArray`` trajectory.

    Returns:
        Array of shape ``(n_steps,)``.
    """
    hk_over_cpT_minus_Wmix = (
        states.partial_molar_enthalpies / (states.cp_mass * states.T)[:, None]
        - states.mean_molecular_weight[:, None]
    )
    species_term = np.sum(hk_over_cpT_minus_Wmix * states.net_production_rates, axis=1)
    area_term = _area_change_term(states, L, As, A1) if area_change else 0.0

    return _drhodz_formula(
        states.vel, states.cp_mass, states.T, states.mean_molecular_weight,
        species_term, area_term,
    )


def drhodz_per_rxn(states, L=0.1, As=0.2, A1=0.2, area_change=False, rxnNum=None):
    """Per-reaction density gradient along a trajectory.

    Args:
        rxnNum: Single reaction index, list of indices, or ``None`` for
            all reactions.

    Returns:
        Array of shape ``(n_steps, n_selected_rxns)``. The identity
        ``sum(axis=1) == drhodz`` holds only when ``area_change=False``.
    """
    nu_fwd = states.product_stoich_coeffs
    nu_rev = states.reactant_stoich_coeffs
    delta_N_full = np.sum(nu_fwd, axis=0) - np.sum(nu_rev, axis=0)

    if rxnNum is None:
        rxns = slice(None)
    elif isinstance(rxnNum, list):
        rxns = rxnNum
    else:
        rxns = [rxnNum]

    rj = states.net_rates_of_progress[:, rxns]
    hj = states.delta_enthalpy[:, rxns]
    delta_N = delta_N_full[rxns]

    species_term = rj * (
        hj / (states.cp_mass * states.T)[:, None]
        - states.mean_molecular_weight[:, None] * delta_N
    )
    area_term = (
        _area_change_term(states, L, As, A1)[:, None] if area_change else 0.0
    )

    return _drhodz_formula(
        states.vel[:, None], states.cp_mass[:, None], states.T[:, None],
        states.mean_molecular_weight[:, None],
        species_term, area_term,
    )


# ─────────────────── drhodz wrappers (single state) ──────────────────

def drhodz_at_state(gas, T, rho, Y, v, area_change_term=0.0):
    """Total density gradient at a single ``(T, ρ, Y, v)`` state.

    Mutates the gas thermo state to ``(T, ρ, Y)``.
    """
    gas.TDY = T, rho, Y
    cp = gas.cp_mass
    W = gas.mean_molecular_weight
    hk = gas.partial_molar_enthalpies
    wdot = gas.net_production_rates
    species_term = float(np.sum((hk / (cp * T) - W) * wdot))

    return _drhodz_formula(v, cp, T, W, species_term, area_change_term)


def drhodz_per_rxn_at_state(gas, T, rho, Y, v, area_change_term=0.0):
    """Per-reaction density gradient at a single ``(T, ρ, Y, v)`` state.

    Mutates the gas thermo state to ``(T, ρ, Y)``.

    Returns:
        Array of length ``gas.n_reactions``.
    """
    gas.TDY = T, rho, Y
    cp = gas.cp_mass
    W = gas.mean_molecular_weight
    rj = gas.net_rates_of_progress
    hj = gas.delta_enthalpy
    delta_N = (gas.product_stoich_coeffs - gas.reactant_stoich_coeffs).sum(axis=0)
    species_term = rj * (hj / (cp * T) - W * delta_N)

    return _drhodz_formula(v, cp, T, W, species_term, area_change_term)


# ────────────────────── HRR wrappers (single state) ──────────────────

def hrr_at_state(gas, T, rho, Y) -> float:
    """Volumetric heat release rate at a single ``(T, ρ, Y)`` state.

    Returns:
        Heat release rate in W/m³, positive for exothermic.
        Mutates ``gas.TDY``.
    """
    gas.TDY = T, rho, Y

    return float(gas.heat_release_rate)


def hrr_per_rxn_at_state(gas, T, rho, Y) -> np.ndarray:
    """Per-reaction volumetric heat release rate (W/m³).

    Each entry is ``-Δh_j · r_j`` for reaction ``j``. Mutates ``gas.TDY``.

    Returns:
        Array of length ``gas.n_reactions``.
    """
    gas.TDY = T, rho, Y

    return -gas.delta_enthalpy * gas.net_rates_of_progress


# ─────────────── shared FD helpers for state-sensitivity ─────────────

def _fd_state_partials(
    gas, eval_fn, T, rho, Y, baseline, eps=1e-6, normalize=True,
):
    """Local-FD ``∂g/∂(T, ρ, Y)`` for an observable evaluated by ``eval_fn``.

    Args:
        eval_fn: ``eval_fn(gas, T, rho, Y) -> float`` returns the scalar
            observable at the given thermo state.
        normalize: If ``True``, partials are log-derivatives
            ``A_s = ∂ln(g)/∂ln(s)``; otherwise raw ``A_s = ∂g/∂s``.

    Returns:
        ``(A_T, A_rho, A_Y)``. ``Y`` components with mass fraction ≤ 0
        contribute zero (FD on a species that doesn't exist would be
        ill-defined).
    """
    if normalize:
        denom = baseline
        A_T = (eval_fn(gas, T * (1 + eps), rho, Y) - baseline) / (eps * denom)
        A_rho = (eval_fn(gas, T, rho * (1 + eps), Y) - baseline) / (eps * denom)
    else:
        A_T = (eval_fn(gas, T * (1 + eps), rho, Y) - baseline) / (T * eps)
        A_rho = (eval_fn(gas, T, rho * (1 + eps), Y) - baseline) / (rho * eps)

    n_species = Y.size
    A_Y = np.zeros(n_species)
    for i in range(n_species):
        if Y[i] <= 0.0:
            continue
        Y_pert = Y.copy()
        Y_pert[i] = Y[i] * (1 + eps)
        if normalize:
            A_Y[i] = (eval_fn(gas, T, rho, Y_pert) - baseline) / (eps * denom)
        else:
            A_Y[i] = (eval_fn(gas, T, rho, Y_pert) - baseline) / (Y[i] * eps)

    return A_T, A_rho, A_Y


# ──────────────── terminal_fn (SUNDIALS terminal / projection) ─────────

def _terminal_drhodz_tot(gas, y, geometry, Wk, n_rxns, species_idx):
    g = float(drhodz_at_state(
        gas, float(y[I_T]), float(y[I_RHO]), y[N_SHOCK:], float(y[I_V]), 0.0,
    ))
    if abs(g) < 1e-30:
        return g, None, None
    direct = drhodz_per_rxn_at_state(
        gas, float(y[I_T]), float(y[I_RHO]), y[N_SHOCK:], float(y[I_V]), 0.0,
    )
    J = _shock_jacobian(gas, y, geometry, Wk)
    scale = y[I_RHO] * y[I_A] / (geometry.rho1 * geometry.A1)
    f_z = y[I_V] * scale
    dg_dy = (J[I_RHO] - g * J[I_Z]) / f_z

    return g, direct, dg_dy


def _terminal_hrr_tot(gas, y, geometry, Wk, n_rxns, species_idx):
    T = float(y[I_T])
    rho = float(y[I_RHO])
    Y = np.asarray(y[N_SHOCK:], dtype=float)
    g = hrr_at_state(gas, T, rho, Y)
    if abs(g) < 1e-30:
        return g, None, None
    direct = hrr_per_rxn_at_state(gas, T, rho, Y)

    # HRR depends only on (T, ρ, Y); no velocity term in dg/dy.
    A_T, A_rho, A_Y = _fd_state_partials(
        gas, hrr_at_state, T, rho, Y, baseline=g, normalize=False,
    )
    n_state = N_SHOCK + Wk.size
    dg_dy = np.zeros(n_state)
    dg_dy[I_T] = A_T
    dg_dy[I_RHO] = A_rho
    dg_dy[N_SHOCK:] = A_Y

    return g, direct, dg_dy


def _terminal_T(gas, y, geometry, Wk, n_rxns, species_idx):
    n_state = N_SHOCK + Wk.size
    g = float(y[I_T])
    dg_dy = np.zeros(n_state)
    dg_dy[I_T] = 1.0

    return g, None, dg_dy


def _terminal_Y(gas, y, geometry, Wk, n_rxns, species_idx):
    if species_idx is None:
        raise ValueError("observable='Y' requires species_idx")
    n_state = N_SHOCK + Wk.size
    idx = N_SHOCK + species_idx
    g = float(y[idx])
    if abs(g) < 1e-30:
        return g, None, None
    dg_dy = np.zeros(n_state)
    dg_dy[idx] = 1.0

    return g, None, dg_dy


def _terminal_X(gas, y, geometry, Wk, n_rxns, species_idx):
    if species_idx is None:
        raise ValueError("observable='X' requires species_idx")
    gas.set_unnormalized_mass_fractions(y[N_SHOCK:])
    gas.TD = float(y[I_T]), float(y[I_RHO])
    W_mix = float(gas.mean_molecular_weight)
    Y_k = float(y[N_SHOCK + species_idx])
    W_k = Wk[species_idx]
    g = W_mix * Y_k / W_k
    if abs(g) < 1e-30:
        return g, None, None
    n_state = N_SHOCK + Wk.size
    dg_dy = np.zeros(n_state)
    # ∂X_k/∂Y_j = (W_mix/W_k)·δ_jk − X_k·W_mix/W_j
    dg_dy[N_SHOCK:] = -g * W_mix / Wk
    dg_dy[N_SHOCK + species_idx] += W_mix / W_k

    return g, None, dg_dy


def _terminal_conc(gas, y, geometry, Wk, n_rxns, species_idx):
    if species_idx is None:
        raise ValueError("observable='conc' requires species_idx")
    rho = float(y[I_RHO])
    Y_k = float(y[N_SHOCK + species_idx])
    W_k = Wk[species_idx]
    g = rho * Y_k / W_k
    if abs(g) < 1e-30:
        return g, None, None
    n_state = N_SHOCK + Wk.size
    dg_dy = np.zeros(n_state)
    dg_dy[I_RHO] = Y_k / W_k
    dg_dy[N_SHOCK + species_idx] = rho / W_k

    return g, None, dg_dy


def _terminal_P(gas, y, geometry, Wk, n_rxns, species_idx):
    gas.set_unnormalized_mass_fractions(y[N_SHOCK:])
    gas.TD = float(y[I_T]), float(y[I_RHO])
    W_mix = float(gas.mean_molecular_weight)
    P = float(gas.P)
    n_state = N_SHOCK + Wk.size
    dg_dy = np.zeros(n_state)
    dg_dy[I_RHO] = P / float(y[I_RHO])
    dg_dy[I_T] = P / float(y[I_T])
    # ∂P/∂Y_j = P·W_mix/W_j  (from s(P) = s(ρ) + s(T) − s(W_mix),
    # s(W_mix) = −W_mix·Σ Y_i/W_i · s(Y_i))
    dg_dy[N_SHOCK:] = P * W_mix / Wk

    return P, None, dg_dy


# ─────────────── native_project_fn (Cantera chain rule) ───────────────

def _all_species_sens(sens_step: np.ndarray, idx: dict, gas) -> np.ndarray:
    n_species = gas.n_species
    n_rxns = sens_step.shape[1]
    out = np.zeros((n_species, n_rxns), dtype=float)
    for j in range(n_species):
        if j in idx["species"]:
            out[j] = sens_step[idx["species"][j]]

    return out


def _density_sensitivity_normalized(
    sens_step: np.ndarray, idx: dict, snap: dict, gas, dW_over_W: np.ndarray,
) -> np.ndarray:
    """Normalized ``s(ρ)`` derived from whichever state variables the
    reactor exposes.

    * Incident-shock reactor: ``ρ`` is a state variable → read directly.
    * 0-D const-V Mole reactor: closed system, ``ρ = M_tot / V`` with
      ``V`` fixed → fall back through ``s(W)`` and species sums.
    * 0-D const-P Mole reactor: ``ρ = W·P / (R·T)`` with ``P`` fixed →
      ``s(ρ) = s(W) − s(T)``.
    """
    if "density" in idx:
        return sens_step[idx["density"]]
    if "volume" in idx:
        return -sens_step[idx["volume"]]
    dT = sens_step[idx["temperature"]] if "temperature" in idx else 0.0

    return dW_over_W - dT


def _project_T(species_idx, sens_state_list, snapshot_list, idx, gas):
    sens_arr = np.stack(sens_state_list, axis=0)

    return sens_arr[:, idx["temperature"], :]


def _project_Y(species_idx, sens_state_list, snapshot_list, idx, gas):
    sens_arr = np.stack(sens_state_list, axis=0)

    return sens_arr[:, idx["species"][species_idx], :]


def _project_X(species_idx, sens_state_list, snapshot_list, idx, gas):
    sens_arr = np.stack(sens_state_list, axis=0)
    n_steps, _, n_rxns = sens_arr.shape
    Wk = np.asarray(gas.molecular_weights, dtype=float)
    out = np.zeros((n_steps, n_rxns), dtype=float)
    for step in range(n_steps):
        snap = snapshot_list[step]
        dY = _all_species_sens(sens_arr[step], idx, gas)
        dW_over_W = -np.sum(
            (snap["Y"][:, None] / Wk[:, None]) * dY, axis=0,
        ) * snap["W_mix"]
        # X_k = (Y_k / W_k) · W_mix; W_k const → s(X_k) = s(Y_k) + s(W_mix)
        out[step] = dY[species_idx] + dW_over_W

    return out


def _project_conc(species_idx, sens_state_list, snapshot_list, idx, gas):
    sens_arr = np.stack(sens_state_list, axis=0)
    n_steps, _, n_rxns = sens_arr.shape
    Wk = np.asarray(gas.molecular_weights, dtype=float)
    out = np.zeros((n_steps, n_rxns), dtype=float)
    for step in range(n_steps):
        snap = snapshot_list[step]
        dY = _all_species_sens(sens_arr[step], idx, gas)
        dW_over_W = -np.sum(
            (snap["Y"][:, None] / Wk[:, None]) * dY, axis=0,
        ) * snap["W_mix"]
        # c_i = (ρ/W_i)·Y_i; W_i const → s(c_i) = s(ρ) + s(Y_i)
        drho = _density_sensitivity_normalized(
            sens_arr[step], idx, snap, gas, dW_over_W,
        )
        out[step] = drho + dY[species_idx]

    return out


def _project_P(species_idx, sens_state_list, snapshot_list, idx, gas):
    sens_arr = np.stack(sens_state_list, axis=0)
    n_steps, _, n_rxns = sens_arr.shape
    Wk = np.asarray(gas.molecular_weights, dtype=float)
    out = np.zeros((n_steps, n_rxns), dtype=float)
    for step in range(n_steps):
        snap = snapshot_list[step]
        dY = _all_species_sens(sens_arr[step], idx, gas)
        dW_over_W = -np.sum(
            (snap["Y"][:, None] / Wk[:, None]) * dY, axis=0,
        ) * snap["W_mix"]
        # P = ρ·R·T/W → s(P) = s(ρ) + s(T) − s(W)
        dT = sens_arr[step, idx["temperature"], :]
        drho = _density_sensitivity_normalized(
            sens_arr[step], idx, snap, gas, dW_over_W,
        )
        out[step] = drho + dT - dW_over_W

    return out


def _project_drhodz_tot(species_idx, sens_state_list, snapshot_list, idx, gas):
    sens_arr = np.stack(sens_state_list, axis=0)
    n_steps, _, n_rxns = sens_arr.shape
    out = np.zeros((n_steps, n_rxns), dtype=float)
    for step in range(n_steps):
        out[step] = _drhodz_chain_rule(sens_arr[step], idx, snapshot_list[step], gas)

    return out


def _project_hrr_tot(species_idx, sens_state_list, snapshot_list, idx, gas):
    sens_arr = np.stack(sens_state_list, axis=0)
    n_steps, _, n_rxns = sens_arr.shape
    out = np.zeros((n_steps, n_rxns), dtype=float)
    for step in range(n_steps):
        out[step] = _hrr_chain_rule(sens_arr[step], idx, snapshot_list[step], gas)

    return out


def _drhodz_chain_rule(sens_step: np.ndarray, idx: dict, snap: dict, gas) -> np.ndarray:
    """Normalized ``s(drhodz_tot) = ∂ln(drhodz)/∂ln(k_j)`` at one step.

    Direct partial: ``per_rxn[j] / drhodz`` (``drhodz`` is linear in
    each reaction's rate of progress; per-reaction share is the direct
    sensitivity).

    Indirect partial: chain-rule through state variables (T, ρ, Y, v).
    The state-side partials ``∂drhodz/∂s`` are computed by local FD
    on ``drhodz_at_state`` — no reactor re-run.
    """
    if "velocity" not in snap:
        raise ValueError(
            "drhodz_tot sensitivity requires the incident-shock reactor "
            "(no velocity component on 0-D reactors)"
        )
    T = snap["T"]
    rho = snap["rho"]
    Y = snap["Y"]
    v = snap["velocity"]

    saved_state = gas.TPY
    try:
        def eval_at(g, T_, rho_, Y_):
            return drhodz_at_state(g, T_, rho_, Y_, v, area_change_term=0.0)

        baseline = drhodz_at_state(gas, T, rho, Y, v, area_change_term=0.0)
        if abs(baseline) < 1e-30:
            return np.zeros(sens_step.shape[1])

        per_rxn = drhodz_per_rxn_at_state(gas, T, rho, Y, v, area_change_term=0.0)
        direct = per_rxn / baseline

        A_T, A_rho, A_Y = _fd_state_partials(
            gas, eval_at, T, rho, Y, baseline=baseline, normalize=True,
        )
        # Velocity partial — drhodz depends explicitly on v
        eps = 1e-6
        A_v = (drhodz_at_state(gas, T, rho, Y, v * (1 + eps), 0.0) - baseline) \
            / (eps * baseline)

        s_T = sens_step[idx["temperature"], :]
        s_v = sens_step[idx["velocity"], :] if "velocity" in idx else 0.0
        s_rho = sens_step[idx["density"], :] if "density" in idx else 0.0
        dY = _all_species_sens(sens_step, idx, gas)

        indirect = A_T * s_T + A_rho * s_rho + A_v * s_v
        indirect = indirect + (A_Y[:, None] * dY).sum(axis=0)
    finally:
        gas.TPY = saved_state

    return direct + indirect


def _hrr_chain_rule(sens_step: np.ndarray, idx: dict, snap: dict, gas) -> np.ndarray:
    """Normalized ``s(HRR_tot) = ∂ln(HRR)/∂ln(k_j)`` at one step.

    Direct partial: per-reaction ``−Δh_j · r_j / HRR`` (``HRR`` is
    linear in each reaction's rate of progress).

    Indirect partial: chain-rule through state variables (T, ρ, Y).
    No velocity dependence — HRR is purely thermochemistry.
    """
    T = snap["T"]
    rho = snap["rho"]
    Y = snap["Y"]

    saved_state = gas.TPY
    try:
        baseline = hrr_at_state(gas, T, rho, Y)
        if abs(baseline) < 1e-30:
            return np.zeros(sens_step.shape[1])

        per_rxn = hrr_per_rxn_at_state(gas, T, rho, Y)
        direct = per_rxn / baseline

        A_T, A_rho, A_Y = _fd_state_partials(
            gas, hrr_at_state, T, rho, Y, baseline=baseline, normalize=True,
        )

        s_T = sens_step[idx["temperature"], :]
        dY = _all_species_sens(sens_step, idx, gas)
        Wk = np.asarray(gas.molecular_weights, dtype=float)
        dW_over_W = -np.sum(
            (snap["Y"][:, None] / Wk[:, None]) * dY, axis=0,
        ) * snap["W_mix"]
        s_rho = _density_sensitivity_normalized(sens_step, idx, snap, gas, dW_over_W)

        indirect = A_T * s_T + A_rho * s_rho
        indirect = indirect + (A_Y[:, None] * dY).sum(axis=0)
    finally:
        gas.TPY = saved_state

    return direct + indirect


# ────────────────────────────── registry ──────────────────────────────

@dataclass(frozen=True)
class Observable:
    """One sensitivity observable: terminal/projection callables + metadata.

    Attributes:
        name: Observable identifier (matches the dispatch key in
            ``OBSERVABLES``).
        requires_species_idx: If ``True`` the observable is per-species
            and callers must supply ``species_idx``.
        requires_incident_shock: If ``True`` the observable references
            shock-only state (e.g. velocity) and 0-D reactors cannot
            compute it.
        terminal_fn: ``(gas, y, geometry, Wk, n_rxns, species_idx) ->
            (g, direct_partial, dg_dy)`` at a single state. Used by
            SUNDIALS adjoint and forward-sensitivity paths. ``direct``
            is the explicit ``∂g/∂p`` term, non-zero only for
            observables that depend on rates explicitly (drhodz, HRR).
        native_project_fn: ``(species_idx, sens_state_list, snapshot_list,
            idx, gas) -> sens_array``. Chain-rules Cantera-native
            normalized state sensitivities into the observable's
            normalized sensitivity over a trajectory.
    """
    name: str
    requires_species_idx: bool
    requires_incident_shock: bool
    terminal_fn: Callable
    native_project_fn: Callable


OBSERVABLES: dict[str, Observable] = {
    "T": Observable(
        name="T",
        requires_species_idx=False,
        requires_incident_shock=False,
        terminal_fn=_terminal_T,
        native_project_fn=_project_T,
    ),
    "P": Observable(
        name="P",
        requires_species_idx=False,
        requires_incident_shock=False,
        terminal_fn=_terminal_P,
        native_project_fn=_project_P,
    ),
    "Y": Observable(
        name="Y",
        requires_species_idx=True,
        requires_incident_shock=False,
        terminal_fn=_terminal_Y,
        native_project_fn=_project_Y,
    ),
    "X": Observable(
        name="X",
        requires_species_idx=True,
        requires_incident_shock=False,
        terminal_fn=_terminal_X,
        native_project_fn=_project_X,
    ),
    "conc": Observable(
        name="conc",
        requires_species_idx=True,
        requires_incident_shock=False,
        terminal_fn=_terminal_conc,
        native_project_fn=_project_conc,
    ),
    "drhodz_tot": Observable(
        name="drhodz_tot",
        requires_species_idx=False,
        requires_incident_shock=True,
        terminal_fn=_terminal_drhodz_tot,
        native_project_fn=_project_drhodz_tot,
    ),
    "HRR_tot": Observable(
        name="HRR_tot",
        requires_species_idx=False,
        requires_incident_shock=False,
        terminal_fn=_terminal_hrr_tot,
        native_project_fn=_project_hrr_tot,
    ),
}


def terminal(
    name: str,
    gas: ct.Solution,
    y: np.ndarray,
    geometry: _Geometry,
    Wk: np.ndarray,
    n_rxns: int,
    species_idx: int | None,
) -> tuple[float, np.ndarray | None, np.ndarray | None]:
    """Dispatch to the named observable's ``terminal_fn``.

    Returns:
        ``(g, direct_partial, dg_dy)`` — see :class:`Observable` for the
        contract.

    Raises:
        NotImplementedError: If ``name`` is not a registered observable.
    """
    try:
        obs = OBSERVABLES[name]
    except KeyError as e:
        raise NotImplementedError(
            f"sensitivity does not support observable={name!r}; "
            f"valid: {sorted(OBSERVABLES)}"
        ) from e

    return obs.terminal_fn(gas, y, geometry, Wk, n_rxns, species_idx)


def project_native(
    name: str,
    species_idx: int | None,
    sens_state_list: list[np.ndarray],
    snapshot_list: list[dict],
    idx: dict,
    gas: ct.Solution,
) -> np.ndarray:
    """Dispatch to the named observable's Cantera-native chain rule.

    Returns:
        Sensitivity array of shape ``(n_steps, n_rxns)``.

    Raises:
        NotImplementedError: If ``name`` is not a registered observable.
    """
    try:
        obs = OBSERVABLES[name]
    except KeyError as e:
        raise NotImplementedError(
            f"sensitivity does not support observable={name!r}; "
            f"valid: {sorted(OBSERVABLES)}"
        ) from e

    return obs.native_project_fn(species_idx, sens_state_list, snapshot_list, idx, gas)
