"""Sensitivity analysis with auto-selected method.

Backends:

- ``"forward_sens"`` — SUNDIALS CVODES forward sensitivity over the
  incident-shock state vector. One augmented ODE solve covers all
  observables; cost scales with ``N_rxns``. Default for incident-shock.
- ``"adjoint"`` — SUNDIALS CVODES adjoint sensitivity. Cost scales
  with the number of output time points, not the mechanism size.
- ``"native"`` — Cantera's forward sensitivity API. Supported for
  0-D reactors only.
- ``"fd"`` — finite-difference perturbation across reactions,
  optionally parallelized. Used for very large mechanisms where
  parallel FD beats forward_sens, and as a sanity oracle.

``method="auto"`` routes incident-shock to ``forward_sens`` and
0-D reactors to ``native``; parallel FD takes over when
``n_workers > 1`` and the mechanism is large enough that parallel FD
overtakes the analytical path.
"""
from __future__ import annotations

import multiprocessing as mp
import os
from typing import Literal

import numpy as np
import cantera as ct

from frhodo.simulation.shock.sundials_sens import (
    compute_adjoint_sensitivity,
    compute_forward_sensitivity,
)
from frhodo.simulation.shock.incident_shock_reactor import run_incident_shock
from frhodo.simulation.shock.state import zero_d_mode_from_label
from frhodo.simulation.shock.zero_d_reactor import run_zero_d


Observable = Literal["T", "P", "drhodz_tot", "Y", "X", "conc"]
Method = Literal["auto", "native", "fd", "adjoint", "forward_sens"]

_SPECIES_OBSERVABLES = {"Y", "X", "conc"}

# Crossover heuristic: parallel FD overtakes single-threaded native when
# the mech is large enough that native's augmented-Jacobian work
# outweighs the N_rxns/N_workers parallel scaling. ~3000 rxns is a
# rough empirical landmark on commodity hardware.
_AUTO_FD_THRESHOLD_RXNS = 3000

_INCIDENT_SHOCK_NAME = "Incident Shock Reactor"


def compute_sensitivity(
    mech,
    reactor_state,
    shock,
    observable: Observable,
    species_idx: int | None = None,
    time_grid: np.ndarray | None = None,
    method: Method = "auto",
    n_workers: int = 1,
    eps: float = 1e-2,
) -> tuple[np.ndarray, np.ndarray]:
    """Sensitivity of ``observable`` to each reaction's rate constant.

    Args:
        mech: Loaded mechanism.
        reactor_state: Runtime reactor settings (``t_end``, tolerances,
            reactor name).
        shock: Post-shock state with ``T_reactor``, ``P_reactor``,
            ``thermo_mix``, plus shock-only fields used by incident-shock.
        observable: Observable name (see module docstring).
        species_idx: Required when ``observable`` is per-species (``Y``,
            ``X``, ``conc``); ignored otherwise.
        time_grid: Optional output time grid; defaults to the solver's
            internal grid for ``native``/``fd``, or a 50-point linear
            grid for SUNDIALS methods.
        method: Backend selector — ``"auto"`` resolves to one of the
            other values; see module docstring.
        n_workers: Process-pool size for parallel FD; ignored unless
            ``method="fd"`` or ``method="auto"`` resolves to ``"fd"``.
        eps: Multiplicative perturbation magnitude for FD.

    Returns:
        ``(t, sens)`` with ``sens.shape == (len(t), n_rxns)`` and
        Cantera-normalized entries
        ``sens[i, k] = ∂ln(observable[i])/∂ln(k_rxn=k)``.

    Raises:
        ValueError: If a per-species observable is requested without
            ``species_idx``.
    """
    if observable in _SPECIES_OBSERVABLES and species_idx is None:
        raise ValueError(
            f"observable={observable!r} requires species_idx; got None"
        )

    resolved = _resolve_method(method, mech, observable, reactor_state, n_workers)
    if resolved == "native":
        return _compute_native(
            mech, reactor_state, shock,
            observable=observable, species_idx=species_idx,
            time_grid=time_grid,
        )
    if resolved == "adjoint":
        return compute_adjoint_sensitivity(
            mech, reactor_state, shock,
            observable=observable, species_idx=species_idx,
            time_grid=time_grid,
        )
    if resolved == "forward_sens":
        return compute_forward_sensitivity(
            mech, reactor_state, shock,
            observable=observable, species_idx=species_idx,
            time_grid=time_grid,
        )

    return _compute_fd(
        mech, reactor_state, shock,
        observable=observable, species_idx=species_idx,
        time_grid=time_grid, eps=eps, n_workers=n_workers,
    )


def _resolve_method(
    method: Method, mech, observable: str, reactor_state, n_workers: int,
) -> str:
    """Pick the cheapest method for the current call.

    Incident-shock routes to ``forward_sens``; 0-D reactors stay on
    ``native``. Parallel FD takes over for very large mechanisms when
    workers are available.
    """
    if method != "auto":
        return method
    if n_workers > 1 and mech.gas.n_reactions > _AUTO_FD_THRESHOLD_RXNS:
        return "fd"
    if reactor_state.name == _INCIDENT_SHOCK_NAME:
        return "forward_sens"

    return "native"


# ──────────────────────────── native path ────────────────────────────

def _compute_native(
    mech, reactor_state, shock,
    observable: str, species_idx: int | None,
    time_grid: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Cantera-native sensitivity via ``add_sensitivity_reaction``.

    0-D reactors only. The integration grid is the solver's internal
    grid; resampled to ``time_grid`` on return when supplied.
    """
    gas = mech.gas
    saved_state = gas.TPX

    try:
        gas.TPX = shock.T_reactor, shock.P_reactor, shock.thermo_mix
        reactor = _build_reactor(gas, reactor_state, shock)
        network = ct.ReactorNet([reactor])
        network.rtol = reactor_state.ode_rtol
        network.atol = reactor_state.ode_atol

        n_rxns = gas.n_reactions
        for k in range(n_rxns):
            reactor.add_sensitivity_reaction(k)

        idx = _state_index_map(reactor, gas)
        t_list: list[float] = []
        sens_state_list: list[np.ndarray] = []
        snapshot_list: list[dict] = []

        while network.time < reactor_state.t_end:
            network.step()
            if network.time > reactor_state.t_end:
                break
            t_list.append(network.time)
            sens_state_list.append(np.asarray(network.sensitivities()))
            snapshot_list.append(_capture_snapshot(reactor, gas))

        if not t_list:
            return np.zeros(0), np.zeros((0, n_rxns))

        t = np.asarray(t_list, dtype=float)
        sens = _project_to_observable(
            observable, species_idx,
            sens_state_list, snapshot_list, idx, gas,
        )
    finally:
        gas.TPX = saved_state

    return _resample(t, sens, time_grid)


def _state_index_map(reactor, gas) -> dict:
    """Resolve component indices for the chain-rule formulas.

    Names returned by ``component_name``/``component_index`` differ per
    reactor type; we look up by canonical name and tolerate misses so
    chain rule branches can guard on presence.
    """
    out: dict = {"species": {}}
    for name in (
        "mass", "volume", "internal_energy", "enthalpy",
        "temperature", "density", "velocity",
    ):
        try:
            i = reactor.component_index(name)
            if i >= 0:
                out[name] = i
        except Exception:
            pass

    for j, sp in enumerate(gas.species_names):
        try:
            i = reactor.component_index(sp)
            if i >= 0:
                out["species"][j] = i
        except Exception:
            pass

    return out


def _capture_snapshot(reactor, gas) -> dict:
    """Thermo snapshot used by the native chain-rule projection."""
    snap = {
        "T": float(gas.T),
        "P": float(gas.P),
        "rho": float(gas.density),
        "Y": np.asarray(gas.Y, dtype=float).copy(),
        "W_mix": float(gas.mean_molecular_weight),
    }
    velocity = getattr(reactor, "_v", None)
    if velocity is not None:
        snap["velocity"] = float(velocity)

    return snap


def _project_to_observable(
    observable: str, species_idx: int | None,
    sens_state_list: list[np.ndarray],
    snapshot_list: list[dict],
    idx: dict, gas,
) -> np.ndarray:
    """Project state-variable sensitivities into the requested observable.

    Inputs are Cantera-normalized: ``sens[i, j] = d ln(y_i) / d ln(k_j)``.
    Outputs preserve the same normalization for the observable.

    Chain rule shortcuts on normalized sensitivities:
    - product rule: s(A·B) = s(A) + s(B)
    - ratio rule:   s(A/B) = s(A) - s(B)
    - mixture MW: s(W_mix) = -W_mix · Σ(Y_i · s(Y_i) / W_i)
    """
    sens_arr = np.stack(sens_state_list, axis=0)
    n_steps = sens_arr.shape[0]
    n_rxns = sens_arr.shape[2]

    Wk = np.asarray(gas.molecular_weights, dtype=float)

    if observable == "T":
        return sens_arr[:, idx["temperature"], :]

    if observable == "Y":
        return sens_arr[:, idx["species"][species_idx], :]

    out = np.zeros((n_steps, n_rxns), dtype=float)
    for step in range(n_steps):
        snap = snapshot_list[step]
        dY = _all_species_sens(sens_arr[step], idx, gas)
        # s(W_mix) = -W_mix · Σ(Y_i · s(Y_i) / W_i); the leading W_mix is
        # absorbed because we only ever multiply by 1/W_mix downstream.
        dW_over_W = -np.sum((snap["Y"][:, None] / Wk[:, None]) * dY, axis=0) * snap["W_mix"]

        if observable == "X":
            # X_i = Y_i · W_mix / W_i  ⇒  s(X_i) = s(Y_i) - s(W_mix)·(-1) ...
            # equivalently: s(X_i) = s(Y_i) + s(W_mix), using product rule.
            # Wait: X_i = (Y_i / W_i) · W_mix; s(constant) = 0, so
            #   s(X_i) = s(Y_i) + s(W_mix).
            dY_i = dY[species_idx]
            out[step] = dY_i + dW_over_W
        elif observable == "conc":
            # c_i = (ρ / W_i) · Y_i; W_i constant ⇒ s(c_i) = s(ρ) + s(Y_i).
            drho = _density_sensitivity_normalized(
                sens_arr[step], idx, snap, gas, dW_over_W,
            )
            dY_i = dY[species_idx]
            out[step] = drho + dY_i
        elif observable == "P":
            # P = ρ·R·T / W_mix ⇒ s(P) = s(ρ) + s(T) - s(W_mix).
            dT = sens_arr[step, idx["temperature"], :]
            drho = _density_sensitivity_normalized(
                sens_arr[step], idx, snap, gas, dW_over_W,
            )
            out[step] = drho + dT - dW_over_W
        else:
            raise ValueError(
                f"native path does not support observable={observable!r}"
            )

    return out


def _all_species_sens(sens_step: np.ndarray, idx: dict, gas) -> np.ndarray:
    n_species = gas.n_species
    n_rxns = sens_step.shape[1]
    out = np.zeros((n_species, n_rxns), dtype=float)
    for j in range(n_species):
        if j in idx["species"]:
            out[j] = sens_step[idx["species"][j]]

    return out


def _density_sensitivity_normalized(
    sens_step: np.ndarray, idx: dict, snap: dict, gas,
    dW_over_W: np.ndarray,
) -> np.ndarray:
    """Normalized ``s(ρ)`` derived from whichever state variables the
    0-D reactor exposes. Inputs and output are normalized
    (``s(y) = d ln(y) / d ln(k)``).

    - ``IdealGasReactor`` (const V): closed system ⇒ s(m) = 0,
      so s(ρ) = s(m) − s(V) = −s(V).
    - ``IdealGasConstPressureReactor``: ρ = P·W/(R·T), P fixed ⇒
      s(ρ) = s(W) − s(T).
    """
    if "density" in idx:
        return sens_step[idx["density"]]
    if "volume" in idx:
        return -sens_step[idx["volume"]]

    dT = sens_step[idx["temperature"]] if "temperature" in idx else 0.0

    return dW_over_W - dT


# ────────────────────────────── FD path ──────────────────────────────

def _compute_fd(
    mech, reactor_state, shock,
    observable: str, species_idx: int | None,
    time_grid: np.ndarray | None,
    eps: float,
    n_workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Finite-difference sensitivity via per-reaction multiplier perturbations.

    Each reaction's rate constant is scaled by ``(1 + eps)`` in a
    separate full reactor solve. Dispatches to serial or parallel
    workers based on ``n_workers``.
    """
    gas = mech.gas
    n_rxns = gas.n_reactions
    gas.set_multiplier(1.0)

    try:
        SIM_base = _run_inline(mech, reactor_state, shock)
        if SIM_base is None:
            return np.zeros(0), np.zeros((0, n_rxns))
        t_base = np.asarray(SIM_base.t_lab(units="SI"), dtype=float)
        obs_base = _observable_trace(SIM_base, observable, species_idx)

        if time_grid is None:
            grid = t_base
        else:
            grid = np.asarray(time_grid, dtype=float)
        obs_base_g = np.interp(grid, t_base, obs_base)

        if n_workers > 1:
            sens = _fd_parallel(
                mech, reactor_state, shock, observable, species_idx,
                eps, n_workers, grid, obs_base_g,
            )
        else:
            sens = _fd_serial(
                mech, reactor_state, shock, observable, species_idx,
                eps, grid, obs_base_g,
            )
    finally:
        gas.set_multiplier(1.0)

    return grid, sens


def _fd_serial(
    mech, reactor_state, shock, observable, species_idx, eps,
    grid, obs_base_g,
) -> np.ndarray:
    gas = mech.gas
    n_rxns = gas.n_reactions
    safe_base = np.where(np.abs(obs_base_g) > 1e-30, obs_base_g, 1.0)
    sens = np.zeros((grid.size, n_rxns), dtype=float)
    for k in range(n_rxns):
        gas.set_multiplier(1.0 + eps, k)
        try:
            SIM = _run_inline(mech, reactor_state, shock)
        finally:
            gas.set_multiplier(1.0, k)
        if SIM is None:
            sens[:, k] = np.nan
            continue
        t_p = np.asarray(SIM.t_lab(units="SI"), dtype=float)
        obs_p = _observable_trace(SIM, observable, species_idx)
        sens[:, k] = (np.interp(grid, t_p, obs_p) - obs_base_g) / (safe_base * eps)

    return sens


def _fd_parallel(
    mech, reactor_state, shock, observable, species_idx, eps,
    n_workers, grid, obs_base_g,
) -> np.ndarray:
    # mp.Pool can't pickle the live mech instance; workers reload from a
    # YAML file. The fork start method (Linux/macOS default) lets workers
    # inherit the mech for free; spawn (Windows) needs the yaml round-
    # trip. We pick fork when available.
    n_rxns = mech.gas.n_reactions
    ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn")

    yaml_text = mech.to_yaml_text()
    task_inputs = _PerturbTask.serialize_state(reactor_state, shock)

    with ctx.Pool(
        processes=n_workers,
        initializer=_PerturbTask.init_worker,
        initargs=(yaml_text, task_inputs, observable, species_idx, eps),
    ) as pool:
        traces = pool.map(_PerturbTask.run_one, range(n_rxns))

    safe_base = np.where(np.abs(obs_base_g) > 1e-30, obs_base_g, 1.0)
    sens = np.zeros((grid.size, n_rxns), dtype=float)
    for k, trace in enumerate(traces):
        if trace is None:
            sens[:, k] = np.nan
            continue
        t_p, obs_p = trace
        sens[:, k] = (np.interp(grid, t_p, obs_p) - obs_base_g) / (safe_base * eps)

    return sens


class _PerturbTask:
    """Worker-side state and per-reaction perturbation entry point."""

    _mech = None
    _reactor_state = None
    _shock = None
    _observable = None
    _species_idx = None
    _eps = None

    @staticmethod
    def serialize_state(reactor_state, shock) -> dict:
        return {
            "reactor_state": reactor_state.model_dump(),
            "shock": {
                "T_reactor": shock.T_reactor,
                "P_reactor": shock.P_reactor,
                "thermo_mix": dict(shock.thermo_mix),
                "u2": shock.u2,
                "rho1": shock.rho1,
                "observable": dict(shock.observable),
            },
        }

    @classmethod
    def init_worker(cls, yaml_text, task_inputs, observable, species_idx, eps):
        from frhodo.simulation.mechanism.mech_fcns import ChemicalMechanism
        from frhodo.simulation.shock.state import RuntimeReactorState
        from types import SimpleNamespace

        gas = ct.Solution(yaml=yaml_text)
        mech = ChemicalMechanism()
        mech.gas = gas
        mech.isLoaded = True
        mech.set_rate_expression_coeffs()
        mech.set_thermo_expression_coeffs()

        cls._mech = mech
        cls._reactor_state = RuntimeReactorState(**task_inputs["reactor_state"])
        cls._shock = SimpleNamespace(**task_inputs["shock"])
        cls._observable = observable
        cls._species_idx = species_idx
        cls._eps = eps

    @classmethod
    def run_one(cls, rxn_idx: int):
        gas = cls._mech.gas
        gas.set_multiplier(1.0 + cls._eps, rxn_idx)
        try:
            SIM = _run_inline(cls._mech, cls._reactor_state, cls._shock)
        finally:
            gas.set_multiplier(1.0, rxn_idx)
        if SIM is None:
            return None
        t = np.asarray(SIM.t_lab(units="SI"), dtype=float)
        obs = _observable_trace(SIM, cls._observable, cls._species_idx)

        return t, obs


# ───────────────────────────── shared bits ─────────────────────────────

def _build_reactor(gas: ct.Solution, reactor_state, shock):
    """Construct a Cantera 0-D reactor for the native sensitivity path.

    Raises:
        ValueError: If ``reactor_state.name`` is not a supported 0-D
            reactor. Incident-shock is handled by the SUNDIALS path
            instead.
    """
    name = reactor_state.name
    if name == "0d Reactor - Constant Volume":
        return ct.IdealGasReactor(gas)
    if name == "0d Reactor - Constant Pressure":
        return ct.IdealGasConstPressureReactor(gas)

    raise ValueError(f"native sensitivity only supports 0-D reactors; got {name!r}")


def _run_inline(mech, reactor_state, shock):
    name = reactor_state.name
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
    if name == "Incident Shock Reactor":
        SIM, details = run_incident_shock(
            mech, reactor_state.t_end,
            shock.T_reactor, shock.P_reactor, shock.thermo_mix, **kwargs,
        )
    elif "0d Reactor" in name:
        kwargs["solve_energy"] = reactor_state.solve_energy
        kwargs["frozen_comp"] = reactor_state.frozen_comp
        mode = zero_d_mode_from_label(name)
        SIM, details = run_zero_d(
            mech, mode, reactor_state.t_end,
            shock.T_reactor, shock.P_reactor, shock.thermo_mix, **kwargs,
        )
    else:
        raise ValueError(f"unknown reactor: {name!r}")
    if SIM is None or not details.get("success", False):
        return None

    return SIM


def _observable_trace(SIM, observable: str, species_idx: int | None) -> np.ndarray:
    if observable == "T":
        return np.asarray(SIM.T(units="SI"), dtype=float)
    if observable == "P":
        return np.asarray(SIM.P(units="SI"), dtype=float)
    if observable == "drhodz_tot":
        return np.asarray(SIM.drhodz_tot(units="SI"), dtype=float)
    if observable in _SPECIES_OBSERVABLES:
        full = np.asarray(getattr(SIM, observable)(units="SI"), dtype=float)
        return full[species_idx, :]

    raise ValueError(f"unknown sensitivity observable: {observable!r}")


def _resample(
    t: np.ndarray, sens: np.ndarray, time_grid: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if time_grid is None:
        return t, sens
    grid = np.asarray(time_grid, dtype=float)
    if t.size < 2:
        out = (
            np.tile(sens[0:1], (grid.size, 1))
            if sens.size else np.zeros((grid.size, sens.shape[1] if sens.ndim > 1 else 0))
        )
        return grid, out
    out = np.empty((grid.size, sens.shape[1]), dtype=float)
    for k in range(sens.shape[1]):
        out[:, k] = np.interp(
            grid, t, sens[:, k],
            left=sens[0, k], right=sens[-1, k],
        )

    return grid, out
