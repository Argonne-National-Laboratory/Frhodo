"""SUNDIALS-driven sensitivity analysis for the incident-shock reactor.

Both adjoint and forward-sensitivity paths share the same contract:
build the reactor + integrator from the shock state, call
:mod:`frhodo.simulation.shock.observables` for the observable's
terminal/projection algebra, and drive ``CVodeAdjInit`` /
``CVodeSensInit`` via :mod:`frhodo.simulation.numerics.sundials`.

Two backends layered over one shared setup:

* :func:`compute_adjoint_sensitivity` — per-output backward solve
  with inline quadrature. Cost scales with ``M`` (output times),
  independent of ``N_rxns``. Best when ``M`` is small.
* :func:`compute_forward_sensitivity` — one augmented forward solve
  via ``CVodeSensInit``. Cost scales with ``N_rxns``, independent of
  ``M``. Best when ``M`` is large.

The shared :func:`_setup_sundials_sens` helper performs the validate
+ gas-save + reactor-build + output-grid construction that both
paths need; the public functions are thin wrappers around the
backend-specific SUNDIALS call sequence.
"""
from __future__ import annotations

from typing import Literal

import cantera as ct
import numpy as np

from frhodo.simulation.numerics.sundials import (
    AdjointProblem, CV_STAGGERED, ForwardSensProblem,
)
from frhodo.simulation.shock.incident_shock_reactor import (
    I_RHO, I_T, I_TS, I_V, I_Z, N_SHOCK,
    IncidentShockReactor,
    _Geometry,
    _shock_jacobian,
    _shock_param_rhs_gradient,
)
from frhodo.simulation.shock.observables import OBSERVABLES, terminal


Observable = Literal["drhodz_tot", "T", "P", "Y", "X", "conc", "HRR_tot"]


def _sens_atol_vector(K: int, atol_T: float, atol_species: float) -> np.ndarray:
    """Per-component absolute tolerance for sens/adjoint state-space vectors.

    Same shape as the forward state (``[z, A, ρ, v, T, t_shock, Y...]``);
    used for the adjoint variable ``λ`` and for each forward-sens
    column ``S_k``.
    """
    atol = np.empty(N_SHOCK + K)
    atol[:N_SHOCK] = 1e-9
    atol[I_T] = atol_T
    atol[N_SHOCK:] = atol_species

    return atol


def _setup_sundials_sens(mech, reactor_state, shock, observable, species_idx):
    """Validate inputs + build the SUNDIALS reactor + cache shared bits.

    Caller is responsible for restoring ``gas.TPX = saved_TPX`` in a
    ``finally`` block.

    Returns:
        ``(gas, n_rxns, Wk, saved_TPX, geometry, reactor)``.

    Raises:
        ValueError: ``reactor_state`` names a non-incident-shock reactor,
            or a per-species observable is requested without ``species_idx``.
        NotImplementedError: ``observable`` is not in :data:`OBSERVABLES`.
    """
    if reactor_state.name != "Incident Shock Reactor":
        raise ValueError(
            f"SUNDIALS sensitivity only supports the incident-shock reactor, "
            f"got reactor_state.name={reactor_state.name!r}"
        )
    if observable not in OBSERVABLES:
        raise NotImplementedError(
            f"sensitivity does not support observable={observable!r}; "
            f"valid: {sorted(OBSERVABLES)}"
        )
    if OBSERVABLES[observable].requires_species_idx and species_idx is None:
        raise ValueError(f"observable={observable!r} requires species_idx")

    gas = mech.gas
    n_rxns = gas.n_reactions
    Wk = np.asarray(gas.molecular_weights, dtype=float).copy()
    saved_TPX = gas.TPX

    geometry = _Geometry(
        rho1=float(shock.rho1),
        A1=float(getattr(shock, "A1", 0.2)),
        L=float(getattr(shock, "L", 0.1)),
        As=float(getattr(shock, "As", 0.2)),
        area_change=bool(getattr(shock, "area_change", False)),
    )
    gas.TPX = shock.T_reactor, shock.P_reactor, shock.thermo_mix

    reactor = IncidentShockReactor(
        gas,
        rho1=geometry.rho1, u_reac=float(shock.u2),
        A1=geometry.A1, As=geometry.As, L=geometry.L,
        area_change=geometry.area_change,
        backend="sundials",
        rtol=reactor_state.ode_rtol,
        atol=reactor_state.ode_atol,
    )

    return gas, n_rxns, Wk, saved_TPX, geometry, reactor


def _build_output_grid(reactor_state, time_grid):
    """Default to a 50-point linear grid covering 1%-99% of ``t_end``.

    Returns:
        Output grid as a 1-D ``np.ndarray``. Caller-supplied
        ``time_grid`` is copied so downstream sorts don't mutate the
        original.
    """
    if time_grid is None:
        return np.linspace(
            float(reactor_state.t_end) * 0.01,
            float(reactor_state.t_end) * 0.99,
            50,
        )

    return np.asarray(time_grid, dtype=float).copy()


# ──────────────────────────── adjoint backend ─────────────────────────

def compute_adjoint_sensitivity(
    mech,
    reactor_state,
    shock,
    observable: Observable,
    species_idx: int | None = None,
    time_grid: np.ndarray | None = None,
    rtol: float = 1e-6,
    atol_T: float = 1e-6,
    atol_species: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Adjoint sensitivity of ``observable`` to each reaction rate constant.

    Args:
        rtol: Backward-solve relative tolerance. Default 1e-6 (not
            ``reactor_state.ode_rtol``) because Cantera's Linux build
            does not export ``CVodeSetMaxNumStepsB``, so the backward
            solve is capped at SUNDIALS' default 500 steps per output
            time; tightening ``rtol`` past 1e-6 there causes the
            backward integrator to hit the cap on stiff problems. Use
            :func:`compute_forward_sensitivity` for stiff problems
            where adjoint ``rtol`` can't be relaxed.
        atol_T: Absolute tolerance applied to the temperature component
            of the adjoint state.
        atol_species: Absolute tolerance applied to the species
            components of the adjoint state and to the quadrature.

    Returns:
        ``(t, sens)`` with ``sens.shape == (len(t), n_rxns)`` and
        Cantera-normalized entries
        ``sens[m, j] = ∂ln(observable[m])/∂ln(k_j)``.

    Raises:
        ValueError: From :func:`_setup_sundials_sens` for unsupported
            reactor types or missing ``species_idx``.
        NotImplementedError: From :func:`_setup_sundials_sens` for
            unrecognized observables.
    """
    try:
        gas, n_rxns, Wk, saved_TPX, geometry, reactor = _setup_sundials_sens(
            mech, reactor_state, shock, observable, species_idx,
        )
    except (ValueError, NotImplementedError):
        raise
    except Exception:
        return np.zeros(0), np.zeros((0, mech.gas.n_reactions))

    try:
        atol_vec = _sens_atol_vector(Wk.size, atol_T, atol_species)

        # SUNDIALS often calls rhsB and jacB at the same t (BDF reuses
        # the Jacobian between Newton iterations on the same step); memoize
        # ``_shock_jacobian`` on (t, id(y)) within a single backward sweep
        # to avoid redundant Cantera derivative work.
        jac_cache: dict = {"key": None, "J": None}

        def _jac_at(t, y):
            key = (t, y.ctypes.data)
            if jac_cache["key"] == key:
                return jac_cache["J"]
            J = _shock_jacobian(gas, y, geometry, Wk)
            jac_cache["key"] = key
            jac_cache["J"] = J

            return J

        def rhsB(t, y, lam, lamdot):
            J = _jac_at(t, y)
            lamdot[:] = -(J.T @ lam)

        def jacB(t, y, lam, flam, JB):
            J = _jac_at(t, y)
            JB[:] = -J.T

        def quad_rhsB(t, y, lam, qdot):
            dgdp = _shock_param_rhs_gradient(gas, y, geometry, Wk)
            qdot[:] = lam @ dgdp

        adj = AdjointProblem(
            reactor._integ,
            rhsB=rhsB, jacB=jacB, quad_rhsB=quad_rhsB,
            n_quadrature=n_rxns, n_checkpoints=150,
            rtolB=rtol, atolB=atol_vec,
            rtolQB=rtol, atolQB=atol_species,
            # Raise the backward-solve step cap when the setter is
            # available (Cantera Windows). On Linux the symbol is
            # absent and SUNDIALS' default 500 applies.
            max_backward_steps=50_000,
        )

        output_t = _build_output_grid(reactor_state, time_grid)
        positive_t = sorted({float(t) for t in output_t if t > 0.0})
        try:
            forward_y = adj.run_forward(positive_t)
        except Exception:
            return np.zeros(0), np.zeros((0, n_rxns))

        sens = np.zeros((output_t.size, n_rxns), dtype=float)
        # CVODES adjoint is most efficient when backward sweeps walk t_m
        # in monotonically decreasing order — the checkpoint scheme is
        # optimized for that traversal.
        for idx in np.argsort(output_t)[::-1]:
            t_m = float(output_t[idx])
            y_m = reactor.initial_state if t_m <= 0.0 else forward_y[t_m]
            g_m, direct, dg_dy = terminal(
                observable, gas, y_m, geometry, Wk, n_rxns, species_idx,
            )
            if dg_dy is None:
                continue
            if t_m <= 0.0:
                if direct is not None:
                    sens[idx] = direct / g_m
                continue

            _, quadrature = adj.solve_backward(t_m, dg_dy)
            sens_unnorm = -quadrature
            if direct is not None:
                sens_unnorm = sens_unnorm + direct
            sens[idx] = sens_unnorm / g_m

    finally:
        gas.TPX = saved_TPX

    return output_t, sens


# ────────────────────── forward sensitivity backend ────────────────────

def compute_forward_sensitivity(
    mech,
    reactor_state,
    shock,
    observable: Observable,
    species_idx: int | None = None,
    time_grid: np.ndarray | None = None,
    rtol: float = 1e-6,
    atol_T: float = 1e-6,
    atol_species: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward sensitivity of ``observable`` via ``CVodeSensInit``.

    Args:
        rtol: Sensitivity-solve relative tolerance. For very large
            mechanisms (Aramco-scale, ~3000 reactions) the sensitivity
            accuracy is bounded by the **state** step size, not the
            sens tolerance — in ``CV_STAGGERED`` mode the sens solve
            inherits the state's step size and tightening ``rtol``
            here only adds Newton iterations per step. Tighten
            ``reactor_state.ode_rtol`` (e.g. to ``1e-10``) to improve
            sens accuracy on stiff large-mech problems.
        atol_T: Absolute tolerance applied to the temperature component
            of each sensitivity column.
        atol_species: Absolute tolerance applied to the species
            components of each sensitivity column.

    Returns:
        ``(t, sens)`` with ``sens.shape == (len(t), n_rxns)`` and
        Cantera-normalized entries
        ``sens[m, j] = ∂ln(observable[m])/∂ln(k_j)``.

    Raises:
        ValueError: From :func:`_setup_sundials_sens` for unsupported
            reactor types or missing ``species_idx``.
        NotImplementedError: From :func:`_setup_sundials_sens` for
            unrecognized observables.
    """
    try:
        gas, n_rxns, Wk, saved_TPX, geometry, reactor = _setup_sundials_sens(
            mech, reactor_state, shock, observable, species_idx,
        )
    except (ValueError, NotImplementedError):
        raise
    except Exception:
        return np.zeros(0), np.zeros((0, mech.gas.n_reactions))

    try:
        atol_vec = _sens_atol_vector(Wk.size, atol_T, atol_species)

        def sens_rhs(Ns, t, y, ydot, yS, ySdot):
            J = _shock_jacobian(gas, y, geometry, Wk)
            dfdp = _shock_param_rhs_gradient(gas, y, geometry, Wk)
            for k in range(Ns):
                ySdot[k][:] = J @ yS[k] + dfdp[:, k]

        fs = ForwardSensProblem(
            reactor._integ,
            n_params=n_rxns,
            sens_rhs=sens_rhs,
            ism=CV_STAGGERED,
            rtol=rtol,
            atol=atol_vec,
        )

        output_t = _build_output_grid(reactor_state, time_grid)
        positive_t = sorted({float(t) for t in output_t if t > 0.0})
        try:
            sens_states = fs.run_forward(positive_t)
        except Exception:
            return np.zeros(0), np.zeros((0, n_rxns))

        sens = np.zeros((output_t.size, n_rxns), dtype=float)
        for idx in range(output_t.size):
            t_m = float(output_t[idx])
            if t_m <= 0.0:
                y_m = reactor.initial_state
                S_m = np.zeros((y_m.size, n_rxns))
            else:
                y_m, S_m = sens_states[t_m]

            g_m, direct, dg_dy = terminal(
                observable, gas, y_m, geometry, Wk, n_rxns, species_idx,
            )
            if dg_dy is None:
                continue

            # Chain rule: ∂g/∂p_j = Σ_i ∂g/∂y_i · S[i, j]
            indirect = dg_dy @ S_m
            sens_unnorm = indirect
            if direct is not None:
                sens_unnorm = sens_unnorm + direct
            sens[idx] = sens_unnorm / g_m

    finally:
        gas.TPX = saved_TPX

    return output_t, sens
