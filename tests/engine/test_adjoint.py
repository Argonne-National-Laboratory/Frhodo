"""Adjoint sensitivity analysis — component checks and end-to-end vs native.

Component-level tests isolate failure modes that an end-to-end
agreement-with-native test would conflate (Jacobian algebra, backward
solver, quadrature). The end-to-end test on cycloheptane validates the
assembled pipeline against the existing forward-sensitivity oracle.
"""
import pathlib
from types import SimpleNamespace

import cantera as ct
import numpy as np
import pytest
from scipy.integrate import simpson

from frhodo.simulation.numerics.sundials import AdjointProblem, CVodeIntegrator
from frhodo.simulation.shock.incident_shock_reactor import (
    I_A, I_RHO, I_T, I_V, I_Z, N_SHOCK,
    IncidentShockReactor, _Geometry, _shock_derivatives, _shock_jacobian,
    _shock_param_rhs_gradient,
)
from frhodo.simulation.shock.sensitivity import compute_sensitivity
from frhodo.simulation.shock.shock_solver import ShockJumpSolver
from frhodo.simulation.shock.state import RuntimeReactorState


PRE_SHOCK = {
    "T1": 294.15,
    "P1": 5.01 * 133.322368421,
    "u1": 120e-3 / 116.557292e-6,
    "mix": {"Kr": 0.96, "cC7H14": 0.04},
}


@pytest.fixture(scope="module")
def cyc7_shock(loaded_cycloheptane):
    """``compute_sensitivity``-shaped post-shock descriptor."""
    props = ShockJumpSolver(loaded_cycloheptane.gas, PRE_SHOCK)
    assert props.success
    r = props.res

    return SimpleNamespace(
        T_reactor=r.T2,
        P_reactor=r.P2,
        thermo_mix=dict(PRE_SHOCK["mix"]),
        u2=r.u2,
        rho1=r.rho1,
        A1=0.2,
        As=0.2,
        L=0.1,
        area_change=False,
        observable={"main": "Density Gradient", "sub": 0},
    )


@pytest.fixture(scope="module")
def cyc7_reactor_state():
    return RuntimeReactorState(
        name="Incident Shock Reactor",
        t_end=5.0e-6,
        ode_solver="BDF",
        ode_rtol=1e-8,
        ode_atol=1e-11,
    )


@pytest.fixture(scope="module")
def cyc7_state_vector(loaded_cycloheptane, cyc7_shock):
    """Mid-trajectory state vector where chemistry has populated most species.

    The initial post-shock state has many zero mass fractions; FD checks
    on those columns are ill-conditioned because the perturbed value
    crosses zero. Stepping the reactor forward gives a state with all
    species at non-trivial mass fractions.
    """
    gas = loaded_cycloheptane.gas
    gas.TPX = cyc7_shock.T_reactor, cyc7_shock.P_reactor, cyc7_shock.thermo_mix
    reactor = IncidentShockReactor(
        gas,
        rho1=cyc7_shock.rho1, u_reac=cyc7_shock.u2,
        A1=cyc7_shock.A1, As=cyc7_shock.As, L=cyc7_shock.L,
        area_change=cyc7_shock.area_change,
        backend="scipy", method="BDF", rtol=1e-8, atol=1e-11,
    )
    reactor.advance(3.0e-6)

    return reactor.dense_eval(np.array([3.0e-6]))[0]


@pytest.fixture
def cyc7_geometry(cyc7_shock):
    return _Geometry(
        rho1=cyc7_shock.rho1,
        A1=cyc7_shock.A1,
        L=cyc7_shock.L,
        As=cyc7_shock.As,
        area_change=cyc7_shock.area_change,
    )


def _f(gas, y, Wk, geometry):
    """Wrap ``_shock_derivatives`` to take a state vector and a geometry."""
    z, A, rho, v, T = y[I_Z], y[I_A], y[I_RHO], y[I_V], y[I_T]
    gas.set_unnormalized_mass_fractions(y[N_SHOCK:])
    gas.TD = float(T), float(rho)

    return _shock_derivatives(
        gas, float(z), float(A), float(v), Wk,
        rho1=geometry.rho1, A1=geometry.A1, L=geometry.L, As=geometry.As,
        area_change=geometry.area_change,
    )


class TestShockJacobian:
    """``_shock_jacobian`` matches central finite differences of ``_shock_derivatives``."""

    def test_jacobian_matches_fd_column_by_column(
        self, loaded_cycloheptane, cyc7_state_vector, cyc7_geometry,
    ):
        """Each well-conditioned column of J matches central FD on the RHS."""
        gas = loaded_cycloheptane.gas
        y = cyc7_state_vector
        Wk = np.asarray(gas.molecular_weights, dtype=float).copy()

        J = _shock_jacobian(gas, y, cyc7_geometry, Wk)

        scales = np.maximum(np.abs(y), 1.0)
        worst = 0.0
        worst_col = -1
        for k in (I_Z, I_A, I_RHO, I_V, I_T) + tuple(range(N_SHOCK, y.size)):
            if k >= N_SHOCK and abs(y[k]) < 1e-15:
                continue
            eps = 1e-7 * scales[k]
            y_plus = y.copy(); y_plus[k] += eps
            y_minus = y.copy(); y_minus[k] -= eps
            fd_col = (_f(gas, y_plus, Wk, cyc7_geometry)
                      - _f(gas, y_minus, Wk, cyc7_geometry)) / (2.0 * eps)
            j_norm = float(np.linalg.norm(J[:, k]))
            if j_norm < 1e-30:
                continue
            ref = np.maximum(np.abs(J[:, k]), j_norm * 1e-12)
            rel_err = float(np.max(np.abs(J[:, k] - fd_col) / ref))
            if rel_err > worst:
                worst = rel_err
                worst_col = k

        assert worst < 1e-2, (
            f"J disagrees with central FD at column {worst_col}: rel err {worst:.2e}"
        )



class TestBackwardSolve:
    """``AdjointProblem`` on a constant-``J`` toy ODE matches the matrix exponential."""

    def test_constant_jacobian_matches_matrix_exponential(self):
        """For ``dy/dt = A·y`` with constant ``A``, the adjoint backward solve
        of ``dλ/dt = −Aᵀ·λ`` from ``t_m`` to 0 yields ``λ(0) = exp(Aᵀ·t_m)·λ_T``.
        """
        from scipy.linalg import expm
        rng = np.random.default_rng(1)
        n = 4
        A = rng.standard_normal((n, n)) * 0.5
        t_m = 1.0
        lam_T = rng.standard_normal(n)
        y0 = rng.standard_normal(n)

        def rhs(t, y, ydot):
            ydot[:] = A @ y

        def jac(t, y, fy, J):
            J[:] = A

        def rhsB(t, y, lam, lamdot):
            lamdot[:] = -(A.T @ lam)

        def jacB(t, y, lam, flam, JB):
            JB[:] = -A.T

        def quad_rhsB(t, y, lam, qdot):
            qdot[:] = 0.0  # no parameter — quadrature is unused but required for shape

        integ = CVodeIntegrator(n, rhs, jac=jac, rtol=1e-10, atol=1e-13)
        integ.reinit(0.0, y0)
        adj = AdjointProblem(
            integ, rhsB=rhsB, jacB=jacB, quad_rhsB=quad_rhsB,
            n_quadrature=1, n_checkpoints=50,
            rtolB=1e-10, atolB=1e-13, rtolQB=1e-10, atolQB=1e-13,
        )
        adj.run_forward([t_m])
        lam_at_0, _ = adj.solve_backward(t_m, lam_T)

        expected = expm(A.T * t_m) @ lam_T
        rel = float(
            np.max(np.abs(lam_at_0 - expected) / np.maximum(np.abs(expected), 1e-10))
        )
        assert rel < 1e-6, f"backward solve at t=0: rel err {rel:.2e}"


class TestQuadrature:
    """Simpson integral over the assembled ``λᵀ·∂f/∂p`` integrand matches closed form."""

    def test_simpson_of_sin_cos_product_matches_closed_form(self):
        omega = 7.0
        T = 1.0
        t_grid = np.linspace(0.0, T, 4001)
        lam_vals = np.sin(omega * t_grid)
        dfdp_vals = np.cos(omega * t_grid)
        integrand = lam_vals * dfdp_vals

        result = simpson(integrand, x=t_grid)
        closed_form = np.sin(omega * T) ** 2 / (2.0 * omega)

        assert abs(result - closed_form) < 1e-6, (
            f"Simpson got {result}, expected {closed_form}"
        )


class TestParameterGradient:
    """``∂f/∂p_j`` matches finite differences via ``gas.set_multiplier``."""

    def test_dfdp_matches_fd_on_each_reaction(
        self, loaded_cycloheptane, cyc7_state_vector, cyc7_geometry,
    ):
        gas = loaded_cycloheptane.gas
        y = cyc7_state_vector
        Wk = np.asarray(gas.molecular_weights, dtype=float).copy()
        n_rxns = gas.n_reactions

        dfdp = _shock_param_rhs_gradient(gas, y, cyc7_geometry, Wk)
        eps = 1e-4
        worst = 0.0
        try:
            for j in range(n_rxns):
                gas.set_multiplier(1.0 + eps, j)
                try:
                    f_plus = _f(gas, y, Wk, cyc7_geometry)
                finally:
                    gas.set_multiplier(1.0, j)
                gas.set_multiplier(1.0 - eps, j)
                try:
                    f_minus = _f(gas, y, Wk, cyc7_geometry)
                finally:
                    gas.set_multiplier(1.0, j)
                fd_col = (f_plus - f_minus) / (2.0 * eps)
                scale = max(float(np.max(np.abs(dfdp[:, j]))),
                            float(np.max(np.abs(fd_col))))
                if scale < 1e-10:
                    continue
                rel_err = float(np.max(np.abs(dfdp[:, j] - fd_col)) / scale)
                worst = max(worst, rel_err)
        finally:
            gas.set_multiplier(1.0)

        assert worst < 1e-3, (
            f"∂f/∂p disagrees with central FD by max rel err {worst:.2e}"
        )


_OBSERVABLES = [
    ("T",          None,        1e-4),
    ("P",          None,        1e-4),
    ("Y",          "cC7H14",    1e-3),
    ("X",          "cC7H14",    1e-3),
    ("conc",       "cC7H14",    1e-3),
    # drhodz_tot terminal condition involves the analytical shock
    # Jacobian — its agreement floor is wider than the state observables.
    ("drhodz_tot", None,        1e-3),
]


@pytest.fixture(scope="module")
def cyc7_forward_sens(loaded_cycloheptane, cyc7_reactor_state, cyc7_shock):
    """Forward sensitivities for every observable, computed once per module."""
    time_grid = np.linspace(5e-7, cyc7_reactor_state.t_end * 0.95, 25)
    out: dict[tuple[str, str | None], tuple[np.ndarray, np.ndarray]] = {}
    for obs, species, _ in _OBSERVABLES:
        species_idx = (
            loaded_cycloheptane.gas.species_index(species)
            if species is not None else None
        )
        t_f, sens_f = compute_sensitivity(
            loaded_cycloheptane, cyc7_reactor_state, cyc7_shock,
            observable=obs, species_idx=species_idx,
            time_grid=time_grid, method="forward_sens",
        )
        out[(obs, species)] = (t_f, sens_f)

    return time_grid, out


@pytest.mark.parametrize("obs,species,rtol_bound", _OBSERVABLES)
def test_adjoint_matches_forward_sens(
    obs, species, rtol_bound,
    loaded_cycloheptane, cyc7_reactor_state, cyc7_shock, cyc7_forward_sens,
):
    """Adjoint and forward-sensitivity solve the same continuous problem
    by independent algorithms; their results must agree. Both build on
    ``_shock_jacobian`` and ``_shock_param_rhs_gradient`` (each
    independently FD-validated elsewhere in this module), so an
    agreement check here pins down the assembly logic — adjoint
    backward solve + quadrature versus CVodeSensInit forward
    augmentation.
    """
    time_grid, forward = cyc7_forward_sens
    t_f, sens_f = forward[(obs, species)]

    species_idx = (
        loaded_cycloheptane.gas.species_index(species)
        if species is not None else None
    )
    t_a, sens_a = compute_sensitivity(
        loaded_cycloheptane, cyc7_reactor_state, cyc7_shock,
        observable=obs, species_idx=species_idx,
        time_grid=time_grid, method="adjoint",
    )
    np.testing.assert_allclose(t_a, t_f, rtol=0.0, atol=1e-12)
    assert sens_a.shape == sens_f.shape

    scale = np.maximum(np.abs(sens_f), 1.0)
    rel = np.abs(sens_a - sens_f) / scale
    max_rel = float(np.nanmax(rel))
    assert max_rel < rtol_bound, (
        f"adjoint vs forward_sens on observable={obs!r} species={species!r}: "
        f"max rel error {max_rel:.2e} exceeded bound {rtol_bound:.0e}"
    )


@pytest.fixture(scope="module")
def gri30_mech(tmp_path_factory):
    """GRI 3.0 mechanism via Cantera's bundled YAML."""
    import shutil
    import cantera
    from frhodo.simulation.mechanism.mechanism_loader import MechanismLoader

    src = pathlib.Path(cantera.__file__).parent / "data" / "gri30.yaml"
    dst = tmp_path_factory.mktemp("gri30") / "gri30.yaml"
    shutil.copy(src, dst)

    return MechanismLoader(silent=True).load({"mech": dst, "Cantera_Mech": dst})


@pytest.fixture(scope="module")
def gri30_setup(gri30_mech):
    """Post-shock state + reactor state for GRI 3.0 pre-ignition window."""
    pre = {
        "T1": 800.0,
        "P1": 0.1 * 101325,
        "u1": 1200.0,
        "mix": {"CH4": 0.05, "O2": 0.10, "N2": 0.85},
    }
    sj = ShockJumpSolver(gri30_mech.gas, pre)
    assert sj.success
    r = sj.res
    shock = SimpleNamespace(
        T_reactor=r.T2, P_reactor=r.P2, thermo_mix=dict(pre["mix"]),
        u2=r.u2, rho1=r.rho1, A1=0.2, As=0.2, L=0.1, area_change=False,
        observable={"main": "Density Gradient", "sub": 0},
    )
    # Pre-ignition window — ignition for CH4/O2 at T2 ~ 1300 K lands
    # past ~150 microseconds; we stay below ~50 microseconds.
    reactor_state = RuntimeReactorState(
        name="Incident Shock Reactor",
        t_end=5.0e-5,
        ode_solver="BDF",
        ode_rtol=1e-8,
        ode_atol=1e-11,
    )

    return shock, reactor_state


def test_forward_sens_runs_on_gri30(gri30_mech, gri30_setup):
    """``forward_sens`` runs cleanly on GRI 3.0 (325 reactions).

    Smoke test: the larger Jacobian and chain-rule structure of GRI 3.0
    push more code paths than cycloheptane (the cyc7 cross-check
    against adjoint already pins correctness). This test confirms the
    pipeline still completes and produces sane outputs at scale.
    """
    shock, reactor_state = gri30_setup
    time_grid = np.linspace(1e-5, reactor_state.t_end * 0.95, 10)

    t_f, sens_f = compute_sensitivity(
        gri30_mech, reactor_state, shock,
        observable="drhodz_tot", time_grid=time_grid, method="forward_sens",
    )
    n_rxns = gri30_mech.gas.n_reactions
    assert sens_f.shape == (time_grid.size, n_rxns)
    assert np.all(np.isfinite(sens_f))
    # Pre-ignition sensitivities are non-trivial — at least one reaction
    # carries a Cantera-normalized magnitude well above floor noise.
    assert np.max(np.abs(sens_f)) > 1e-3


@pytest.mark.skip(
    reason=(
        "Cantera's bundled SUNDIALS does not export CVodeSetMaxNumStepsB; "
        "the backward solve is capped at SUNDIALS' default 500 steps per "
        "output time, which is insufficient for stiff backward integration "
        "on GRI 3.0. forward_sens covers this case."
    ),
)
def test_adjoint_runs_on_gri30(gri30_mech, gri30_setup):
    """Tracked here so the limitation is visible. Re-enable once a future
    Cantera release exports the backward integrator setters."""
    shock, reactor_state = gri30_setup
    time_grid = np.linspace(1e-5, reactor_state.t_end * 0.95, 10)
    _, sens_f = compute_sensitivity(
        gri30_mech, reactor_state, shock,
        observable="drhodz_tot", time_grid=time_grid, method="forward_sens",
    )
    _, sens_a = compute_sensitivity(
        gri30_mech, reactor_state, shock,
        observable="drhodz_tot", time_grid=time_grid, method="adjoint",
    )
    scale = np.maximum(np.abs(sens_f), 1.0)
    max_rel = float(np.nanmax(np.abs(sens_a - sens_f) / scale))
    assert max_rel < 1e-2
