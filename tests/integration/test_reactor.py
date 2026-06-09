"""``run_incident_shock`` end-to-end on Shock1.

Solves zone-2 post-shock state via ``ShockJumpSolver``, runs the reactor to
``t_end``, inspects the resulting ``ReactorOutput``.
"""
import numpy as np
import pytest

from frhodo.simulation.shock.incident_shock_reactor import run_incident_shock
from frhodo.simulation.shock.shock_solver import ShockJumpSolver

# Inputs match example/experiment/shock1.exp.
T1_K = 294.15
P1_PA = 5.01 * 133.322368421
U1_MPS = 120e-3 / 116.557292e-6
MIX = {"Kr": 0.96, "cC7H14": 0.04}


@pytest.fixture(params=["BDF", "CVODES"])
def shock_sim(request, loaded_cycloheptane):
    """Run an incident-shock reactor sim under each ODE backend.

    Parametrized: scipy BDF and SUNDIALS CVODES (via
    :class:`IncidentShockReactor` ``backend="sundials"`` / our ctypes binding).
    """
    props = ShockJumpSolver(
        loaded_cycloheptane.gas,
        {"T1": T1_K, "P1": P1_PA, "u1": U1_MPS, "mix": dict(MIX)},
    )
    assert props.success

    r = props.res
    SIM, details = run_incident_shock(
        loaded_cycloheptane,
        t_end=5.0e-5,
        T_reac=r.T2,
        P_reac=r.P2,
        mix=MIX,
        u_reac=r.u2,
        rho1=r.rho1,
        ODE_solver=request.param,
        observable={"main": "Density Gradient", "sub": 0},
    )
    assert details["success"], f"shock reactor failed: {details.get('message')}"
    return SIM


class TestIncidentShockReactor:
    def test_simulation_result_returned(self, shock_sim):
        assert shock_sim is not None

    def test_simulation_marked_successful(self, shock_sim):
        assert shock_sim.success is True

    def test_independent_var_starts_at_zero(self, shock_sim):
        assert shock_sim.independent_var[0] == pytest.approx(0.0, abs=1e-12)

    def test_independent_var_reaches_t_end(self, shock_sim):
        # independent_var is in seconds (matches t_end input).
        assert shock_sim.independent_var[-1] == pytest.approx(5.0e-5, rel=1e-3), (
            f"sim ended at {shock_sim.independent_var[-1]} (expected 5e-5 s)"
        )

    def test_observable_array_is_finite(self, shock_sim):
        assert np.isfinite(shock_sim.observable).all(), (
            "observable array contains non-finite values; ODE may have diverged"
        )

    def test_observable_changes_over_time(self, shock_sim):
        obs = np.asarray(shock_sim.observable)
        assert obs.std() > 0.0, f"observable is constant: {obs[0]}"


class TestSIMPropertyCaching:
    """``SimProperty.__call__`` lazily computes and caches values per unit system.

    Production reads a property many times during plotting; recomputing
    would either (a) waste cycles, or (b) break aliasing assumptions in
    the plot widgets. Both invariants need explicit locks.
    """

    def test_repeated_call_returns_same_array_object(self, shock_sim):
        """Identity, not just equality: callers rely on object identity
        to avoid stale array references in plot widgets."""
        first = shock_sim.drhodz_tot(units="SI")
        second = shock_sim.drhodz_tot(units="SI")
        assert first is second, "SimProperty recomputed instead of returning cache"

    def test_si_and_cgs_both_populated_after_dual_request(self, shock_sim):
        prop = shock_sim.drhodz_tot
        prop(units="SI")
        prop(units="CGS")

        assert prop.value["SI"].size > 0, "SI cache empty after explicit SI request"
        assert prop.value["CGS"].size > 0, "CGS cache empty after explicit CGS request"

    def test_cgs_value_equals_si_times_conversion(self, shock_sim):
        """drhodz_tot uses conversion factor 1e-5 (g/cm^3/cm <- kg/m^3/m * 1e-5)."""
        si = shock_sim.drhodz_tot(units="SI")
        cgs = shock_sim.drhodz_tot(units="CGS")
        np.testing.assert_allclose(
            cgs, si * 1e-5, rtol=1e-12,
            err_msg="CGS conversion does not match the documented factor",
        )

    def test_unitless_property_has_no_conversion(self, shock_sim):
        """T (temperature) is the same in SI and CGS — its conversion
        is None, so both caches must hold the identical array."""
        si = shock_sim.T(units="SI")
        cgs = shock_sim.T(units="CGS")
        assert si is cgs, "T cache should alias SI to CGS when conversion is None"


class TestIncidentShockReactorTrajectorySnapshot:
    """Shock1 reactor trajectory snapshot at rtol=1e-3."""

    EXPECTED_T_INITIAL = 1616.2904
    EXPECTED_T_FINAL = 1326.0463
    EXPECTED_RHO_INITIAL = 0.13045567
    EXPECTED_RHO_FINAL = 0.15118594
    EXPECTED_OBS_MAX = 9.2855657e-05
    EXPECTED_OBS_FIRST = 2.1216120e-05
    EXPECTED_OBS_LAST = 5.358688750e-07

    def test_initial_temperature(self, shock_sim):
        T = shock_sim.T(units="SI")
        np.testing.assert_allclose(
            T[0], self.EXPECTED_T_INITIAL, rtol=1e-3,
            err_msg=f"T[0] drifted: {T[0]} vs expected {self.EXPECTED_T_INITIAL}",
        )

    def test_final_temperature(self, shock_sim):
        T = shock_sim.T(units="SI")
        np.testing.assert_allclose(
            T[-1], self.EXPECTED_T_FINAL, rtol=1e-3,
            err_msg=f"T[-1] drifted: {T[-1]} vs expected {self.EXPECTED_T_FINAL}",
        )

    def test_initial_density(self, shock_sim):
        rho = shock_sim.rho(units="SI")
        np.testing.assert_allclose(
            rho[0], self.EXPECTED_RHO_INITIAL, rtol=1e-3,
            err_msg=f"rho[0] drifted: {rho[0]} vs {self.EXPECTED_RHO_INITIAL}",
        )

    def test_final_density(self, shock_sim):
        rho = shock_sim.rho(units="SI")
        np.testing.assert_allclose(
            rho[-1], self.EXPECTED_RHO_FINAL, rtol=1e-3,
            err_msg=f"rho[-1] drifted: {rho[-1]} vs {self.EXPECTED_RHO_FINAL}",
        )

    def test_observable_first_value(self, shock_sim):
        obs = np.asarray(shock_sim.observable)
        np.testing.assert_allclose(
            obs[0], self.EXPECTED_OBS_FIRST, rtol=1e-3,
            err_msg=f"observable[0] drift: {obs[0]} vs {self.EXPECTED_OBS_FIRST}",
        )

    def test_observable_last_value(self, shock_sim):
        obs = np.asarray(shock_sim.observable)
        # Tail value is ~6e-7, three orders of magnitude below the peak,
        # so rtol alone is too strict. atol = 0.1% of peak.
        np.testing.assert_allclose(
            obs[-1], self.EXPECTED_OBS_LAST,
            rtol=1e-3, atol=self.EXPECTED_OBS_MAX * 1e-3,
            err_msg=f"observable[-1] drift: {obs[-1]} vs {self.EXPECTED_OBS_LAST}",
        )

    def test_observable_max_matches_snapshot(self, shock_sim):
        obs = np.asarray(shock_sim.observable)
        np.testing.assert_allclose(
            obs.max(), self.EXPECTED_OBS_MAX, rtol=1e-3,
            err_msg=(
                f"peak observable drifted: {obs.max()} vs "
                f"{self.EXPECTED_OBS_MAX}"
            ),
        )
