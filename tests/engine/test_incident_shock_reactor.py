"""``IncidentShockReactor._rhs`` — Goldsmith / Speth shock-tube state derivative.

State vector ``[z, A, rho, v, T, t_shock, Y_1..Y_K]``. Tests pin the
analytical zero-rate case (inert gas, no reactions) and the input-validation
guards.
"""
import cantera as ct
import numpy as np
import pytest

from frhodo.simulation.shock.incident_shock_reactor import IncidentShockReactor


@pytest.fixture
def inert_h2o2_state():
    """h2o2 mech in pure-Ar composition; all wdot vanish."""
    gas = ct.Solution("h2o2.yaml")
    gas.TPX = 1500.0, 20000.0, "AR:1.0"
    return gas


@pytest.fixture
def reactor(inert_h2o2_state):
    return IncidentShockReactor(
        inert_h2o2_state,
        rho1=inert_h2o2_state.density, u_reac=1000.0,
        L=0.1, As=0.2, A1=0.2, area_change=False,
    )


def _initial_state(gas, A=0.2, v=1000.0):
    return np.hstack((0.0, A, gas.density, v, gas.T, 0.0, gas.Y))


class TestRhsInert:
    """When wdot=0, only z and t_shock evolve; energy/density/momentum are flat."""

    def test_output_shape(self, reactor, inert_h2o2_state):
        y0 = _initial_state(inert_h2o2_state)
        ydot = reactor._rhs(0.0, y0)
        assert ydot.shape == (inert_h2o2_state.n_species + 6,)

    def test_dA_dt_is_zero_without_area_change(self, reactor, inert_h2o2_state):
        ydot = reactor._rhs(0.0, _initial_state(inert_h2o2_state))
        np.testing.assert_allclose(ydot[1], 0.0, atol=1e-12)

    def test_drho_dt_is_zero_for_frozen_chemistry(self, reactor, inert_h2o2_state):
        ydot = reactor._rhs(0.0, _initial_state(inert_h2o2_state))
        np.testing.assert_allclose(ydot[2], 0.0, atol=1e-12)

    def test_dv_dt_is_zero_for_frozen_chemistry(self, reactor, inert_h2o2_state):
        ydot = reactor._rhs(0.0, _initial_state(inert_h2o2_state))
        np.testing.assert_allclose(ydot[3], 0.0, atol=1e-12)

    def test_dT_dt_is_zero_for_frozen_chemistry(self, reactor, inert_h2o2_state):
        ydot = reactor._rhs(0.0, _initial_state(inert_h2o2_state))
        np.testing.assert_allclose(ydot[4], 0.0, atol=1e-12)

    def test_dY_dt_is_zero_for_frozen_chemistry(self, reactor, inert_h2o2_state):
        ydot = reactor._rhs(0.0, _initial_state(inert_h2o2_state))
        np.testing.assert_allclose(ydot[6:], 0.0, atol=1e-12)

    def test_dz_dt_equals_velocity_at_initial_conditions(self, reactor, inert_h2o2_state):
        v = 1000.0
        y0 = _initial_state(inert_h2o2_state, v=v)
        ydot = reactor._rhs(0.0, y0)
        np.testing.assert_allclose(ydot[0], v, rtol=1e-12)

    def test_dt_shock_dt_is_unity_at_initial_conditions(self, reactor, inert_h2o2_state):
        ydot = reactor._rhs(0.0, _initial_state(inert_h2o2_state))
        np.testing.assert_allclose(ydot[5], 1.0, rtol=1e-12)

    def test_scaling_factor_applied(self, inert_h2o2_state):
        """ydot is scaled by ``rho * A / (rho1 * A1)``. With rho1 set
        to half the post-shock density, all derivatives double."""
        v = 1000.0
        rho_post = inert_h2o2_state.density
        scaled = IncidentShockReactor(
            inert_h2o2_state, rho1=rho_post / 2.0, u_reac=v,
            L=0.1, As=0.2, A1=0.2, area_change=False,
        )
        ydot = scaled._rhs(0.0, _initial_state(inert_h2o2_state, v=v))
        np.testing.assert_allclose(ydot[0], 2.0 * v, rtol=1e-12)
        np.testing.assert_allclose(ydot[5], 2.0, rtol=1e-12)


class TestRhsAreaChange:
    def test_dA_dt_positive_when_area_change_enabled(self, inert_h2o2_state):
        reactor = IncidentShockReactor(
            inert_h2o2_state,
            rho1=inert_h2o2_state.density, u_reac=1000.0,
            L=0.1, As=0.2, A1=0.2, area_change=True,
        )
        y = _initial_state(inert_h2o2_state, v=1000.0)
        y[0] = 0.05  # mid-tube position
        ydot = reactor._rhs(0.0, y)
        assert ydot[1] > 0.0, (
            f"dA/dt should be positive at z=L/2 with v>0; got {ydot[1]}"
        )


class TestRhsInvalidState:
    def test_raises_on_negative_temperature(self, reactor, inert_h2o2_state):
        y = _initial_state(inert_h2o2_state)
        y[4] = -100.0
        with pytest.raises(Exception, match="Temperature is invalid"):
            reactor._rhs(0.0, y)

    def test_raises_on_nan_temperature(self, reactor, inert_h2o2_state):
        y = _initial_state(inert_h2o2_state)
        y[4] = np.nan
        with pytest.raises(Exception, match="Temperature is invalid"):
            reactor._rhs(0.0, y)

    def test_raises_on_negative_density(self, reactor, inert_h2o2_state):
        y = _initial_state(inert_h2o2_state)
        y[2] = -1.0
        with pytest.raises(Exception, match="Density is invalid"):
            reactor._rhs(0.0, y)

    def test_raises_on_nan_density(self, reactor, inert_h2o2_state):
        y = _initial_state(inert_h2o2_state)
        y[2] = np.nan
        with pytest.raises(Exception, match="Density is invalid"):
            reactor._rhs(0.0, y)
