"""``variables.drhodz`` and ``drhodz_per_rxn`` snapshots and the
``sum(drhodz_per_rxn, axis=1) == drhodz`` algebraic identity.
"""
import cantera as ct
import numpy as np
import pytest

from frhodo.simulation.shock.reactor_output import drhodz, drhodz_per_rxn

# Three post-shock states at fixed P and mixture, with the SolutionArray
# extras (z, A, vel) populated.
TEMPERATURES_K = [1500.0, 2000.0, 2500.0]
PRESSURE_PA = 20_000.0
MIX = {"Kr": 0.96, "cC7H14": 0.04}
EXTRAS = {"z": 0.05, "A": 0.2, "vel": 200.0}

# Snapshot outputs from current code (Cycloheptane mech).
EXPECTED_DRHODZ = [
    0.5637810743389414,
    16.258964930373153,
    25.240999170693865,
]
EXPECTED_DRHODZ_PER_RXN_SHAPE = (3, 66)


@pytest.fixture(scope="module")
def states(loaded_cycloheptane):
    sa = ct.SolutionArray(loaded_cycloheptane.gas, extra=list(EXTRAS.keys()))
    for T in TEMPERATURES_K:
        loaded_cycloheptane.gas.TPX = T, PRESSURE_PA, MIX
        sa.append(loaded_cycloheptane.gas.state, **EXTRAS)
    return sa


class TestDrhodz:
    def test_shape_matches_state_count(self, states):
        result = drhodz(states)
        assert result.shape == (len(TEMPERATURES_K),), (
            f"drhodz returned {result.shape}, expected ({len(TEMPERATURES_K)},)"
        )

    def test_values_are_finite(self, states):
        assert np.isfinite(drhodz(states)).all()

    @pytest.mark.parametrize("idx,expected", list(enumerate(EXPECTED_DRHODZ)))
    def test_value_matches_snapshot(self, states, idx, expected):
        actual = float(drhodz(states)[idx])
        np.testing.assert_allclose(
            actual, expected, rtol=1e-12,
            err_msg=(
                f"drhodz at T={TEMPERATURES_K[idx]}K drifted: "
                f"expected {expected!r}, got {actual!r}"
            ),
        )

class TestDrhodzPerRxn:
    def test_shape_is_states_by_reactions(self, states):
        result = drhodz_per_rxn(states)
        assert result.shape == EXPECTED_DRHODZ_PER_RXN_SHAPE

    def test_values_are_finite(self, states):
        assert np.isfinite(drhodz_per_rxn(states)).all()

    def test_sum_over_reactions_equals_drhodz_total(self, states):
        """``sum(drhodz_per_rxn, axis=1) == drhodz``."""
        per_rxn = drhodz_per_rxn(states)
        total = drhodz(states)
        np.testing.assert_allclose(
            per_rxn.sum(axis=1), total, rtol=1e-12,
            err_msg=(
                "sum(drhodz_per_rxn, axis=1) does not equal drhodz; "
                f"diffs: {(per_rxn.sum(axis=1) - total).tolist()}"
            ),
        )

    def test_rxn_num_subset_matches_full(self, states):
        full = drhodz_per_rxn(states)
        sub = drhodz_per_rxn(states, rxnNum=[0, 1, 2])
        assert sub.shape == (full.shape[0], 3)
        np.testing.assert_allclose(sub, full[:, [0, 1, 2]], rtol=1e-12)


class TestDrhodzAreaChange:
    def test_drhodz_with_area_change_is_finite(self, states):
        assert np.isfinite(drhodz(states, area_change=True)).all()

    def test_drhodz_per_rxn_with_area_change_is_finite(self, states):
        assert np.isfinite(drhodz_per_rxn(states, area_change=True)).all()

    def test_area_change_shifts_drhodz(self, states):
        off = drhodz(states, area_change=False)
        on = drhodz(states, area_change=True)
        assert not np.allclose(off, on), (
            "area_change=True should change drhodz; got identical arrays"
        )


class TestDrhodzAtEquilibrium:
    """At chemical equilibrium, all net rates of progress vanish, so
    ``drhodz_per_rxn`` and ``drhodz`` should both be zero."""

    @pytest.fixture(scope="class")
    def equilibrium_states(self, loaded_cycloheptane):
        gas = loaded_cycloheptane.gas
        sa = ct.SolutionArray(gas, extra=list(EXTRAS.keys()))
        for T in [1500.0, 2000.0]:
            gas.TPX = T, PRESSURE_PA, MIX
            gas.equilibrate("TP")
            sa.append(gas.state, **EXTRAS)
        return sa

    def test_drhodz_at_equilibrium_is_zero(self, equilibrium_states):
        np.testing.assert_allclose(
            drhodz(equilibrium_states), 0.0, atol=1e-8,
            err_msg="drhodz must vanish at chemical equilibrium",
        )

    def test_drhodz_per_rxn_at_equilibrium_is_zero(self, equilibrium_states):
        np.testing.assert_allclose(
            drhodz_per_rxn(equilibrium_states), 0.0, atol=1e-8,
            err_msg="every per-reaction contribution must vanish at equilibrium",
        )

