"""``core.reactors.check_rxn_rates`` rate-magnitude diagnostic.

Returns 1-based indices of reactions whose forward or reverse rate
constant exceeds the per-molecularity limit (``[1e9, 1e15, 1e21]`` for
uni-/bi-/termolecular). Three-body reactions get their effective
molecularity bumped by 1 before the lookup.
"""
import cantera as ct
import pytest

from frhodo.simulation.mechanism.mech_fcns import check_rxn_rates


@pytest.fixture
def h2o2_at_shock_conditions():
    gas = ct.Solution("h2o2.yaml")
    gas.TPX = 1500.0, 20000.0, "H2:0.04, O2:0.02, AR:0.94"
    return gas


@pytest.fixture
def h2o2_with_inflated_first_rate():
    base = ct.Solution("h2o2.yaml")
    rxns = list(base.reactions())
    r0 = rxns[0]
    rxns[0] = ct.Reaction(
        reactants=r0.reactants, products=r0.products,
        rate=ct.ArrheniusRate(1.0e20, 0.0, 0.0),
    )
    gas = ct.Solution(thermo="ideal-gas", kinetics="gas",
                      species=base.species(), reactions=rxns)
    gas.TPX = 1500.0, 20000.0, "H2:0.04, O2:0.02, AR:0.94"
    return gas


class TestCheckRxnRates:
    def test_normal_h2o2_flags_nothing(self, h2o2_at_shock_conditions):
        flagged = check_rxn_rates(h2o2_at_shock_conditions)
        assert flagged == [], (
            f"normal h2o2 rates are below diagnostic limits; got {flagged}"
        )

    def test_inflated_first_rate_is_flagged(self, h2o2_with_inflated_first_rate):
        flagged = check_rxn_rates(h2o2_with_inflated_first_rate)
        assert 1 in flagged, (
            f"reaction 0 with A=1e20 should be flagged (1-based); got {flagged}"
        )

    def test_returns_one_based_indices(self, h2o2_with_inflated_first_rate):
        flagged = check_rxn_rates(h2o2_with_inflated_first_rate)
        assert min(flagged) >= 1, (
            f"indices must be 1-based; got minimum {min(flagged)}"
        )

    def test_handles_three_body_reactions(self, h2o2_at_shock_conditions):
        """h2o2 contains three-body reactions like H+H(+M)<=>H2(+M); the
        function must traverse them without raising."""
        types = {
            r.reaction_type for r in h2o2_at_shock_conditions.reactions()
        }
        assert any("three-body" in t for t in types), (
            f"fixture must include a three-body reaction; got types {types}"
        )
        check_rxn_rates(h2o2_at_shock_conditions)
