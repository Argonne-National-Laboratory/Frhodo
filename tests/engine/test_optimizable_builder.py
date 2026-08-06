"""``OptimizableSetBuilder`` toggle + ``build`` contract.

The builder is the GUI's source of truth for which reactions and
coefficients are marked optimizable. Mech-tree widgets toggle entries
here; the orchestrator calls ``build(mech)`` to get an immutable
``OptimizableSet`` for the optimizer.
"""
import cantera as ct
import pytest

from frhodo.optimize.parameters import (
    OptimizableSet,
    OptimizableSetBuilder,
)



@pytest.fixture
def builder():
    return OptimizableSetBuilder()


@pytest.fixture
def mech_with_one_arrhenius_rxn(loaded_cycloheptane):
    """Cycloheptane mech, no opt flags toggled."""
    return loaded_cycloheptane


@pytest.fixture
def arrh_idx(loaded_cycloheptane):
    """Index of the first ArrheniusRate rxn in Cycloheptane (rxn 0/1/7
    are PlogRate; rxn 2 is the first Arrhenius)."""
    for i, rxn in enumerate(loaded_cycloheptane.gas.reactions()):
        if type(rxn.rate) is ct.ArrheniusRate:
            return i
    raise AssertionError("Cycloheptane should contain at least one Arrhenius rxn")


@pytest.fixture
def plog_idx(loaded_cycloheptane):
    """Index of the first PlogRate rxn in Cycloheptane (rxn 0)."""
    for i, rxn in enumerate(loaded_cycloheptane.gas.reactions()):
        if type(rxn.rate) is ct.PlogRate:
            return i
    raise AssertionError("Cycloheptane should contain at least one Plog rxn")


class TestReactionToggle:
    def test_default_no_reactions_optimizable(self, builder):
        assert builder.is_reaction_optimizable(0) is False

    def test_set_reaction_optimizable(self, builder):
        builder.set_reaction_optimizable(0, True)
        assert builder.is_reaction_optimizable(0) is True

    def test_unset_reaction_optimizable(self, builder):
        builder.set_reaction_optimizable(5, True)
        builder.set_reaction_optimizable(5, False)
        assert builder.is_reaction_optimizable(5) is False


class TestCoefficientBoundState:
    def test_default_no_coefficients_optimizable(self, builder):
        assert builder.is_coefficient_optimizable(0, "rate", "A") is False

    def test_set_coefficient_optimizable(self, builder):
        builder.set_coefficient_optimizable(0, "rate", "A", True)
        assert builder.is_coefficient_optimizable(0, "rate", "A") is True

    def test_unset_coefficient_optimizable(self, builder):
        builder.set_coefficient_optimizable(0, "rate", "A", True)
        builder.set_coefficient_optimizable(0, "rate", "A", False)
        assert builder.is_coefficient_optimizable(0, "rate", "A") is False


class TestBuild:
    def test_empty_builder_yields_empty_set(self, builder, mech_with_one_arrhenius_rxn):
        result = builder.build(mech_with_one_arrhenius_rxn)
        assert isinstance(result, OptimizableSet)
        assert result.is_empty()

    def test_arrhenius_rate_selection_alone_yields_all_three_coefficients(
        self, builder, mech_with_one_arrhenius_rxn, arrh_idx,
    ):
        """The rate uncertainty is what selects a reaction: marking it
        with no per-coefficient state fits the full A / n / Ea set."""
        builder.set_reaction_optimizable(arrh_idx, True)
        result = builder.build(mech_with_one_arrhenius_rxn)

        arrh_coefs = [c for c in result.coefficients if c.rxn_idx == arrh_idx]
        assert arrh_idx in result.optimizable_reactions
        assert {c.coef_name for c in arrh_coefs} == {
            "pre_exponential_factor", "temperature_exponent", "activation_energy",
        }

    def test_coef_bound_without_rate_selection_yields_no_entry(
        self, builder, mech_with_one_arrhenius_rxn, arrh_idx,
    ):
        """Coefficient uncertainties bound a reaction, they never select
        one: without the rate-level mark the set stays empty."""
        mech = mech_with_one_arrhenius_rxn
        bnds_key = next(iter(mech.coeffs_bnds[arrh_idx]))
        coef_name = next(iter(mech.coeffs_bnds[arrh_idx][bnds_key]))
        builder.set_coefficient_optimizable(arrh_idx, bnds_key, coef_name, True)
        result = builder.build(mech)
        assert result.is_empty()

    def test_coef_bound_does_not_reduce_the_fitted_coefficients(
        self, builder, mech_with_one_arrhenius_rxn, arrh_idx,
    ):
        """Bounding one coefficient narrows it; it does not drop the
        others from the fit."""
        mech = mech_with_one_arrhenius_rxn
        bnds_key = next(iter(mech.coeffs_bnds[arrh_idx]))
        coef_name = next(iter(mech.coeffs_bnds[arrh_idx][bnds_key]))
        builder.set_reaction_optimizable(arrh_idx, True)
        builder.set_coefficient_optimizable(arrh_idx, bnds_key, coef_name, True)

        result = builder.build(mech)
        arrh_coefs = [c for c in result.coefficients if c.rxn_idx == arrh_idx]
        assert len(arrh_coefs) == 3, (
            f"bounding {coef_name!r} should leave all three Arrhenius "
            f"coefficients in the fit, got {len(arrh_coefs)}"
        )

    def test_pressure_dep_reaction_emits_all_rate_coefs_unconditionally(
        self, builder, mech_with_one_arrhenius_rxn, plog_idx,
    ):
        """Pressure-dependent rxns (Plog/Falloff/Lindemann/Sri/Tsang/Troe)
        are recast as Troe and the full Troe parameterization is optimized.
        Marking the rxn alone — with no per-coef toggles — must yield all
        rate-limb coefs so coef_opt aligns slot-for-slot with the
        post-upgrade fit return.  Pre-upgrade Plog has only ``low_rate``
        and ``high_rate`` keys, so this emits 6 entries."""
        mech = mech_with_one_arrhenius_rxn
        builder.set_reaction_optimizable(plog_idx, True)
        result = builder.build(mech)

        plog_coefs = [c for c in result.coefficients if c.rxn_idx == plog_idx]
        assert len(plog_coefs) == 6, (
            f"Plog rxn {plog_idx} should emit 6 coefs (3 low + 3 high) "
            f"despite no per-coef toggles, got {len(plog_coefs)}"
        )
        bnds_keys = {c.bnds_key for c in plog_coefs}
        assert bnds_keys == {"low_rate", "high_rate"}

    def test_pressure_dep_reaction_unmarked_yields_no_entries(
        self, builder, mech_with_one_arrhenius_rxn, plog_idx,
    ):
        """The reaction-level gate is still required: an un-marked Plog
        rxn yields no coefficients regardless of per-coef toggles."""
        mech = mech_with_one_arrhenius_rxn
        # Toggle a per-coef even though reaction-level is False
        builder.set_coefficient_optimizable(
            plog_idx, "low_rate", "pre_exponential_factor", True,
        )
        result = builder.build(mech)
        plog_coefs = [c for c in result.coefficients if c.rxn_idx == plog_idx]
        assert len(plog_coefs) == 0
