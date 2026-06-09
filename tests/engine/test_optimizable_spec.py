"""Tests for the typed optimization-input models."""
import cantera as ct
import numpy as np
import pytest
from pydantic import ValidationError

from frhodo.api import (
    AlgorithmSettings,
    AlgorithmStage,
    CoefUncertainty,
    OptimizableRate,
    OptimizableSpec,
    OptimizableSpecBuilder,
    RateUncertainty,
)


def _first_arrhenius_idx(mech):
    return next(
        i for i, r in enumerate(mech.gas.reactions())
        if type(r.rate) is ct.ArrheniusRate
    )


class TestCoefUncertainty:
    def test_factor_resolves(self):
        cu = CoefUncertainty(factor=2.0)
        assert cu.resolve(10.0, coef_name="A") == (5.0, 20.0)

    def test_delta_resolves(self):
        cu = CoefUncertainty(delta=3.0)
        assert cu.resolve(10.0, coef_name="Ea") == (7.0, 13.0)

    def test_bounds_resolves(self):
        cu = CoefUncertainty(bounds=(1.0, 100.0))
        assert cu.resolve(50.0, coef_name="T3") == (1.0, 100.0)

    def test_bounds_must_bracket_nominal(self):
        cu = CoefUncertainty(bounds=(1.0, 5.0))
        with pytest.raises(ValueError, match="bracket"):
            cu.resolve(10.0, coef_name="T3")

    @pytest.mark.parametrize("kwargs", [
        {},
        {"factor": 1.0, "delta": 1.0},
        {"factor": 1.0, "bounds": (1.0, 2.0)},
        {"delta": 1.0, "bounds": (1.0, 2.0)},
        {"factor": 1.0, "delta": 1.0, "bounds": (1.0, 2.0)},
    ])
    def test_exactly_one_mode(self, kwargs):
        with pytest.raises(ValidationError):
            CoefUncertainty(**kwargs)


class TestRateUncertainty:
    def test_default(self):
        ru = RateUncertainty()
        assert ru.factor == 2.0

    def test_explicit(self):
        ru = RateUncertainty(factor=10.0)
        assert ru.factor == 10.0

    def test_factor_must_be_positive(self):
        with pytest.raises(ValidationError):
            RateUncertainty(factor=-1.0)


class TestOptimizableSpecBuild:
    def test_resolves_against_mech(self, loaded_cycloheptane):
        # rxn 0 of Cycloheptane is Arrhenius
        spec = OptimizableSpec(rates=[
            OptimizableRate(rxn_idx=0, rate=RateUncertainty(factor=2.0)),
        ])
        result = spec.build(loaded_cycloheptane)
        assert not result.is_empty()
        assert all(c.rxn_idx == 0 for c in result.coefficients)

    def test_to_builder_exposes_recast_query_and_matches_build(self, loaded_cycloheptane):
        # recast_to_troe consumes the builder and queries
        # is_coefficient_optimizable; build() must just wrap it.
        spec = OptimizableSpec(rates=[
            OptimizableRate(rxn_idx=0, rate=RateUncertainty(factor=2.0)),
        ])
        builder = spec.to_builder(loaded_cycloheptane)
        assert builder.is_reaction_optimizable(0)
        assert callable(builder.is_coefficient_optimizable)
        from_builder = builder.build(loaded_cycloheptane)
        assert from_builder.coefficients == spec.build(loaded_cycloheptane).coefficients

    def test_rejects_invalid_rxn_idx(self, loaded_cycloheptane):
        spec = OptimizableSpec(rates=[
            OptimizableRate(rxn_idx=99999, rate=RateUncertainty()),
        ])
        with pytest.raises(ValueError, match="out of range"):
            spec.build(loaded_cycloheptane)

    def test_rejects_duplicate_rxn_idx(self, loaded_cycloheptane):
        spec = OptimizableSpec(rates=[
            OptimizableRate(rxn_idx=0, rate=RateUncertainty()),
            OptimizableRate(rxn_idx=0, rate=RateUncertainty()),
        ])
        with pytest.raises(ValueError, match="duplicate"):
            spec.build(loaded_cycloheptane)

    def test_applies_rate_uncertainty_to_mech(self, loaded_cycloheptane):
        spec = OptimizableSpec(rates=[
            OptimizableRate(rxn_idx=0, rate=RateUncertainty(factor=3.0)),
        ])
        spec.build(loaded_cycloheptane)
        assert loaded_cycloheptane.rate_bnds[0]["value"] == 3.0
        assert loaded_cycloheptane.rate_bnds[0]["type"] == "F"

    def test_applies_coef_factor_override(self, loaded_cycloheptane):
        arrh_idx = _first_arrhenius_idx(loaded_cycloheptane)
        spec = OptimizableSpec(rates=[
            OptimizableRate(
                rxn_idx=arrh_idx,
                rate=RateUncertainty(factor=2.0),
                coefficients={
                    "pre_exponential_factor": CoefUncertainty(factor=5.0),
                },
            ),
        ])
        spec.build(loaded_cycloheptane)
        bnds_key = next(iter(loaded_cycloheptane.coeffs_bnds[arrh_idx]))
        d = loaded_cycloheptane.coeffs_bnds[arrh_idx][bnds_key]["pre_exponential_factor"]
        assert d["value"] == 5.0
        assert d["type"] == "F"

    def test_optimize_subset_restricts_coefs(self, loaded_cycloheptane):
        arrh_idx = _first_arrhenius_idx(loaded_cycloheptane)
        spec = OptimizableSpec(rates=[
            OptimizableRate(
                rxn_idx=arrh_idx, rate=RateUncertainty(),
                optimize=["pre_exponential_factor"],
            ),
        ])
        result = spec.build(loaded_cycloheptane)
        coef_names = {c.coef_name for c in result.coefficients}
        assert coef_names == {"pre_exponential_factor"}

    def test_rejects_pdep_with_coef_override(self, loaded_cycloheptane):
        import cantera as ct

        pdep_idx = next(
            (
                i for i, r in enumerate(loaded_cycloheptane.gas.reactions())
                if isinstance(r.rate, (
                    ct.FalloffRate, ct.LindemannRate, ct.TroeRate, ct.PlogRate,
                    ct.SriRate, ct.TsangRate,
                ))
            ),
            None,
        )
        if pdep_idx is None:
            pytest.skip("test mech has no pressure-dependent reactions")

        spec = OptimizableSpec(rates=[
            OptimizableRate(
                rxn_idx=pdep_idx, rate=RateUncertainty(),
                coefficients={"A_0": CoefUncertainty(factor=2.0)},
            ),
        ])
        with pytest.raises(ValueError, match="recast to Troe"):
            spec.build(loaded_cycloheptane)


class TestOptimizableSpecBuilder:
    def test_set_and_clear(self):
        b = OptimizableSpecBuilder()
        b.set_rxn(0, enabled=True, rate=RateUncertainty(factor=5.0))
        spec = b.build()
        assert len(spec.rates) == 1
        assert spec.rates[0].rxn_idx == 0

        b.clear_rxn(0)
        spec = b.build()
        assert spec.rates == []

    def test_set_with_enabled_false_clears(self):
        b = OptimizableSpecBuilder()
        b.set_rxn(0, enabled=True, rate=RateUncertainty(factor=5.0))
        b.set_rxn(0, enabled=False)
        spec = b.build()
        assert spec.rates == []

    def test_default_rate_round_trip(self):
        b = OptimizableSpecBuilder()
        b.set_default_rate(RateUncertainty(factor=7.5))
        spec = b.build()
        assert spec.default_rate.factor == 7.5


class TestAlgorithmSettings:
    def test_defaults(self):
        s = AlgorithmSettings()
        assert s.global_stage.algorithm == "RBFOpt"
        assert s.local_stage.algorithm == "Subplex"
        assert s.global_stage.enabled is True
        assert s.local_stage.enabled is True

    def test_to_legacy_dict_shape(self):
        s = AlgorithmSettings()
        d = s.to_legacy_dict()
        for stage_key in ("global", "local"):
            assert stage_key in d
            assert "algorithm" in d[stage_key]
            assert "initial_step" in d[stage_key]
            assert "stop_criteria_type" in d[stage_key]
            assert "stop_criteria_val" in d[stage_key]
            assert "run" in d[stage_key]

    def test_stage_disable_round_trips(self):
        s = AlgorithmSettings(
            global_stage=AlgorithmStage(algorithm="RBFOpt", enabled=False),
        )
        assert s.global_stage.enabled is False
        assert s.to_legacy_dict()["global"]["run"] is False

    def test_unknown_algorithm_label_rejected_on_translation(self):
        s = AlgorithmSettings(
            local_stage=AlgorithmStage(algorithm="NotAReal"),
        )
        with pytest.raises(ValueError, match="unknown optimization algorithm"):
            s.to_legacy_dict()


class TestOptimizableSetSlotIndex:
    """The OptimizableSet helpers let callers interpret result.x slot-by-slot."""

    def test_slot_index_finds_coefficient(self, loaded_cycloheptane):
        arrh_idx = _first_arrhenius_idx(loaded_cycloheptane)
        spec = OptimizableSpec(rates=[
            OptimizableRate(
                rxn_idx=arrh_idx, rate=RateUncertainty(),
                optimize=["pre_exponential_factor", "activation_energy"],
            ),
        ])
        result = spec.build(loaded_cycloheptane)
        idx = result.slot_index(arrh_idx, "pre_exponential_factor")
        assert idx is not None
        assert result.coefficients[idx].coef_name == "pre_exponential_factor"

    def test_slot_index_returns_none_for_missing(self, loaded_cycloheptane):
        arrh_idx = _first_arrhenius_idx(loaded_cycloheptane)
        spec = OptimizableSpec(rates=[
            OptimizableRate(
                rxn_idx=arrh_idx, rate=RateUncertainty(),
                optimize=["pre_exponential_factor"],
            ),
        ])
        result = spec.build(loaded_cycloheptane)
        assert result.slot_index(99, "X") is None

    def test_rxn_slots_returns_all_indices_for_rxn(self, loaded_cycloheptane):
        arrh_idx = _first_arrhenius_idx(loaded_cycloheptane)
        spec = OptimizableSpec(rates=[
            OptimizableRate(rxn_idx=arrh_idx, rate=RateUncertainty()),
        ])
        result = spec.build(loaded_cycloheptane)
        slots = result.rxn_slots(arrh_idx)
        assert len(slots) == len(result.coefficients)
        assert slots == tuple(range(len(slots)))
