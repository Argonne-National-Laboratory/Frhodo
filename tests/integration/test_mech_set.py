"""``ChemicalMechanism.set_mechanism`` rebuild round-trip and
``modify_reactions`` rate-coefficient editor.

Each test uses a fresh ``ChemicalMechanism`` (function-scoped fixture)
because both methods mutate ``self.gas``.
"""
import collections
from copy import deepcopy

import cantera as ct
import numpy as np
import pytest

from frhodo.experiment import ExperimentalShock
from frhodo.simulation.mechanism.mechanism_loader import MechanismLoader
from frhodo.optimize.parameters import (
    OptimizableSetBuilder,
    build_rxn_coef_opt,
    build_rxn_rate_opt,
)


@pytest.fixture
def fresh_mech(cycloheptane_paths):
    return MechanismLoader().load(cycloheptane_paths)


class TestSetMechanismRoundTrip:
    def test_rebuild_does_not_raise(self, fresh_mech):
        """``set_mechanism`` runs to completion on the loaded mech."""
        fresh_mech.set_mechanism(fresh_mech.reset_mech)

    def test_rebuild_preserves_reaction_count(self, fresh_mech):
        original_n = fresh_mech.gas.n_reactions
        fresh_mech.set_mechanism(fresh_mech.reset_mech)
        assert fresh_mech.gas.n_reactions == original_n, (
            f"reaction count drift after set_mechanism: "
            f"{original_n} -> {fresh_mech.gas.n_reactions}"
        )

    def test_rebuild_preserves_species_count(self, fresh_mech):
        original_n = fresh_mech.gas.n_species
        fresh_mech.set_mechanism(fresh_mech.reset_mech)
        assert fresh_mech.gas.n_species == original_n

    def test_rebuild_preserves_reaction_type_breakdown(self, fresh_mech):
        before = collections.Counter(
            r.reaction_type for r in fresh_mech.gas.reactions()
        )
        fresh_mech.set_mechanism(fresh_mech.reset_mech)
        after = collections.Counter(
            r.reaction_type for r in fresh_mech.gas.reactions()
        )
        assert before == after, (
            f"reaction-type breakdown changed after rebuild:\n"
            f"  before: {dict(before)}\n"
            f"  after:  {dict(after)}"
        )

    def test_rebuild_preserves_three_body_efficiencies(self, fresh_mech):
        """Regression for the Cantera 3 ``rxn.third_body.efficiencies`` rewire."""
        before = {
            i: dict(r.third_body.efficiencies)
            for i, r in enumerate(fresh_mech.gas.reactions())
            if r.reaction_type.startswith("three-body")
        }
        assert before, "fixture has no three-body reactions; test is meaningless"

        fresh_mech.set_mechanism(fresh_mech.reset_mech)

        after = {
            i: dict(r.third_body.efficiencies)
            for i, r in enumerate(fresh_mech.gas.reactions())
            if r.reaction_type.startswith("three-body")
        }
        assert before == after, (
            f"three-body efficiencies changed after rebuild:\n"
            f"  before: {before}\n"
            f"  after:  {after}"
        )

    def test_rebuild_preserves_falloff_efficiencies(self, fresh_mech):
        before = {
            i: dict(r.third_body.efficiencies)
            for i, r in enumerate(fresh_mech.gas.reactions())
            if r.reaction_type.startswith("falloff")
        }
        assert before, "fixture has no falloff reactions; test is meaningless"

        fresh_mech.set_mechanism(fresh_mech.reset_mech)

        after = {
            i: dict(r.third_body.efficiencies)
            for i, r in enumerate(fresh_mech.gas.reactions())
            if r.reaction_type.startswith("falloff")
        }
        assert before == after


class TestModifyReactions:
    """``modify_reactions`` edits Arrhenius coefficients in place.

    Reaction 2 in the Cycloheptane mech (H + cC7H14 <=> H2 + cC7H13) is
    a plain Arrhenius reaction; the forward rate constant scales linearly
    with the pre-exponential factor.
    """

    ARR_RXN_IDX = 2

    @pytest.fixture
    def state(self, fresh_mech):
        """Set a known TPX so forward_rate_constants is reproducible."""
        fresh_mech.set_TPX(1500.0, 20_000.0, {"Kr": 0.96, "cC7H14": 0.04})
        return fresh_mech

    def test_doubling_pre_exponential_doubles_forward_rate(self, state):
        k_before = state.gas.forward_rate_constants[self.ARR_RXN_IDX]

        new_coeffs = deepcopy(state.coeffs)
        new_coeffs[self.ARR_RXN_IDX][0]["pre_exponential_factor"] *= 2.0
        state.modify_reactions(new_coeffs)

        # set_TPX again because modify_reactions mutates the gas object.
        state.set_TPX(1500.0, 20_000.0, {"Kr": 0.96, "cC7H14": 0.04})
        k_after = state.gas.forward_rate_constants[self.ARR_RXN_IDX]

        assert k_after / k_before == pytest.approx(2.0, rel=1e-12), (
            f"k_forward should double when A doubles; got ratio {k_after / k_before}"
        )

    def test_unchanged_coeffs_leave_rate_unchanged(self, state):
        """Calling modify_reactions with identical coeffs should be a no-op."""
        k_before = state.gas.forward_rate_constants[self.ARR_RXN_IDX]

        state.modify_reactions(deepcopy(state.coeffs))

        state.set_TPX(1500.0, 20_000.0, {"Kr": 0.96, "cC7H14": 0.04})
        k_after = state.gas.forward_rate_constants[self.ARR_RXN_IDX]

        assert k_after == pytest.approx(k_before, rel=1e-15), (
            "rate drifted when coefficients were unchanged"
        )

    def test_single_int_rxnIdx_is_accepted(self, state):
        """``rxnIdxs=int`` should be auto-wrapped to a single-element list."""
        new_coeffs = deepcopy(state.coeffs)
        new_coeffs[self.ARR_RXN_IDX][0]["pre_exponential_factor"] *= 3.0

        # Should not raise:
        state.modify_reactions(new_coeffs, rxnIdxs=self.ARR_RXN_IDX)

        state.set_TPX(1500.0, 20_000.0, {"Kr": 0.96, "cC7H14": 0.04})
        new_A = state.gas.reaction(self.ARR_RXN_IDX).rate.pre_exponential_factor
        original_A = state.coeffs[self.ARR_RXN_IDX][0]["pre_exponential_factor"]
        assert new_A == pytest.approx(3.0 * original_A, rel=1e-12)

    def test_modify_reactions_silently_ignores_chebyshev(self, loaded_chebyshev):
        """``modify_reactions`` is a no-op for ChebyshevRate."""
        loaded_chebyshev.gas.TPX = 1500.0, ct.one_atm, {"H2O": 1.0}
        kf_before = float(loaded_chebyshev.gas.forward_rate_constants[1])

        loaded_chebyshev.coeffs[1] = "anything"
        loaded_chebyshev.modify_reactions(loaded_chebyshev.coeffs, rxnIdxs=[1])

        loaded_chebyshev.gas.TPX = 1500.0, ct.one_atm, {"H2O": 1.0}
        kf_after = float(loaded_chebyshev.gas.forward_rate_constants[1])
        assert kf_after == pytest.approx(kf_before, rel=1e-15), (
            f"Chebyshev kf changed under modify_reactions: "
            f"{kf_before} -> {kf_after}"
        )


class TestModifyReactionsTroe:
    """``modify_reactions`` editing a Troe reaction.

    Cycloheptane reaction 36 (C2H6 (+M) <=> 2 CH3 (+M)) is the only
    Troe rxn in the bundled mech; it carries 4 falloff coefficients.
    """

    TROE_RXN_IDX = 36

    @pytest.fixture
    def state(self, fresh_mech):
        fresh_mech.set_TPX(1500.0, 20_000.0, {"Kr": 0.96, "cC7H14": 0.04})

        return fresh_mech

    def test_only_low_rate_changed_preserves_falloff_coeffs(self, state):
        """Changing only the low-rate Arrhenius limb of a Troe reaction
        must preserve the existing falloff coefficients on the rebuilt
        rate. Cantera 3.2 silently accepts ``None`` for the falloff arg
        (yielding ``[nan, inf, inf]``); older versions error with
        ``TroeRate::setFalloffCoeffs: ... Received 0``. Either way the
        rxn is broken if unchanged components default to ``None``.
        """
        original_falloff = np.array(
            state.gas.reaction(self.TROE_RXN_IDX).rate.falloff_coeffs
        )

        new_coeffs = deepcopy(state.coeffs)
        new_coeffs[self.TROE_RXN_IDX]["low_rate"]["pre_exponential_factor"] *= 2.0

        state.modify_reactions(new_coeffs, rxnIdxs=self.TROE_RXN_IDX)

        rxn = state.gas.reaction(self.TROE_RXN_IDX)
        new_falloff = np.array(rxn.rate.falloff_coeffs)
        assert isinstance(rxn.rate, ct.TroeRate)
        assert np.isfinite(new_falloff).all(), (
            f"falloff_coeffs corrupted to non-finite: {new_falloff}"
        )
        np.testing.assert_array_equal(
            new_falloff, original_falloff,
            err_msg=(
                f"falloff_coeffs changed when only low_rate was edited: "
                f"{original_falloff} -> {new_falloff}"
            ),
        )

    def test_only_high_rate_changed_preserves_low_rate_and_falloff(self, state):
        rxn0 = state.gas.reaction(self.TROE_RXN_IDX)
        original_low_A = rxn0.rate.low_rate.pre_exponential_factor
        original_falloff = np.array(rxn0.rate.falloff_coeffs)

        new_coeffs = deepcopy(state.coeffs)
        new_coeffs[self.TROE_RXN_IDX]["high_rate"]["pre_exponential_factor"] *= 2.0

        state.modify_reactions(new_coeffs, rxnIdxs=self.TROE_RXN_IDX)

        rxn = state.gas.reaction(self.TROE_RXN_IDX)
        new_falloff = np.array(rxn.rate.falloff_coeffs)
        assert np.isfinite(new_falloff).all()
        np.testing.assert_array_equal(new_falloff, original_falloff)
        assert rxn.rate.low_rate.pre_exponential_factor == pytest.approx(
            original_low_A, rel=1e-15,
        ), "low_rate was dropped when only high_rate was edited"

    def test_unchanged_coeffs_leave_rate_unchanged(self, state):
        """Identical coeffs should be a full no-op for Troe too."""
        k_before = float(state.gas.forward_rate_constants[self.TROE_RXN_IDX])

        state.modify_reactions(deepcopy(state.coeffs), rxnIdxs=self.TROE_RXN_IDX)

        state.set_TPX(1500.0, 20_000.0, {"Kr": 0.96, "cC7H14": 0.04})
        k_after = float(state.gas.forward_rate_constants[self.TROE_RXN_IDX])
        assert k_after == pytest.approx(k_before, rel=1e-15), (
            "Troe rate drifted when coefficients were unchanged"
        )


class TestSetMechanismRebuildChebyshev:
    """``set_mechanism`` rebuild preserves Chebyshev reactions."""

    def test_rebuild_preserves_reaction_count(self, loaded_chebyshev):
        original_n = loaded_chebyshev.gas.n_reactions
        loaded_chebyshev.set_mechanism(loaded_chebyshev.reset_mech)
        assert loaded_chebyshev.gas.n_reactions == original_n

    def test_rebuild_preserves_chebyshev_rate_type(self, loaded_chebyshev):
        loaded_chebyshev.set_mechanism(loaded_chebyshev.reset_mech)
        rebuilt = loaded_chebyshev.gas.reaction(1)
        assert rebuilt.reaction_type == "Chebyshev"
        assert isinstance(rebuilt.rate, ct.ChebyshevRate)

    def test_rebuild_preserves_chebyshev_rate_constant(self, loaded_chebyshev):
        loaded_chebyshev.gas.TPX = 1500.0, ct.one_atm, {"H2O": 1.0}
        kf_before = float(loaded_chebyshev.gas.forward_rate_constants[1])

        loaded_chebyshev.set_mechanism(loaded_chebyshev.reset_mech)

        loaded_chebyshev.gas.TPX = 1500.0, ct.one_atm, {"H2O": 1.0}
        kf_after = float(loaded_chebyshev.gas.forward_rate_constants[1])
        assert kf_after == pytest.approx(kf_before, rel=1e-12), (
            f"Chebyshev kf drifted across rebuild: {kf_before} -> {kf_after}"
        )


def _build_recast_inputs(mech, target_idx, thermo_mix, *, F=2.0):
    """Mark ``target_idx`` fully optimizable, run the public builders.

    Returns ``(optimizables, rxn_coef_opt, rxn_rate_opt)`` ready to feed
    :meth:`ChemicalMechanism.recast_to_troe`.
    """
    mech.rate_bnds[target_idx]["value"] = F
    mech.rate_bnds[target_idx]["type"] = "F"

    builder = OptimizableSetBuilder()
    builder.set_reaction_optimizable(target_idx, True)
    for bnds_key, sub in mech.coeffs_bnds[target_idx].items():
        if bnds_key == "falloff_parameters":
            continue
        for coef_name in sub:
            if not isinstance(coef_name, str):
                continue
            d = sub[coef_name]
            d["value"] = F
            d["type"] = "F"
            builder.set_coefficient_optimizable(target_idx, bnds_key, coef_name, True)

    optimizable_set = builder.build(mech)
    coef_opt = list(optimizable_set.coefficients)
    shocks = [
        ExperimentalShock.from_dict({
            "T_reactor": T, "P_reactor": 20_000.0,
            "thermo_mix": thermo_mix,
        })
        for T in (1500.0, 1700.0)
    ]
    rxn_coef_opt = build_rxn_coef_opt(mech, coef_opt, shocks)
    rxn_rate_opt = build_rxn_rate_opt(mech, rxn_coef_opt)

    return builder, rxn_coef_opt, rxn_rate_opt


def _sample_kf(mech, rxnIdx, T, P, mix):
    mech.set_TPX(T, P, mix)
    return float(mech.gas.forward_rate_constants[rxnIdx])


class TestRecastToTroe:
    """``ChemicalMechanism.recast_to_troe`` routes every Cantera rate
    type Frhodo supports.

    The fixture mech (``loaded_all_rate_types``) carries one reaction
    per rate type Cantera exposes through Frhodo:

      * idx 0 — ArrheniusRate (skip path)
      * idx 1 — LindemannRate (Falloff-family, fresh refit, no rebuild)
      * idx 2 — TroeRate (no-op refit; just install)
      * idx 3 — PlogRate (structural recast, rebuild)
      * idx 4 — ChebyshevRate (structural recast, rebuild)
      * idx 5 — SriRate (Falloff-family, fresh refit, no rebuild)
    """

    MIX = {"AR": 1.0}
    SAMPLE_T = 1500.0
    SAMPLE_P = 101_325.0
    KF_PRESERVATION_RTOL = 0.5

    def test_arrhenius_rxn_is_skipped(self, loaded_all_rate_types):
        mech = loaded_all_rate_types
        builder, rxn_coef_opt, rxn_rate_opt = _build_recast_inputs(mech, 0, self.MIX)
        rxn_type_before = type(mech.gas.reaction(0).rate)

        rxns_changed, mech_rebuilt = mech.recast_to_troe(
            rxn_coef_opt, rxn_rate_opt, builder,
        )

        assert rxns_changed == [], "ArrheniusRate must not be touched"
        assert mech_rebuilt is False
        assert type(mech.gas.reaction(0).rate) is rxn_type_before

    def test_lindemann_rxn_refit_to_troe(self, loaded_all_rate_types):
        mech = loaded_all_rate_types
        builder, rxn_coef_opt, rxn_rate_opt = _build_recast_inputs(mech, 1, self.MIX)

        rxns_changed, mech_rebuilt = mech.recast_to_troe(
            rxn_coef_opt, rxn_rate_opt, builder,
        )

        assert 1 in rxns_changed
        assert mech_rebuilt is False, (
            "Lindemann shares the falloff_parameters slot; no structural rebuild"
        )
        assert mech.coeffs[1]["falloff_type"] == "Troe"
        assert len(mech.coeffs[1]["falloff_parameters"]) == 4

    def test_troe_rxn_no_structural_change(self, loaded_all_rate_types):
        mech = loaded_all_rate_types
        builder, rxn_coef_opt, rxn_rate_opt = _build_recast_inputs(mech, 2, self.MIX)

        rxns_changed, mech_rebuilt = mech.recast_to_troe(
            rxn_coef_opt, rxn_rate_opt, builder,
        )

        assert 2 not in rxns_changed, (
            "TroeRate already matches the target form; recast should leave it alone"
        )
        assert mech_rebuilt is False
        assert isinstance(mech.gas.reaction(2).rate, ct.TroeRate)

    def test_plog_rxn_recast_to_troe(self, loaded_all_rate_types):
        mech = loaded_all_rate_types
        kf_before = _sample_kf(mech, 3, self.SAMPLE_T, self.SAMPLE_P, self.MIX)
        builder, rxn_coef_opt, rxn_rate_opt = _build_recast_inputs(mech, 3, self.MIX)

        rxns_changed, mech_rebuilt = mech.recast_to_troe(
            rxn_coef_opt, rxn_rate_opt, builder,
        )

        assert 3 in rxns_changed
        assert mech_rebuilt is True
        assert isinstance(mech.gas.reaction(3).rate, ct.TroeRate)
        kf_after = _sample_kf(mech, 3, self.SAMPLE_T, self.SAMPLE_P, self.MIX)
        assert kf_after == pytest.approx(kf_before, rel=self.KF_PRESERVATION_RTOL), (
            f"Plog → Troe drift: {kf_before:.4e} → {kf_after:.4e}"
        )

    def test_chebyshev_rxn_recast_to_troe(self, loaded_all_rate_types):
        mech = loaded_all_rate_types
        kf_before = _sample_kf(mech, 4, self.SAMPLE_T, self.SAMPLE_P, self.MIX)
        builder, rxn_coef_opt, rxn_rate_opt = _build_recast_inputs(mech, 4, self.MIX)

        rxns_changed, mech_rebuilt = mech.recast_to_troe(
            rxn_coef_opt, rxn_rate_opt, builder,
        )

        assert 4 in rxns_changed
        assert mech_rebuilt is True
        assert isinstance(mech.gas.reaction(4).rate, ct.TroeRate)
        kf_after = _sample_kf(mech, 4, self.SAMPLE_T, self.SAMPLE_P, self.MIX)
        assert kf_after == pytest.approx(kf_before, rel=self.KF_PRESERVATION_RTOL), (
            f"Chebyshev → Troe drift: {kf_before:.4e} → {kf_after:.4e}"
        )

    def test_chebyshev_sampling_spans_declared_validity_range(self, loaded_all_rate_types):
        mech = loaded_all_rate_types
        rxn = mech.gas.reaction(4)
        T_lo, T_hi = rxn.rate.temperature_range
        P_lo, P_hi = rxn.rate.pressure_range
        _, rxn_coef_opt, _ = _build_recast_inputs(mech, 4, self.MIX)
        rc = rxn_coef_opt[0]
        # sampled over the Chebyshev's own validity range, not the
        # experimental shock window (1500-1700 K, 20 kPa)
        assert rc["T"].min() == pytest.approx(T_lo, rel=0.02)
        assert rc["T"].max() == pytest.approx(T_hi, rel=0.02)
        assert rc["P"].min() == pytest.approx(P_lo, rel=0.02)
        assert rc["P"].max() == pytest.approx(P_hi, rel=0.02)

    def test_sri_rxn_refit_to_troe(self, loaded_all_rate_types):
        mech = loaded_all_rate_types
        builder, rxn_coef_opt, rxn_rate_opt = _build_recast_inputs(mech, 5, self.MIX)

        rxns_changed, mech_rebuilt = mech.recast_to_troe(
            rxn_coef_opt, rxn_rate_opt, builder,
        )

        assert 5 in rxns_changed
        assert mech_rebuilt is False
        assert mech.coeffs[5]["falloff_type"] == "Troe"
        assert len(mech.coeffs[5]["falloff_parameters"]) == 4

    def test_recast_records_fit_log_rms(self, loaded_all_rate_types):
        """recast_to_troe records a finite per-reaction held-out log-RMS."""
        mech = loaded_all_rate_types
        for idx in (1, 3, 4):  # Lindemann falloff, Plog, Chebyshev
            builder, rxn_coef_opt, rxn_rate_opt = _build_recast_inputs(mech, idx, self.MIX)
            mech.recast_to_troe(rxn_coef_opt, rxn_rate_opt, builder)
            assert idx in mech.recast_log_rms
            assert np.isfinite(mech.recast_log_rms[idx])
            print(f"recast R{idx + 1} log-RMS = {mech.recast_log_rms[idx]:.4f}")


class TestRecastPdepAtPressure:
    """``ChemicalMechanism.recast_pdep_at_pressure`` flattens every
    pressure-dependent rxn to Arrhenius form valid at the target P.

    Uses the same ``loaded_all_rate_types`` fixture as ``TestRecastToTroe``
    (one rxn per supported rate type at fixed indices 0-5).
    """

    MIX = {"AR": 1.0}
    P = 101_325.0

    def test_returns_new_mech_source_unchanged(self, loaded_all_rate_types):
        before = [type(r.rate) for r in loaded_all_rate_types.gas.reactions()]
        loaded_all_rate_types.recast_pdep_at_pressure(self.P, self.MIX)
        after = [type(r.rate) for r in loaded_all_rate_types.gas.reactions()]
        assert before == after, "source mech was mutated"

    def test_arrhenius_passes_through_unchanged(self, loaded_all_rate_types):
        new_mech = loaded_all_rate_types.recast_pdep_at_pressure(self.P, self.MIX)
        assert isinstance(new_mech.gas.reaction(0).rate, ct.ArrheniusRate)
        rate_orig = loaded_all_rate_types.gas.reaction(0).rate
        rate_new = new_mech.gas.reaction(0).rate
        assert rate_new.pre_exponential_factor == pytest.approx(
            rate_orig.pre_exponential_factor, rel=1e-12,
        )
        assert rate_new.temperature_exponent == pytest.approx(
            rate_orig.temperature_exponent, rel=1e-12,
        )
        assert rate_new.activation_energy == pytest.approx(
            rate_orig.activation_energy, rel=1e-12,
        )

    @pytest.mark.parametrize("rxn_idx", [1, 2, 5])
    def test_falloff_family_becomes_three_body_arrhenius(
        self, loaded_all_rate_types, rxn_idx,
    ):
        """Lindemann, Troe, SRI → three-body Arrhenius with efficiencies preserved."""
        rxn_orig = loaded_all_rate_types.gas.reaction(rxn_idx)
        if rxn_orig.third_body is None:
            pytest.skip(f"rxn {rxn_idx} has no third body")
        eff_orig = dict(rxn_orig.third_body.efficiencies)

        new_mech = loaded_all_rate_types.recast_pdep_at_pressure(self.P, self.MIX)
        rxn_new = new_mech.gas.reaction(rxn_idx)

        assert isinstance(rxn_new.rate, ct.ArrheniusRate)
        assert rxn_new.third_body is not None
        assert dict(rxn_new.third_body.efficiencies) == eff_orig, (
            f"rxn {rxn_idx} efficiencies dropped or altered"
        )

    @pytest.mark.parametrize("rxn_idx", [3, 4])
    def test_plog_chebyshev_becomes_pure_arrhenius(
        self, loaded_all_rate_types, rxn_idx,
    ):
        """Plog and Chebyshev → pure Arrhenius (no (+M) marker)."""
        new_mech = loaded_all_rate_types.recast_pdep_at_pressure(self.P, self.MIX)
        rxn_new = new_mech.gas.reaction(rxn_idx)
        assert isinstance(rxn_new.rate, ct.ArrheniusRate)
        assert rxn_new.third_body is None, (
            f"rxn {rxn_idx} unexpectedly carries a (+M) factor"
        )

    def test_plog_recast_matches_at_target_pressure(self, loaded_all_rate_types):
        """A pure Plog rxn at a single P-bracket is itself an Arrhenius;
        fit should match within the linear-LS tolerance."""
        T = 1500.0
        loaded_all_rate_types.gas.TPX = T, self.P, self.MIX
        k_orig = float(loaded_all_rate_types.gas.forward_rate_constants[3])

        new_mech = loaded_all_rate_types.recast_pdep_at_pressure(self.P, self.MIX)
        new_mech.gas.TPX = T, self.P, self.MIX
        k_new = float(new_mech.gas.forward_rate_constants[3])

        assert k_new == pytest.approx(k_orig, rel=1e-3), (
            f"Plog recast mismatch at (T={T}, P={self.P}): "
            f"{k_orig:.4e} -> {k_new:.4e}"
        )

    def test_yaml_round_trip_after_recast(self, loaded_all_rate_types, tmp_path):
        """Recast → YAML → reload preserves the recast rate types."""
        new_mech = loaded_all_rate_types.recast_pdep_at_pressure(self.P, self.MIX)
        yaml_path = tmp_path / "recast.yaml"
        yaml_path.write_text(new_mech.to_yaml_text())

        reloaded = ct.Solution(yaml=yaml_path.read_text())
        for idx in range(reloaded.n_reactions):
            assert isinstance(reloaded.reaction(idx).rate, ct.ArrheniusRate), (
                f"reloaded rxn {idx} should be ArrheniusRate; "
                f"got {type(reloaded.reaction(idx).rate).__name__}"
            )


class TestRecastReactionAtPressure:
    """``ChemicalMechanism.recast_reaction_at_pressure`` flattens one
    pressure-dependent rxn in place, reversibly.

    Same ``loaded_all_rate_types`` fixture (one rxn per rate type at fixed
    indices 0-5): 0 Arrhenius, 1 Lindemann, 2 Troe, 3 Plog, 4 Chebyshev,
    5 SRI.
    """

    MIX = {"AR": 1.0}
    P = 101_325.0
    T = 1500.0

    def test_arrhenius_rxn_is_noop(self, loaded_all_rate_types):
        mech = loaded_all_rate_types
        changed = mech.recast_reaction_at_pressure(0, self.P, self.MIX)
        assert changed is False, "non-pdep rxn must not be recast"
        assert isinstance(mech.gas.reaction(0).rate, ct.ArrheniusRate)
        assert mech.is_reaction_recast(0) is False

    def test_plog_recast_in_place_becomes_arrhenius(self, loaded_all_rate_types):
        mech = loaded_all_rate_types
        mech.set_TPX(self.T, self.P, self.MIX)
        k_before = float(mech.gas.forward_rate_constants[3])

        changed = mech.recast_reaction_at_pressure(3, self.P, self.MIX)

        assert changed is True
        assert mech.is_reaction_recast(3) is True
        rxn_new = mech.gas.reaction(3)
        assert isinstance(rxn_new.rate, ct.ArrheniusRate)
        assert rxn_new.third_body is None, "Plog recast must not carry (+M)"

        mech.set_TPX(self.T, self.P, self.MIX)
        k_after = float(mech.gas.forward_rate_constants[3])
        assert k_after == pytest.approx(k_before, rel=1e-3), (
            f"Plog recast mismatch at target P: {k_before:.4e} -> {k_after:.4e}"
        )

    def test_falloff_recast_becomes_three_body_arrhenius(self, loaded_all_rate_types):
        mech = loaded_all_rate_types
        eff_orig = dict(mech.gas.reaction(1).third_body.efficiencies)

        changed = mech.recast_reaction_at_pressure(1, self.P, self.MIX)

        assert changed is True
        rxn_new = mech.gas.reaction(1)
        assert isinstance(rxn_new.rate, ct.ArrheniusRate)
        assert rxn_new.third_body is not None
        assert dict(rxn_new.third_body.efficiencies) == eff_orig, (
            "falloff efficiencies dropped or altered on recast"
        )

    def test_recast_leaves_other_reactions_unchanged(self, loaded_all_rate_types):
        mech = loaded_all_rate_types
        before = [type(r.rate) for r in mech.gas.reactions()]

        mech.recast_reaction_at_pressure(3, self.P, self.MIX)

        after = [type(r.rate) for r in mech.gas.reactions()]
        for idx, (b, a) in enumerate(zip(before, after)):
            if idx == 3:
                continue
            assert a is b, f"rxn {idx} type changed: {b.__name__} -> {a.__name__}"

    def test_revert_restores_original_rate_and_kf(self, loaded_all_rate_types):
        mech = loaded_all_rate_types
        mech.set_TPX(self.T, self.P, self.MIX)
        k_before = float(mech.gas.forward_rate_constants[3])

        mech.recast_reaction_at_pressure(3, self.P, self.MIX)
        reverted = mech.revert_reaction_recast(3)

        assert reverted is True
        assert mech.is_reaction_recast(3) is False
        assert isinstance(mech.gas.reaction(3).rate, ct.PlogRate)

        mech.set_TPX(self.T, self.P, self.MIX)
        k_after = float(mech.gas.forward_rate_constants[3])
        assert k_after == pytest.approx(k_before, rel=1e-12), (
            f"revert did not restore the original rate: {k_before:.6e} -> {k_after:.6e}"
        )

    def test_revert_without_recast_is_noop(self, loaded_all_rate_types):
        mech = loaded_all_rate_types
        reverted = mech.revert_reaction_recast(0)
        assert reverted is False


class TestUpdateMechCoefOpt:
    """``fit_fcn.update_mech_coef_opt`` applies an optimizer step to mech state."""

    ARR_RXN_IDX = 2

    @pytest.fixture
    def state(self, fresh_mech):
        fresh_mech.set_TPX(1500.0, 20_000.0, {"Kr": 0.96, "cC7H14": 0.04})
        return fresh_mech

    def _coef_opt(self):
        from frhodo.optimize.parameters import OptimizableCoefficient

        return [OptimizableCoefficient(
            rxn_idx=self.ARR_RXN_IDX,
            coef_name="pre_exponential_factor",
            coef_idx=0,
            coeffs_key=0,
            bnds_key="rate",
        )]

    def test_single_coef_change_propagates_to_rate(self, state):
        from frhodo.optimize.cost.fit_fcn import update_mech_coef_opt

        coef_opt = self._coef_opt()
        original_A = state.coeffs[self.ARR_RXN_IDX][0]["pre_exponential_factor"]

        update_mech_coef_opt(state, coef_opt, [2.0 * original_A])

        state.set_TPX(1500.0, 20_000.0, {"Kr": 0.96, "cC7H14": 0.04})
        actual_A = state.gas.reaction(self.ARR_RXN_IDX).rate.pre_exponential_factor
        assert actual_A == pytest.approx(2.0 * original_A, rel=1e-12), (
            f"update_mech_coef_opt did not propagate A: expected {2.0*original_A}, got {actual_A}"
        )

    def test_unchanged_coef_leaves_state_unchanged(self, state):
        """Idempotency check: passing the same value should be a no-op."""
        from frhodo.optimize.cost.fit_fcn import update_mech_coef_opt

        coef_opt = self._coef_opt()
        original_A = state.coeffs[self.ARR_RXN_IDX][0]["pre_exponential_factor"]

        update_mech_coef_opt(state, coef_opt, [original_A])

        state.set_TPX(1500.0, 20_000.0, {"Kr": 0.96, "cC7H14": 0.04})
        actual_A = state.gas.reaction(self.ARR_RXN_IDX).rate.pre_exponential_factor
        assert actual_A == pytest.approx(original_A, rel=1e-15)
