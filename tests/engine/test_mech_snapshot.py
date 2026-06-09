"""Per-rxn uncertainty + optimizable-state snapshot across mech reloads.

Covers :mod:`frhodo.simulation.mechanism.mech_snapshot` — the engine
side of the reload-preservation feature. Three outcomes per rxn under
test: no signature match, signature match with full restore, signature
match with partial restore (different rate type or coefficient
mismatch).
"""
import cantera as ct
import numpy as np
import pytest

from frhodo.optimize.parameters import OptimizableSetBuilder
from frhodo.simulation.mechanism.mech_snapshot import (
    capture_state,
    coeffs_equal,
    restore_state,
    rxn_signature,
    signatures_for_gas,
)


class TestRxnSignature:
    """Identity = reactants + products + reversibility. Coefficient
    values are excluded so a rxn matches across rate-type changes."""

    def test_identical_reactions_share_signature(self, loaded_all_rate_types):
        sig_a = rxn_signature(loaded_all_rate_types.gas.reaction(0))
        sig_b = rxn_signature(loaded_all_rate_types.gas.reaction(0))

        assert sig_a == sig_b

    def test_different_species_differ(self, loaded_all_rate_types):
        sig_0 = rxn_signature(loaded_all_rate_types.gas.reaction(0))
        sig_3 = rxn_signature(loaded_all_rate_types.gas.reaction(3))

        assert sig_0 != sig_3, "different species lists must produce different signatures"

    def test_signature_is_hashable(self, loaded_all_rate_types):
        sig = rxn_signature(loaded_all_rate_types.gas.reaction(0))

        assert {sig: True}[sig] is True


class TestSignaturesForGas:
    """``signatures_for_gas`` returns one entry per reaction; duplicates
    share a signature."""

    def test_length_matches_n_reactions(self, loaded_all_rate_types):
        sigs = signatures_for_gas(loaded_all_rate_types.gas)

        assert len(sigs) == loaded_all_rate_types.gas.n_reactions

    def test_duplicate_pair_shares_signature(self, loaded_all_rate_types):
        """The test mech contains two rxns sharing reactants + products
        (Lindemann at idx 1, SRI at idx 5). Without ``dupe_idx`` they
        share one signature; the snapshot stacks their states."""
        sigs = signatures_for_gas(loaded_all_rate_types.gas)

        assert sigs[1] == sigs[5]


class TestCoeffsEqual:
    @pytest.mark.parametrize("a, b", [
        (1.0, 1.0),
        ([1.0, 2.0], [1.0, 2.0]),
        ({"A": 1.0, "B": 2.0}, {"A": 1.0, "B": 2.0}),
        (np.array([1.0, 2.0]), np.array([1.0, 2.0])),
        (float("nan"), float("nan")),
    ])
    def test_equal_values(self, a, b):
        assert coeffs_equal(a, b), f"expected {a!r} == {b!r}"

    @pytest.mark.parametrize("a, b", [
        (1.0, 2.0),
        ([1.0, 2.0], [1.0, 2.0, 3.0]),
        ({"A": 1.0}, {"B": 1.0}),
        (np.array([1.0, 2.0]), np.array([1.0, 3.0])),
    ])
    def test_unequal_values(self, a, b):
        assert not coeffs_equal(a, b), f"expected {a!r} != {b!r}"

    def test_tolerance_applied(self):
        assert coeffs_equal(1.0, 1.0 + 1e-12, rtol=1e-9, atol=0.0)


class TestCaptureStateShape:
    def test_state_count_matches_n_reactions(self, loaded_all_rate_types):
        """Each snapshot value is a list; the total entries across all
        signatures equals the reaction count (duplicates stack)."""
        opt = OptimizableSetBuilder()
        snapshot = capture_state(loaded_all_rate_types, opt)

        total = sum(len(states) for states in snapshot.values())
        assert total == loaded_all_rate_types.gas.n_reactions

    def test_duplicate_signatures_stack(self, loaded_all_rate_types):
        opt = OptimizableSetBuilder()
        snapshot = capture_state(loaded_all_rate_types, opt)
        dup_sig = rxn_signature(loaded_all_rate_types.gas.reaction(1))

        assert len(snapshot[dup_sig]) == 2, (
            "rxns 1 and 5 share a signature; their states should stack"
        )

    def test_records_rate_unc_value_and_type(self, loaded_all_rate_types):
        opt = OptimizableSetBuilder()
        loaded_all_rate_types.rate_bnds[0]["value"] = 2.5
        loaded_all_rate_types.rate_bnds[0]["type"] = "%"

        snapshot = capture_state(loaded_all_rate_types, opt)
        sig = rxn_signature(loaded_all_rate_types.gas.reaction(0))

        assert snapshot[sig][0]["rate_unc"]["value"] == 2.5
        assert snapshot[sig][0]["rate_unc"]["type"] == "%"

    def test_records_optimizable_flags(self, loaded_all_rate_types):
        opt = OptimizableSetBuilder()
        opt.set_reaction_optimizable(0, True)
        opt.set_coefficient_optimizable(
            0, "rate", "activation_energy", True,
        )

        snapshot = capture_state(loaded_all_rate_types, opt)
        sig = rxn_signature(loaded_all_rate_types.gas.reaction(0))

        assert snapshot[sig][0]["rate_optimizable"] is True
        assert (
            snapshot[sig][0]["coef_state"][("rate", "activation_energy")][
                "optimizable"
            ]
            is True
        )


class TestRestoreStateExactMatch:
    """Re-snapshot then restore on the same mech instance — every value
    survives the round trip and no rxn is flagged as partial-match."""

    def test_rate_unc_round_trip(self, loaded_all_rate_types):
        opt = OptimizableSetBuilder()
        loaded_all_rate_types.rate_bnds[0]["value"] = 3.0
        loaded_all_rate_types.rate_bnds[0]["type"] = "F"

        snapshot = capture_state(loaded_all_rate_types, opt)
        loaded_all_rate_types.rate_bnds[0]["value"] = np.nan
        loaded_all_rate_types.rate_bnds[0]["type"] = "F"
        restored, partial = restore_state(
            loaded_all_rate_types, opt, snapshot,
        )

        assert 0 in restored
        assert partial == set()
        assert loaded_all_rate_types.rate_bnds[0]["value"] == 3.0

    def test_per_coef_unc_round_trip(self, loaded_all_rate_types):
        opt = OptimizableSetBuilder()
        loaded_all_rate_types.coeffs_bnds[0]["rate"]["activation_energy"][
            "value"
        ] = 1500.0
        loaded_all_rate_types.coeffs_bnds[0]["rate"]["activation_energy"][
            "type"
        ] = "±"

        snapshot = capture_state(loaded_all_rate_types, opt)
        loaded_all_rate_types.coeffs_bnds[0]["rate"]["activation_energy"][
            "value"
        ] = np.nan
        loaded_all_rate_types.coeffs_bnds[0]["rate"]["activation_energy"][
            "type"
        ] = "F"

        restored, partial = restore_state(
            loaded_all_rate_types, opt, snapshot,
        )

        assert 0 in restored
        assert partial == set()
        bnds = loaded_all_rate_types.coeffs_bnds[0]["rate"]["activation_energy"]
        assert bnds["value"] == 1500.0
        assert bnds["type"] == "±"

    def test_optimizable_flag_round_trip(self, loaded_all_rate_types):
        opt = OptimizableSetBuilder()
        opt.set_reaction_optimizable(0, True)
        snapshot = capture_state(loaded_all_rate_types, opt)
        opt.reset()

        restore_state(loaded_all_rate_types, opt, snapshot)

        assert opt.is_reaction_optimizable(0)


class TestRestoreStateNoMatch:
    def test_empty_snapshot_returns_empty_sets(self, loaded_all_rate_types):
        opt = OptimizableSetBuilder()
        restored, partial = restore_state(loaded_all_rate_types, opt, {})

        assert restored == set()
        assert partial == set()

    def test_none_snapshot_returns_empty_sets(self, loaded_all_rate_types):
        opt = OptimizableSetBuilder()
        restored, partial = restore_state(loaded_all_rate_types, opt, None)

        assert restored == set()
        assert partial == set()


class TestRestoreStateSigMatchCoefMismatch:
    """Signature matches but the coef structure/values differ.

    Partial fires only when the user had non-default state on a coef
    that doesn't transfer — a sister mech with slightly different
    resetVals on un-touched coefs is *not* partial because the user
    has no visible state to mourn."""

    def test_changed_resetval_no_user_state_not_partial(
        self, loaded_all_rate_types,
    ):
        """User only set rate-level state (the k box). Sister mech has
        a slightly different coef resetVal but the user didn't touch
        that coef — partial would be noise."""
        opt = OptimizableSetBuilder()
        loaded_all_rate_types.rate_bnds[0]["value"] = 2.0
        loaded_all_rate_types.rate_bnds[0]["type"] = "F"

        snapshot = capture_state(loaded_all_rate_types, opt)
        loaded_all_rate_types.coeffs_bnds[0]["rate"]["activation_energy"][
            "resetVal"
        ] = 9999.0

        restored, partial = restore_state(
            loaded_all_rate_types, opt, snapshot,
        )

        assert 0 in restored
        assert 0 not in partial, (
            "coef resetVal drift on an un-touched coef must not flag partial"
        )

    def test_changed_resetval_with_user_state_triggers_partial(
        self, loaded_all_rate_types,
    ):
        opt = OptimizableSetBuilder()
        opt.set_reaction_optimizable(0, True)
        loaded_all_rate_types.coeffs_bnds[0]["rate"]["activation_energy"][
            "value"
        ] = 100.0
        loaded_all_rate_types.coeffs_bnds[0]["rate"]["activation_energy"][
            "type"
        ] = "±"

        snapshot = capture_state(loaded_all_rate_types, opt)
        loaded_all_rate_types.coeffs_bnds[0]["rate"]["activation_energy"][
            "resetVal"
        ] = 9999.0

        restored, partial = restore_state(
            loaded_all_rate_types, opt, snapshot,
        )

        assert 0 in restored
        assert 0 in partial, (
            "user-touched coef with mismatched resetVal on an optimized "
            "rxn must flag partial"
        )

    def test_orphan_old_key_with_user_state_triggers_partial(
        self, loaded_all_rate_types,
    ):
        """User touched a coef on the old mech. The new mech's coef
        key space doesn't include it (e.g. rate type change) — partial."""
        opt = OptimizableSetBuilder()
        opt.set_reaction_optimizable(0, True)
        snapshot = capture_state(loaded_all_rate_types, opt)
        sig = list(snapshot.keys())[0]
        snapshot[sig][0]["coef_state"][("low_rate", "fake_coef")] = {
            "value": 1.5, "type": "±", "resetVal": 1.0, "optimizable": False,
        }

        restored, partial = restore_state(
            loaded_all_rate_types, opt, snapshot,
        )

        assert 0 in partial, (
            "orphan old key with user state on optimized rxn must flag partial"
        )

    def test_partial_skipped_when_rate_not_optimizable(
        self, loaded_all_rate_types,
    ):
        """If the user never marked the rate optimizable, the rxn isn't
        being optimized — coef mismatches are noise and partial must
        not fire."""
        opt = OptimizableSetBuilder()
        loaded_all_rate_types.coeffs_bnds[0]["rate"]["activation_energy"][
            "value"
        ] = 100.0
        loaded_all_rate_types.coeffs_bnds[0]["rate"]["activation_energy"][
            "type"
        ] = "±"

        snapshot = capture_state(loaded_all_rate_types, opt)
        loaded_all_rate_types.coeffs_bnds[0]["rate"]["activation_energy"][
            "resetVal"
        ] = 9999.0

        restored, partial = restore_state(
            loaded_all_rate_types, opt, snapshot,
        )

        assert 0 in restored
        assert 0 not in partial, (
            "non-optimized rxn must not be flagged partial even with "
            "coef mismatch"
        )

    def test_orphan_old_key_no_user_state_not_partial(
        self, loaded_all_rate_types,
    ):
        opt = OptimizableSetBuilder()
        snapshot = capture_state(loaded_all_rate_types, opt)
        sig = list(snapshot.keys())[0]
        snapshot[sig][0]["coef_state"][("low_rate", "fake_coef")] = {
            "value": np.nan, "type": "F", "resetVal": 1.0, "optimizable": False,
        }

        restored, partial = restore_state(
            loaded_all_rate_types, opt, snapshot,
        )

        assert 0 not in partial, (
            "orphan key with default state must not flag partial"
        )

    def test_optimizable_flag_alone_triggers_partial_on_orphan(
        self, loaded_all_rate_types,
    ):
        """Even without typed value/type, marking a coef optimizable
        is user state that should survive — orphan flags partial,
        provided the rate is also marked optimizable."""
        opt = OptimizableSetBuilder()
        opt.set_reaction_optimizable(0, True)
        snapshot = capture_state(loaded_all_rate_types, opt)
        sig = list(snapshot.keys())[0]
        snapshot[sig][0]["coef_state"][("low_rate", "fake_coef")] = {
            "value": float("nan"),
            "type": "F",
            "resetVal": 1.0,
            "optimizable": True,
        }

        restored, partial = restore_state(
            loaded_all_rate_types, opt, snapshot,
        )

        assert 0 in partial


class TestRestoreStateResetValStaysFromNewMech:
    """The user wants resetVal to track the *new* mech's coefficient
    value — restore must not write the prior resetVal back."""

    def test_resetval_not_overwritten(self, loaded_all_rate_types):
        opt = OptimizableSetBuilder()
        snapshot = capture_state(loaded_all_rate_types, opt)
        new_reset = 12345.0
        loaded_all_rate_types.coeffs_bnds[0]["rate"]["activation_energy"][
            "resetVal"
        ] = new_reset

        restore_state(loaded_all_rate_types, opt, snapshot)

        assert (
            loaded_all_rate_types.coeffs_bnds[0]["rate"]["activation_energy"][
                "resetVal"
            ]
            == new_reset
        )
