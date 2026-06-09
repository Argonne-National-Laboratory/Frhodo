"""Chemkin -> Cantera mechanism load via ``ChemicalMechanism``.

Snapshot values pinned to ``example/mechanism/cycloheptane``.
"""
import collections

import cantera as ct
import numpy as np
import pytest

from frhodo.simulation.mechanism.mech_fcns import ChemicalMechanism, Uncertainty
from frhodo.simulation.mechanism.mechanism_loader import MechanismLoader

# Snapshot counts captured from current Cycloheptane.mech / .therm.
# Treat drift as a regression unless the example file itself was edited.
EXPECTED_N_SPECIES = 36
EXPECTED_N_REACTIONS = 66
EXPECTED_N_ELEMENTS = 5
EXPECTED_FIRST_SPECIES = "C8H18"
EXPECTED_LAST_SPECIES = "Ar"
EXPECTED_REACTION_TYPE_COUNTS = {
    "Arrhenius": 44,
    "three-body-Arrhenius": 7,
    "pressure-dependent-Arrhenius": 14,
    "falloff-Troe": 1,
}


class TestChemkinToCanteraLoad:
    def test_load_succeeds(self, cycloheptane_paths):
        mech = MechanismLoader().load(cycloheptane_paths)
        assert mech.isLoaded is True

    def test_writes_cantera_yaml(self, cycloheptane_paths):
        MechanismLoader().load(cycloheptane_paths)
        out = cycloheptane_paths["Cantera_Mech"]
        assert out.exists(), f"expected Cantera YAML at {out}"
        text = out.read_text()
        assert "phases:" in text and "species:" in text and "reactions:" in text

    def test_reports_species_and_reaction_counts_in_message(self, cycloheptane_paths):
        loader = MechanismLoader()
        loader.load(cycloheptane_paths)
        joined = "\n".join(loader.messages)
        assert f"{EXPECTED_N_SPECIES} species" in joined
        assert f"{EXPECTED_N_REACTIONS} reactions" in joined


class TestLoadIntoExistingMech:
    """``MechanismLoader.load(paths, mech=existing)`` populates the
    supplied instance in place so external holders of the reference
    (the GUI's ``convert_units``, etc.) see the loaded gas without
    needing to rebind."""

    def test_returns_same_instance_when_mech_supplied(self, cycloheptane_paths):
        from frhodo.simulation.mechanism import ChemicalMechanism

        target = ChemicalMechanism()
        result = MechanismLoader().load(cycloheptane_paths, mech=target)
        assert result is target

    def test_populates_existing_instance(self, cycloheptane_paths):
        from frhodo.simulation.mechanism import ChemicalMechanism

        target = ChemicalMechanism()
        assert not hasattr(target, "gas")
        MechanismLoader().load(cycloheptane_paths, mech=target)
        assert target.isLoaded is True
        assert target.gas is not None
        assert target.gas.n_species == EXPECTED_N_SPECIES

    def test_external_reference_sees_populated_state(self, cycloheptane_paths):
        """Mirrors the GUI bug: a sibling object captures the mech at
        construction (when it's empty) and must see the loaded gas
        afterwards without being told to refresh."""
        from frhodo.simulation.mechanism import ChemicalMechanism

        target = ChemicalMechanism()
        captured = target
        MechanismLoader().load(cycloheptane_paths, mech=target)
        assert captured.isLoaded is True
        assert captured.gas.n_reactions == EXPECTED_N_REACTIONS

    def test_reload_overwrites_existing_attrs(self, cycloheptane_paths, tmp_path):
        from frhodo.simulation.mechanism import ChemicalMechanism

        target = ChemicalMechanism()
        MechanismLoader().load(cycloheptane_paths, mech=target)
        first_gas = target.gas

        # Same paths but a fresh output location so the conversion runs again.
        second_paths = dict(cycloheptane_paths)
        second_paths["Cantera_Mech"] = tmp_path / "reload.yaml"
        MechanismLoader().load(second_paths, mech=target)

        assert target.isLoaded is True
        assert target.gas is not first_gas
        assert target.gas.n_species == EXPECTED_N_SPECIES


class TestLoadedGasStructure:
    """Inspect the resulting Cantera ``Solution`` object.

    Uses the module-scoped ``loaded_cycloheptane`` fixture so we parse
    the Chemkin file once per module, not once per test.
    """

    def test_species_count(self, loaded_cycloheptane):
        assert loaded_cycloheptane.gas.n_species == EXPECTED_N_SPECIES

    def test_reaction_count(self, loaded_cycloheptane):
        assert loaded_cycloheptane.gas.n_reactions == EXPECTED_N_REACTIONS

    def test_element_count(self, loaded_cycloheptane):
        assert loaded_cycloheptane.gas.n_elements == EXPECTED_N_ELEMENTS

    def test_first_and_last_species(self, loaded_cycloheptane):
        names = loaded_cycloheptane.gas.species_names
        assert names[0] == EXPECTED_FIRST_SPECIES
        assert names[-1] == EXPECTED_LAST_SPECIES

    def test_reaction_type_breakdown(self, loaded_cycloheptane):
        """Count of each ``reaction_type`` string."""
        counts = collections.Counter(
            r.reaction_type for r in loaded_cycloheptane.gas.reactions()
        )
        assert dict(counts) == EXPECTED_REACTION_TYPE_COUNTS, (
            f"reaction-type breakdown changed:\n"
            f"  expected: {EXPECTED_REACTION_TYPE_COUNTS}\n"
            f"  actual:   {dict(counts)}"
        )


class TestRateExpressionCoeffs:
    """Verify ``set_rate_expression_coeffs`` populates per-reaction state."""

    def test_coeffs_has_one_entry_per_reaction(self, loaded_cycloheptane):
        assert len(loaded_cycloheptane.coeffs) == EXPECTED_N_REACTIONS

    def test_reset_mech_classifies_each_reaction(self, loaded_cycloheptane):
        types = collections.Counter(
            r["rxnType"] for r in loaded_cycloheptane.reset_mech
        )
        # Mapping is reaction_type -> Frhodo's display name
        expected = {
            "Arrhenius Reaction": 44,
            "Three Body Reaction": 7,
            "Plog Reaction": 14,
            "Falloff Reaction": 1,
        }
        assert dict(types) == expected, (
            f"Frhodo reaction-type classifier diverged from Cantera: "
            f"expected {expected}, got {dict(types)}"
        )

    @pytest.mark.parametrize(
        "rxn_type,required_arrhenius_keys",
        [
            ("Arrhenius Reaction", {"activation_energy", "pre_exponential_factor", "temperature_exponent"}),
            ("Three Body Reaction", {"activation_energy", "pre_exponential_factor", "temperature_exponent"}),
        ],
    )
    def test_arrhenius_coeffs_have_three_parameters(
        self, loaded_cycloheptane, rxn_type, required_arrhenius_keys
    ):
        for idx, entry in enumerate(loaded_cycloheptane.reset_mech):
            if entry["rxnType"] != rxn_type:
                continue
            coeffs = loaded_cycloheptane.coeffs[idx][0]
            missing = required_arrhenius_keys - set(coeffs.keys())
            assert not missing, (
                f"reaction {idx} ({rxn_type}) missing keys {missing}; "
                f"present: {set(coeffs.keys())}"
            )


class TestSetTPX:
    """``set_TPX`` validates inputs before delegating to Cantera."""

    MIX = {"Kr": 0.96, "cC7H14": 0.04}

    def test_valid_inputs_apply_state_to_gas(self, loaded_cycloheptane):
        result = loaded_cycloheptane.set_TPX(1500.0, 20_000.0, self.MIX)

        assert result["success"] is True, f"unexpected error: {result['message']}"
        assert loaded_cycloheptane.gas.T == pytest.approx(1500.0)
        assert loaded_cycloheptane.gas.P == pytest.approx(20_000.0, rel=1e-6)

    def test_empty_X_sets_only_TP(self, loaded_cycloheptane):
        # Arrange: seed a known composition, then call set_TPX without X.
        loaded_cycloheptane.gas.TPX = 1000.0, 10_000.0, {"Kr": 1.0}

        result = loaded_cycloheptane.set_TPX(1500.0, 20_000.0)  # X defaults to []

        assert result["success"] is True
        assert loaded_cycloheptane.gas.T == pytest.approx(1500.0)
        kr_idx = loaded_cycloheptane.gas.species_index("Kr")
        assert loaded_cycloheptane.gas.X[kr_idx] == pytest.approx(1.0), (
            "empty X should leave the prior composition untouched"
        )

    @pytest.mark.parametrize("bad_T", [-1.0, 0.0, np.nan])
    def test_invalid_temperature_returns_error(self, loaded_cycloheptane, bad_T):
        result = loaded_cycloheptane.set_TPX(bad_T, 20_000.0, self.MIX)

        assert result["success"] is False, f"set_TPX accepted bad T={bad_T}"
        assert any("Temperature is invalid" in m for m in result["message"]), (
            f"expected 'Temperature is invalid' message, got: {result['message']}"
        )

    @pytest.mark.parametrize("bad_P", [-1.0, 0.0, np.nan])
    def test_invalid_pressure_returns_error(self, loaded_cycloheptane, bad_P):
        result = loaded_cycloheptane.set_TPX(1500.0, bad_P, self.MIX)

        assert result["success"] is False, f"set_TPX accepted bad P={bad_P}"
        assert any("Pressure is invalid" in m for m in result["message"])

    def test_unknown_species_returns_error_naming_species(self, loaded_cycloheptane):
        result = loaded_cycloheptane.set_TPX(1500.0, 20_000.0, {"Xx_FAKE": 1.0})

        assert result["success"] is False
        assert any("Xx_FAKE" in m for m in result["message"]), (
            f"error message should name the missing species; got: {result['message']}"
        )


class TestForwardRateConstantsAreInvariant:
    """``forward_rate_constants`` snapshot at T=1500K, P=20kPa, Kr/cC7H14 mix."""

    EXPECTED_SUM = 1.0149422651e12  # at T=1500K, P=20kPa, Kr/cC7H14 mix
    EXPECTED_KF_2 = 4.8496184420e10   # Arrhenius: H + cC7H14 <=> H2 + cC7H13
    EXPECTED_KF_36 = 156.79680515     # falloff: C2H6 (+M) <=> 2 CH3 (+M)
    EXPECTED_KF_42 = 3.4617012892e11  # three-body: C2H5 + H + M <=> C2H6 + M

    @pytest.fixture
    def state(self, loaded_cycloheptane):
        loaded_cycloheptane.gas.TPX = 1500.0, 20_000.0, {"Kr": 0.96, "cC7H14": 0.04}
        return loaded_cycloheptane

    def test_total_kf_sum_matches_snapshot(self, state):
        actual = float(np.sum(state.gas.forward_rate_constants))
        np.testing.assert_allclose(
            actual, self.EXPECTED_SUM, rtol=1e-9,
            err_msg="sum(forward_rate_constants) drift — chemistry has changed",
        )

    @pytest.mark.parametrize("idx,expected", [
        (2, EXPECTED_KF_2),
        (36, EXPECTED_KF_36),
        (42, EXPECTED_KF_42),
    ])
    def test_individual_kf_matches_snapshot(self, state, idx, expected):
        actual = float(state.gas.forward_rate_constants[idx])
        np.testing.assert_allclose(
            actual, expected, rtol=1e-9,
            err_msg=f"kf[{idx}] drift — chemistry of reaction {idx} has changed",
        )


class TestThirdBodyConcentration:
    """``ChemicalMechanism.M`` returns the third-body collision rate.

    For an Arrhenius (non-third-body) reaction the result is just
    ``density_mole``. For a three-body reaction the value is the same
    quantity weighted by efficiencies (snapshot-pinned).
    """

    @pytest.fixture
    def state(self, loaded_cycloheptane):
        loaded_cycloheptane.gas.TPX = 1500.0, 20_000.0, {"Kr": 0.96, "cC7H14": 0.04}
        return loaded_cycloheptane

    def test_arrhenius_reaction_falls_back_to_density_mole(self, state):
        # rxn 2 is plain Arrhenius — Cantera's third_body_concentrations[2]
        # is NaN, so M should fall back to gas.density_mole.
        assert state.M(2) == pytest.approx(state.gas.density_mole, rel=1e-12), (
            "non-third-body reactions should fall through to density_mole"
        )

    def test_three_body_reaction_matches_cantera_third_body_concentration(self, state):
        # rxn 42: C2H5 + H + M <=> C2H6 + M.
        ct_M = state.gas.third_body_concentrations[42]
        assert state.M(42) == pytest.approx(ct_M, rel=1e-12)
        assert state.M(42) == pytest.approx(1.603631400569e-3, rel=1e-9)

    def test_falloff_reaction_matches_cantera_third_body_concentration(self, state):
        # rxn 36 is C2H6 (+M) <=> 2 CH3 (+M) (falloff-Troe).
        ct_M = state.gas.third_body_concentrations[36]
        assert state.M(36) == pytest.approx(ct_M, rel=1e-12)

    def test_TPX_vector_mode_returns_array(self, state):
        # Vector mode iterates (T[i], P[i], X) and returns array of M values.
        T = np.array([1000.0, 1500.0, 2000.0])
        P = np.array([20_000.0, 20_000.0, 20_000.0])  # fixed P
        X = {"Kr": 0.96, "cC7H14": 0.04}
        result = state.M(42, [T, P, X])
        assert result.shape == (3,)
        assert np.isfinite(result).all()
        # At fixed P with rising T, density (and hence M) decreases (PV=nRT).
        assert result[0] > result[1] > result[2], (
            f"expected M to decrease with T at fixed P; got {result}"
        )


class TestChebyshevLoad:
    """Cycloheptane has zero Chebyshev reactions; covered by the
    ``loaded_chebyshev`` fixture (synthetic in-memory YAML)."""

    def test_two_reactions_total(self, loaded_chebyshev):
        assert loaded_chebyshev.gas.n_reactions == 2

    def test_first_reaction_is_arrhenius(self, loaded_chebyshev):
        assert loaded_chebyshev.gas.reaction(0).reaction_type == "Arrhenius"

    def test_second_reaction_is_chebyshev(self, loaded_chebyshev):
        rxn = loaded_chebyshev.gas.reaction(1)
        assert rxn.reaction_type == "Chebyshev"
        assert isinstance(rxn.rate, ct.ChebyshevRate)

    def test_chebyshev_classified_in_reset_mech(self, loaded_chebyshev):
        """``set_rate_expression_coeffs`` must label Chebyshev reactions
        with the ``"Chebyshev Reaction"`` rxnType so downstream code
        (GUI tree, optimizer) can dispatch on it."""
        types = [entry["rxnType"] for entry in loaded_chebyshev.reset_mech]
        cheb_types = [t for t in types if "Chebyshev" in t]
        assert len(cheb_types) == 1, (
            f"expected one Chebyshev classification in reset_mech; got {types}"
        )


class TestUncertainty:
    """``Uncertainty.__call__`` evaluates per-coefficient bound functions."""

    def _make_rate_unc(self, value, unc_type):
        rate_bnds = [{"value": value, "type": unc_type}]
        return Uncertainty("rate", 0, rate_bnds=rate_bnds)

    @pytest.mark.parametrize(
        "value,unc_type,x,expected",
        [
            (2.0, "F", 100.0, [50.0, 200.0]),       # factor: x/u, x*u
            (0.10, "%", 100.0, [100/1.10, 100*1.10]),  # percent
            (5.0, "±", 100.0, [95.0, 105.0]),       # +/-
        ],
    )
    def test_rate_bound_formulas(self, value, unc_type, x, expected):
        unc = self._make_rate_unc(value, unc_type)
        result = unc(x)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_nan_value_returns_nan_pair(self):
        unc = self._make_rate_unc(np.nan, "F")
        result = unc(100.0)
        assert np.isnan(result).all(), f"expected [nan, nan], got {result}"

    def test_falloff_parameters_key_returns_nan_pair(self):
        coeffs_bnds = [{"falloff_parameters": {"foo": {"resetVal": 1.0, "value": 1.0, "type": "F"}}}]
        unc = Uncertainty(
            "coef", 0, coeffs_bnds=coeffs_bnds, key="falloff_parameters", coef_name="foo",
        )
        assert np.isnan(unc(1.0)).all(), (
            "Uncertainty for falloff_parameters key should always return [nan, nan]"
        )


class TestOptimizerRatesAndBounds:
    """Tests for the ``optimize.misc_fcns`` helpers that drive the optimizer
    -- they read from a loaded mech and depend on its state."""

    def test_rates_returns_log_forward_rate_constants(self, loaded_cycloheptane):
        """``rates(rxn_coef_opt, mech)`` returns ln(k_f) at each (T, P) point."""
        from frhodo.simulation.mechanism.coef_helpers import rates

        T_pts = np.array([1500.0, 2000.0])
        P_pts = np.array([20_000.0, 20_000.0])
        rxn_coef_opt = [{"rxnIdx": 2, "T": T_pts, "P": P_pts}]
        loaded_cycloheptane.set_TPX(1500.0, 20_000.0, {"Kr": 0.96, "cC7H14": 0.04})

        result = rates(rxn_coef_opt, loaded_cycloheptane)

        # Result is ln(k); recompute expected by manually advancing TPX.
        expected = []
        for T in T_pts:
            loaded_cycloheptane.set_TPX(T, 20_000.0)
            expected.append(np.log(loaded_cycloheptane.gas.forward_rate_constants[2]))
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_set_bnds_returns_three_arrays_for_arrhenius(self, loaded_cycloheptane):
        """For an Arrhenius reaction with NaN limits, ``set_bnds`` returns
        a ``{lower, upper, exist}`` dict where each array has 3 elements."""
        from frhodo.simulation.mechanism.coef_helpers import set_bnds

        keys = [{"coeffs_bnds": "rate"}] * 3
        coefNames = [
            "activation_energy",
            "pre_exponential_factor",
            "temperature_exponent",
        ]
        bnds = set_bnds(loaded_cycloheptane, rxnIdx=2, keys=keys, coefNames=coefNames)

        assert set(bnds.keys()) == {"lower", "upper", "exist"}
        assert len(bnds["lower"]) == 3, f"expected 3 lower bounds, got {len(bnds['lower'])}"
        assert len(bnds["upper"]) == 3, f"expected 3 upper bounds, got {len(bnds['upper'])}"
        # exist is per-coef: [Ea_exists, A_exists, n_exists]; with default
        # NaN limits in the loaded mech, all should be [False, False].
        assert (bnds["exist"] == False).all(), (
            f"loaded mech has no explicit limits; 'exist' should be all False, got {bnds['exist']}"
        )

    def test_set_bnds_lower_below_upper(self, loaded_cycloheptane):
        from frhodo.simulation.mechanism.coef_helpers import set_bnds

        keys = [{"coeffs_bnds": "rate"}] * 3
        coefNames = [
            "activation_energy",
            "pre_exponential_factor",
            "temperature_exponent",
        ]
        bnds = set_bnds(loaded_cycloheptane, rxnIdx=2, keys=keys, coefNames=coefNames)
        assert (bnds["lower"] < bnds["upper"]).all(), (
            f"lower bound exceeds upper for some coef: "
            f"lower={bnds['lower']}, upper={bnds['upper']}"
        )
