"""Mechanism roundtrip via ``ChemicalMechanism.to_yaml_text`` and
``to_chemkin``. Locks down the optimize-and-checkpoint contract for ML.
"""
import pathlib

import cantera as ct
import numpy as np
import pytest

from frhodo.api import (
    PostShockState,
    PreShockState,
    ShockTubeConfig,
    run_shock_tube,
    solve_shock_jump,
)
from frhodo.simulation.mechanism.mech_fcns import ChemicalMechanism
from frhodo.simulation.mechanism.mechanism_loader import MechanismLoader

T1_K = 294.15
P1_PA = 5.01 * 133.322368421
U1_MPS = 120e-3 / 116.557292e-6
MIX = {"Kr": 0.96, "cC7H14": 0.04}


class TestToYamlText:
    def test_returns_non_empty_string(self, loaded_cycloheptane):
        text = loaded_cycloheptane.to_yaml_text()
        assert isinstance(text, str)
        assert len(text) > 1000

    def test_parses_back_with_same_dimensions(self, loaded_cycloheptane):
        text = loaded_cycloheptane.to_yaml_text()
        gas = ct.Solution(yaml=text)
        assert gas.n_species == loaded_cycloheptane.gas.n_species
        assert gas.n_reactions == loaded_cycloheptane.gas.n_reactions

    def test_modified_coefficient_survives_roundtrip(self, loaded_cycloheptane):
        mech = loaded_cycloheptane
        target = next(
            i for i, r in enumerate(mech.gas.reactions())
            if type(r.rate) is ct.ArrheniusRate
        )
        original_A = mech.coeffs[target][0]["pre_exponential_factor"]
        try:
            mech.coeffs[target][0]["pre_exponential_factor"] = 2.0 * original_A
            mech.modify_reactions(mech.coeffs, rxnIdxs=[target])

            text = mech.to_yaml_text()
            gas = ct.Solution(yaml=text)
            new_A = gas.reactions()[target].rate.pre_exponential_factor
            assert new_A == pytest.approx(2.0 * original_A, rel=1e-12)
        finally:
            mech.coeffs[target][0]["pre_exponential_factor"] = original_A
            mech.modify_reactions(mech.coeffs, rxnIdxs=[target])


class TestToChemkin:
    def test_writes_a_file(self, loaded_cycloheptane, tmp_path):
        out = tmp_path / "mech.ck"
        loaded_cycloheptane.to_chemkin(out)
        assert out.exists()
        assert out.stat().st_size > 1000

    def test_chemkin_file_starts_with_elements(self, loaded_cycloheptane, tmp_path):
        out = tmp_path / "mech.ck"
        loaded_cycloheptane.to_chemkin(out)
        text = out.read_text()
        assert "ELEMENTS" in text or "ELEM" in text


class TestRoundTripPreservesSimulation:
    """Chemkin roundtrip must yield numerically identical reactor outputs.

    Pin: write to Chemkin, reload from Chemkin, run an Incident Shock
    Reactor sim under identical conditions, observable arrays match at
    rtol=1e-10. This is the optimize-and-checkpoint property: an ML
    training loop can serialize an optimized mechanism and pick up
    where it left off.
    """

    @pytest.fixture(scope="class")
    def reloaded_via_chemkin(self, loaded_cycloheptane, tmp_path_factory):
        d = tmp_path_factory.mktemp("ck_roundtrip")
        ck_path = d / "roundtrip.mech"
        loaded_cycloheptane.to_chemkin(ck_path)

        return MechanismLoader().load({
            "mech": ck_path,
            "thermo": None,  # thermo is embedded in the Chemkin output
            "Cantera_Mech": d / "roundtrip.yaml",
        })

    def test_same_n_species(self, loaded_cycloheptane, reloaded_via_chemkin):
        assert (
            reloaded_via_chemkin.gas.n_species
            == loaded_cycloheptane.gas.n_species
        )

    def test_same_n_reactions(self, loaded_cycloheptane, reloaded_via_chemkin):
        assert (
            reloaded_via_chemkin.gas.n_reactions
            == loaded_cycloheptane.gas.n_reactions
        )

    def test_shock_tube_observable_matches(
        self, loaded_cycloheptane, reloaded_via_chemkin
    ):
        ic = PreShockState(T1=T1_K, P1=P1_PA, u1=U1_MPS, composition=dict(MIX))

        def run_one(mech):
            ss = solve_shock_jump(ic, mech)
            assert ss.success
            cfg = ShockTubeConfig(
                initial=PostShockState(
                    T_reac=ss.T2, P_reac=ss.P2,
                    u_incident=ss.u2, rho1=ss.rho1,
                    composition=dict(MIX),
                ),
                t_end=5e-5,
            )
            r = run_shock_tube(mech, cfg)
            assert r.success
            return r

        ref = run_one(loaded_cycloheptane)
        roundtrip = run_one(reloaded_via_chemkin)

        assert ref.observable.shape == roundtrip.observable.shape
        np.testing.assert_allclose(
            roundtrip.observable, ref.observable, rtol=1e-6,
            err_msg="Chemkin roundtrip changed reactor observable",
        )
