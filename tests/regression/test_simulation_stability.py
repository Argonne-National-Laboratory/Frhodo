"""Level-1 single-call deterministic regression — accuracy gate.

Fixed ``ShockTubeConfig`` against h2o2; output arrays compared against
committed reference values. Tolerances pin the math at IEEE-reorder
noise level so refactors that change algorithm behavior fail.
"""
import pathlib

import cantera as ct
import numpy as np
import pytest

from frhodo.api import PostShockState, ShockTubeConfig, SolverSettings, run_shock_tube
from frhodo.simulation.mechanism import ChemicalMechanism


FIXTURES = pathlib.Path(__file__).parent / "fixtures"
REFERENCE_NPZ = FIXTURES / "h2o2_shock_tube_reference.npz"

T_REAC = 1500.0
P_REAC = 20_000.0
COMPOSITION = {"H2": 0.04, "O2": 0.02, "AR": 0.94}
U_INCIDENT = 1029.0
RHO1 = 0.05
T_END = 5e-5


@pytest.fixture(scope="module")
def h2o2_mech():
    mech = ChemicalMechanism()
    mech.gas = ct.Solution("h2o2.yaml")
    mech.set_rate_expression_coeffs()
    mech.set_thermo_expression_coeffs()
    mech.isLoaded = True
    return mech


@pytest.fixture(scope="module")
def reference_result(h2o2_mech):
    cfg = ShockTubeConfig(
        initial=PostShockState(
            T_reac=T_REAC, P_reac=P_REAC,
            u_incident=U_INCIDENT, rho1=RHO1,
            composition=COMPOSITION,
        ),
        t_end=T_END,
        solver=SolverSettings(solver="BDF"),
    )
    result = run_shock_tube(h2o2_mech, cfg)
    assert result.success, f"reference run failed: {result.message}"

    return result


def _capture_reference(result):
    FIXTURES.mkdir(parents=True, exist_ok=True)
    np.savez(
        REFERENCE_NPZ,
        t=result.t, T=result.T, P=result.P, rho=result.rho,
        Y=result.Y, observable=result.observable,
        species=np.array(result.species),
    )


class TestH2O2ShockTubeReference:
    def test_reference_fixture_exists(self, reference_result):
        if not REFERENCE_NPZ.exists():
            _capture_reference(reference_result)
            pytest.fail(
                f"reference fixture was missing; captured to {REFERENCE_NPZ}. "
                "Re-run to verify."
            )

    def test_t_grid_matches_reference(self, reference_result):
        ref = np.load(REFERENCE_NPZ, allow_pickle=False)
        np.testing.assert_array_equal(reference_result.t, ref["t"])

    def test_observable_matches_reference(self, reference_result):
        ref = np.load(REFERENCE_NPZ, allow_pickle=False)
        np.testing.assert_allclose(
            reference_result.observable, ref["observable"],
            rtol=1e-10, atol=1e-12,
        )

    def test_T_matches_reference(self, reference_result):
        ref = np.load(REFERENCE_NPZ, allow_pickle=False)
        np.testing.assert_allclose(
            reference_result.T, ref["T"], rtol=1e-10, atol=1e-12,
        )

    def test_P_matches_reference(self, reference_result):
        ref = np.load(REFERENCE_NPZ, allow_pickle=False)
        np.testing.assert_allclose(
            reference_result.P, ref["P"], rtol=1e-10, atol=1e-12,
        )

    def test_rho_matches_reference(self, reference_result):
        ref = np.load(REFERENCE_NPZ, allow_pickle=False)
        np.testing.assert_allclose(
            reference_result.rho, ref["rho"], rtol=1e-10, atol=1e-12,
        )

    def test_Y_matches_reference(self, reference_result):
        ref = np.load(REFERENCE_NPZ, allow_pickle=False)
        np.testing.assert_allclose(
            reference_result.Y, ref["Y"], rtol=1e-10, atol=1e-15,
        )

    def test_species_set_unchanged(self, reference_result):
        ref = np.load(REFERENCE_NPZ, allow_pickle=False)
        ref_species = tuple(s.item() for s in ref["species"])
        assert reference_result.species == ref_species
