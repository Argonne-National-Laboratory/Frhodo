"""``run_zero_d`` end-to-end on a textbook H2/O2 ignition.

Captures snapshot temperatures, densities, and the OH peak under both
constant-volume and constant-pressure modes so a future reactor-class
swap (Mole vs Mass variants, integrator changes) can't silently drift
the result.
"""
import pathlib
import shutil

import cantera as ct
import numpy as np
import pytest

from frhodo.simulation import run_zero_d
from frhodo.simulation.mechanism.mechanism_loader import MechanismLoader


# H2/O2/AR diluted, autoignition ~75 microseconds at 1200 K / 1 atm.
T_REAC = 1200.0
P_REAC = 101325.0
MIX = {"H2": 2, "O2": 1, "AR": 7}
T_END = 2.0e-4


@pytest.fixture(scope="module")
def loaded_h2o2(tmp_path_factory):
    """Cantera's bundled h2o2 YAML reloaded through Frhodo's loader."""
    src = pathlib.Path(ct.__file__).parent / "data" / "h2o2.yaml"
    dst = tmp_path_factory.mktemp("h2o2") / "h2o2.yaml"
    shutil.copy(src, dst)

    return MechanismLoader(silent=True).load({"mech": dst, "Cantera_Mech": dst})


@pytest.fixture(scope="module")
def const_v_sim(loaded_h2o2):
    sim, det = run_zero_d(
        loaded_h2o2, "constant_volume", t_end=T_END,
        T_reac=T_REAC, P_reac=P_REAC, mix=MIX,
        solve_energy=True, frozen_comp=False, rtol=1e-9, atol=1e-13,
        t_lab_save=np.linspace(1e-7, T_END, 40),
    )
    assert det["success"], f"const-V failed: {det.get('message')}"

    return sim


@pytest.fixture(scope="module")
def const_p_sim(loaded_h2o2):
    sim, det = run_zero_d(
        loaded_h2o2, "constant_pressure", t_end=T_END,
        T_reac=T_REAC, P_reac=P_REAC, mix=MIX,
        solve_energy=True, frozen_comp=False, rtol=1e-9, atol=1e-13,
        t_lab_save=np.linspace(1e-7, T_END, 40),
    )
    assert det["success"], f"const-P failed: {det.get('message')}"

    return sim


class TestConstantVolumeIgnition:
    """Snapshot values at rtol=1e-3."""

    EXPECTED_T_INITIAL = 1200.0
    EXPECTED_T_FINAL = 2949.575
    EXPECTED_RHO = 0.3205887  # constant for closed const-V
    EXPECTED_OH_MAX = 1.23324e-2

    def test_initial_temperature(self, const_v_sim):
        np.testing.assert_allclose(
            const_v_sim.states.T[0], self.EXPECTED_T_INITIAL, rtol=1e-9,
        )

    def test_final_temperature(self, const_v_sim):
        np.testing.assert_allclose(
            const_v_sim.states.T[-1], self.EXPECTED_T_FINAL, rtol=1e-3,
        )

    def test_density_conserved(self, const_v_sim):
        rho = np.asarray(const_v_sim.states.density)
        # Closed const-V: density should be exactly conserved
        np.testing.assert_allclose(rho, self.EXPECTED_RHO, rtol=1e-6)

    def test_OH_peak(self, const_v_sim):
        OH = np.asarray(const_v_sim.states("OH").Y).ravel()
        np.testing.assert_allclose(OH.max(), self.EXPECTED_OH_MAX, rtol=1e-3)


class TestConstantPressureIgnition:
    """Snapshot values at rtol=1e-3."""

    EXPECTED_T_INITIAL = 1200.0
    EXPECTED_T_FINAL = 2402.597
    EXPECTED_RHO_INITIAL = 0.3205887
    EXPECTED_RHO_FINAL = 0.170338
    EXPECTED_OH_MAX = 9.3027e-3

    def test_initial_temperature(self, const_p_sim):
        np.testing.assert_allclose(
            const_p_sim.states.T[0], self.EXPECTED_T_INITIAL, rtol=1e-9,
        )

    def test_final_temperature(self, const_p_sim):
        np.testing.assert_allclose(
            const_p_sim.states.T[-1], self.EXPECTED_T_FINAL, rtol=1e-3,
        )

    def test_initial_density(self, const_p_sim):
        np.testing.assert_allclose(
            const_p_sim.states.density[0], self.EXPECTED_RHO_INITIAL, rtol=1e-3,
        )

    def test_final_density_drops(self, const_p_sim):
        np.testing.assert_allclose(
            const_p_sim.states.density[-1], self.EXPECTED_RHO_FINAL, rtol=1e-3,
        )

    def test_OH_peak(self, const_p_sim):
        OH = np.asarray(const_p_sim.states("OH").Y).ravel()
        np.testing.assert_allclose(OH.max(), self.EXPECTED_OH_MAX, rtol=1e-3)


class TestFrozenChemistry:
    """``frozen_comp=True`` should hold composition fixed."""

    def test_const_v_frozen_keeps_initial_composition(self, loaded_h2o2):
        sim, det = run_zero_d(
            loaded_h2o2, "constant_volume", t_end=T_END,
            T_reac=T_REAC, P_reac=P_REAC, mix=MIX,
            solve_energy=False, frozen_comp=True, rtol=1e-9, atol=1e-13,
            t_lab_save=np.array([T_END]),
        )
        assert det["success"]
        Y_init = np.asarray(sim.states.Y[0])
        Y_final = np.asarray(sim.states.Y[-1])
        np.testing.assert_allclose(Y_final, Y_init, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(
            sim.states.T[-1], sim.states.T[0], rtol=1e-9,
        )
