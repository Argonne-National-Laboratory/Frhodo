"""Tests for the Tranter shock-experiment file loader.

The bundled ``example/experiment/shock1.exp`` is the v1 (Tranter) format:
ConfigParser-style ``[Mixture]`` and ``[Expt Params]`` sections. The
loader converts T1/P1/u1/P4 into Cantera units (K, Pa, m/s, Pa) on the
way out.

Snapshot reference values come from the bundled file. Update only on
intentional file changes.
"""
import pytest

from frhodo.common.units import Convert_Units
from frhodo.experiment import ExperimentLoader

SHOCK1_EXP_RAW = {
    "Kr": 0.96,
    "cC7H14": 0.04,
    # raw (display) values before unit conversion
    "P1_torr": 5.01,
    "T1_C": 21.00,
    "P4_psi": 30.00,
    "tOpt_us": 116.557292,
    "PT_spacing_mm": 120.0,
    "SampRate_Hz": 50_000_000.0,
}


@pytest.fixture
def convert_units(loaded_cycloheptane):
    """Build a ``Convert_Units`` instance over the real loaded mechanism."""
    return Convert_Units(loaded_cycloheptane)


@pytest.fixture
def loaded_shock1(convert_units, example_mech_dir):
    exp = ExperimentLoader(convert_units)
    exp_file = example_mech_dir.parent / "experiment" / "shock1.exp"
    return exp.parameters(exp_file)


class TestShock1Parameters:
    def test_mixture_composition(self, loaded_shock1):
        mix = loaded_shock1["exp_mix"]
        assert mix == {"Kr": 0.96, "cC7H14": 0.04}, (
            f"unexpected mixture: {mix}"
        )

    def test_thermo_mix_matches_exp_mix(self, loaded_shock1):
        assert loaded_shock1["exp_mix"] == loaded_shock1["thermo_mix"]

    def test_T1_converted_to_kelvin(self, loaded_shock1):
        assert loaded_shock1["T1"] == pytest.approx(294.15, abs=1e-6), (
            f"T1: 21°C should be 294.15 K, got {loaded_shock1['T1']}"
        )

    def test_P1_converted_to_pascals(self, loaded_shock1):
        # 5.01 torr * 133.32... Pa/torr ~ 668.0 Pa.
        # Frhodo's torr->Pa factor is the one in Convert_Units.conv2ct.
        assert loaded_shock1["P1"] == pytest.approx(667.945, rel=1e-3), (
            f"P1: 5.01 torr expected ~668 Pa, got {loaded_shock1['P1']}"
        )

    def test_u1_derived_from_PT_spacing_over_tOpt(self, loaded_shock1):
        # 120 mm / 116.557292 us in mm/us = 1.0295 mm/us; converted to m/s
        # by Convert_Units (mm/us -> m/s is *1000): expect ~1029.5 m/s.
        assert loaded_shock1["u1"] == pytest.approx(1029.54, rel=1e-3), (
            f"u1: expected ~1029.5 m/s, got {loaded_shock1['u1']}"
        )

    def test_P4_converted_to_pascals(self, loaded_shock1):
        # 30 psi * 6894.76 Pa/psi ~ 206843 Pa
        assert loaded_shock1["P4"] == pytest.approx(206_842.7, rel=1e-3), (
            f"P4: 30 psi expected ~206843 Pa, got {loaded_shock1['P4']}"
        )

    def test_sample_rate_in_megahertz(self, loaded_shock1):
        # The loader divides Hz by 1e6 (settings.py:419).
        assert loaded_shock1["Sample_Rate"] == pytest.approx(50.0, rel=1e-9), (
            f"Sample_Rate should be 50 MHz, got {loaded_shock1['Sample_Rate']}"
        )


class TestExperimentFileErrors:
    def test_zero_P1_raises(self, convert_units, tmp_path):
        bad = tmp_path / "bad.exp"
        bad.write_text(
            "[Mixture]\nMol_0_Formula=\"Ar\"\nMol_0_Mol frc=1.0\n"
            "[Expt Params]\nP1=0\nT1=300\nP4=10\nSampRate=1e6\n"
            "tOpt=100\nPT Spacing=100\n"
        )
        exp = ExperimentLoader(convert_units)
        with pytest.raises(Exception, match="P1 is zero"):
            exp.parameters(bad)

    def test_missing_files_log_warnings(self, convert_units, tmp_path, caplog):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        exp = ExperimentLoader(convert_units)
        with caplog.at_level("WARNING", logger="frhodo.experiment.parsers"):
            data = list(exp.load_data(99, empty_dir))
        warnings = [r for r in caplog.records if r.levelno >= 30]
        assert warnings, "missing shock files must log a WARNING"
        assert any("is missing" in r.getMessage() for r in warnings)
        assert all(d is not None for d in data)


class TestExpDataLoader:
    """``experiment.exp_data`` parses Shock*.rho (CSV: t [μs], ρ [g/cm^3])."""

    @pytest.fixture
    def loaded_rho(self, convert_units, example_mech_dir):
        exp = ExperimentLoader(convert_units)
        rho_file = example_mech_dir.parent / "experiment" / "shock1.rho"
        return exp.exp_data(rho_file)

    def test_returns_two_column_array(self, loaded_rho):
        import numpy as np
        arr = np.asarray(loaded_rho)
        assert arr.ndim == 2 and arr.shape[1] == 2, (
            f"expected (N, 2) array, got shape {arr.shape}"
        )

    def test_first_row_matches_file(self, loaded_rho):
        # First line of Shock1.rho: 1.493735E-1, 3.551242E-4
        import numpy as np
        np.testing.assert_allclose(
            np.asarray(loaded_rho)[0],
            [1.493735e-1, 3.551242e-4],
            rtol=1e-6,
        )

    def test_density_column_is_finite_and_varying(self, loaded_rho):
        import numpy as np
        rho_col = np.asarray(loaded_rho)[:, 1]
        assert np.isfinite(rho_col).all(), "density column has non-finite values"
        assert rho_col.std() > 0, "density column should not be constant"


class TestRawSignalLoader:
    """``experiment.raw_signal`` parses Shock*raw1.sig."""

    @pytest.fixture
    def loaded_sig(self, convert_units, example_mech_dir):
        exp = ExperimentLoader(convert_units)
        sig_file = example_mech_dir.parent / "experiment" / "shock1raw1.sig"
        return exp.raw_signal(sig_file)

    def test_returns_array(self, loaded_sig):
        import numpy as np
        arr = np.asarray(loaded_sig)
        assert arr.size > 0, "raw signal loader returned empty array"

    def test_signal_values_are_finite(self, loaded_sig):
        import numpy as np
        assert np.isfinite(np.asarray(loaded_sig)).all(), (
            "raw signal contains non-finite values"
        )


class TestTranterV0Parser:
    """Synthetic v0-format file (no [Mixture] section, no Expt Params)."""

    V0_CONTENT = (
        '"[Expt Parameters]"\n'
        '"[Thermochemistry]"\n'
        ' Ar;0.96\n'
        ' cC7H14;0.04\n'
        '\n'
        '"[Start Conditions]"\n'
        '0.0\n'        # ignored
        '5.01\n'       # P1
        '21.0\n'       # T1
        '0.0\n'        # ignored
        '116.557\n'    # final value used as time spacing
        '\n'
    )

    def test_parses_synthetic_v0_file(self, convert_units, tmp_path):
        v0_file = tmp_path / "v0.exp"
        v0_file.write_text(self.V0_CONTENT)
        exp = ExperimentLoader(convert_units)

        params = exp.parameters(v0_file)

        # P1, T1 go through unit conversion (torr->Pa, °C->K).
        # u1 is computed as 120 / time_spacing, then mm/μs -> m/s (×1000).
        assert exp.load_style == "tranter_v0_1", (
            f"v0 file should set load_style='tranter_v0_1', got {exp.load_style}"
        )
        assert "Ar" in params["exp_mix"]
        assert params["exp_mix"]["Ar"] == pytest.approx(0.96, rel=1e-9)
