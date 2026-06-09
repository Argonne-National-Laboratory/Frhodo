"""Pure-function tests for unit and order-of-magnitude helpers."""
import math

import numpy as np
import pytest

from frhodo.common.units import Bisymlog, OoM, RoundToSigFigs


class TestOoM:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (1.0, 0),
            (10.0, 1),
            (999.0, 2),  # below 10^3
            (1000.0, 3),
            (0.1, -1),
            (0.001, -3),
            (-1234.0, 3),  # sign-independent
        ],
    )
    def test_powers_of_ten(self, value, expected):
        assert OoM(value) == expected

    def test_zero_returns_zero(self):
        assert OoM(0.0) == 0

    def test_array_input_returns_array(self):
        result = OoM(np.array([1.0, 100.0, 0.01]))
        assert np.array_equal(result, np.array([0, 2, -2]))


class TestRoundToSigFigs:
    @pytest.mark.parametrize(
        "value,sig_figs,expected",
        [
            (1234.5678, 3, 1230.0),
            (1234.5678, 5, 1234.6),
            (0.0012345, 2, 0.0012),
            (-987.65, 2, -990.0),
            (0.0, 3, 0.0),
        ],
    )
    def test_known_values(self, value, sig_figs, expected):
        result = float(RoundToSigFigs(value, sig_figs))
        assert math.isclose(result, expected, rel_tol=0, abs_tol=1e-10), (
            f"RoundToSigFigs({value}, {sig_figs}) = {result}, expected {expected}"
        )

    def test_array_input(self):
        result = RoundToSigFigs(np.array([1234.0, 0.005678]), 2)
        np.testing.assert_allclose(result, np.array([1200.0, 0.0057]), rtol=1e-10)


class TestBisymlog:
    """Symmetric log transform used by the bisymlog matplotlib scale.

    ``transform(y)`` maps real-valued y to a log-like scale that is
    sign-preserving and finite at zero. ``invTransform`` reverses it
    exactly within numerical precision.
    """

    @pytest.fixture
    def bisymlog(self):
        return Bisymlog(C=1.0)

    def test_zero_maps_to_zero(self, bisymlog):
        result = bisymlog.transform(np.array([0.0]))
        assert result[0] == pytest.approx(0.0, abs=1e-15)

    def test_transform_is_sign_preserving(self, bisymlog):
        """f(-y) = -f(y) for symmetric log scales."""
        y = np.array([0.5, 1.0, 5.0, 100.0])
        np.testing.assert_allclose(
            bisymlog.transform(-y), -bisymlog.transform(y),
            rtol=1e-12,
            err_msg="bisymlog transform broke sign-preservation",
        )

    def test_round_trip_recovers_input(self, bisymlog):
        """transform → invTransform must reproduce the original values."""
        y = np.array([-1000.0, -1.5, -0.5, 0.5, 1.5, 1000.0])
        recovered = bisymlog.invTransform(bisymlog.transform(y))
        np.testing.assert_allclose(recovered, y, rtol=1e-12)

    def test_nan_input_yields_nan_output(self, bisymlog):
        y = np.array([1.0, np.nan, 2.0])
        result = bisymlog.transform(y)
        assert np.isnan(result[1]), f"nan input should give nan output, got {result[1]}"
        assert np.isfinite(result[0]) and np.isfinite(result[2])

    def test_set_C_heuristically_on_constant_data_returns_fallback(self):
        """When max == min the heuristic falls back to 1/log(1000) and leaves C=None."""
        bs = Bisymlog()
        result = bs.set_C_heuristically(np.array([2.5, 2.5, 2.5]))
        assert result == pytest.approx(1.0 / np.log(1000), rel=1e-12)
        assert bs.C is None, "constant-data path must not set self.C"

    def test_invTransform_without_C_raises(self):
        """``invTransform`` requires C; a clear exception is the contract."""
        bs = Bisymlog()  # C left unset
        with pytest.raises(Exception, match="C is unspecified"):
            bs.invTransform(np.array([0.5]))

    def test_int_input_returns_finite_floats(self, bisymlog):
        """matplotlib's mouse-coord pipeline can hand integer-typed pixel
        arrays to the inverse transform. Without a float-cast on the
        result buffer, np.zeros_like(int) + np.nan assignment crashed
        with ``ValueError: cannot convert float NaN to integer``."""
        result = bisymlog.transform(np.array([-1, 0, 1, 5]))
        assert result.dtype == np.float64
        assert np.isfinite(result).all()

        recovered = bisymlog.invTransform(result.astype(int))
        assert recovered.dtype == np.float64
        assert np.isfinite(recovered).all()

    def test_list_input_is_promoted_to_array(self, bisymlog):
        """Plain-list inputs must work; ``y[idx]`` would otherwise fail."""
        result = bisymlog.transform([0.5, 1.0, 2.0])
        assert result.shape == (3,)
        assert np.isfinite(result).all()

    @pytest.mark.parametrize("y_t", [-3.0, -0.5, 0.5, 3.0])
    def test_invTransform_derivative_matches_central_fd(self, bisymlog, y_t):
        """Analytic derivative of ``invTransform`` agrees with central FD.

        ``y_t = 0`` is skipped: the second derivative has a sign jump
        there, so central FD picks up an O(h) error that swamps the
        agreement we're testing.  Continuity at zero is covered by
        :meth:`test_invTransform_derivative_continuous_at_zero`.
        """
        h = 1e-6
        fd = (
            bisymlog.invTransform(np.array([y_t + h]))[0]
            - bisymlog.invTransform(np.array([y_t - h]))[0]
        ) / (2 * h)
        analytic = float(bisymlog.invTransform_derivative(y_t))
        assert analytic == pytest.approx(fd, rel=1e-6, abs=1e-9)

    def test_invTransform_derivative_continuous_at_zero(self, bisymlog):
        """``invTransform`` is C^1 through zero; derivative must not jump."""
        left = float(bisymlog.invTransform_derivative(-1e-12))
        right = float(bisymlog.invTransform_derivative(1e-12))
        at_zero = float(bisymlog.invTransform_derivative(0.0))
        assert left == pytest.approx(at_zero, rel=1e-9)
        assert right == pytest.approx(at_zero, rel=1e-9)

    def test_invTransform_derivative_without_C_raises(self):
        bs = Bisymlog()
        with pytest.raises(Exception, match="C is unspecified"):
            bs.invTransform_derivative(0.5)


class TestConvertUnitsArrhenius:
    """Convert_Units depends only on ``mech`` (not a GUI parent)."""

    def test_basic_temperature_conversion(self, loaded_cycloheptane):
        from frhodo.common.units import Convert_Units

        conv = Convert_Units(loaded_cycloheptane)
        assert conv(0.0, "°C") == pytest.approx(273.15)
        assert conv(273.15, "°C", "out") == pytest.approx(0.0)

    def test_arrhenius_reaches_mech_via_self_mech(self, loaded_cycloheptane):
        """``_arrhenius`` looks up the reaction off ``self.mech.gas`` directly."""
        from frhodo.common.units import Convert_Units

        conv = Convert_Units(loaded_cycloheptane)
        coeffs = [["activation_energy", "Ea", 4184.0]]
        result = conv._arrhenius(0, coeffs, "Cantera2Chemkin")
        # Cantera2Chemkin Ea divides by 4.184e3 (J/cal): 4184 → 1.0 cal/mol.
        assert result[0][2] == pytest.approx(1.0)
