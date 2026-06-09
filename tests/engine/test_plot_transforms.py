"""Tests for the matplotlib transform classes used by Frhodo's custom scales.

The matplotlib ``Scale`` wrappers (``BiSymmetricLogScale``, ``AbsoluteLogScale``)
need a real axis context, but the inner ``Transform`` subclasses are
standalone — they implement ``transform_non_affine(x) -> x'`` against
plain numpy arrays. These transforms back the y-axis on the signal plot,
so a regression here visibly distorts user-facing graphs.
"""
import numpy as np
import pytest

from frhodo.gui.plots.custom_mplscale import AbsoluteLogScale, BiSymmetricLogScale


class TestAbsLogTransform:
    """log10(|x|) — the abs-log y-scale used for signed signals."""

    @pytest.fixture
    def transform(self):
        return AbsoluteLogScale.AbsLogTransform()

    def test_positive_value_matches_log10_abs(self, transform):
        x = np.array([1.0, 10.0, 100.0])
        np.testing.assert_allclose(transform.transform_non_affine(x), [0.0, 1.0, 2.0])

    def test_negative_input_is_treated_as_absolute(self, transform):
        """abslog of a sign-flipped input must equal abslog of the original."""
        x = np.array([3.0, 50.0, 1234.0])
        np.testing.assert_allclose(
            transform.transform_non_affine(-x),
            transform.transform_non_affine(x),
            rtol=1e-12,
        )

    def test_round_trip_to_inverted_then_back(self, transform):
        x = np.array([0.1, 1.0, 100.0, 10000.0])
        inv = transform.inverted()
        np.testing.assert_allclose(
            inv.transform_non_affine(transform.transform_non_affine(x)),
            x, rtol=1e-12,
        )


class TestBiSymLogTransform:
    """Symmetric log: linear near zero, log-like at large |x|. Used by
    matplotlib axes via ``BiSymmetricLogScale``; instantiated directly here."""

    C = 1.0

    @pytest.fixture
    def transform(self):
        return BiSymmetricLogScale.BiSymLogTransform(C=self.C)

    def test_zero_maps_to_zero(self, transform):
        result = transform.transform_non_affine(np.array([0.0]))
        assert result[0] == pytest.approx(0.0, abs=1e-15)

    def test_sign_preservation(self, transform):
        x = np.array([0.5, 5.0, 500.0])
        np.testing.assert_allclose(
            transform.transform_non_affine(-x),
            -transform.transform_non_affine(x),
            rtol=1e-12,
        )

    def test_round_trip_recovers_input(self, transform):
        x = np.array([-100.0, -1.5, -0.1, 0.1, 1.5, 100.0])
        inv = transform.inverted()
        np.testing.assert_allclose(
            inv.transform_non_affine(transform.transform_non_affine(x)),
            x, rtol=1e-12,
        )

    def test_inverted_transform_is_self_inverse_of_forward(self, transform):
        """Calling ``.inverted()`` twice on the forward transform yields a
        forward-equivalent (round-trip stays the identity)."""
        x = np.array([2.0, 20.0, 200.0])
        twice_inverted = transform.inverted().inverted()
        np.testing.assert_allclose(
            twice_inverted.transform_non_affine(x),
            transform.transform_non_affine(x),
            rtol=1e-12,
        )
