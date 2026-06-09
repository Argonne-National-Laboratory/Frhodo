"""Pure-function tests for the adaptive loss / robust statistics helpers."""
import numpy as np
import pytest

from frhodo._vendor.opendsm.adaptive_loss import (
    adaptive_weights,
    generalized_loss_derivative,
    generalized_loss_fcn,
    generalized_loss_weights,
    get_C,
)
from frhodo._vendor.opendsm.stats_basic import weighted_quantile


class TestGeneralizedLossFcn:
    """The generalized loss has closed forms at integer values of ``alpha``.

    See Barron 2019, "A General and Adaptive Robust Loss Function".
    """

    def test_alpha2_is_half_squared(self):
        """At alpha=2 the loss is the standard L2 loss: 0.5 * x^2."""
        x = np.array([0.0, 1.0, -2.0, 3.5])
        np.testing.assert_allclose(
            generalized_loss_fcn(x, alpha=2.0),
            0.5 * x**2,
            rtol=1e-12,
        )

    def test_alpha1_is_smoothed_l1(self):
        x = np.array([0.0, 1.0, -2.0, 3.5])
        np.testing.assert_allclose(
            generalized_loss_fcn(x, alpha=1.0),
            np.sqrt(x**2 + 1) - 1,
            rtol=1e-12,
        )

    def test_zero_input_yields_zero_loss(self):
        for alpha in [-2.0, 0.0, 1.0, 2.0]:
            assert generalized_loss_fcn(np.array([0.0]), alpha=alpha) == pytest.approx(0.0, abs=1e-15)

    def test_loss_is_symmetric_in_x(self):
        x = np.linspace(-5, 5, 21)
        for alpha in [-2.0, 0.0, 1.0, 2.0]:
            np.testing.assert_allclose(
                generalized_loss_fcn(x, alpha=alpha),
                generalized_loss_fcn(-x, alpha=alpha),
                rtol=1e-12,
                err_msg=f"asymmetry at alpha={alpha}",
            )

    def test_loss_is_monotone_increasing_in_abs_x(self):
        """For any fixed alpha, loss(|x|) is non-decreasing as |x| grows."""
        x = np.linspace(0, 5, 51)
        for alpha in [-2.0, 0.0, 1.0, 2.0]:
            loss = generalized_loss_fcn(x, alpha=alpha)
            diffs = np.diff(loss)
            assert (diffs >= -1e-12).all(), f"non-monotone at alpha={alpha}"


class TestWeightedQuantile:
    """Numba-jitted; arguments must be passed positionally as numpy arrays."""

    @pytest.mark.parametrize("q", [0.25, 0.5, 0.75])
    def test_uniform_weights_match_numpy_quantile(self, q):
        rng = np.random.default_rng(0)
        data = rng.standard_normal(1000)
        wq = weighted_quantile(data, np.array([q]))
        ref = np.quantile(data, q)
        assert wq[0] == pytest.approx(ref, rel=1e-2), (
            f"weighted vs unweighted quantile diverged at q={q}: "
            f"weighted={wq[0]}, numpy={ref}"
        )

    def test_zero_weight_excludes_point(self):
        """A sample with zero weight should not influence the median."""
        data = np.array([1.0, 2.0, 3.0, 1e9])
        weights = np.array([1.0, 1.0, 1.0, 0.0])
        median = weighted_quantile(data, np.array([0.5]), weights)[0]
        assert median == pytest.approx(2.0, abs=1.0), (
            f"zero-weight outlier still influenced result: median={median}"
        )


class TestGeneralizedLossDerivative:
    """At each special-case ``alpha``, the derivative must match the finite
    difference of ``generalized_loss_fcn`` at the same point."""

    @pytest.mark.parametrize("alpha", [-2.0, 0.0, 1.0, 2.0])
    @pytest.mark.parametrize("x", [-2.0, -0.5, 0.5, 2.0])
    def test_matches_finite_difference(self, alpha, x):
        h = 1e-6
        analytic = float(generalized_loss_derivative(np.array([x]), scale=1.0, alpha=alpha)[0])
        numeric = (
            generalized_loss_fcn(np.array([x + h]), alpha=alpha)[0]
            - generalized_loss_fcn(np.array([x - h]), alpha=alpha)[0]
        ) / (2 * h)
        assert analytic == pytest.approx(numeric, rel=1e-5, abs=1e-7), (
            f"derivative mismatch at alpha={alpha}, x={x}: "
            f"analytic={analytic:.6g}, finite-diff={numeric:.6g}"
        )

    def test_derivative_is_zero_at_zero(self):
        """All Barron loss variants are even, so f'(0) = 0."""
        for alpha in [-2.0, 0.0, 1.0, 2.0]:
            d = generalized_loss_derivative(np.array([0.0]), alpha=alpha)[0]
            assert d == pytest.approx(0.0, abs=1e-15), (
                f"derivative at x=0 should be zero for alpha={alpha}, got {d}"
            )


class TestGeneralizedLossWeights:
    """Weights are used for IRLS-style robust regression."""

    def test_unit_weight_at_zero_residual(self):
        for alpha in [-2.0, 0.0, 1.0, 2.0]:
            w = generalized_loss_weights(np.array([0.0]), alpha=alpha)[0]
            assert w == pytest.approx(1.0, abs=1e-12), (
                f"weight at x=0 must be 1 (no down-weighting) for alpha={alpha}, got {w}"
            )

    def test_weights_decrease_with_residual_magnitude_for_robust_alpha(self):
        """For ``alpha=0`` (Charbonnier), weights monotonically decrease as |x| grows."""
        x = np.linspace(0.0, 5.0, 11)
        w = generalized_loss_weights(x, alpha=0.0)
        assert (np.diff(w) <= 1e-12).all(), (
            f"Charbonnier weights should be non-increasing in |x|, got {w}"
        )

    def test_min_weight_floor_is_respected(self):
        x = np.array([100.0])
        w = generalized_loss_weights(x, alpha=0.0, min_weight=0.1)[0]
        assert w >= 0.1, f"min_weight=0.1 floor not enforced; got w={w}"


class TestGetC:
    """``get_C`` produces the loss scale parameter ``C`` from residuals."""

    def test_returns_positive_value_on_normal_data(self):
        rng = np.random.default_rng(42)
        resid = rng.standard_normal(500)
        C = get_C(resid, mu=0.0, sigma=3.0)
        assert C > 0, f"get_C should be positive on standard-normal residuals, got {C}"

    def test_zero_residuals_falls_through_to_OoM_floor(self):
        """When IQR collapses to zero (constant data), the function falls back
        to the OoM floor — must still return a non-NaN, finite scalar."""
        resid = np.zeros(50)
        C = get_C(resid, mu=0.0, sigma=3.0)
        assert np.isfinite(C), f"get_C returned non-finite value on zero data: {C}"

    @pytest.mark.parametrize("algo", ["iqr_legacy", "iqr", "mad", "stdev"])
    def test_weighted_path_returns_positive(self, algo):
        rng = np.random.default_rng(7)
        resid = rng.standard_normal(300)
        weights = np.full(300, 0.5)
        weights[:30] = 0.0
        C = get_C(resid, mu=0.0, sigma=3.0, algo=algo, weights=weights)
        assert C > 0 and np.isfinite(C), f"{algo!r} weighted C invalid: {C}"


class TestAdaptiveWeightsThreading:
    """``adaptive_weights`` should propagate ``weights`` through mu, C, alpha."""

    def test_C_scalar_post_multiplies_C(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal(500)
        _, C1, _ = adaptive_weights(x, alpha=2.0, C_scalar=1.0)
        _, C2, _ = adaptive_weights(x, alpha=2.0, C_scalar=3.0)
        assert C2 == pytest.approx(3.0 * C1, rel=1e-9), (
            f"C_scalar=3 did not triple C: C1={C1}, C2={C2}"
        )

    def test_zero_weights_exclude_outliers_from_C(self):
        """A massive outlier with weight=0 should not inflate C."""
        rng = np.random.default_rng(2)
        x = rng.standard_normal(500)
        x_outlier = x.copy()
        x_outlier[0] = 1e6
        weights = np.ones(500)
        weights[0] = 0.0
        _, C_clean, _ = adaptive_weights(x, alpha=2.0)
        _, C_masked, _ = adaptive_weights(x_outlier, weights=weights, alpha=2.0)
        # Masked C should be close to the clean C, not blown up by the outlier.
        assert C_masked == pytest.approx(C_clean, rel=0.5), (
            f"zero-weight outlier still influenced C: clean={C_clean}, masked={C_masked}"
        )

    def test_empty_weights_treated_as_unweighted(self):
        """Legacy callers passed `np.array([])` to mean 'no weights'."""
        rng = np.random.default_rng(3)
        x = rng.standard_normal(200)
        w_none, C_none, _ = adaptive_weights(x, alpha=2.0)
        w_empty, C_empty, _ = adaptive_weights(x, weights=np.array([]), alpha=2.0)
        np.testing.assert_allclose(w_none, w_empty, rtol=1e-12)
        assert C_none == pytest.approx(C_empty, rel=1e-12)
