"""Wavelet-based pointwise noise estimator.

Pins the contract of :func:`frhodo.experiment.uncertainty.estimate_pointwise_sigma`:
- accurate σ(t) recovery on white-Gaussian noise across magnitudes,
- robustness to sharp signal features (real edges live in the
  SUREShrink-denoised signal, not the σ estimate),
- heteroscedastic noise detection (σ tracks slow trends),
- works in the optimizer's residual scale (Linear / Log / Bisymlog)
  so the σ feeding CheKiPEUQ matches the cost function's likelihood,
- ``bounds_from_sigma`` round-trips through the scale transform.
"""
import numpy as np
import pytest

from frhodo.common.scale import Scale
from frhodo.experiment.uncertainty import (
    bounds_from_sigma,
    estimate_pointwise_sigma,
    smooth_centerline,
)


LINEAR = Scale("Linear")


@pytest.fixture
def signal_with_transition():
    """Smooth signal + sharp transition — exercises signal-bleed-resistance."""
    t = np.linspace(0.0, 1.0, 4096)
    base = np.tanh(20.0 * (t - 0.3))

    return t, base


class TestEstimatePointwiseSigma:
    @pytest.mark.parametrize("true_sigma", [0.005, 0.05, 0.5])
    def test_recovers_homoscedastic_white_noise(self, true_sigma):
        """On a smooth signal + homoscedastic noise, σ̂ recovers the noise
        std (the centerline tracks the smooth trend, residual = noise)."""
        n = 4096
        t = np.linspace(0.0, 1.0, n)
        base = np.exp(-2.0 * t)  # smooth, no sharp feature in the noise
        rng = np.random.default_rng(42)
        y = base + rng.normal(0.0, true_sigma, n)

        est = estimate_pointwise_sigma(y, scale=LINEAR)

        assert est.shape == y.shape
        assert est.mean() == pytest.approx(true_sigma, rel=0.25), (
            f"mean σ estimate should be within 25% of true σ; got {est.mean():.4g} "
            f"vs {true_sigma:.4g}"
        )

    def test_no_inflation_at_clean_sharp_feature(self):
        """A sharp feature at low noise (the shock-arrival regime): the
        σ-adaptive bandwidth shrinks where the data is clean, so the
        center tracks the drop and σ is *not* inflated there — it stays
        below the noisy-tail σ."""
        n = 4096
        t = np.linspace(0.0, 1.0, n)
        base = np.exp(-t / 0.03)  # sharp clean drop at the start
        sigma_t = 0.004 + 0.05 * t  # clean at the start, noisy in the tail
        rng = np.random.default_rng(0)
        y = base + sigma_t * rng.standard_normal(n)

        est = estimate_pointwise_sigma(y, scale=LINEAR)

        start_sigma = float(est[:200].max())
        tail_sigma = float(est[-500:].mean())
        assert start_sigma < tail_sigma, (
            f"σ at the clean sharp start ({start_sigma:.3g}) should stay below "
            f"the noisy-tail σ ({tail_sigma:.3g}) — no inflation at the feature"
        )

    def test_tracks_heteroscedastic_growth(self):
        """If real σ grows linearly across the trace, the estimator
        should pick that up — its whole reason for being per-point."""
        n = 4096
        t = np.linspace(0.0, 1.0, n)
        rng = np.random.default_rng(1)
        sigma_true = np.linspace(0.01, 0.1, n)
        y = np.zeros(n) + sigma_true * rng.standard_normal(n)

        est = estimate_pointwise_sigma(y, scale=LINEAR)

        early = est[100:500].mean()
        late = est[-500:-100].mean()
        assert late > 3.0 * early, (
            f"heteroscedastic σ should rise; early={early:.3g}, late={late:.3g}"
        )

    def test_handles_odd_length(self):
        rng = np.random.default_rng(1)
        y = rng.normal(0.0, 0.1, 4095)

        est = estimate_pointwise_sigma(y, scale=LINEAR)

        assert est.shape == y.shape
        assert est.mean() == pytest.approx(0.1, rel=0.2)

    def test_returns_nan_for_too_short(self):
        est = estimate_pointwise_sigma(np.array([0.0, 1.0, 2.0]), scale=LINEAR)

        assert est.shape == (3,)
        assert np.isnan(est).all()

    def test_sigma_is_continuous(self):
        """σ̂(t) is smooth: jump-to-neighbor variation is small relative to
        its magnitude (no jagged band). Smooth signal + homoscedastic noise."""
        n = 4096
        t = np.linspace(0.0, 1.0, n)
        base = np.exp(-2.0 * t)
        rng = np.random.default_rng(7)
        y = base + rng.normal(0.0, 0.02, n)

        est = estimate_pointwise_sigma(y, scale=LINEAR)

        jumps = np.abs(np.diff(est))
        median_sigma = np.median(est)
        assert jumps.max() < 0.5 * median_sigma, (
            f"σ̂(t) jagged: max jump {jumps.max():.4g} vs median σ {median_sigma:.4g}"
        )


class TestSmoothCenterline:
    def test_tracks_clean_signal(self):
        """On clean data the σ-adaptive centerline follows the signal to
        within the local-linear smoother's curvature bias (a few % of
        range at the steepest point — it smooths, it doesn't interpolate)."""
        n = 1024
        t = np.linspace(0.0, 1.0, n)
        y = np.exp(-3.0 * t)

        mu_scaled = smooth_centerline(y, scale=LINEAR)

        assert np.max(np.abs(mu_scaled - y)) < 0.03 * (y.max() - y.min())

    def test_smooths_noisy_signal(self):
        """Heavy noise → centerline is smoother than the raw signal."""
        n = 1024
        rng = np.random.default_rng(0)
        t = np.linspace(0.0, 1.0, n)
        base = np.exp(-3.0 * t)
        y = base + rng.normal(0.0, 0.1, n)

        mu_scaled = smooth_centerline(y, scale=LINEAR)

        assert np.std(np.diff(mu_scaled)) < 0.5 * np.std(np.diff(y))

    def test_short_input_returns_forward_y(self):
        y = np.array([1.0, 2.0, 3.0])

        mu = smooth_centerline(y, scale=LINEAR)

        np.testing.assert_array_equal(mu, y)


class TestBoundsFromSigma:
    def test_shape_is_N_by_2(self):
        y = np.linspace(0.0, 1.0, 32)
        sigma = np.full_like(y, 0.05)

        bounds = bounds_from_sigma(y, sigma, sigma_multiple=3.0, scale=LINEAR)

        assert bounds.shape == (32, 2)

    def test_linear_scale_symmetric(self):
        y = np.array([1.0, 2.0, 3.0])
        sigma = np.array([0.1, 0.2, 0.3])

        bounds = bounds_from_sigma(y, sigma, sigma_multiple=2.0, scale=LINEAR)

        assert bounds[1, 0] == pytest.approx(2.0 - 2.0 * 0.2)
        assert bounds[1, 1] == pytest.approx(2.0 + 2.0 * 0.2)

    def test_log_scale_round_trips(self):
        """``bounds_from_sigma`` in Log space must satisfy
        ``log10(upper) - log10(y) == k·σ`` so the cost function's own
        log-transform on the bounds recovers the same σ CheKiPEUQ
        receives via ``sigma_multiple``."""
        y = np.array([1.0, 10.0, 100.0])
        sigma = np.array([0.05, 0.05, 0.05])
        log_scale = Scale("Log")

        bounds = bounds_from_sigma(y, sigma, sigma_multiple=2.0, scale=log_scale)

        log_upper_delta = np.log10(bounds[:, 1]) - np.log10(y)
        log_lower_delta = np.log10(y) - np.log10(bounds[:, 0])
        np.testing.assert_allclose(log_upper_delta, 2.0 * sigma)
        np.testing.assert_allclose(log_lower_delta, 2.0 * sigma)

    def test_shape_mismatch_raises(self):
        y = np.zeros(10)
        sigma = np.zeros(5)

        with pytest.raises(ValueError, match="shape mismatch"):
            bounds_from_sigma(y, sigma, sigma_multiple=3.0, scale=LINEAR)


class TestEnvelopeInvariance:
    """σ is the data's local scatter, so the ±1.96σ band envelopes ~95%
    of the data and stays finite in *every* scale — including scales
    whose forward transform is singular at a zero-crossing."""

    def _band_pct(self, y, scale):
        sigma = estimate_pointwise_sigma(y, scale=scale)
        center = smooth_centerline(y, scale=scale)
        z = scale.forward(y)
        finite = np.isfinite(z) & np.isfinite(sigma) & np.isfinite(center)
        inside = np.abs(z[finite] - center[finite]) <= 1.96 * sigma[finite]

        return 100.0 * inside.sum() / finite.sum(), sigma[finite]

    @pytest.mark.parametrize("mode", ["Linear", "Bisymlog", "AbsoluteLog"])
    def test_band_envelopes_zero_crossing_data(self, mode):
        """A signed, zero-crossing decay: band must contain ~95% and σ
        must stay finite (no delta-method blow-up at the crossings)."""
        rng = np.random.default_rng(11)
        n = 2048
        t = np.linspace(0.0, 1.0, n)
        signal = 3.0e-4 * np.exp(-t / 0.05) - 1.0e-5  # crosses zero
        y = signal + 1.5e-6 * rng.standard_normal(n)
        scale = Scale(mode, calibration_data=y)

        band_pct, sigma = self._band_pct(y, scale)

        assert np.all(np.isfinite(sigma)), f"{mode}: σ has non-finite values"
        assert 88.0 <= band_pct <= 99.0, (
            f"{mode}: band encloses {band_pct:.1f}% of data, expected ~95%"
        )


class TestEstimatePointwiseSigmaScales:
    def test_log_scale_recovers_relative_noise(self):
        """For multiplicative noise on a decaying signal, the log-scale
        σ estimate should be roughly constant (matches the relative
        noise level) rather than the strongly-varying linear σ."""
        rng = np.random.default_rng(2)
        n = 4096
        t = np.linspace(0.0, 1.0, n)
        base = 1e-4 * np.exp(-3 * t)
        rel_noise = 0.1
        y = base * np.exp(rng.normal(0.0, rel_noise, n))

        sigma_log = estimate_pointwise_sigma(y, scale=Scale("Log"))

        expected = rel_noise / np.log(10.0)
        assert sigma_log.mean() == pytest.approx(expected, rel=0.3)

    def test_linear_vs_log_differ_for_multi_decade_data(self):
        """Linear σ varies wildly across the trace; log σ stays bounded.
        Both estimates are valid in their own scale — they just are
        not the same quantity."""
        rng = np.random.default_rng(3)
        n = 4096
        base = np.logspace(-4, -7, n)
        y = base * np.exp(rng.normal(0.0, 0.15, n))

        sigma_linear = estimate_pointwise_sigma(y, scale=LINEAR)
        sigma_log = estimate_pointwise_sigma(y, scale=Scale("Log"))

        assert sigma_linear.max() / sigma_linear.min() > 10.0
        assert sigma_log.max() / sigma_log.min() < 5.0


class TestScale:
    def test_linear_round_trip(self):
        y = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
        s = Scale("Linear")

        np.testing.assert_array_equal(s.inverse(s.forward(y)), y)

    def test_log_round_trip(self):
        y = np.array([1e-7, 1e-4, 1.0, 1e4])
        s = Scale("Log")

        np.testing.assert_allclose(s.inverse(s.forward(y)), y, rtol=1e-12)

    def test_bisymlog_calibrates_from_data(self):
        y = np.array([-100.0, -1.0, 0.0, 1.0, 100.0])
        s = Scale("Bisymlog", calibration_data=y)

        assert s.bisymlog is not None
        assert s.bisymlog.C is not None
        np.testing.assert_allclose(s.inverse(s.forward(y)), y, rtol=1e-10)

    def test_unknown_mode_rejected(self):
        with pytest.raises(ValueError, match="unknown scale"):
            Scale("Bogus")
