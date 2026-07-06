"""Unit tests for the experiment-level M-location aggregation."""
import numpy as np
import pytest

from frhodo.optimize.cost.aggregation import (
    ALPHA_MAX,
    ALPHA_MIN,
    coverage_weights,
    irls_weights_asym,
    ln_z_asym,
    model_error_floor,
    psi_asym,
    rho_asym,
    solve_m_location,
)



class TestAsymmetricLossPieces:
    def test_ln_z_asym_at_alpha_two_is_gaussian_normalizer(self):
        """Both halves are Gaussian at α=2, so Z_asym = sqrt(2π)."""
        expected = float(np.log(np.sqrt(2.0 * np.pi)))
        assert ln_z_asym(2.0) == pytest.approx(expected, abs=1e-5), (
            f"ln_z_asym(2)={ln_z_asym(2.0):.8f} should equal ln sqrt(2pi)={expected:.8f}"
        )

    def test_rho_is_quadratic_on_good_side_for_any_alpha(self):
        z = np.array([-3.0, -1.0, -0.2])
        for alpha in [1.0, 1.5, 2.0]:
            np.testing.assert_allclose(
                rho_asym(z, alpha), 0.5 * z * z, rtol=1e-12,
                err_msg=f"good side must be quadratic at alpha={alpha}",
            )

    def test_rho_upper_side_is_pseudo_huber_at_alpha_one(self):
        z = np.array([0.5, 1.0, 4.0])
        expected = np.sqrt(z * z + 1.0) - 1.0
        np.testing.assert_allclose(rho_asym(z, 1.0), expected, rtol=1e-12)

    @pytest.mark.parametrize("alpha", [1.0, 1.3, 1.7, 2.0])
    @pytest.mark.parametrize("z0", [-2.0, -0.5, 0.5, 2.0])
    def test_psi_matches_finite_difference(self, alpha, z0):
        h = 1e-6
        analytic = psi_asym(np.array([z0]), alpha)[0]
        numeric = (
            rho_asym(np.array([z0 + h]), alpha)[0]
            - rho_asym(np.array([z0 - h]), alpha)[0]
        ) / (2 * h)
        assert analytic == pytest.approx(numeric, rel=1e-5), (
            f"psi mismatch at alpha={alpha}, z={z0}: {analytic:.8g} vs FD {numeric:.8g}"
        )

    def test_irls_weights_are_one_on_good_side_and_decreasing_above(self):
        z = np.array([-1.0, 0.0, 0.5, 2.0, 8.0])
        w = irls_weights_asym(z, 1.0)
        assert w[0] == 1.0 and w[1] == 1.0
        assert np.all(np.diff(w[2:]) < 0), f"weights must decrease with z: {w}"
        assert np.all((w > 0) & (w <= 1.0))


class TestMLocation:
    def test_alpha_two_recovers_weighted_mean(self):
        losses = np.array([1.0, 2.0, 3.0, 10.0])
        weights = np.array([1.0, 2.0, 3.0, 4.0])
        expected = float(np.average(losses, weights=weights))
        result = solve_m_location(losses, weights, alpha=2.0)
        assert result.mu == pytest.approx(expected, rel=1e-9), (
            f"alpha=2 must give the weighted mean {expected}, got {result.mu}"
        )

    def test_single_shock_degenerates_to_its_loss(self):
        result = solve_m_location(np.array([3.7]))
        assert result.mu == 3.7
        assert result.alpha == ALPHA_MAX

    def test_identical_losses_return_that_value(self):
        result = solve_m_location(np.full(6, 2.5))
        assert result.mu == pytest.approx(2.5)
        assert result.c == 0.0

    def test_shift_equivariance(self):
        rng = np.random.default_rng(7)
        losses = 1.0 + 0.1 * rng.standard_normal(20)
        losses[-1] = 3.0
        base = solve_m_location(losses)
        shifted = solve_m_location(losses + 5.0)
        assert shifted.mu == pytest.approx(base.mu + 5.0, abs=1e-6), (
            "M-location must be shift-equivariant"
        )

    @pytest.mark.parametrize("alpha", [1.0, 1.5, 2.0])
    def test_monotone_in_every_sample_at_fixed_alpha(self, alpha):
        """Worsening any shock can never lower the objective (fixed α)."""
        rng = np.random.default_rng(3)
        losses = 1.0 + 0.2 * rng.standard_normal(12)
        base = solve_m_location(losses, alpha=alpha).mu
        for idx in [0, 5, int(np.argmax(losses))]:
            for delta in [0.1, 1.0, 10.0]:
                bumped = losses.copy()
                bumped[idx] += delta
                mu = solve_m_location(bumped, alpha=alpha).mu
                assert mu >= base - 1e-10, (
                    f"raising sample {idx} by {delta} lowered mu: {base} -> {mu}"
                )

    def test_adaptive_alpha_keeps_decreases_negligible(self):
        """α re-estimation may move μ* slightly when a tail grows; any
        decrease must stay far below the loss scale c."""
        rng = np.random.default_rng(3)
        losses = 1.0 + 0.2 * rng.standard_normal(12)
        base = solve_m_location(losses)
        bumped = losses.copy()
        bumped[int(np.argmax(losses))] += 10.0
        result = solve_m_location(bumped)
        assert result.mu >= base.mu - 0.05 * base.c, (
            f"adaptive-alpha decrease too large: {base.mu} -> {result.mu} (c={base.c})"
        )

    def test_upper_outlier_saturates_at_the_bulk(self):
        losses = np.array([0.95, 0.98, 1.0, 1.02, 1.05, 10.0])
        result = solve_m_location(losses)
        mean = float(losses.mean())
        assert 0.9 < result.mu < 1.1, (
            f"robust location {result.mu:.3f} must sit at the bulk (~1.0), "
            f"not be dragged toward the mean {mean:.3f}"
        )
        assert result.irls_weights[-1] < 0.05, (
            f"the outlier must be heavily discounted, got weight "
            f"{result.irls_weights[-1]:.4f}"
        )

    def test_asymmetry_good_side_pulls_harder_than_bad_side(self):
        bulk = np.array([0.95, 0.98, 1.0, 1.02, 1.05])
        up = solve_m_location(np.append(bulk, 1.0 + 3.0)).mu
        down = solve_m_location(np.append(bulk, 1.0 - 3.0)).mu
        pull_up = up - 1.0
        pull_down = 1.0 - down
        assert pull_down > pull_up, (
            f"quadratic good side must pull harder: down {pull_down:.4f} "
            f"vs up {pull_up:.4f}"
        )

    def test_adaptive_alpha_near_two_for_clean_campaign(self):
        rng = np.random.default_rng(11)
        losses = 1.0 + 0.05 * rng.standard_normal(40)
        result = solve_m_location(losses)
        assert result.alpha > 1.6, (
            f"clean campaign should keep alpha near 2, got {result.alpha:.3f}"
        )

    def test_adaptive_alpha_drops_under_upper_contamination(self):
        rng = np.random.default_rng(11)
        clean = 1.0 + 0.05 * rng.standard_normal(40)
        contaminated = clean.copy()
        contaminated[:6] = 1.0 + np.array([0.4, 0.5, 0.6, 0.7, 0.8, 1.0])
        alpha_clean = solve_m_location(clean).alpha
        alpha_dirty = solve_m_location(contaminated).alpha
        assert alpha_dirty < alpha_clean - 0.05, (
            f"contamination should lower alpha: clean {alpha_clean:.3f}, "
            f"dirty {alpha_dirty:.3f}"
        )

    def test_rejects_non_finite_losses(self):
        with pytest.raises(ValueError, match="finite"):
            solve_m_location(np.array([1.0, np.inf, 2.0]))

    def test_few_shocks_fall_back_to_alpha_two(self):
        result = solve_m_location(np.array([1.0, 1.1, 5.0]))
        assert result.alpha == ALPHA_MAX

    def test_alpha_bounds_exported(self):
        assert ALPHA_MIN == 1.0 and ALPHA_MAX == 2.0


class TestCoverageWeights:
    def test_sparse_shock_outweighs_cluster_members(self):
        rng = np.random.default_rng(5)
        cluster = np.column_stack([
            0.55 + 0.005 * rng.standard_normal(10),
            2.07 + 0.005 * rng.standard_normal(10),
        ])
        sparse = np.array([[0.75, 2.05], [0.62, 1.40]])
        feats = np.vstack([cluster, sparse])
        w = coverage_weights(feats)
        assert w[10:].min() > w[:10].max(), (
            f"sparse shocks {w[10:]} must outweigh cluster members {w[:10]}"
        )
        assert w.mean() == pytest.approx(1.0, rel=1e-9)

    def test_clip_bounds_respected(self):
        rng = np.random.default_rng(5)
        cluster = np.column_stack([
            0.55 + 0.001 * rng.standard_normal(30),
            2.07 + 0.001 * rng.standard_normal(30),
        ])
        lone = np.array([[5.0, -3.0]])
        w = coverage_weights(np.vstack([cluster, lone]), clip=(0.2, 5.0))
        ratio = w.max() / w.min()
        assert ratio <= 5.0 / 0.2 + 1e-9, f"clip violated: ratio {ratio:.1f}"

    def test_fewer_than_four_shocks_get_uniform_weights(self):
        w = coverage_weights(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
        np.testing.assert_array_equal(w, np.ones(3))

    def test_zero_variance_axis_is_dropped(self):
        rng = np.random.default_rng(9)
        t_axis = np.sort(rng.uniform(0.5, 0.8, 12))
        with_const = np.column_stack([t_axis, np.full(12, 2.07)])
        only_t = t_axis[:, None]
        np.testing.assert_allclose(
            coverage_weights(with_const), coverage_weights(only_t), rtol=1e-12,
            err_msg="a constant-pressure campaign must reduce to the 1D case",
        )

    def test_all_identical_conditions_get_uniform_weights(self):
        w = coverage_weights(np.tile([0.6, 2.0], (8, 1)))
        np.testing.assert_array_equal(w, np.ones(8))


class TestModelErrorFloor:
    def test_recovers_shared_floor_exactly(self):
        rng = np.random.default_rng(2)
        sigma = np.exp(rng.uniform(np.log(0.01), np.log(1.0), 30))
        floor = 0.25
        r = np.sqrt(sigma**2 + floor**2)
        assert model_error_floor(r, sigma) == pytest.approx(floor, rel=1e-9)

    def test_zero_when_residuals_within_noise(self):
        sigma = np.array([0.1, 0.2, 0.5, 1.0])
        assert model_error_floor(0.8 * sigma, sigma) == 0.0

    def test_high_excess_shocks_do_not_inflate_floor(self):
        """Optimizable misfit concentrates in a few shocks; the floor
        must track the best-fit bulk, not the misfit tail."""
        sigma = np.full(20, 0.1)
        floor = 0.05
        r = np.sqrt(sigma**2 + floor**2)
        r[:4] = 10.0
        assert model_error_floor(r, sigma) == pytest.approx(floor, rel=1e-6)

    def test_non_finite_entries_ignored(self):
        sigma = np.full(4, 0.1)
        floor = 0.3
        r = np.sqrt(sigma**2 + floor**2)
        r[0] = np.inf
        assert model_error_floor(r, sigma) == pytest.approx(floor, rel=1e-6)

    def test_all_non_finite_returns_zero(self):
        result = model_error_floor(np.array([np.inf, np.nan]), np.ones(2))
        assert result == 0.0


class TestManufacturedTailFromMeasurementOnlyScale:
    """Standardizing by measurement noise alone turns the cleanest shocks
    into the apparent worst — irreducible model error dominates their tiny
    σ̄ — and the robust layer then trims the most informative experiments.
    The total-error scale removes the artifact."""

    def _losses(self, divisor_kind):
        rng = np.random.default_rng(4)
        n = 30
        sigma = np.exp(np.linspace(np.log(0.02), np.log(1.0), n))
        floor = 0.3
        r = np.sqrt(sigma**2 + floor**2) * (1.0 + 0.03 * rng.standard_normal(n))
        if divisor_kind == "measurement":
            losses = r / sigma
        else:
            losses = r / np.sqrt(sigma**2 + floor**2)

        return losses, sigma

    def test_measurement_only_scale_floors_alpha(self):
        losses, _ = self._losses("measurement")
        result = solve_m_location(losses)
        assert result.alpha < 1.2, (
            f"manufactured tail should drive alpha to the floor, "
            f"got {result.alpha:.3f}"
        )

    def test_cleanest_shocks_top_the_manufactured_tail(self):
        losses, sigma = self._losses("measurement")
        assert np.argmax(losses) == np.argmin(sigma), (
            "the lowest-noise shock must show the largest standardized loss"
        )

    def test_total_scale_lifts_alpha_and_fades_no_shock(self):
        measurement = solve_m_location(self._losses("measurement")[0])
        total = solve_m_location(self._losses("total")[0])
        assert total.alpha > measurement.alpha + 0.4, (
            f"floored scale must lift alpha well off the floor: "
            f"{measurement.alpha:.3f} -> {total.alpha:.3f}"
        )
        assert total.alpha > 1.4, (
            f"floored scale should keep alpha high, got {total.alpha:.3f}"
        )
        assert total.irls_weights.min() > 0.5, (
            f"no shock should be heavily faded, got min weight "
            f"{total.irls_weights.min():.3f}"
        )


class TestStrictlyInsideClip:
    """The optimizer start point must land strictly inside the box even
    for negative bounds (lb·(1+ε) sits outside when lb < 0)."""

    def test_negative_bounds_clip_stays_inside(self):
        lb = -np.log(2.0) * np.ones(4)
        ub = np.log(2.0) * np.ones(4)
        raw = np.array([-5.0, 5.0, 0.0, -0.7])
        margin = 1e-9 * (ub - lb)
        clipped = np.clip(raw, lb + margin, ub - margin)
        assert np.all(clipped > lb) and np.all(clipped < ub), (
            f"clip left values outside the box: {clipped} vs [{lb[0]}, {ub[0]}]"
        )
