"""``calculate.optimize.fit_fcn`` helpers: ``rescale_loss_fcn``, ``_log_ratio``."""
import numpy as np
import pytest

from frhodo.optimize.cost.fit_fcn import (
    _degenerate_trace_output,
    _log_ratio,
    rescale_loss_fcn,
)


class TestRescaleLossFcn:
    """Maps the loss array onto the input array's quantile range."""

    def test_scaled_loss_endpoints_match_input_endpoints(self):
        """The transformed loss should span [x.min, x.max]."""
        x = np.linspace(0.0, 10.0, 11)
        loss = np.linspace(100.0, 900.0, 11)  # arbitrary loss scale

        scaled = rescale_loss_fcn(x, loss)

        assert scaled.min() == pytest.approx(x.min(), rel=1e-9), (
            f"scaled loss min should equal x.min()={x.min()}, got {scaled.min()}"
        )
        assert scaled.max() == pytest.approx(x.max(), rel=1e-9), (
            f"scaled loss max should equal x.max()={x.max()}, got {scaled.max()}"
        )

    def test_scaling_is_linear(self):
        """A linear loss vs. linear x should produce a linear output."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        loss = 2.0 * x + 7.0  # exactly linear in x

        scaled = rescale_loss_fcn(x, loss)

        diffs = np.diff(scaled)
        np.testing.assert_allclose(diffs, diffs[0], rtol=1e-9), (
            "linear loss should produce linearly spaced output"
        )

    def test_constant_loss_returns_loss_unchanged(self):
        """Degenerate case: zero variance triggers the early-return branch."""
        x = np.linspace(0.0, 10.0, 11)
        loss = np.full_like(x, 5.0)

        scaled = rescale_loss_fcn(x, loss)

        np.testing.assert_array_equal(scaled, loss), (
            "constant loss should pass through unchanged (avoids divide-by-zero)"
        )

    def test_constant_x_returns_loss_unchanged(self):
        """If x is constant, rescaling is undefined; function falls through."""
        x = np.full(11, 3.14)
        loss = np.linspace(100.0, 900.0, 11)

        scaled = rescale_loss_fcn(x, loss)

        np.testing.assert_array_equal(scaled, loss)


class TestLogRatio:
    """``log10`` of the larger-over-smaller ratio, elementwise."""

    def test_equal_inputs_give_zero(self):
        a = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(_log_ratio(a, a), 0.0, atol=1e-15)

    def test_obs_exp_greater_uses_exp_over_sim(self):
        obs_exp = np.array([10.0, 100.0])
        obs_sim = np.array([1.0, 10.0])
        np.testing.assert_allclose(_log_ratio(obs_exp, obs_sim), [1.0, 1.0], rtol=1e-12)

    def test_obs_sim_greater_uses_sim_over_exp(self):
        obs_exp = np.array([1.0, 10.0])
        obs_sim = np.array([10.0, 100.0])
        np.testing.assert_allclose(_log_ratio(obs_exp, obs_sim), [1.0, 1.0], rtol=1e-12)

    def test_mixed_branches_per_element(self):
        obs_exp = np.array([10.0, 1.0, 5.0])
        obs_sim = np.array([1.0, 10.0, 5.0])
        np.testing.assert_allclose(
            _log_ratio(obs_exp, obs_sim),
            [1.0, 1.0, 0.0],
            rtol=1e-12, atol=1e-15,
        )

    def test_output_is_non_negative(self):
        rng = np.random.default_rng(0)
        obs_exp = rng.uniform(0.1, 10.0, size=50)
        obs_sim = rng.uniform(0.1, 10.0, size=50)
        result = _log_ratio(obs_exp, obs_sim)
        assert (result >= 0.0).all(), (
            f"_log_ratio must be non-negative; got min {result.min()}"
        )

    def test_preserves_input_shape(self):
        obs_exp = np.array([[1.0], [2.0], [3.0]])  # shape (3,1) — same as resid_func indexing
        obs_sim = np.array([[2.0], [1.0], [3.0]])
        result = _log_ratio(obs_exp, obs_sim)
        assert result.shape == obs_exp.shape


class TestDegenerateTraceOutput:
    """Penalty short-circuit for simulations that collapse to <2 timesteps."""

    @pytest.fixture
    def fake_shock(self):
        class _S:
            pass

        return _S()

    def test_one_point_trace_signals_undefined_loss(self, fake_shock):
        ind_var = np.array([[0.0]])
        obs_sim = np.array([[1.5]])
        out = _degenerate_trace_output(
            fake_shock, ind_var, obs_sim, coef_opt=[],
            var={"loss_alpha": 2.0},
        )
        assert np.isinf(out["loss"])
        assert out["resid"].shape == (1,)
        assert out["weights"].size == 1
        assert out["aggregate_weights"].size == 1

    def test_empty_trace_signals_undefined_loss(self, fake_shock):
        ind_var = np.array([]).reshape(0, 1)
        obs_sim = np.array([]).reshape(0, 1)
        out = _degenerate_trace_output(
            fake_shock, ind_var, obs_sim, coef_opt=[],
            var={"loss_alpha": 2.0},
        )
        assert np.isinf(out["loss"])

    def test_keys_match_normal_output_for_aggregation(self, fake_shock):
        """The penalty dict must carry every key that ``append_output`` will
        aggregate per shock; otherwise concatenation fails for the run.
        """
        out = _degenerate_trace_output(
            fake_shock,
            np.array([[0.0]]), np.array([[0.0]]),
            coef_opt=[], var={"loss_alpha": 2.0},
        )
        required = {
            "wsse", "resid", "resid_outlier", "loss", "weights",
            "aggregate_weights", "obs_sim_interp", "obs_exp", "obs_bounds",
            "shock", "independent_var", "observable", "t_unc",
            "loss_alpha", "KDE",
        }
        assert required <= set(out.keys()), required - set(out.keys())

    def test_adaptive_loss_alpha_falls_back_to_two(self, fake_shock):
        out = _degenerate_trace_output(
            fake_shock,
            np.array([[0.0]]), np.array([[0.0]]),
            coef_opt=[], var={"loss_alpha": "Adaptive"},
        )
        assert out["loss_alpha"] == 2.0
