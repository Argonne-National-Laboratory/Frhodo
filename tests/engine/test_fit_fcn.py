"""``calculate.optimize.fit_fcn`` helpers: ``_log_ratio`` and friends."""
import types

import numpy as np
import pytest

from frhodo.optimize.cost.fit_fcn import (
    CostFunction,
    _degenerate_trace_output,
    _log_ratio,
)



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
            "aggregate_weights", "obs_sim_interp", "obs_exp",
            "shock", "independent_var", "observable", "t_unc",
            "loss_alpha",
        }
        assert required <= set(out.keys()), required - set(out.keys())

    def test_adaptive_loss_alpha_falls_back_to_two(self, fake_shock):
        out = _degenerate_trace_output(
            fake_shock,
            np.array([[0.0]]), np.array([[0.0]]),
            coef_opt=[], var={"loss_alpha": "Adaptive"},
        )
        assert out["loss_alpha"] == 2.0


class TestStagedWarmup:
    """Warmup must compile the kernel cache in one worker before fan-out
    (numba's on-disk cache rename races under concurrent Windows writers)."""

    class _RecordingPool:
        def __init__(self):
            self.map_sizes = []

        def map(self, fcn, args):
            self.map_sizes.append(len(args))

            return [None] * len(args)

    def _cost_function_with_pool(self, pool, n_fit_rxns=0):
        fake = types.SimpleNamespace(
            pool=pool,
            shocks2run=[object()],
            x0=np.zeros(1),
            fit_all_coeffs=lambda rates: np.zeros(1),
            _build_fit_args=lambda rates: [object()] * n_fit_rxns,
            _build_var_dict=lambda: {},
            coef_opt=[],
        )

        return fake

    def test_single_task_precedes_fanout(self):
        pool = self._RecordingPool()
        fake = self._cost_function_with_pool(pool)
        CostFunction.warmup_workers(fake, 8, np.zeros(1))
        assert pool.map_sizes == [1, 8], (
            f"warmup must stage [1, n_workers], got {pool.map_sizes}"
        )

    def test_single_worker_pool_warms_once(self):
        pool = self._RecordingPool()
        fake = self._cost_function_with_pool(pool)
        CostFunction.warmup_workers(fake, 1, np.zeros(1))
        assert pool.map_sizes == [1], (
            f"single-worker warmup must not fan out, got {pool.map_sizes}"
        )

    def test_per_reaction_fits_stage_before_everything(self):
        """Each pooled fit task warms alone so the fit-path kernels are
        cached before fit_all_coeffs can fan them out mid-run."""
        pool = self._RecordingPool()
        fake = self._cost_function_with_pool(pool, n_fit_rxns=3)
        CostFunction.warmup_workers(fake, 8, np.zeros(1))
        assert pool.map_sizes == [1, 1, 1, 1, 8], (
            f"warmup must stage fit tasks singly then [1, n_workers], "
            f"got {pool.map_sizes}"
        )


class TestCalibrateErrorFloor:
    """One probe at the start point raises each shock's standardizing
    scale from measurement noise to total expected error."""

    def _fake(self, sigma_bars, probe_losses, fit_result=np.zeros(1)):
        shocks = [
            types.SimpleNamespace(sigma_bar=float(s), sigma_total=float(s))
            for s in sigma_bars
        ]
        logged = []
        fake = types.SimpleNamespace(
            shocks2run=shocks,
            x0=np.zeros(1),
            fit_all_coeffs=lambda rates: fit_result,
            _build_var_dict=lambda: {},
            _dispatch=lambda x, var: [{"loss": float(l)} for l in probe_losses],
            _log=logged.append,
            _model_error_floor=0.0,
            _start_loss_raw=None,
        )

        return fake, shocks, logged

    def test_sets_total_scale_from_campaign_floor(self):
        sigma = np.array([0.1, 0.2, 0.4, 0.8])
        floor = 0.3
        losses = np.sqrt(sigma**2 + floor**2) / sigma
        fake, shocks, _ = self._fake(sigma, losses)
        CostFunction.calibrate_error_floor(fake, np.zeros(1))
        assert fake._model_error_floor == pytest.approx(floor, rel=1e-6)
        for shock, s_bar in zip(shocks, sigma):
            expected = np.sqrt(s_bar**2 + floor**2)
            assert shock.sigma_total == pytest.approx(expected, rel=1e-6), (
                f"sigma_total for sigma_bar={s_bar} should be {expected}"
            )

    def test_failed_start_fit_leaves_scales_untouched(self):
        sigma = np.array([0.1, 0.2, 0.4, 0.8])
        fake, shocks, logged = self._fake(sigma, np.ones(4), fit_result=None)
        CostFunction.calibrate_error_floor(fake, np.zeros(1))
        for shock, s_bar in zip(shocks, sigma):
            assert shock.sigma_total == s_bar
        assert any("skipped" in msg for msg in logged)

    def test_overflowed_probe_shocks_are_ignored(self):
        sigma = np.full(4, 0.1)
        floor = 0.2
        losses = np.sqrt(sigma**2 + floor**2) / sigma
        losses[0] = np.inf
        fake, _, _ = self._fake(sigma, losses)
        CostFunction.calibrate_error_floor(fake, np.zeros(1))
        assert fake._model_error_floor == pytest.approx(floor, rel=1e-6)
