"""Pre-fit screening: core math, and the campaign driver on a real mech."""
import cantera as ct
import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from frhodo.api import (
    ExperimentShock,
    ObservableSettings,
    OptimizableSpec,
    OptimizationRequest,
    PostShockState,
    _to_internal_shock,
)
from frhodo.optimize import screening
from frhodo.optimize._worker_context import MechBuildPayload, WorkerContext
from frhodo.optimize.cost import fit_fcn
from frhodo.optimize.cost.settings import CostSettings
from frhodo.optimize.pool import PersistentWorkerPool
from frhodo.optimize.sensitivity_cache import SensitivityCache
from frhodo.optimize.screening import (
    ASSUMED_MOVABILITY,
    SUGGEST_LOSS_FRACTION,
    ScreeningResult,
    _run_start_sim,
    experiment_influence,
    influential_mask,
    uniqueness_weights,
    campaign_aggregate,
    effective_rank,
    screen_campaign,
    shock_scores,
    spectral_gap_rank,
    suggested_set,
)
from frhodo.optimize.shock_prep import shock_sigma_bar
from frhodo.simulation.shock.state import RuntimeReactorState



class TestShockScores:
    def test_importance_is_weighted_mean_abs_sensitivity(self):
        S = np.array([[1.0, -2.0], [3.0, 0.0]])
        w = np.array([1.0, 3.0])
        r = np.zeros(2)
        importance, leverage, loss, footprint = shock_scores(S, w, r)
        np.testing.assert_allclose(importance, [(1 + 9) / 4, (2 + 0) / 4])
        np.testing.assert_allclose(leverage, 0.0)
        assert loss == 0.0
        np.testing.assert_allclose(footprint, importance)

    def test_slope_tracks_residual_alignment(self):
        """A reaction whose sensitivity aligns with the residual carries
        a descent slope; an orthogonal one carries none."""
        r = np.array([1.0, -1.0])
        S = np.column_stack([r, np.array([1.0, 1.0])])
        w = np.ones(2)
        _, slope, loss, _ = shock_scores(S, w, r)
        np.testing.assert_allclose(slope, [-2.0, 0.0])
        assert loss == pytest.approx(1.0)

    def test_chain_converts_to_residual_metric(self):
        """The chain factor scales sensitivities into the residual's
        transformed, standardized units before the slope contraction."""
        r = np.array([1.0, -1.0])
        S = np.column_stack([r, np.array([1.0, 1.0])])
        w = np.ones(2)
        _, slope, _, footprint = shock_scores(S, w, r, chain=np.array([0.5, 0.5]))
        np.testing.assert_allclose(slope, [-1.0, 0.0])
        np.testing.assert_allclose(footprint, [0.5, 0.5])

    def test_zero_weights_give_zero_scores(self):
        importance, leverage, loss, footprint = shock_scores(
            np.ones((3, 2)), np.zeros(3), np.ones(3),
        )
        np.testing.assert_allclose(importance, 0.0)
        np.testing.assert_allclose(leverage, 0.0)
        assert loss == 0.0
        np.testing.assert_allclose(footprint, 0.0)


class TestCampaignAggregate:
    def test_weighted_mean_over_shocks(self):
        per_shock = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        agg = campaign_aggregate(per_shock, np.array([3.0, 1.0]))
        np.testing.assert_allclose(agg, [0.75, 0.25])


class TestEffectiveRank:
    def test_rank_one_campaign(self):
        """All shocks constrain the same direction -> erank ~ 1."""
        g = np.array([1.0, 1.0, 0.0])
        G = np.stack([g, 2 * g, 0.5 * g])
        erank, svals = effective_rank(G, np.ones(3))
        assert erank == pytest.approx(1.0, abs=1e-9)
        assert svals[0] > 0

    def test_orthogonal_shocks_count_separately(self):
        G = np.eye(3)
        erank, _ = effective_rank(G, np.ones(3))
        assert erank == pytest.approx(3.0, abs=1e-9)

    def test_zero_matrix(self):
        erank, _ = effective_rank(np.zeros((2, 4)), np.ones(2))
        assert erank == 0.0


class TestSpectralGapRank:
    def test_clear_kink_found(self):
        scores = np.array([1.0, 0.9, 0.8, 1e-4, 9e-5, 8e-5])
        assert spectral_gap_rank(scores) == 3

    def test_smooth_spectrum_has_no_kink(self):
        scores = np.geomspace(1.0, 1e-3, 20)
        assert spectral_gap_rank(scores) is None

    def test_short_spectrum_has_no_kink(self):
        assert spectral_gap_rank(np.array([1.0, 0.5])) is None


class TestSuggestedSet:
    def test_materiality_threshold_is_absolute(self):
        materiality = np.array([0.5, 0.005, 0.0])
        keep = suggested_set(materiality, loss_start=1.0, fraction=0.01)
        np.testing.assert_array_equal(keep, [True, False, False])

    def test_nothing_material_selects_nothing(self):
        keep = suggested_set(np.array([1e-9, 1e-8]), loss_start=1.0)
        assert not keep.any()

    def test_degenerate_zero_loss(self):
        keep = suggested_set(np.array([1.0]), loss_start=0.0)
        assert not keep.any()


class TestUniquenessWeights:
    def test_duplicates_share_weight(self):
        """Three identical experiments + one orthogonal: the copies get
        ~1/3 relative weight, the unique one full weight."""
        g = np.array([1.0, 0.0, 0.0])
        u = np.array([0.0, 1.0, 0.0])
        W = uniqueness_weights(np.stack([g, 2 * g, 0.5 * g, u]))
        assert W[3] / W[0] == pytest.approx(3.0)
        np.testing.assert_allclose(W[:3], W[0])

    def test_orthogonal_experiments_weight_equally(self):
        W = uniqueness_weights(np.eye(4))
        np.testing.assert_allclose(W, W[0])

    def test_tp_position_is_irrelevant(self):
        """Two experiments constraining the same direction share weight
        no matter how far apart their conditions are — collinearity is
        the only input."""
        g = np.array([3.0, 1.0])
        W = uniqueness_weights(np.stack([g, 10.0 * g, [-1.0, 3.0]]))
        assert W[0] == pytest.approx(W[1])
        assert W[2] > W[0]

    def test_zero_influence_rows_get_unit_weight(self):
        W = uniqueness_weights(np.array([[1.0, 0.0], [0.0, 0.0]]))
        assert np.all(np.isfinite(W))
        assert W[1] > 0


class TestInfluenceMask:
    def test_zero_influence_experiments_dropped(self):
        slopes = np.array([[1.0, 2.0], [0.0, 0.0], [1e-6, 0.0]])
        keep = influential_mask(slopes)
        np.testing.assert_array_equal(keep, [True, False, False])
        np.testing.assert_allclose(experiment_influence(slopes),
                                   [3.0, 0.0, 1e-6])

    def test_all_zero_keeps_everything(self):
        keep = influential_mask(np.zeros((3, 2)))
        assert keep.all(), "degenerate campaigns must not drop shocks"

    def test_weak_but_alive_experiment_survives(self):
        # ~0.4% of the leader's influence: noise-inflated, not dead.
        slopes = np.array([[1.0, 2.0], [0.005, 0.008]])
        keep = influential_mask(slopes)
        assert keep.all(), (
            f"exclusion must only cut essentially-zero-signal shocks, "
            f"dropped {list(np.nonzero(~keep)[0])}"
        )


@pytest.mark.slow
class TestScreenCampaign:
    """The driver on the real cycloheptane mech with synthetic shocks."""

    BOUNDED = (0, 5)

    def _campaign(self, mech):
        t = np.linspace(1e-7, 5e-5, 60)
        shocks = []
        for num, (T, P) in enumerate([(1500.0, 2e4), (1650.0, 4e4)], start=1):
            exp = ExperimentShock(
                t=t, observable=np.zeros_like(t),
                initial=PostShockState(
                    T_reac=T, P_reac=P, u2=181.85, rho1=0.0230433,
                    composition={"Kr": 0.96, "cC7H14": 0.04},
                ),
                t_end=5e-5,
            )
            request = OptimizationRequest(
                shocks=[exp], optimizable=OptimizableSpec(rates=[]),
                reactor_state=self._reactor(),
                cost=self._cost(), observable=ObservableSettings(),
            )
            shock = _to_internal_shock(exp, request, mech)
            shock.num = num
            shocks.append(shock)

        return shocks

    def _reactor(self):
        state = RuntimeReactorState(
            name="Incident Shock Reactor", t_end=5e-5, t_unit_conv=1e-6,
            sim_interp_factor=1, ode_solver="BDF", ode_rtol=1e-4,
            ode_atol=1e-7,
        )

        return state

    def _cost(self):
        return CostSettings(scale="Linear", loss_alpha=2.0, loss_c=1.0)

    def test_scores_shape_and_suggested_need_no_bounds(
        self, loaded_cycloheptane,
    ):
        """Screening ranks every reaction with no user-set uncertainties;
        the suggested set follows leverage under assumed movability."""
        mech = loaded_cycloheptane
        result = screen_campaign(
            mech, self._campaign(mech), self._reactor(), self._cost(),
        )

        n = mech.gas.n_reactions
        assert len(result.importance) == n
        assert np.all(np.isfinite(result.importance))
        assert np.all(np.isfinite(result.leverage))
        assert result.loss_start > 0
        assert not result.skipped_shocks

        lev = np.asarray(result.leverage)
        expected = (lev * np.log(ASSUMED_MOVABILITY)
                    > SUGGEST_LOSS_FRACTION * result.loss_start)
        np.testing.assert_array_equal(result.suggested, expected)

        # The fuel-decomposition reaction (rxn 0) must register on a
        # density-gradient campaign of 96% Kr / 4% cC7H14.
        assert result.importance[0] > 0
        assert 1.0 <= result.effective_rank <= 2.0
        assert result.ranking()[0] == int(np.argmax(lev))

    def test_sim_cache_skips_solves_and_invalidates_on_coeff_change(
        self, loaded_cycloheptane, monkeypatch,
    ):
        """A re-screen with the cache runs zero solves and reproduces the
        result exactly; a real coefficient change re-solves."""
        mech = loaded_cycloheptane
        calls = {"sim": 0, "sens": 0}
        real_sim = screening._run_start_sim
        real_sens = screening.compute_sensitivity

        def counting_sim(*args, **kwargs):
            calls["sim"] += 1

            return real_sim(*args, **kwargs)

        def counting_sens(*args, **kwargs):
            calls["sens"] += 1

            return real_sens(*args, **kwargs)

        monkeypatch.setattr(screening, "_run_start_sim", counting_sim)
        monkeypatch.setattr(screening, "compute_sensitivity", counting_sens)

        cache = SensitivityCache()
        first = screen_campaign(
            mech, self._campaign(mech), self._reactor(), self._cost(),
            sim_cache=cache,
        )
        assert calls == {"sim": 2, "sens": 2}
        assert len(cache) == 2

        second = screen_campaign(
            mech, self._campaign(mech), self._reactor(), self._cost(),
            sim_cache=cache,
        )
        assert calls == {"sim": 2, "sens": 2}, (
            f"cache hit must skip both solves, got {calls}"
        )
        np.testing.assert_array_equal(second.leverage, first.leverage)
        np.testing.assert_array_equal(second.footprints, first.footprints)

        idx = next(
            i for i, r in enumerate(mech.gas.reactions())
            if type(r.rate) is ct.ArrheniusRate
        )
        original = mech.coeffs[idx][0]["pre_exponential_factor"]
        mech.coeffs[idx][0]["pre_exponential_factor"] = original * 1.5
        try:
            mech.modify_reactions(mech.coeffs, rxnIdxs=idx)
            screen_campaign(
                mech, self._campaign(mech), self._reactor(), self._cost(),
                sim_cache=cache,
            )
        finally:
            mech.coeffs[idx][0]["pre_exponential_factor"] = original
            mech.modify_reactions(mech.coeffs, rxnIdxs=idx)
        assert calls == {"sim": 4, "sens": 4}, (
            f"a coefficient change must invalidate the cache, got {calls}"
        )
        assert len(cache) == 2, "stale-fingerprint entries must be pruned"

    def test_worker_pool_prefetch_matches_serial(
        self, loaded_cycloheptane, monkeypatch,
    ):
        """Pooled screening reproduces the serial result exactly, stages
        the first task alone, and fills the cache."""
        mech = loaded_cycloheptane
        serial = screen_campaign(
            mech, self._campaign(mech), self._reactor(), self._cost(),
            sim_cache=SensitivityCache(),
        )

        class SerialMapPool:
            map_calls = 0

            def map(self, func, iterable, chunksize=1):
                SerialMapPool.map_calls += 1

                return [func(args) for args in iterable]

        monkeypatch.setattr(
            fit_fcn, "_pool_worker_ctx", WorkerContext(mech=mech),
        )
        cache = SensitivityCache()
        pooled = screen_campaign(
            mech, self._campaign(mech), self._reactor(), self._cost(),
            sim_cache=cache, worker_pool=SerialMapPool(),
        )

        assert SerialMapPool.map_calls == 2, (
            "the first task must stage alone before the fan-out"
        )
        np.testing.assert_array_equal(pooled.leverage, serial.leverage)
        np.testing.assert_array_equal(pooled.footprints, serial.footprints)
        assert len(cache) == 2, "pool results must land in the cache"

    def test_worker_pool_failures_skip_shocks(self, loaded_cycloheptane):
        """Shocks the pool can't solve are skipped like the serial path;
        a campaign with none left raises."""
        mech = loaded_cycloheptane

        class FailingPool:
            def map(self, func, iterable, chunksize=1):
                return [None for _ in iterable]

        with pytest.raises(ValueError, match="no usable shocks"):
            screen_campaign(
                mech, self._campaign(mech), self._reactor(), self._cost(),
                sim_cache=SensitivityCache(), worker_pool=FailingPool(),
            )

    def test_worker_pool_end_to_end(self, loaded_cycloheptane):
        """Real spawned workers rebuild the mechanism from the payload
        and reproduce the serial screening."""
        mech = loaded_cycloheptane
        serial = screen_campaign(
            mech, self._campaign(mech), self._reactor(), self._cost(),
            sim_cache=SensitivityCache(),
        )
        payload = MechBuildPayload(
            reset_mech=mech.reset_mech,
            thermo_coeffs=mech.thermo_coeffs,
            coeffs=mech.coeffs,
            coeffs_bnds=mech.coeffs_bnds,
            rate_bnds=mech.rate_bnds,
        )
        worker_pool = PersistentWorkerPool()
        try:
            pool = worker_pool.acquire(workers=2, payload=payload)
            pooled = screen_campaign(
                mech, self._campaign(mech), self._reactor(), self._cost(),
                sim_cache=SensitivityCache(), worker_pool=pool,
            )
        finally:
            worker_pool.close()

        np.testing.assert_allclose(
            pooled.leverage, serial.leverage, rtol=1e-9,
            err_msg="worker-solved screening must match serial",
        )
        assert pooled.skipped_shocks == serial.skipped_shocks == []


class TestCoeffsVersion:
    def test_bumps_on_coefficient_change_only(self, loaded_cycloheptane):
        mech = loaded_cycloheptane
        before = mech.coeffs_version
        mech.modify_reactions(mech.coeffs)
        assert mech.coeffs_version == before, (
            "a no-op modify must not bump the version"
        )

        idx = next(
            i for i, r in enumerate(mech.gas.reactions())
            if type(r.rate) is ct.ArrheniusRate
        )
        original = mech.coeffs[idx][0]["pre_exponential_factor"]
        mech.coeffs[idx][0]["pre_exponential_factor"] = original * 1.5
        try:
            mech.modify_reactions(mech.coeffs, rxnIdxs=idx)
            assert mech.coeffs_version == before + 1, (
                "a real coefficient change must bump the version"
            )
        finally:
            mech.coeffs[idx][0]["pre_exponential_factor"] = original
            mech.modify_reactions(mech.coeffs, rxnIdxs=idx)


@pytest.mark.slow
class TestLeverageOracle:
    """Screening leverage must match a finite-difference slope of the
    same windowed, standardized loss for the leading reactions."""

    def test_top_leverage_matches_fd_slope(self, loaded_cycloheptane):
        mech = loaded_cycloheptane
        reactor = RuntimeReactorState(
            name="Incident Shock Reactor", t_end=5e-5, t_unit_conv=1e-6,
            sim_interp_factor=1, ode_solver="BDF", ode_rtol=1e-6,
            ode_atol=1e-9,
        )
        cost = CostSettings(scale="Linear", loss_alpha=2.0, loss_c=1.0)
        t = np.linspace(1e-7, 4.5e-5, 200)
        exp = ExperimentShock(
            t=t, observable=np.zeros_like(t),
            initial=PostShockState(
                T_reac=1500.0, P_reac=2e4, u2=181.85, rho1=0.0230433,
                composition={"Kr": 0.96, "cC7H14": 0.04},
            ),
            t_end=5e-5,
        )
        request = OptimizationRequest(
            shocks=[exp], optimizable=OptimizableSpec(rates=[]),
            reactor_state=reactor, cost=cost,
            observable=ObservableSettings(),
        )
        shock = _to_internal_shock(exp, request, mech)
        shock.num = 1

        result = screen_campaign(mech, [shock], reactor, cost)
        lev = np.asarray(result.leverage)

        def loss_at(mult, j):
            mech.gas.set_multiplier(mult, j)
            try:
                t_sim, obs_sim = _run_start_sim(mech, reactor, shock)
            finally:
                mech.gas.set_multiplier(1.0, j)
            f = CubicSpline(t_sim, obs_sim)
            t_exp = shock.exp_data_trim[:, 0]
            window = (t_exp >= t_sim[0]) & (t_exp <= t_sim[-1])
            sigma = shock_sigma_bar(shock, "Linear")
            resid = (shock.exp_data_trim[window, 1]
                     - f(t_exp[window])) / sigma
            w = shock.weights_trim[window]

            return float(w @ resid**2 / w.sum())

        eps = 1e-3
        for j in np.argsort(lev)[::-1][:2]:
            fd = abs(loss_at(np.exp(eps), int(j))
                     - loss_at(np.exp(-eps), int(j))) / (2 * eps)
            assert lev[j] == pytest.approx(fd, rel=0.1), (
                f"rxn {j}: screening leverage {lev[j]:.3e} vs FD slope "
                f"{fd:.3e}"
            )
