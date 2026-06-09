"""Tests for ``frhodo.simulation.mechanism.troe``.

Covers:
  * ``falloff_parameters`` and ``Troe`` constructor pinning
  * ``multistart_nn`` end-to-end recovery on a known Troe surface
  * ``ln_Troe_jac`` analytic vs 4th-order central FD agreement
"""
import nlopt
import numpy as np
import pytest

from frhodo.simulation.mechanism.troe import (
    Troe,
    _BOUNDS_HIGH,
    _BOUNDS_LOW,
    _PHYS_FCENT_MIN,
    _data_aware_bounds,
    _physically_valid,
    bisymlog_C,
    bisymlog_base,
    default_Troe_coefNames,
    falloff_parameters,
    multistart_nn,
)
from frhodo.simulation.mechanism.troe_kernels import (
    Arrhenius_constraint_kernel,
    Fcent_calc,
    Fcent_constraint_kernel,
    Ru,
    ln_Troe,
    ln_Troe_jac,
    ln_k_max,
    max_ln_val,
    objective_l2_kernel,
    set_x_from_opt_kernel,
)
from frhodo.simulation.mechanism.troe_nn import (
    DECODER_PER_CAND,
    K_CANDIDATES,
    capture_to_normalized_np,
    get_model,
    normalized_to_capture_np,
    raw_to_normalized_np,
    reset_model_cache,
)


# Synthetic 10-element Arrhenius+Fcent vector mimicking a Troe rxn:
# [Ea_low, A_low, n_low, Ea_high, A_high, n_high, A_Fcent, T3, T1, T2]
SYNTHETIC_X0 = [
    -7.1e6, 2.3e12, -0.9,
    0.0, 7.4e10, -0.37,
    0.7346, 94.0, 1756.0, 5182.0,
]

T_GRID = np.array([1000.0, 1500.0, 2000.0])


def _make_M(T, P=1e5):
    return 1e-3 * np.full_like(T, P) / (8.314 * T)


def _synthesize_troe_surface():
    """Generate (T, M, ln_k) from a known Troe parameter set on a 3x3 grid."""
    truth_capture = np.array([
        1.5e8, 5e15, -1.0,
        1.0e8, 1e13, 0.5,
        0.7, 250.0, 1500.0, 4000.0,
    ])
    truth_lna = truth_capture.copy()
    truth_lna[1] = np.log(truth_capture[1])
    truth_lna[4] = np.log(truth_capture[4])

    T_vals = np.array([800.0, 1500.0, 2200.0])
    P_atm = np.array([0.1, 1.0, 10.0])
    n_P, n_T = len(P_atm), len(T_vals)
    T = np.tile(T_vals, (n_P, 1))
    P_2d = np.tile(P_atm.reshape(-1, 1), (1, n_T))
    M = (P_2d * 101325.0) / (Ru * T)
    ln_k = ln_Troe(T, M, *truth_lna)

    return T, M, ln_k, truth_capture


class TestFalloffParametersInit:
    @pytest.fixture
    def falloff(self):
        ln_k = np.log([1e10, 1e11, 1e12])
        algo = {
            "is_P_limit": [False, False],
            "loss_fcn_param": [2.0, 1.0],
        }
        return falloff_parameters(
            T=T_GRID,
            M=_make_M(T_GRID),
            ln_k=ln_k,
            x0=np.array(SYNTHETIC_X0),
            algo_options=algo,
        )

    def test_x0_length_is_ten(self, falloff):
        assert len(falloff.x0) == 10

    def test_alter_idx_full_when_neither_p_limit(self, falloff):
        assert falloff.alter_idx == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    def test_loss_alpha_passes_through(self, falloff):
        assert falloff.loss_alpha == 2.0
        assert falloff.loss_scale == 1.0

    def test_temperature_window_defaults(self, falloff):
        assert falloff.Tmin == 273
        assert falloff.Tmax == 6000

    def test_low_p_limit_drops_low_rate(self):
        ln_k = np.log([1e10, 1e11, 1e12])
        algo = {"is_P_limit": [True, False], "loss_fcn_param": [2.0, 1.0]}
        falloff = falloff_parameters(
            T=T_GRID, M=_make_M(T_GRID), ln_k=ln_k,
            x0=np.array(SYNTHETIC_X0), algo_options=algo,
        )
        assert 0 not in falloff.alter_idx
        assert 1 not in falloff.alter_idx
        assert 2 not in falloff.alter_idx
        assert 3 in falloff.alter_idx

    def test_high_p_limit_drops_high_rate(self):
        ln_k = np.log([1e10, 1e11, 1e12])
        algo = {"is_P_limit": [False, True], "loss_fcn_param": [2.0, 1.0]}
        falloff = falloff_parameters(
            T=T_GRID, M=_make_M(T_GRID), ln_k=ln_k,
            x0=np.array(SYNTHETIC_X0), algo_options=algo,
        )
        assert 0 in falloff.alter_idx
        assert 3 not in falloff.alter_idx
        assert 4 not in falloff.alter_idx
        assert 5 not in falloff.alter_idx

    def test_both_p_limits_optimize_only_fcent(self):
        ln_k = np.log([1e10, 1e11, 1e12])
        algo = {"is_P_limit": [True, True], "loss_fcn_param": [2.0, 1.0]}
        falloff = falloff_parameters(
            T=T_GRID, M=_make_M(T_GRID), ln_k=ln_k,
            x0=np.array(SYNTHETIC_X0), algo_options=algo,
        )
        assert all(i >= 6 for i in falloff.alter_idx), (
            f"both P-limits should drop Arrhenius coefs; got {falloff.alter_idx}"
        )

    def test_degenerate_fcent_locks_to_T2_only(self):
        """When Fcent T3/T1 collapse to ~zero, only T2 is varied."""
        ln_k = np.log([1e10, 1e11, 1e12])
        algo = {"is_P_limit": [False, False], "loss_fcn_param": [2.0, 1.0]}
        x0 = np.array(SYNTHETIC_X0).copy()
        x0[7] = 1e-30
        x0[8] = 1e-30
        falloff = falloff_parameters(
            T=T_GRID, M=_make_M(T_GRID), ln_k=ln_k,
            x0=x0, algo_options=algo,
        )
        assert falloff.Fcent_idx == [9]


class TestTroeInit:
    @pytest.fixture
    def troe_full(self):
        rates = np.array([1e10, 1e11, 1e12])
        return Troe(
            rates=rates, T=T_GRID, P=np.full_like(T_GRID, 1e5),
            M=_make_M,
            x0=np.array(SYNTHETIC_X0),
            coefNames=default_Troe_coefNames,
            bnds=[],
            is_falloff_limit=np.zeros(10, dtype=bool),
        )

    def test_x0_length_is_ten(self, troe_full):
        assert len(troe_full.x0) == 10

    def test_x0_extends_to_ten_when_six_provided(self):
        rates = np.array([1e10, 1e11, 1e12])
        x0_short = list(SYNTHETIC_X0[:6])
        troe = Troe(
            rates=rates, T=T_GRID, P=np.full_like(T_GRID, 1e5),
            M=_make_M, x0=x0_short, coefNames=default_Troe_coefNames,
            is_falloff_limit=np.zeros(10, dtype=bool),
        )
        assert len(troe.x0) == 10
        np.testing.assert_allclose(
            troe.x0[6:], [1.0, 1e-30, 1e-30, 1500.0],
        )

    def test_does_not_mutate_caller_x0_list(self):
        """``Troe.__init__`` must not alias-mutate the caller's x0 list."""
        rates = np.array([1e10, 1e11, 1e12])
        x0 = list(SYNTHETIC_X0[:6])
        x0_snapshot = list(x0)
        Troe(
            rates=rates, T=T_GRID, P=np.full_like(T_GRID, 1e5),
            M=_make_M, x0=x0, coefNames=default_Troe_coefNames,
            is_falloff_limit=np.zeros(10, dtype=bool),
        )
        assert x0 == x0_snapshot

    def test_alter_idx_dict_has_required_keys(self, troe_full):
        for key in ("low_rate", "high_rate", "pre_exponential_factor", "all"):
            assert key in troe_full.alter_idx

    def test_low_high_rate_split(self, troe_full):
        """Each Arrhenius coef ends in "_0" (low) or "_inf" (high)."""
        for n, name in enumerate(default_Troe_coefNames):
            if name.endswith("_0"):
                assert n in troe_full.alter_idx["low_rate"]
                assert n not in troe_full.alter_idx["high_rate"]
            elif name.endswith("_inf"):
                assert n in troe_full.alter_idx["high_rate"]
                assert n not in troe_full.alter_idx["low_rate"]


class TestMultistartNn:
    def test_returns_expected_schema(self):
        T, M, ln_k, _ = _synthesize_troe_surface()
        result = multistart_nn(T, M, ln_k)
        assert set(result.keys()) == {"x", "fval", "nfev", "elapsed", "k_refined"}
        assert result["x"].shape == (10,)
        assert np.isfinite(result["fval"])
        assert result["nfev"] > 0
        assert result["elapsed"] > 0
        assert result["k_refined"] >= 2

    def test_recovers_synthetic_rates_within_rms(self):
        T, M, ln_k, _ = _synthesize_troe_surface()
        result = multistart_nn(T, M, ln_k)
        assert result["fval"] < 0.05, (
            f"Expected rate-space log-RMS < 0.05 on noise-free Troe surface, "
            f"got {result['fval']:.4f}"
        )

    def test_deterministic(self):
        T, M, ln_k, _ = _synthesize_troe_surface()
        result_a = multistart_nn(T, M, ln_k)
        result_b = multistart_nn(T, M, ln_k)
        assert result_a["fval"] == pytest.approx(result_b["fval"], rel=1e-9)
        np.testing.assert_allclose(result_a["x"], result_b["x"], rtol=1e-9)

    def test_output_is_capture_form(self):
        """``x[1]`` and ``x[4]`` should be raw A (not ln_A)."""
        T, M, ln_k, _ = _synthesize_troe_surface()
        result = multistart_nn(T, M, ln_k)
        assert result["x"][1] > 0
        assert result["x"][4] > 0


class TestFalloffParametersFit:
    """End-to-end ``falloff_parameters.fit()`` exercises the AUGLAG polish
    and the analytical ``calc_s`` scaling on the smooth Troe manifold."""

    def _build_polish_inputs(self):
        """Synthetic surface + NN seed → polish-ready (T, M, ln_k, x0_lna)."""
        T, M, ln_k, _ = _synthesize_troe_surface()
        ms = multistart_nn(T, M, ln_k)
        x0_lna = ms["x"].copy()
        x0_lna[1] = np.log(x0_lna[1])
        x0_lna[4] = np.log(x0_lna[4])
        algo = {
            "algorithm": nlopt.LN_SBPLX,
            "xtol_rel": 1e-6,
            "ftol_rel": 1e-6,
            "initial_step": 1e-3,
            "max_eval": 2000,
            "loss_fcn_param": [2.0, 1.0],
            "is_P_limit": [False, False],
        }

        return T, M, ln_k, x0_lna, algo

    def test_fit_returns_full_capture_form_vector(self):
        T, M, ln_k, x0_lna, algo = self._build_polish_inputs()
        fp = falloff_parameters(T, M, ln_k, x0_lna, algo)
        res = fp.fit()
        assert res["x"].shape == (10,)
        assert res["x"][1] > 0
        assert res["x"][4] > 0

    def test_fit_does_not_increase_residual(self):
        """Polish must not make the rate-space fit worse than the seed."""
        T, M, ln_k, x0_lna, algo = self._build_polish_inputs()
        seed_rms = float(np.sqrt(np.mean(
            ((ln_Troe(T, M, *x0_lna) - ln_k).flatten()) ** 2,
        )))
        fp = falloff_parameters(T, M, ln_k, x0_lna, algo)
        res = fp.fit()
        post_lna = res["x"].copy()
        post_lna[1] = np.log(post_lna[1])
        post_lna[4] = np.log(post_lna[4])
        post_rms = float(np.sqrt(np.mean(
            ((ln_Troe(T, M, *post_lna) - ln_k).flatten()) ** 2,
        )))
        assert post_rms <= seed_rms + 1e-6, (
            f"polish worsened fit: seed RMS={seed_rms:.4e}, polished={post_rms:.4e}"
        )

    def test_calc_s_at_origin_is_finite_and_positive(self):
        """The analytical scaling factors are real numbers > 0 at p0_opt=0."""
        T, M, ln_k, x0_lna, algo = self._build_polish_inputs()
        fp = falloff_parameters(T, M, ln_k, x0_lna, algo)
        fp.p0 = fp.x0.copy()
        fp.p0[-3:] = fp.convert_Fcent(fp.p0[-3:], "base2opt")
        fp.p0 = fp.p0[fp.alter_idx]
        fp.s = np.ones_like(fp.p0)

        s = fp.calc_s(np.zeros_like(fp.p0))

        assert np.all(np.isfinite(s)), f"calc_s produced non-finite scales: {s}"
        assert np.all(s > 0), f"calc_s produced non-positive scales: {s}"


class TestObjectiveLossAlpha:
    """``loss_alpha == 2`` (the default) bypasses ``adaptive_weights``;
    other values exercise the robust-loss path."""

    def _make_fp(self, alpha):
        T, M, ln_k, _ = _synthesize_troe_surface()
        ms = multistart_nn(T, M, ln_k)
        x0_lna = ms["x"].copy()
        x0_lna[1] = np.log(x0_lna[1])
        x0_lna[4] = np.log(x0_lna[4])
        algo = {
            "algorithm": nlopt.LN_SBPLX,
            "xtol_rel": 1e-6,
            "ftol_rel": 1e-6,
            "initial_step": 1e-3,
            "max_eval": 100,
            "loss_fcn_param": [alpha, 1.0],
            "is_P_limit": [False, False],
        }

        return falloff_parameters(T, M, ln_k, x0_lna, algo), T, M, ln_k

    def test_alpha_2_objective_is_sum_squared_residuals(self):
        """L2 path: objective at any x_fit must equal ``sum(resid**2)``."""
        fp, T, M, ln_k = self._make_fp(alpha=2.0)
        fp.p0 = fp.x0.copy()
        fp.p0[-3:] = fp.convert_Fcent(fp.p0[-3:], "base2opt")
        fp.p0 = fp.p0[fp.alter_idx]
        fp.s = np.ones_like(fp.p0)
        x_fit = np.zeros_like(fp.p0)

        obj_value = fp.objective(x_fit)

        x_full = fp.set_x_from_opt(x_fit)
        resid = (ln_Troe(T, M, *x_full) - ln_k).flatten()
        expected = float(np.sum(resid * resid))
        assert obj_value == pytest.approx(expected, rel=1e-15)

    def test_alpha_2_bypasses_adaptive_weights(self, monkeypatch):
        """``adaptive_weights`` must not be called when ``loss_alpha == 2``."""
        import frhodo.simulation.mechanism.troe as troe_mod

        call_count = {"n": 0}

        def _spy(*args, **kwargs):
            call_count["n"] += 1
            raise RuntimeError("adaptive_weights should not be called for alpha=2")

        monkeypatch.setattr(troe_mod, "adaptive_weights", _spy)

        fp, _, _, _ = self._make_fp(alpha=2.0)
        fp.p0 = fp.x0.copy()
        fp.p0[-3:] = fp.convert_Fcent(fp.p0[-3:], "base2opt")
        fp.p0 = fp.p0[fp.alter_idx]
        fp.s = np.ones_like(fp.p0)

        fp.objective(np.zeros_like(fp.p0))
        fp.calc_s(np.zeros_like(fp.p0))

        assert call_count["n"] == 0

    def test_alpha_not_2_still_uses_adaptive_weights(self, monkeypatch):
        """Non-default alpha must route through the robust-loss path."""
        import frhodo.simulation.mechanism.troe as troe_mod

        call_count = {"n": 0}
        real_aw = troe_mod.adaptive_weights

        def _spy(*args, **kwargs):
            call_count["n"] += 1
            return real_aw(*args, **kwargs)

        monkeypatch.setattr(troe_mod, "adaptive_weights", _spy)

        fp, _, _, _ = self._make_fp(alpha=1.0)
        fp.p0 = fp.x0.copy()
        fp.p0[-3:] = fp.convert_Fcent(fp.p0[-3:], "base2opt")
        fp.p0 = fp.p0[fp.alter_idx]
        fp.s = np.ones_like(fp.p0)

        fp.objective(np.zeros_like(fp.p0))

        assert call_count["n"] == 1


class TestSetXFromOptCache:
    """``set_x_from_opt`` memoizes on its input so the three calls per
    inner query (objective + two constraints with the same x_fit)
    collapse to one compute + two cache hits."""

    def _prep_fp(self):
        T, M, ln_k, _ = _synthesize_troe_surface()
        ms = multistart_nn(T, M, ln_k)
        x0_lna = ms["x"].copy()
        x0_lna[1] = np.log(x0_lna[1])
        x0_lna[4] = np.log(x0_lna[4])
        algo = {
            "algorithm": nlopt.LN_SBPLX,
            "xtol_rel": 1e-6, "ftol_rel": 1e-6,
            "initial_step": 1e-3, "max_eval": 100,
            "loss_fcn_param": [2.0, 1.0], "is_P_limit": [False, False],
        }
        fp = falloff_parameters(T, M, ln_k, x0_lna, algo)
        fp.p0 = fp.x0.copy()
        fp.p0[-3:] = fp.convert_Fcent(fp.p0[-3:], "base2opt")
        fp.p0 = fp.p0[fp.alter_idx]
        fp.s = np.ones_like(fp.p0)

        return fp

    def test_repeated_call_returns_equal_result(self):
        """Cache hit must match the freshly-computed value."""
        fp = self._prep_fp()
        x_fit = np.full(len(fp.p0), 0.5)
        first = fp.set_x_from_opt(x_fit)
        second = fp.set_x_from_opt(x_fit)
        np.testing.assert_array_equal(first, second)

    def test_cache_miss_when_input_changes(self):
        """Different x_fit content must recompute, not return stale val."""
        fp = self._prep_fp()
        x_fit_a = np.full(len(fp.p0), 0.5)
        x_fit_b = np.full(len(fp.p0), 0.6)
        result_a_first = fp.set_x_from_opt(x_fit_a)
        _ = fp.set_x_from_opt(x_fit_b)
        result_a_after = fp.set_x_from_opt(x_fit_a)
        np.testing.assert_array_equal(result_a_first, result_a_after)
        assert not np.array_equal(result_a_first, fp.set_x_from_opt(x_fit_b))

    def test_returned_array_is_safe_to_mutate(self):
        """Caller mutation of the returned array must not corrupt the cache."""
        fp = self._prep_fp()
        x_fit = np.full(len(fp.p0), 0.5)
        first = fp.set_x_from_opt(x_fit)
        first[1] = 999.0
        second = fp.set_x_from_opt(x_fit)
        assert second[1] != 999.0

    def test_cache_avoids_recompute_under_repeated_call(self, monkeypatch):
        """Spy on ``set_x_from_opt_kernel`` — it should run once across N
        calls with identical x_fit, not N times."""
        from frhodo.simulation.mechanism import troe as troe_mod

        fp = self._prep_fp()
        call_count = {"n": 0}
        real_kernel = troe_mod.set_x_from_opt_kernel

        def _spy(*args, **kwargs):
            call_count["n"] += 1
            return real_kernel(*args, **kwargs)

        monkeypatch.setattr(troe_mod, "set_x_from_opt_kernel", _spy)
        x_fit = np.full(len(fp.p0), 0.5)
        fp.set_x_from_opt(x_fit)
        fp.set_x_from_opt(x_fit)
        fp.set_x_from_opt(x_fit)

        assert call_count["n"] == 1


class TestDataAwareBounds:
    """``_data_aware_bounds`` returns the static box when the worst-case
    ``ln_k_0`` at the corner stays below ``max_ln_val``, and a narrower
    box otherwise."""

    @staticmethod
    def _corner_lnk(T_lo, T_hi, low, high):
        """Worst-case |ln_k_0| at the bound corners over [T_lo, T_hi]."""
        ln_A_box = max(abs(low[1]), abs(high[1]))
        n_box = max(abs(low[2]), abs(high[2]))
        ea_box = max(abs(low[0]), abs(high[0]))

        return ln_A_box + n_box * np.log(T_hi) + ea_box / (Ru * T_lo)

    def test_warm_grid_keeps_static_ea_bound(self):
        """Combustion-range T (>= 800 K) — static Ea bounds already safe."""
        T = np.array([[800.0, 1500.0, 2500.0], [800.0, 1500.0, 2500.0]])
        low, high = _data_aware_bounds(T)
        assert low[0] == pytest.approx(_BOUNDS_LOW[0])
        assert high[0] == pytest.approx(_BOUNDS_HIGH[0])

    def test_cold_grid_narrows_ea_bound(self):
        """Cold T floor (50 K) — static Ea would let ln_k_0 escape the clamp."""
        T = np.array([[50.0, 200.0]])
        low, high = _data_aware_bounds(T)
        assert abs(high[0]) < abs(_BOUNDS_HIGH[0])
        assert low[0] == -high[0]

    def test_returned_bounds_keep_corner_lnk_below_max(self):
        """Whatever T is, the corner ln_k_0 stays inside ``max_ln_val``."""
        for T_lo, T_hi in [(50.0, 200.0), (300.0, 800.0), (1000.0, 3000.0)]:
            T = np.array([[T_lo, T_hi]])
            low, high = _data_aware_bounds(T)
            corner = self._corner_lnk(T_lo, T_hi, low, high)
            assert corner <= max_ln_val, (
                f"T=[{T_lo},{T_hi}]: corner ln_k_0={corner:.2f} exceeds "
                f"max_ln_val={max_ln_val:.2f}"
            )

    def test_non_ea_bounds_unchanged(self):
        T = np.array([[50.0, 200.0]])
        low, high = _data_aware_bounds(T)
        for i in (1, 2, 4, 5, 6, 7, 8, 9):
            assert low[i] == _BOUNDS_LOW[i]
            assert high[i] == _BOUNDS_HIGH[i]


# Per-slot FD step sizes for the analytical-Jacobian comparison.
# Optimal for 4th-order central FD is h ~ eps_mach^(1/5) * |x| ~ 1e-3 * |x|;
# we use 1e-5 * |x| so truncation error ~ 1e-20 and float64 round-off
# ~ 1e-16 * scale dominates. Absolute floors protect when x is near zero.
_JAC_REL_H = np.array([
    1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-6, 1e-5, 1e-5, 1e-5,
])
_JAC_ABS_H = np.array([
    1e-3, 1e-6, 1e-6, 1e-3, 1e-6, 1e-6, 1e-6, 1e-3, 1e-3, 1e-3,
])


def _fd4_central(T, M, x_lna, k, h):
    """4th-order central FD partial of ln_Troe w.r.t. the k-th param."""
    x_p1 = x_lna.copy(); x_p1[k] += h
    x_p2 = x_lna.copy(); x_p2[k] += 2 * h
    x_m1 = x_lna.copy(); x_m1[k] -= h
    x_m2 = x_lna.copy(); x_m2[k] -= 2 * h
    f_p2 = ln_Troe(T, M, *x_p2)
    f_p1 = ln_Troe(T, M, *x_p1)
    f_m1 = ln_Troe(T, M, *x_m1)
    f_m2 = ln_Troe(T, M, *x_m2)

    return (-f_p2 + 8 * f_p1 - 8 * f_m1 + f_m2) / (12 * h)


def _smooth_jacobian_cases(n_cases=5, seed=42):
    """Random warm-grid cases with params in ln_Troe's smooth (un-clamped) region.

    ln_Troe clamps ln_k_0 / ln_k_inf when they hit numerical extrema, but
    ln_Troe_jac assumes the un-clamped function. Sampling Ea / |ln_A| /
    grid T inside a safe envelope keeps the comparison meaningful.
    """
    rng = np.random.default_rng(seed)
    cases = []
    for _ in range(n_cases):
        n_P = int(rng.integers(3, 7))
        n_T = int(rng.integers(5, 12))
        T_lo = float(rng.uniform(800.0, 1200.0))
        T_hi = float(rng.uniform(1500.0, 2500.0))
        T_1d = np.linspace(T_lo, T_hi, n_T)
        T = np.tile(T_1d, (n_P, 1)).astype(np.float64)
        P_levels = np.geomspace(
            rng.uniform(1e-8, 1e-7), rng.uniform(1e-4, 1e-3), n_P,
        )
        M = np.tile(P_levels[:, None], (1, n_T)).astype(np.float64)
        x_lna = np.array([
            rng.uniform(-2e5, 2e5),
            rng.uniform(np.log(1e8), np.log(1e14)),
            rng.uniform(-1.5, 1.5),
            rng.uniform(-2e5, 2e5),
            rng.uniform(np.log(1e10), np.log(1e14)),
            rng.uniform(-1.5, 1.5),
            rng.uniform(0.2, 0.8),
            rng.uniform(100.0, 3000.0),
            rng.uniform(100.0, 3000.0),
            rng.uniform(500.0, 8000.0),
        ])
        cases.append((T, M, x_lna))

    return cases


_PARAM_NAMES = [
    "Ea_0", "ln_A_0", "n_0", "Ea_inf", "ln_A_inf", "n_inf",
    "A_Fc", "T3", "T1", "T2",
]


class TestLnTroeJacobian:
    """Analytic ``ln_Troe_jac`` agrees with high-accuracy FD on the smooth manifold."""

    @pytest.fixture(scope="class")
    def cases(self):
        return _smooth_jacobian_cases()

    @pytest.mark.parametrize("slot", range(10), ids=_PARAM_NAMES)
    def test_partial_matches_fd_mean_sup_rel_below_1e_minus_5(self, cases, slot):
        sup_rels = []
        for T, M, x_lna in cases:
            _, J_a = ln_Troe_jac(T, M, *x_lna)
            J_a_slot = J_a[:, :, slot]
            h = max(_JAC_REL_H[slot] * abs(x_lna[slot]), _JAC_ABS_H[slot])
            J_n_slot = _fd4_central(T, M, x_lna, slot, h)
            diff_max = float(np.abs(J_a_slot - J_n_slot).max())
            ref = float(np.abs(J_n_slot).max())
            sup_rels.append(diff_max / max(ref, 1e-30))
        mean_sup_rel = float(np.mean(sup_rels))

        assert mean_sup_rel < 1e-5, (
            f"{_PARAM_NAMES[slot]}: mean sup-rel FD disagreement "
            f"{mean_sup_rel:.3e} exceeds 1e-5 — analytical Jacobian drifted "
            f"from ln_Troe"
        )

    def test_forward_pass_matches_ln_troe(self, cases):
        for T, M, x_lna in cases:
            ln_k_direct = ln_Troe(T, M, *x_lna)
            ln_k_jac, _ = ln_Troe_jac(T, M, *x_lna)
            np.testing.assert_allclose(
                ln_k_jac, ln_k_direct, rtol=1e-12, atol=1e-14,
                err_msg="ln_Troe_jac's forward pass diverged from ln_Troe",
            )

    def test_jacobian_shape_matches_grid_and_param_count(self, cases):
        for T, M, x_lna in cases:
            _, J = ln_Troe_jac(T, M, *x_lna)
            assert J.shape == (T.shape[0], T.shape[1], 10)


class TestPhysicallyValid:
    """``_physically_valid`` rejects param vectors with Fcent outside
    ``[_PHYS_FCENT_MIN, 1]`` at the data T extrema, non-positive A_inf,
    or ln_k_inf exceeding the production rate cap."""

    T_GRID_2D = np.array([[800.0, 1500.0, 2500.0]])
    GOOD = np.array([
        1.5e8, 5e15, -1.0,
        1.0e8, 1e13, 0.5,
        0.7, 250.0, 1500.0, 4000.0,
    ])

    def test_good_params_accepted(self):
        assert _physically_valid(self.GOOD, self.T_GRID_2D)

    def test_fcent_above_one_rejected(self):
        x = self.GOOD.copy()
        x[6] = 0.0   # A_Fc=0; T2 contribution dominates → Fcent > 1
        x[9] = 1.0   # T2 small enough that exp(-T2/T) is close to 1
        assert not _physically_valid(x, self.T_GRID_2D)

    def test_non_positive_A_inf_rejected(self):
        x = self.GOOD.copy()
        x[4] = -1.0
        assert not _physically_valid(x, self.T_GRID_2D)

    def test_rate_cap_violation_rejected(self):
        x = self.GOOD.copy()
        x[4] = 1e80
        assert not _physically_valid(x, self.T_GRID_2D)


class TestSkipPolishGate:
    """``Troe.fit`` skips the AUGLAG-SBPLX polish when the multistart
    log-RMS is below ``_POLISH_SKIP_THRESH`` and the params are
    physically valid."""

    @staticmethod
    def _build_troe():
        """Synthetic Troe surface — multistart fits it well, so the
        gate has a chance to fire."""
        T_vals = np.array([800.0, 1500.0, 2200.0])
        P_atm = np.array([0.1, 1.0, 10.0])
        n_P, n_T = len(P_atm), len(T_vals)
        T_2d = np.tile(T_vals, (n_P, 1))
        P_2d = np.tile(P_atm.reshape(-1, 1), (1, n_T))
        M_2d = (P_2d * 101325.0) / (Ru * T_2d)
        truth_capture = np.array([
            1.5e8, 5e15, -1.0,
            1.0e8, 1e13, 0.5,
            0.7, 250.0, 1500.0, 4000.0,
        ])
        truth_lna = truth_capture.copy()
        truth_lna[1] = np.log(truth_capture[1])
        truth_lna[4] = np.log(truth_capture[4])
        rates = np.exp(ln_Troe(T_2d, M_2d, *truth_lna))

        return Troe(
            rates=rates.flatten(),
            T=T_2d.flatten(),
            P=(P_2d * 101325.0).flatten(),
            M=lambda T_arg, P_arg: (P_arg / (Ru * T_arg)),
            x0=truth_capture,
            coefNames=default_Troe_coefNames,
            is_falloff_limit=np.zeros(10, dtype=bool),
        )

    def test_zero_threshold_runs_polish(self, monkeypatch):
        """``_POLISH_SKIP_THRESH = 0`` disables the gate — every fit runs polish."""
        import frhodo.simulation.mechanism.troe as troe_mod

        polish_calls = {"n": 0}
        real_fit = troe_mod.falloff_parameters.fit

        def _spy(self_fp):
            polish_calls["n"] += 1
            return real_fit(self_fp)

        monkeypatch.setattr(troe_mod.falloff_parameters, "fit", _spy)
        monkeypatch.setattr(troe_mod, "_POLISH_SKIP_THRESH", 0.0)
        troe = self._build_troe()
        troe.fit()
        assert polish_calls["n"] == 1

    def test_low_rms_skips_polish(self, monkeypatch):
        """With threshold above ``ms['fval']`` and the validity check
        passing, the gate fires and polish is skipped."""
        import frhodo.simulation.mechanism.troe as troe_mod

        polish_calls = {"n": 0}
        real_fit = troe_mod.falloff_parameters.fit

        def _spy(self_fp):
            polish_calls["n"] += 1
            return real_fit(self_fp)

        monkeypatch.setattr(troe_mod.falloff_parameters, "fit", _spy)
        monkeypatch.setattr(troe_mod, "_POLISH_SKIP_THRESH", 10.0)
        # NN output for this synthetic surface lands in a Fcent-overparameterized
        # region that fails the physical check; force it True to exercise the
        # gate-fires path. The True-path of _physically_valid is covered by
        # ``TestPhysicallyValid``.
        monkeypatch.setattr(troe_mod, "_physically_valid", lambda *_a, **_k: True)
        troe = self._build_troe()
        result = troe.fit()
        assert polish_calls["n"] == 0
        assert result.shape == (10,)

    def test_invalid_params_still_polish_even_under_threshold(self, monkeypatch):
        """An NN seed that would be physically invalid forces polish even
        if the RMS happens to be below threshold."""
        import frhodo.simulation.mechanism.troe as troe_mod

        polish_calls = {"n": 0}
        real_fit = troe_mod.falloff_parameters.fit

        def _spy(self_fp):
            polish_calls["n"] += 1
            return real_fit(self_fp)

        def _always_invalid(*_args, **_kwargs):
            return False

        monkeypatch.setattr(troe_mod.falloff_parameters, "fit", _spy)
        monkeypatch.setattr(troe_mod, "_POLISH_SKIP_THRESH", 10.0)
        monkeypatch.setattr(troe_mod, "_physically_valid", _always_invalid)
        troe = self._build_troe()
        troe.fit()
        assert polish_calls["n"] == 1


class TestSkipTrfGate:
    """``multistart_nn`` skips the multistart-TRF refinement when the NN's
    top-conf ``predicted_rms`` is below ``_TRF_SKIP_THRESH`` and the
    seed's measured residual confirms it."""

    @staticmethod
    def _build_inputs():
        """Synthetic Troe surface fed directly to ``multistart_nn`` as
        flat 1D arrays."""
        T_vals = np.array([800.0, 1500.0, 2200.0])
        P_atm = np.array([0.1, 1.0, 10.0])
        n_P, n_T = len(P_atm), len(T_vals)
        T_2d = np.tile(T_vals, (n_P, 1))
        P_2d = np.tile(P_atm.reshape(-1, 1), (1, n_T))
        M_2d = (P_2d * 101325.0) / (Ru * T_2d)
        truth_capture = np.array([
            1.5e8, 5e15, -1.0,
            1.0e8, 1e13, 0.5,
            0.7, 250.0, 1500.0, 4000.0,
        ])
        truth_lna = truth_capture.copy()
        truth_lna[1] = np.log(truth_capture[1])
        truth_lna[4] = np.log(truth_capture[4])
        ln_k = ln_Troe(T_2d, M_2d, *truth_lna)

        return T_2d, M_2d, ln_k

    def test_zero_threshold_runs_trf(self, monkeypatch):
        """``_TRF_SKIP_THRESH = 0`` disables the gate — TRF always runs."""
        import frhodo.simulation.mechanism.troe as troe_mod

        trf_calls = {"n": 0}
        real_trf = troe_mod._trf_from_start

        def _spy(*args, **kwargs):
            trf_calls["n"] += 1
            return real_trf(*args, **kwargs)

        monkeypatch.setattr(troe_mod, "_trf_from_start", _spy)
        monkeypatch.setattr(troe_mod, "_TRF_SKIP_THRESH", 0.0)
        T, M, ln_k = self._build_inputs()
        ms = multistart_nn(T, M, ln_k)
        assert trf_calls["n"] >= 1
        assert ms["k_refined"] >= 1

    def test_high_threshold_skips_trf(self, monkeypatch):
        """When predicted_rms and actual NN-seed RMS are both below the
        threshold, the gate fires and TRF is skipped entirely."""
        import frhodo.simulation.mechanism.troe as troe_mod

        trf_calls = {"n": 0}
        real_trf = troe_mod._trf_from_start

        def _spy(*args, **kwargs):
            trf_calls["n"] += 1
            return real_trf(*args, **kwargs)

        monkeypatch.setattr(troe_mod, "_trf_from_start", _spy)
        monkeypatch.setattr(troe_mod, "_TRF_SKIP_THRESH", 1e6)
        T, M, ln_k = self._build_inputs()
        ms = multistart_nn(T, M, ln_k)
        assert trf_calls["n"] == 0
        assert ms["k_refined"] == 0
        assert ms["nfev"] == 0

    def test_invalid_seed_forces_trf_even_under_threshold(self, monkeypatch):
        """A predicted-confident-but-physically-invalid NN seed still
        triggers TRF refinement instead of being returned directly."""
        import frhodo.simulation.mechanism.troe as troe_mod

        trf_calls = {"n": 0}
        real_trf = troe_mod._trf_from_start

        def _spy(*args, **kwargs):
            trf_calls["n"] += 1
            return real_trf(*args, **kwargs)

        monkeypatch.setattr(troe_mod, "_trf_from_start", _spy)
        monkeypatch.setattr(troe_mod, "_TRF_SKIP_THRESH", 1e6)
        monkeypatch.setattr(troe_mod, "_physically_valid", lambda *_a, **_k: False)
        T, M, ln_k = self._build_inputs()
        ms = multistart_nn(T, M, ln_k)
        assert trf_calls["n"] >= 1
        assert ms["k_refined"] >= 1


class TestFusedKernels:
    """``set_x_from_opt_kernel`` and the three callback kernels pin the
    polish-inner-loop math now living in Numba. These tests check each
    kernel against its closed-form expected value so a Numba port that
    silently drifts is caught at unit-test time."""

    def test_set_x_from_opt_kernel_recovers_p0_at_origin(self):
        """``x_fit_opt = 0`` should give result[slot] == p0[slot] for
        Arrhenius slots, and the bisymlog inverse of p0 for Fcent slots."""
        x_state = np.array([
            5e7, 30.0, -1.0,
            4e7, 25.0, 0.5,
            0.7, 100.0, 200.0, 1500.0,
        ])
        alter_idx = np.arange(10, dtype=np.int64)
        p0 = x_state.copy()
        s = np.ones(10)
        x_fit_opt = np.zeros(10)

        out = set_x_from_opt_kernel(
            x_fit_opt, alter_idx, p0, s, x_state, bisymlog_C, bisymlog_base,
        )

        # Arrhenius slots: identity.
        np.testing.assert_allclose(out[:7], p0[:7])
        # Fcent T-slots inverse-transformed and (T3, T1) clamped > 1e8 → 1e30.
        for j in (7, 8):
            assert out[j] == 1e30, (
                f"slot {j} expected clamp to 1e30 (large positive opt "
                f"value), got {out[j]}"
            )
        expected_T2 = bisymlog_C * (bisymlog_base ** p0[9] - 1.0)
        assert np.isinf(out[9]) or out[9] == pytest.approx(expected_T2, rel=1e-12)

    def test_set_x_from_opt_kernel_small_T3_T1_clamped_to_1e_neg30(self):
        """Slots 7, 8 below 10 (after bisymlog inverse) clamp to 1e-30 —
        a Lindemann-limit guard preserved from the Python version."""
        # Choose p0 so that bisymlog_inverse(p0[7..9]) is in (0, 10).
        # bisymlog_inverse(y) = C*(e^y - 1); for output < 10 with C ≈ 0.58,
        # need e^y - 1 < 10/C ≈ 17.2 → y < ln(18.2) ≈ 2.9.
        x_state = np.array([
            5e7, 30.0, -1.0, 4e7, 25.0, 0.5,
            0.7, 1.0, 1.0, 0.0,
        ])
        alter_idx = np.arange(10, dtype=np.int64)
        p0 = x_state.copy()
        s = np.ones(10)
        x_fit_opt = np.zeros(10)

        out = set_x_from_opt_kernel(
            x_fit_opt, alter_idx, p0, s, x_state, bisymlog_C, bisymlog_base,
        )

        assert out[7] == 1e-30
        assert out[8] == 1e-30

    def test_objective_l2_kernel_is_zero_when_pred_equals_target(self):
        """``ln_k == ln_Troe(x)`` gives zero residual sum."""
        x_lna = np.array([
            5e7, np.log(1e12), -1.0,
            4e7, np.log(1e10), 0.5,
            0.7, 250.0, 1500.0, 4000.0,
        ])
        T = np.array([[1000.0, 1500.0, 2000.0], [1000.0, 1500.0, 2000.0]])
        M = _make_M(T, P=1e5)
        ln_k = ln_Troe(T, M, *x_lna)

        out = objective_l2_kernel(T, M, ln_k, x_lna)

        assert out == pytest.approx(0.0, abs=1e-20)

    def test_objective_l2_kernel_matches_python_sum(self):
        """Closed-form check: kernel output equals sum((ln_Troe - ln_k)^2)."""
        x_lna = np.array([
            5e7, np.log(1e12), -1.0,
            4e7, np.log(1e10), 0.5,
            0.7, 250.0, 1500.0, 4000.0,
        ])
        T = np.array([[1000.0, 1500.0, 2000.0], [1000.0, 1500.0, 2000.0]])
        M = _make_M(T, P=1e5)
        ln_k = np.zeros_like(T)

        resid_sq_sum = float(np.sum((ln_Troe(T, M, *x_lna) - ln_k) ** 2))
        out = objective_l2_kernel(T, M, ln_k, x_lna)

        assert out == pytest.approx(resid_sq_sum, rel=1e-12)

    def test_Fcent_constraint_kernel_feasible_when_Fcent_in_unit_interval(self):
        """Fcent(T) ∈ (Fcent_min, 1) at Tmin, Tmax → constraint < 0."""
        x = np.array([
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.5, 1000.0, 1000.0, 500.0,
        ])
        out = Fcent_constraint_kernel(x, 273.0, 6000.0, 1e-6)
        Fcent = Fcent_calc(np.array([273.0, 6000.0]), 0.5, 1000.0, 1000.0, 500.0)
        expected = max(
            (1e-6 - Fcent).max(),
            (Fcent - 1.0).max(),
        ) * 1e8

        assert out == pytest.approx(expected, rel=1e-12)
        assert out < 0, f"feasible Fcent should give negative constraint, got {out}"

    def test_Arrhenius_constraint_kernel_matches_max_over_row(self):
        """Kernel returns max(ln_k_0, ln_k_inf) - ln_k_max across all
        temperatures in the row passed in."""
        x = np.array([
            5e7, np.log(1e12), -1.0,
            4e7, np.log(1e10), 0.5,
            0.7, 250.0, 1500.0, 4000.0,
        ])
        T_row = np.array([1000.0, 1500.0, 2000.0])

        out = Arrhenius_constraint_kernel(x, T_row, ln_k_max)

        Ea_0, ln_A_0, n_0 = x[0], x[1], x[2]
        Ea_inf, ln_A_inf, n_inf = x[3], x[4], x[5]
        ln_k_0 = ln_A_0 + n_0 * np.log(T_row) - Ea_0 / (Ru * T_row)
        ln_k_inf = ln_A_inf + n_inf * np.log(T_row) - Ea_inf / (Ru * T_row)
        expected = float(np.max([ln_k_0, ln_k_inf])) - ln_k_max

        assert out == pytest.approx(expected, rel=1e-12)


class TestNumpyNN:
    """The production NN runs in pure numpy. These tests pin the output
    mapping, the forward-pass shape contract, and the round-trip between
    capture-form physical params and the normalized [-1, 1] training target."""

    def test_normalized_to_capture_at_zero_is_neutral(self):
        """``norm == 0`` should map to physically neutral values:
        Ea, n, T2 to 0; lnA to exp(0) = 1; T3, T1 to 10^0 = 1;
        A_Fc to its sigmoid identity value (already 0 in normalized space,
        which corresponds to sigmoid'd output 0)."""
        out = normalized_to_capture_np(np.zeros((1, 10)))[0]
        assert out[0] == pytest.approx(0.0)            # Ea_0
        assert out[1] == pytest.approx(1.0)            # A_0 = exp(0)
        assert out[2] == pytest.approx(0.0)            # n_0
        assert out[3] == pytest.approx(0.0)            # Ea_inf
        assert out[4] == pytest.approx(1.0)            # A_inf
        assert out[5] == pytest.approx(0.0)            # n_inf
        assert out[6] == pytest.approx(0.0)            # A_Fc (passes through)
        assert out[7] == pytest.approx(1.0)            # T3 = 10^0
        assert out[8] == pytest.approx(1.0)            # T1 = 10^0
        assert out[9] == pytest.approx(0.0)            # T2

    def test_normalized_to_capture_boundary_ranges(self):
        """At ``norm == +1`` and ``norm == -1`` the mapping should reach
        the expected physical ranges that C1 was sized for."""
        plus = normalized_to_capture_np(np.ones((1, 10)))[0]
        minus = normalized_to_capture_np(-np.ones((1, 10)))[0]

        assert plus[0] == pytest.approx(3e9)
        assert minus[0] == pytest.approx(-3e9)
        assert plus[1] == pytest.approx(np.exp(100.0))
        assert minus[1] == pytest.approx(np.exp(-100.0))
        assert plus[2] == pytest.approx(50.0)
        assert minus[2] == pytest.approx(-50.0)
        assert plus[7] == pytest.approx(1e15)
        assert minus[7] == pytest.approx(1e-15)
        assert plus[9] == pytest.approx(1e5 - 1.0)
        assert minus[9] == pytest.approx(-(1e5 - 1.0))

    def test_capture_normalized_round_trip(self):
        """``capture_to_normalized_np`` must be the inverse of
        ``normalized_to_capture_np`` away from the boundary."""
        rng = np.random.default_rng(0)
        norm_in = rng.uniform(-0.95, 0.95, size=(50, 10))
        norm_in[:, 6] = rng.uniform(0.02, 0.98, size=50)  # slot 6 already in (0, 1)
        capture = normalized_to_capture_np(norm_in)
        norm_back = capture_to_normalized_np(capture)
        np.testing.assert_allclose(norm_in, norm_back, rtol=1e-8, atol=1e-9)

    def test_raw_to_normalized_slot_6_uses_sigmoid(self):
        """Slot 6 (A_Fc) must use sigmoid; other slots use tanh."""
        raw = np.zeros((1, 10))
        raw[0, 6] = 10.0  # large positive: sigmoid → ~1
        out = raw_to_normalized_np(raw)
        assert out[0, 6] > 0.999, "slot 6 should be sigmoid-bounded into (0, 1)"
        # Other slots stay at tanh(0) = 0
        for j in (0, 1, 2, 3, 4, 5, 7, 8, 9):
            assert abs(out[0, j]) < 1e-12, f"slot {j} should be tanh(0) = 0"

    def test_forward_shapes(self):
        """The numpy network must return ``(preds, log_σ, predicted_rms,
        conf)`` with the contracted shapes."""
        reset_model_cache()
        model, _stats = get_model()
        feats = np.zeros((2, 9, 3), dtype=np.float64)
        mask = np.ones((2, 9), dtype=bool)
        preds, log_sig, pred_rms, conf = model.forward(feats, mask)
        assert preds.shape == (2, K_CANDIDATES, 10)
        assert log_sig.shape == (2, K_CANDIDATES, 10)
        assert pred_rms.shape == (2, K_CANDIDATES)
        assert conf.shape == (2, K_CANDIDATES)

    def test_forward_bounds(self):
        """preds within the normalized contract; predicted_rms > 0."""
        reset_model_cache()
        model, _stats = get_model()
        rng = np.random.default_rng(1)
        feats = rng.standard_normal((1, 9, 3))
        mask = np.ones((1, 9), dtype=bool)
        preds, _log_sig, pred_rms, _conf = model.forward(feats, mask)
        # Slots 0-5, 7-9 use tanh → [-1, 1]; slot 6 uses sigmoid → (0, 1).
        tanh_slots = [0, 1, 2, 3, 4, 5, 7, 8, 9]
        assert (preds[..., tanh_slots] >= -1.0).all() and (preds[..., tanh_slots] <= 1.0).all()
        assert (preds[..., 6] >= 0.0).all() and (preds[..., 6] <= 1.0).all()
        assert (pred_rms > 0).all(), "softplus output must be strictly positive"

    def test_forward_is_deterministic(self):
        """No dropout at inference: identical inputs produce identical outputs."""
        reset_model_cache()
        model, _stats = get_model()
        rng = np.random.default_rng(2)
        feats = rng.standard_normal((1, 9, 3))
        mask = np.ones((1, 9), dtype=bool)
        a = model.forward(feats, mask)
        b = model.forward(feats, mask)
        for arr_a, arr_b in zip(a, b):
            np.testing.assert_array_equal(arr_a, arr_b)

    def test_mask_changes_output(self):
        """The mask actually gates points — masked-out points must not
        contribute to the pooled representation."""
        reset_model_cache()
        model, _stats = get_model()
        rng = np.random.default_rng(3)
        feats = rng.standard_normal((1, 9, 3))
        mask_all = np.ones((1, 9), dtype=bool)
        mask_half = np.array([[True] * 5 + [False] * 4], dtype=bool)
        out_all = model.forward(feats, mask_all)[0]
        out_half = model.forward(feats, mask_half)[0]
        assert not np.allclose(out_all, out_half), "mask must affect the pool"
