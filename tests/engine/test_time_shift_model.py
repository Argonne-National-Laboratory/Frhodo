"""Elastic-net global time-shift model (parametric t-uncertainty)."""
import numpy as np
import pytest

from frhodo.optimize.time_shift_model import build_shift_features, regularized_shifts



def _synthetic_conditions(n, rng, *, noise_species=True):
    """n shocks: T,P vary; AR is a constant bath; fuel varies; optional noise."""
    T = rng.uniform(1000.0, 2000.0, n)
    P = rng.uniform(0.5, 2.0, n)
    fuel = rng.uniform(0.01, 0.05, n)
    conditions = []
    for i in range(n):
        mix = {"AR": 0.96, "fuel": float(fuel[i])}
        if noise_species:
            mix["noise"] = float(rng.uniform(0.0, 0.1))
        conditions.append((float(T[i]), float(P[i]), mix))

    return conditions, T


class TestBuildShiftFeatures:
    def test_drops_constant_species(self):
        conds, _ = _synthetic_conditions(10, np.random.default_rng(0))
        _, names = build_shift_features(conds)
        assert "AR" not in names, "constant bath gas should be dropped"
        assert "fuel" in names and "noise" in names

    def test_includes_quadratic_TP_terms(self):
        conds, _ = _synthetic_conditions(8, np.random.default_rng(1))
        X, names = build_shift_features(conds)
        assert names[:5] == ["T", "P", "T^2", "P^2", "T*P"]
        assert X.shape == (8, len(names))


class TestRegularizedShifts:
    def test_recovers_linear_T_trend(self):
        rng = np.random.default_rng(0)
        conds, T = _synthetic_conditions(40, rng)
        dt_true = 1.0e-6 + 2.0e-6 * (T - 1500.0) / 500.0
        t_star = dt_true + rng.normal(0.0, 1.0e-7, T.size)

        shifts, info = regularized_shifts(conds, t_star, t_unc=1.0e-5)

        assert info["model"] is not None
        r = np.corrcoef(shifts, dt_true)[0, 1]
        assert r > 0.9, f"prediction should track the true trend (r={r:.3f})"

    def test_shrinks_irrelevant_feature(self):
        rng = np.random.default_rng(0)
        conds, T = _synthetic_conditions(40, rng)
        dt_true = 1.0e-6 + 2.0e-6 * (T - 1500.0) / 500.0
        t_star = dt_true + rng.normal(0.0, 1.0e-7, T.size)

        _, info = regularized_shifts(conds, t_star, t_unc=1.0e-5)
        names = info["feature_names"]
        coef = np.abs(info["coefficients"])
        noise_c = coef[names.index("noise")]
        T_c = coef[names.index("T")]
        assert noise_c < 0.1 * T_c, (
            f"elastic net should suppress the unrelated species "
            f"(noise={noise_c:.2e}, T={T_c:.2e})"
        )

    def test_predictions_clamped_to_t_unc(self):
        rng = np.random.default_rng(2)
        conds, T = _synthetic_conditions(30, rng)
        t_star = 1.0e-3 * (T - 1500.0)  # huge shifts, far outside the bound
        t_unc = 5.0e-6

        shifts, _ = regularized_shifts(conds, t_star, t_unc=t_unc)
        assert np.all(np.abs(shifts) <= t_unc + 1e-15)

    def test_too_few_shocks_falls_back(self):
        conds, _ = _synthetic_conditions(2, np.random.default_rng(3))
        t_star = np.array([3.0e-6, -3.0e-6])
        shifts, info = regularized_shifts(conds, t_star, t_unc=1.0e-6)
        assert info["model"] is None
        assert np.all(np.abs(shifts) <= 1.0e-6 + 1e-15)
