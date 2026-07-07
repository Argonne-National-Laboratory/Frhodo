"""``Optimize.run`` stage plumbing: the global → local handoff."""
import types

import nlopt
import numpy as np
import pytest

from frhodo.optimize.algorithms import Optimize



CENTER = np.array([0.3, -0.2])


class _FakeCostFunction:
    """Records stage-end evaluations and floor recalibrations."""

    def __init__(self):
        self.i = 0
        self.opt_type = "local"
        self.recalibrated_at = []

    def __call__(self, s, optimizing=False):
        s = np.asarray(s)
        fval = float(np.sum((s - CENTER) ** 2))
        result = (fval, s, "shocks")

        return result

    def calibrate_error_floor(self, s):
        self.recalibrated_at.append(np.asarray(s, dtype=float).copy())


def _options(run_global=True):
    stage = {
        "algorithm": nlopt.LN_SBPLX,
        "initial_step": 0.1,
        "stop_criteria_type": "Iteration Maximum",
        "stop_criteria_val": 80,
        "xtol_rel": 1e-8,
        "ftol_rel": 1e-10,
        "initial_pop_multiplier": 1.0,
    }
    options = {
        "global": dict(stage, run=run_global),
        "local": dict(stage, run=True),
    }

    return options


def _run(run_global=True):
    fake = _FakeCostFunction()
    evals = []

    def obj_fcn(s, grad=None):
        evals.append((fake.opt_type, np.asarray(s, dtype=float).copy()))

        return float(np.sum((np.asarray(s) - CENTER) ** 2))

    bnds = {"lower": np.full(2, -1.0), "upper": np.full(2, 1.0)}
    optimize = Optimize(obj_fcn, np.zeros(2), bnds, _options(run_global), fake)
    res = optimize.run()

    return res, fake, evals


class TestStageHandoff:
    def test_local_stage_starts_at_global_optimum(self):
        res, _, evals = _run()
        global_s = res["global"]["s"]
        first_local = next(x for stage, x in evals if stage == "local")
        np.testing.assert_allclose(
            first_local, global_s, atol=1e-12,
            err_msg="local stage must start from the global stage's optimum",
        )
        np.testing.assert_allclose(
            global_s, CENTER, atol=1e-3,
            err_msg="global stage should have found the quadratic center",
        )

    def test_error_floor_recalibrated_at_stage_boundary(self):
        res, fake, _ = _run()
        assert len(fake.recalibrated_at) == 1, (
            f"expected one boundary recalibration, got "
            f"{len(fake.recalibrated_at)}"
        )
        np.testing.assert_allclose(fake.recalibrated_at[0], res["global"]["s"])

    def test_local_only_run_does_not_recalibrate(self):
        res, fake, _ = _run(run_global=False)
        assert "global" not in res
        assert fake.recalibrated_at == []
        np.testing.assert_allclose(res["local"]["s"], CENTER, atol=1e-3)

    def test_stage_results_carry_scaler_and_coefficient_vectors(self):
        res, _, _ = _run()
        for stage in ("global", "local"):
            assert "s" in res[stage] and "x" in res[stage], (
                f"{stage} result must expose both scaler-space and "
                f"coefficient-space optima"
            )


class _AnisotropicCostFunction:
    """Quadratic with a 1e4 curvature spread between the two slots."""

    CENTER = np.array([0.4, -0.3])
    CURV = np.array([1.0, 1e4])

    def __init__(self):
        self.i = 0
        self.opt_type = "local"

    def value(self, s):
        d = np.asarray(s, dtype=float) - self.CENTER
        value = float(np.dot(self.CURV, d * d))

        return value

    def __call__(self, s, optimizing=False):
        result = (self.value(s), np.asarray(s), "shocks")

        return result


def _whitened_options(max_eval=120):
    stage = {
        "algorithm": "whitened_sbplx",
        "initial_step": 0.1,
        "stop_criteria_type": "Iteration Maximum",
        "stop_criteria_val": max_eval,
        "xtol_rel": 1e-10,
        "ftol_rel": 1e-12,
        "initial_pop_multiplier": 1.0,
    }
    options = {
        "global": dict(stage, run=False),
        "local": dict(stage, run=True),
    }

    return options


def _run_whitened(x0, bnds_lo, bnds_hi, max_eval=120, fake=None):
    fake = fake or _AnisotropicCostFunction()

    def obj_fcn(s, grad=None):
        return fake.value(s)

    bnds = {
        "lower": np.asarray(bnds_lo, dtype=float),
        "upper": np.asarray(bnds_hi, dtype=float),
    }
    optimize = Optimize(obj_fcn, np.asarray(x0, dtype=float), bnds,
                        _whitened_options(max_eval), fake)
    res = optimize.run()

    return res["local"], fake


class TestWhitenedSubplex:
    def test_converges_on_anisotropic_quadratic(self):
        res, fake = _run_whitened(np.zeros(2), [-1.0, -1.0], [1.0, 1.0])
        np.testing.assert_allclose(
            res["s"], fake.CENTER, atol=1e-3,
            err_msg="whitened Subplex should reach the quadratic center "
                    "despite the 1e4 curvature spread",
        )

    def test_probe_evaluations_count_against_budget(self):
        res, _ = _run_whitened(np.zeros(2), [-1.0, -1.0], [1.0, 1.0],
                               max_eval=40)
        assert res["nfev"] <= 40, (
            f"stage budget 40 exceeded: nfev {res['nfev']} — the probe "
            "must be charged to the stage, not added on top"
        )

    def test_optimum_respects_bounds_when_center_outside_box(self):
        res, _ = _run_whitened(np.zeros(2), [-0.2, -0.2], [0.2, 0.2])
        assert np.all(res["s"] >= -0.2 - 1e-12) and np.all(res["s"] <= 0.2 + 1e-12), (
            f"optimum {res['s']} left the box [-0.2, 0.2]^2"
        )
        np.testing.assert_allclose(
            res["s"], [0.2, -0.2], atol=1e-3,
            err_msg="constrained optimum should sit on the box face "
                    "nearest the center",
        )

    def test_infeasible_start_falls_back_to_plain_subplex(self):
        class _InfStart(_AnisotropicCostFunction):
            def value(self, s):
                s = np.asarray(s, dtype=float)
                if np.array_equal(s, np.zeros(2)):
                    return float("inf")

                return super().value(s)

        res, fake = _run_whitened(np.zeros(2), [-1.0, -1.0], [1.0, 1.0],
                                  fake=_InfStart())
        assert np.isfinite(res["fval"]), (
            "fallback plain Subplex should still return a finite optimum"
        )
        np.testing.assert_allclose(res["s"], fake.CENTER, atol=1e-2)


class _IsotropicCostFunction(_AnisotropicCostFunction):
    """Quadratic with equal curvature in both slots."""

    CURV = np.array([2.0, 2.0])


class TestWhitenedSubplexAutoSwitch:
    def test_isotropic_curvature_skips_whitening(self):
        fake = _IsotropicCostFunction()
        res, _ = _run_whitened(np.zeros(2), [-1.0, -1.0], [1.0, 1.0],
                               fake=fake)
        assert "whitening skipped" in res["message"], (
            f"spread ~1 should fall through to plain Subplex; message "
            f"was: {res['message']}"
        )
        np.testing.assert_allclose(
            res["s"], fake.CENTER, atol=1e-3,
            err_msg="fall-through Subplex should still converge",
        )

    def test_anisotropic_curvature_keeps_whitening(self):
        res, _ = _run_whitened(np.zeros(2), [-1.0, -1.0], [1.0, 1.0])
        assert "whitening skipped" not in res["message"], (
            "1e4 curvature spread must stay on the whitened path"
        )

    def test_fall_through_charges_probe_to_budget(self):
        fake = _IsotropicCostFunction()
        res, _ = _run_whitened(np.zeros(2), [-1.0, -1.0], [1.0, 1.0],
                               max_eval=40, fake=fake)
        assert res["nfev"] <= 40, (
            f"fall-through nfev {res['nfev']} exceeds the stage budget "
            "40 — probe evals must be charged, not added on top"
        )

    def test_infinite_probe_curvature_keeps_finite_scales(self):
        class _CliffCostFunction(_AnisotropicCostFunction):
            def value(self, s):
                s = np.asarray(s, dtype=float)
                if s[1] > 0.5:
                    return float("inf")

                return super().value(s)

        fake = _CliffCostFunction()
        res, _ = _run_whitened(np.array([0.0, 0.45]), [-1.0, -1.0],
                               [1.0, 1.0], fake=fake)
        assert np.all(np.isfinite(res["s"])), (
            f"optimum {res['s']} must stay finite when a probe hits the "
            "infeasible cliff"
        )
        assert np.isfinite(res["fval"])


class TestSmurfCollinearAttribution:
    def test_identical_profiles_get_minimum_norm_split(self):
        """Reactions with identical sensitivity profiles cannot be
        distinguished by one experiment; the ridged solve must split
        their shared multiplier equally instead of taking large
        opposite-signed moves."""
        from frhodo.optimize.smurf import _solve_multipliers

        rng = np.random.default_rng(0)
        n_pts = 40
        profile = rng.standard_normal(n_pts)
        a_matrix = np.column_stack([
            profile, profile, rng.standard_normal(n_pts),
        ])
        true_delta = np.array([0.6, 0.0, -0.3])
        resid = a_matrix @ true_delta
        weights = np.ones(n_pts)

        delta, _w = _solve_multipliers(
            a_matrix, resid, weights, loss_c=1.0, loss_alpha=2.0)

        assert delta[0] == pytest.approx(delta[1], abs=1e-8), (
            f"collinear pair split unevenly: {delta[0]} vs {delta[1]}"
        )
        assert delta[0] + delta[1] == pytest.approx(0.6, rel=0.05), (
            f"collinear pair sum {delta[0] + delta[1]} lost the "
            f"identifiable total 0.6"
        )
        assert delta[2] == pytest.approx(-0.3, rel=0.05), (
            f"independent reaction disturbed: {delta[2]}"
        )
