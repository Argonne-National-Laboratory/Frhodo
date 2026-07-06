"""``Optimize.run`` stage plumbing: the global → local handoff."""
import types

import nlopt
import numpy as np

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
