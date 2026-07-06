"""Post-fit audit: bound utilization, at-bounds detection, flat directions."""
import numpy as np
import pytest

from frhodo.optimize.audit import (
    AT_BOUND_FRACTION,
    FLAT_REL_TOL,
    audit_optimum,
    bound_utilization,
    slot_labels,
    slot_rxn_indices,
)



class FakeInputs:
    def __init__(self, lower, upper, rxn_coef_opt):
        self.rxn_rate_opt = {"bnds": {"lower": np.asarray(lower, dtype=float),
                                      "upper": np.asarray(upper, dtype=float)}}
        self.rxn_coef_opt = rxn_coef_opt


def two_rxn_coef_opt():
    coef_opt = [
        {"rxnIdx": 0, "T": [1000.0, 2000.0]},
        {"rxnIdx": 4, "T": [1500.0]},
    ]

    return coef_opt


class TestBoundUtilization:
    def test_side_aware_fractions(self):
        util = bound_utilization(
            s=[0.5, -0.25, 0.0],
            lower=[-1.0, -1.0, -1.0],
            upper=[1.0, 1.0, 1.0],
        )
        np.testing.assert_allclose(util, [0.5, 0.25, 0.0])

    def test_degenerate_side_reports_zero(self):
        util = bound_utilization(s=[0.5], lower=[-1.0], upper=[0.0])
        np.testing.assert_allclose(util, [0.0])


class TestSlotMaps:
    def test_labels_and_rxn_indices_align(self):
        coef_opt = two_rxn_coef_opt()
        assert slot_labels(coef_opt) == [
            "R1 @ 1000 K", "R1 @ 2000 K", "R5 @ 1500 K",
        ]
        assert slot_rxn_indices(coef_opt) == [0, 0, 4]


class TestAuditOptimum:
    def _run(self, s, fit_fun):
        inputs = FakeInputs([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0],
                            two_rxn_coef_opt())
        res = {"local": {"s": np.asarray(s, dtype=float)}}
        lines = []
        audit_optimum(res, inputs, fit_fun, lines.append)
        audit = res["local"]["audit"]

        return audit, lines

    def test_at_bounds_and_flat_detection(self):
        # Slot 0 pressed to its upper bound; slot 2 does not enter the
        # objective at all.
        def fit_fun(s, quiet=False):
            return 1.0 + s[0] ** 2 + s[1] ** 2

        s_opt = [1.0 - 0.5 * AT_BOUND_FRACTION * 2.0, 0.2, 0.3]
        audit, lines = self._run(s_opt, fit_fun)
        assert audit["at_bounds"] == [("R1 @ 1000 K", "upper")]
        assert audit["flat"] == ["R5 @ 1500 K"]
        assert any("At bounds: 1" in line for line in lines)
        assert any("Unconstrained at the optimum: 1" in line for line in lines)

    def test_clean_optimum_logs_single_quiet_line(self):
        def fit_fun(s, quiet=False):
            return 1.0 + float(np.sum(np.square(s)))

        audit, lines = self._run([0.2, -0.3, 0.4], fit_fun)
        assert audit["at_bounds"] == []
        assert audit["flat"] == []
        assert lines == [
            "Post-fit audit: no coefficients at bounds, none unconstrained",
        ]

    def test_rxn_utilization_is_per_reaction_max(self):
        def fit_fun(s, quiet=False):
            return 1.0 + float(np.sum(np.square(s)))

        audit, _ = self._run([0.5, -0.8, 0.4], fit_fun)
        assert audit["rxn_utilization"] == {0: pytest.approx(0.8),
                                            4: pytest.approx(0.4)}

    def test_final_evaluation_restores_the_optimum(self):
        # Probes mutate shared mech/shock state; the last evaluation
        # must land back on the optimum.
        calls = []

        def fit_fun(s, quiet=False):
            calls.append(np.array(s, dtype=float))

            return 1.0 + float(np.sum(np.square(s)))

        s_opt = [0.2, -0.3, 0.4]
        self._run(s_opt, fit_fun)
        np.testing.assert_allclose(calls[-1], s_opt)

    def test_missing_stage_is_a_noop(self):
        res = {}
        audit_optimum(res, FakeInputs([-1.0], [1.0], []), None, lambda m: None)
        assert "audit" not in res

    def test_flat_probe_steps_into_the_interior(self):
        # Optimum exactly at the upper bound: the probe must step down
        # into the box, never past the bound.
        probes = []

        def fit_fun(s, quiet=False):
            probes.append(np.array(s, dtype=float))

            return 1.0 + s[0] ** 2 + s[1] ** 2 + s[2] ** 2

        self._run([1.0, 0.0, 0.0], fit_fun)
        assert all(p[0] <= 1.0 for p in probes), (
            f"probe exceeded the upper bound: {probes}"
        )

    def test_flat_tolerance_boundary(self):
        # dObj/obj right below the tolerance counts as flat; the span
        # step is 0.01 * 2.0, so a linear coefficient this small stays
        # under FLAT_REL_TOL.
        slope = 0.9 * FLAT_REL_TOL / 0.02

        def fit_fun(s, quiet=False):
            return 1.0 + slope * s[2] + s[0] ** 2 + s[1] ** 2

        audit, _ = self._run([0.2, 0.2, 0.0], fit_fun)
        assert audit["flat"] == ["R5 @ 1500 K"]
