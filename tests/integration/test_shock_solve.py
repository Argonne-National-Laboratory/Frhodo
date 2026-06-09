"""Snapshot tests for the shock-tube post-shock state solver.

``ShockJumpSolver.solve`` computes T2, P2, T5, P5 from incident shock
conditions using the Frosh / perfect-gas iteration. The reference
values below were captured with the bundled Cycloheptane mech.

Two cases are pinned:

* ``shock1`` — Shock1.exp baseline (P1=5.01 torr, T1=21°C, 96% Kr / 4%
  cC7H14, u1 derived from PT spacing / tOpt = 120 mm / 116.557292 us).
* ``hot_lowM`` — synthetic mid-Mach Ar/cC7H14 case for diversity
  (T1=350 K, P1=2666.4 Pa, u1=800 m/s).

Each case is solved under three GUI-supported known-variable
combinations: zone-1 (T1,P1,u1), zone-2 (T1,T2,P2), zone-5 (T1,T5,P5).
All three combos must converge to the same physical state — the snapshot
is identical across them.

Tolerances are loose (rtol=1e-3) so cross-Cantera-version drift in the
NASA polynomials and Mach iteration does not cause spurious failures,
but tight enough that any logic regression will fail loudly.
"""
import numpy as np
import pytest

from frhodo.common.errors import FailureReason, ShockJumpError
from frhodo.simulation.shock.shock_solver import ShockJumpResult, ShockJumpSolver


SHOCK1 = {
    "T1": 294.15,
    "P1": 5.01 * 133.322368421,
    "u1": 120e-3 / 116.557292e-6,
    "mix": {"Kr": 0.96, "cC7H14": 0.04},
    "expected": {
        "T1": 294.1500, "P1": 667.9451, "u1": 1029.5366,
        "T2": 1616.2904, "P2": 20778.2875, "u2": 181.8542,
        "T5": 2954.7343, "P5": 147139.9444, "u5": 46.9463,
        "Mach1": 4.9316, "rho1": 0.023043,
    },
}

HOT_LOWM = {
    "T1": 350.0,
    "P1": 2666.4,
    "u1": 800.0,
    "mix": {"Ar": 0.95, "cC7H14": 0.05},
    "expected": {
        "T1": 350.0000, "P1": 2666.4000, "u1": 800.0000,
        "T2": 755.3694, "P2": 20870.8985, "u2": 220.5797,
        "T5": 1153.0358, "P5": 93432.5426, "u5": 75.2128,
        "Mach1": 2.5560, "rho1": 0.039273,
    },
}

CASES = {"shock1": SHOCK1, "hot_lowM": HOT_LOWM}


def _combo_inputs(case, combo):
    """Build a known-variable dict for the given combo using `case` reference values."""
    e = case["expected"]
    if combo == "zone1_TPu":
        return {"T1": e["T1"], "P1": e["P1"], "u1": e["u1"], "mix": dict(case["mix"])}
    if combo == "zone2_TTP":
        return {"T1": e["T1"], "T2": e["T2"], "P2": e["P2"], "mix": dict(case["mix"])}
    if combo == "zone5_TTP":
        return {"T1": e["T1"], "T5": e["T5"], "P5": e["P5"], "mix": dict(case["mix"])}
    raise ValueError(combo)


@pytest.fixture(params=["shock1", "hot_lowM"])
def case_name(request):
    return request.param


@pytest.fixture(params=["zone1_TPu", "zone2_TTP", "zone5_TTP"])
def combo(request):
    return request.param


@pytest.fixture
def shock_props(loaded_cycloheptane, case_name, combo):
    case = CASES[case_name]
    shock_vars = _combo_inputs(case, combo)
    props = ShockJumpSolver(loaded_cycloheptane.gas, shock_vars)
    assert props.success, f"shock solver returned success=False for {case_name}/{combo}"
    return props, case


class TestShockSolveCombos:
    """Each (case, combo) must reproduce the pinned physical state.

    Same expected values across all three combos: any combination of
    knowns must converge to the same shock state.
    """

    @pytest.mark.parametrize("attr", [
        "T1", "P1", "u1",
        "T2", "P2", "u2",
        "T5", "P5", "u5",
    ])
    def test_field_matches_snapshot(self, shock_props, attr):
        props, case = shock_props
        actual = getattr(props.res, attr)
        expected = case["expected"][attr]
        np.testing.assert_allclose(
            actual, expected, rtol=1e-3,
            err_msg=f"{attr}: snapshot drift "
                    f"(expected {expected}, got {actual})",
        )

    def test_mach_matches_snapshot(self, shock_props):
        props, case = shock_props
        np.testing.assert_allclose(
            props.res.Mach1, case["expected"]["Mach1"], rtol=1e-3,
        )

    def test_post_shock_temperature_increases_through_zones(self, shock_props):
        """Physical sanity: T1 < T2 < T5 for an incident-then-reflected shock."""
        props, _ = shock_props
        r = props.res
        assert r.T1 < r.T2 < r.T5, (
            f"non-monotone temperatures: T1={r.T1}, T2={r.T2}, T5={r.T5}"
        )

    def test_post_shock_pressure_increases_through_zones(self, shock_props):
        props, _ = shock_props
        r = props.res
        assert r.P1 < r.P2 < r.P5, (
            f"non-monotone pressures: P1={r.P1}, P2={r.P2}, P5={r.P5}"
        )

    def test_zone_5_velocity_is_low(self, shock_props):
        """Reflected-shock zone is approximately stagnant."""
        props, _ = shock_props
        u5 = props.res.u5
        assert abs(u5) < 100.0, f"zone 5 should be near-stagnant; got u5={u5} m/s"


class TestShockSolveInputContract:
    """Inputs to ``ShockJumpSolver`` must not be mutated and must round-trip cleanly."""

    def test_input_dict_not_mutated(self, loaded_cycloheptane):
        case = SHOCK1
        shock_vars = _combo_inputs(case, "zone1_TPu")
        original = {k: (dict(v) if isinstance(v, dict) else v) for k, v in shock_vars.items()}
        ShockJumpSolver(loaded_cycloheptane.gas, shock_vars)
        assert shock_vars == original, (
            "ShockJumpSolver() mutated the caller's shock_vars dict; "
            "callers cannot reuse it for a second solve."
        )


class TestShockSolveFailures:
    """Failure paths surface as ``success=False`` with a typed error attached.

    Production callers (GUI, api.py) check ``.success``; ML callers can
    inspect ``.error.reason`` for failure-mode-aware retry logic.
    """

    def test_missing_mix_raises_input_invalid(self, loaded_cycloheptane):
        with pytest.raises(ShockJumpError) as exc_info:
            ShockJumpSolver(
                loaded_cycloheptane.gas,
                {"T1": 300.0, "P1": 1e5, "u1": 1000.0},  # no 'mix'
            )
        assert exc_info.value.reason is FailureReason.INPUT_INVALID

    def test_negative_temperature_marks_failure(self, loaded_cycloheptane):
        props = ShockJumpSolver(
            loaded_cycloheptane.gas,
            {"T1": -300.0, "P1": 1e5, "u1": 1000.0, "mix": {"Ar": 1.0}},
        )
        assert props.success is False
        assert isinstance(props.error, ShockJumpError)
        assert props.error.reason is FailureReason.TEMPERATURE_INVALID

    def test_negative_pressure_marks_failure(self, loaded_cycloheptane):
        props = ShockJumpSolver(
            loaded_cycloheptane.gas,
            {"T1": 300.0, "P1": -1e5, "u1": 1000.0, "mix": {"Ar": 1.0}},
        )
        assert props.success is False
        assert isinstance(props.error, ShockJumpError)
        assert props.error.reason is FailureReason.PRESSURE_INVALID

    def test_subsonic_input_marks_failure(self, loaded_cycloheptane):
        """u1 below the local speed of sound has no physical incident shock.

        The Frosh / perfect-gas iteration cannot find a self-consistent
        post-shock state and must surface the failure rather than
        returning silent garbage.
        """
        props = ShockJumpSolver(
            loaded_cycloheptane.gas,
            {"T1": 300.0, "P1": 1e5, "u1": 50.0, "mix": {"Ar": 1.0}},
        )
        assert props.success is False
        assert props.error is not None
        assert props.error.reason in (
            FailureReason.PERFECT_GAS_NOT_CONVERGED,
            FailureReason.FROSH_NOT_CONVERGED,
            FailureReason.TEMPERATURE_INVALID,
        )

    def test_failure_emits_warning_via_module_logger(self, loaded_cycloheptane, caplog):
        with caplog.at_level("WARNING", logger="frhodo.simulation.shock.shock_solver"):
            ShockJumpSolver(
                loaded_cycloheptane.gas,
                {"T1": -300.0, "P1": 1e5, "u1": 1000.0, "mix": {"Ar": 1.0}},
            )
        warnings = [r for r in caplog.records if r.levelno >= 30]
        assert warnings, "shock-jump failure must log a WARNING record"
        assert "Shock jump solver failed" in warnings[0].getMessage()


class TestShockJumpResultModel:
    """``ShockJumpSolver.res`` is a typed pydantic ``ShockJumpResult``.

    Attribute access replaces the old dict-style ``res[var]``. Lazy
    properties (P4) compute on access, never during ``solve``.
    """

    def test_res_is_shock_jump_result(self, loaded_cycloheptane):
        case = SHOCK1
        props = ShockJumpSolver(loaded_cycloheptane.gas, _combo_inputs(case, "zone1_TPu"))
        assert isinstance(props.res, ShockJumpResult)

    def test_attribute_access_matches_zone_state(self, loaded_cycloheptane):
        case = SHOCK1
        props = ShockJumpSolver(loaded_cycloheptane.gas, _combo_inputs(case, "zone1_TPu"))
        r = props.res
        e = case["expected"]
        np.testing.assert_allclose(r.T1, e["T1"], rtol=1e-3)
        np.testing.assert_allclose(r.P2, e["P2"], rtol=1e-3)
        np.testing.assert_allclose(r.T5, e["T5"], rtol=1e-3)
        np.testing.assert_allclose(r.Mach1, e["Mach1"], rtol=1e-3)

    def test_p4_is_none_without_mix_driver(self, loaded_cycloheptane):
        """No driver-section composition supplied → P4 is ``None``,
        not a NaN sentinel and never computed."""
        case = SHOCK1
        props = ShockJumpSolver(loaded_cycloheptane.gas, _combo_inputs(case, "zone1_TPu"))
        assert props.res.X_driver is None
        assert props.res.P4 is None

    def test_p4_lazy_not_in_dict_field_set(self, loaded_cycloheptane):
        """``P4`` is a ``cached_property``; until accessed it should not
        appear in the model's set fields. Confirms the lazy contract."""
        case = SHOCK1
        props = ShockJumpSolver(loaded_cycloheptane.gas, _combo_inputs(case, "zone1_TPu"))
        # Direct fields populated; P4 is computed-only (cached_property),
        # so reading model_fields_set must not contain "P4" until accessed.
        assert "P4" not in props.res.model_fields_set
        assert "T1" in props.res.model_fields_set

    def test_p4_computed_when_mix_driver_supplied(self, loaded_cycloheptane):
        """With ``mix_driver`` provided, ``P4`` resolves to a positive
        finite pressure consistent with the shock-tube relation
        (P4 > P1 since the driver section is at higher pressure)."""
        case = SHOCK1
        shock_vars = _combo_inputs(case, "zone1_TPu")
        shock_vars["mix_driver"] = {"H2": 1.0}
        props = ShockJumpSolver(loaded_cycloheptane.gas, shock_vars)
        p4 = props.res.P4
        assert p4 is not None
        assert np.isfinite(p4)
        assert p4 > props.res.P1, (
            f"P4 should exceed P1 (driver is high-pressure side); "
            f"got P4={p4}, P1={props.res.P1}"
        )

    def test_p4_caches_across_reads(self, loaded_cycloheptane):
        """Second read returns the same value without recomputation —
        ``cached_property`` semantics."""
        case = SHOCK1
        shock_vars = _combo_inputs(case, "zone1_TPu")
        shock_vars["mix_driver"] = {"H2": 1.0}
        props = ShockJumpSolver(loaded_cycloheptane.gas, shock_vars)
        first = props.res.P4
        second = props.res.P4
        assert first == second

    def test_lazy_mach_chain(self, loaded_cycloheptane):
        """Mach_i / a_i / gamma_i are all ``cached_property``; reading
        Mach_i should pull a_i and gamma_i through the chain and cache
        each. After reading, all three appear in ``__dict__``."""
        case = SHOCK1
        props = ShockJumpSolver(loaded_cycloheptane.gas, _combo_inputs(case, "zone1_TPu"))
        r = props.res
        # None of the lazy properties cached yet.
        for name in ("Mach1", "Mach2", "Mach5", "a1", "a2", "a5", "gamma1", "gamma2", "gamma5"):
            assert name not in r.__dict__, f"{name} cached before access"

        m2 = r.Mach2
        assert np.isfinite(m2) and m2 > 0
        # Reading Mach2 caches Mach2, a2, gamma2 (via the chain).
        for name in ("Mach2", "a2", "gamma2"):
            assert name in r.__dict__, f"{name} not cached after Mach2 read"
        # Untouched zones stay uncached.
        assert "Mach1" not in r.__dict__
        assert "gamma5" not in r.__dict__

    def test_post_shock_mach_is_subsonic(self, loaded_cycloheptane):
        """Behind an incident shock the flow is subsonic relative to the
        shock front: Mach2 < 1 < Mach1 always."""
        case = SHOCK1
        props = ShockJumpSolver(loaded_cycloheptane.gas, _combo_inputs(case, "zone1_TPu"))
        r = props.res
        assert r.Mach1 > 1.0, f"incident-shock Mach must be supersonic, got {r.Mach1}"
        assert r.Mach2 < 1.0, f"post-incident Mach must be subsonic, got {r.Mach2}"


class TestShockSolveDictOnlyContract:
    """``mix`` and ``mix_driver`` must be dicts. String compositions are
    rejected at the solver boundary; the public api.py normalizes them
    to dicts before calling."""

    def test_string_mix_rejected(self, loaded_cycloheptane):
        with pytest.raises(ShockJumpError) as exc_info:
            ShockJumpSolver(
                loaded_cycloheptane.gas,
                {"T1": 300.0, "P1": 1e5, "u1": 1000.0, "mix": "Ar:1.0"},
            )
        assert exc_info.value.reason is FailureReason.INPUT_INVALID

    def test_string_mix_driver_rejected(self, loaded_cycloheptane):
        with pytest.raises(ShockJumpError) as exc_info:
            ShockJumpSolver(
                loaded_cycloheptane.gas,
                {"T1": 300.0, "P1": 1e5, "u1": 1000.0,
                 "mix": {"Ar": 1.0}, "mix_driver": "H2:1.0"},
            )
        assert exc_info.value.reason is FailureReason.INPUT_INVALID

    def test_unknown_species_raises_input_invalid(self, loaded_cycloheptane):
        """Bad species name surfaces as INPUT_INVALID immediately at __init__,
        per the failure contract — does NOT set success=False."""
        with pytest.raises(ShockJumpError) as exc_info:
            ShockJumpSolver(
                loaded_cycloheptane.gas,
                {"T1": 300.0, "P1": 1e5, "u1": 1000.0, "mix": {"NonExistentSpecies": 1.0}},
            )
        assert exc_info.value.reason is FailureReason.INPUT_INVALID
