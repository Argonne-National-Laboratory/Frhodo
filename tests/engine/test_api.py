"""Edge-case sweep for ``frhodo.api``.

These tests pin the public API surface as a regression net for
internal rewrites. Every callable in ``frhodo.api`` gets at least one
happy-path snapshot test plus failure-mode coverage.
"""
import pathlib

import numpy as np
import pytest
from pydantic import ValidationError

import frhodo
from frhodo.api import (
    PostShockState,
    PreShockState,
    ShockTubeConfig,
    SolverSettings,
    ZeroDConfig,
    kJ_per_mol,
    kcal_per_mol,
    load_mechanism,
    parse_composition,
    run_shock_tube,
    run_shock_tubes,
    run_zero_d,
    solve_shock_jump,
)


T1_K = 294.15
P1_PA = 5.01 * 133.322368421
U1_MPS = 120e-3 / 116.557292e-6
MIX = {"Kr": 0.96, "cC7H14": 0.04}
T_END = 5.0e-5


@pytest.fixture(scope="module")
def initial_shock1():
    return PreShockState(T1=T1_K, P1=P1_PA, u1=U1_MPS, composition=dict(MIX))


@pytest.fixture(scope="module")
def shock_state(loaded_cycloheptane, initial_shock1):
    return solve_shock_jump(initial_shock1, loaded_cycloheptane)


@pytest.fixture(scope="module")
def shock1_cfg(shock_state):
    return ShockTubeConfig(
        initial=PostShockState(
            T_reac=shock_state.T2, P_reac=shock_state.P2,
            u_incident=shock_state.u2, rho1=shock_state.rho1,
            composition=dict(MIX),
        ),
        t_end=T_END,
    )


class TestSolveShockJump:
    def test_success(self, shock_state):
        assert shock_state.success

    def test_zone1_pinned(self, shock_state):
        assert shock_state.T1 == pytest.approx(T1_K)
        assert shock_state.P1 == pytest.approx(P1_PA)
        assert shock_state.u1 == pytest.approx(U1_MPS)

    def test_temperature_ordering(self, shock_state):
        assert shock_state.T1 < shock_state.T2 < shock_state.T5

    def test_pressure_ordering(self, shock_state):
        assert shock_state.P1 < shock_state.P2 < shock_state.P5

    def test_zone2_T_snapshot(self, shock_state):
        assert shock_state.T2 == pytest.approx(1616.29, rel=1e-3)

    def test_zone5_P_snapshot(self, shock_state):
        assert shock_state.P5 == pytest.approx(147139.9, rel=1e-3)

    def test_rho1_positive(self, shock_state):
        assert shock_state.rho1 > 0


class TestRunShockTubeHappyPath:
    @pytest.fixture(scope="class")
    def result(self, loaded_cycloheptane, shock1_cfg):
        return run_shock_tube(loaded_cycloheptane, shock1_cfg)

    def test_success(self, result):
        assert result.success

    def test_shape_consistency(self, result):
        n = result.t.size
        assert result.T.shape == (n,)
        assert result.P.shape == (n,)
        assert result.rho.shape == (n,)
        assert result.observable.shape == (n,)

    def test_t_starts_at_zero(self, result):
        assert result.t[0] == pytest.approx(0.0, abs=1e-12)

    def test_t_reaches_t_end(self, result):
        assert result.t[-1] == pytest.approx(T_END, rel=1e-3)

    def test_observable_finite(self, result):
        assert np.isfinite(result.observable).all()

    def test_species_count_matches_mechanism(self, result, loaded_cycloheptane):
        assert len(result.species) == loaded_cycloheptane.gas.n_species

    def test_no_qt_imports(self):
        import sys

        import frhodo.api  # noqa: F401
        for mod in list(sys.modules):
            if mod.startswith("frhodo.api"):
                m = sys.modules[mod]
                src = getattr(m, "__file__", "")
                if src and "frhodo" in src:
                    assert "frhodo/gui" not in (m.__name__ or ""), (
                        f"{m.__name__} should not be imported by frhodo.api"
                    )


def _post_shock(shock_state, composition=MIX):
    return PostShockState(
        T_reac=shock_state.T2, P_reac=shock_state.P2,
        u_incident=shock_state.u2, rho1=shock_state.rho1,
        composition=dict(composition),
    )


class TestRunShockTubeFailureModes:
    def test_unknown_species_returns_failed_result(
        self, loaded_cycloheptane, shock_state
    ):
        cfg = ShockTubeConfig(
            initial=PostShockState(
                T_reac=shock_state.T2, P_reac=shock_state.P2,
                u_incident=shock_state.u2, rho1=shock_state.rho1,
                composition={"NOT_A_REAL_SPECIES": 1.0},
            ),
            t_end=T_END,
        )
        result = run_shock_tube(loaded_cycloheptane, cfg)
        assert result.success is False
        assert result.message

    def test_nan_T_reac_rejected_at_config_build(self, shock_state):
        with pytest.raises(ValidationError):
            PostShockState(
                T_reac=float("nan"), P_reac=shock_state.P2,
                u_incident=shock_state.u2, rho1=shock_state.rho1,
                composition=dict(MIX),
            )

    def test_negative_t_end_rejected_at_config_build(self, shock_state):
        with pytest.raises(ValidationError):
            ShockTubeConfig(initial=_post_shock(shock_state), t_end=-1.0)

    def test_zero_t_end_rejected_at_config_build(self, shock_state):
        with pytest.raises(ValidationError):
            ShockTubeConfig(initial=_post_shock(shock_state), t_end=0.0)

    def test_unknown_solver_rejected_at_config_build(self, shock_state):
        with pytest.raises(ValidationError):
            ShockTubeConfig(
                initial=_post_shock(shock_state), t_end=T_END,
                solver=SolverSettings(solver="NotARealSolver"),
            )


class TestRunShockTubeAcrossSolvers:
    @pytest.mark.parametrize("solver", ["BDF", "LSODA", "Radau", "CVODES"])
    def test_solver_succeeds(self, loaded_cycloheptane, shock_state, solver):
        cfg = ShockTubeConfig(
            initial=_post_shock(shock_state), t_end=T_END,
            solver=SolverSettings(solver=solver),
        )
        result = run_shock_tube(loaded_cycloheptane, cfg)
        assert result.success, (
            f"solver {solver} failed: {result.failure_reason} {result.message}"
        )
        assert result.t[-1] == pytest.approx(T_END, rel=1e-2)


class TestRunShockTubeReentrancy:
    """Two threads sharing a ``ChemicalMechanism`` produce the same results as one.

    Cantera's ``Solution`` is correctness-safe across threads but
    offers no parallelism (GIL contention). This pins the correctness
    half of that contract.
    """

    def test_two_threads_share_one_mechanism_match_sequential(
        self, loaded_cycloheptane, shock1_cfg
    ):
        import threading

        ref = run_shock_tube(loaded_cycloheptane, shock1_cfg)
        assert ref.success

        results = [None, None]

        def worker(i):
            results[i] = run_shock_tube(loaded_cycloheptane, shock1_cfg)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for r in results:
            assert r is not None and r.success
            np.testing.assert_allclose(r.observable, ref.observable, rtol=1e-6)


class TestRunShockTubes:
    def test_sequential_workers_none_runs(self, loaded_cycloheptane, shock1_cfg):
        results = run_shock_tubes(loaded_cycloheptane, [shock1_cfg, shock1_cfg])
        assert len(results) == 2
        assert all(r.success for r in results)
        np.testing.assert_allclose(results[0].observable, results[1].observable)

    def test_sequential_workers_one_runs(self, loaded_cycloheptane, shock1_cfg):
        results = run_shock_tubes(loaded_cycloheptane, [shock1_cfg], workers=1)
        assert len(results) == 1 and results[0].success

    def test_identical_inputs_give_identical_outputs(
        self, loaded_cycloheptane, shock1_cfg
    ):
        results = run_shock_tubes(loaded_cycloheptane, [shock1_cfg] * 5)
        for r in results[1:]:
            np.testing.assert_allclose(r.observable, results[0].observable, rtol=1e-12)


class TestLoadMechanism:
    def test_yaml_single_arg(self, tmp_path, example_mech_dir):
        src = example_mech_dir / "cycloheptane.mech"
        thermo = example_mech_dir / "cycloheptane.therm"
        yaml_out = tmp_path / "cyc7.converted.yaml"
        load_mechanism(src, thermo=thermo, converted_yaml=yaml_out)

        mech = load_mechanism(yaml_out)
        assert mech.gas.n_species > 0
        assert mech.gas.n_reactions > 0

    def test_chemkin_auto_derives_output_path(self, tmp_path, example_mech_dir):
        src_copy = tmp_path / "cycloheptane.mech"
        thermo_copy = tmp_path / "cycloheptane.therm"
        src_copy.write_bytes((example_mech_dir / "cycloheptane.mech").read_bytes())
        thermo_copy.write_bytes((example_mech_dir / "cycloheptane.therm").read_bytes())

        mech = load_mechanism(src_copy, thermo=thermo_copy)

        expected = src_copy.with_suffix(".converted.yaml")
        assert expected.exists()
        assert mech.gas.n_species > 0

    def test_chemkin_with_thermo_and_explicit_output(
        self, tmp_path, example_mech_dir
    ):
        out = tmp_path / "explicit.yaml"
        mech = load_mechanism(
            example_mech_dir / "cycloheptane.mech",
            thermo=example_mech_dir / "cycloheptane.therm",
            converted_yaml=out,
        )
        assert out.exists()
        assert mech.gas.n_reactions > 0


class TestParseComposition:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("AR:0.5, KR:0.5", {"AR": 0.5, "KR": 0.5}),
            ("Kr:0.96,cC7H14:0.04", {"Kr": 0.96, "cC7H14": 0.04}),
            ("H2:1.0", {"H2": 1.0}),
            ("AR: 0.5,  KR: 0.5", {"AR": 0.5, "KR": 0.5}),
            ("AR:0.5\nKR:0.5", {"AR": 0.5, "KR": 0.5}),
        ],
    )
    def test_valid_forms(self, text, expected):
        assert parse_composition(text) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "AR:0.5, KR=0.5",
            "AR 0.5",
            "AR:notanumber",
            ":0.5",
            "",
            "   ",
        ],
    )
    def test_invalid_raises(self, text):
        with pytest.raises(ValueError):
            parse_composition(text)


class TestUnitHelpers:
    def test_kcal_per_mol_matches_cantera_factor(self):
        # 1 kcal = 4184 J; 1 mol = 1e-3 kmol; therefore 1 kcal/mol = 4.184e6 J/kmol.
        assert kcal_per_mol(1.0) == pytest.approx(4.184e6, rel=0, abs=0)
        assert kcal_per_mol(5.0) == pytest.approx(5 * 4.184e6)
        assert kcal_per_mol(0.0) == 0.0
        assert kcal_per_mol(-1.0) == pytest.approx(-4.184e6)

    def test_kJ_per_mol_matches_cantera_factor(self):
        # 1 kJ = 1000 J; 1 mol = 1e-3 kmol; therefore 1 kJ/mol = 1e6 J/kmol.
        assert kJ_per_mol(1.0) == pytest.approx(1.0e6)
        assert kJ_per_mol(4.184) == pytest.approx(kcal_per_mol(1.0))


class TestPackageRootExports:
    def test_public_names_resolve(self):
        for name in frhodo.__all__:
            assert getattr(frhodo, name, None) is not None, name

    def test_load_mechanism_at_package_root(self):
        assert frhodo.load_mechanism is load_mechanism

    def test_run_shock_tube_at_package_root(self):
        assert frhodo.run_shock_tube is run_shock_tube


class TestSimulationResultFlatField:
    """SimulationResult exposes optional properties as named fields, not a dict."""

    @pytest.fixture(scope="class")
    def result(self, loaded_cycloheptane, shock1_cfg):
        return run_shock_tube(loaded_cycloheptane, shock1_cfg)

    def test_core_arrays_populated_on_success(self, result):
        n = result.t.size
        for arr in (result.T, result.P, result.rho, result.observable):
            assert arr.shape == (n,)

    def test_optional_properties_are_arrays_or_none(self, result):
        for name in ("h_tot", "s_tot", "wdot", "HRR_tot", "drhodz_tot"):
            value = getattr(result, name)
            assert value is None or isinstance(value, np.ndarray), (
                f"{name} must be ndarray or None; got {type(value).__name__}"
            )

    def test_cantera_array_attached(self, result):
        # SolutionArray-like: needs .species_names and len()
        assert hasattr(result.cantera_array, "species_names")
        assert len(result.cantera_array) == result.t.size

    def test_no_extras_field(self, result):
        """Flat slots replaced the old extras Mapping."""
        assert not hasattr(result, "extras")


class TestCompositionDictOnly:
    """String-form composition is rejected; users must call parse_composition."""

    def test_post_shock_string_composition_raises(self):
        with pytest.raises(ValidationError):
            PostShockState(
                T_reac=1500.0, P_reac=20000.0,
                u_incident=1029.0, rho1=0.4,
                composition="Kr:0.96, cC7H14:0.04",
            )

    def test_zero_d_string_composition_raises(self):
        with pytest.raises(ValidationError):
            ZeroDConfig(
                mode="constant_volume",
                T_reac=1500.0, P_reac=20000.0,
                composition="Kr:1.0", t_end=1e-3,
            )

    def test_pre_shock_string_composition_raises(self):
        with pytest.raises(ValidationError):
            PreShockState(T1=300.0, P1=1e5, u1=1000.0, composition="Ar:1.0")

    def test_dict_composition_round_trips(self):
        comp = {"Kr": 0.96, "cC7H14": 0.04}
        cfg = ShockTubeConfig(
            initial=PostShockState(
                T_reac=1500.0, P_reac=20000.0,
                u_incident=1029.0, rho1=0.4, composition=comp,
            ),
            t_end=5e-5,
        )
        assert cfg.initial.composition == comp


class TestRunShockTubesNewSignature:
    """``run_shock_tubes`` accepts ``ChemicalMechanism | str | Path`` only."""

    def test_path_string_loads_inline(self, tmp_path, example_mech_dir, shock1_cfg):
        yaml = tmp_path / "cyc7.yaml"
        from frhodo import load_mechanism as lm

        lm(
            example_mech_dir / "cycloheptane.mech",
            thermo=example_mech_dir / "cycloheptane.therm",
            converted_yaml=yaml,
        )
        results = run_shock_tubes(str(yaml), [shock1_cfg])
        assert len(results) == 1 and results[0].success

    def test_path_object_loads_inline(self, tmp_path, example_mech_dir, shock1_cfg):
        yaml = tmp_path / "cyc7.yaml"
        from frhodo import load_mechanism as lm

        lm(
            example_mech_dir / "cycloheptane.mech",
            thermo=example_mech_dir / "cycloheptane.therm",
            converted_yaml=yaml,
        )
        results = run_shock_tubes(yaml, [shock1_cfg])
        assert len(results) == 1 and results[0].success


class TestShockTubeDiscriminator:
    """The ``initial`` field is a discriminated union; ``kind`` selects the model."""

    def test_pre_shock_validates(self):
        s = PreShockState(T1=294.0, P1=601.0, u1=1029.0, composition={"Ar": 1.0})
        assert s.kind == "pre_shock"

    def test_post_shock_validates(self):
        s = PostShockState(T_reac=1500.0, P_reac=2e5, u_incident=1029.0,
                           rho1=0.4, composition={"Ar": 1.0})
        assert s.kind == "post_shock"

    def test_missing_kind_resolves_to_either_via_unique_fields(self):
        """Pydantic discriminated union dispatches on ``kind`` literal."""
        cfg = ShockTubeConfig(
            initial={
                "kind": "post_shock",
                "T_reac": 1500.0, "P_reac": 2e5,
                "u_incident": 1029.0, "rho1": 0.4,
                "composition": {"Ar": 1.0},
            },
            t_end=5e-5,
        )
        assert isinstance(cfg.initial, PostShockState)

    def test_unknown_kind_rejected(self):
        with pytest.raises(ValidationError):
            ShockTubeConfig(
                initial={"kind": "between_shocks", "T1": 300.0, "P1": 1e5, "u1": 1000.0},
                t_end=5e-5,
            )


class TestShockTubeAutoJump:
    """``run_shock_tube`` with a ``PreShockState`` matches the explicit two-call path."""

    def test_pre_shock_auto_solves_jump(self, loaded_cycloheptane):
        cfg = ShockTubeConfig(
            initial=PreShockState(T1=T1_K, P1=P1_PA, u1=U1_MPS,
                                  composition=dict(MIX)),
            t_end=T_END,
        )
        result = run_shock_tube(loaded_cycloheptane, cfg)
        assert result.success
        assert result.shock is not None
        assert result.shock.T2 > result.shock.T1
        assert result.shock.initial is cfg.initial

    def test_post_shock_leaves_shock_field_none(
        self, loaded_cycloheptane, shock_state
    ):
        cfg = ShockTubeConfig(initial=_post_shock(shock_state), t_end=T_END)
        result = run_shock_tube(loaded_cycloheptane, cfg)
        assert result.success
        assert result.shock is None

    def test_auto_jump_trace_matches_explicit_two_call(
        self, loaded_cycloheptane, initial_shock1, shock1_cfg
    ):
        """Equivalence: PreShockState auto-jump = solve_shock_jump + PostShockState run.
        Both paths share the same ShockJumpSolver internals, so the resulting
        reactor traces must match to machine precision.
        """
        # Reference: explicit two-call path
        reference = run_shock_tube(loaded_cycloheptane, shock1_cfg)
        assert reference.success

        # Auto-jump path with the same PreShockState
        cfg_pre = ShockTubeConfig(initial=initial_shock1, t_end=T_END)
        auto = run_shock_tube(loaded_cycloheptane, cfg_pre)
        assert auto.success

        np.testing.assert_allclose(auto.t, reference.t, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(
            auto.observable, reference.observable, rtol=1e-12, atol=1e-15,
            err_msg="observable trace must match between auto-jump and explicit two-call",
        )
        np.testing.assert_allclose(auto.T, reference.T, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(auto.P, reference.P, rtol=1e-12, atol=1e-15)

    def test_auto_jump_failure_returns_failed_result(self, loaded_cycloheptane):
        """A pre-shock state that the jump solver cannot resolve must surface
        as ``success=False`` with the failed shock attached."""
        cfg = ShockTubeConfig(
            initial=PreShockState(
                T1=300.0, P1=1e5, u1=50.0,  # subsonic u1 → no jump
                composition={"Ar": 1.0},
            ),
            t_end=1e-3,
        )
        result = run_shock_tube(loaded_cycloheptane, cfg)
        assert result.success is False
        assert result.shock is not None
        assert result.shock.success is False
        assert result.failure_reason is not None


class TestSolverObservableSettings:
    """Composition: ShockTubeConfig embeds SolverSettings + ObservableSettings."""

    def test_default_solver(self):
        cfg = ShockTubeConfig(
            initial=PostShockState(
                T_reac=1500.0, P_reac=2e5, u_incident=1029.0, rho1=0.4,
                composition={"Ar": 1.0},
            ),
            t_end=5e-5,
        )
        assert cfg.solver.solver == "CVODES"
        assert cfg.solver.sim_interp_factor == 1
        assert cfg.observable.main == "Density Gradient"

    def test_custom_settings_round_trip(self):
        cfg = ShockTubeConfig(
            initial=PostShockState(
                T_reac=1500.0, P_reac=2e5, u_incident=1029.0, rho1=0.4,
                composition={"Ar": 1.0},
            ),
            t_end=5e-5,
            solver=SolverSettings(solver="LSODA", rtol=1e-8, atol=1e-12),
        )
        assert cfg.solver.solver == "LSODA"
        assert cfg.solver.rtol == 1e-8
