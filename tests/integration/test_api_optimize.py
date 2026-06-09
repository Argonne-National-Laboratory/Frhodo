"""End-to-end test for :func:`frhodo.api.optimize_residual`.

Exercises the typed ``OptimizationRequest`` surface:
:class:`OptimizableSpec` selects targets,
:class:`RateUncertainty` / :class:`CoefUncertainty` set bounds,
:class:`AlgorithmSettings` configures the optimizer. Runs a tiny
local-only optimization (2 iterations) against a synthetic shock so
the test stays under a few seconds.
"""
import cantera as ct
import numpy as np
import pytest

from frhodo.api import (
    AlgorithmSettings,
    AlgorithmStage,
    CostSettings,
    ExperimentShock,
    IterationUpdate,
    ObservableSettings,
    OptimizableRate,
    OptimizableSpec,
    OptimizationCallbacks,
    OptimizationRequest,
    OptimizationResult,
    PostShockState,
    RateUncertainty,
    StageComplete,
    StartInfo,
    apply_optimization_result,
    optimize_residual,
)
from frhodo.simulation.mechanism.mechanism_loader import MechanismLoader
from frhodo.simulation.shock.state import RuntimeReactorState


def _first_arrhenius_idx(mech):
    return next(
        i for i, r in enumerate(mech.gas.reactions())
        if type(r.rate) is ct.ArrheniusRate
    )


def _first_recastable_pdep_idx(mech):
    """First reaction that ``recast_to_troe`` actually refits.

    Skips Arrhenius (nothing to recast) and Troe (already the target
    form). The match is typically a Plog/Chebyshev reaction, whose
    recast rebuilds the Solution and exercises the bounds-freezing path.
    """
    skip = (ct.ArrheniusRate, ct.TroeRate)
    idx = next(
        i for i, r in enumerate(mech.gas.reactions())
        if type(r.rate) not in skip
    )

    return idx


@pytest.fixture
def loaded_cycloheptane(loaded_cycloheptane):
    """Function-scoped override that restores mutated mech state after the
    test runs. ``OptimizableSpec.build`` mutates rate_bnds/coeffs_bnds and
    the optimizer's ``update_mech_coef_opt`` mutates mech.coeffs in place;
    later tests must not see those mutations.
    """
    import copy

    mech = loaded_cycloheptane
    snapshot = {
        "coeffs": copy.deepcopy(mech.coeffs),
        "coeffs_bnds": copy.deepcopy(mech.coeffs_bnds),
        "rate_bnds": copy.deepcopy(mech.rate_bnds),
    }
    yield mech
    mech.coeffs = snapshot["coeffs"]
    mech.coeffs_bnds = snapshot["coeffs_bnds"]
    mech.rate_bnds = snapshot["rate_bnds"]
    mech.modify_reactions(mech.coeffs)


def _synthetic_shock():
    t = np.linspace(1e-7, 5e-5, 50)
    return ExperimentShock(
        t=t,
        observable=np.zeros_like(t),
        initial=PostShockState(
            T_reac=1500.0, P_reac=20000.0,
            u_incident=181.85, rho1=0.0230433,
            composition={"Kr": 0.96, "cC7H14": 0.04},
        ),
        t_end=5e-5,
    )


def _cost_settings():
    return CostSettings(
        obj_fcn_type="Residual",
        scale="Linear",
        bisymlog_scaling_factor=1.0,
        loss_alpha=2.0,
        loss_c=1.0,
        bayes_dist_type="Automatic",
        bayes_unc_sigma=2.0,
    )


def _local_only(max_iters=2):
    """Tiny algorithm settings: skip the global stage, run local for ``max_iters``."""
    return AlgorithmSettings(
        global_stage=AlgorithmStage(
            algorithm="RBFOpt", enabled=False, stop_value=1.0,
        ),
        local_stage=AlgorithmStage(
            algorithm="Nelder-Mead Simplex",
            enabled=True, max_eval=max_iters,
            stop_value=float(max_iters),
        ),
    )


def _reactor_state():
    return RuntimeReactorState(
        name="Incident Shock Reactor", t_end=5e-5, t_unit_conv=1e-6,
        sim_interp_factor=1, ode_solver="BDF", ode_rtol=1e-4, ode_atol=1e-7,
    )


def _build_request(mech, max_iters=2):
    return OptimizationRequest(
        shocks=[_synthetic_shock()],
        optimizable=OptimizableSpec(rates=[
            OptimizableRate(
                rxn_idx=_first_arrhenius_idx(mech),
                rate=RateUncertainty(factor=2.0),
            ),
        ]),
        reactor_state=_reactor_state(),
        cost=_cost_settings(),
        algorithm=_local_only(max_iters),
        observable=ObservableSettings(),
    )


@pytest.mark.slow
class TestOptimizeResidualTypedRequest:
    def test_returns_optimization_result(self, loaded_cycloheptane):
        request = _build_request(loaded_cycloheptane)
        result = optimize_residual(loaded_cycloheptane, request)
        assert isinstance(result, OptimizationResult)

    def test_success_path_has_finite_fval(self, loaded_cycloheptane):
        request = _build_request(loaded_cycloheptane)
        result = optimize_residual(loaded_cycloheptane, request)
        assert result.success, f"optimize_residual failed: {result.message}"
        assert np.isfinite(result.fval), f"fval was {result.fval}"
        assert result.nfev >= 1

    def test_x_has_one_entry_per_optimized_coefficient(self, loaded_cycloheptane):
        request = _build_request(loaded_cycloheptane)
        result = optimize_residual(loaded_cycloheptane, request)
        assert result.optimizable_used is not None
        assert result.x.size == len(result.optimizable_used.coefficients)

    def test_optimizable_used_attached_to_result(self, loaded_cycloheptane):
        request = _build_request(loaded_cycloheptane)
        result = optimize_residual(loaded_cycloheptane, request)
        assert result.optimizable_used is not None
        assert not result.optimizable_used.is_empty()

    def test_on_iteration_invoked_with_iteration_update(self, loaded_cycloheptane):
        request = _build_request(loaded_cycloheptane)
        updates: list[IterationUpdate] = []
        cb = OptimizationCallbacks(on_iteration=updates.append)
        optimize_residual(loaded_cycloheptane, request, callbacks=cb)
        assert len(updates) >= 1, "on_iteration was never invoked"
        first = updates[0]
        assert isinstance(first, IterationUpdate)
        assert first.iter >= 0
        assert first.stage in ("global", "local")
        assert np.isfinite(first.fval)
        assert first.is_best is True

    def test_on_start_fires_with_start_info(self, loaded_cycloheptane):
        request = _build_request(loaded_cycloheptane)
        starts: list[StartInfo] = []
        cb = OptimizationCallbacks(on_start=starts.append)
        optimize_residual(loaded_cycloheptane, request, callbacks=cb)
        assert len(starts) == 1
        info = starts[0]
        assert info.n_shocks == len(request.shocks)
        assert not info.optimizable_used.is_empty()
        assert info.recast_rxns == ()

    def test_log_reports_recast_fit_rms_for_pdep_rxn(self, cycloheptane_paths):
        # A structural recast (Plog -> Troe) rebuilds the Solution, which
        # the shared module mech's snapshot-restore can't undo; load a
        # throwaway mech instead.
        mech = MechanismLoader().load(cycloheptane_paths)
        pdep_idx = _first_recastable_pdep_idx(mech)
        request = OptimizationRequest(
            shocks=[_synthetic_shock()],
            optimizable=OptimizableSpec(rates=[
                OptimizableRate(
                    rxn_idx=pdep_idx, rate=RateUncertainty(factor=2.0),
                ),
            ]),
            reactor_state=_reactor_state(),
            cost=_cost_settings(),
            algorithm=_local_only(2),
            observable=ObservableSettings(),
        )
        messages: list[str] = []
        cb = OptimizationCallbacks(log=messages.append)
        optimize_residual(mech, request, callbacks=cb)
        recast_lines = [m for m in messages if "recast to Troe: fit log-RMS" in m]
        assert recast_lines, f"no recast-RMS line logged; got {messages}"
        assert f"R{pdep_idx + 1} " in recast_lines[0]

    def test_on_stage_complete_fires_per_stage(self, loaded_cycloheptane):
        request = _build_request(loaded_cycloheptane)
        completed: list[StageComplete] = []
        cb = OptimizationCallbacks(on_stage_complete=completed.append)
        optimize_residual(loaded_cycloheptane, request, callbacks=cb)
        # Local-only request: exactly one stage completion
        assert len(completed) == 1
        sc = completed[0]
        assert sc.stage == "local"
        assert np.isfinite(sc.fval)
        assert sc.shock_evals == sc.nfev * len(request.shocks)

    def test_is_best_flag_tracks_minimum(self, loaded_cycloheptane):
        request = _build_request(loaded_cycloheptane, max_iters=5)
        updates: list[IterationUpdate] = []
        cb = OptimizationCallbacks(on_iteration=updates.append)
        optimize_residual(loaded_cycloheptane, request, callbacks=cb)
        # The first update is always is_best (best so far is itself);
        # subsequent is_best=True ⇒ fval strictly improved.
        best_seen = float("inf")
        for u in updates:
            if u.is_best:
                assert u.fval < best_seen, (
                    f"is_best=True but fval {u.fval} not better than {best_seen}"
                )
                best_seen = u.fval

    def test_empty_optimizable_spec_returns_failed_result(self, loaded_cycloheptane):
        request = OptimizationRequest(
            shocks=[_synthetic_shock()],
            optimizable=OptimizableSpec(rates=[]),
            reactor_state=_reactor_state(),
            cost=_cost_settings(),
            algorithm=_local_only(),
        )
        result = optimize_residual(loaded_cycloheptane, request)
        assert not result.success
        assert "empty" in result.message.lower()

    def test_no_qt_dependency(self, loaded_cycloheptane):
        request = _build_request(loaded_cycloheptane, max_iters=1)
        optimize_residual(loaded_cycloheptane, request)
        import sys

        assert "qtpy" not in sys.modules.get("frhodo.api").__dict__


class TestApplyOptimizationResult:
    def test_writes_yaml_file(self, loaded_cycloheptane, tmp_path):
        request = _build_request(loaded_cycloheptane)
        result = optimize_residual(loaded_cycloheptane, request)
        assert result.success

        out = tmp_path / "optimized.yaml"
        apply_optimization_result(loaded_cycloheptane, result, save_path=out)
        assert out.exists()
        assert out.read_text().startswith("generator") or "phases:" in out.read_text()

    def test_writes_chemkin_file(self, loaded_cycloheptane, tmp_path):
        request = _build_request(loaded_cycloheptane)
        result = optimize_residual(loaded_cycloheptane, request)
        assert result.success

        out = tmp_path / "optimized.inp"
        apply_optimization_result(loaded_cycloheptane, result, save_path=out)
        assert out.exists()

    def test_unsupported_suffix_raises(self, loaded_cycloheptane, tmp_path):
        request = _build_request(loaded_cycloheptane)
        result = optimize_residual(loaded_cycloheptane, request)
        out = tmp_path / "out.zzz"
        with pytest.raises(ValueError, match="unsupported save_path suffix"):
            apply_optimization_result(loaded_cycloheptane, result, save_path=out)

    def test_in_place_modification_when_no_save_path(
        self, loaded_cycloheptane,
    ):
        request = _build_request(loaded_cycloheptane)
        result = optimize_residual(loaded_cycloheptane, request)
        assert result.success
        assert result.x.size > 0

        # Capture the optimized coefficient values, apply, then verify
        # mech.coeffs reflects the post-optimization x.
        coef_opt = list(result.optimizable_used.coefficients)
        apply_optimization_result(loaded_cycloheptane, result)
        for i, c in enumerate(coef_opt):
            stored = loaded_cycloheptane.coeffs[c.rxn_idx][c.coeffs_key][c.coef_name]
            assert stored == result.x[i], (
                f"rxn {c.rxn_idx} coef {c.coef_name}: stored={stored} x[i]={result.x[i]}"
            )

    def test_rejects_mismatched_x_size(self, loaded_cycloheptane):
        request = _build_request(loaded_cycloheptane)
        result = optimize_residual(loaded_cycloheptane, request)
        # Manually craft a result with the wrong x length
        from dataclasses import replace
        bad = replace(result, x=np.array([1.0]))
        with pytest.raises(ValueError, match="entries"):
            apply_optimization_result(loaded_cycloheptane, bad)

    def test_rejects_missing_optimizable_used(self, loaded_cycloheptane):
        request = _build_request(loaded_cycloheptane)
        result = optimize_residual(loaded_cycloheptane, request)
        from dataclasses import replace
        bad = replace(result, optimizable_used=None)
        with pytest.raises(ValueError, match="optimizable_used"):
            apply_optimization_result(loaded_cycloheptane, bad)


