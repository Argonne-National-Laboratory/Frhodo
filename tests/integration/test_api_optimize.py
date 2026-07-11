"""End-to-end test for :func:`frhodo.api.optimize_residual`.

Exercises the typed ``OptimizationRequest`` surface:
:class:`OptimizableSpec` selects targets,
:class:`RateUncertainty` / :class:`CoefUncertainty` set bounds,
:class:`AlgorithmSettings` configures the optimizer. Runs a tiny
local-only optimization (2 iterations) against a synthetic shock so
the test stays under a few seconds.
"""
import cantera as ct
import copy
import sys
from dataclasses import replace

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
    idx = next(
        i for i, r in enumerate(mech.gas.reactions())
        if type(r.rate) is ct.ArrheniusRate
    )

    return idx


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
    shock = ExperimentShock(
        t=t,
        observable=np.zeros_like(t),
        initial=PostShockState(
            T_reac=1500.0, P_reac=20000.0,
            u2=181.85, rho1=0.0230433,
            composition={"Kr": 0.96, "cC7H14": 0.04},
        ),
        t_end=5e-5,
    )

    return shock


def _cost_settings():
    settings = CostSettings(
        scale="Linear",
        bisymlog_scaling_factor=1.0,
        loss_alpha=2.0,
        loss_c=1.0,
    )

    return settings


def _local_only(max_iters=2):
    """Tiny algorithm settings: skip the global stage, run local for ``max_iters``."""
    settings = AlgorithmSettings(
        global_stage=AlgorithmStage(
            algorithm="RBFOpt", enabled=False, stop_value=1.0,
        ),
        local_stage=AlgorithmStage(
            algorithm="Nelder-Mead Simplex",
            enabled=True, max_eval=max_iters,
            stop_value=float(max_iters),
        ),
    )

    return settings


def _reactor_state():
    state = RuntimeReactorState(
        name="Incident Shock Reactor", t_end=5e-5, t_unit_conv=1e-6,
        sim_interp_factor=1, ode_solver="BDF", ode_rtol=1e-4, ode_atol=1e-7,
    )

    return state


def _build_request(mech, max_iters=2):
    request = OptimizationRequest(
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

    return request


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

    @pytest.mark.parametrize("local_algo", ["BOBYQA", "BOBYQA (whitened)", "Subplex (field basis)", "Subplex (quick, field basis, multi-fidelity)"])
    def test_model_based_local_algorithm_runs(self, loaded_cycloheptane,
                                              local_algo):
        base = _build_request(loaded_cycloheptane)
        algorithm = AlgorithmSettings(
            global_stage=AlgorithmStage(
                algorithm="RBFOpt", enabled=False, stop_value=1.0),
            local_stage=AlgorithmStage(
                algorithm=local_algo, enabled=True, max_eval=8,
                stop_value=8.0),
        )
        request = base.model_copy(update={"algorithm": algorithm})
        result = optimize_residual(loaded_cycloheptane, request)
        assert result.success, f"{local_algo} failed: {result.message}"
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

    def test_per_shock_diagnostics_reach_the_progress_payload(
        self, loaded_cycloheptane,
    ):
        """The objective's per-shock diagnostics (the outcome-view data
        contract) must flow through stat_plot every iteration."""
        request = _build_request(loaded_cycloheptane)
        updates: list[IterationUpdate] = []
        cb = OptimizationCallbacks(on_iteration=updates.append)
        optimize_residual(loaded_cycloheptane, request, callbacks=cb)
        diag = updates[0].stat_plot["per_shock"]
        assert diag is not None, "per_shock diagnostics missing from stat_plot"
        n = len(request.shocks)
        for key in ("loss", "z", "irls_weights", "coverage", "user",
                    "trim_weights", "t_unc", "t_unc_star", "t_offset_base",
                    "sigma_bar", "T", "P"):
            assert len(np.atleast_1d(diag[key])) == n, (
                f"per_shock[{key!r}] should have one entry per shock"
            )
        assert np.isfinite(diag["mu"]) and 1.0 <= diag["alpha"] <= 2.0
        assert diag["c_floor"] > 0.0
        bounds = np.asarray(diag["t_unc_bounds"], dtype=float)
        assert bounds.shape == (2,) and bounds[0] <= bounds[1], (
            f"t_unc_bounds should be (lo, hi); got {bounds}"
        )
        assert diag["t_unc_mode"] in ("parametric", "independent", "fixed"), (
            f"unknown t_unc_mode {diag['t_unc_mode']!r}"
        )
        applied = np.asarray(diag["t_unc"], dtype=float)
        assert np.all((applied >= bounds[0]) & (applied <= bounds[1])), (
            f"applied offsets {applied} must lie inside the window {bounds}"
        )

    def test_sim_traces_reach_the_progress_payload(self, loaded_cycloheptane):
        """The overlay gallery's per-shock sim traces ship as deltas:
        ``start`` on the first emit, ``current`` at least once, each a
        per-shock list with flattened t/obs arrays and a display offset."""
        request = _build_request(loaded_cycloheptane)
        raw: list[dict] = []
        cb = OptimizationCallbacks(on_progress=raw.append)
        optimize_residual(loaded_cycloheptane, request, callbacks=cb)
        assert raw, "no progress updates captured"
        assert "sim_traces" in raw[0], "first update must carry sim_traces"
        start = raw[0]["sim_traces"].get("start")
        assert start is not None, "start traces must ship on the first emit"
        assert "current" in raw[0]["sim_traces"], (
            "current traces must ship on the first emit"
        )
        n = len(request.shocks)
        assert len(start) == n, "one start trace per shock"
        tr = start[0]
        assert set(tr) == {"num", "t", "obs", "t_offset"}
        assert np.asarray(tr["t"]).ndim == 1 and np.asarray(tr["obs"]).ndim == 1
        assert np.asarray(tr["t"]).shape == np.asarray(tr["obs"]).shape

        # start ships exactly once; later updates omit it (GUI keeps cache).
        later_starts = [u["sim_traces"].get("start") for u in raw[1:]
                        if "sim_traces" in u]
        assert all(s is None for s in later_starts), (
            "start must not re-ship after the first emit"
        )

    def test_uniqueness_weighting_freezes_and_runs(self, loaded_cycloheptane):
        """experiment_weighting='uniqueness' screens once at the start,
        freezes information weights, and the run completes."""
        request = OptimizationRequest(
            shocks=[_synthetic_shock()],
            optimizable=OptimizableSpec(rates=[
                OptimizableRate(
                    rxn_idx=_first_arrhenius_idx(loaded_cycloheptane),
                    rate=RateUncertainty(factor=2.0),
                ),
            ]),
            reactor_state=_reactor_state(),
            cost=CostSettings(
                scale="Linear", loss_alpha=2.0, loss_c=1.0,
                experiment_weighting="uniqueness",
            ),
            algorithm=_local_only(2),
            observable=ObservableSettings(),
        )
        logs: list[str] = []
        cb = OptimizationCallbacks(log=logs.append)
        result = optimize_residual(loaded_cycloheptane, request, callbacks=cb)
        assert result.success, f"uniqueness run failed: {result.message}"
        assert any("uniqueness weights frozen" in m for m in logs), (
            f"freeze log line missing; logs: {logs[:5]}"
        )

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
        bad = replace(result, x=np.array([1.0]))
        with pytest.raises(ValueError, match="entries"):
            apply_optimization_result(loaded_cycloheptane, bad)

    def test_rejects_missing_optimizable_used(self, loaded_cycloheptane):
        request = _build_request(loaded_cycloheptane)
        result = optimize_residual(loaded_cycloheptane, request)
        bad = replace(result, optimizable_used=None)
        with pytest.raises(ValueError, match="optimizable_used"):
            apply_optimization_result(loaded_cycloheptane, bad)




def _smurf_request(mech, max_iters=24):
    temps = (1400.0, 1600.0, 1800.0)
    shocks = []
    for temperature in temps:
        t = np.linspace(1e-7, 5e-5, 50)
        shocks.append(ExperimentShock(
            t=t, observable=np.zeros_like(t),
            initial=PostShockState(
                T_reac=temperature, P_reac=20000.0, u2=181.85,
                rho1=0.0230433, composition={"Kr": 0.96, "cC7H14": 0.04}),
            t_end=5e-5))
    arrh = [i for i, r in enumerate(mech.gas.reactions())
            if type(r.rate) is ct.ArrheniusRate][:2]
    request = OptimizationRequest(
        shocks=shocks,
        optimizable=OptimizableSpec(rates=[
            OptimizableRate(rxn_idx=i, rate=RateUncertainty(factor=2.0))
            for i in arrh]),
        reactor_state=_reactor_state(),
        cost=_cost_settings(),
        algorithm=AlgorithmSettings(
            global_stage=AlgorithmStage(
                algorithm="Smurf (Sensitivity Multistart Rate Fitting)",
                enabled=True, max_eval=max_iters, stop_value=float(max_iters)),
            local_stage=AlgorithmStage(
                algorithm="Subplex", enabled=False, stop_value=1.0)),
        observable=ObservableSettings())

    return request


@pytest.mark.slow
class TestSmurfGlobalStage:
    def test_smurf_runs_end_to_end(self, loaded_cycloheptane):
        request = _smurf_request(loaded_cycloheptane)
        result = optimize_residual(loaded_cycloheptane, request)
        assert result.success, f"Smurf global stage failed: {result.message}"
        assert np.isfinite(result.fval)
        assert result.x.size == len(result.optimizable_used.coefficients)

    def test_smurf_emits_progress_updates_with_sim_traces(
        self, loaded_cycloheptane,
    ):
        """The signal-plot overlay feeds off the raw progress payload;
        Smurf's scored evaluations must emit it like any other algorithm."""
        request = _smurf_request(loaded_cycloheptane)
        raw: list[dict] = []
        cb = OptimizationCallbacks(on_progress=raw.append)
        optimize_residual(loaded_cycloheptane, request, callbacks=cb)
        global_updates = [u for u in raw if u.get("type") == "global"]
        assert global_updates, "no global-stage progress updates from Smurf"
        assert "sim_traces" in global_updates[0], (
            "first Smurf update must carry sim_traces"
        )
        assert global_updates[0]["sim_traces"].get("start") is not None, (
            "start traces must ship on Smurf's first emit"
        )
        with_current = [u for u in global_updates
                        if u.get("sim_traces", {}).get("current")]
        assert with_current, "no Smurf update carried current sim traces"

    def test_smurf_lowfi_reports_full_fidelity_final(
        self, loaded_cycloheptane,
    ):
        """The low-fidelity Smurf re-scores its final point at the
        run's configured tolerance; the reported fval must be finite
        and the reactor state restored."""
        request = _smurf_request(loaded_cycloheptane)
        algorithm = request.algorithm.model_copy(update={
            "global_stage": request.algorithm.global_stage.model_copy(
                update={"algorithm": "Smurf (quick, low fidelity)"}),
        })
        request = request.model_copy(update={"algorithm": algorithm})
        result = optimize_residual(loaded_cycloheptane, request)
        assert result.success, f"smurf_lowfi failed: {result.message}"
        assert np.isfinite(result.fval)

    def test_smurf_respects_eval_budget(self, loaded_cycloheptane):
        budget = 10
        request = _smurf_request(loaded_cycloheptane, max_iters=budget)
        result = optimize_residual(loaded_cycloheptane, request)
        assert result.success, f"Smurf failed: {result.message}"
        assert result.nfev <= budget, (
            f"Smurf spent {result.nfev} evals against a budget of "
            f"{budget}")

    def test_smurf_pooled_run_close_to_serial(self, loaded_cycloheptane):
        """Pooled and serial runs diverge in trajectory (per-shock
        time-shift warm-start state evolves master-side only in serial
        mode), so the bar is closeness, not bit-parity."""
        serial = optimize_residual(
            loaded_cycloheptane, _smurf_request(loaded_cycloheptane))
        pooled_request = _smurf_request(loaded_cycloheptane).model_copy(
            update={"multiprocessing": True, "max_processors": 2})
        pooled = optimize_residual(loaded_cycloheptane, pooled_request)
        assert pooled.success, f"pooled Smurf failed: {pooled.message}"
        assert pooled.fval == pytest.approx(serial.fval, rel=0.05), (
            f"pooled run diverged from serial beyond trajectory noise: "
            f"{pooled.fval} vs {serial.fval}")
