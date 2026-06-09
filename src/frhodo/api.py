"""Public API for the Frhodo simulation engine.

Wraps ``ChemicalMechanism``, the reactor entry points, and
``ShockJumpSolver`` behind typed inputs and frozen result dataclasses.

Example::

    from frhodo import load_mechanism, ShockTubeConfig, run_shock_tube

    mech = load_mechanism("mech.yaml")
    cfg = ShockTubeConfig(
        T_reac=1500.0, P_reac=20_000.0,
        composition={"Kr": 0.96, "cC7H14": 0.04},
        u_incident=1029.0, rho1=0.4, t_end=5e-5,
    )
    result = run_shock_tube(mech, cfg)
    if result.success:
        print(result.t.shape, result.observable.shape)
"""
import multiprocessing as mp
import sys
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import cantera as ct
import numpy as np

from frhodo.simulation.mechanism import ChemicalMechanism
from frhodo.simulation.mechanism.mechanism_loader import MechanismLoader
from frhodo.simulation.shock import (
    RuntimeReactorState,
    ShockJumpSolver,
    run_incident_shock as _run_incident_shock,
    run_zero_d as _run_zero_d,
)
from frhodo.common.errors import FailureReason
from frhodo.common import (
    ObservableSettings,
    PostShockState,
    PreShockState,
    ShockTubeConfig,
    SolverSettings,
    ZeroDConfig,
)
from frhodo.common.scale import Scale
from frhodo.common.units import kJ_per_mol, kcal_per_mol
from frhodo.experiment.profiles import (
    ExperimentShock,
    WeightProfile,
)
from frhodo.experiment.uncertainty import (
    bounds_from_sigma,
    estimate_pointwise_sigma,
)
from frhodo.optimize._worker_context import MechBuildPayload, WorkerContext
from frhodo.optimize.algorithm_settings import AlgorithmSettings, AlgorithmStage
from frhodo.optimize.cost.settings import CostSettings
from frhodo.optimize.parameters import OptimizableSet
from frhodo.optimize.request import OptimizationRequest
from frhodo.optimize.spec import (
    CoefUncertainty,
    OptimizableRate,
    OptimizableSpec,
    OptimizableSpecBuilder,
    RateUncertainty,
)


_YAML_SUFFIXES = (".yaml", ".yml")


def load_mechanism(
    mech: str | Path,
    *,
    thermo: str | Path | None = None,
    converted_yaml: str | Path | None = None,
) -> ChemicalMechanism:
    """Load a Cantera mechanism from YAML, CTI, or Chemkin input.

    YAML inputs load directly. CTI and Chemkin inputs are converted to
    YAML.

    Args:
        mech: Path to the mechanism source file.
        thermo: Optional Chemkin thermodynamic database; ignored for
            YAML inputs.
        converted_yaml: Where to write the converted YAML. Defaults to
            a sibling of ``mech`` with a ``.converted.yaml`` suffix.

    Returns:
        A populated :class:`ChemicalMechanism` ready to feed the
        simulation runners.
    """
    mech_path = Path(mech)
    if converted_yaml is None:
        if mech_path.suffix in _YAML_SUFFIXES:
            converted_yaml_path = mech_path
        else:
            converted_yaml_path = mech_path.with_suffix(".converted.yaml")
    else:
        converted_yaml_path = Path(converted_yaml)

    paths = {
        "mech": mech_path,
        "thermo": Path(thermo) if thermo is not None else None,
        "Cantera_Mech": converted_yaml_path,
    }

    return MechanismLoader().load(paths)


def parse_composition(text: str) -> dict[str, float]:
    """Parse a Cantera-style composition string into ``dict[str, float]``.

    Entries are ``NAME:fraction`` pairs separated by commas or newlines.
    Whitespace around names and fractions is ignored.

    Raises:
        ValueError: For malformed entries (missing separator, empty
            name, non-numeric fraction) or when the parsed result is
            empty.
    """
    parsed: dict[str, float] = {}
    for raw_part in text.replace("\n", ",").split(","):
        part = raw_part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(
                f"composition entry {part!r} is missing 'name:fraction' separator"
            )
        name, frac = part.split(":", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"composition entry {part!r} has empty species name")
        try:
            parsed[name] = float(frac)
        except ValueError as exc:
            raise ValueError(
                f"composition entry {part!r} has non-numeric fraction"
            ) from exc
    if not parsed:
        raise ValueError("composition string parsed to an empty dict")

    return parsed


@dataclass(frozen=True)
class SimulationResult:
    """Frozen output of :func:`run_shock_tube` / :func:`run_zero_d`.

    Attributes:
        success: ``True`` when the integrator completed; trajectory
            arrays are non-empty only in this case.
        failure_reason: Typed failure category, or ``None`` on success.
        message: Human-readable diagnostic.
        t, T, P, rho, Y: Trajectory arrays.
        species: Species names parallel to ``Y``'s columns.
        observable: The configured observable trace.
        h_tot, s_tot, wdot, HRR_tot, drhodz_tot: Optional derived
            traces; ``None`` when the underlying reactor did not
            expose them.
        shock: Resolved jump conditions when the run started from a
            :class:`PreShockState`; ``None`` otherwise.
        cantera_array: Underlying ``ct.SolutionArray``. Tied to the
            installed Cantera version — not part of the public
            contract.
    """
    success: bool
    failure_reason: FailureReason | None = None
    message: str = ""
    t: np.ndarray = field(default_factory=lambda: np.array([]))
    T: np.ndarray = field(default_factory=lambda: np.array([]))
    P: np.ndarray = field(default_factory=lambda: np.array([]))
    rho: np.ndarray = field(default_factory=lambda: np.array([]))
    Y: np.ndarray = field(default_factory=lambda: np.array([]))
    species: tuple[str, ...] = ()
    observable: np.ndarray = field(default_factory=lambda: np.array([]))
    h_tot: np.ndarray | None = None
    s_tot: np.ndarray | None = None
    wdot: np.ndarray | None = None
    HRR_tot: np.ndarray | None = None
    drhodz_tot: np.ndarray | None = None
    shock: "ShockState | None" = None
    cantera_array: ct.SolutionArray | None = None


@dataclass(frozen=True)
class ShockState:
    """Pre- and post-shock conditions from ``solve_shock_jump`` (zones 1, 2, 5).

    When attached to :attr:`SimulationResult.shock`, ``initial`` carries
    the original :class:`PreShockState` the auto-jump consumed.
    """
    success: bool
    message: str = ""
    initial: PreShockState | None = None
    T1: float = 0.0
    P1: float = 0.0
    u1: float = 0.0
    T2: float = 0.0
    P2: float = 0.0
    u2: float = 0.0
    rho1: float = 0.0
    T5: float = 0.0
    P5: float = 0.0


def run_shock_tube(mech: ChemicalMechanism, cfg: ShockTubeConfig) -> SimulationResult:
    """Run one Incident Shock Reactor simulation.

    Auto-solves the normal-shock jump when ``cfg.initial`` is a
    :class:`PreShockState`; otherwise consumes the supplied
    :class:`PostShockState` directly.

    ``ChemicalMechanism`` is not thread-safe — :meth:`exclusive`
    raises :class:`RuntimeError` on concurrent entry. For parallelism,
    use :func:`run_shock_tubes` with ``workers > 1``.

    Returns:
        :class:`SimulationResult`. On jump-solver failure, returns a
        result with ``success=False`` and the failure reason set; on
        reactor failure, the same with the integrator's diagnostic.
    """
    if isinstance(cfg.initial, PreShockState):
        shock_state = solve_shock_jump(cfg.initial, mech)
        if not shock_state.success:
            return SimulationResult(
                success=False,
                failure_reason=FailureReason.FROSH_NOT_CONVERGED,
                message=shock_state.message or "shock jump solver failed",
                shock=shock_state,
            )
        T_reac = shock_state.T2
        P_reac = shock_state.P2
        u_incident = shock_state.u2
        rho1 = shock_state.rho1
        composition = cfg.initial.composition
    else:
        shock_state = None
        T_reac = cfg.initial.T_reac
        P_reac = cfg.initial.P_reac
        u_incident = cfg.initial.u_incident
        rho1 = cfg.initial.rho1
        composition = cfg.initial.composition

    kwargs: dict = {
        "u_reac": u_incident,
        "rho1": rho1,
        "A1": cfg.A1,
        "As": cfg.As,
        "L": cfg.L,
        "area_change": cfg.area_change,
        "ODE_solver": cfg.solver.solver,
        "sim_int_f": cfg.solver.sim_interp_factor,
        "observable": {"main": cfg.observable.main, "sub": cfg.observable.sub},
    }
    if cfg.solver.rtol is not None:
        kwargs["rtol"] = cfg.solver.rtol
    if cfg.solver.atol is not None:
        kwargs["atol"] = cfg.solver.atol
    if cfg.save_times is not None:
        kwargs["t_lab_save"] = np.asarray(cfg.save_times, dtype=float)

    SIM, details = _run_incident_shock(
        mech, cfg.t_end, T_reac, P_reac,
        composition, **kwargs,
    )
    result = _to_simulation_result(SIM, details)

    return dataclasses.replace(result, shock=shock_state)


def run_zero_d(mech: ChemicalMechanism, cfg: ZeroDConfig) -> SimulationResult:
    """Run one 0-D reactor simulation (constant volume or constant pressure).

    Returns:
        :class:`SimulationResult`. ``shock`` is always ``None`` for
        0-D runs.
    """
    kwargs: dict = {
        "solve_energy": cfg.solve_energy,
        "frozen_comp": cfg.frozen_comp,
        "rtol": cfg.solver.rtol if cfg.solver.rtol is not None else 1e-4,
        "atol": cfg.solver.atol if cfg.solver.atol is not None else 1e-7,
        "sim_int_f": cfg.solver.sim_interp_factor,
        "observable": {"main": cfg.observable.main, "sub": cfg.observable.sub},
    }
    if cfg.save_times is not None:
        kwargs["t_lab_save"] = np.asarray(cfg.save_times, dtype=float)

    SIM, details = _run_zero_d(
        mech, cfg.mode, cfg.t_end, cfg.T_reac, cfg.P_reac,
        cfg.composition, **kwargs,
    )

    return _to_simulation_result(SIM, details)


def solve_shock_jump(initial: PreShockState, mech: ChemicalMechanism) -> ShockState:
    """Compute zone-2 and zone-5 conditions from a :class:`PreShockState`.

    Returns:
        :class:`ShockState`. ``success=False`` carries the convergence
        diagnostic in ``message`` and leaves the zone fields at their
        zero defaults.
    """
    shock_vars = {
        "T1": initial.T1,
        "P1": initial.P1,
        "u1": initial.u1,
        "mix": dict(initial.composition),
    }
    with mech.exclusive():
        solver = ShockJumpSolver(mech.gas, shock_vars)
        if not solver.success:
            return ShockState(
                success=False, message="Shock jump solver failed",
                initial=initial,
            )

    r = solver.res

    return ShockState(
        success=True,
        initial=initial,
        T1=r.T1, P1=r.P1, u1=r.u1, rho1=r.rho1,
        T2=r.T2, P2=r.P2, u2=r.u2,
        T5=r.T5, P5=r.P5,
    )


_CORE_PROPERTIES = ("T", "P", "rho", "Y")
_OPTIONAL_PROPERTIES = ("h_tot", "s_tot", "wdot", "HRR_tot", "drhodz_tot")


def _fetch_property(SIM, name: str) -> np.ndarray | None:
    getter = getattr(SIM, name, None)
    if not callable(getter):
        return None
    try:
        return np.asarray(getter(units="SI"))
    except Exception:
        return None


def _to_simulation_result(SIM, details: dict) -> SimulationResult:
    success = bool(details.get("success", False))
    raw_msg = details.get("message", "")
    if isinstance(raw_msg, list):
        message = "\n".join(str(m) for m in raw_msg)
    else:
        message = str(raw_msg)

    failure_reason = details.get("failure_reason") if not success else None

    if SIM is None or not success:
        return SimulationResult(
            success=success,
            failure_reason=failure_reason,
            message=message,
        )

    states = getattr(SIM, "states", None)
    if states is None or len(states) == 0:
        return SimulationResult(
            success=False,
            failure_reason=FailureReason.SOLVER_FAILURE,
            message=message or "no states produced",
        )

    species = tuple(states.species_names)
    observable = np.asarray(SIM.observable) if hasattr(SIM, "observable") else np.array([])
    independent_var = (
        np.asarray(SIM.independent_var) if hasattr(SIM, "independent_var") else np.array([])
    )

    core_values: dict[str, np.ndarray] = {}
    for name in _CORE_PROPERTIES:
        value = _fetch_property(SIM, name)
        core_values[name] = value if value is not None else np.array([])

    optional_values: dict[str, np.ndarray | None] = {
        name: _fetch_property(SIM, name) for name in _OPTIONAL_PROPERTIES
    }

    return SimulationResult(
        success=True,
        message=message,
        t=independent_var,
        T=core_values["T"],
        P=core_values["P"],
        rho=core_values["rho"],
        Y=core_values["Y"],
        species=species,
        observable=observable,
        h_tot=optional_values["h_tot"],
        s_tot=optional_values["s_tot"],
        wdot=optional_values["wdot"],
        HRR_tot=optional_values["HRR_tot"],
        drhodz_tot=optional_values["drhodz_tot"],
        cantera_array=states,
    )


def run_shock_tubes(
    mech: ChemicalMechanism | str | Path,
    cfgs: Sequence[ShockTubeConfig],
    *,
    workers: int | None = None,
) -> list[SimulationResult]:
    """Batch sibling of :func:`run_shock_tube`.

    Args:
        mech: Loaded mechanism or path to one. When parallel and on
            Windows, must be a path; on Linux/macOS either works.
        cfgs: Configurations to run, one result per config.
        workers: ``None`` or ``1`` runs sequentially in the current
            process. ``>1`` uses ``multiprocessing.Pool`` with
            ``fork`` on Linux/macOS (workers inherit ``mech``) or
            ``spawn`` on Windows (each worker loads its own copy).

    Returns:
        List of :class:`SimulationResult` parallel to ``cfgs``.

    Raises:
        ValueError: ``workers > 1`` on Windows with a live
            :class:`ChemicalMechanism` instance — spawn workers can't
            inherit it, so pass a path instead.
    """
    if workers is None or workers == 1:
        loaded = _resolve_mech(mech)
        return [run_shock_tube(loaded, cfg) for cfg in cfgs]

    if sys.platform == "win32":
        if isinstance(mech, ChemicalMechanism):
            raise ValueError(
                "run_shock_tubes(workers>1) on Windows requires mech as a "
                "path string or pathlib.Path; spawn workers cannot inherit "
                "a loaded ChemicalMechanism."
            )
        ctx = mp.get_context("spawn")
        initargs = (str(Path(mech)),)
        init = _spawn_worker_init
    else:
        _FORK_HANDOFF.mech = _resolve_mech(mech) if isinstance(mech, ChemicalMechanism) else None
        _FORK_HANDOFF.mech_path = None if _FORK_HANDOFF.mech is not None else str(Path(mech))
        ctx = mp.get_context("fork")
        initargs = ()
        init = _fork_worker_init

    try:
        with ctx.Pool(processes=workers, initializer=init, initargs=initargs) as pool:
            return pool.map(_pool_run_one, list(cfgs))
    finally:
        _FORK_HANDOFF.mech = None
        _FORK_HANDOFF.mech_path = None


def _resolve_mech(mech: ChemicalMechanism | str | Path) -> ChemicalMechanism:
    if isinstance(mech, ChemicalMechanism):
        return mech

    return load_mechanism(mech)


_FORK_HANDOFF = WorkerContext()
_RUN_CTX = WorkerContext()


def _fork_worker_init():
    if _FORK_HANDOFF.mech is not None:
        _RUN_CTX.mech = _FORK_HANDOFF.mech
    elif _FORK_HANDOFF.mech_path is not None:
        _RUN_CTX.mech = load_mechanism(_FORK_HANDOFF.mech_path)


def _spawn_worker_init(mech_path: str):
    _RUN_CTX.mech = load_mechanism(mech_path)


def _pool_run_one(cfg: ShockTubeConfig) -> SimulationResult:
    mech = _RUN_CTX.mech
    assert mech is not None, "worker pool dispatched a task before init set _RUN_CTX.mech"

    return run_shock_tube(mech, cfg)


@dataclass(frozen=True)
class OptimizationResult:
    """Output of :func:`optimize_residual`.

    Attributes:
        success: ``True`` when the optimizer finished naturally.
        message: Human-readable termination reason.
        aborted: ``True`` when stopped via :attr:`OptimizationCallbacks.abort`.
        x: Best coefficient vector found.
        fval: Objective value at ``x``.
        nfev: Total evaluations consumed.
        elapsed_s: Wall-clock seconds.
        optimizable_used: Resolved :class:`OptimizableSet` the
            optimizer actually worked on (post-recast). Use to
            interpret ``x`` slot-by-slot via
            ``optimizable_used.slot_index(...)``.
        raw: Full dict returned by the optimization driver, for
            callers that need fields not exposed on this dataclass.
    """
    success: bool
    message: str = ""
    aborted: bool = False
    x: np.ndarray = field(default_factory=lambda: np.array([]))
    fval: float = float("inf")
    nfev: int = 0
    elapsed_s: float = 0.0
    optimizable_used: "OptimizableSet | None" = None
    raw: Mapping | None = None


@dataclass(frozen=True)
class StartInfo:
    """Engine-emitted snapshot at optimization start, after any recast.

    Fires once before the first iteration. Always fires, even when no
    recast happened, so clients can stash ``optimizable_used`` for
    per-iteration x-vector interpretation.
    """
    optimizable_used: "OptimizableSet"
    recast_rxns: tuple[int, ...] = ()
    n_shocks: int = 0
    max_processors: int = 1


@dataclass(frozen=True)
class IterationUpdate:
    """Engine-emitted snapshot at one optimization iteration."""
    iter: int
    fval: float
    x: np.ndarray
    is_best: bool
    stage: str
    elapsed_s: float
    ind_var: np.ndarray | None = None
    observable: np.ndarray | None = None
    stat_plot: Mapping | None = None


@dataclass(frozen=True)
class StageComplete:
    """Engine-emitted summary at the end of one optimization stage."""
    stage: str
    success: bool
    message: str
    fval: float
    nfev: int
    elapsed_s: float
    shock_evals: int


@dataclass
class OptimizationCallbacks:
    """Domain-event hooks the optimizer fires during a run.

    Clients render however they want. ``log`` carries free-form text
    progress messages; routing severity through Python's ``logging``
    module (the GUI handler maps ``WARNING+`` to the blinking-tab
    alert) is the diagnostic-emission path.

    ``on_iteration`` receives a typed :class:`IterationUpdate`;
    ``on_progress`` receives the raw per-iteration dict the optimizer
    emits internally and is intended for consumers that need fields
    not exposed on :class:`IterationUpdate` (e.g. ``stat_plot`` for
    live diagnostic plots). Both fire from the same source when set.

    ``display_shock_provider`` returns the currently-rendered shock at
    call time; the optimizer calls it once per iteration so the
    consumer can switch the rendered shock mid-run. When ``None`` the
    engine falls back to ``request.display_shock_index``.
    """
    on_start: Callable[[StartInfo], None] | None = None
    on_iteration: Callable[[IterationUpdate], None] | None = None
    on_progress: Callable[[Mapping], None] | None = None
    on_stage_complete: Callable[[StageComplete], None] | None = None
    log: Callable[[str], None] | None = None
    abort: Callable[[], bool] | None = None
    display_shock_provider: Callable[[], object] | None = None
    worker_pool: Any = None  # PersistentWorkerPool — Any to avoid circular import


def optimize_residual(
    mech: ChemicalMechanism,
    request: "OptimizationRequest",
    *,
    callbacks: OptimizationCallbacks | None = None,
) -> OptimizationResult:
    """Run a residual-objective optimization against ``mech``.

    The request is translated to the internal :class:`OptimizableSet`,
    legacy shock objects, and algorithm dict the optimizer loop
    expects. Callers interpret ``result.x`` slot-by-slot through
    ``result.optimizable_used.slot_index(...)`` /
    ``.rxn_slots(...)``.

    Args:
        mech: Mechanism to optimize. Mutated in place during the run.
        request: Typed bundle of all inputs (shocks, optimizable spec,
            reactor settings, cost / algorithm settings).
        callbacks: Optional event hooks. Defaults to all-quiet.

    Returns:
        :class:`OptimizationResult`. Always populated even on abort
        or failure; check ``success`` and ``aborted``.
    """
    import scipy.stats

    from frhodo.simulation.mechanism.coef_helpers import rates as compute_rates
    from frhodo.optimize.parameters import build_rxn_coef_opt, build_rxn_rate_opt
    from frhodo.optimize.residual import OptimizeRunInputs

    cb = callbacks or OptimizationCallbacks()

    optimizable_builder = request.optimizable.to_builder(mech)
    optimizable_set = optimizable_builder.build(mech)
    coef_opt = list(optimizable_set.coefficients)
    if not coef_opt:
        return OptimizationResult(
            success=False,
            message="OptimizableSpec is empty (no reactions or coefficients selected)",
            optimizable_used=optimizable_set,
        )

    shocks2run = [_to_internal_shock(s, request, mech) for s in request.shocks]

    rxn_coef_opt = build_rxn_coef_opt(mech, coef_opt, shocks2run)
    rxn_rate_opt = build_rxn_rate_opt(mech, rxn_coef_opt)

    _, mech_rebuilt = mech.recast_to_troe(
        rxn_coef_opt, rxn_rate_opt, optimizable_builder,
    )

    if cb.log is not None and mech.recast_log_rms:
        for rxnIdx in sorted(mech.recast_log_rms):
            log_rms = mech.recast_log_rms[rxnIdx]
            cb.log(f"R{rxnIdx + 1} recast to Troe: fit log-RMS = {log_rms:.4f}")

    if mech_rebuilt:
        optimizable_set = request.optimizable.build(mech)
        coef_opt = list(optimizable_set.coefficients)
        rxn_coef_opt = build_rxn_coef_opt(mech, coef_opt, shocks2run)
        rxn_rate_opt = build_rxn_rate_opt(mech, rxn_coef_opt)

    lb, ub = rxn_rate_opt["bnds"]["lower"], rxn_rate_opt["bnds"]["upper"]
    initial_scalers = compute_rates(rxn_coef_opt, mech) - rxn_rate_opt["x0"]
    initial_scalers = np.clip(initial_scalers, lb * (1 + 1e-9), ub * (1 - 1e-9))

    default_display_shock = None
    if request.display_shock_index is not None:
        default_display_shock = shocks2run[request.display_shock_index]

    inputs = OptimizeRunInputs(
        mech=mech,
        shocks2run=shocks2run,
        coef_opt=coef_opt,
        rxn_coef_opt=rxn_coef_opt,
        rxn_rate_opt=rxn_rate_opt,
        initial_scalers=initial_scalers,
        reactor_state=request.reactor_state,
        time_unc=request.time_uncertainty,
        cost_settings=request.cost,
        opt_settings_optimize=request.algorithm.to_legacy_dict(),
        dist=scipy.stats.norm,
        multiprocessing=request.multiprocessing,
        max_processors=request.max_processors,
        random_t_uncertainty=request.random_t_uncertainty,
    )

    return _run_optimization_engine(
        inputs,
        optimizable_set=optimizable_set,
        default_display_shock=default_display_shock,
        callbacks=callbacks,
    )


def _run_optimization_engine(
    inputs: "OptimizeRunInputs",
    *,
    optimizable_set: "OptimizableSet",
    default_display_shock: Any = None,
    callbacks: OptimizationCallbacks | None = None,
) -> OptimizationResult:
    """Drive the residual-loop optimizer with prepared engine inputs.

    Wraps :func:`frhodo.optimize.residual.optimize_residual` to emit
    the api-level :class:`OptimizationCallbacks` events and shape the
    raw driver dict into an :class:`OptimizationResult`.
    """
    from frhodo.optimize.residual import (
        OptimizeRunCallbacks,
        optimize_residual as _run_residual_loop,
    )

    cb = callbacks or OptimizationCallbacks()
    mech = inputs.mech

    if cb.on_start is not None:
        cb.on_start(StartInfo(
            optimizable_used=optimizable_set,
            n_shocks=len(inputs.shocks2run),
            max_processors=inputs.max_processors,
        ))

    progress_adapter = _ProgressAdapter(cb.on_iteration) if cb.on_iteration else None
    if progress_adapter is not None and cb.on_progress is not None:
        def raw_progress(update):
            progress_adapter(update)
            cb.on_progress(update)
    elif progress_adapter is not None:
        raw_progress = progress_adapter
    elif cb.on_progress is not None:
        raw_progress = cb.on_progress
    else:
        raw_progress = None

    log_cb = (lambda msg: cb.log(str(msg))) if cb.log is not None else None
    shock_provider = (
        cb.display_shock_provider
        if cb.display_shock_provider is not None
        else (lambda s=default_display_shock: s)
    )

    engine_callbacks = OptimizeRunCallbacks(
        display_shock_provider=shock_provider,
        abort_check=cb.abort,
        log_callback=log_cb,
        progress_callback=raw_progress,
        mech_payload=MechBuildPayload(
            reset_mech=mech.reset_mech,
            thermo_coeffs=mech.thermo_coeffs,
            coeffs=mech.coeffs,
            coeffs_bnds=mech.coeffs_bnds,
            rate_bnds=mech.rate_bnds,
        ) if inputs.multiprocessing else None,
        worker_pool=cb.worker_pool,
        mech=mech,
    )

    raw = _run_residual_loop(inputs, engine_callbacks)

    aborted = bool(cb.abort and cb.abort())
    if raw is None:
        return OptimizationResult(
            success=False, aborted=aborted,
            message="optimization aborted" if aborted else "optimization failed",
            optimizable_used=optimizable_set,
        )

    if cb.on_stage_complete is not None:
        for stage_name in ("global", "local"):
            stage = raw.get(stage_name)
            if stage is None:
                continue
            cb.on_stage_complete(StageComplete(
                stage=stage_name,
                success=bool(stage.get("success", False)),
                message=str(stage.get("message", "")),
                fval=float(stage.get("fval", float("inf"))),
                nfev=int(stage.get("nfev", 0)),
                elapsed_s=float(stage.get("time", 0.0)),
                shock_evals=int(stage.get("nfev", 0)) * len(inputs.shocks2run),
            ))

    last = raw.get("local") or raw.get("global") or {}

    return OptimizationResult(
        success=bool(last.get("success", False)),
        aborted=aborted,
        message=str(last.get("message", "")),
        x=np.asarray(last.get("x", [])),
        fval=float(last.get("fval", float("inf"))),
        nfev=int(last.get("nfev", 0)),
        elapsed_s=float(last.get("time", 0.0)),
        optimizable_used=optimizable_set,
        raw=raw,
    )


def apply_optimization_result(
    mech: ChemicalMechanism,
    result: OptimizationResult,
    *,
    save_path: str | Path | None = None,
) -> None:
    """Apply optimized coefficients back to ``mech`` in place.

    Uses ``result.optimizable_used`` to map each entry of ``result.x``
    to its (reaction, coefficient) slot. When ``save_path`` is given,
    writes the updated mechanism in the format implied by the suffix:

    * ``.yaml`` / ``.yml`` → Cantera YAML
    * ``.inp`` / ``.dat`` / ``.mech`` → Chemkin

    Raises:
        ValueError: ``result.optimizable_used`` is ``None`` (result
            came from a failed/aborted run that did not finalize),
            ``result.x`` is the wrong length for the recorded
            optimizable set, or ``save_path`` has an unsupported
            suffix.
    """
    from frhodo.optimize.cost.fit_fcn import update_mech_coef_opt

    if result.optimizable_used is None:
        raise ValueError(
            "OptimizationResult.optimizable_used is None; cannot map x to coefficients"
        )
    if result.x.size != len(result.optimizable_used.coefficients):
        raise ValueError(
            f"result.x has {result.x.size} entries but optimizable_used has "
            f"{len(result.optimizable_used.coefficients)} coefficients"
        )

    coef_opt = list(result.optimizable_used.coefficients)
    update_mech_coef_opt(mech, coef_opt, result.x)

    if save_path is not None:
        path = Path(save_path)
        suffix = path.suffix.lower()
        if suffix in (".yaml", ".yml"):
            path.write_text(mech.to_yaml_text())
        elif suffix in (".inp", ".dat", ".mech"):
            mech.to_chemkin(path)
        else:
            raise ValueError(
                f"apply_optimization_result: unsupported save_path suffix "
                f"{suffix!r}; expected one of .yaml, .yml, .inp, .dat, .mech"
            )


class _ProgressAdapter:
    """Translates the legacy per-iteration progress dict into an
    :class:`IterationUpdate`, tracking is-best and elapsed wall time."""

    def __init__(self, on_iteration: Callable[[IterationUpdate], None]):
        import time

        self._on_iteration = on_iteration
        self._best_fval: float = float("inf")
        self._t0 = time.perf_counter()

    def __call__(self, update: Mapping) -> None:
        import time

        fval = float(update.get("obj_fcn", float("inf")))
        is_best = fval < self._best_fval
        if is_best:
            self._best_fval = fval

        self._on_iteration(IterationUpdate(
            iter=int(update.get("i", 0)),
            fval=fval,
            x=np.asarray(update.get("x", [])),
            is_best=is_best,
            stage=str(update.get("type", "")),
            elapsed_s=time.perf_counter() - self._t0,
            ind_var=update.get("ind_var"),
            observable=update.get("observable"),
            stat_plot=update.get("stat_plot"),
        ))


def _to_internal_shock(shock: "ExperimentShock", request: "OptimizationRequest", mech):
    """Convert an :class:`ExperimentShock` into the legacy shape the
    optimizer consumes (reactor inlet state + experimental trace +
    per-sample weights / uncertainties).

    Samples outside ``weight_profile.cutoff_*`` get weight 0 so the
    downstream ``_trim_shocks`` drops them; samples inside carry the
    ``WeightProfile`` envelope scaled by ``shock.scalar_weight``.
    """
    from frhodo.experiment import ExperimentalShock

    if isinstance(shock.initial, PreShockState):
        ss = solve_shock_jump(shock.initial, mech)
        if not ss.success:
            raise ValueError(
                f"shock-jump solver failed for shock initial={shock.initial}: "
                f"{ss.message}"
            )
        T_reac, P_reac = ss.T2, ss.P2
        u2, rho1 = ss.u2, ss.rho1
        T1, P1, u1 = ss.T1, ss.P1, ss.u1
    else:
        T_reac, P_reac = shock.initial.T_reac, shock.initial.P_reac
        u2 = shock.initial.u_incident
        rho1 = shock.initial.rho1
        T1, P1, u1 = float("nan"), float("nan"), float("nan")

    t = shock.t_array()
    obs = shock.observable_array()
    exp_data = np.column_stack([t, obs])

    duration = float(t[-1] - t[0]) if t.size >= 2 else 1.0
    t_percent = 100.0 * (t - t[0]) / max(duration, 1e-30)

    weight_profile = shock.weight_profile or request.default_weight_profile
    envelope = weight_profile.evaluate(t_percent)
    in_window = (t_percent >= weight_profile.cutoff_pre) & (
        t_percent <= weight_profile.cutoff_post
    )
    normalized_weights = np.where(in_window, envelope * shock.scalar_weight, 0.0)

    abs_uncertainties = np.zeros((0, 2))
    sigma_t = np.array([])
    if request.cost.obj_fcn_type == "Bayesian":
        scale = Scale(request.cost.scale, calibration_data=obs)
        sigma_t = estimate_pointwise_sigma(obs, scale=scale)
        abs_uncertainties = bounds_from_sigma(
            obs, sigma_t,
            sigma_multiple=request.cost.bayes_unc_sigma,
            scale=scale,
        )

    return ExperimentalShock.from_dict({
        "T1": T1, "P1": P1, "u1": u1, "rho1": rho1, "u2": u2,
        "T_reactor": T_reac, "P_reactor": P_reac,
        "thermo_mix": dict(shock.initial.composition),
        "observable": {
            "main": request.observable.main,
            "sub": request.observable.sub,
        },
        "exp_data": exp_data,
        "normalized_weights": normalized_weights,
        "weights": normalized_weights.copy(),
        "sigma_t": sigma_t,
        "abs_uncertainties": abs_uncertainties,
        "opt_time_offset": 0.0,
    })


__all__ = [
    "AlgorithmSettings",
    "AlgorithmStage",
    "ChemicalMechanism",
    "CoefUncertainty",
    "CostSettings",
    "ExperimentShock",
    "IterationUpdate",
    "ObservableSettings",
    "OptimizableRate",
    "OptimizableSet",
    "OptimizableSpec",
    "OptimizableSpecBuilder",
    "OptimizationCallbacks",
    "OptimizationRequest",
    "OptimizationResult",
    "PostShockState",
    "PreShockState",
    "RateUncertainty",
    "ShockState",
    "StageComplete",
    "StartInfo",
    "ShockTubeConfig",
    "SimulationResult",
    "SolverSettings",
    "WeightProfile",
    "ZeroDConfig",
    "kJ_per_mol",
    "kcal_per_mol",
    "apply_optimization_result",
    "load_mechanism",
    "optimize_residual",
    "parse_composition",
    "run_shock_tube",
    "run_shock_tubes",
    "run_zero_d",
    "solve_shock_jump",
]
