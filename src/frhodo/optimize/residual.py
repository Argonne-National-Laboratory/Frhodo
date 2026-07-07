"""``optimize_residual`` — pure-Python optimization loop.

Owns the ``mp.Pool`` lifetime, builds the cost function, and runs the
algorithm dispatch. Logging, progress, and abort are callbacks.
"""
import multiprocessing as mp
import traceback
from dataclasses import dataclass, field, replace
from typing import Any, Callable

import numpy as np

from frhodo.common.scale import Scale
from frhodo.common.units import Bisymlog
from frhodo.experiment.uncertainty import correlation_length, sigma_bar
from frhodo.optimize._worker_context import MechBuildPayload
from frhodo.optimize.screening import (
    influential_mask,
    screen_campaign,
    uniqueness_weights,
)
from frhodo.optimize.shock_prep import trim_shocks
from frhodo.optimize.algorithms import Optimize
from frhodo.optimize.cost.fit_fcn import CostFunction, initialize_parallel_worker
from frhodo.optimize.cost.settings import CostSettings



@dataclass(frozen=True)
class OptimizeRunInputs:
    """Engine-level optimization run inputs.

    Bundles the optimization payload (already-prepped shocks +
    coefficient-target structures + initial scalers) with run-time
    knobs (reactor state, cost / algorithm settings, parallelism). The
    api and GUI worker both construct this directly.
    """
    mech: Any
    shocks2run: list
    coef_opt: list
    rxn_coef_opt: list
    rxn_rate_opt: dict
    initial_scalers: np.ndarray
    reactor_state: Any
    time_unc: float
    cost_settings: "CostSettings"
    opt_settings_optimize: dict
    multiprocessing: bool = True
    max_processors: int = 1
    random_t_uncertainty: bool = True
    # Frozen per-shock balance weights (uniqueness mode); None lets the
    # cost function apply the configured mode itself.
    experiment_weights: Any = None


@dataclass
class OptimizeRunCallbacks:
    """Side-effect hooks the optimizer fires during a run.

    All optional. ``mech_payload`` is the multiprocessing-worker init
    payload — required only when ``inputs.multiprocessing=True``.
    ``worker_pool`` is an optional :class:`PersistentWorkerPool` for
    spawn-cost amortization; when present, the run reuses its workers
    rather than spawning a fresh ``mp.Pool``.
    """
    display_shock_provider: Callable[[], object] | None = None
    abort_check: Callable[[], bool] | None = None
    log_callback: Callable[[str], None] | None = None
    progress_callback: Callable[[dict], None] | None = None
    mech_payload: "MechBuildPayload | None" = None
    worker_pool: Any = None  # PersistentWorkerPool — Any to avoid circular import
    mech: Any = None  # ChemicalMechanism — read for struct_version


def _apply_uniqueness_weighting(inputs: OptimizeRunInputs, log):
    """Freeze the information-uniqueness balance factor at the start point.

    Screens the campaign once (two solves per shock), drops experiments
    whose simulation responds to no reaction (mutating ``shocks2run`` in
    place), and returns the uniqueness factor for the kept experiments;
    the cost function multiplies live coverage weights into it (the
    hybrid balance). Both the factor and the mask are built from
    capability footprints — residual-free, so contaminated data cannot
    masquerade as unique information. Returns ``None`` on screening
    failure — the cost function then falls back to plain geometric
    coverage, logged.
    """
    try:
        result = screen_campaign(
            inputs.mech, inputs.shocks2run, inputs.reactor_state,
            inputs.cost_settings,
        )
    except Exception as e:
        log(
            "Uniqueness weighting unavailable "
            f"({e}); falling back to coverage weighting"
        )

        return None

    footprints = np.asarray(result.footprints)
    by_num = {num: row for num, row in zip(result.shock_nums, footprints)}
    ordered = []
    kept_shocks = []
    dropped = []
    for shock in inputs.shocks2run:
        num = int(getattr(shock, "num", 0) or 0)
        if num in by_num:
            ordered.append(by_num[num])
            kept_shocks.append(shock)
        else:
            dropped.append(num)
    G = np.asarray(ordered)

    keep = influential_mask(G)
    excluded = [int(getattr(s, "num", 0) or 0)
                for s, k in zip(kept_shocks, keep) if not k]
    if excluded or dropped:
        log(
            f"Excluding {len(excluded) + len(dropped)} experiment(s) with "
            f"no optimizable signal: {sorted(excluded + dropped)}"
        )
    inputs.shocks2run[:] = [s for s, k in zip(kept_shocks, keep) if k]
    weights = uniqueness_weights(G[keep])
    notable = [
        f"shock {int(getattr(s, 'num', 0) or 0)} ({w:.1f}x)"
        for s, w in zip(inputs.shocks2run, weights) if w > 2.0
    ]
    upweights = ""
    if notable:
        upweights = f"; unique-information upweights: {', '.join(notable)}"
    log(
        "Experiment balance: uniqueness weights frozen at the start "
        f"point ({len(inputs.shocks2run)} experiments){upweights}"
    )

    return weights


def optimize_residual(
    inputs: OptimizeRunInputs,
    callbacks: OptimizeRunCallbacks | None = None,
    *,
    debug: bool = False,
):
    """Run a residual optimization over reaction-rate coefficients.

    Args:
        inputs: Frozen bundle of prepared optimization data + run
            settings (reactor, cost, algorithm, parallelism).
        callbacks: Optional event hooks. ``mech_payload`` must be set
            when ``inputs.multiprocessing`` is ``True``.
        debug: When ``True``, algorithm exceptions propagate; when
            ``False`` they are caught and the run returns ``None``.

    Returns:
        Optimizer result dict, or ``None`` on failure / user abort.
    """
    cb = callbacks or OptimizeRunCallbacks()
    log = cb.log_callback or (lambda msg: None)
    progress = cb.progress_callback or (lambda update: None)
    abort = cb.abort_check or (lambda: False)

    pool_is_persistent = False
    if inputs.multiprocessing and cb.mech_payload is not None:
        if cb.worker_pool is not None and cb.mech is not None:
            pool = cb.worker_pool.acquire(
                workers=inputs.max_processors,
                mech=cb.mech,
                payload=cb.mech_payload,
            )
            pool_is_persistent = True
        else:
            pool = mp.Pool(
                processes=inputs.max_processors,
                initializer=initialize_parallel_worker,
                initargs=(cb.mech_payload,),
            )
    else:
        pool = None

    trim_shocks(inputs.shocks2run, inputs.cost_settings)

    if (inputs.cost_settings.experiment_weighting == "uniqueness"
            and inputs.experiment_weights is None):
        balance = _apply_uniqueness_weighting(inputs, log)
        if balance is not None:
            inputs = replace(inputs, experiment_weights=balance)

    fit_fun = CostFunction(
        inputs,
        pool=pool,
        display_shock_provider=cb.display_shock_provider,
        log_callback=log,
        progress_callback=progress,
    )

    if pool is not None:
        fit_fun.warmup_workers(inputs.max_processors, inputs.initial_scalers)
    fit_fun.calibrate_error_floor(inputs.initial_scalers)

    def eval_fun(s, grad=None):
        if abort():
            log("\nOptimization aborted")
            raise Exception("Optimization terminated by user")

        return fit_fun(s)

    optimize = Optimize(
        eval_fun,
        inputs.initial_scalers,
        inputs.rxn_rate_opt["bnds"],
        inputs.opt_settings_optimize,
        fit_fun,
    )
    try:
        res = optimize.run()
    except Exception as e:
        if debug:
            if pool is not None and not pool_is_persistent:
                pool.close()
            raise
        res = None
        if "Optimization terminated by user" not in str(e):
            log("\n" + traceback.format_exc())
    finally:
        if pool is not None and not pool_is_persistent:
            pool.close()

    return res
