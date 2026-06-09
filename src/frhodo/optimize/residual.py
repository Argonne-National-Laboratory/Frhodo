"""``optimize_residual`` — pure-Python optimization loop.

Owns the ``mp.Pool`` lifetime, builds the cost function, and runs the
algorithm dispatch. Logging, progress, and abort are callbacks.
"""
import multiprocessing as mp
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from frhodo.common.units import Bisymlog
from frhodo.optimize._worker_context import MechBuildPayload
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
    dist: Any
    multiprocessing: bool = True
    max_processors: int = 1
    random_t_uncertainty: bool = True


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


def _trim_shocks(shocks2run: list, cost_settings: "CostSettings") -> None:
    """Pre-mask each shock to the points where the weight profile is non-zero.

    Mutates each ``shock`` in place — populates ``weights_trim``,
    ``exp_data_trim``, ``abs_uncertainties_trim``, and ``bisymlog``.
    Skipping zero-weight rows up front saves work inside every cost
    evaluation.
    """
    for shock in shocks2run:
        weights = shock.normalized_weights
        exp_bounds = np.nonzero(weights)[0]
        shock.weights_trim = weights[exp_bounds]
        shock.exp_data_trim = shock.exp_data[exp_bounds, :]
        if shock.abs_uncertainties.size > 0:
            shock.abs_uncertainties_trim = shock.abs_uncertainties[exp_bounds, :]

        if cost_settings.scale == "Bisymlog":
            bisymlog = Bisymlog(
                C=None, scaling_factor=cost_settings.bisymlog_scaling_factor,
            )
            bisymlog.set_C_heuristically(shock.exp_data_trim[:, 1])
            shock.bisymlog = bisymlog
        else:
            shock.bisymlog = None


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
        debug: Re-raise the algorithm's exception instead of returning
            ``None``.

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

    _trim_shocks(inputs.shocks2run, inputs.cost_settings)

    fit_fun = CostFunction(
        inputs,
        pool=pool,
        display_shock_provider=cb.display_shock_provider,
        log_callback=log,
        progress_callback=progress,
    )

    if pool is not None:
        fit_fun.warmup_workers(inputs.max_processors, inputs.initial_scalers)

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
