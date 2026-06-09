"""Qt adapter around :func:`frhodo.api._run_optimization_engine`.

``Worker(QRunnable)`` runs the optimization in a Qt thread pool;
``WorkerSignals`` carries log/progress/result back to the GUI. All
inputs are snapshotted into the worker at construction — the worker
holds no reference to the GUI ``Main`` window during the run.
"""
import sys
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from qtpy.QtCore import QObject, QRunnable, Signal, Slot

from frhodo.api import OptimizationCallbacks, _run_optimization_engine
from frhodo.optimize.residual import OptimizeRunInputs
from frhodo.simulation.mechanism.coef_helpers import rates


@dataclass(frozen=True)
class WorkerInputs:
    """Frozen snapshot of every parameter the optimizer needs.

    Captured at :class:`Worker` construction so the run is decoupled
    from any further GUI state mutation. Holding a frozen dataclass
    rather than a live ``Main`` reference is what lets the worker run
    safely off the GUI thread without locks.
    """

    mech: Any
    shocks2run: list
    coef_opt: list
    rxn_coef_opt: list
    rxn_rate_opt: dict
    optimizable_set: Any
    cost_settings: Any
    opt_settings: dict
    reactor_state: Any
    time_unc_value: float
    time_unc_random: bool
    max_processors: int
    multiprocessing: bool
    dist: Any
    display_shock_provider: Callable[[], object]
    worker_pool: Any = None  # PersistentWorkerPool


class Worker(QRunnable):
    """``QRunnable`` that drives the optimizer off the GUI thread.

    Emits :class:`frhodo.api.OptimizationResult` via
    ``signals.result``; per-iteration raw progress dicts via
    ``signals.update``; human-readable log lines via ``signals.log``.
    """

    def __init__(self, inputs: WorkerInputs):
        super().__init__()
        self.signals = WorkerSignals()
        self.inputs = inputs
        self._abort = False

        lb, ub = inputs.rxn_rate_opt["bnds"]["lower"], inputs.rxn_rate_opt["bnds"]["upper"]
        initial_scalers = rates(inputs.rxn_coef_opt, inputs.mech) - inputs.rxn_rate_opt["x0"]
        self.initial_scalers = np.clip(
            initial_scalers, lb * (1 + 1e-9), ub * (1 - 1e-9),
        )

    def optimize_coeffs(self):
        inputs = self.inputs
        callbacks = OptimizationCallbacks(
            on_progress=self.signals.update.emit,
            log=self.signals.log.emit,
            abort=lambda: self._abort,
            display_shock_provider=inputs.display_shock_provider,
            worker_pool=inputs.worker_pool,
        )
        engine_inputs = OptimizeRunInputs(
            mech=inputs.mech,
            shocks2run=inputs.shocks2run,
            coef_opt=inputs.coef_opt,
            rxn_coef_opt=inputs.rxn_coef_opt,
            rxn_rate_opt=inputs.rxn_rate_opt,
            initial_scalers=self.initial_scalers,
            reactor_state=inputs.reactor_state,
            time_unc=inputs.time_unc_value,
            cost_settings=inputs.cost_settings,
            opt_settings_optimize=inputs.opt_settings,
            dist=inputs.dist,
            multiprocessing=inputs.multiprocessing,
            max_processors=inputs.max_processors,
            random_t_uncertainty=inputs.time_unc_random,
        )

        return _run_optimization_engine(
            engine_inputs,
            optimizable_set=inputs.optimizable_set,
            callbacks=callbacks,
        )

    @Slot()
    def run(self):
        try:
            result = self.optimize_coeffs()
        except Exception:
            tb_text = traceback.format_exc()
            traceback.print_exc()
            self.signals.error.emit((sys.exc_info()[:2], tb_text))
            return

        self.signals.result.emit(result)

    def abort(self):
        self._abort = True


class WorkerSignals(QObject):
    finished = Signal()
    error = Signal(tuple)
    result = Signal(object)
    update = Signal(object)
    progress = Signal(int, str)
    log = Signal(str)
    abort = Signal()
