"""GUI runner for pre-fit reaction screening.

Gathers the shocks an optimization would run, executes
:func:`frhodo.optimize.screening.screen_campaign` on a worker thread
under the mechanism's exclusive lock, and formats the ranking table for
the log.
"""
import traceback

import numpy as np
from qtpy.QtCore import QObject, QRunnable, Signal

from frhodo.optimize._worker_context import MechBuildPayload
from frhodo.optimize.cost.settings import CostSettings
from frhodo.optimize.screening import SUGGEST_LOSS_FRACTION, screen_campaign



class ScreeningSignals(QObject):
    done = Signal(object)
    error = Signal(str)


class ScreeningRunnable(QRunnable):
    """Runs the screening solves off the GUI thread.

    Holds ``mech.exclusive()`` for the duration so concurrent GUI
    simulations don't race the screening reactor solves. When the
    persistent worker pool is already running, the per-shock solves are
    farmed across it (acquiring re-initializes the workers to the
    current mechanism if the user edited coefficients since spawn); a
    pool that isn't up yet stays down — screening never triggers the
    worker-fleet launch on its own.
    """

    def __init__(self, mech, shocks, reactor_state, cost_settings,
                 worker_pool=None, workers=0):
        super().__init__()
        self.signals = ScreeningSignals()
        self._mech = mech
        self._shocks = shocks
        self._reactor_state = reactor_state
        self._cost_settings = cost_settings
        self._worker_pool = worker_pool
        self._workers = workers

    def _acquire_pool(self):
        if (self._worker_pool is None or self._workers < 2
                or not self._worker_pool.running):
            return None
        payload = MechBuildPayload(
            reset_mech=self._mech.reset_mech,
            thermo_coeffs=self._mech.thermo_coeffs,
            coeffs=self._mech.coeffs,
            coeffs_bnds=self._mech.coeffs_bnds,
            rate_bnds=self._mech.rate_bnds,
        )
        try:
            pool = self._worker_pool.acquire(
                workers=self._workers, payload=payload,
            )
        except Exception:
            return None

        return pool

    def run(self):
        try:
            with self._mech.exclusive():
                result = screen_campaign(
                    self._mech, self._shocks, self._reactor_state,
                    self._cost_settings, worker_pool=self._acquire_pool(),
                )
        except Exception:
            self.signals.error.emit(traceback.format_exc())

            return

        self.signals.done.emit(result)


def gather_screening_shocks(parent) -> list:
    """The shocks an optimization run would use, weights prepared."""
    shocks = []
    for series in parent.series.shock:
        for shock in series:
            if not shock.include or "exp_data" in shock.err:
                continue
            shocks.append(shock)
    if not shocks:
        shocks = [parent.display_shock]

    prepared = []
    for shock in shocks:
        if shock.exp_data.size == 0:
            continue
        weight_var = [
            shock.weight_max, shock.weight_min,
            shock.weight_shift, shock.weight_k,
        ]
        if np.isnan(np.hstack(weight_var)).any():
            parent.weight.update(shock=shock)
        shock.weights = parent.series.weights(shock.exp_data[:, 0], shock)
        shock.opt_time_offset = shock.time_offset
        prepared.append(shock)

    return prepared


def cost_settings_from_gui(parent) -> CostSettings:
    opt_settings = parent.optimization_settings
    settings = CostSettings(
        scale=opt_settings.get("obj_fcn", "scale"),
        bisymlog_scaling_factor=parent.plot.signal.bisymlog.scaling_factor,
        loss_alpha=opt_settings.get("obj_fcn", "alpha"),
        loss_c=opt_settings.get("obj_fcn", "c"),
        experiment_weighting=opt_settings.get("obj_fcn", "experiment_weighting"),
    )

    return settings


def format_screening_summary(result) -> str:
    """One log line: what the tree's ranking cannot show."""
    n_suggested = int(np.sum(result.suggested))
    if result.spectral_gap_rank is not None:
        kink = f"kink after rank {result.spectral_gap_rank}"
    else:
        kink = "no clear kink"
    summary = (
        f"Screening: {n_suggested} suggested reaction(s), "
        f"effective rank {result.effective_rank:.1f}, {kink}, "
        f"start loss {result.loss_start:.3e}"
    )
    if result.skipped_shocks:
        summary += f"; skipped shocks {result.skipped_shocks}"

    return summary
