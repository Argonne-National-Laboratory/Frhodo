"""GUI controller for the optimization run lifecycle.

Owns input validation, per-iteration HoF + plot cadence, and run
finalization. Drives the Qt :class:`Worker` off the GUI thread and
receives engine events through :class:`frhodo.api.OptimizationCallbacks`.
"""
import multiprocessing as mp
import traceback
from timeit import default_timer as timer
from typing import Any

import numpy as np
from qtpy.QtCore import Qt, QObject, QRunnable, Signal
from scipy import stats

from frhodo.gui.workers.optimize_worker import Worker, WorkerInputs
from frhodo.optimize.cost.fit_fcn import update_mech_coef_opt
from frhodo.optimize.cost.settings import CostSettings
from frhodo.optimize.parameters import build_rxn_coef_opt, build_rxn_rate_opt


class _RecastSignals(QObject):
    done = Signal(bool, bool)
    error = Signal(str)


class _RecastRunnable(QRunnable):
    """Runs :meth:`ChemicalMechanism.recast_to_troe` off the GUI thread.

    Holds ``mech``'s exclusive lock for the duration so concurrent GUI
    edits don't race with the recast's mutations to ``coeffs`` /
    ``coeffs_bnds`` / ``reset_mech``. Emits ``done(rxns_changed,
    mech_rebuilt)`` on completion; the orchestrator catches it on the
    GUI thread to refresh the tree and start the optimizer worker.
    """

    def __init__(self, mech, rxn_coef_opt, rxn_rate_opt, optimizable_set):
        super().__init__()
        self.signals = _RecastSignals()
        self._mech = mech
        self._rxn_coef_opt = rxn_coef_opt
        self._rxn_rate_opt = rxn_rate_opt
        self._optimizable_set = optimizable_set

    def run(self):
        try:
            with self._mech.exclusive():
                rxns_changed, mech_rebuilt = self._mech.recast_to_troe(
                    self._rxn_coef_opt,
                    self._rxn_rate_opt,
                    self._optimizable_set,
                )
        except Exception:
            self.signals.error.emit(traceback.format_exc())
            return

        self.signals.done.emit(bool(rxns_changed), bool(mech_rebuilt))


class Multithread_Optimize:
    """Qt-thread controller for optimization runs.

    Builds the engine inputs from GUI state, owns the per-run
    hall-of-fame and plot-cadence state, and finalizes the run when
    the worker completes.
    """

    dist = stats.gennorm

    def __init__(self, parent):
        self.parent = parent
        parent.run_control.optimize_running = False
        parent.run_control.multiprocessing = True

        self.HoF: Any = []
        self.abort = False
        self._last_plot_timer = 0.0
        self._time_between_plots = 0.0
        self.coef_opt: list = []
        self.rxn_coef_opt: list = []
        self.rxn_rate_opt: dict = {}
        self.shocks2run: list = []
        self.optimizable: Any = None
        self._saved_auto_fit: bool | None = None
        self._max_processors_pending: int = 1
        self._recast_cancelled: bool = False
        self._iter_width: int = 5

        parent.action_Run.triggered.connect(self.start_threads)
        parent.action_Abort.triggered.connect(self.abort_workers)

    def start_threads(self) -> None:
        parent = self.parent
        parent.run_control.multiprocessing = parent.multiprocessing_box.isChecked()

        prepared = self._prepare_run()
        if prepared is None:
            return
        shocks2run, max_processors = prepared

        parent.run_control.max_processors = max_processors
        parent.run_control.abort = self.abort
        self.shocks2run = shocks2run
        self._max_processors_pending = max_processors
        self._recast_cancelled = False

        if self._has_recast_work():
            parent.log.append(
                "Recasting pressure-dependent reactions to Troe form…",
                alert=False,
            )
            recast = _RecastRunnable(
                parent.mech,
                self.rxn_coef_opt,
                self.rxn_rate_opt,
                parent.optimizables,
            )
            recast.signals.done.connect(self._after_recast, Qt.QueuedConnection)
            recast.signals.error.connect(self._on_recast_error, Qt.QueuedConnection)
            parent.threadpool.start(recast)

            return

        self._after_recast(False, False)

    def _has_recast_work(self) -> bool:
        """True if any optimizable rxn still needs a Troe recast."""
        import cantera as ct

        mech = self.parent.mech
        for rxn_coef in self.rxn_coef_opt:
            rate = mech.gas.reaction(rxn_coef["rxnIdx"]).rate
            if not isinstance(rate, (ct.ArrheniusRate, ct.TroeRate)):
                return True

        return False

    def _on_recast_error(self, tb: str) -> None:
        self.parent.log.append(f"Recast failed:\n{tb}", alert=True)
        self.parent.run_control.optimize_running = False
        self._restore_auto_fit()

    def _after_recast(self, rxns_changed: bool, mech_rebuilt: bool) -> None:
        """GUI-thread continuation after the background recast finishes.

        Refreshes the tree if the mech structurally changed, rebuilds
        ``rxn_coef_opt`` / ``rxn_rate_opt`` against the post-recast mech,
        and starts the optimizer :class:`Worker`. A no-op if the user
        aborted while the recast was in flight.
        """
        parent = self.parent
        if self._recast_cancelled:
            parent.run_control.optimize_running = False
            self._restore_auto_fit()

            return

        if mech_rebuilt:
            parent.save.chemkin_format(
                parent.mech.gas,
                parent.path_set.optimized_mech(file_out="recast_mech"),
            )
            parent.tree.set_trees(parent.mech)

        if rxns_changed:
            self._initialize_opt(parent.mech)

        if parent.mech.recast_log_rms:
            for rxnIdx in sorted(parent.mech.recast_log_rms):
                log_rms = parent.mech.recast_log_rms[rxnIdx]
                parent.log.append(
                    f"R{rxnIdx + 1} recast to Troe: fit log-RMS = {log_rms:.4f}",
                    alert=False,
                )

        max_processors = self._max_processors_pending
        self._iter_width = self._compute_iter_width()
        parent.log.append(
            f"Initializing {max_processors:d} worker processes…",
            alert=False,
        )
        parent.plot.opt.clear_plot()

        self.worker = Worker(
            self._build_worker_inputs(self.shocks2run, max_processors),
        )
        self.worker.signals.result.connect(self.on_worker_done)
        self.worker.signals.finished.connect(self._thread_complete)
        self.worker.signals.update.connect(self._on_iteration_safe)
        self.worker.signals.progress.connect(self._on_worker_progress)
        self.worker.signals.log.connect(parent.log.append)
        self.worker.signals.error.connect(self._on_worker_error, Qt.QueuedConnection)
        self.worker.signals.abort.connect(self.worker.abort)

        if not parent.run_control.abort:
            parent.threadpool.start(self.worker)

    def _prepare_run(self) -> tuple[list, int] | None:
        """Validate GUI state and prep ``coef_opt`` / ``rxn_coef_opt`` /
        ``rxn_rate_opt`` for the worker.

        Returns ``(shocks2run, max_processors)`` when ready, else
        ``None`` after logging the rejection.
        """
        parent = self.parent
        parent.path_set.optimized_mech()
        self._last_plot_timer = 0.0
        self._time_between_plots = 0.0

        if parent.directory.invalid:
            parent.log.append("Invalid directory found\n")

            return None
        if parent.run_control.optimize_running:
            parent.log.append("Optimize running flag already set to True\n")

            return None
        if self.parent.optimizables.build(parent.mech).is_empty():
            parent.log.append("No reactions or coefficients set to be optimized\n")

            return None

        shocks2run: list = []
        for series in parent.series.shock:
            for shock in series:
                if not shock.include or "exp_data" in shock.err:
                    shock.SIM = None
                    continue
                shocks2run.append(shock)

        if len(shocks2run) == 0:
            shocks2run = [parent.display_shock]
        else:
            if not parent.load_full_series_box.isChecked():
                parent.log.append(
                    '"Load Full Series Into Memory" must be checked for '
                    "optimization of multiple experiments\n"
                )

                return None
            if len(parent.series_viewer.data_table) == 0:
                parent.log.append("Set Series in Series Viewer and select experiments\n")

                return None

        if len(shocks2run) == 0:
            return None

        for shock in shocks2run:
            shock.opt_time_offset = shock.time_offset
            shock.last_t_unc = None
            shock.last_loss_alpha = None

            weight_var = [
                shock.weight_max, shock.weight_min,
                shock.weight_shift, shock.weight_k,
            ]
            if np.isnan(np.hstack(weight_var)).any():
                parent.weight.update(shock=shock)
            shock.weights = parent.series.weights(
                shock.exp_data[:, 0], shock,
            )

            if np.isnan([shock.T_reactor, shock.P_reactor]).any():
                parent.series.set("zone", shock.zone)

            parent.series.uncertainties(shock)
            parent.series.rate_bnds(shock)

        self.shocks2run = shocks2run
        self._initialize_opt(parent.mech)

        parent.update_user_settings()
        self.abort = False
        self._saved_auto_fit = bool(parent.time_uncertainty.auto_fit)
        # Skip the auto-fit takeover when there's no t_unc bound — the cost
        # function won't search a per-shock offset, so forcing the GUI box
        # to track shock.opt_time_offset would just freeze it at the
        # starting value and look broken.
        if parent.time_uncertainty.value > 0:
            parent.time_uncertainty.auto_fit = True
        parent.run_control.optimize_running = True

        if parent.run_control.multiprocessing:
            cpu_count = mp.cpu_count() + 2
            max_processors = int(min(len(shocks2run), cpu_count))
        else:
            max_processors = 1

        self.HoF = []

        return shocks2run, max_processors

    def _initialize_opt(self, mech) -> None:
        self.optimizable = self.parent.optimizables.build(mech)
        self.coef_opt = list(self.optimizable.coefficients)
        self.rxn_coef_opt = build_rxn_coef_opt(mech, self.coef_opt, self.shocks2run)
        self.rxn_rate_opt = build_rxn_rate_opt(mech, self.rxn_coef_opt)

    def _build_worker_inputs(self, shocks2run: list, max_processors: int) -> WorkerInputs:
        p = self.parent
        opt_settings = p.optimization_settings
        cost_settings = CostSettings(
            obj_fcn_type=opt_settings.get("obj_fcn", "type"),
            scale=opt_settings.get("obj_fcn", "scale"),
            bisymlog_scaling_factor=p.plot.signal.bisymlog.scaling_factor,
            loss_alpha=opt_settings.get("obj_fcn", "alpha"),
            loss_c=opt_settings.get("obj_fcn", "c"),
            bayes_dist_type=opt_settings.get("obj_fcn", "bayes_dist_type"),
            bayes_unc_sigma=opt_settings.get("obj_fcn", "bayes_unc_sigma"),
        )

        return WorkerInputs(
            mech=p.mech,
            shocks2run=shocks2run,
            coef_opt=self.coef_opt,
            rxn_coef_opt=self.rxn_coef_opt,
            rxn_rate_opt=self.rxn_rate_opt,
            optimizable_set=self.optimizable,
            cost_settings=cost_settings,
            opt_settings=opt_settings.settings,
            reactor_state=p.reactor_state,
            time_unc_value=p.time_uncertainty.value,
            time_unc_random=p.time_uncertainty.random,
            max_processors=max_processors,
            multiprocessing=p.run_control.multiprocessing,
            dist=self.dist,
            display_shock_provider=lambda: p.display_shock,
            worker_pool=getattr(p, "worker_pool", None),
        )

    def _on_worker_error(self, payload) -> None:
        """Surface worker-thread tracebacks to the log tab.

        ``Worker.run`` emits ``(exc_info_pair, tb_text)``; without this
        slot the traceback would only go to stderr (invisible in
        production builds) and the user would just see the error message
        with no file/line context.
        """
        tb_text = payload[1] if isinstance(payload, tuple) and len(payload) >= 2 else str(payload)
        self.parent.log.append(f"Optimization worker failed:\n{tb_text}", alert=True)

    def _on_iteration_safe(self, result: dict) -> None:
        """Run ``_on_iteration`` with a traceback-capturing guard.

        ``_on_iteration`` mutates the GUI (tree, plot, time-offset box)
        and Qt swallows exceptions thrown from signal slots — they'd
        only appear on stderr, which is invisible in the bundled app.
        Catch + log so the user can actually see what broke.
        """
        try:
            self._on_iteration(result)
        except Exception:
            tb_text = traceback.format_exc()
            self.parent.log.append(
                f"Iteration callback failed:\n{tb_text}", alert=True,
            )

    def _on_iteration(self, result: dict, write_log: bool = True) -> None:
        """Per-iteration update: HoF, log line, tree refresh, plot cadence."""
        parent = self.parent
        if not self.HoF:
            self.HoF = result
            # Workers spawned and the first iteration just landed — log
            # "Optimization starting" now so it appears after the pool
            # init message instead of racing it on the worker thread.
            parent.log.append(
                "\nOptimization starting\n\n"
                "   Iteration\t\t Objective Func\tBest Objetive Func",
                alert=False,
            )
        elif result["obj_fcn"] < self.HoF["obj_fcn"]:
            self.HoF = result

        obj_fcn_str = f"{result['obj_fcn']:.3e}"
        for old, new in (("e+", "e"), ("e0", "e"), ("e-0", "e-")):
            obj_fcn_str = obj_fcn_str.replace(old, new)
        result["obj_fcn_str"] = obj_fcn_str

        if write_log:
            i = result["i"]
            opt_type = result["type"][0].upper()
            width = self._iter_width
            log_str = (
                f"\t{opt_type.upper()} {i:>{width}d}"
                f"\t\t{obj_fcn_str:>10s}"
                f"\t\t{self.HoF['obj_fcn_str']:>10s}"
            )
            ode_err = result.get("ode_error")
            if ode_err:
                log_str += f"  [{ode_err}]"
            parent.log.append(log_str, alert=False)

        parent.tree.update_coef_rate_from_opt(self.coef_opt, result["x"])

        if timer() - self._last_plot_timer > self._time_between_plots:
            plot_start_time = timer()
            ind_var = result["ind_var"]
            observable = result["observable"]
            if ind_var is None and observable is None:
                parent.run_single()
            else:
                parent.plot.signal.update_sim(ind_var[:, 0], observable[:, 0])
            parent.plot.opt.update(result["stat_plot"])

            if result.get("display_t_offset") is not None:
                self._sync_display_t_offset_box(result["display_t_offset"])

            current_time_to_plot = timer() - plot_start_time
            if current_time_to_plot * 0.1 > self._time_between_plots:
                self._time_between_plots = current_time_to_plot * 0.1

            self._last_plot_timer = timer()

    def _on_worker_progress(self, perc_completed, time_left) -> None:
        self.parent.update_progress(perc_completed, time_left)

    def _thread_complete(self) -> None:
        pass

    def on_worker_done(self, result) -> None:
        """Finalize a run: apply best coefs, save mech, refresh UI, alert."""
        raw = result.raw
        parent = self.parent
        parent.run_control.optimize_running = False
        self._commit_final_time_offsets()
        self._restore_auto_fit()

        if raw is None or len(raw) == 0:
            if self.HoF and self.coef_opt:
                update_mech_coef_opt(parent.mech, self.coef_opt, self.HoF["x"])
                parent.tree.update_coef_rate_from_opt(self.coef_opt, self.HoF["x"])
                parent.run_single()
            return

        if "local" in raw:
            update_mech_coef_opt(parent.mech, self.coef_opt, raw["local"]["x"])
        else:
            update_mech_coef_opt(parent.mech, self.coef_opt, raw["global"]["x"])

        for opt_type, res in raw.items():
            total_shock_eval = (res["nfev"] + 1) * len(self.shocks2run)
            message = res["message"][:1].lower() + res["message"][1:]
            parent.log.append(f"\n{opt_type.capitalize()} {message}")
            parent.log.append(f"\telapsed time:\t{res['time']:.2f}", alert=False)
            parent.log.append(f"\tAvg Std Residual:\t{res['fval']:.3e}", alert=False)
            parent.log.append(f"\topt iters:\t\t{res['nfev'] + 1:.0f}", alert=False)
            parent.log.append(f"\tshock evals:\t{total_shock_eval:.0f}", alert=False)
            parent.log.append(f"\tsuccess:\t\t{res['success']}", alert=False)

        parent.log.append("\n", alert=False)
        parent.save.chemkin_format(parent.mech.gas, parent.path_set.optimized_mech())
        parent.path_set.mech()
        parent.tree._copy_expanded_tab_rates()
        parent.app.alert(parent, 5 * 1000)

    def abort_workers(self) -> None:
        if hasattr(self, "worker"):
            self.worker.signals.abort.emit()
            self.parent.run_control.abort = True
        else:
            # Abort hit during the background recast — the recast itself
            # can't be interrupted, but flag the continuation slot so it
            # skips Worker startup when the recast eventually returns.
            self._recast_cancelled = True

        self.abort = True
        self.parent.run_control.optimize_running = False
        self._restore_auto_fit()

    def _compute_iter_width(self) -> int:
        """Width to pad the iteration counter so the log column doesn't drift.

        Reads the ``stop_criteria_val`` from the active optimizer config —
        when the user picked an iteration cap, that's the largest value
        ``result["i"]`` will ever reach in this run. Falls back to a
        conservative width if the cap is time-based or missing.
        """
        settings = self.parent.optimization_settings.settings
        caps: list[int] = []
        for opt_type in ("global", "local"):
            cfg = settings.get(opt_type, {})
            if cfg.get("stop_criteria_type") == "Iteration Maximum":
                try:
                    caps.append(int(cfg.get("stop_criteria_val", 0)))
                except (TypeError, ValueError):
                    continue
        if not caps:
            return 5

        return max(2, len(str(max(caps))))

    def _restore_auto_fit(self) -> None:
        if self._saved_auto_fit is None:
            return
        self.parent.time_uncertainty.auto_fit = self._saved_auto_fit
        self._saved_auto_fit = None

    def _sync_display_t_offset_box(self, t_offset: float) -> None:
        """Write the optimizer's current offset for the displayed shock to the box.

        Called from the live-plot cadence, so the spinbox tracks the optimizer
        value as it explores; the cadence coalesces bursts, showing the most
        recent value and dropping the rest. ``shock.time_offset`` is updated
        too so the post-sim consistency check in :meth:`update_user_settings`
        stays stable.
        """
        parent = self.parent
        shock = getattr(parent, "display_shock", None)
        if shock is None:
            return
        t_unit_conv = parent.reactor_state.t_unit_conv or 1.0
        box_value = t_offset / t_unit_conv
        for box in parent.time_offset_box.twin:
            box.blockSignals(True)
            box.setValue(box_value)
            box.blockSignals(False)
        synced = parent.time_offset_box.value() * t_unit_conv
        shock.time_offset = synced
        parent.time_uncertainty.offset = synced

    def _commit_final_time_offsets(self) -> None:
        """Persist the optimizer's final per-shock offsets onto each shock.

        After the optimizer exits, each shock's effective time offset is
        ``opt_time_offset + last_t_unc``. Writing it back to
        ``shock.time_offset`` makes the optimizer's result the new baseline
        the GUI displays and re-fits from.
        """
        for shock in self.shocks2run:
            base = float(getattr(shock, "opt_time_offset", shock.time_offset))
            last = getattr(shock, "last_t_unc", None)
            shock.time_offset = base + (float(last) if last is not None else 0.0)
