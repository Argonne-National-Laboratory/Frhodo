#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This file is part of Frhodo. Copyright © 2020, UChicago Argonne, LLC
# and licensed under BSD-3-Clause. See License.txt in the top-level
# directory for license and copyright information.

import os
import sys
import threading
import platform
import multiprocessing
import pathlib
import ctypes
import re
import signal

# Under WSLg, point Qt at the runtime dir that holds the wayland-0
# socket; the default /run/user/$UID lacks it. Must run before qtpy import.
if "microsoft" in platform.uname().release.lower() and os.path.isdir("/mnt/wslg/runtime-dir"):
    os.environ.setdefault("QT_QPA_PLATFORM", "wayland")
    os.environ["XDG_RUNTIME_DIR"] = "/mnt/wslg/runtime-dir"

from qtpy.QtWidgets import QMainWindow, QApplication, QMessageBox, QFileDialog
from qtpy import uic, QtCore, QtGui

import numpy as np

from frhodo.gui.plots.plot_main import All_Plots as plot
from frhodo.gui.widgets.misc_widget import MessageWindow
from frhodo.simulation.mechanism import mech_fcns

from frhodo.common import units as convert_units
from frhodo import __version__
from frhodo.common.errors import MechanismLoadError
from frhodo.simulation.mechanism.mechanism_loader import (
    FORMAT_TABLE,
    MechanismLoader,
)
from frhodo.simulation.shock.incident_shock_reactor import run_incident_shock
from frhodo.simulation.shock.reactor_output import ReactorOutput
from frhodo.simulation.shock.zero_d_reactor import run_zero_d
from frhodo.simulation.shock.state import (
    RuntimeReactorState,
    zero_d_mode_from_label as _zero_d_mode_from_label,
)
import platformdirs

from frhodo.gui.runtime_paths import RuntimePaths
from frhodo.gui import session
from frhodo.gui.state import (
    LoadState, RunControlState, ShockSelectionState, TimeUncertaintyState,
)
from frhodo.gui.widgets.options_panel_widgets import apply_auto_fit_time_offset
from frhodo.optimize.parameters import OptimizableSetBuilder
from frhodo.optimize.pool import PersistentWorkerPool, default_worker_count
from frhodo.optimize._worker_context import MechBuildPayload
from frhodo.gui.widgets import (
    config_io,
    error_window,
    help_menu,
    options_panel_widgets,
    program_settings,
    save_widget,
    settings,
    sim_explorer_widget,
)



if (
    os.environ["QT_API"] == "pyside2"
):  # Silence warning: "Qt WebEngine seems to be initialized from a plugin."
    QApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)

# Handle high resolution displays:  Minimum recommended resolution 1280 x 960
if hasattr(QtCore.Qt, "AA_EnableHighDpiScaling"):
    QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
if hasattr(QtCore.Qt, "AA_UseHighDpiPixmaps"):
    QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

_THIS_FILE = pathlib.Path(__file__).resolve()
path = {
    "package": _THIS_FILE.parent / "gui",  # frhodo/gui/  (ui/ lives here)
    "main": _THIS_FILE.parent,             # frhodo/      (package root)
}

path["appdata"] = pathlib.Path(
    platformdirs.user_config_dir(appname="Frhodo", appauthor=False, roaming=True)
)
path["appdata"].mkdir(parents=True, exist_ok=True)

_startup_failed = False


def _mark_startup_failed():
    global _startup_failed
    _startup_failed = True


# The Chemkin THERMO keyword opens a block and so must start its own line;
# "THER" is the accepted abbreviation.
_CHEMKIN_THERMO_KEYWORD = re.compile(r"\s*ther(mo?)?\b", re.IGNORECASE)


def _chemkin_embeds_thermo(mech_path):
    """True when a Chemkin mech file carries its own ``THERMO`` block."""
    with open(mech_path, "r", errors="replace") as f:
        for line in f:
            if _CHEMKIN_THERMO_KEYWORD.match(line.split("!")[0]):
                return True

    return False


class Main(QMainWindow):
    def closeEvent(self, event):
        """Cooperative shutdown on user-driven window close.

        Signals abort to long-running workers, drains the threadpool
        within a grace period. Falls back to ``os._exit`` only if a
        worker still hasn't released after the timeout.
        """
        if hasattr(self, "user_settings"):
            try:
                self.user_settings.save()  # persist window/panel geometry
            except Exception:
                pass  # a save failure must never block shutdown

        if hasattr(self, "optimize"):
            self.optimize.abort_workers()

        if hasattr(self, "worker_pool"):
            self.worker_pool.close()

        drained = True
        if hasattr(self, "threadpool"):
            drained = self.threadpool.waitForDone(5_000)

        event.accept()
        self.app.quit()
        if not drained:
            os._exit(0)

    def __init__(self, app, path):
        super().__init__()
        self.app = app
        self.runtime_paths = RuntimePaths.from_package(
            package=path["package"], appdata=path["appdata"],
        )
        self.path_set = settings.Path(self, path)
        uic.loadUi(str(self.runtime_paths.package / "ui" / "main_window.ui"), self)  # ~0.4 sec
        self.setWindowIcon(
            QtGui.QIcon(str(self.runtime_paths.graphics / "main_icon.png"))
        )

        # Start threadpools
        self.threadpool = QtCore.QThreadPool()
        self.threadpool.setMaxThreadCount(
            2
        )  # 1 for GUI (implicit) + 1 for calc

        # Set selected tabs
        for tab_widget in [self.option_tab_widget, self.plot_tab_widget]:
            tab_widget.setCurrentIndex(0)

        # Set Clipboard
        self.clipboard = QApplication.clipboard()

        self.shock_selection = ShockSelectionState()
        self.time_uncertainty = TimeUncertaintyState()
        self.load_state = LoadState()
        self.run_control = RunControlState()
        self.optimizables = OptimizableSetBuilder()
        self.reactor_state = RuntimeReactorState(t_unit_conv=1)
        self.SIM = ReactorOutput()
        self.mech = mech_fcns.ChemicalMechanism()
        self.convert_units = convert_units.Convert_Units(self.mech)
        self.worker_pool = PersistentWorkerPool()
        self.series = settings.series(self)

        self.sim_explorer = sim_explorer_widget.SIM_Explorer_Widgets(self)
        self.plot = plot(self)
        options_panel_widgets.Initialize(self)

        # Setup save sim
        self.save_sim = save_widget.Save_Dialog(self)
        self.save_sim_button.clicked.connect(self.save_sim.execute)
        self.action_Save.triggered.connect(self.save_sim.execute)
        self.action_Open.triggered.connect(self.load_session)

        self.action_Settings.triggered.connect(self._open_program_settings)

        if _startup_failed:
            sys.exit()

        self.show()
        self.app.processEvents()  # allow everything to draw properly

        # Initialize Settings
        self.initialize_settings()  # ~ 4 sec

        # Setup help menu
        self.version = __version__
        help_menu.HelpMenu(self)

    def initialize_settings(self):
        msgBox = MessageWindow(self, "Loading...")
        self.app.processEvents()

        self.shock_selection = ShockSelectionState(current=1, previous=1)

        self.user_settings = config_io.GUI_settings(self)
        self.user_settings.load()
        self._size_directory_boxes()
        self._restore_window_layout()

        self.load_state.load_full_series = self.load_full_series_box.isChecked()

        # load previous paths if file in path, can be accessed, and is a file
        if (
            "path_file" in self.path
            and os.access(self.path["path_file"], os.R_OK)
            and self.path["path_file"].is_file()
        ):
            self.path_set.load_dir_file(self.path["path_file"])  # ~3.9 sec

        self.update_user_settings()
        # Flush queued signals from the loads above so any downstream
        # run_single calls are absorbed by ``run_block`` before the gate
        # is released and the single startup sim is dispatched.
        self.app.processEvents()
        self.run_control.run_block = False
        self.run_single()
        msgBox.close()

    def _size_directory_boxes(self):
        """Fix each directory text box to a whole number of text lines,
        from live font metrics so display scaling is honored."""
        boxes = [
            (self.exp_main_box, 4),
            (self.mech_main_box, 4),
            (self.sim_main_box, 4),
            (self.path_file_box, 5),
        ]
        for box, n_lines in boxes:
            chrome = 2 * (box.frameWidth() + int(box.document().documentMargin()))
            box.setFixedHeight(n_lines * box.fontMetrics().lineSpacing()
                               + chrome + 2)

    def _restore_window_layout(self):
        """Apply the saved window size and options-panel width.

        On first run (no saved width) the panel is sized to the tab
        bar's preferred width at the current DPI so all tabs show
        without scroll buttons; the minimum width is the lower bound.
        """
        win_cfg = self.user_settings.config.window

        if win_cfg.maximized:
            self.showMaximized()
        elif win_cfg.width and win_cfg.height:
            self.resize(win_cfg.width, win_cfg.height)

        self.app.processEvents()

        floor = self.option_tab_widget.minimumWidth()
        if win_cfg.option_panel_width:
            panel_width = max(win_cfg.option_panel_width, floor)
        else:
            tab_bar = self.option_tab_widget.tabBar()
            panel_width = max(tab_bar.sizeHint().width() + 24, floor)

        total = self.splitter.width()
        self.splitter.setSizes([panel_width, max(1, total - panel_width)])

    def _configure_aux_control(self, checkbox, combobox, files_present, has_embedded):
        """Set an auxiliary-file checkbox and its combobox to match the mech source.

        ``has_embedded`` means the mechanism already carries the data, so the
        separate file is optional: the checkbox only offers it when
        ``files_present``, and an opt-in the user could make survives the reload
        their own click triggers. Without it the checkbox is forced on and
        locked so the user has to supply a file; that forced state is not a
        choice, so it is dropped as soon as a mech carrying the data is
        selected. Signals are blocked around the programmatic state change to
        stop it triggering a second load.
        """
        checkbox.blockSignals(True)
        if has_embedded:
            opted_in = checkbox.isEnabled() and checkbox.isChecked()
            checkbox.setEnabled(files_present)
            checkbox.setChecked(opted_in and files_present)
        else:
            checkbox.setChecked(True)
            checkbox.setDisabled(True)
        checkbox.blockSignals(False)

        combobox.setEnabled(checkbox.isChecked())

    def load_mech(self, event=None):
        if self.mech_select_comboBox.count() == 0:
            return  # if no items return, unsure if this is needed now

        # Specify mech file path
        self.path["mech"] = self.path["mech_main"] / str(
            self.mech_select_comboBox.currentText()
        )
        if not self.path["mech"].is_file():  # if it's not a file, then it was deleted
            self.path_set.mech()  # update mech pulldown choices

            return

        # Separate thermo/transport files only reach Cantera through the Chemkin
        # converter; every other format carries that data inside the mech file.
        is_chemkin = FORMAT_TABLE.get(self.path["mech"].suffix.lower()) == "chemkin"
        thermo_files_present = is_chemkin and self.thermo_select_comboBox.count() > 0
        mech_embeds_thermo = not is_chemkin or _chemkin_embeds_thermo(self.path["mech"])

        self._configure_aux_control(
            self.use_thermo_file_box,
            self.thermo_select_comboBox,
            thermo_files_present,
            mech_embeds_thermo,
        )

        # A missing transport block is not an error for the Chemkin converter, so a
        # separate transport file is always optional and stays opt-in across loads.
        transport_files_present = (
            is_chemkin and self.transport_select_comboBox.count() > 0
        )
        self._configure_aux_control(
            self.use_transport_file_box,
            self.transport_select_comboBox,
            transport_files_present,
            True,
        )

        # Specify transport file path
        if self.use_transport_file_box.isChecked():
            self.path["transport"] = self.path["mech_main"] / str(
                self.transport_select_comboBox.currentText()
            )
        else:
            self.path["transport"] = None

        # Specify thermo file path
        if self.use_thermo_file_box.isChecked():
            if self.thermo_select_comboBox.count() > 0:
                self.path["thermo"] = self.path["mech_main"] / str(
                    self.thermo_select_comboBox.currentText()
                )
            else:
                self.log.append(
                    "Error loading mech:\nNo thermodynamics given", alert=True
                )

                return
        else:
            self.path["thermo"] = None

        # Initialize Mechanism
        self.log.clear([])
        prior_snapshot = self.tree.snapshot_for_reload()
        loader = MechanismLoader()
        try:
            loader.load(self.path, mech=self.mech)
        except MechanismLoadError as e:
            self.log.append(str(e), alert=True)
            self.load_state.mech_loaded = False
            self.mix.update_species()
            self.log._blink(True)

            return
        self.log.append(loader.messages, alert=False)
        self.load_state.mech_loaded = True

        # Pre-spawn the worker pool while the user sets up the run:
        # mp.Pool() returns immediately and worker initializers import
        # the stack in the background, so the first optimization's pool
        # wait shrinks to whatever hasn't finished by then.
        self._prespawn_worker_pool()

        # Initialize tables and trees
        self.tree.handle_reload(prior_snapshot)

        tabIdx = self.plot_tab_widget.currentIndex()
        tabText = self.plot_tab_widget.tabText(tabIdx)
        if tabText == "Signal/Sim":
            observable = self.plot.observable_widget.widget[
                "main_parameter"
            ].currentText()
            self.plot.observable_widget.widget["main_parameter"].currentIndexChanged[
                str
            ].emit(observable)
        elif tabText == "Sim Explorer":
            self.sim_explorer.update_all_main_parameter()

    def _open_program_settings(self, event=None):
        if not getattr(self, "user_settings", None):
            return

        program_settings.ProgramSettingsDialog(self).execute()

    def _prespawn_worker_pool(self):
        if not hasattr(self, "worker_pool") or not self.mech.isLoaded:
            return
        if not self.multiprocessing_box.isChecked():
            return
        if not getattr(self, "user_settings", None):
            return
        opt_cfg = self.user_settings.config.optimization
        if not opt_cfg.pool_prespawn:
            return
        if "PYTEST_CURRENT_TEST" in os.environ:
            # Tests instantiate the app and load mechanisms constantly;
            # spawning a real worker pool per test is pure overhead.
            return
        payload = MechBuildPayload(
            reset_mech=self.mech.reset_mech,
            thermo_coeffs=self.mech.thermo_coeffs,
            coeffs=self.mech.coeffs,
            coeffs_bnds=self.mech.coeffs_bnds,
            rate_bnds=self.mech.rate_bnds,
        )
        # Match the orchestrator's sizing so a later optimize-time
        # acquire never needs a growth respawn.
        workers = opt_cfg.worker_count or default_worker_count()

        def spawn():
            try:
                self.worker_pool.acquire(workers=workers, payload=payload)
            except Exception:
                # Pre-spawning is opportunistic; the optimize path
                # spawns normally if this failed.
                pass

        # Process creation is far too slow for the GUI thread; the pool
        # manager's lock serializes against an optimize-time acquire.
        threading.Thread(target=spawn, daemon=True,
                         name="pool-prespawn").start()

    def load_session(self, event=None):
        if not getattr(self, "user_settings", None):
            return

        cfg_session = self.user_settings.config.session
        start_dir = cfg_session.last_session_file
        if not start_dir:
            start_dir = str(self.path.get("sim_main", ""))

        chosen, _selected = QFileDialog.getOpenFileName(
            self, "Load Session", start_dir,
            "Frhodo Session (*{:s})".format(session.SESSION_SUFFIX),
        )
        if not chosen:
            return

        restored, partial = session.read_session_file(self, chosen)
        cfg_session.last_session_file = chosen

        msg = "Loaded session {:s}: {:d} reactions restored".format(
            pathlib.Path(chosen).name, len(restored),
        )
        if partial:
            msg += ", {:d} flagged for review".format(len(partial))
        self.log.append(msg)

    def shock_choice_changed(self, event):
        if (
            "exp_main" in self.directory.invalid
        ):  # don't allow shock change if problem with exp directory
            return

        self.shock_selection.previous = self.shock_selection.current
        self.shock_selection.current = event

        self.shockRollingList = ["P1", "u1"]  # reset rolling list
        self.rxn_change_history = []  # reset tracking of rxn numbers changed

        if not self.run_control.optimize_running:
            self.log.clear([])
        self.series.change_shock()  # link display_shock to correct set and

    def update_user_settings(self, event=None):
        # This is one is located on the Files tab
        shock = self.display_shock
        self.series.set("series_name", self.exp_series_name_box.text())

        t_unit_conv = self.reactor_state.t_unit_conv
        if (
            self.time_offset_box.value() * t_unit_conv != shock.time_offset
        ):  # if values are different
            self.time_uncertainty.auto_fit = False  # user edit overrides auto-fit
            self.series.set("time_offset", self.time_offset_box.value() * t_unit_conv)
            if hasattr(self.mech_tree, "rxn"):  # checked if copy valid in function
                self.tree._copy_expanded_tab_rates()  # copy rates and time offset

        self.time_uncertainty.value = self.time_unc_box.value() * t_unit_conv
        self.time_uncertainty.random = self.random_t_unc_box.isChecked()

        if event is not None:
            sender = self.sender().objectName()
            if "time_offset" in sender and hasattr(
                self, "SIM"
            ):  # Don't rerun SIM if it exists
                if hasattr(self.SIM, "independent_var") and hasattr(
                    self.SIM, "observable"
                ):
                    self.plot.signal.update_sim(
                        self.SIM.independent_var, self.SIM.observable
                    )
            elif any(
                x in sender
                for x in ["end_time", "sim_interp_factor", "ODE_solver", "rtol", "atol"]
            ):
                self.run_single()
            elif self.display_shock.exp_data.size > 0:  # If exp_data exists
                self.plot.signal.update(update_lim=False)
                self.plot.signal.canvas.draw()

    def keyPressEvent(self, event):
        pass

    def run_single(self, event=None, t_save=None, rxn_changed=False):
        if self.run_control.run_block:
            return
        if not self.load_state.mech_loaded:
            return  # if mech isn't loaded successfully, exit
        if not hasattr(self.mech_tree, "rxn"):
            return  # if mech tree not set up, exit

        shock = self.display_shock

        T_reac, P_reac, mix = (
            shock.T_reactor,
            shock.P_reactor,
            shock.thermo_mix,
        )
        self.tree.update_rates()

        # calculate all properties or observable by sending t_save
        tabIdx = self.plot_tab_widget.currentIndex()
        tabText = self.plot_tab_widget.tabText(tabIdx)
        if tabText == "Sim Explorer":
            t_save = np.array([0])

        SIM_kwargs = {
            "u_reac": shock.u2,
            "rho1": shock.rho1,
            "observable": self.display_shock.observable,
            "t_lab_save": t_save,
            "sim_int_f": self.reactor_state.sim_interp_factor,
            "ODE_solver": self.reactor_state.ode_solver,
            "rtol": self.reactor_state.ode_rtol,
            "atol": self.reactor_state.ode_atol,
        }

        if self.reactor_state.name == "Incident Shock Reactor":
            self.SIM, verbose = run_incident_shock(
                self.mech, self.reactor_state.t_end,
                T_reac, P_reac, mix, **SIM_kwargs
            )
        elif "0d Reactor" in self.reactor_state.name:
            SIM_kwargs["solve_energy"] = self.reactor_state.solve_energy
            SIM_kwargs["frozen_comp"] = self.reactor_state.frozen_comp
            mode = _zero_d_mode_from_label(self.reactor_state.name)
            self.SIM, verbose = run_zero_d(
                self.mech, mode, self.reactor_state.t_end,
                T_reac, P_reac, mix, **SIM_kwargs
            )
        else:
            raise ValueError(f"unknown reactor: {self.reactor_state.name!r}")

        if verbose["success"]:
            self.log._blink(False)
        elif not self.run_control.optimize_running:
            # During optimization the iteration-line annotation carries
            # the compact ODE summary already; dumping the multi-line
            # block here would just clutter the iteration log.
            self.log.append(verbose["message"])

        if self.SIM is not None:
            apply_auto_fit_time_offset(self)
            self.plot.signal.update_sim(
                self.SIM.independent_var, self.SIM.observable, rxn_changed
            )
            if tabText == "Sim Explorer":
                self.sim_explorer.populate_main_parameters()
                self.sim_explorer.update_plot(self.SIM)  # sometimes duplicate updates
        else:
            nan = np.array([np.nan, np.nan])
            self.plot.signal.update_sim(nan, nan)  # make sim plot blank
            if tabText == "Sim Explorer":
                self.sim_explorer.update_plot(None)

            return  # If mech error exit function


def main():
    if platform.system() == "Windows":
        multiprocessing.freeze_support()

        if getattr(sys, "frozen", False):
            ctypes.windll.user32.ShowWindow(
                ctypes.windll.kernel32.GetConsoleWindow(), 0
            )

    app = QApplication(sys.argv)

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    # Wakes the interpreter every 100ms so SIGINT reaches Python.
    heartbeat = QtCore.QTimer()
    heartbeat.timeout.connect(lambda: None)
    heartbeat.start(100)

    sys.excepthook = error_window.excepthookDecorator(app, path, _mark_startup_failed)

    window = Main(app, path)

    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
