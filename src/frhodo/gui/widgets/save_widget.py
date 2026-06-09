# This file is part of Frhodo. Copyright © 2020, UChicago Argonne, LLC
# and licensed under BSD-3-Clause. See License.txt in the top-level
# directory for license and copyright information.

import os
import pathlib
import sys

from copy import deepcopy

from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QShortcut,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from qtpy import uic, QtCore, QtGui

import numpy as np

from frhodo.common.units import PRESSURE_UNITS, pa_per_unit
from frhodo.gui import session
from frhodo.gui.state import SaveDialogState



if (
    os.environ["QT_API"] == "pyside2"
):  # Silence warning: "Qt WebEngine seems to be initialized from a plugin."
    QApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)

# Handle high resolution displays:  Minimum recommended resolution 1280 x 960
if hasattr(QtCore.Qt, "AA_EnableHighDpiScaling"):
    QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
if hasattr(QtCore.Qt, "AA_UseHighDpiPixmaps"):
    QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)


class Save_Dialog(QDialog, QApplication):
    """User-facing Save dialog for simulation traces and modified mechanisms.

    State lives in :class:`SaveDialogState` (typed, validation-assigned).
    The dialog wires its list widgets to the parameter / species /
    reaction selections and routes the OK action through ``Save``.
    """

    def __init__(self, parent):
        super().__init__()
        uic.loadUi(str(parent.path["package"] / "ui" / "save_dialog.ui"), self)
        self.parent = parent

        self.state = SaveDialogState(output_time=np.array([]))

        self.parameters_list_widget.setSelectionMode(QListWidget.ExtendedSelection)
        self.species_list_widget.setSelectionMode(QListWidget.ExtendedSelection)
        self.reactions_list_widget.setSelectionMode(QListWidget.ExtendedSelection)

        self._wrap_in_outer_tabs()

        self.action_Save.triggered.connect(self.accept)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        # installing event filter to track focus
        self.last_focus = None
        for widget in [
            self,
            self.parameters_list_widget,
            self.species_list_widget,
            self.reactions_list_widget,
            self.comment_box,
            self.output_times_box,
            self.include_integrator_time_box,
            self.save_plot_box,
        ]:
            widget.installEventFilter(self)

    def _wrap_in_outer_tabs(self):
        """Move the original simulation widgets into a Simulation tab and
        add a parallel Mechanism tab next to it.
        """
        sim_page = QWidget()
        sim_layout = QGridLayout(sim_page)
        sim_layout.setContentsMargins(0, 0, 0, 0)
        sim_layout.addWidget(self.frame_4, 0, 0, 2, 1)
        sim_layout.addWidget(self.frame_3, 0, 1)
        sim_layout.addWidget(self.frame_2, 1, 1)

        mech_page = self._build_mechanism_page()
        session_page = self._build_session_page()

        self.outer_tab_widget = QTabWidget(self)
        self.outer_tab_widget.setObjectName("outer_tab_widget")
        self.outer_tab_widget.addTab(sim_page, "Simulation")
        self.outer_tab_widget.addTab(mech_page, "Mechanism")
        self.outer_tab_widget.addTab(session_page, "Session")
        self.outer_tab_widget.currentChanged.connect(self._on_outer_tab_changed)

        new_layout = QVBoxLayout()
        new_layout.addWidget(self.outer_tab_widget)
        new_layout.addWidget(self.frame_5)

        # QWidget owns exactly one layout; transfer the old one to a
        # throwaway parent so we can install the new top-level layout.
        QWidget().setLayout(self.layout())
        self.setLayout(new_layout)

    def _build_mechanism_page(self) -> QWidget:
        page = QWidget()
        layout = QGridLayout(page)

        self.mech_info_label = QLabel()
        self.mech_info_label.setWordWrap(True)
        self.mech_info_label.setAlignment(QtCore.Qt.AlignTop)
        layout.addWidget(self.mech_info_label, 0, 0, 1, 2)

        self.recast_box = QCheckBox(
            "Recast pressure-dependent reactions to Arrhenius at:"
        )
        self.recast_box.setToolTip(
            "Falloff (Lindemann/Troe/SRI/Tsang) becomes three-body Arrhenius "
            "with original efficiencies preserved.\n"
            "Plog and Chebyshev become pure Arrhenius.\n"
            "Pure Arrhenius and three-body Arrhenius pass through unchanged."
        )
        layout.addWidget(self.recast_box, 1, 0, 1, 2)

        pressure_row = QHBoxLayout()
        self.recast_pressure_value_box = QDoubleSpinBox()
        self.recast_pressure_value_box.setRange(1.0e-6, 1.0e6)
        self.recast_pressure_value_box.setDecimals(4)
        self.recast_pressure_value_box.setValue(1.0)
        self.recast_pressure_value_box.setEnabled(False)
        pressure_row.addWidget(self.recast_pressure_value_box)

        self.recast_pressure_unit_box = QComboBox()
        self.recast_pressure_unit_box.addItems(PRESSURE_UNITS)
        self.recast_pressure_unit_box.setCurrentText("atm")
        self.recast_pressure_unit_box.setEnabled(False)
        pressure_row.addWidget(self.recast_pressure_unit_box)
        pressure_row.addStretch(1)
        layout.addLayout(pressure_row, 2, 0, 1, 2)

        self.recast_box.toggled.connect(self.recast_pressure_value_box.setEnabled)
        self.recast_box.toggled.connect(self.recast_pressure_unit_box.setEnabled)

        layout.setRowStretch(3, 1)

        return page

    def _build_session_page(self) -> QWidget:
        page = QWidget()
        layout = QGridLayout(page)

        info = QLabel(
            "Save the entire GUI state — directory paths, preferences, "
            "per-shock weights, and per-reaction uncertainties — to a "
            ".frhodo file. Restore it later from the Load button in the "
            "toolbar."
        )
        info.setWordWrap(True)
        info.setAlignment(QtCore.Qt.AlignTop)
        layout.addWidget(info, 0, 0, 1, 2)

        self.autosnapshot_box = QCheckBox(
            "Auto-snapshot the session before optimizing and periodically "
            "during the run"
        )
        self.autosnapshot_box.setToolTip(
            "Writes session_autosave.frhodo to the simulation directory so a "
            "crash mid-optimization is recoverable."
        )
        layout.addWidget(self.autosnapshot_box, 1, 0, 1, 2)

        interval_row = QHBoxLayout()
        interval_row.addWidget(QLabel("Snapshot every"))
        self.snapshot_interval_box = QDoubleSpinBox()
        self.snapshot_interval_box.setRange(1.0, 3600.0)
        self.snapshot_interval_box.setDecimals(0)
        self.snapshot_interval_box.setSuffix(" s")
        interval_row.addWidget(self.snapshot_interval_box)
        interval_row.addStretch(1)
        layout.addLayout(interval_row, 2, 0, 1, 2)

        self.autosnapshot_box.toggled.connect(self.snapshot_interval_box.setEnabled)
        layout.setRowStretch(3, 1)

        return page

    def _refresh_mech_info_label(self):
        gas = getattr(self.parent.mech, "gas", None)
        if gas is None:
            self.mech_info_label.setText("No mechanism loaded.")

            return

        mech_path = self.parent.path.get("mech")
        if mech_path:
            name = pathlib.Path(mech_path).name
        else:
            name = "(unknown)"

        self.mech_info_label.setText(
            "Mechanism: {name}\n"
            "Species: {n_species}\n"
            "Reactions: {n_reactions}\n\n"
            "Click Save to choose the output file and format."
            .format(name=name, n_species=gas.n_species, n_reactions=gas.n_reactions)
        )

    def _on_outer_tab_changed(self, index):
        self.save_plot_box.setVisible(index == 0)

    def _default_mech_output_dir(self) -> str:
        path = self.parent.path
        for key in ("mech_main", "mech"):
            candidate = path.get(key)
            if candidate is None:
                continue
            candidate = pathlib.Path(candidate)
            if candidate.is_file():
                candidate = candidate.parent
            if candidate.exists():
                return str(candidate)

        return str(pathlib.Path.cwd())

    def eventFilter(self, obj, event):  # track last focus excluding pushbuttons
        listWidgets = [
            self.parameters_list_widget,
            self.species_list_widget,
            self.reactions_list_widget,
        ]

        if event.type() == QtCore.QEvent.FocusIn and obj is not self:
            self.last_focus = obj  # track last focus object

        # intercept ctrl + s or ctrl + return to save
        elif event.type() == QtCore.QEvent.KeyPress:
            if event.modifiers() == QtCore.Qt.ControlModifier:
                if event.key() in [QtCore.Qt.Key_S, QtCore.Qt.Key_Return]:
                    self.accept()

                    return True

        # if output times is empty set integrator_time to True
        elif event.type() == QtCore.QEvent.FocusOut and obj is self.output_times_box:
            if (
                len(
                    np.fromstring(
                        self.output_times_box.toPlainText(), sep=",", dtype=float
                    )
                )
                == 0
            ):
                self.include_integrator_time_box.setChecked(True)

        return super().eventFilter(obj, event)

    def execute(self, event=None):
        def setSelected(listWidget, items, match_type=QtCore.Qt.MatchExactly):
            for item in items:
                matching_items = listWidget.findItems(item, match_type)

                for item in matching_items:
                    item.setSelected(True)

        if not self.parent.load_state.mech_loaded:
            return

        gas = self.parent.mech.gas

        sim_ready = (
            self.parent.SIM is not None and getattr(self.parent.SIM, "success", False)
        )
        self.outer_tab_widget.setTabEnabled(0, sim_ready)
        if not sim_ready:
            self.outer_tab_widget.setCurrentIndex(1)

        if sim_ready:
            # Update Save Dialog
            t_unit = self.parent.end_time_units_box.currentText()
            self.output_times_label.setText("Output Save Times " + t_unit)

            # Clear and Populate List Widgets
            self.parameters_list_widget.clear()
            reactor_vars = deepcopy(self.parent.SIM.reactor_var)
            if "Laboratory Time" in reactor_vars:
                del reactor_vars["Laboratory Time"]
            self.parameters_list_widget.addItems(reactor_vars.keys())
            setSelected(self.parameters_list_widget, self.state.parameters)

            # set species list widget
            self.species_list_widget.clear()
            self.species_list_widget.addItems(gas.species_names)
            if not self.state.species:
                setSelected(self.species_list_widget, gas.species_names)
            else:
                setSelected(self.species_list_widget, self.state.species.values())

            # set reactions list widget
            self.reactions_list_widget.clear()
            reactions = []
            for i, rxn in enumerate(self.parent.mech.gas.reaction_equations()):
                reactions.append("R{:<5d}  {:s}".format(i + 1, rxn))
            self.reactions_list_widget.addItems(reactions)
            if not self.state.reactions:
                setSelected(self.reactions_list_widget, reactions)
            else:
                setSelected(
                    self.reactions_list_widget,
                    self.state.reactions.values(),
                    match_type=QtCore.Qt.MatchContains,
                )

        cfg_session = self.parent.user_settings.config.session
        self.autosnapshot_box.setChecked(cfg_session.autosnapshot_enabled)
        self.snapshot_interval_box.setValue(cfg_session.snapshot_interval_s)

        self._refresh_mech_info_label()
        self._on_outer_tab_changed(self.outer_tab_widget.currentIndex())

        self.comment_box.setFocus()
        if (
            self.focusWidget() is self.comment_box
        ):  # select all text if comment box was last edited
            self.comment_box.selectAll()
        self.exec_()  # greedy, no accessing program

    def _set_variables(self):
        gas = self.parent.mech.gas

        self.state.save_plot = self.save_plot_box.isChecked()
        self.state.comment = self.comment_box.toPlainText()

        self.state.parameters = [
            item.text() for item in self.parameters_list_widget.selectedItems()
        ]

        # Get species with correct index
        selected_species = [
            item.text() for item in self.species_list_widget.selectedItems()
        ]
        species: dict = {}
        for n, name in enumerate(gas.species_names):
            if name in selected_species:
                species[n] = name
        self.state.species = species

        # [8:] trims off R#####__  from the string
        selected_rxns = [
            item.text()[8:] for item in self.reactions_list_widget.selectedItems()
        ]
        reactions: dict = {}
        for n, rxn in enumerate(gas.reaction_equations()):
            if rxn in selected_rxns:
                reactions[n] = rxn
        self.state.reactions = reactions

        # Set Simulation output time
        t_save = np.fromstring(
            self.output_times_box.toPlainText(), sep=",", dtype=float
        )
        t_save = np.unique(t_save[t_save >= 0])
        if len(t_save) == 0:
            t_save = np.array([0])
        self.state.output_time = t_save * self.parent.reactor_state.t_unit_conv

        self.state.integrator_time = self.include_integrator_time_box.isChecked()
        if self.include_time_offset_box.isChecked():
            self.state.output_time_offset = self.parent.display_shock.time_offset
        else:
            self.state.output_time_offset = 0.0

    def accept(self):
        index = self.outer_tab_widget.currentIndex()
        if index == 2:
            self._save_session()
        elif index == 1:
            self._save_mechanism()
        else:
            self._save_simulation()

        super().accept()

    def _save_simulation(self):
        parent = self.parent
        self._set_variables()

        parent.run_single(t_save=self.state.output_time)
        parent.save.all(parent.SIM, self.state.model_dump())
        parent.directory.update_icons()

    def _default_session_dir(self) -> str:
        sim_main = self.parent.path.get("sim_main")
        if sim_main and pathlib.Path(sim_main).exists():
            start_dir = str(sim_main)
        else:
            start_dir = str(pathlib.Path.cwd())

        return start_dir

    def _save_session(self):
        parent = self.parent
        cfg_session = parent.user_settings.config.session
        cfg_session.autosnapshot_enabled = self.autosnapshot_box.isChecked()
        cfg_session.snapshot_interval_s = self.snapshot_interval_box.value()

        start_dir = cfg_session.last_session_file or self._default_session_dir()
        chosen, _selected = QFileDialog.getSaveFileName(
            self, "Save Session", start_dir,
            "Frhodo Session (*{:s})".format(session.SESSION_SUFFIX),
        )
        if chosen:
            path = pathlib.Path(chosen)
            if path.suffix.lower() != session.SESSION_SUFFIX:
                path = path.with_suffix(session.SESSION_SUFFIX)

            session.write_session_file(
                parent, path, comment=self.comment_box.toPlainText(),
            )
            cfg_session.last_session_file = str(path)

        parent.user_settings.save()

    def _save_mechanism(self):
        if hasattr(self.parent, "tree"):
            self.parent.tree._drain_pending_render()

        self.state.recast_to_arrhenius = self.recast_box.isChecked()
        unit = self.recast_pressure_unit_box.currentText()
        self.state.recast_pressure_pa = (
            self.recast_pressure_value_box.value() * pa_per_unit[unit]
        )

        start_dir = self.state.mech_output_dir or self._default_mech_output_dir()
        yaml_filter = "Cantera YAML (*.yaml *.yml)"
        chemkin_filter = "Chemkin (*.inp *.dat *.mech)"
        chosen, selected_filter = QFileDialog.getSaveFileName(
            self, "Save Mechanism", start_dir,
            ";;".join([yaml_filter, chemkin_filter]),
        )
        if not chosen:
            return

        path = pathlib.Path(chosen)
        suffix = path.suffix.lower()
        yaml_suffixes = {".yaml", ".yml"}
        chemkin_suffixes = {".inp", ".dat", ".mech"}

        if suffix in yaml_suffixes:
            write_yaml = True
        elif suffix in chemkin_suffixes:
            write_yaml = False
        elif selected_filter == yaml_filter:
            path = path.with_suffix(".yaml")
            write_yaml = True
        else:
            path = path.with_suffix(".inp")
            write_yaml = False

        mech_to_save = self.parent.mech
        if self.state.recast_to_arrhenius:
            composition = self._recast_composition()
            mech_to_save = mech_to_save.recast_pdep_at_pressure(
                self.state.recast_pressure_pa, composition,
            )

        if write_yaml:
            path.write_text(mech_to_save.to_yaml_text())
        else:
            mech_to_save.to_chemkin(path)

        self.state.mech_output_dir = str(path.parent)

    def _recast_composition(self):
        """Composition used to evaluate falloff rates during recast.

        Defaults to the currently-displayed shock's mix; falls back to
        a single species at unit fraction if no shock composition is
        available (so Plog/Chebyshev recasts still work — those don't
        depend on composition).
        """
        shock = getattr(self.parent, "display_shock", None)
        if shock is not None:
            mix = getattr(shock, "thermo_mix", None)
        else:
            mix = None

        if mix:
            composition = dict(mix)
        else:
            composition = {self.parent.mech.gas.species_names[0]: 1.0}

        return composition

    def reject(self):
        super().reject()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    save_dialog_widget = Save_Dialog()
    sys.exit(save_dialog_widget.execute())
