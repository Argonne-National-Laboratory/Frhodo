# This file is part of Frhodo. Copyright © 2020, UChicago Argonne, LLC
# and licensed under BSD-3-Clause. See License.txt in the top-level
# directory for license and copyright information.

import os, sys
from copy import deepcopy

from qtpy.QtWidgets import QDialog, QApplication, QShortcut, QListWidget
from qtpy import uic, QtCore, QtGui

import numpy as np

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
    def __init__(self, parent):
        super().__init__()
        uic.loadUi(str(parent.path["main"] / "UI" / "save_dialog.ui"), self)
        self.parent = parent

        self.var = {
            "comment": "",
            "output_time": np.array([]),
            "integrator_time": False,
            "parameters": [],
            "species": [],
            "reactions": [],
            "save_plot": True,
        }

        # self.parameters_list_widget.setSelectionMode(QListWidget.MultiSelection)
        # self.species_list_widget.setSelectionMode(QListWidget.MultiSelection)
        # self.reactions_list_widget.setSelectionMode(QListWidget.MultiSelection)
        self.parameters_list_widget.setSelectionMode(QListWidget.ExtendedSelection)
        self.species_list_widget.setSelectionMode(QListWidget.ExtendedSelection)
        self.reactions_list_widget.setSelectionMode(QListWidget.ExtendedSelection)

        self.action_Save.triggered.connect(self.accept)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        # Set Shortcuts
        # shortcut_fcn_pair = [['Ctrl+R', lambda: self._reset()], ['Ctrl+C', lambda: self._copy()],
        # ['Ctrl+V', lambda: self._paste()]]
        # for shortcut, fcn in shortcut_fcn_pair:     # TODO: need to fix hover shortcuts not working
        # QShortcut(QtGui.QKeySequence(shortcut), self, activated=fcn, context=QtCore.Qt.WidgetShortcut)

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

        # change selection behavior if shift is selectedItems, CURRENTLY NOT FUNCTIONING
        elif obj in listWidgets:
            if event.type() == QtGui.QKeyEvent:
                if (
                    event.type() == QtCore.QEvent.KeyPress
                    and event.modifiers() == QtCore.Qt.ShiftModifier
                ):
                    print("shift down")
                    self.parameters_list_widget.setSelectionMode(
                        QListWidget.ExtendedSelection
                    )
                    self.species_list_widget.setSelectionMode(
                        QListWidget.ExtendedSelection
                    )
                    self.reactions_list_widget.setSelectionMode(
                        QListWidget.ExtendedSelection
                    )
                elif (
                    event.type() == QtCore.QEvent.KeyRelease
                    and event.modifiers() == QtCore.Qt.ShiftModifier
                ):
                    print("shift up")
                    self.parameters_list_widget.setSelectionMode(
                        QListWidget.MultiSelection
                    )
                    self.species_list_widget.setSelectionMode(
                        QListWidget.MultiSelection
                    )
                    self.reactions_list_widget.setSelectionMode(
                        QListWidget.MultiSelection
                    )

        return super().eventFilter(obj, event)

    def execute(self, event=None):
        def setSelected(listWidget, items, match_type=QtCore.Qt.MatchExactly):
            for item in items:
                matching_items = listWidget.findItems(item, match_type)

                for item in matching_items:
                    item.setSelected(True)

        if not self.parent.mech_loaded:
            return  # if mech isn't loaded successfully, exit
        if self.parent.SIM is None:
            return  # if no successful SIM
        if not self.parent.SIM.success:
            return  # if prior SIM is not successful, exit

        gas = self.parent.mech.gas

        # Update Save Dialog
        t_unit = self.parent.end_time_units_box.currentText()
        self.output_times_label.setText("Output Save Times " + t_unit)

        # Clear and Populate List Widgets
        self.parameters_list_widget.clear()
        reactor_vars = deepcopy(self.parent.SIM.reactor_var)
        if "Laboratory Time" in reactor_vars:
            del reactor_vars["Laboratory Time"]
        self.parameters_list_widget.addItems(reactor_vars.keys())
        setSelected(self.parameters_list_widget, self.var["parameters"])

        # set species list widget
        self.species_list_widget.clear()
        self.species_list_widget.addItems(gas.species_names)
        if not self.var["species"]:
            setSelected(self.species_list_widget, gas.species_names)
        else:
            setSelected(self.species_list_widget, self.var["species"].values())

        # set reactions list widget
        self.reactions_list_widget.clear()
        reactions = []
        for i, rxn in enumerate(self.parent.mech.gas.reaction_equations()):
            reactions.append("R{:<5d}  {:s}".format(i + 1, rxn))
        self.reactions_list_widget.addItems(reactions)
        if not self.var["reactions"]:
            setSelected(self.reactions_list_widget, reactions)
        else:
            setSelected(
                self.reactions_list_widget,
                self.var["reactions"].values(),
                match_type=QtCore.Qt.MatchContains,
            )

        self.comment_box.setFocus()
        # if self.last_focus is not None:
        # self.last_focus.setFocus()
        if (
            self.focusWidget() is self.comment_box
        ):  # select all text if comment box was last edited
            self.comment_box.selectAll()
        # self.show() # can access program with modal window open
        self.exec_()  # greedy, no accessing program

    def _set_variables(self):
        gas = self.parent.mech.gas

        self.var["save_plot"] = self.save_plot_box.isChecked()
        self.var["comment"] = self.comment_box.toPlainText()

        self.var["parameters"] = [
            item.text() for item in self.parameters_list_widget.selectedItems()
        ]

        # Get species with correct index
        selected_species = [
            item.text() for item in self.species_list_widget.selectedItems()
        ]
        self.var["species"] = {}
        for n, species in enumerate(gas.species_names):
            if species in selected_species:
                self.var["species"][n] = species

        # [8:] trims off R#####__  from the string
        selected_rxns = [
            item.text()[8:] for item in self.reactions_list_widget.selectedItems()
        ]
        self.var["reactions"] = {}
        for n, rxn in enumerate(gas.reaction_equations()):
            if rxn in selected_rxns:
                self.var["reactions"][n] = rxn
        # self.var['reactions'] = {i: item.text()[8:] for i, item in enumerate(self.reactions_list_widget.selectedItems())}

        # Set Simulation output time
        t_save = np.fromstring(
            self.output_times_box.toPlainText(), sep=",", dtype=float
        )
        t_save = np.unique(t_save[t_save >= 0])  # Only allow positive unique values
        if len(t_save) == 0:
            t_save = np.array([0])
        self.var["output_time"] = t_save * self.parent.var["reactor"]["t_unit_conv"]

        self.var["integrator_time"] = self.include_integrator_time_box.isChecked()
        if self.include_time_offset_box.isChecked():
            self.var["output_time_offset"] = self.parent.display_shock[
                "time_offset"
            ]  # uses display offset
        else:
            self.var["output_time_offset"] = 0

    def accept(self):
        parent = self.parent
        self._set_variables()

        parent.run_single(t_save=self.var["output_time"])
        parent.save.all(parent.SIM, self.var)
        parent.directory.update_icons()  # update icons in case Sim directory wasn't made previously

        super().accept()  # default action for accept

    def reject(self):
        super().reject()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    save_dialog_widget = Save_Dialog()
    sys.exit(save_dialog_widget.execute())
