# This file is part of Frhodo. Copyright © 2020, UChicago Argonne, LLC
# and licensed under BSD-3-Clause. See License.txt in the top-level
# directory for license and copyright information.

import numpy as np
from calculate import shock_fcns
import mech_widget, misc_widget, thermo_widget, series_viewer_widget, save_output
from qtpy.QtWidgets import *
from qtpy import QtWidgets, QtGui, QtCore
from copy import deepcopy
import re
from timeit import default_timer as timer


class SIM_Explorer_Widgets(QtCore.QObject):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.parent.plot_tab_widget.currentChanged.connect(self.tab_changed)

        self.max_history = 9
        self.widget = []
        self.updating_boxes = False
        self.var_dict = parent.SIM.all_var

        for axis in ["x", "y", "y2"]:
            self.create_choices(axis=axis)

    def create_choices(self, axis):
        parent = self.parent

        spacer = QtWidgets.QSpacerItem(
            10, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum
        )
        main_parameter_box = misc_widget.ItemSearchComboBox()
        # main_parameter_box.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        if axis != "x":
            sub_parameter_box = misc_widget.CheckableSearchComboBox(parent)
        else:
            sub_parameter_box = misc_widget.ItemSearchComboBox()

        self.widget.append([main_parameter_box, sub_parameter_box])

        itemMemory = {}
        main_parameter_box.info = {
            "axis": axis,
            "boxes": self.widget[-1],
            "itemMemory": itemMemory,
        }
        sub_parameter_box.info = {
            "axis": axis,
            "boxes": self.widget[-1],
            "itemMemory": itemMemory,
        }
        sub_parameter_box.setEnabled(False)
        sub_parameter_box.checked = []

        self.populate_main_parameters(comboBoxes=main_parameter_box)

        if axis == "x":
            layout = parent.sim_xchoice_layout
            main_parameter_box.setCurrentIndex(0)
        elif axis == "y":
            layout = parent.sim_ychoice1_layout
            main_parameter_box.setCurrentIndex(3)
        elif axis == "y2":
            layout = parent.sim_ychoice2_layout
            main_parameter_box.setCurrentIndex(0)

        if axis in ["x", "y"]:
            layout.addWidget(main_parameter_box, 0, 0)
            layout.addItem(spacer, 0, 1)
            layout.addWidget(sub_parameter_box, 0, 2)
        elif axis == "y2":  # flip box order if y2
            layout.addWidget(sub_parameter_box, 0, 0)
            layout.addItem(spacer, 0, 1)
            layout.addWidget(main_parameter_box, 0, 2)

        # connect signals
        main_parameter_box.currentIndexChanged[str].connect(self.main_parameter_changed)
        sub_parameter_box.model().itemChanged.connect(self.sub_item_changed)
        sub_parameter_box.currentIndexChanged.connect(self.sub_item_changed)

    def tab_changed(self, idx):  # Run simulation is tab changed to Sim Explorer
        if self.parent.plot_tab_widget.tabText(idx) == "Sim Explorer":
            self.populate_main_parameters()
            self.update_plot(SIM=self.parent.SIM)

    def populate_main_parameters(self, comboBoxes=None):
        if len(self.parent.SIM.reactor_var) > 0:  # if SIM has reactor variables
            reactor_vars = deepcopy(list(self.parent.SIM.reactor_var.keys()))
        else:
            reactor_vars = list(self.var_dict.keys())

        # create list of comboboxes
        if comboBoxes is None:
            comboBoxes = [boxes[0] for boxes in self.widget]
        else:
            comboBoxes = [comboBoxes]

        for comboBox in comboBoxes:
            if comboBox.info["axis"] == "y2":
                reactor_vars.insert(0, "-")

            choice = comboBox.currentText()

            comboBox.blockSignals(True)
            comboBox.clear()
            comboBox.addItems(reactor_vars)
            comboBox.blockSignals(False)
            comboBox.itemList = reactor_vars

            # create memory if doesn't exist
            if list(comboBox.info["itemMemory"].keys()) != ["species", "rxn"]:
                # update item memory for switching variables and keeping selections
                itemMemory = {
                    var: {"selected": None, "checked": []} for var in ["species", "rxn"]
                }
                comboBox.info[
                    "itemMemory"
                ].clear()  # preserve original dictionary to keep link between main/sub
                comboBox.info["itemMemory"].update(itemMemory)

            if (
                choice not in reactor_vars
            ):  # defaults to temperature if variable not found
                choice = "Temperature"
            comboBox.setCurrentText(choice)
            comboBox.setCurrentIndex(reactor_vars.index(choice))

        # keeps selected options
        tab_widget_idx = self.parent.plot_tab_widget.currentIndex()
        if self.parent.plot_tab_widget.tabText(tab_widget_idx) == "Sim Explorer":
            self.update_plot(self.parent.SIM)

    def main_parameter_changed(self, event):
        if not self.parent.mech_loaded:
            return  # if mech isn't loaded successfully, exit

        sender = self.sender()
        subComboBox = sender.info["boxes"][1]
        self.updating_boxes = True  # to prevent multiple plot updates
        subComboBox.clear()
        subComboBox.checked = []

        # Populate subComboBox
        prior = {"selected": None, "checked": []}
        if event == "-":
            subComboBox.setEnabled(False)
        else:
            sub_type = self.var_dict[event]["sub_type"]

            if sub_type is None or not hasattr(self.parent, "mech"):
                subComboBox.setEnabled(False)
            else:
                subComboBox.setEnabled(True)

                # if set(['species', 'rxn']).intersection(sub_type):
                items = []
                if "total" in sub_type:
                    items = ["Total"]

                if "species" in sub_type:
                    itemMemory = deepcopy(
                        subComboBox.info["itemMemory"]["species"]
                    )  # Get prior settings
                    items.extend(self.parent.mech.gas.species_names)
                elif "rxn" in sub_type:
                    itemMemory = deepcopy(
                        subComboBox.info["itemMemory"]["rxn"]
                    )  # Get prior settings
                    rxn_strings = self.parent.mech.gas.reaction_equations()
                    for n, rxn in enumerate(rxn_strings):
                        rxn_strings[n] = "R{:d}:  {:s}".format(
                            n + 1, rxn.replace("<=>", "=")
                        )

                    items.extend(rxn_strings)

                subComboBox.addItems(items)
                for n, text in enumerate(items):
                    # Check if text in prior settings
                    if ":" in text:  # if reaction, strip R# from text
                        text = text.split(":")[1].lstrip()

                    if (
                        itemMemory["selected"] is not None
                        and itemMemory["selected"] == text
                    ):
                        prior["selected"] = n

                    if any(value == text for value in itemMemory["checked"]):
                        prior["checked"].append(n)

        # Update subComboBox from prior settings
        subComboBox.checked = prior["checked"]
        for n in prior["checked"]:
            subComboBox.model().item(n, 0).setCheckState(
                QtCore.Qt.Checked
            )  # check prior selected

        if prior["selected"] is not None:  # set value to last selected
            subComboBox.setCurrentIndex(prior["selected"])

        self.updating_boxes = False
        self.update_plot(self.parent.SIM)

    def update_all_main_parameter(self):
        comboBoxes = [boxes[0] for boxes in self.widget]
        for comboBox in comboBoxes:
            idx = comboBox.currentIndex()
            comboBox.blockSignals(True)  # this forces an update through the widgets
            comboBox.setCurrentIndex(0)
            comboBox.blockSignals(False)
            comboBox.setCurrentIndex(idx)

    def sub_item_changed(self, sender=None):
        def comboBoxValidator(text):
            if ":" in text:  # if reaction, strip R# from text
                text = text.split(":")[1].lstrip()
            return text

        if self.updating_boxes:
            return  # do not run if updating boxes from main_parameter_changed
        sender = self.sender()

        # if sender is widget, not checkbox
        if type(sender) in [
            misc_widget.CheckableSearchComboBox,
            misc_widget.ItemSearchComboBox,
        ]:
            main_choice = sender.info["boxes"][0].currentText()
            sub_type = self.var_dict[main_choice]["sub_type"]
            text = comboBoxValidator(sender.currentText())

            if "species" in sub_type:
                itemMemory = sender.info["itemMemory"]["species"]
            elif "rxn" in sub_type:
                itemMemory = sender.info["itemMemory"]["rxn"]

            if len(text) > 0:  # if text exists, put in memory
                itemMemory["selected"] = text

            self.update_plot(self.parent.SIM)
            return

        # If the sender is a checkbox perform following
        axis = sender.parent().info["axis"]
        mainComboBox = sender.parent().info["boxes"][0]
        main_choice = mainComboBox.currentText()
        comboBox = sender.parent().info["boxes"][1]
        sub_type = self.var_dict[main_choice]["sub_type"]

        checked_idx = []
        for row in range(sender.rowCount()):
            item = sender.item(row)  # get the item in row.
            if item and item.checkState() == 2:  # if the item is checked.
                checked_idx.append(row)

        # find difference between stored list and current, append or remove accordingly
        added = np.setdiff1d(checked_idx, comboBox.checked)
        removed = np.setdiff1d(comboBox.checked, checked_idx)
        if added.size > 0:
            if (
                len(comboBox.checked) > self.max_history
            ):  # if length is greater than max, delete value
                n = 0
                if (
                    "total" in sub_type and 0 == comboBox.checked[0]
                ):  # if total selected, delete next
                    n = 1

                item = comboBox.model().item(comboBox.checked[n], 0)
                comboBox.model().blockSignals(
                    True
                )  # block update if checkstate changed
                item.setCheckState(
                    QtCore.Qt.Unchecked
                )  # uncheck first non Total selection
                comboBox.model().blockSignals(False)
                del comboBox.checked[n]

            comboBox.checked.append(added[0])

        elif removed.size > 0:
            comboBox.checked.remove(removed[0])

        # update itemMemory with stored list
        if "species" in sub_type:
            itemMemory = comboBox.info["itemMemory"]["species"]
        elif "rxn" in sub_type:
            itemMemory = comboBox.info["itemMemory"]["rxn"]
        itemMemory["checked"].clear()

        for n in comboBox.checked:
            text = comboBoxValidator(comboBox.itemList[n])
            itemMemory["checked"].append(text)

        # update selected text to match most recent checked
        if len(comboBox.checked) > 0:  # set value to last appended value in stored list
            comboBox.blockSignals(True)
            comboBox.setCurrentIndex(comboBox.checked[-1])
            comboBox.blockSignals(False)
            text = comboBoxValidator(comboBox.itemList[comboBox.checked[-1]])
            itemMemory["selected"] = text

        self.update_plot(self.parent.SIM)

    def update_plot(self, SIM=None):
        if self.updating_boxes:
            return  # do not update plot if updating boxes
        if not self.parent.mech_loaded:
            return  # if mech isn't loaded successfully, exit
        if SIM is None:
            return  # if SIM hasn't run, exit

        def getData(SIM, var):
            return eval(f"SIM.{var}()")

        label = {"y": [], "y2": []}
        label_order = {"y": [], "y2": []}
        data = {"x": [], "y": [], "y2": []}
        for n, axis in enumerate(["x", "y", "y2"]):
            main_choice = self.widget[n][0].currentText()
            sub_choice = np.array(self.widget[n][1].checked)
            selected_sub_choice = self.widget[n][1].currentIndex()
            itemList = np.array(self.widget[n][1].itemList)

            if main_choice != "-":
                SIM_name = self.var_dict[main_choice]["SIM_name"]
                sub_type = self.var_dict[main_choice]["sub_type"]

                if (
                    self.widget[n][1].count() > 0
                ):  # find all choices other than the selected index
                    sub_choice = sub_choice[sub_choice != selected_sub_choice].astype(
                        int
                    )
                    sub_choice = np.append(
                        selected_sub_choice, np.sort(sub_choice)
                    )  # append list to selected
                else:
                    sub_choice = [
                        0
                    ]  # for types with no sub_choices, default to first array

                if sub_type is None:  # if no Sub_type, do something
                    pass
                elif (
                    "total" in sub_type
                ):  # if total in subtype, subtract 1 (total is 0)
                    sub_choice = np.subtract(sub_choice, 1)

                raw_data = getData(SIM, SIM_name)
                if (
                    raw_data.ndim == 1
                ):  # if raw data is a vector only, give it a second dimension
                    raw_data = raw_data[
                        None, :
                    ]  # this is for times, velocity, temperature

                for n in sub_choice:
                    if n == -1:  # If total then append special
                        data[axis].append(getData(SIM, SIM_name + "_tot"))
                    else:  # if per reaction
                        data[axis].append(raw_data[n, :])

                # set labels for legend
                if len(itemList) > 0:
                    if SIM_name == "delta_h":
                        labels = []
                        for n, choice in enumerate(
                            sub_choice
                        ):  # if delta enthalpy, add t(0) value in e format
                            token = itemList[choice].split(":  ")
                            val = "{:>8.1f}".format(data[axis][n][0])
                            # val = '{:>10.3e}'.format(data[axis][n][0])
                            # val = re.sub("e(-?)0*(\d+)", r"e\1\2", val.replace("e+", "e"))
                            labels.append(
                                "{:s}:  dH(0) = {:s},  {:s}".format(
                                    token[0], val, token[1]
                                )
                            )
                    elif SIM_name == "rate_con_rev":
                        labels = []
                        for n, choice in enumerate(
                            sub_choice
                        ):  # if delta enthalpy, add t(0) value in e format
                            token = itemList[choice].split(":  ")
                            val = "{:>8.5e}".format(data[axis][n][0])
                            # val = '{:>10.3e}'.format(data[axis][n][0])
                            # val = re.sub("e(-?)0*(\d+)", r"e\1\2", val.replace("e+", "e"))
                            labels.append(
                                "{:s}:  kr(0) = {:s},  {:s}".format(
                                    token[0], val, token[1]
                                )
                            )
                    else:  # all other legend entries are entered as is
                        if "total" in sub_type:  # account for subtracting one before
                            sub_choice = sub_choice + 1

                        labels = itemList[sub_choice]

                    label[axis] = np.array(labels)
                    label_order[axis] = np.argsort(sub_choice)

        self.parent.plot.sim_explorer.update(data, label, label_order)
