# This file is part of Frhodo. Copyright © 2020, UChicago Argonne, LLC
# and licensed under BSD-3-Clause. See License.txt in the top-level
# directory for license and copyright information.

import multiprocessing as mp
import re
from copy import deepcopy
from timeit import default_timer as timer

import numpy as np
from qtpy import QtCore, QtGui, QtWidgets

from frhodo.gui.widgets import (
    mech_widget,
    misc_widget,
    save_output,
    series_viewer_widget,
    thermo_widget,
)
from frhodo.simulation.shock.reactor_output import (
    VARIANTS_BY_DISPLAY,
    base_sim_name_for_display,
    sub_types_for_display,
)
from frhodo.simulation.shock.sensitivity import compute_sensitivity


SENSITIVITY_VARIANTS: dict[str, str] = {
    "Temperature Sensitivity Analysis": "T",
    "Pressure Sensitivity Analysis": "P",
    "Density Gradient Sensitivity Analysis": "drhodz_tot",
    "Mass Fraction Sensitivity Analysis": "Y",
    "Mole Fraction Sensitivity Analysis": "X",
    "Concentration Sensitivity Analysis": "conc",
}
_SPECIES_SENSITIVITIES = {"Y", "X", "conc"}


class SIM_Explorer_Widgets(QtCore.QObject):
    """X/Y/Y2 axis selector boxes for the Sim Explorer tab.

    Owns the three pairs of (main, sub) dropdowns, populates them
    from the active :class:`ReactorOutput`, and caches sensitivity
    calculations so switching observables doesn't recompute.
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.parent.plot_tab_widget.currentChanged.connect(self.tab_changed)

        self.max_history = 9
        self.widget = []
        self.updating_boxes = False

        # Cache keyed by (gas_id, sim_id, observable, species_idx) →
        # (t_arr, sens_arr). gas_id and sim_id invalidate on mech reload
        # or rerun; observable / species_idx separate variants so
        # switching dropdowns doesn't recompute already-cached cases.
        self._sensitivity_cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}

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
        sim = self.parent.SIM
        if sim is not None and len(sim.reactor_var) > 0:
            reactor_vars = deepcopy(list(sim.reactor_var.keys()))
        else:
            reactor_vars = list(VARIANTS_BY_DISPLAY)

        # create list of comboboxes
        if comboBoxes is None:
            comboBoxes = [boxes[0] for boxes in self.widget]
        else:
            comboBoxes = [comboBoxes]

        for comboBox in comboBoxes:
            local_vars = list(reactor_vars)
            if comboBox.info["axis"] == "y2":
                local_vars.insert(0, "-")
            if comboBox.info["axis"] in ("y", "y2"):
                local_vars.extend(SENSITIVITY_VARIANTS.keys())

            choice = comboBox.currentText()

            comboBox.blockSignals(True)
            comboBox.clear()
            comboBox.addItems(local_vars)
            comboBox.blockSignals(False)
            comboBox.itemList = local_vars

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
                choice not in local_vars
            ):  # defaults to temperature if variable not found
                choice = "Temperature"
            comboBox.setCurrentText(choice)
            comboBox.setCurrentIndex(local_vars.index(choice))

        # keeps selected options
        tab_widget_idx = self.parent.plot_tab_widget.currentIndex()
        if self.parent.plot_tab_widget.tabText(tab_widget_idx) == "Sim Explorer":
            self.update_plot(self.parent.SIM)

    def main_parameter_changed(self, event):
        if not self.parent.load_state.mech_loaded:
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
        elif event in SENSITIVITY_VARIANTS:
            sub_type = ["rxn"]
            subComboBox.setEnabled(True)
            itemMemory = deepcopy(subComboBox.info["itemMemory"]["rxn"])
            rxn_strings = self.parent.mech.gas.reaction_equations()
            items = [
                "R{:d}:  {:s}".format(n + 1, rxn.replace("<=>", "="))
                for n, rxn in enumerate(rxn_strings)
            ]
            subComboBox.addItems(items)
            for n, text in enumerate(items):
                stripped = text.split(":")[1].lstrip() if ":" in text else text
                if itemMemory["selected"] == stripped:
                    prior["selected"] = n
                if any(value == stripped for value in itemMemory["checked"]):
                    prior["checked"].append(n)
        else:
            sub_type = sub_types_for_display(event)

            if sub_type is None or not hasattr(self.parent, "mech"):
                subComboBox.setEnabled(False)
            else:
                subComboBox.setEnabled(True)

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
            sub_type = self._sub_type_for(main_choice)
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
        sub_type = self._sub_type_for(main_choice)

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
        if not self.parent.load_state.mech_loaded:
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

            if main_choice == "-":
                continue

            if main_choice in SENSITIVITY_VARIANTS:
                sens_result = self._compute_sensitivity_if_needed(SIM, main_choice)
                if sens_result is None:
                    continue
                _, sens = sens_result

                if self.widget[n][1].count() > 0:
                    sub_choice = sub_choice[sub_choice != selected_sub_choice].astype(int)
                    sub_choice = np.append(selected_sub_choice, np.sort(sub_choice))
                else:
                    sub_choice = []

                for rxn_idx in sub_choice:
                    data[axis].append(sens[:, int(rxn_idx)])

                if len(itemList) > 0 and len(sub_choice) > 0:
                    label[axis] = np.array(itemList[sub_choice])
                    label_order[axis] = np.argsort(sub_choice)
                continue

            SIM_name = base_sim_name_for_display(main_choice)
            sub_type = sub_types_for_display(main_choice)

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
                    for n, choice in enumerate(sub_choice):
                        token = itemList[choice].split(":  ")
                        val = "{:>8.1f}".format(data[axis][n][0])
                        labels.append(
                            "{:s}:  dH(0) = {:s},  {:s}".format(
                                token[0], val, token[1]
                            )
                        )
                elif SIM_name == "rate_con_rev":
                    labels = []
                    for n, choice in enumerate(sub_choice):
                        token = itemList[choice].split(":  ")
                        val = "{:>8.5e}".format(data[axis][n][0])
                        labels.append(
                            "{:s}:  kr(0) = {:s},  {:s}".format(
                                token[0], val, token[1]
                            )
                        )
                else:
                    if "total" in sub_type:
                        sub_choice = sub_choice + 1
                    labels = itemList[sub_choice]

                label[axis] = np.array(labels)
                label_order[axis] = np.argsort(sub_choice)

        self.parent.plot.sim_explorer.update(data, label, label_order)

    def _sub_type_for(self, main_choice):
        if main_choice in SENSITIVITY_VARIANTS:
            return ["rxn"]

        return sub_types_for_display(main_choice)

    def _sensitivity_selected(self) -> bool:
        return any(
            boxes[0].currentText() in SENSITIVITY_VARIANTS
            for boxes in self.widget
        )

    def _sim_explorer_active(self) -> bool:
        idx = self.parent.plot_tab_widget.currentIndex()

        return self.parent.plot_tab_widget.tabText(idx) == "Sim Explorer"

    def _sensitivity_species_idx(self) -> int:
        """The species index used for species-keyed sensitivity variants.

        Prefers the first species checked in any axis whose itemMemory
        carries a species selection (the user's most recent intent).
        Falls back to species 0 when nothing is selected.
        """
        species_names = self.parent.mech.gas.species_names
        for boxes in self.widget:
            memory = boxes[0].info.get("itemMemory", {}).get("species", {})
            for picked in memory.get("checked", []):
                if picked in species_names:
                    return species_names.index(picked)
            selected = memory.get("selected")
            if selected in species_names:
                return species_names.index(selected)

        return 0

    def _compute_sensitivity_if_needed(
        self, SIM, variant_label: str,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Return (t, sens) for the requested sensitivity variant,
        computing on cache miss. Returns None when the Sim Explorer tab
        isn't active so the (expensive) sensitivity reactor never runs
        from elsewhere.
        """
        if not self._sim_explorer_active():
            return None
        if SIM is None or not getattr(SIM, "success", False):
            return None

        observable = SENSITIVITY_VARIANTS[variant_label]
        species_idx = (
            self._sensitivity_species_idx() if observable in _SPECIES_SENSITIVITIES
            else None
        )

        gas = self.parent.mech.gas
        cache_key = (id(gas), id(SIM), observable, species_idx)
        cached = self._sensitivity_cache.get(cache_key)
        if cached is not None:
            return cached

        shock = self.parent.display_shock
        t_grid = SIM.t_lab(units="SI") if hasattr(SIM, "t_lab") else None

        t, sens = compute_sensitivity(
            self.parent.mech,
            reactor_state=self.parent.reactor_state,
            shock=shock,
            observable=observable,
            species_idx=species_idx,
            time_grid=t_grid,
            method="auto",
            n_workers=max(1, mp.cpu_count() // 2),
        )
        self._sensitivity_cache[cache_key] = (t, sens)

        return t, sens
