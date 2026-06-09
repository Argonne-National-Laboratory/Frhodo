# This file is part of Frhodo. Copyright © 2020, UChicago Argonne, LLC
# and licensed under BSD-3-Clause. See License.txt in the top-level
# directory for license and copyright information.
"""Mechanism tree view: reaction rows, editable coefficients, uncertainty boxes.

Defines the Qt model/view scaffolding behind the Mechanism tab. The
top-level controller is :class:`Tree`; reaction-row helpers
(:class:`rateExpCoefficient`, :class:`Uncertainty`, ...) live further
down and are instantiated by :class:`Tree` as the user expands the
tree.
"""
import sys, ast, re
from copy import deepcopy
from functools import partial
from timeit import default_timer as timer
from typing import Any, Protocol

import cantera as ct
import numpy as np
from scipy.optimize import root_scalar
from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import QSortFilterProxyModel
from qtpy.QtGui import QStandardItem
from qtpy.QtWidgets import (
    QAction, QGridLayout, QHeaderView, QLabel, QLineEdit,
    QMenu, QToolTip, QWidget,
)

from frhodo.common.units import PRESSURE_UNITS, pa_per_unit
from frhodo.gui.state import LoadState, RunControlState
from frhodo.gui.widgets import misc_widget
from frhodo.simulation.mechanism import ChemicalMechanism
from frhodo.simulation.mechanism.coef_helpers import arrhenius_coefNames
from frhodo.simulation.mechanism.mech_fcns import _FALLOFF_FAMILY
from frhodo.simulation.mechanism.mech_snapshot import (
    capture_state,
    restore_state,
    signatures_for_gas,
)


class MechWidgetHost(Protocol):
    """Subset of the ``Main`` window's surface that :class:`Tree` reads.

    Documents the contract between the mech-tree widget and its parent
    so tests can build minimal stubs and so future renames break loudly
    at type-check time. ``Main`` satisfies this structurally.
    """
    mech: ChemicalMechanism
    optimizables: Any                  # OptimizableSetBuilder
    load_state: LoadState
    run_control: RunControlState
    rxn_change_history: list[int]
    display_shock: Any
    reactor_state: Any
    convert_units: Any
    series: Any
    mech_tree: Any                     # QTreeView with .rxn list of dicts
    mech_tree_filter_box: Any
    tab_select_comboBox: Any
    num_sim_lines_box: Any
    log: Any                           # has ``.append(msg, alert=...)``
    plot: Any                          # has ``.signal.update_history``
    tree: Any                          # back-ref to the Tree instance
    clipboard: Any

    def run_single(self, *, rxn_changed: bool = False) -> None: ...


default_coef_abbreviation = {
    "pre_exponential_factor": "A",
    "temperature_exponent": "n",
    "activation_energy": "Ea",
}

coef_abbreviation = {key: default_coef_abbreviation[key] for key in arrhenius_coefNames}

# Display order for the mech table: A, n, Ea. Falloff repeats the same
# ordering for the low_rate then high_rate limb.
_ARRHENIUS_DISPLAY_ORDER = (
    "pre_exponential_factor",
    "temperature_exponent",
    "activation_energy",
)


def _display_indices(rate_count: int) -> list[int]:
    """Indices into the per-rxn coefficient list, ordered for table display.

    ``rate_count`` is 1 for Arrhenius (3 coefficients total) and 2 for
    falloff (6 coefficients: 3 per limb).
    """
    base = [arrhenius_coefNames.index(name) for name in _ARRHENIUS_DISPLAY_ORDER]

    return [b + 3 * r for r in range(rate_count) for b in base]


_ARRHENIUS_DISPLAY_IDX = _display_indices(1)  # [1, 2, 0]
_FALLOFF_DISPLAY_IDX = _display_indices(2)    # [1, 2, 0, 4, 5, 3]


def _arrhenius_coef_rows(rxnIdx, mech):
    return [
        [abbr, coefName, mech.coeffs[rxnIdx][0]]
        for coefName, abbr in coef_abbreviation.items()
    ]


def _pressure_dep_coef_rows(rxnIdx, rxn, mech):
    coeffs = []
    for key in ("high", "low"):
        if isinstance(rxn.rate, ct.PlogRate):
            n = len(mech.coeffs[rxnIdx]) - 1 if key == "high" else 0
        else:
            n = f"{key}_rate"
        for coefName, abbr in coef_abbreviation.items():
            coeffs.append([f"{abbr}_{key}", coefName, mech.coeffs[rxnIdx][n]])

    return coeffs


# Rate-level optimization is supported for every reaction type whose fit
# path the orchestrator understands: native Arrhenius / Plog / Falloff
# variants, Three Body Arrhenius, and Chebyshev (recast to Troe before
# fitting). Coefficient-level editing additionally requires a per-coef
# widget layout, which Chebyshev's 2D matrix doesn't have.
RATE_OPTIMIZABLE_TYPES = (
    "Arrhenius Reaction",
    "Three Body Reaction",
    "Plog Reaction",
    "Falloff Reaction",
    "Chebyshev Reaction",
)
COEF_OPTIMIZABLE_TYPES = (
    "Arrhenius Reaction",
    "Three Body Reaction",
    "Plog Reaction",
    "Falloff Reaction",
)
PRESSURE_DEPENDENT_TYPES = (
    "Plog Reaction",
    "Falloff Reaction",
    "Chebyshev Reaction",
)


def _rxn_type_tags(rxn_type: str, rate_class: str) -> frozenset:
    """Lowercase tokens that match the rxn in the tree filter's ``type:X``
    syntax. ``rxn_type`` is the broad category (``Arrhenius Reaction``,
    ``Falloff Reaction``, ...); ``rate_class`` is the Cantera rate class
    name (``TroeRate``, ``LindemannRate``, ...) so the user can filter
    sub-variants of ``Falloff Reaction`` separately."""
    tags: set[str] = set()
    if rxn_type == "Arrhenius Reaction":
        tags.add("arrhenius")
    elif rxn_type == "Three Body Reaction":
        tags.update({"arrhenius", "three-body", "threebody"})
    elif rxn_type == "Plog Reaction":
        tags.update({"plog", "pressure-sensitive"})
    elif rxn_type == "Falloff Reaction":
        tags.update({"falloff", "pressure-sensitive"})
        sub = {
            "LindemannRate": "lindemann",
            "TroeRate": "troe",
            "SriRate": "sri",
            "TsangRate": "tsang",
        }.get(rate_class)
        if sub:
            tags.add(sub)
    elif rxn_type == "Chebyshev Reaction":
        tags.update({"chebyshev", "pressure-sensitive"})

    return frozenset(tags)


_TYPE_TAG_ROLE = QtCore.Qt.UserRole + 1
_TYPE_TOKEN_RE = re.compile(r"^type:(\S+)$", re.IGNORECASE)


def silentSetValue(obj, value):
    """Set a Qt widget's value without re-firing its change signals."""
    obj.blockSignals(True)
    obj.setValue(value)
    obj.blockSignals(False)


def keysFromBox(box, mech):
    """Resolve ``(coef_key, bnds_key)`` from an uncertainty/coef box's metadata."""
    coefAbbr, rxnIdx = box.info["coefAbbr"], box.info["rxnNum"]
    rxn = mech.gas.reactions()[rxnIdx]

    return mech.get_coeffs_keys(rxn, coefAbbr, rxnIdx=rxnIdx)


_RXN_WIDGET_KEYS = ("coef", "rateBox", "formulaBox", "valueBox", "uncBox")


def _clear_rate_slots(rxn_dict: dict) -> None:
    """Drop every widget ref from ``rxn_dict``. Wired to the rxnRate
    widget's ``destroyed`` signal — the per-rxn widgets all share one
    parent ``QStandardItem``, so the rxnRate teardown is the latest
    we can reliably hear that they've all gone away."""
    for key in _RXN_WIDGET_KEYS:
        rxn_dict.pop(key, None)


def _clear_coef_slots(rxn_dict: dict, coef_idx: int) -> None:
    """Null one coefficient's widget refs in ``rxn_dict``. Wired to the
    coefficient widget's ``destroyed`` signal so the slot is dead-but-
    addressable in the window before :func:`_clear_rate_slots` pops the
    keys entirely (signal-order is not guaranteed)."""
    for key in ("formulaBox", "valueBox"):
        boxes = rxn_dict.get(key)
        if isinstance(boxes, list) and coef_idx < len(boxes):
            boxes[coef_idx] = None
    uncs = rxn_dict.get("uncBox")
    if isinstance(uncs, list) and coef_idx + 1 < len(uncs):
        uncs[coef_idx + 1] = None


class RecastPressureDialog(QtWidgets.QDialog):
    """Pick a pressure + unit to recast one pdep reaction to Arrhenius.

    Mirrors the Save dialog's recast inputs (value box + unit dropdown).
    The value is interpreted in the selected unit; :meth:`pressure_pa`
    returns it in pascals.
    """

    def __init__(self, parent, *, default_value=1.0, default_unit="atm", eqn=""):
        super().__init__(parent)
        self.setWindowTitle("Convert to Arrhenius at pressure")

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QLabel(f"Recast {eqn} to Arrhenius at:"))

        row = QtWidgets.QHBoxLayout()
        self.value_box = QtWidgets.QDoubleSpinBox()
        self.value_box.setRange(1.0e-6, 1.0e6)
        self.value_box.setDecimals(4)
        self.unit_box = QtWidgets.QComboBox()
        self.unit_box.addItems(PRESSURE_UNITS)
        if default_unit not in PRESSURE_UNITS:
            default_unit = "atm"
        self.unit_box.setCurrentText(default_unit)
        if default_value is not None and np.isfinite(default_value) and default_value > 0:
            self.value_box.setValue(default_value)
        else:
            self.value_box.setValue(1.0)
        row.addWidget(self.value_box)
        row.addWidget(self.unit_box)
        row.addStretch(1)
        layout.addLayout(row)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def pressure_pa(self):
        unit = self.unit_box.currentText()
        pressure_pa = self.value_box.value() * pa_per_unit[unit]

        return pressure_pa


class Tree(QtCore.QObject):
    """Controller for the Mechanism tab's reaction tree view.

    Owns the Qt model + proxy + filter, builds and rebuilds rows from
    a :class:`ChemicalMechanism`, and routes per-row edits back into
    the mech's coefficient state. ``run_sim_on_change`` is the master
    flag that lets bulk loads suppress per-edit re-simulations.
    """

    def __init__(self, parent: MechWidgetHost):
        super().__init__(parent)
        self.run_sim_on_change = True
        self.copyRates = False
        self.convert = parent.convert_units

        self.timer = QtCore.QTimer()

        self._pending_render: dict | None = None
        self._render_timer = QtCore.QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.timeout.connect(self._drain_pending_render)

        self.model = QtGui.QStandardItemModel(parent.mech_tree)
        self.proxy_model = QSortFilterProxyModel(parent.mech_tree)
        self.proxy_model.setSourceModel(self.model)
        parent.mech_tree.setModel(self.proxy_model)
        self.tree_filter = TreeFilter(
            parent, self.proxy_model, self.model, self._set_mech_widgets
        )

        self.color = {
            "variable_rxn": QtGui.QBrush(QtGui.QColor(188, 0, 188)),
            "fixed_rxn": QtGui.QBrush(QtGui.QColor(0, 0, 0)),
            "partial_match_rxn": QtGui.QBrush(QtGui.QColor(0x6E, 0, 0)),
        }
        self._partial_match_idxs: set = set()

        parent.mech_tree.setRootIsDecorated(False)
        parent.mech_tree.setIndentation(21)
        parent.mech_tree.setExpandsOnDoubleClick(False)
        parent.mech_tree.clicked.connect(self.item_clicked)

        # Set up right click popup menu and linked expand/collapse
        parent.mech_tree.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        parent.mech_tree.customContextMenuRequested.connect(self._popup_menu)
        parent.mech_tree.expanded.connect(
            lambda sender: self._tabExpanded(sender, True)
        )
        parent.mech_tree.collapsed.connect(
            lambda sender: self._tabExpanded(sender, False)
        )

    def item_clicked(self, event):
        ix = self.proxy_model.mapToSource(event)
        item = self.model.itemFromIndex(ix)
        if not hasattr(item, "info"):
            return

        rxnNum = item.info["rxnNum"]
        tree = self.parent().mech_tree
        if tree.isExpanded(event):
            tree.collapse(event)
            self._tabExpanded(event, expanded=False)
            item.info["isExpanded"] = False
        else:
            if not hasattr(item, "info"):
                return  # skip if not a reaction, hence no info
            if not item.info["hasExpanded"]:  # forward event if tab has never expanded
                self._tabExpanded(event, expanded=True)
            else:
                self._set_mech_widgets(item)
            tree.expand(event)
            item.info["isExpanded"] = True

    def set_trees(self, mech, partial_match_idxs=None):
        parent = self.parent()
        self._partial_match_idxs = set(partial_match_idxs or ())
        self.model.removeRows(0, self.model.rowCount())
        if "Chemkin" in parent.tab_select_comboBox.currentText():
            self.mech_tree_type = "Chemkin"
        else:
            self.mech_tree_type = "Bilbo"
        self.mech_tree_data = self._set_mech_tree_data(self.mech_tree_type, mech)
        self._set_mech_tree(self.mech_tree_data)
        self._apply_initial_foreground()

    def _apply_initial_foreground(self):
        """Color each row at tree-build time so the user sees optimizable
        state without having to expand the row.

        Dark red wins over purple — partial-match flags a coef mismatch
        that needs review even when the rate-level restore made the rxn
        optimizable.
        """
        for i in range(self.model.rowCount()):
            item = self.model.item(i, 0)
            if not hasattr(item, "info"):
                continue
            self._paint_rxn_foreground(item.info["rxnNum"], item=item)

    def _paint_rxn_foreground(self, rxnNum, *, item=None):
        """Apply text color for one rxn based on current state.

        Priority: partial-match (dark red) > optimizable (purple) >
        default (black).
        """
        parent = self.parent()
        if item is None:
            row_dict = parent.mech_tree.rxn[rxnNum]
            item = row_dict.get("item")
            if item is None:
                return
        if rxnNum in self._partial_match_idxs:
            item.setForeground(self.color["partial_match_rxn"])
        elif parent.optimizables.is_reaction_optimizable(rxnNum):
            item.setForeground(self.color["variable_rxn"])
        else:
            item.setForeground(self.color["fixed_rxn"])

    def _clear_partial_on_user_interaction(self, rxnNum):
        """User touched a box on a partial-match rxn — drop the flag and
        let the normal optimizable/fixed coloring take over.

        Skipped during programmatic widget construction (the
        ``_building_widgets`` guard) so pre-building widgets for a
        previously-opened partial rxn doesn't undo the dark-red flag
        before the user has seen it.
        """
        if getattr(self, "_building_widgets", False):
            return
        if rxnNum not in self._partial_match_idxs:
            return
        self._partial_match_idxs.discard(rxnNum)
        self._paint_rxn_foreground(rxnNum)

    def snapshot_for_reload(self):
        """Snapshot enough state to rebuild the tree after the mech loader
        replaces the underlying :class:`ChemicalMechanism`.

        Returns ``None`` if nothing is loaded yet. Otherwise returns a
        dict with three keys: ``"mech"`` (engine-side snapshot from
        :func:`mech_snapshot.capture_state`), ``"opened"`` (set of
        per-rxn signatures for rows the user expanded at least once, so
        their widgets get pre-built), and ``"expanded"`` (subset of
        ``opened`` that was visually expanded at snapshot time so the
        reload restores the visual collapsed/expanded state per-rxn).
        """
        parent = self.parent()
        mech = parent.mech
        if not parent.load_state.mech_loaded:
            return None

        sigs = signatures_for_gas(mech.gas)
        opened: set = set()
        expanded: set = set()
        for i in range(self.model.rowCount()):
            item = self.model.item(i, 0)
            if not hasattr(item, "info") or not item.info.get("hasExpanded"):
                continue
            if i >= len(sigs):
                continue
            sig = sigs[i]
            opened.add(sig)
            proxy_idx = self.proxy_model.mapFromSource(item.index())
            if parent.mech_tree.isExpanded(proxy_idx):
                expanded.add(sig)

        return {
            "mech": capture_state(mech, parent.optimizables),
            "opened": opened,
            "expanded": expanded,
        }

    def handle_reload(self, prior_snapshot):
        """Refresh the tree after a mech reload.

        Called by ``app.load_mech`` after the loader has populated the
        ``ChemicalMechanism``. Two modes:

        * ``prior_snapshot is None``: first load (or post-error). Just
          rebuild the tree.
        * ``prior_snapshot`` present: clear optimizables, restore engine
          state per matched signature, rebuild the tree with partial-
          match flags, and pre-build widgets for rxns the user had
          opened so the restored uncertainty boxes are ready (rows stay
          collapsed; user expands to see).
        """
        parent = self.parent()
        if prior_snapshot is None:
            self.set_trees(parent.mech)
            return

        parent.optimizables.reset()
        restored_idxs, partial_idxs = restore_state(
            parent.mech, parent.optimizables, prior_snapshot["mech"],
        )
        self.set_trees(parent.mech, partial_match_idxs=partial_idxs)
        opened = prior_snapshot["opened"]
        expanded = prior_snapshot.get("expanded", set())
        if not opened:
            return

        sigs = signatures_for_gas(parent.mech.gas)
        for i, sig in enumerate(sigs):
            if i not in restored_idxs or sig not in opened:
                continue

            item = self.model.item(i, 0)
            if not hasattr(item, "info") or item.info.get("hasExpanded"):
                continue

            item.info["hasExpanded"] = True
            self._set_mech_widgets(item)

            if sig in expanded:
                proxy_idx = self.proxy_model.mapFromSource(item.index())
                parent.mech_tree.expand(proxy_idx)
                item.info["isExpanded"] = True

        self._apply_initial_foreground()

    def _set_mech_tree_data(self, selection, mech):
        data = []
        for rxnIdx, rxn in enumerate(mech.gas.reactions()):
            rxn_type = mech.reaction_type(rxn)
            rate_class = type(rxn.rate).__name__
            entry = {
                "num": rxnIdx, "eqn": rxn.equation,
                "type": rxn_type, "rateClass": rate_class,
            }
            if isinstance(rxn.rate, ct.ArrheniusRate):
                entry["coeffs"] = _arrhenius_coef_rows(rxnIdx, mech)
                entry["coeffs_order"] = _ARRHENIUS_DISPLAY_IDX
            elif isinstance(rxn.rate, (ct.PlogRate, *_FALLOFF_FAMILY)):
                entry["coeffs"] = _pressure_dep_coef_rows(rxnIdx, rxn, mech)
                entry["coeffs_order"] = _FALLOFF_DISPLAY_IDX
            data.append(entry)

        return data

    def _set_mech_tree(self, rxn_matrix):
        parent = self.parent()
        tree = parent.mech_tree
        self.model.setHorizontalHeaderLabels(["Reaction"])

        tree.setUpdatesEnabled(False)
        tree.rxn = []
        for rxn in rxn_matrix:
            L1 = QtGui.QStandardItem(
                f" R{rxn['num']+1:d}:   {rxn['eqn'].replace('<=>', '=')}"
            )
            L1.setEditable(False)
            L1.setToolTip(rxn["type"])
            L1.setData(
                _rxn_type_tags(rxn["type"], rxn.get("rateClass", "")),
                _TYPE_TAG_ROLE,
            )
            L1.info = {
                "tree": tree.objectName(),
                "type": "rxn tab",
                "rxnNum": rxn["num"],
                "rxnType": rxn["type"],
                "hasExpanded": False,
                "isExpanded": False,
                "rxn_details": rxn,
                "row": [],
            }
            self.model.appendRow(L1)
            tree.rxn.append(
                {
                    "item": L1,
                    "num": rxn["num"],
                    "rxnType": rxn["type"],
                    "dependent": False,
                }
            )

        tree.setUpdatesEnabled(True)
        tree.sortByColumn(0, QtCore.Qt.AscendingOrder)
        tree.header().setStretchLastSection(True)
        tree.header().setSectionResizeMode(
            0, QHeaderView.ResizeToContents
        )

    def _set_mech_widgets(self, sender):
        L1 = sender
        rxn = L1.info["rxn_details"]
        rxnNum = rxn["num"]
        parent = self.parent()
        tree = parent.mech_tree
        self._building_widgets = True

        L1.removeRows(0, L1.rowCount())

        if rxn["type"] in COEF_OPTIMIZABLE_TYPES:
            self._build_coef_rxn_row(L1, rxn, parent, tree)
        elif rxn["type"] == "Chebyshev Reaction":
            self._build_rate_only_row(L1, rxn, parent, tree, with_unc=True)
        else:
            self._build_rate_only_row(L1, rxn, parent, tree, with_unc=False)

        self.update_box_reset_values(rxnNum)
        self.update_rates(rxnNum)
        self._building_widgets = False

    def _attach_rate_widget(self, L1, rxn, parent, tree, unc=None):
        """Build the per-rxn rate-uncertainty header widget."""
        info = {"type": "rateUnc", "rxnNum": rxn["num"]}
        if unc is None:
            widget = rxnRate(parent, info, rxnType=rxn["type"])
        else:
            widget = rxnRate(
                parent, info, rxnType=rxn["type"],
                unc_type=unc["type"], unc_value=unc["value"],
            )
        L1.info["row"].append({"item": QtGui.QStandardItem(""), "widget": widget})
        L1.appendRow([L1.info["row"][-1]["item"]])
        mIndex = self.proxy_model.mapFromSource(L1.info["row"][-1]["item"].index())
        tree.setIndexWidget(mIndex, widget)

        return widget

    def _build_coef_rxn_row(self, L1, rxn, parent, tree):
        rxnNum = rxn["num"]
        widget = self._attach_rate_widget(
            L1, rxn, parent, tree,
            unc={
                "type": parent.mech.rate_bnds[rxnNum]["type"],
                "value": parent.mech.rate_bnds[rxnNum]["value"],
            },
        )
        widget.uncValBox.valueChanged.connect(self.update_uncertainties)

        len_coef = len(rxn["coeffs_order"])
        rxn_dict = tree.rxn[rxnNum]
        rxn_dict.update({
            "coef": rxn["coeffs"],
            "rateBox": widget.valueBox,
            "formulaBox": [None] * len_coef,
            "valueBox": [None] * len_coef,
            "uncBox": [widget.uncValBox] + [None] * len_coef,
        })
        widget.destroyed.connect(lambda _, d=rxn_dict: _clear_rate_slots(d))

        for coefNum in rxn["coeffs_order"]:
            self._attach_coef_widget(L1, rxn, coefNum, rxn_dict, parent, tree)

    def _attach_coef_widget(self, L1, rxn, coefNum, rxn_dict, parent, tree):
        rxnNum = rxn["num"]
        conv_type = f"Cantera2{self.mech_tree_type}"

        coef = deepcopy(rxn["coeffs"][coefNum])
        coef[2] = coef[2][coef[1]]
        coef = self.convert._arrhenius(rxnNum, [coef], conv_type)[0]

        if rxn["type"] in ("Arrhenius Reaction", "Three Body Reaction"):
            bnds_key = "rate"
        elif "high" in coef[0]:
            bnds_key = "high_rate"
        else:
            bnds_key = "low_rate"

        unc_type = parent.mech.coeffs_bnds[rxnNum][bnds_key][coef[1]]["type"]
        unc_value = parent.mech.coeffs_bnds[rxnNum][bnds_key][coef[1]]["value"]
        if unc_type not in ("F", "%"):
            unc_value = self.convert._arrhenius(
                rxnNum, [[*coef[:2], unc_value]], conv_type,
            )[0][2]

        info = {
            "type": rxn["type"], "rxnNum": rxnNum, "coefNum": coefNum, "label": "",
            "coef": coef[0:2], "coefAbbr": coef[0],
            "coefName": coef[1], "coefVal": coef[2],
        }
        widget = rateExpCoefficient(
            parent, coef, info, unc_type=unc_type, unc_value=unc_value,
        )
        widget.Label.setToolTip(coef[1].replace("_", " ").title())
        if self.mech_tree_type == "Bilbo":
            widget.valueBox.setSingleStep(0.01)
        elif self.mech_tree_type == "Chemkin":
            widget.valueBox.setSingleStep(0.1)

        widget.formulaBox.setInitialFormula()
        widget.formulaBox.valueChanged.connect(self.update_value)
        widget.valueBox.valueChanged.connect(self.update_value)
        widget.valueBox.resetValueChanged.connect(self.update_mech_reset_value)
        widget.uncValBox.valueChanged.connect(self.update_uncertainties)

        rxn_dict["formulaBox"][coefNum] = widget.formulaBox
        rxn_dict["valueBox"][coefNum] = widget.valueBox
        rxn_dict["uncBox"][coefNum + 1] = widget.uncValBox
        widget.destroyed.connect(
            lambda _, d=rxn_dict, c=coefNum: _clear_coef_slots(d, c),
        )

        L1.info["row"].append({"item": QtGui.QStandardItem(""), "widget": widget})
        L1.appendRow([L1.info["row"][-1]["item"]])
        mIndex = self.proxy_model.mapFromSource(L1.info["row"][-1]["item"].index())
        tree.setIndexWidget(mIndex, widget)

    def _build_rate_only_row(self, L1, rxn, parent, tree, *, with_unc: bool):
        """Rate-level widget only — no per-coef edits. Used by Chebyshev
        (which is recast to Troe before fitting) and unsupported types.
        """
        rxnNum = rxn["num"]
        if with_unc:
            widget = self._attach_rate_widget(
                L1, rxn, parent, tree,
                unc={
                    "type": parent.mech.rate_bnds[rxnNum]["type"],
                    "value": parent.mech.rate_bnds[rxnNum]["value"],
                },
            )
            widget.uncValBox.valueChanged.connect(self.update_uncertainties)
            unc_box = widget.uncValBox
        else:
            widget = self._attach_rate_widget(L1, rxn, parent, tree)
            unc_box = None

        rxn_dict = tree.rxn[rxnNum]
        rxn_dict.update({
            "coef": [], "rateBox": widget.valueBox,
            "formulaBox": [None], "valueBox": [None],
            "uncBox": [unc_box],
        })
        widget.destroyed.connect(lambda _, d=rxn_dict: _clear_rate_slots(d))

    def currentRxn(self):
        tree = self.parent().mech_tree
        sender_idx = tree.selectedIndexes()[0]
        ix = self.proxy_model.mapToSource(sender_idx)
        selected = self.model.itemFromIndex(ix)
        if hasattr(selected, "info"):
            rxnNum = selected.info["rxnNum"]
            return tree.rxn[rxnNum]
        else:
            return None

    def _rxn_at_cursor(self):
        """Resolve the reaction under the mouse cursor, or ``None``.

        The context menu resolves its reaction from the cursor position
        so a plain right-click (which need not move the selection) still
        targets the row under the pointer. A click on a child coefficient
        row walks up to its parent reaction.
        """
        tree = self.parent().mech_tree
        pos = tree.viewport().mapFromGlobal(QtGui.QCursor.pos())
        idx = tree.indexAt(pos)
        if not idx.isValid():
            return None

        ix = self.proxy_model.mapToSource(idx)
        item = self.model.itemFromIndex(ix)
        while item is not None and not hasattr(item, "info"):
            item = item.parent()
        if item is None:
            return None

        result = tree.rxn[item.info["rxnNum"]]

        return result

    def _schedule_render(self, rxnIdxs, *, run_sim=False):
        """Queue a coalesced render. Multiple calls in one event-loop tick
        collapse into a single ``modify_reactions`` + ``update_rates`` +
        ``run_single`` pass when the timer fires."""
        if not rxnIdxs and not run_sim:
            return
        if self._pending_render is None:
            self._pending_render = {"rxnIdxs": set(), "run_sim": False}
        self._pending_render["rxnIdxs"].update(int(i) for i in rxnIdxs)
        if run_sim:
            self._pending_render["run_sim"] = True
        if not self._render_timer.isActive():
            self._render_timer.start(0)

    def _drain_pending_render(self):
        if self._pending_render is None:
            return
        pending = self._pending_render
        self._pending_render = None
        parent = self.parent()
        if not parent.load_state.mech_loaded:
            return
        mech = parent.mech
        rxnIdxs = sorted(pending["rxnIdxs"])
        if rxnIdxs:
            mech.modify_reactions(mech.coeffs, rxnIdxs=rxnIdxs)
        if pending["run_sim"]:
            parent.run_single(rxn_changed=True)
        elif rxnIdxs:
            self.update_rates(rxnNum=rxnIdxs)

    def _clip_coef_to_limits(self, rxnNum, coef_key, bnds_key, coefName, coefBox):
        """Clip ``mech.coeffs[rxnNum][coef_key][coefName]`` into the
        coef's current bound interval and refresh the spinbox display.

        Returns ``True`` when a clip happened.
        """
        parent = self.parent()
        mech = parent.mech
        coefValue = mech.coeffs[rxnNum][coef_key][coefName]
        limits = mech.coeffs_bnds[rxnNum][bnds_key][coefName]["limits"]()
        if np.isnan(limits).all():
            return False
        clipped = float(np.clip(coefValue, limits[0], limits[1]))
        if clipped == coefValue:
            return False
        mech.coeffs[rxnNum][coef_key][coefName] = clipped
        conv_type = "Cantera2" + self.mech_tree_type
        coeffs = [*coefBox.info["coef"], clipped]
        display_val = self.convert._arrhenius(rxnNum, deepcopy([coeffs]), conv_type)[0][2]
        silentSetValue(coefBox, display_val)

        return True

    def update_value(self, event):
        def getRateConst(parent, rxnNum, coef_key, coefName, value):
            shock = parent.display_shock
            parent.mech.coeffs[rxnNum][coef_key][coefName] = value
            parent.mech.modify_reactions(parent.mech.coeffs, rxnIdxs=[rxnNum])
            mech_out = parent.mech.set_TPX(
                shock.T_reactor, shock.P_reactor, shock.thermo_mix,
            )
            if not mech_out["success"]:
                parent.log.append(mech_out["message"])
                return
            return parent.mech.gas.forward_rate_constants[rxnNum]

        parent = self.parent()
        sender = self.sender()
        mech = parent.mech
        rxnNum, coefNum, coefName = (
            sender.info["rxnNum"],
            sender.info["coefNum"],
            sender.info["coefName"],
        )

        # track changes to rxns
        if rxnNum in parent.rxn_change_history:
            parent.rxn_change_history.remove(rxnNum)
        parent.rxn_change_history.append(rxnNum)

        # Convert coeffs to cantera
        conv_type = self.mech_tree_type + "2Cantera"
        coeffs = [*sender.info["coef"], event]
        cantera_value = self.convert._arrhenius(rxnNum, deepcopy([coeffs]), conv_type)

        rateLimits = parent.display_shock.rate_bnds[rxnNum]
        coef_key, bnds_key = keysFromBox(sender, mech)

        coef_bnds_dict = mech.coeffs_bnds[rxnNum][bnds_key][coefName]
        coefLimits = coef_bnds_dict["limits"]()

        outside_limits = True
        if not np.isnan(
            coefLimits
        ).all():  # if coef limits exist, default to using these
            if cantera_value[0][2] < coefLimits[0]:
                cantera_value[0][2] = coefLimits[0]
            elif cantera_value[0][2] > coefLimits[1]:
                cantera_value[0][2] = coefLimits[1]
            else:
                outside_limits = False

            if outside_limits:
                conv_type = "Cantera2" + self.mech_tree_type
                coeffs = [*sender.info["coef"], cantera_value[0][2]]
                value = self.convert._arrhenius(rxnNum, deepcopy([coeffs]), conv_type)
                silentSetValue(sender, value[0][2])

        elif not np.isnan(
            rateLimits
        ).all():  # if rate limits exist, and coef limits do not
            rateCon = getRateConst(
                parent, rxnNum, coef_key, coefName, cantera_value[0][2]
            )

            if rateCon < rateLimits[0]:
                limViolation = 0
            elif rateCon > rateLimits[1]:
                limViolation = 1
            else:
                outside_limits = False

            if outside_limits:
                # Calculate correct coef value
                fcn = (
                    lambda x: getRateConst(parent, rxnNum, coef_key, coefName, x)
                    - rateLimits[limViolation]
                )
                x0 = coef_bnds_dict["resetVal"]
                if x0 == 0:
                    x1 = 1e-9
                else:
                    x1 = x0 * (1 - 1e-9)
                sol = root_scalar(fcn, x0=x0, x1=x1, method="secant")
                cantera_value[0][2] = sol.root

                # Update box
                conv_type = "Cantera2" + self.mech_tree_type
                coeffs = [*sender.info["coef"], cantera_value[0][2]]
                value = self.convert._arrhenius(rxnNum, deepcopy([coeffs]), conv_type)
                silentSetValue(sender, value[0][2])

        mech.coeffs[rxnNum][coef_key][coefName] = cantera_value[0][2]
        dependentRxnIdxs = self._updateDependents()
        rxnIdxs = {rxnNum, *dependentRxnIdxs}
        self._schedule_render(rxnIdxs, run_sim=self.run_sim_on_change)

    def update_rates(self, rxnNum=None):
        parent = self.parent()
        shock = parent.display_shock

        if not parent.load_state.mech_loaded:
            return

        if rxnNum is None:
            rxnNumRange = range(parent.mech.gas.n_reactions)
        elif isinstance(rxnNum, (list, tuple, np.ndarray, range)):
            rxnNumRange = list(rxnNum)
        else:
            rxnNumRange = [int(rxnNum)]

        if not rxnNumRange:
            return

        if rxnNum is None:
            rxn_rate = parent.series.rates(shock)
        else:
            rxn_rate = parent.series.rates(shock, rxnIdxs=rxnNumRange)
        if rxn_rate is None:
            return

        num_reac_all = np.sum(parent.mech.gas.reactant_stoich_coeffs, axis=0)
        for idx in rxnNumRange:
            rxn_row = parent.mech_tree.rxn[idx]
            if "rateBox" not in rxn_row:
                continue
            rxn_rate_box = rxn_row["rateBox"]
            if rxn_rate_box is None:
                continue
            conv = np.power(1e3, num_reac_all[idx] - 1)
            rxn_rate_box.setValue(np.multiply(rxn_rate[idx], conv))

        self._copy_expanded_tab_rates()

    def update_uncertainties(self, event=None, sender=None):
        parent = self.parent()
        mech = parent.mech

        # update uncertainty spinbox
        if event is not None:  # individual uncertainty is being updated
            if sender is None:
                sender = self.sender()

            rxnNum = sender.info["rxnNum"]
            self._clear_partial_on_user_interaction(rxnNum)
            if "coefName" in sender.info:  # this means the coef unc was changed
                coefNum, coefName, coefAbbr = (
                    sender.info["coefNum"],
                    sender.info["coefName"],
                    sender.info["coefAbbr"],
                )

                # get correct uncertainty diction based on reaction type
                coef_key, bnds_key = keysFromBox(sender, mech)

                rxn_dict = parent.mech_tree.rxn[rxnNum]
                uncBoxes = rxn_dict.get("uncBox")
                if uncBoxes is None or coefNum + 1 >= len(uncBoxes):
                    return
                uncBox = uncBoxes[coefNum + 1]
                if uncBox is None:
                    return

                coefUncDict = mech.coeffs_bnds[rxnNum][bnds_key][coefName]
                coefUncDict["value"] = uncVal = uncBox.uncValue
                coefUncDict["type"] = uncBox.uncType
                limits = coefUncDict["limits"]()

                opt = uncVal != uncBox.minimumBaseValue
                parent.optimizables.set_coefficient_optimizable(
                    rxnNum, bnds_key, coefName, opt,
                )

                coefValue = mech.coeffs[rxnNum][coef_key][coefName]
                if coefValue < limits[0] or coefValue > limits[1]:
                    valueBoxes = rxn_dict.get("valueBox")
                    if valueBoxes is None or coefNum >= len(valueBoxes):
                        return
                    coefBox = valueBoxes[coefNum]
                    if coefBox is not None and self._clip_coef_to_limits(
                        rxnNum, coef_key, bnds_key, coefName, coefBox,
                    ):
                        self._schedule_render(
                            [rxnNum], run_sim=self.run_sim_on_change,
                        )

                return
            else:
                rxnNumRange = [rxnNum]
        else:
            rxnNumRange = range(mech.gas.n_reactions)

        for rxnNum in rxnNumRange:  # update all rate uncertainties
            rxn = parent.mech_tree.rxn[rxnNum]
            if rxn["rxnType"] not in RATE_OPTIMIZABLE_TYPES:
                parent.optimizables.set_reaction_optimizable(rxnNum, False)
                continue
            if "uncBox" not in rxn:
                continue

            mech.rate_bnds[rxnNum]["value"] = uncVal = rxn["uncBox"][0].uncValue
            mech.rate_bnds[rxnNum]["type"] = rxn["uncBox"][0].uncType
            if (
                np.isnan(uncVal) or uncVal == rxn["uncBox"][0].minimumBaseValue
            ):  # not optimized
                parent.optimizables.set_reaction_optimizable(rxnNum, False)
                rxn["item"].setForeground(self.color["fixed_rxn"])
            else:
                parent.optimizables.set_reaction_optimizable(rxnNum, True)
                rxn["item"].setForeground(self.color["variable_rxn"])

        parent.series.rate_bnds(parent.display_shock)

    def update_coef_rate_from_opt(self, coef_opt, x):
        parent = self.parent()

        conv_type = "Cantera2" + self.mech_tree_type
        touched = set()
        for i, c in enumerate(coef_opt):
            rxnIdx, coefIdx = c.rxn_idx, c.coef_idx
            coeffs_key = c.coeffs_key
            touched.add(rxnIdx)

            if coeffs_key == "falloff_parameters":
                if isinstance(parent.mech.coeffs[rxnIdx][coeffs_key], tuple):
                    parent.mech.coeffs[rxnIdx][coeffs_key] = list(
                        parent.mech.coeffs[rxnIdx][coeffs_key]
                    )
                parent.mech.coeffs[rxnIdx][coeffs_key][coefIdx] = x[i]
                continue

            coefName = list(parent.mech.coeffs[rxnIdx][coeffs_key].keys())[coefIdx]
            parent.mech.coeffs[rxnIdx][coeffs_key][coefName] = x[i]
            coeffs = ["", coefName, x[i]]
            value = self.convert._arrhenius(rxnIdx, [coeffs], conv_type)
            if coeffs_key == "low_rate":
                coefIdx += 3
            valueBoxes = parent.mech_tree.rxn[rxnIdx].get("valueBox")
            if valueBoxes is None or coefIdx >= len(valueBoxes):
                continue
            coefBox = valueBoxes[coefIdx]
            if coefBox is None:
                continue
            silentSetValue(coefBox, value[0][2])

        if touched:
            indices = sorted(touched)
            parent.mech.modify_reactions(parent.mech.coeffs, rxnIdxs=indices)
            self.update_rates(rxnNum=indices)

    def update_mech_reset_value(self, event):
        parent = self.parent()
        sender = self.sender()
        rxnNum, coefName = sender.info["rxnNum"], sender.info["coefName"]

        coeffs = [*sender.info["coef"], event]
        conv_type = self.mech_tree_type + "2Cantera"
        cantera_value = self.convert._arrhenius(rxnNum, deepcopy([coeffs]), conv_type)

        _, bnds_key = keysFromBox(sender, parent.mech)
        parent.mech.coeffs_bnds[rxnNum][bnds_key][coefName]["resetVal"] = (
            cantera_value[0][2]
        )
        self.update_uncertainties(event, sender)

    def update_box_reset_values(self, rxnNum=None):
        parent = self.parent()
        mech = parent.mech
        conv_type = "Cantera2" + self.mech_tree_type

        if rxnNum is not None:
            if type(rxnNum) in [list, np.ndarray]:
                rxnNumRange = rxnNum
            else:
                rxnNumRange = [rxnNum]
        else:
            rxnNumRange = range(mech.gas.n_reactions)

        for rxnNum in rxnNumRange:
            rxn = parent.mech_tree.rxn[rxnNum]
            if (
                rxn["rxnType"] not in COEF_OPTIMIZABLE_TYPES
                or "valueBox" not in rxn
            ):
                continue

            valBoxes = parent.mech_tree.rxn[rxnNum]["valueBox"]
            for n, valBox in enumerate(valBoxes):
                if valBox is None:
                    continue
                coefName = valBox.info["coefName"]
                coef_key, bnds_key = keysFromBox(valBox, mech)

                resetVal = mech.coeffs_bnds[rxnNum][bnds_key][coefName]["resetVal"]
                coeffs = [*valBox.info["coef"], deepcopy(resetVal)]
                value = self.convert._arrhenius(rxnNum, [coeffs], conv_type)
                valBox.reset_value = value[0][2]

    def update_display_type(self):
        parent = self.parent()
        mech = parent.mech
        conv_type = f"Cantera2{self.mech_tree_type}"
        for rxnNum, rxn in enumerate(mech.coeffs):
            if "valueBox" not in parent.mech_tree.rxn[rxnNum]:
                continue

            valBoxes = parent.mech_tree.rxn[rxnNum]["valueBox"]
            uncBoxes = parent.mech_tree.rxn[rxnNum]["uncBox"]
            for n, valBox in enumerate(valBoxes):  # update value boxes
                if valBox is None:  # in case there is no valbox because not arrhenius
                    continue

                coefNum = valBox.info["coefNum"]
                coefName = valBox.info["coefName"]
                coef_key, bnds_key = keysFromBox(valBox, mech)
                coeffs = [*valBox.info["coef"], rxn[coef_key][coefName]]
                coeffs = self.convert._arrhenius(rxnNum, [coeffs], conv_type)[0]

                silentSetValue(valBox, coeffs[2])  # update value
                valBox.info["coefAbbr"] = f"{coeffs[0]}:"  # update abbreviation
                valBox.info["label"].setText(valBox.info["coefAbbr"])
                if self.mech_tree_type == "Bilbo":  # update step size
                    valBox.setSingleStep(0.01)
                elif self.mech_tree_type == "Chemkin":
                    valBox.setSingleStep(0.1)

                if n + 1 >= len(uncBoxes):
                    continue
                uncBox = uncBoxes[n + 1]
                if uncBox is None:
                    continue
                if uncBox.uncType == "±":
                    if uncBox.info["coef"][1] == "pre_exponential_factor":
                        continue

                    uncVal = mech.coeffs_bnds[rxnNum][bnds_key][coefName]["value"]
                    coeffs = [*valBox.info["coef"], deepcopy(uncVal)]
                    uncVal = self.convert._arrhenius(rxnNum, [coeffs], conv_type)[0][2]
                    silentSetValue(uncBox, uncVal)  # update value

        self.update_box_reset_values()
        dependentRxnIdxs = self._updateDependents()
        if dependentRxnIdxs:
            mech.modify_reactions(mech.coeffs, rxnIdxs=dependentRxnIdxs)
            self.update_rates(rxnNum=dependentRxnIdxs)

    def _tabExpanded(
        self, sender_idx, expanded
    ):  # set uncboxes to not set upon first expand
        parent = self.parent()
        ix = self.proxy_model.mapToSource(sender_idx)
        sender = self.model.itemFromIndex(ix)
        if hasattr(sender, "info"):
            rxnNum = sender.info["rxnNum"]
        else:
            return

        if expanded:
            if sender.info["hasExpanded"]:
                return
            else:
                sender.info["hasExpanded"] = True
                self._set_mech_widgets(sender)

            self._copy_expanded_tab_rates()
        else:
            if rxnNum in parent.rxn_change_history:
                parent.rxn_change_history.remove(rxnNum)
                if (
                    parent.num_sim_lines_box.value() > 1
                ):  # update history only if history tracked
                    parent.plot.signal.update_history()  # update plot history lines

    def _popup_menu(self, event):
        def setCopyRates(self, event):
            self.copyRates = event
            self._copy_expanded_tab_rates()

        tree = self.parent().mech_tree
        rxn = self._rxn_at_cursor()

        popup_menu = QMenu(tree)

        copyRatesAction = QAction("Auto Copy Rates", checkable=True)
        copyRatesAction.setChecked(self.copyRates)
        popup_menu.addAction(copyRatesAction)
        copyRatesAction.triggered.connect(lambda event: setCopyRates(self, event))

        popup_menu.addSeparator()
        popup_menu.addAction("Collapse All", lambda: tree.collapseAll())
        popup_menu.addSeparator()
        popup_menu.addAction("Reset All", lambda: self._reset_all())

        # this causes independent/dependent to not show if right click is not on rxn
        if rxn is not None and "Arrhenius Reaction" in rxn["rxnType"]:
            popup_menu.addSeparator()

            dependentAction = QAction("Set Dependent", checkable=True)
            dependentAction.setChecked(rxn["dependent"])
            popup_menu.addAction(dependentAction)
            dependentAction.triggered.connect(
                lambda event: self._setDependence(rxn, event)
            )

        if rxn is not None and not self.parent().run_control.optimize_running:
            mech = self.parent().mech
            if mech.is_reaction_recast(rxn["num"]):
                popup_menu.addSeparator()
                popup_menu.addAction(
                    "Revert to pressure-dependent",
                    lambda: self._revert_rxn_recast(rxn),
                )
            elif rxn["rxnType"] in PRESSURE_DEPENDENT_TYPES:
                popup_menu.addSeparator()
                popup_menu.addAction(
                    "Convert to Arrhenius at pressure…",
                    lambda: self._convert_rxn_to_pressure(rxn),
                )

        popup_menu.exec_(
            QtGui.QCursor.pos()
        )  # don't use exec_ twice or it will cause a double popup

    def _recast_composition(self):
        """Composition for evaluating falloff rates during a recast.

        The displayed shock's mix, or the first species at unit fraction
        when no mix is available (Plog/Chebyshev are M-independent).
        """
        parent = self.parent()
        shock = getattr(parent, "display_shock", None)
        if shock is not None:
            mix = getattr(shock, "thermo_mix", None)
        else:
            mix = None

        if mix:
            composition = dict(mix)
        else:
            composition = {parent.mech.gas.species_names[0]: 1.0}

        return composition

    def _reactor_pressure_default(self):
        """Pressure value + unit of the active shock zone's input boxes.

        Reads the selected zone's ``P{zone}_value_box`` /
        ``P{zone}_units_box`` (zone 2 = incident, 5 = reflected) so the
        recast dialog opens on the pressure the user already sees.
        Returns ``(1.0, "atm")`` when unavailable.
        """
        parent = self.parent()
        shock = getattr(parent, "display_shock", None)
        if shock is not None:
            zone = getattr(shock, "zone", None)
        else:
            zone = None

        value, unit = 1.0, "atm"
        if zone is not None:
            unit_box = getattr(parent, f"P{int(zone)}_units_box", None)
            value_box = getattr(parent, f"P{int(zone)}_value_box", None)
            if unit_box is not None and value_box is not None:
                candidate = unit_box.currentText().strip().strip("[]")
                if candidate in PRESSURE_UNITS and value_box.value() > 0:
                    value = value_box.value()
                    unit = candidate

        default = (value, unit)

        return default

    def _convert_rxn_to_pressure(self, rxn):
        parent = self.parent()
        rxnNum = rxn["num"]
        default_value, default_unit = self._reactor_pressure_default()
        eqn = rxn["item"].text().strip()

        dialog = RecastPressureDialog(
            parent, default_value=default_value, default_unit=default_unit, eqn=eqn,
        )
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return

        changed = parent.mech.recast_reaction_at_pressure(
            rxnNum, dialog.pressure_pa(), self._recast_composition(),
        )
        if not changed:
            return

        self._refresh_rxn_after_recast(rxnNum)
        self._schedule_render(set(), run_sim=True)

    def _revert_rxn_recast(self, rxn):
        parent = self.parent()
        rxnNum = rxn["num"]
        reverted = parent.mech.revert_reaction_recast(rxnNum)
        if not reverted:
            return

        self._refresh_rxn_after_recast(rxnNum)
        self._schedule_render(set(), run_sim=True)

    def _refresh_rxn_after_recast(self, rxnNum):
        """Rebuild one reaction's row after its rate type changed.

        Recasting swaps the rxn's Cantera rate type (e.g. Plog ->
        Arrhenius), so its cached type, coefficient rows, and widget
        slots are stale. Re-reads the entry from the mech, resets the
        row to its un-built state, and re-expands it through the normal
        lazy-build path if it was open.
        """
        parent = self.parent()
        tree = parent.mech_tree
        entry = self._set_mech_tree_data(self.mech_tree_type, parent.mech)[rxnNum]
        row_dict = tree.rxn[rxnNum]
        L1 = row_dict["item"]

        proxy_idx = self.proxy_model.mapFromSource(L1.index())
        was_expanded = tree.isExpanded(proxy_idx)
        if was_expanded:
            tree.collapse(proxy_idx)

        L1.removeRows(0, L1.rowCount())
        for key in _RXN_WIDGET_KEYS:
            row_dict.pop(key, None)
        row_dict["rxnType"] = entry["type"]
        row_dict["dependent"] = False

        L1.setToolTip(entry["type"])
        L1.setData(
            _rxn_type_tags(entry["type"], entry.get("rateClass", "")),
            _TYPE_TAG_ROLE,
        )
        L1.info["rxnType"] = entry["type"]
        L1.info["rxn_details"] = entry
        L1.info["row"] = []
        L1.info["hasExpanded"] = False
        L1.info["isExpanded"] = False

        self._paint_rxn_foreground(rxnNum)

        if was_expanded:
            tree.expand(proxy_idx)

    def _reset_all(self):
        parent = self.parent()
        prior_run_sim = self.run_sim_on_change
        self.run_sim_on_change = False
        try:
            mech = parent.mech
            resetRxnIdxs = set()
            for rxn in parent.mech_tree.rxn:
                if (
                    rxn["rxnType"] not in COEF_OPTIMIZABLE_TYPES
                    or "valueBox" not in rxn
                ):
                    continue
                for box in rxn["valueBox"]:
                    if box is None:
                        continue
                    rxnNum, coefName = box.info["rxnNum"], box.info["coefName"]
                    coef_key, bnds_key = keysFromBox(box, mech)
                    resetCoef = mech.coeffs_bnds[rxnNum][bnds_key][coefName]["resetVal"]
                    mech.coeffs[rxnNum][coef_key][coefName] = resetCoef
                    box._reset(silent=True)
                    resetRxnIdxs.add(rxnNum)

            dependentRxnIdxs = self._updateDependents()
        finally:
            self.run_sim_on_change = prior_run_sim

        self._schedule_render(resetRxnIdxs | set(dependentRxnIdxs), run_sim=True)

    def _copy_expanded_tab_rates(self):
        parent = self.parent()

        def copy_to_clipboard(values):
            parent.clipboard.clear()
            parent.clipboard.setText(
                "\t".join(values)
            )  # tab for new column, new line for new row

        if not self.copyRates:
            return
        elif parent.run_control.optimize_running:
            return

        values = []
        for rxnNum, rxn in enumerate(parent.mech_tree.rxn):
            mIndex = self.proxy_model.mapFromSource(rxn["item"].index())
            if parent.mech_tree.isExpanded(mIndex):
                values.append(str(rxn["rateBox"].value))

        t_unit_conv = parent.reactor_state.t_unit_conv
        values.append(
            str(parent.display_shock.time_offset / t_unit_conv)
        )  # add time offset
        if np.shape(values)[0] > 0:  # only clear clipboard and copy if values exist
            self.timer.singleShot(
                50, lambda: copy_to_clipboard(values)
            )  # 50 ms to prevent errors

    def _find_mech_item(self, item):
        if not hasattr(item, "info"):
            return None
        parent = self.parent()
        rxnNum = item.info["rxnNum"]
        tree = parent.mech_tree
        if item is tree.rxn[rxnNum]["item"] or item in tree.rxn[rxnNum]["valueBox"]:
            return tree.rxn[rxnNum]

        return None

    def _setDependence(self, rxn, isDependent):
        rxn["dependent"] = isDependent
        unc_boxes = rxn.get("uncBox") or []
        value_boxes = rxn.get("valueBox") or []
        formula_boxes = rxn.get("formulaBox") or []
        if isDependent:
            for box in unc_boxes:
                if box is None:
                    continue
                box.blockSignals(True)
                try:
                    box.setValue(-1)
                finally:
                    box.blockSignals(False)

            for valueBox, formulaBox in zip(value_boxes, formula_boxes):
                if valueBox is None or formulaBox is None:
                    continue
                valueBox.hide()
                formulaBox.show()

            dependentRxnIdxs = self._updateDependents()
            self._schedule_render(dependentRxnIdxs, run_sim=self.run_sim_on_change)
        else:
            for valueBox, formulaBox in zip(value_boxes, formula_boxes):
                if valueBox is None or formulaBox is None:
                    continue
                silentSetValue(valueBox, float(formulaBox.text()))
                formulaBox.hide()
                valueBox.show()

    def _updateDependents(self):
        """Re-evaluate every dependent rxn's formula against current coeffs.

        Mutates ``mech.coeffs`` in place. Returns the indices of the
        reactions that changed so the caller can fold them into a
        coalesced render.
        """
        parent = self.parent()
        mech = parent.mech
        updated = []
        for rxnNum, rxn in enumerate(parent.mech_tree.rxn):
            if not rxn.get("dependent"):
                continue
            updated.append(rxnNum)
            formula_boxes = rxn.get("formulaBox") or []
            for box in formula_boxes:
                if box is None:
                    continue
                box_rxn, coefName = box.info["rxnNum"], box.info["coefName"]
                coef_key, bnds_key = keysFromBox(box, mech)
                current_value = mech.coeffs[box_rxn][coef_key][coefName]
                reset_value = mech.coeffs_bnds[box_rxn][bnds_key][coefName]["resetVal"]
                if reset_value == 0.0:
                    eqn = "+" + str(current_value)
                else:
                    eqn = "*" + str(current_value / reset_value)
                box.applyFormula(emit=False, adjustment=eqn)

        return updated


class QSortFilterProxyModel(QtCore.QSortFilterProxyModel):
    def __init__(self, parent=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self._type_filter: frozenset = frozenset()

    def setTypeFilter(self, tokens: frozenset) -> None:
        """Restrict the filter to rows whose ``_TYPE_TAG_ROLE`` tags
        intersect ``tokens``. Empty set disables the type filter and
        falls back to text-only matching."""
        self._type_filter = frozenset(t.lower() for t in tokens)
        self.invalidateFilter()

    def filterAcceptsRow(self, row, parent):
        if parent.isValid():  # Do not apply the filter to child elements
            return True
        if not super().filterAcceptsRow(row, parent):
            return False
        if not self._type_filter:
            return True
        model = self.sourceModel()
        idx = model.index(row, 0)
        tags = model.data(idx, _TYPE_TAG_ROLE) or frozenset()

        return bool(self._type_filter & tags)

    def lessThan(self, left, right):
        rxnNum = lambda text: int(text[2:].split(":")[0])
        leftData = self.sourceModel().data(left)
        rightData = self.sourceModel().data(right)

        try:
            return rxnNum(leftData) < rxnNum(rightData)
        except ValueError:
            return leftData < rightData


class TreeFilter:
    def __init__(self, parent, proxy_model, model, _set_mech_widgets):
        self.parent = parent
        self.filter_input = parent.mech_tree_filter_box
        self.tree = parent.mech_tree
        self.proxy_model = proxy_model
        self.model = model
        self._set_mech_widgets = _set_mech_widgets

        self.filter_input.textChanged.connect(self.textChanged)

    def textChanged(self, event):
        type_tokens, text_tokens = self._split_filter_tokens(event)

        regexp = ["^.*"]
        for txt in text_tokens:
            if txt == "|":
                regexp.append(txt)
            elif txt == "&":
                continue
            elif len(txt.strip()) > 0:
                txt = txt.replace("*", ".*")
                regexp.append(rf"(?=.*\b{txt}\b)")
        regexp.append(".*$")
        regexp = "".join(regexp)

        self.proxy_model.setFilterRegularExpression(regexp)
        self.proxy_model.setTypeFilter(frozenset(type_tokens))
        self.update_match_tooltip(self.proxy_model.rowCount())
        self.expand_items()

    @staticmethod
    def _split_filter_tokens(event: str) -> tuple[list, list]:
        """Pull ``type:X`` tokens out of the user's filter input. Returns
        ``(type_tokens_lowercased, remaining_text_tokens)``."""
        type_tokens: list = []
        text_tokens: list = []
        for tok in event.strip().split(" "):
            m = _TYPE_TOKEN_RE.match(tok)
            if m:
                type_tokens.append(m.group(1).lower())
            else:
                text_tokens.append(tok)
        return type_tokens, text_tokens

    def update_match_tooltip(self, num, show=True):
        if num == 1:
            self.filter_input.setToolTip(f"{num:d} match")
        else:
            self.filter_input.setToolTip(f"{num:d} matches")

        if show:
            pos = self.filter_input.mapToGlobal(QtCore.QPoint(0, 0))
            width = self.filter_input.sizeHint().width()
            pos.setX(pos.x() + width * 2)
            height = self.filter_input.sizeHint().height()
            pos.setY(pos.y() + int(height / 4))

            QToolTip.showText(pos, self.filter_input.toolTip())

    def expand_items(self):
        tree = self.parent.mech_tree
        for row_idx in range(self.proxy_model.rowCount()):
            proxy_idx = self.proxy_model.index(row_idx, 0)
            idx = self.proxy_model.mapToSource(proxy_idx)
            item = self.model.itemFromIndex(idx)
            rxnNum = item.info["rxnNum"]

            if item.info["isExpanded"]:
                tree.expand(proxy_idx)
                self._set_mech_widgets(item)


class rateExpCoefficient(QWidget):  # rate expression coefficient
    def __init__(self, parent, coef, info, *args, **kwargs):
        QWidget.__init__(self)

        self.Label = QLabel(self.tr("{:s}:".format(coef[0])))
        info["label"] = self.Label

        exclude_keys = ["unc_value", "unc_type", "info"]
        valueBox_kwargs = {
            k: kwargs[k] for k in set(list(kwargs.keys())) - set(exclude_keys)
        }
        self.valueBox = misc_widget.ScientificDoubleSpinBox(
            parent=parent, *args, **valueBox_kwargs
        )
        self.valueBox.info = info
        self.valueBox.setValue(coef[2])
        self.valueBox.setSingleIntStep(0.01)
        self.valueBox.setMaximumWidth(75)  # This matches the coefficients
        self.valueBox.setToolTip("Coefficient Value")

        self.formulaBox = ScientificLineEdit(parent)
        self.formulaBox.info = info
        self.formulaBox.setValue(coef[2])
        self.formulaBox.setMaximumWidth(75)  # This matches the coefficients
        self.formulaBox.setSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed
        )
        self.formulaBox.hide()

        info["mainValueBox"] = self.valueBox

        if "unc_value" in kwargs and "unc_type" in kwargs:
            self.unc = Uncertainty(
                parent,
                "coef",
                info,
                value=kwargs["unc_value"],
                unc_choice=kwargs["unc_type"],
            )
        else:
            self.unc = Uncertainty(parent, "coef", info)

        self.uncValBox = self.unc.valBox
        self.uncTypeBox = self.unc.typeBox
        if coef[1] == "pre_exponential_factor":  # pre_exponential_factor does not support ± uncertainty types
            for uncType in ["±", "+", "-"]:  # Remove uncTypes from box
                self.uncTypeBox.removeItem(self.uncTypeBox.findText(uncType))

        spacer = QtWidgets.QSpacerItem(
            20, 10, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        end_spacer = QtWidgets.QSpacerItem(
            15, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum
        )

        layout = QGridLayout(self)
        layout.setContentsMargins(0, 2, 4, 2)

        layout.addWidget(self.Label, 0, 0)
        layout.addItem(spacer, 0, 1)
        layout.addWidget(self.formulaBox, 0, 2)
        layout.addWidget(self.valueBox, 0, 2)
        layout.addItem(end_spacer, 0, 3)
        layout.addWidget(self.unc, 0, 4)


class rxnRate(QWidget):
    def __init__(
        self, parent, info, rxnType="Arrhenius Reaction", label="", *args, **kwargs
    ):
        QWidget.__init__(self, parent)
        self.parent = parent

        self.Label = QLabel(self.tr("k"))
        self.Label.setToolTip("Reaction Rate [mol, cm, s]")

        exclude_keys = ["unc_value", "unc_type"]
        valueBox_kwargs = {
            k: kwargs[k] for k in set(list(kwargs.keys())) - set(exclude_keys)
        }
        self.valueBox = ScientificLineEditReadOnly(parent, *args, **valueBox_kwargs)
        self.valueBox.setMaximumWidth(75)  # This matches the coefficients
        self.valueBox.setDecimals(4)
        self.valueBox.setReadOnly(True)
        if "value" in kwargs:
            self.valueBox.setValue(kwargs["value"])

        spacer = QtWidgets.QSpacerItem(
            20, 10, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )

        layout = QGridLayout(self)
        layout.setContentsMargins(0, 2, 4, 2)

        layout.addWidget(self.Label, 0, 0)
        layout.addItem(spacer, 0, 1)
        layout.addWidget(self.valueBox, 0, 2)

        if rxnType in RATE_OPTIMIZABLE_TYPES:
            info["mainValueBox"] = self.valueBox

            if "unc_value" in kwargs and "unc_type" in kwargs:
                self.unc = Uncertainty(
                    parent,
                    "rate",
                    info,
                    value=kwargs["unc_value"],
                    unc_choice=kwargs["unc_type"],
                )
            else:
                self.unc = Uncertainty(parent, "rate", info)

            self.uncValBox = self.unc.valBox
            self.uncTypeBox = self.unc.typeBox

            end_spacer = QtWidgets.QSpacerItem(
                15, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum
            )
            layout.addItem(end_spacer, 0, 3)
            layout.addWidget(self.unc, 0, 4)
        else:
            end_spacer = QtWidgets.QSpacerItem(
                117, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum
            )
            layout.addItem(end_spacer, 0, 3)


class Uncertainty(QWidget):
    def __init__(self, parent, type, info, *args, **kwargs):
        super().__init__(parent)
        self.parent = parent
        self.info = info

        self.typeBox = misc_widget.SearchComboBox()  # uncertainty type
        self.typeBox.info = info
        self.typeBox.lineEdit().setReadOnly(
            True
        )  # defeats point of widget, but I like the look
        if type == "coef":
            self.typeBox.addItems(["F", "%", "±", "+", "-"])
        elif type == "rate":
            self.typeBox.addItems(["F", "%"])

        tooltipTxt = [
            '<html><table border="0" cellspacing="1" cellpadding="0">'
            '<tr><td style="padding-top:0; padding-bottom:6; padding-left:0; padding-right:4;"><p>F</p></td>',
            '<td style="padding-top:0; padding-bottom:6; padding-left:4; padding-right:0;"><p>Uncertainty Factor</p></td></tr>',
            '<tr><td style="padding-top:6; padding-bottom:6; padding-left:0; padding-right:4;"><p>%</p></td>',
            '<td style="padding-top:6; padding-bottom:6; padding-left:4; padding-right:0;"><p>Percent Uncertainty (%/100)</p></td></tr>',
            '<tr><td style="padding-top:6; padding-bottom:0; padding-left:0; padding-right:4;"><p>±</p></td>',
            '<td style="padding-top:6; padding-bottom:0; padding-left:4; padding-right:0;"><p>Plus or Minus</p></td></tr></table></html>',
        ]
        self.typeBox.setToolTip("".join(tooltipTxt))
        self.priorUncType = self.typeBox.currentText()
        self.typeBox.currentIndexChanged[str].connect(self.uncTypeChanged)

        self.uncMax = 100
        if "value" in kwargs:
            self.valBox = UncertaintyBox(
                parent, self.uncMax, value=kwargs["value"]
            )  # uncertainty value
        else:
            self.valBox = UncertaintyBox(parent, self.uncMax)  # uncertainty value
        self.valBox.info = info

        if "unc_choice" in kwargs:
            self.typeBox.setCurrentText(kwargs["unc_choice"])
            self.uncTypeChanged(
                kwargs["unc_choice"], update=False
            )  # initialize uncertainty type
        else:
            self.valBox.setUncType(self.priorUncType)  # initialize uncertainty type

        if "value" in kwargs:
            self.valBox.setValue(kwargs["value"])

        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        layout.addWidget(self.typeBox, 0, 0)
        layout.addWidget(self.valBox, 0, 1)

    def uncTypeChanged(self, event, update=True):
        def plus_minus_values():
            coefBox = self.info["mainValueBox"]
            return coefBox.strDecimals, coefBox.singleStep() * 10, sys.float_info.max

        self.valBox.setUncType(event)  # pass event change to uncValBox
        if self.priorUncType == "±" and update:
            self.valBox.setValue(
                self.valBox.minimum()
            )  # if prior uncertainty was +-, reset

        uncVal = self.valBox.uncValue

        if event == "F":  # only happens on event change, must have been % or ±

            if not np.isnan(uncVal) and update:
                self.valBox.setValue(uncVal + 1)
            self.valBox.setMinimum(1)
        elif self.priorUncType == "F":
            self.valBox.setMinimum(0)
            if update:
                self.valBox.setValue(uncVal - 1)

        if event in ["F", "%"]:
            self.valBox.setDecimals(2)
            self.valBox.setSingleStep(0.25)
            self.valBox.setMaximum(self.uncMax)
        else:
            dec, step, maxval = plus_minus_values()
            self.valBox.setDecimals(dec)
            self.valBox.setSingleStep(step)
            self.valBox.setMaximum(maxval)
            if update:
                self.valBox.setValue(-step)

        self.priorUncType = event
        if update:
            self.parent.tree.update_uncertainties(event, self.typeBox)


class UncValidator(QtGui.QValidator):
    def _isNum(self, s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    def validate(self, string, position):
        if self._isNum(string) or string == "-":
            state = QtGui.QValidator.Acceptable
        elif string == "" or string[position - 1].lower() in "e.-+":
            state = QtGui.QValidator.Intermediate
        else:
            state = QtGui.QValidator.Invalid
        return (state, string, position)


class UncertaintyBox(misc_widget.ScientificDoubleSpinBox):
    def __init__(self, parent, maxUnc, value=None, *args, **kwargs):
        super().__init__(parent=parent, *args, **kwargs)

        self.tree = self.parent().tree

        self.validator = UncValidator()
        self.setKeyboardTracking(False)
        self.setAccelerated(True)

        self.setMinimumWidth(55)
        self.setMaximumWidth(55)
        self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

        self.minimumBaseValue = 1
        self.maximumBaseValue = maxUnc
        self.setSingleIntStep(0.25)
        self.setDecimals(2)
        if value is None:
            self.setValue(-1)
        else:
            self.setValue(value)
        self.setToolTip(
            "Coefficient Uncertainty\n\nBased on reset values\nReset values default to initial mechanism"
        )

        self.uncValue = self.value()
        self.valueChanged.connect(
            self.setUncVal
        )  # update uncertainty value based on value of box
        self.lineEdit().installEventFilter(self)

    def eventFilter(self, obj, event):  # clear if text is -
        if (
            event.type() == QtCore.QEvent.MouseButtonPress
            and obj is self.lineEdit()
            and event.button() == QtCore.Qt.LeftButton
            and self.text() == "-"
        ):
            self.lineEdit().clear()
            return True
        else:
            return super().eventFilter(obj, event)

    def validate(self, text, position):
        return self.validator.validate(text, position)

    def setSingleStep(self, event):
        super(UncertaintyBox, self).setSingleStep(
            event
        )  # don't want to overwrite all functionality
        self.setMinimum(self.minimumBaseValue)
        self.setMaximum(self.maximumBaseValue)

    def setMinimum(self, event):
        self.minimumBaseValue = event
        super(UncertaintyBox, self).setMinimum(
            event - self.singleStep()
        )  # don't want to overwrite all functionality

    def setMaximum(self, event):
        self.maximumBaseValue = event
        super(UncertaintyBox, self).setMaximum(
            event + self.singleStep()
        )  # don't want to overwrite all functionality

    def valueFromText(self, text):
        if text == "-":
            value = self.maximum()
        else:
            value = float(text)
        return value

    def textFromValue(self, value):
        if value < self.minimumBaseValue or value > self.maximumBaseValue:
            string = "-"
        else:
            string = super(UncertaintyBox, self).textFromValue(
                value
            )  # don't want to overwrite all functionality
        return string

    def stepBy(self, steps):
        if self.specialValueText() and self.value() == self.minimum():
            text = self.textFromValue(self.minimum())
        else:
            text = self.cleanText()

        old_val = self.value()
        if self.value() < self.minimumBaseValue or self.value() > self.maximumBaseValue:
            val = old_val + self.singleStep() * steps
        else:
            val = (
                old_val
                + np.power(10, misc_widget.OoM(old_val)) * self.singleStep() * steps
            )

        if val < self.singleStep():
            val = 0

        new_string = "{:g}".format(val)
        self.lineEdit().setText(new_string)
        self.setValue(float(new_string))

    def setUncVal(self, event):
        if event < self.minimumBaseValue or event > self.maximumBaseValue:
            self.uncValue = np.nan
        elif self.uncType in ["F", "%"]:
            self.uncValue = event
        else:  # if +-, need to convert to cantera units
            tree = self.tree
            rxnNum = self.info["rxnNum"]
            coeffs = [*self.info["coef"], event]
            conv_type = tree.mech_tree_type + "2Cantera"
            cantera_value = tree.convert._arrhenius(
                rxnNum, deepcopy([coeffs]), conv_type
            )
            self.uncValue = cantera_value[0][2]

    def setUncType(self, event):
        self.uncType = event


class ScientificLineEditReadOnly(QLineEdit):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.parent = parent
        self.setDecimals(3)
        self.setValue(0)

        if "setDecimals" in kwargs:
            self.setDecimals(kwargs["setDecimals"])

        if "value" in kwargs:
            self.setValue(kwargs["value"])

    def mousePressEvent(self, event):  # selects all text on click
        self.selectAll()

    def setValue(self, value):
        self.value = value  # stores full-precision value; display text uses fewer digits
        self.setText("{:.{dec}g}".format(value, dec=self.decimals))

    def setDecimals(self, value):
        self.decimals = int(value)


class ScientificLineEdit(QLineEdit):
    valueChanged = QtCore.Signal(float)

    def __init__(self, parent, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.parent = parent
        self.setValue(0)
        self.displayMode = "value"
        self.setReadOnly(True)

        if "setDecimals" in kwargs:
            self.setDecimals(kwargs["setDecimals"])

        if "value" in kwargs:
            self.setValue(kwargs["value"])

        self.textEdited.connect(self.updateFormula)
        self.editingFinished.connect(lambda: self.applyFormula(emit=True))

    def mousePressEvent(self, event):
        if self.isReadOnly():
            if event.button() == QtCore.Qt.LeftButton:
                self.displayMode = "formula"
                self.setReadOnly(False)

                if hasattr(
                    self, "formula"
                ):  # if this has been clicked before and has a formula show it
                    self.setText("={:s}".format(self.formula))

        else:
            super(ScientificLineEdit, self).mousePressEvent(
                event
            )  # don't want to overwrite all functionality

    def setValue(self, value, emit=False):
        self.value = (
            value  # Full precision. For box value set below as float(self.text())
        )
        self.setText("{:g}".format(value))
        if emit:
            self.valueChanged.emit(value)

    def setInitialFormula(self):
        abbr = self.info["coefAbbr"]
        if abbr == "log(A)":
            abbr = "A"
        self.formula = "{:s}{:d}".format(abbr, self.info["rxnNum"] + 1)

    def updateFormula(self):
        self.formula = self.text().replace("=", "").replace("^", "**")

    def applyFormula(self, emit, adjustment="*1.0"):
        parent = self.parent
        mech = parent.mech
        rxnNum, coefName = self.info["rxnNum"], self.info["coefName"]
        coef_key, bnds_key = keysFromBox(self, mech)
        tree = parent.mech_tree

        formula = self.formula
        if formula is None or formula.strip() == "":
            self.setInitialFormula()
            formula = self.formula.replace("=", "")

        try:
            tree_ast = ast.parse(formula, mode="eval")
        except SyntaxError:
            self.setInitialFormula()
            formula = self.formula.replace("=", "")
            tree_ast = ast.parse(formula, mode="eval")
        names = [node.id for node in ast.walk(tree_ast) if isinstance(node, ast.Name)]

        var: dict = {}
        prior_value = mech.coeffs[rxnNum][coef_key][coefName]
        for name in names:
            match = re.split(r"(\d+)", name)
            if len(match) < 2 or not match[1]:
                continue
            subRxnNum = int(match[1]) - 1
            if subRxnNum < 0 or subRxnNum >= len(tree.rxn):
                continue
            formula = formula.replace(name, f"var[{name!r}]")
            if subRxnNum == rxnNum:
                var[name] = mech.coeffs_bnds[rxnNum][bnds_key][coefName]["resetVal"]
                if (
                    len(names) == 1
                    and formula[-1] == "]"
                    and adjustment not in ["*1.0", "+0.0"]
                ):
                    formula += adjustment
                    self.formula += adjustment
            else:
                for box in tree.rxn[subRxnNum].get("valueBox") or []:
                    if box is None:
                        continue
                    if box.info["coefName"] == coefName:
                        var[name] = mech.coeffs[subRxnNum][coef_key][coefName]

        try:
            value = eval(formula, {"__builtins__": {}}, {"var": var})
        except Exception:
            self.setValue(prior_value, emit=False)

            return

        mech.coeffs[rxnNum][coef_key][coefName] = value
        conv_type = "Cantera2" + parent.tree.mech_tree_type
        coeffs = [*self.info["coef"], value]
        display_val = parent.convert_units._arrhenius(
            rxnNum, deepcopy([coeffs]), conv_type,
        )[0][2]

        self.displayMode = "value"
        self.setReadOnly(True)
        self.setValue(display_val, emit)
