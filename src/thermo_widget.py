# This file is part of Frhodo. Copyright © 2020, UChicago Argonne, LLC
# and licensed under BSD-3-Clause. See License.txt in the top-level
# directory for license and copyright information.

import sys, ast, re
import misc_widget
import cantera as ct
import numpy as np
from copy import deepcopy
from scipy.optimize import root_scalar
from qtpy.QtWidgets import *
from qtpy import QtWidgets, QtGui, QtCore


def silentSetValue(obj, value):
    obj.blockSignals(True)  # stop changing text from signaling
    obj.setValue(value)
    obj.blockSignals(False)  # allow signals again


class Tree(QtCore.QObject):
    def __init__(self, parent):
        super().__init__(parent)
        self.run_sim_on_change = True
        self.copyRates = False
        self.convert = parent.convert_units

        self.color = {
            "variable_rxn": QtGui.QBrush(QtGui.QColor(188, 0, 188)),
            "fixed_rxn": QtGui.QBrush(QtGui.QColor(0, 0, 0)),
        }

        self.tree = QTreeView()
        parent.thermo_tree_container.addWidget(self.tree, 0, 0)

        self.tree.setRootIsDecorated(False)
        self.tree.setIndentation(21)
        # self.tree.itemClicked.connect(lambda event: self.item_clicked(event))
        # self.tree.resized.connect(lambda: self.maxViewableItems())
        return
        self.model = ThermoModel()
        self.model.setHorizontalHeaderLabels(["Species"])
        # self.model = QtCore.QAbstractItemModel()
        self.tree.setModel(self.model)

        # for n in range(0,100):
        # L1 = QtWidgets.QTreeWidgetItem()
        # self.tree.addTopLevelItem(L1)
        # L1.setText(0, f"item {n:d}")
        # for item in ["String AA", "String BB", "String CC" ]:
        # L2 = QtWidgets.QTreeWidgetItem([item])
        # L1.addChild(L2)

        # self.tree.maxViewableItems()


class TreeView(QTreeView):
    resized = QtCore.Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.topLevelItemList = []
        self.viewableItems = []
        self.atItemMax = False

        self.collapsed.connect(
            lambda event: self.setViewableItems(event, type="collapse")
        )
        self.expanded.connect(lambda event: self.setViewableItems(event, type="expand"))

    def clear(self):
        self.topLevelItemList = []
        self.viewableItems = []
        self.atItemMax = False
        super(TreeView, self).clear()

    def addTopLevelItem(self, item):
        if (
            not hasattr(self, "maxItems")
            or self.topLevelItemCount + 1 < self.maxViewableItems()
        ):
            super(TreeView, self).addTopLevelItem(item)
            if not hasattr(self, "maxItems"):
                self.collapsedItemHeight = self.sizeHintForRow(0)
            self.atItemMax = False
            self.topLevelItemList.append(item)
            self.viewableItems.append(item)
        else:
            self.atItemMax = True

    def resizeEvent(self, event):
        self.resized.emit()
        super(TreeView, self).resizeEvent(event)


class ThermoModel(QtCore.QAbstractItemModel):
    def __init__(self, parent=None):
        super(ThermoModel, self).__init__(parent)
        self.rootNodes = self._getRootNodes()

    def _getRootNodes(self):
        raise NotImplementedError()

    def index(self, row, column, parent):
        if not parent.isValid():
            return self.createIndex(row, column, self.rootNodes[row])
        parentNode = parent.internalPointer()
        return self.createIndex(row, column, parentNode.subnodes[row])

    def parent(self, index):
        if not index.isValid():
            return QModelIndex()
        node = index.internalPointer()
        if node.parent is None:
            return QModelIndex()
        else:
            return self.createIndex(node.parent.row, 0, node.parent)

    def reset(self):
        self.rootNodes = self._getRootNodes()
        QAbstractItemModel.reset(self)

    def rowCount(self, parent):
        if not parent.isValid():
            return len(self.rootNodes)
        node = parent.internalPointer()
        return len(node.subnodes)
