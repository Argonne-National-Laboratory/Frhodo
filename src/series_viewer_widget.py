# This file is part of Frhodo. Copyright © 2020, UChicago Argonne, LLC
# and licensed under BSD-3-Clause. See License.txt in the top-level 
# directory for license and copyright information.

from calculate import shock_fcns
import numpy as np
from qtpy.QtWidgets import *
from qtpy import QtWidgets, QtGui, QtCore
from copy import deepcopy

class Series_Viewer:
    def __init__(self, parent, *args, **kwargs):
        self.parent = parent

        self.tree = TreeWidget()
        parent.series_viewer_container.addWidget(self.tree, 0, 0)
        self._initialize_tree()
         
        self.full_row_selected = False
        self.data_table = []
        self.L1 = []
        self.L2 = []
        self.last_selected_table = None
        
        parent.add_series_button.clicked.connect(self._add_series_table)
        self.tree.resized.connect(lambda: self.updateSizeHint())

    def _initialize_tree(self):
        tree = self.tree
        header_labels = ['Series']
        num_cols = 1                        # THIS SHOULD BE VARIABLE BASED ON INPUT DATA
        tree.setColumnCount(num_cols)
        if len(header_labels) < num_cols:   # Pad named columns with blank named columns 
            header_labels.extend((['']*(num_cols-len(header_labels))))
        tree.setHeaderLabels(header_labels)
        tree.setHeaderHidden(True)
        tree.uniformRowHeights = False

        tree.setRootIsDecorated(False)
        # tree.setHeaderHidden(True)
        tree.setIndentation(0)
        tree.itemClicked.connect(self.item_clicked)
            
    def item_clicked(self, event):
        if event.isExpanded():
            event.setExpanded(False)
        else:
            event.setExpanded(True)
    
    def _add_series_table(self, event):
        parent = self.parent
        tree = self.tree
        
        for table in self.data_table:   # skip adding the same set
            if parent.path['exp_main'] == table.main_path:
                return
        
        L1 = QtWidgets.QTreeWidgetItem(tree)
        L2 = QtWidgets.QTreeWidgetItem(L1)
        self.L1.append(L1)
        self.L2.append(L2)
        
        L1.setText(0, parent.series.name[parent.series.idx])
        parent.series.added_to_table(parent.series.idx)    # update series to inform
        
        path, shocks = parent.series.path[-1], parent.series.shock[-1]
        self.data_table.append(DataSetsTable(parent, self, path, shocks))
        
        # L2.setSizeHint(0, self.data_table[-1].viewportSizeHint())
        # table_holder_widget = QWidget()
        # layout = QGridLayout(table_holder_widget)
        # layout.setContentsMargins(0,0,6,6)
        # layout.addWidget(self.data_table[-1], 0, 0)
        
        L2.treeWidget().setItemWidget(L2, 0, self.data_table[-1])
        L1.addChild(L2)
        if len(self.data_table) == 1:
            self.collapsedItemHeight = self.tree.itemsSizeHint().height()
        self.updateSizeHint()
    
    def updateSizeHint(self, visible_items=2):
        treeWidth  = self.tree.maximumViewportSize().width()
        # reduce treeHeight by header height
        treeHeight = self.tree.maximumViewportSize().height() - self.tree.header().height()
        if len(self.data_table) > 0:
            if len(self.data_table) <= (visible_items+1):   # limit size to show other sets
                treeHeight -= self.collapsedItemHeight*(len(self.data_table)-1)
            else:
                treeHeight -= self.collapsedItemHeight*visible_items
            
            for n, L2 in enumerate(self.L2):
                tableWidth  = self.data_table[n].getSize().width()
                tableHeight = self.data_table[n].getSize().height()+self.collapsedItemHeight
                # tableHeight = self.tree.itemsSizeHint().height()
                width = np.min([treeWidth, tableWidth])
                height = np.min([treeHeight, tableHeight])
                L2.setSizeHint(0, QtCore.QSize(width, height))
            
    def update(self, shock_idx=None):
        if not self.data_table: return # in case no series have been added
        
        parent = self.parent
        for table in self.data_table:
            if parent.path['exp_main'] == table.main_path:
                if shock_idx is None:
                    table._update()
                else:
                    table._update(shock_idx)
                break


class TreeWidget(QTreeWidget):
    resized = QtCore.Signal()
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def resizeEvent(self, event):
        self.resized.emit()
        super(TreeWidget, self).resizeEvent(event)
    
    def sizeHint(self):
        height = 2 * self.frameWidth() # border around tree
        if not self.isHeaderHidden():
            header = self.header()
            headerSizeHint = header.sizeHint()
            height += headerSizeHint.height()
        rows = 0
        it = QTreeWidgetItemIterator(self)
        while it.value() is not None:
            rows += 1
            index = self.indexFromItem(it.value())
            height += self.rowHeight(index)
            it += 1
        if self.isHeaderHidden():
            return QtCore.QSize(2*self.frameWidth(), height)
        else:
            return QtCore.QSize(header.length() + 2 * self.frameWidth(), height)

    def itemsSizeHint(self):
        # height = 2 * self.frameWidth() # border around tree
        height = 0
        # if not self.isHeaderHidden():
            # header = self.header()
            # headerSizeHint = header.sizeHint()
            # height += headerSizeHint.height()
        rows = 0
        it = QTreeWidgetItemIterator(self)
        while it.value() is not None:
            rows += 1
            index = self.indexFromItem(it.value())
            height += self.rowHeight(index)
            it += 1
        return QtCore.QSize(self.header().length() + 2 * self.frameWidth(), height)
                

class DataSetsTable(QTableWidget):
    def __init__(self, parent, series_holder, path, shock, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parent = parent
        self.series_holder = series_holder
        self.shock = shock
        self.exp_path = path
        self.convert_units = parent.convert_units
        
        self.copyShockNum = True
        
        self._create_table()
        
        # Connect Signals
        self.itemSelectionChanged.connect(self._selection_change)
        
        # Set right click menu
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._popup_menu)
        
        # Set Shortcuts
        shortcut_fcn_pair = [['Ctrl+C', lambda: self._copy()]]
        for shortcut, fcn in shortcut_fcn_pair:
            QShortcut(QtGui.QKeySequence(shortcut), self, activated=fcn, context=QtCore.Qt.WidgetShortcut)
        
    def _create_table(self):
        parent = self.parent
        
        self.main_path = parent.path['exp_main']
        self.verticalHeader().setDefaultSectionSize(23)
        
        parent.series.in_table[-1] = True
        self.all_shocks = [int(shock['num']) for shock in self.shock]
        
        row_names = ['Shock {:.0f}'.format(n) for n in self.all_shocks]
        self.setRowCount(len(row_names))
        self.setVerticalHeaderLabels(row_names)
       
        self.column_names = ['incl.', 'T1', 'P1', 'P4', 'T2', 'P2', 'T5', 'P5', 'mix']
        self.setColumnCount(len(self.column_names))
        self.setHorizontalHeaderLabels(self.column_names)
        
        self.include_box = []
        for idx, shock_num in enumerate(self.all_shocks):
            cell_widget = QWidget()
            self.include_box.append(QCheckBox())
            self.include_box[-1].info = {'shock_num': shock_num, 'shock_idx': idx}
            self.include_box[-1].toggled.connect(self._toggle_checkbox)
            self.include_box[-1].setChecked(False)
            self.include_box[-1].setFocusPolicy(QtCore.Qt.NoFocus)
            # create layout to center checkbox in cell
            checkbox_layout = QHBoxLayout(cell_widget)
            checkbox_layout.addWidget(self.include_box[-1])
            checkbox_layout.setAlignment(QtCore.Qt.AlignCenter)
            checkbox_layout.setContentsMargins(0,0,0,0)
            cell_widget.setLayout(checkbox_layout)
            self.setCellWidget(idx, 0, cell_widget)
        
        self.horizontalHeader().setSectionsMovable(True)
        self.horizontalHeader().setDragEnabled(True)
        self.horizontalHeader().setDragDropMode(QAbstractItemView.InternalMove)
        
        self.horizontalHeader().setStretchLastSection(True)
        
        self.SizeAdjustPolicy(2)    # makes table adjust contents on viewport resize
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        # self.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.resizeColumnsToContents()
        
        # self.verticalHeader().setSectionsMovable(True)
        # self.verticalHeader().setDragEnabled(True)
        # self.verticalHeader().setDragDropMode(QAbstractItemView.InternalMove)

        self._update()
        
    def _update(self, shock_idx=None):
        parent = self.parent
        data = self.shock
        
        if shock_idx is None:
            shock_idxs = range(0, len(self.all_shocks))
        else:
            shock_idxs = [shock_idx]

        for idx in shock_idxs:
            # Skip runs that did not load
            if 'exp_data' in data[idx]['err']:
                self._toggle_checkbox(idx=idx, bool=False)
                continue
            
            # Check that proper variables exist
            IC = [data[idx][key] for key in ['T1', 'P1', 'u1']]
            if np.isnan(IC).any() or len(data[idx]['thermo_mix']) == 0:
                continue
            
            parent.series.thermo_mix(shock = data[idx])
            
            shock_vars = {'T1': data[idx]['T1'], 'P1': data[idx]['P1'], 
                          'u1': data[idx]['u1'], 'mix': data[idx]['thermo_mix']}

            shock = shock_fcns.Properties(parent.mech.gas, shock_vars, parent = parent)
            
            if not shock.success:
                self._toggle_checkbox(idx=idx, bool=False)
                data[idx]['err'].append('SIM Failure')
                continue
                
            for var in ['rho1', 'T2', 'P2', 'u2', 'T5', 'P5']:
                data[idx][var] = shock.res[var]
                
            for n, var in enumerate(self.column_names):
                if var in ['T1', 'P1', 'P4', 'T2', 'P2', 'T5', 'P5']:
                    if var != 'P4':
                        unit = eval('str(parent.' + var + '_units_box.currentText())')
                    elif var == 'P4':
                        unit = 'psi'    # TODO: THIS IS HARDCODED UNTIL A BETTER SOLUTION IS NEEDED
                    display_value = self.convert_units(data[idx][var], unit, unit_dir='out')
                    text  = '{:.2f}'.format(display_value)
                    self.setItem(idx, n, QTableWidgetItem(text))
            
        self.resizeColumnsToContents()
    
    def getSize(self):
        w = self.verticalHeader().width() + 2  # +2 seems to be needed
        for i in range(self.columnCount()):
            w += self.columnWidth(i)  # seems to include gridline (on my machine)
        h = self.horizontalHeader().height() + 2
        for i in range(self.rowCount()):
            h += self.rowHeight(i)
        return QtCore.QSize(w, h)
    
    def _selection_change(self):
        parent = self.parent
        num_columns = self.columnCount()
        
        if self.series_holder.last_selected_table is None:          # for clearing other tables if selected
            self.series_holder.last_selected_table = self
        elif self.series_holder.last_selected_table is not self:
            self.series_holder.last_selected_table.blockSignals(True)
            self.series_holder.last_selected_table.clearSelection()
            self.series_holder.last_selected_table.blockSignals(False)
            self.series_holder.last_selected_table = self
        
        self.full_row_selected = False
        
        # If only one row is selected:
        selected_rows = [idx.row() for idx in self.selectedIndexes()]
        selected_columns = [idx.column() for idx in self.selectedIndexes()]
        if all(x == selected_rows[0] for x in selected_rows):   # only 1 row is selected
            if selected_columns == list(range(num_columns)):    # full row is selected
                self.full_row_selected = True
                n = self.selectedIndexes()[0].row()
                
                if parent.path['exp_main'] != self.exp_path:
                    parent.exp_main_box.setText(str(self.exp_path))       # set to selected table's exp path
                parent.shock_choice_box.setValue(self.all_shocks[n]) # set to selected row's shock num
                return

    def _toggle_checkbox(self, event=None, **kwargs):
        def silentSetChecked(box, bool):
            box.blockSignals(True)
            box.setChecked(bool)          # set checkbox silently
            box.blockSignals(False)
        
        def shockError(idx):
            shock_err = list(dict.fromkeys(self.shock[idx]['err'])) # remove duplicate values
            if len(shock_err) > 0 and shock_err != ['raw_data']:    # ignore error if raw data doesn't exist
                self.shock[idx]['include'] = False
                #self.include_box[idx].setChecked(False)
                silentSetChecked(self.include_box[idx], False)
                return True
            else:
                return False

        if all(key in kwargs for key in ['idx', 'bool']):   # if changed programmatically
            chkboxIdx, bool = kwargs['idx'], kwargs['bool']
            if shockError(chkboxIdx): return
            
            self.shock[chkboxIdx]['include'] = bool  # set value in data
            silentSetChecked(self.include_box[chkboxIdx], bool)

        elif type(self.sender()) == QCheckBox:              # if changed through widget
            # set value to be the status of checkbox
            bool = self.sender().isChecked()
            chkboxIdx = self.sender().info['shock_idx']
            if shockError(chkboxIdx): return
                
            self.shock[chkboxIdx]['include'] = bool
            # set all selected rows to be the same
            selected_rows = [idx.row() for idx in self.selectedIndexes()]
            selected_columns = [idx.column() for idx in self.selectedIndexes()]
            for row, column in zip(selected_rows, selected_columns):
                columnName = self.horizontalHeaderItem(column).text()
                if row != chkboxIdx and columnName == 'incl.':
                    self.shock[row]['include'] = bool
                    silentSetChecked(self.include_box[row], bool)

    
    def keyPressEvent(self, event):
        key = {'change_shock': {'up': QtCore.Qt.Key_Up, 'down': QtCore.Qt.Key_Down},
               'select_toggle': {'return': QtCore.Qt.Key_Return, 'enter': QtCore.Qt.Key_Enter,
               'delete': QtCore.Qt.Key_Delete, 'space': QtCore.Qt.Key_Space}}
        if self.full_row_selected:
            selected_row = self.selectedIndexes()[0].row()
            if event.key() in key['change_shock'].values():
                if event.key() == key['change_shock']['up']:
                    self.selectRow(selected_row - 1)
                else:
                    self.selectRow(selected_row + 1)
                return
            elif event.key() in key['select_toggle'].values():
                if event.key() == key['select_toggle']['delete']:
                    self.include_box[selected_row].setChecked(False)
                else:
                    include_exp = self.include_box[selected_row].isChecked()
                    self.include_box[selected_row].setChecked(not include_exp)
                return
        
        super(DataSetsTable, self).keyPressEvent(event)   # don't want to overwrite all shortcuts     
    
    def _popup_menu(self, event):
        def setCopyShockNumAction(self, event):
            self.copyShockNum = event 
            
        popup_menu = QMenu(self)
        popup_menu.addAction('Copy', lambda: self._copy(), 'Ctrl+C')
        
        popup_menu.addSeparator()
        
        copyShockNumAction = QAction('Copy: Incl. Exp Num', checkable=True)
        copyShockNumAction.setChecked(self.copyShockNum)
        popup_menu.addAction(copyShockNumAction)
        copyShockNumAction.triggered.connect(lambda event: setCopyShockNumAction(self, event))
        
        popup_menu.exec_(QtGui.QCursor.pos())
        
    def _copy(self):
        parent = self.parent
        
        selected_rows = [idx.row() for idx in self.selectedIndexes()]
        selected_columns = [idx.column() for idx in self.selectedIndexes()]
                    
        text = ''
        old_row = -1
        # TODO: Make robust for groups of individual cells being copied
        for n, (row, column) in enumerate(zip(selected_rows, selected_columns)):
            if n != 0:  # not on the first line
                if row != old_row:
                    text += '\n'
                else:
                    text += '\t'
                
            if self.copyShockNum and row != old_row:    # add shock number
                text += '{:d}\t'.format(self.all_shocks[row])
                
            if hasattr(self.item(row, column), 'text'):
                text += self.item(row, column).text()
                
            old_row = row
                
        parent.clipboard.clear()
        parent.clipboard.setText(text) # tab for new column, new line for new row
                
    
class CustomHeader(QHeaderView):        # https://blog.qt.io/blog/2014/04/11/qt-weekly-5-widgets-on-a-qheaderview/
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    # def _populateTree(self, children, parent=None):
        # if parent is None:
            # parent = self.root_model.invisibleRootItem()
          
        # for child in sorted(children):
            # child_item = QtGui.QStandardItem(child)
            # parent.appendRow(child_item)
            # if isinstance(children, dict):
                # self._populateTree(children[child], child_item)