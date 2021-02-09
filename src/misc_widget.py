# This file is part of Frhodo. Copyright © 2020, UChicago Argonne, LLC
# and licensed under BSD-3-Clause. See License.txt in the top-level 
# directory for license and copyright information.

import re, sys
import numpy as np
from qtpy.QtWidgets import *
from qtpy import QtWidgets, QtGui, QtCore
from calculate.convert_units import OoM
    
# Regular expression to find floats. Match groups are the whole string, the
# whole coefficient, the decimal part of the coefficient, and the exponent
# part.
_float_re = re.compile(r'(([+-]?\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)')

def valid_float_string(string):
    match = _float_re.search(string)
    return match.groups()[0] == string if match else False
    
class FloatValidator(QtGui.QValidator):
    def validate(self, string, position):
        if valid_float_string(string):
            state = QtGui.QValidator.Acceptable
        elif string == "" or string[position-1].lower() in 'e.-+':
            state = QtGui.QValidator.Intermediate
        else:
            state = QtGui.QValidator.Invalid
        return (state, string, position)

    def fixup(self, text):
        match = _float_re.search(text)
        return match.groups()[0] if match else ""
       
class ScientificDoubleSpinBox(QtWidgets.QDoubleSpinBox):
    resetValueChanged = QtCore.Signal(float)
    def __init__(self, reset_popup=True, *args, **kwargs):
        self.validator = FloatValidator()
        if 'numFormat' in kwargs:
            self.numFormat = kwargs.pop('numFormat')
        else:
            self.numFormat = 'g'
        
        self.setStrDecimals(6)  # number of decimals displayed
        super().__init__(*args, **kwargs)
        self.cb = QApplication.clipboard()
        self.setKeyboardTracking(False)
        self.setMinimum(-sys.float_info.max)
        self.setMaximum(sys.float_info.max)
        self.setDecimals(int(np.floor(np.log10(sys.float_info.max))))   # big for setting value
        self.setSingleStep(0.1)
        self.setSingleIntStep(1)
        self.setSingleExpStep(0.1)
        self.setAccelerated(True)
        # self.installEventFilter(self)
        
        if 'value' in kwargs:
            self.setValue(kwargs['value'])
        else:
            self.setValue(0)
            
        self._set_reset_value(self.value())
        
        if reset_popup:
            # Set popup
            self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
            self.customContextMenuRequested.connect(self._popup_menu)
            
            # Set Shortcuts
            shortcut_fcn_pair = [['Ctrl+R', lambda: self._reset()], ['Ctrl+C', lambda: self._copy()],
                                 ['Ctrl+V', lambda: self._paste()]]
            for shortcut, fcn in shortcut_fcn_pair:     # TODO: need to fix hover shortcuts not working
                QShortcut(QtGui.QKeySequence(shortcut), self, activated=fcn, context=QtCore.Qt.WidgetShortcut)
    
    # def eventFilter(self, obj, event):              # event filter to allow hover shortcuts
        # if event.type() == QtCore.QEvent.Enter: 
            # self.setFocus()
            # return True
        # elif event.type() == QtCore.QEvent.Leave:
            # return False
        # else:
            # return super().eventFilter(obj, event)
    
    def _popup_menu(self, event):
        popup_menu = QMenu(self)
        popup_menu.addAction('Reset', lambda: self._reset(), 'Ctrl+R')
        popup_menu.addSeparator()
        popup_menu.addAction('Copy', lambda: self._copy(), 'Ctrl+C')
        popup_menu.addAction('Paste', lambda: self._paste(), 'Ctrl+V')
        popup_menu.addSeparator()
        popup_menu.addAction('Set Reset Value', lambda: self._set_reset_value(self.value()))
        popup_menu.exec_(QtGui.QCursor.pos())
    
    def _reset(self, silent=False):
        self.blockSignals(True)     # needed because shortcut isn't always signalling valueChanged.emit
        self.setValue(self.reset_value)
        self.blockSignals(False)
        if not silent:
            self.valueChanged.emit(self.reset_value)
    
    def setStrDecimals(self, value: int):
        self.strDecimals = value
    
    def setSingleIntStep(self, value: float):
        self.singleIntStep = value

    def setSingleExpStep(self, value: float):
        self.singleExpStep = value

    def _set_reset_value(self, value):
        self.reset_value = value
        self.resetValueChanged.emit(self.reset_value)
    
    def _copy(self):
        self.selectAll()
        cb = self.cb
        cb.clear(mode=cb.Clipboard)
        cb.setText(self.textFromValue(self.value()), mode=cb.Clipboard)
    
    def _paste(self):
        previous_value = self.text()
        if self.fixup(self.cb.text()):
            self.setValue(float(self.fixup(self.cb.text())))
        else:
            self.setValue(float(previous_value))
    
    def keyPressEvent(self, event):
        if event.matches(QtGui.QKeySequence.Paste):
            self._paste()
        
        super(ScientificDoubleSpinBox, self).keyPressEvent(event)   # don't want to overwrite all shortcuts
    
    def validate(self, text, position):
        return self.validator.validate(text, position)

    def fixup(self, text):
        return self.validator.fixup(text)

    def valueFromText(self, text):
        return float(text)

    def textFromValue(self, value):
        """Modified form of the 'g' format specifier."""
        if 'g' in self.numFormat:
            # if full number showing and number decimals less than str, switch to number decimals
            if abs(OoM(value)) < self.strDecimals and self.decimals() < self.strDecimals:
                string = "{:.{dec}{numFormat}}".format(value, dec=int(abs(OoM(value)))+1+self.decimals(), numFormat=self.numFormat)
            else:
                string = "{:.{dec}{numFormat}}".format(value, dec=self.strDecimals, numFormat=self.numFormat)
        elif 'e' in self.numFormat:
            string = "{:.{dec}{numFormat}}".format(value, dec=self.strDecimals, numFormat=self.numFormat)
        string = re.sub("e(-?)0*(\d+)", r"e\1\2", string.replace("e+", "e"))

        return string
    
    def stepBy(self, steps):
        if self.specialValueText() and self.value() == self.minimum():
            text = self.textFromValue(self.minimum())
        else:    
            text = self.cleanText()
        
        old_val = float(text)
        if self.numFormat == 'g' and abs(OoM(old_val)) < self.strDecimals:    # my own custom g
            val = old_val + self.singleIntStep*steps
        else:
            old_OoM = OoM(old_val)
            val = old_val + np.power(10, old_OoM)*self.singleExpStep*steps
            new_OoM = OoM(val)
            if old_OoM > new_OoM:   # needed to step down by new amount 1E5 -> 9.9E6
                if self.numFormat == 'g' and abs(new_OoM) < self.strDecimals:
                    val = old_val + self.singleIntStep*steps
                else:
                    val = old_val + np.power(10, new_OoM)*self.singleExpStep*steps

        self.setValue(val)

        
class SearchComboBox(QComboBox):
    def __init__(self, parent=None):
        super(SearchComboBox, self).__init__(parent)

        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setEditable(True)
        self.setInsertPolicy(QComboBox.NoInsert)
        
        # add a filter model to filter matching items
        self.pFilterModel = QtCore.QSortFilterProxyModel(self)
        self.pFilterModel.setFilterCaseSensitivity(QtCore.Qt.CaseInsensitive)
        self.pFilterModel.setSourceModel(self.model())

        # add a completer, which uses the filter model
        self.completer = QCompleter(self.pFilterModel, self)
        
        # always show all (filtered) completions
        self.completer.setFilterMode(QtCore.Qt.MatchContains)
        self.completer.setCompletionMode(QCompleter.UnfilteredPopupCompletion)
        self.completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
        self.setCompleter(self.completer)
        
        # connect signals
        self.lineEdit().textEdited.connect(self.pFilterModel.setFilterFixedString)
        self.lineEdit().editingFinished.connect(self.on_completer_activated)
        self.completer.activated[str].connect(self.on_completer_activated)

    # on selection of an item from the completer, select the corresponding item from combobox 
    def on_completer_activated(self, text=None):
        if text is None:
            text = self.lineEdit().text()
        
        old_idx = self.currentIndex()
        if text:
            idx = self.findText(text)
            
            if idx < 0:         # if new text not found, revert to prior text
                idx = old_idx
        else:                   # if no text found, revert to prior
            idx = old_idx
            
        self.setCurrentIndex(idx)
        self.activated[str].emit(self.itemText(idx))

    # on model change, update the models of the filter and completer as well 
    def setModel(self, model):
        super().setModel(model)
        self.pFilterModel.setSourceModel(model)
        self.completer.setModel(self.pFilterModel)

    # on model column change, update the model column of the filter and completer as well
    def setModelColumn(self, column):
        self.completer.setCompletionColumn(column)
        self.pFilterModel.setFilterKeyColumn(column)
        super().setModelColumn(column)    
    
    def setNewStyleSheet(self, down_arrow_path):
        fontInfo = QtGui.QFontInfo(self.font())
        family = fontInfo.family()
        font_size = fontInfo.pixelSize()
        
         # stylesheet because of a border on the arrow that I dislike
        stylesheet = ["QComboBox { color: black;  font-size: " + str(font_size) + "px;",
            "font-family: " + family + ";  margin: 0px 0px 1px 1px;  border: 0px;",
            "padding: 1px 0px 0px 0px;}", # This (useless) line resolves a bug with the font color
            "QComboBox::drop-down { border: 0px; }" # Replaces the whole arrow of the combo box
            "QComboBox::down-arrow { image: url(" + down_arrow_path + ");",
            "width: 14px; height: 14px; }"]
        
        self.setStyleSheet(' '.join(stylesheet))         

        
class ItemSearchComboBox(SearchComboBox):   # track items in itemList
    def __init__(self, parent=None):
        super().__init__(parent)
        self.itemList = []
        self.completer.activated.connect(self.on_completer_activated)
    
    def addItem(self, item):
        super().addItem(item)
        self.itemList.append(item)
    
    def removeItem(self, idx):
        super().removeItem(idx)
        del self.itemList[idx]
    
    def clear(self):
        super().clear()
        self.itemList = []
        
        
class CheckableSearchComboBox(ItemSearchComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setView(QTreeView())
        self.view().setHeaderHidden(True)
        self.view().setIndentation(0)
        
        self.view().header().setMinimumSectionSize(0)   # set minimum to 0
        self.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        
        #self.setModelColumn(1)  # sets column for text to the second column
        self.view().setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        
        self.cb = parent.clipboard

        # Set popup
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(lambda event: self._popup_menu(event))
        
        # Set Shortcuts
        shortcut_fcn_pair = [['Ctrl+R', lambda: self._reset()]]
        for shortcut, fcn in shortcut_fcn_pair:
            QShortcut(QtGui.QKeySequence(shortcut), self, activated=fcn, context=QtCore.Qt.WidgetShortcut)

        # Connect Signals
        self.view().pressed.connect(self.handleItemPressed)
    
    def handleItemPressed(self, index):
        self.setCurrentIndex(index.row())
        self.hidePopup()

    def addItem(self, item, model=None):
        super().addItem(item)
    
        checkbox_item = self.model().item(self.count()-1, 0)
        checkbox_item.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
        checkbox_item.setCheckState(QtCore.Qt.Unchecked)

        #self.view().resizeColumnToContents(0)

    def addItems(self, items):
        for item in items:
            self.addItem(item)

    def itemChecked(self, index):
        item = self.model().item(index, 0)
        return item.checkState() == QtCore.Qt.Checked

    def sizeHint(self):
        base = super().sizeHint()
        height = base.height()
        
        width = 0
        if type(self.view()) is QTreeView:                  # if the view is a QTreeView
            for i in range(self.view().header().count()):   # add size hint for each column
                width += self.view().sizeHintForColumn(i)
        else:
            width += self.view().sizeHintForColumn(0)
        
        if self.count() > self.maxVisibleItems():                       # if scrollbar visible
            width += self.view().verticalScrollBar().sizeHint().width() # add width of scrollbar          
            
        width += 2                              # TODO: do this properly, I think this is padding
        
        return QtCore.QSize(width, height)
    
    def _popup_menu(self, event):
        popup_menu = QMenu(self)
        popup_menu.addAction('Reset', lambda: self._reset_checkboxes(), 'Ctrl+R')
        popup_menu.addSeparator()
        popup_menu.addAction('Copy', lambda: self._copy(), 'Ctrl+C')
        popup_menu.addAction('Paste', lambda: self._paste(), 'Ctrl+V')
        popup_menu.exec_(QtGui.QCursor.pos())
        
    def _reset_checkboxes(self):
        for i in range(self.count()):
            item = self.model().item(i, 0)
            if self.itemChecked(i):
                item.setCheckState(QtCore.Qt.Unchecked)     # uncheck all
    
    def _copy(self):
        text = str(self.currentText())
        self.cb.clear()
        self.cb.setText(text) # tab for new column, new line for new row
    
    def _paste(self):
        self.lineEdit().setText(self.cb.text())

class MessageWindow(QWidget):
    def __init__(self, parent, text):
        super().__init__(parent=parent)
        n = 7 # Margin size
        layout = QVBoxLayout()
        layout.setContentsMargins(n+1, n, n+1, n)
        self.label = QLabel(text)
        layout.addWidget(self.label)
        self.setLayout(layout)
        

        self.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.CustomizeWindowHint | QtCore.Qt.FramelessWindowHint)
        self.show()