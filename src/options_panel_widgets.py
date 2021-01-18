# This file is part of Frhodo. Copyright © 2020, UChicago Argonne, LLC
# and licensed under BSD-3-Clause. See License.txt in the top-level 
# directory for license and copyright information.

import numpy as np
import nlopt, pathlib, os
import mech_widget, misc_widget, thermo_widget, series_viewer_widget, shock_fcns, save_output
from optimize.mech_optimize import Multithread_Optimize
from qtpy.QtWidgets import *
from qtpy import QtWidgets, QtGui, QtCore
from copy import deepcopy


class Initialize(QtCore.QObject):
    def __init__(self, parent):
        super().__init__(parent)

        parent.log = Log(parent.option_tab_widget, parent.log_box,
            parent.clear_log_button, parent.copy_log_button)
        
        # Setup and Connect Directory Widgets
        parent.directory = Directories(parent)
        
        # Connect and Reorder settings boxes
        box_list = [parent.shock_choice_box, parent.time_offset_box]
        
        self._set_user_settings_boxes(box_list)
        
        # Create toolbar experiment number spinbox
        parent.toolbar_shock_choice_box = QtWidgets.QSpinBox()
        parent.toolbar_shock_choice_box.setKeyboardTracking(False)
        parent.toolbar_shock_choice_box.label = QtWidgets.QAction('Shock # ')
        parent.toolbar_shock_choice_box.label.setEnabled(False)
        parent.toolbar_shock_choice_box.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        
        parent.toolBar.insertAction(parent.action_Run, parent.toolbar_shock_choice_box.label)
        parent.toolBar.insertWidget(parent.action_Run, parent.toolbar_shock_choice_box)
        parent.toolBar.insertSeparator(parent.action_Run)
        parent.toolBar.setStyleSheet("QToolButton:disabled { color: black } " + 
                                     "QToolButton:enabled { color: black }")   # alter color

        # Set twinned boxes
        self.twin = [[parent.time_offset_box, parent.time_offset_twin_box], # main box first
                     [parent.shock_choice_box, parent.toolbar_shock_choice_box]]
        for boxes in self.twin:
            for box in boxes:
                box.twin = boxes
                box.setValue(boxes[0].value())   # set all values to be main
                box.setMinimum(boxes[0].minimum())
                box.setMaximum(boxes[0].maximum())
                if box is not parent.shock_choice_box: # prevent double signals, boxes changed in settings
                    box.valueChanged.connect(self.twin_change)  

        # Connect optimization widgets
        parent.optimization_settings = Optimization(parent)
        
        # Create list of shock boxes (units and values) and connect them to function
        parent.shock_widgets = Shock_Settings(parent)
        
        # Setup tables
        parent.mix = Mix_Table(parent)
        parent.weight = Weight_Parameters_Table(parent)
        
        # Setup reactor settings
        parent.reactor_settings = Reactor_Settings(parent)
        
        # Setup and Connect Tree Widgets
        Tables_Tab(parent)
        
        # Optimize Widgets
        parent.save = save_output.Save(parent)
        parent.optimize = Multithread_Optimize(parent)       
        parent.run_optimize_button.clicked.connect(lambda: parent.optimize.start_threads())
    
    def _set_user_settings_boxes(self, box_list):
        parent = self.parent()
        box_list[0].valueChanged.connect(parent.shock_choice_changed)
        for box in box_list[1:]:
            if isinstance(box, QtWidgets.QDoubleSpinBox) or isinstance(box, QtWidgets.QSpinBox):
                box.valueChanged.connect(parent.update_user_settings)
            elif isinstance(box, QtWidgets.QComboBox):
                box.currentIndexChanged[int].connect(parent.update_user_settings)
            elif isinstance(box, QtWidgets.QCheckBox):
                box.stateChanged.connect(parent.update_user_settings)
            elif isinstance(box, QtWidgets.QTextEdit):
                box.textChanged.connect(parent.update_user_settings)
            
        box_list[0], box_list[1] = box_list[1], box_list[0] # switch list order
        for i in range(len(box_list)-1):             # Sets the box order
            parent.setTabOrder(box_list[i], box_list[i+1])
            
    def twin_change(self, event):
        if self.sender() is self.sender().twin[0]:   # if box is main, update others
            for box in self.sender().twin:
                if box is not self.sender():
                    box.blockSignals(True)           # stop changing text from signaling
                    box.setValue(event)
                    box.blockSignals(False)          # allow signals again
        else:                                                       
            self.sender().twin[0].setValue(event)    # if box isn't main, update main
        

class Directories(QtCore.QObject):
    def __init__(self, parent):
        super().__init__(parent)
        parent = self.parent()
        
        parent.exp_main_box.textChanged.connect(self.select)
        parent.exp_main_button.clicked.connect(self.select)
        parent.mech_main_box.textChanged.connect(self.select)
        parent.mech_main_button.clicked.connect(self.select)
        parent.sim_main_box.textChanged.connect(self.select)
        parent.sim_main_button.clicked.connect(self.select)
        parent.path_file_box.textChanged.connect(self.select)
        parent.path_file_load_button.clicked.connect(self.select)
        parent.path_file_save_button.clicked.connect(self.save)
        
        parent.exp_series_name_box.textChanged.connect(parent.update_user_settings)
        parent.mech_select_comboBox.activated[str].connect(parent.load_mech)    # call function if opened, even if not changed
        parent.use_thermo_file_box.stateChanged.connect(parent.load_mech)
        
        parent.load_full_series_box.stateChanged.connect(self.set_load_full_set)
        self.set_load_full_set()
        
        self.x_icon = QtGui.QPixmap(str(parent.path['graphics']/'x_icon.png'))
        self.check_icon = QtGui.QPixmap(str(parent.path['graphics']/'check_icon.png'))
        self.update_icons()
    
    def preset(self, selection):
        parent = self.parent
        
        parent.preset_settings_choice.setCurrentIndex(parent.preset_settings_choice.findText(selection))
        parent.preset_box.setPlainText(parent.path['Settings'][selection])
        parent.user_settings.load(parent.path['Settings'][selection])
    
    def select(self):
        parent = self.parent()
        
        key = '_'.join(self.sender().objectName().split("_")[:-1])
        if 'path_file_load' in key:
            key = 'path_file'
            dialog = 'load'
        else:
            dialog = 'select'
            
        type = self.sender().objectName().split("_")[-1]
        if 'button' in type:
            description_text = eval('parent.' + key + '_box.placeholderText()')
            initial_dir = pathlib.Path.home()     # set user as initial folder
            if dialog in 'select':
                # if this path exists, set previous folder as initial folder
                if key in parent.path and parent.path[key].exists() and len(parent.path[key].parts) > 1:
                    initial_dir = parent.path[key].parents[0]
                    
                path = QFileDialog.getExistingDirectory(parent, description_text, str(initial_dir))                
            elif dialog in 'load':
                if key in parent.path and len(parent.path[key].parts) > 1:
                    initial_dir = parent.path[key].parents[0] # set path_file as initial folder
                    
                    # if initial_dir doesn't exist or can't be accessed, choose source folder
                    if not os.access(parent.path[key], os.R_OK) or not initial_dir.is_dir():  
                        initial_dir = parent.path['main']
                
                path = QFileDialog.getOpenFileName(parent, description_text, str(initial_dir), 'ini (*.ini)')
                path = path[0]
                
            if path:
                path = pathlib.Path(path).resolve() # convert to absolute path
                
                if dialog in 'load': # if load is selected and path is valid
                    parent.path_set.load_dir_file(path)
                    
                eval('parent.' + key + '_box.setPlainText(str(path))')
                parent.path[key] = path
                parent.user_settings.save(save_all = False)

        elif 'box' in type:
            text = self.sender().toPlainText()
            def fn(parent, text):
                return self.sender().setPlainText(text)
            
            self.QTextEdit_function(self.sender(), fn, parent, text)
            parent.path[key] = pathlib.Path(text) 
        
            # select will modify box, this section is under if box to prevent double calling
            self.update_icons()
            if 'mech_main' in key and 'mech_main' not in self.invalid:  # Mech path changed: update mech combobox
                parent.path_set.set_watch_dir()  # update watched directory
                parent.path_set.mech()
                # if no mechs found, do not try to load, return
                if parent.mech_select_comboBox.count() == 0: return
                
                # if mech not in current path load mech
                if 'mech' not in parent.path:
                    parent.load_mech()
                else:   # load mech if path or mech name has changed
                    mech_name = str(parent.mech_select_comboBox.currentText())
                    mech_name_changed = mech_name != parent.path['mech'].name
                    
                    mech_path = parent.path['mech_main']
                    mech_path_changed = mech_path != parent.path['mech'].parents[0]
                    
                    if mech_name_changed or mech_path_changed:
                        parent.load_mech()
                
                if parent.mech.isLoaded:     # this is causing the mix table to be blanked out
                    parent.mix.update_species()
                    # parent.mix.setItems(parent.mech.gas.species_names)
            elif 'exp_main' in key and 'exp_main' not in self.invalid:  # Exp path changed: reload list of shocks and load data
                series_name = parent.exp_series_name_box.text() 
                if parent.exp_main_box.toPlainText() not in parent.series.path: # if series already exists, don't create new
                    parent.series.add_series()
                    
                    if not series_name or series_name in parent.series.name:
                        exp_path = parent.path['exp_main']
                        parent.exp_series_name_box.setText(str(exp_path.name))
                else:
                    parent.series.change_series()
                    parent.exp_series_name_box.setText(parent.display_shock['series_name'])
                    
                    if not series_name:
                        parent.exp_series_name_box.setText(str(exp_path.name))

    def save(self):
        parent = self.parent()
       
        description_text = 'Save Directory Settings'
        default_location = str(parent.path['path_file'])
        path = QFileDialog.getSaveFileName(parent, description_text, default_location, 
                           "Configuration file (*.ini)")
        
        if path[0] and 'exp_main' not in self.invalid:
            parent.path_set.save_dir_file(path[0])
            parent.path_file_box.setPlainText(path[0])
            parent.user_settings.save(save_all = False)
        elif self.invalid:
            parent.log.append('Could not save directory settings:\nInvalid directory found')
            
    
    def QTextEdit_function(self, object, fn, *args, **kwargs):
        object.blockSignals(True)             # stop changing text from signalling
        old_position = object.textCursor().position()    # find old cursor position
        fn(*args, **kwargs)
        
        cursor = object.textCursor()          # create new cursor (I don't know why)
        cursor.setPosition(old_position)      # move new cursor to old pos
        object.setTextCursor(cursor)          # switch current cursor with newly made
        object.blockSignals(False)            # allow signals again
    
    def update_icons(self, invalid=[]): # This also checks if paths are valid
        parent = self.parent()
        
        key_names = ['path_file', 'exp_main', 'mech_main', 'sim_main']
                                
        self.invalid = deepcopy(invalid)
        for key in key_names:
            if key is 'path_file':
                if key in parent.path and os.access(parent.path[key], os.R_OK) and parent.path[key].is_file():
                    eval('parent.' + key + '_label.setPixmap(self.check_icon)')
                else:
                    eval('parent.' + key + '_label.setPixmap(self.x_icon)')
            else:
                
                if key in self.invalid:
                    eval('parent.' + key + '_label.setPixmap(self.x_icon)')
                elif (key in parent.path and os.access(parent.path[key], os.R_OK) 
                    and parent.path[key].is_dir() and str(parent.path[key]) != '.'):

                    eval('parent.' + key + '_label.setPixmap(self.check_icon)')
                else:
                    if key is not 'sim_main':    # not invalid if sim folder missing, can create later
                        self.invalid.append(key)
                    eval('parent.' + key + '_label.setPixmap(self.x_icon)')
            eval('parent.' + key + '_label.show()')
            
    def set_load_full_set(self, event=None):
        parent = self.parent()
        parent.load_full_series = parent.load_full_series_box.isChecked()
        if event:
            parent.series.load_full_series()
            # parent.series_viewer._update(load_full_series = parent.load_full_series)
    

class Shock_Settings(QtCore.QObject):
    def __init__(self, parent):
        super().__init__(parent)
        self._set_shock_boxes()
        self.convert_units = self.parent().convert_units
        self.error_msg = []
        
    def _set_shock_boxes(self):
        parent = self.parent()
        
        shock_var_list = ['T1', 'P1', 'u1', 'T2', 'P2', 'T5', 'P5']
        shock_box_list = []
        for shock_var in shock_var_list:
            value_box = eval('parent.' + shock_var + '_value_box')
            unit_box = eval('parent.' + shock_var + '_units_box')
            value_box.valueChanged.connect(self._shock_value_changed)
            unit_box.currentIndexChanged[str].connect(lambda: self._shock_unit_changed())
            shock_box_list.append(unit_box)
            shock_box_list.append(value_box)
        
        # Reorder tab list for shock boxes
        for i in range(len(shock_box_list)-1):             # Sets the box order
            parent.setTabOrder(shock_box_list[i], shock_box_list[i+1])
    
    def set_shock_value_box(self, var_type):
        parent = self.parent()
        
        unit = eval('str(parent.' + var_type + '_units_box.currentText())')
        value = parent.display_shock[var_type]
        minimum_value = self.convert_units(0.05, unit, unit_dir='out')
        display_value = self.convert_units(value, unit, unit_dir='out')
        if np.isnan(display_value):
            display_value = 0
        
        eval('parent.' + var_type + '_value_box.blockSignals(True)')
        eval('parent.' + var_type + '_value_box.setMinimum(' + str(minimum_value) + ')')
        eval('parent.' + var_type + '_value_box.setValue(' + str(display_value) + ')')
        eval('parent.' + var_type + '_value_box.blockSignals(False)')
    
    def _shock_value_changed(self, event):
        parent = self.parent()
        var_type = self.sender().objectName().split('_')[0]   
        
        # Get unit type and convert to SIM units
        units = eval('str(parent.' + var_type + '_units_box.currentText())')
        parent.display_shock[var_type] = self.convert_units(event, units, unit_dir = 'in')
        
        self.solve_postshock(var_type)
    
    def _shock_unit_changed(self):
        parent = self.parent()
        var_type = self.sender().objectName().split('_')[0]
        
        # Update spinbox
        self.set_shock_value_box(var_type)
        parent.plot.signal.update_info_text(redraw=True)    # update info text box
    
    def solve_postshock(self, var_type):
        parent = self.parent()
        print2log = True
        if parent.path_set.loading_dir_file and len(parent.series.current['species_alias']) > 0:
            print2log = False

        if not hasattr(parent.mech.gas, 'species_names'): # Check mechanism is loaded
            return
        
        # Check that the variables exist to calculate post shock conditions
        IC = [parent.display_shock[key] for key in ['T1', 'P1']]
        if not np.isnan(IC).any() and len(parent.display_shock['thermo_mix']) > 0:  # if T1, P1, thermo_mix all valid
            IC = [parent.display_shock[key] for key in ['u1', 'T2', 'P2', 'T5', 'P5']] 
            nonzero_count = np.count_nonzero(~np.isnan(IC)) # count existing values of secondary IC's
            if nonzero_count == 0:
                self.error_msg.append('Not enough shock variables to calculate postshock conditions')
        else:
            self.error_msg.append('Not enough shock variables to calculate postshock conditions')
        
        for species in parent.display_shock['thermo_mix']:
            if species not in parent.mech.gas.species_names:
                self.error_msg.append('Species: {:s} is not in the mechanism'.format(species))

        if len(self.error_msg) > 0:
            if print2log:   # do not print to log if loading_dir_file
                for err in self.error_msg:
                    parent.log.append(err)
                
            self.error_msg = [] # reset error message and return
            return
        
        # Setup variables to be sent to shock solver
        # Assume T1, mix + variables from selected zone are known variables
        shock_vars = {'T1': parent.display_shock['T1'], 'mix': parent.display_shock['thermo_mix']}
        if '1' in var_type:
            shock_vars['P1'] = parent.display_shock['P1']
            shock_vars['u1'] = parent.display_shock['u1']
        elif '2' in var_type:
            shock_vars['T2'] = parent.display_shock['T2']
            shock_vars['P2'] = parent.display_shock['P2']
        elif '5' in var_type:
            shock_vars['T5'] = parent.display_shock['T5']
            shock_vars['P5'] = parent.display_shock['P5']
        
        # Solve for new values
        shock = shock_fcns.Properties(parent.mech.gas, shock_vars, parent=parent)
        self.success = shock.success
        
        if shock.success:
            parent.log._blink(False)
        else:
            return
            
        # Update new values and run sim
        # Remove set shock_vars
        vars = list(set(shock_vars.keys())^set(['u1', 'T1', 'P1', 'T2', 'P2', 'T5', 'P5']))
        for var in vars:
            parent.display_shock[var] = shock.res[var]
            self.set_shock_value_box(var)
        
        # Set reactor conditions
        parent.series.set('zone', parent.display_shock['zone'])
        
        parent.display_shock['u2'] = shock.res['u2']
        parent.display_shock['rho1'] = shock.res['rho1']
        
        parent.tree.update_rates()    # Updates the rate constants
        parent.tree.update_uncertainties() # update rate constants uncertainty
        parent.run_single()
                

class Reactor_Settings(QtCore.QObject):
    def __init__(self, parent):
        super().__init__(parent)
        self._set_reactor_boxes()
        self.update_reactor_choice(event=None)
        self.update_reactor_variables(event=None)
        
    def _set_reactor_boxes(self):
        parent = prnt = self.parent()
        
        boxes = [prnt.solve_energy_box, prnt.frozen_comp_box, prnt.end_time_units_box, 
                 prnt.end_time_value_box, prnt.ODE_solver_box, prnt.sim_interp_factor_box, 
                 prnt.ODE_rtol_box, prnt.ODE_atol_box]
    
        for box in boxes:
            if isinstance(box, QtWidgets.QDoubleSpinBox) or isinstance(box, QtWidgets.QSpinBox):
                box.valueChanged.connect(self.update_reactor_variables)
            elif isinstance(box, QtWidgets.QComboBox):
                box.currentIndexChanged[int].connect(self.update_reactor_variables)
            elif isinstance(box, QtWidgets.QCheckBox):
                box.stateChanged.connect(self.update_reactor_variables)
            elif isinstance(box, QtWidgets.QTextEdit):
                box.textChanged.connect(self.update_reactor_variables)
        
        prnt.reactor_select_box.currentIndexChanged[str].connect(self.update_reactor_choice)
        
    def update_reactor_variables(self, event=None):
        parent = self.parent()
        
        parent.var['reactor']['solve_energy'] = parent.solve_energy_box.isChecked()
        parent.var['reactor']['frozen_comp'] = parent.frozen_comp_box.isChecked()
        
        # Set Simulation time
        if 'μs' in parent.end_time_units_box.currentText():
            t_unit_conv = parent.var['reactor']['t_unit_conv'] = 1E-6
        elif 'ms' in parent.end_time_units_box.currentText():
            t_unit_conv = parent.var['reactor']['t_unit_conv'] = 1E-3
        elif 's' in parent.end_time_units_box.currentText():
            t_unit_conv = parent.var['reactor']['t_unit_conv'] = 1
        
        t_unit = parent.end_time_units_box.currentText()
        parent.time_offset_box.setSuffix(' ' + t_unit)
        
        parent.var['reactor']['ode_solver'] = parent.ODE_solver_box.currentText()
        parent.var['reactor']['ode_rtol'] = 10**parent.ODE_rtol_box.value()
        parent.var['reactor']['ode_atol'] = 10**parent.ODE_atol_box.value()
        parent.var['reactor']['t_end'] = parent.end_time_value_box.value()*t_unit_conv
        parent.var['reactor']['sim_interp_factor'] = parent.sim_interp_factor_box.value()
        
        if event is not None:
            sender = self.sender().objectName()
            parent.run_single()
                
            # if 'time_offset' in sender and hasattr(self, 'SIM'): # Don't rerun SIM if it exists
                # if hasattr(self.SIM, 'independent_var') and hasattr(self.SIM, 'observable'):
                    # self.plot.signal.update_sim(self.SIM.independent_var, self.SIM.observable)
            # elif any(x in sender for x in ['end_time', 'sim_interp_factor', 'ODE_solver', 'rtol', 'atol']):
                # self.run_single()
            # elif self.display_shock['exp_data'].size > 0: # If exp_data exists
                # self.plot.signal.update(update_lim=False)
                # self.plot.signal.canvas.draw()
        
    def update_reactor_choice(self, event=None):
        parent = self.parent()
        
        parent.var['reactor']['name'] = parent.reactor_select_box.currentText()
        parent.plot.observable_widget.populate_mainComboBox()   # update observables (delete density gradient from 0d)
        
        # hide/show choices based on selection
        if parent.var['reactor']['name'] == 'Incident Shock Reactor':
            parent.zero_d_choice_frame.hide()
            parent.solver_frame.show()
            parent.series.set('zone', 2)
        elif '0d Reactor' in parent.var['reactor']['name']:
            parent.zero_d_choice_frame.show()
            parent.solver_frame.hide()
            parent.series.set('zone', 5)
            
        if event is not None:
            sender = self.sender().objectName()
            parent.run_single()


class CheckableTabWidget(QTabWidget): # defunct TODO: this would be a good way to select the zone
    checkBoxList = []
    def addTab(self, widget, title):
        QTabWidget.addTab(self, widget, title)
        checkBox = QCheckBox()
        self.checkBoxList.append(checkBox)
        self.tabBar().setTabButton(self.tabBar().count()-1, QTabBar.LeftSide, checkBox)
        self.connect(checkBox, QtCore.SIGNAL('stateChanged(int)'), lambda checkState: self.__emitStateChanged(checkBox, checkState))

    def isChecked(self, index):
        return self.tabBar().tabButton(index, QTabBar.LeftSide).checkState() != QtCore.Qt.Unchecked

    def setCheckState(self, index, checkState):
        self.tabBar().tabButton(index, QTabBar.LeftSide).setCheckState(checkState)

    def __emitStateChanged(self, checkBox, checkState):
        index = self.checkBoxList.index(checkBox)
        self.emit(QtCore.SIGNAL('stateChanged(int, int)'), index, checkState)

        
class Mix_Table(QtCore.QObject):
    def __init__(self, parent):
        super().__init__(parent)
        self.table = self.parent().mix_table

        stylesheet = ["QHeaderView::section{",  # stylesheet because windows 10 doesn't show borders on the bottom
            "border-top:0px solid #D8D8D8;",
            "border-left:0px solid #D8D8D8;",
            "border-right:1px solid #D8D8D8;",
            "border-bottom:1px solid #D8D8D8;",
            # "background-color:white;",                                        # this matches windows 10 theme 
            "background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,"     # this matches windows 7 theme perfectly
                                 "stop: 0 #ffffff, stop: 1.0 #f1f2f4);"
            "padding:4px;",
        "}",
        "QTableCornerButton::section{",
            "border-top:0px solid #D8D8D8;",
            "border-left:0px solid #D8D8D8;",
            "border-right:1px solid #D8D8D8;",
            "border-bottom:1px solid #D8D8D8;",
            "background-color:white;",
        "}"]
        
        header = self.table.horizontalHeader()   
        header.setStyleSheet(' '.join(stylesheet))
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Interactive)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.Fixed)
        header.resizeSection(2, 60) # Force size of Mol Frac column
        header.setFixedHeight(24)
        
        self.setItems(species=[], exp_mix=[], alias=[])
        self.table.itemChanged.connect(self.update_mix)
        
    def create_thermo_boxes(self, species=[]):
        species.insert(0, '')
        self.thermoSpecies_box = []
        # create down_arrow_path with forward slashes as required by QT stylesheet url
        down_arrow_path = '"' + str((self.parent().path['graphics']/'arrowdown.png').as_posix()) + '"'
        for row in range(self.table.rowCount()):
            self.thermoSpecies_box.append(misc_widget.SearchComboBox())
            self.thermoSpecies_box[-1].addItems(species)
            self.thermoSpecies_box[-1].currentIndexChanged[str].connect(self.update_mix)
            self.table.setCellWidget(row, 1, self.thermoSpecies_box[-1])
            self.thermoSpecies_box[-1].setNewStyleSheet(down_arrow_path)   
    
    def create_molFrac_boxes(self, allMolFrac=[]):
        self.molFrac_box = []
        for row in range(self.table.rowCount()):
            if len(allMolFrac) - 1 < row:
                molFrac = 0
            else:
                molFrac = allMolFrac[row]
                
            self.molFrac_box.append(misc_widget.ScientificDoubleSpinBox(parent=self.parent(), value=molFrac))
            self.molFrac_box[-1].setMinimum(0)
            self.molFrac_box[-1].setMaximum(1)
            self.molFrac_box[-1].setSingleStep(0.001)
            self.molFrac_box[-1].setSpecialValueText('-')
            self.molFrac_box[-1].setFrame(False)
            self.molFrac_box[-1].valueChanged.connect(self.update_mix)
            self.table.setCellWidget(row, 2, self.molFrac_box[-1])
    
    def update_mix(self, event=None):
        def isPopStr(str):  # is populated string
            return not not str.strip()
        
        def isValidRow(table, row):
            if self.molFrac_box[row].value() == 0:
                return False
            if table.item(row, 0) is not None and isPopStr(table.item(row, 0).text()):
                return True
            elif str(self.thermoSpecies_box[row].currentText()):
                return True
            else:
                return False
        
        parent = self.parent()
        valid_row = []
        for row in range(self.table.rowCount()): 
            if isValidRow(self.table, row):
                valid_row.append(row)
        
        save_species_alias = False  # do not save aliases if no original alias and none added
        if len(parent.series.current['species_alias']) > 0:
            save_species_alias = True

        # parent.series.current['species_alias'] = {} # set to empty dict and create from boxes
        parent.display_shock['exp_mix'] = {}
        for row in valid_row:
            molFrac = self.molFrac_box[row].value()
            thermo_name = str(self.thermoSpecies_box[row].currentText())
            if self.table.item(row, 0) is None:
                exp_name = thermo_name
            else:
                exp_name = self.table.item(row, 0).text()

            if thermo_name: # If experimental and thermo name exist update aliases
                if self.table.item(row, 0) is not None and isPopStr(exp_name):
                    parent.series.current['species_alias'][exp_name] = thermo_name
            elif exp_name in parent.series.current['species_alias']:
                del parent.series.current['species_alias'][exp_name]
            
            parent.display_shock['exp_mix'][exp_name] = molFrac
        
        # if path_file exists and species_aliases exist and not loading preset, save aliases
        if save_species_alias or len(parent.series.current['species_alias']) > 0:
            if parent.path['path_file'].is_file() and not parent.path_set.loading_dir_file:    
                parent.path_set.save_aliases(parent.path['path_file'])
        
        parent.series.thermo_mix()
        parent.shock_widgets.solve_postshock('T1') # Updates Post-Shock conditions and SIM  
        
    def setItems(self, species=[], exp_mix=[], alias=[]):
        self.table.blockSignals(True)
        self.table.clearContents()
        self.create_thermo_boxes(species)
        if not exp_mix:
            self.create_molFrac_boxes([])
        else:
            self.create_molFrac_boxes([*exp_mix.values()])
           
            for n, (name, molFrac) in enumerate(exp_mix.items()):
                self.table.setItem(n, 0, QTableWidgetItem(name))
                if name in alias:
                    box = self.thermoSpecies_box[n]
                    box.blockSignals(True)
                    box.setCurrentIndex(box.findText(alias[name]))
                    box.blockSignals(False)
        
        # self.table.resizeColumnsToContents()
        self.table.blockSignals(False)
        if len(species) > 0 and species != ['']:
            self.update_mix()
               
    def update_species(self):   # may be better to pass variables than call from parent?
        parent = self.parent()
        exp_mix = parent.display_shock['exp_mix']
        species_alias = parent.series.current['species_alias']
        if hasattr(parent.mech.gas, 'species_names'):   # if mech exists, set mix table with mech species in thermo box
            self.setItems(parent.mech.gas.species_names, 
                    exp_mix = exp_mix, alias=species_alias)
        else:
            self.setItems([], exp_mix=exp_mix, alias=species_alias)


class Weight_Parameters_Table(QtCore.QObject):
    def __init__(self, parent):
        super().__init__(parent)
        self.table = self.parent().weight_fcn_table

        stylesheet = ["QHeaderView::section{",  # stylesheet because windows 10 doesn't show borders on the bottom
            "border-top:0px solid #D8D8D8;",
            "border-left:0px solid #D8D8D8;",
            "border-right:1px solid #D8D8D8;",
            "border-bottom:1px solid #D8D8D8;",
            # "background-color:white;",                                        # this matches windows 10 theme 
            "background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,"     # this matches windows 7 theme perfectly
                                 "stop: 0 #ffffff, stop: 1.0 #f1f2f4);"
            "padding:4px;}",
        "QTableCornerButton::section{",
            "border-top:0px solid #D8D8D8;",
            "border-left:0px solid #D8D8D8;",
            "border-right:1px solid #D8D8D8;",
            "border-bottom:1px solid #D8D8D8;",
            "background-color:white;}"]
        
        header = self.table.horizontalHeader()   
        header.setStyleSheet(' '.join(stylesheet))
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        header.setFixedHeight(24)
        
        self.table.setSpan(0, 0, 1, 2)  # make first row span entire length
        
        self.create_boxes()
        self.table.itemChanged.connect(self.update)
    
    def create_boxes(self):
        parent = self.parent()
        # self.table.setStyleSheet("QTableWidget::item { margin-left: 10px }")
        # TODO: Change to saved variables
        self.boxes = {'weight_max': [], 'weight_min': [], 'weight_shift': [], 'weight_k': []}
        self.prop = {'start': {'weight_max':   {'value': 100,   'singleStep': 1,    'maximum': 100, 
                                                'minimum': 0,   'decimals': 3,      'suffix': '%'},
                               'weight_min':   {'value': 0,     'singleStep': 1,    'maximum': 100, 
                                                'minimum': 0,   'decimals': 3,      'suffix': '%'},
                               'weight_shift': {'value': 4.5,   'singleStep': 0.1,  'maximum': 100,
                                                'minimum': 0,   'decimals': 3,      'suffix': '%'},              
                               'weight_k':     {'value': 0,     'singleStep': 0.01, 'decimals': 3,
                                                'minimum': 0}},
                     'end':   {'weight_min':   {'value': 0,     'singleStep': 1,    'maximum': 100, 
                                                'minimum': 0,   'decimals': 3,      'suffix': '%'},
                               'weight_shift': {'value': 36.0,  'singleStep': 0.1,  'maximum': 100,
                                                'minimum': 0,   'decimals': 3,      'suffix': '%'},
                               'weight_k':     {'value': 0.3,   'singleStep': 0.01, 'decimals': 3,
                                                'minimum': 0}}}
                          
        for j, col in enumerate(['start', 'end']):
            for i, row in enumerate(self.prop[col]):
                box_val = self.prop[col][row]['value']
                box = misc_widget.ScientificDoubleSpinBox(parent=self.parent(), value=box_val)
               
                box.setSingleStep(self.prop[col][row]['singleStep'])
                box.setStrDecimals(self.prop[col][row]['decimals'])
                box.setMinimum(self.prop[col][row]['minimum'])
                if 'suffix' in self.prop[col][row]:
                    box.setSuffix(self.prop[col][row]['suffix'])
                if 'maximum' in self.prop[col][row]:
                    box.setMaximum(self.prop[col][row]['maximum'])
                box.setFrame(False)
                box.info = [col, row]
                
                box.valueChanged.connect(self.update)
                self.table.setCellWidget(i+j, j, box)
                self.boxes[row].append(box)
    
    def set_boxes(self, shock=None):
        parent = self.parent()
        if shock is None:
            shock = parent.display_shock
            
        for j, col in enumerate(['start', 'end']):
            for i, row in enumerate(self.prop[col]):
                box_val = shock[row][j]
                box = self.boxes[row][j]
                box.blockSignals(True)
                box.setValue(box_val)
                box.blockSignals(False)
    
    def update(self, event=None, shock=None):
        parent = self.parent()
        update_plot = False
        if shock is None:           # if no shock given, must be from widgets
            shock = parent.display_shock
            update_plot = True
        
        shock['weight_max'] = [self.boxes['weight_max'][0].value()]
        shock['weight_min'] = [box.value() for box in self.boxes['weight_min']]
        shock['weight_shift'] = [box.value() for box in self.boxes['weight_shift']]
        shock['weight_k'] = [box.value() for box in self.boxes['weight_k']]

        if parent.display_shock['exp_data'].size > 0 and update_plot: # If exp_data exists
            parent.plot.signal.update(update_lim=False)
            parent.plot.signal.canvas.draw()
        
         
class Tables_Tab(QtCore.QObject):
    def __init__(self, parent):
        super().__init__(parent)
        parent = self.parent()
        self.tabwidget = parent.tab_stacked_widget
        
        # Initialize and Connect Tree Widgets
        parent.tree = mech_widget.Tree(parent)                  # TODO: make dict of trees and types
        parent.tree_thermo = thermo_widget.Tree(parent)         # TODO: MAKE THIS
        parent.series_viewer = series_viewer_widget.Series_Viewer(parent)
        
        selector = parent.tab_select_comboBox
        selector.currentIndexChanged[str].connect(self.select)       
    
    def select(self, event):
        parent = self.parent()
        if 'Mechanism' in event:
            self.tabwidget.setCurrentWidget(self.tabwidget.findChild(QWidget, 'mech_tab'))
            if 'Bilbo' in event:
                parent.tree.mech_tree_type = 'Bilbo'
            elif 'Chemkin' in event:
                parent.tree.mech_tree_type = 'Chemkin'
            
            if parent.mech_loaded:  # if mech is loaded successfully, update display type
                parent.tree.update_display_type()
        elif 'Thermodynamics' in event:
            self.tabwidget.setCurrentWidget(self.tabwidget.findChild(QWidget, 'thermo_tab'))
        elif 'Series Viewer' in event:
            self.tabwidget.setCurrentWidget(self.tabwidget.findChild(QWidget, 'series_viewer_tab'))
  
  
class Log:
    def __init__(self, tab_widget, log_box, clear_log_button, copy_log_button):
        self.tab_widget = tab_widget
        self.log = log_box
        self.log_tab = self.tab_widget.findChild(QWidget, 'log_tab')
        self.log_tab_idx = self.tab_widget.indexOf(self.log_tab)
        self.color = {'base': self.tab_widget.tabBar().tabTextColor(self.log_tab_idx),
                      'gold': QtGui.QColor(255, 191, 0)}
        self.current_color = self.color['base']
        self.blink_status = False
        self.log.setTabStopWidth(QtGui.QFontMetricsF(self.log.font()).width(' ') * 6)

        # self.tab_widget.tabBar().setStyleSheet('background-color: yellow')

        # Connect Log Functions
        self.tab_widget.currentChanged.connect(self._tab_widget_change)
        clear_log_button.clicked.connect(self.clear)
        copy_log_button.clicked.connect(self.copy)
        
    def append(self, message, alert=True):
        if isinstance(message, list):
            message = '\n'.join(message)
            
        self.log.append('{}'.format(message))
        if alert and self.tab_widget.currentIndex() != self.log_tab_idx:
            self._blink(True)
    
    def _tab_widget_change(self, event):
        if event == self.log_tab_idx:
            self._blink(False)
    
    def _blink(self, blink_on):
        if blink_on:
            if not self.blink_status:   # if not blinking, set timer and start
                self.timer = QtCore.QTimer()
                self.timer.timeout.connect(lambda: self._blink(True))
                self.timer.start(500)
                
            self.blink_status = True
            if self.current_color is self.color['base']:
                self.tab_widget.tabBar().setTabTextColor(self.log_tab_idx, self.color['gold'])
                self.current_color =  self.color['gold']
            elif self.current_color is self.color['gold']:
                self.tab_widget.tabBar().setTabTextColor(self.log_tab_idx, self.color['base'])
                self.current_color =  self.color['base']
        elif not blink_on or self.blink_status:
            self.blink_status = False
            if hasattr(self, 'timer'):
                self.timer.stop()
            self.tab_widget.tabBar().setTabTextColor(self.log_tab_idx, self.color['base'])
            self.current_color =  self.color['base']
    
    def clear(self, event=None):
        if event is not None:
            self._blink(False)
        self.log.clear()
         
    def copy(self, event):
        def fn(self):
            self.log.selectAll()
            self.log.copy()
        
        self.QTextEdit_function(self.log, fn, self)
        
    def QTextEdit_function(self, object, fn, *args, **kwargs):
        signal = object.blockSignals(True)    # stop changing text from signaling
        old_position = object.textCursor().position()    # find old cursor position
        cursor = object.textCursor()          # create new cursor (I don't know why)
        cursor.movePosition(old_position)          # move new cursor to old pos
        fn(*args, **kwargs)
        object.setTextCursor(cursor)          # switch current cursor with newly made
        object.blockSignals(signal)           # allow signals again
  
  
optAlgorithm = {'DIRECT': nlopt.GN_DIRECT, 
                'DIRECT-L': nlopt.GN_DIRECT_L,
                'MLSL (Multi-Level Single-Linkage)': nlopt.GN_MLSL_LDS, #GN_MLSL
                'ISRES': nlopt.GN_ISRES,
                'CRS (Controlled Random Search)': nlopt.GN_CRS2_LM,
                'Evolutionary': nlopt.GN_ESCH,
                'Nelder-Mead Simplex': nlopt.LN_NELDERMEAD,
                'Subplex': nlopt.LN_SBPLX,
                'COBYLA': nlopt.LN_COBYLA,
                'BOBYQA': nlopt.LN_BOBYQA}   
                
class Optimization(QtCore.QObject):
    def __init__(self, parent): # TODO: Setting tab order needs to happen here
        super().__init__(parent)
        parent = self.parent()
        self.settings = {'obj_fcn': {}, 'global': {}, 'local': {}}
        
        for box in [parent.loss_alpha_box, parent.loss_c_box, parent.bayes_unc_sigma_box]:
            box.valueChanged.connect(self.update_obj_fcn_settings)
        for box in [parent.obj_fcn_type_box, parent.obj_fcn_scale_box, parent.bayes_dist_type_box]:
            box.currentTextChanged.connect(self.update_obj_fcn_settings)
        
        self.update_obj_fcn_settings() # initialize settings
        
        parent.multiprocessing_box  # checkbox
        
        self.widgets = {'global': {'run': parent.global_opt_enable_box,
                                   'algorithm': parent.global_opt_choice_box, 'initial_step': [],
                                   'xtol_rel': [], 'ftol_rel': []},
                        'local': {'run': parent.local_opt_enable_box,
                                   'algorithm': parent.local_opt_choice_box, 'initial_step': [],
                                   'xtol_rel': [], 'ftol_rel': []}}
        
        self.labels = {'global': [parent.global_text_1, parent.global_text_2, parent.global_text_3],
                       'local':  [parent.local_text_1, parent.local_text_2, parent.local_text_3]}
        
        self._create_spinboxes()
        
        for opt_type, boxes in self.widgets.items():
            for var_type, box in boxes.items():
                self.widgets[opt_type][var_type].info = {'opt_type': opt_type, 'var': var_type}
                
                if isinstance(box, QtWidgets.QDoubleSpinBox) or isinstance(box, QtWidgets.QSpinBox):
                    box.valueChanged.connect(self.update_opt_settings)
                elif isinstance(box, QtWidgets.QComboBox):
                    box.currentIndexChanged[int].connect(self.update_opt_settings)
                elif isinstance(box, QtWidgets.QCheckBox):
                    box.stateChanged.connect(self.update_opt_settings)
        
        self.update_opt_settings()
     
    def _create_spinboxes(self):
        parent = self.parent()
        layout = {'global': parent.global_opt_layout, 'local': parent.local_opt_layout}
        vars = {'global': {'initial_step': 1E-2, 'xtol_rel': 1E-4, 'ftol_rel': 5E-4},
                'local':  {'initial_step': 1E-2, 'xtol_rel': 1E-4, 'ftol_rel': 1E-3}}
        
        spinbox = misc_widget.ScientificDoubleSpinBox
        for opt_type, layout in layout.items():
            for n, (var_type, val) in enumerate(vars[opt_type].items()):
                self.widgets[opt_type][var_type] = spinbox(parent=parent, value=val, numFormat='e')
                self.widgets[opt_type][var_type].setSingleStep(0.1)
                self.widgets[opt_type][var_type].setStrDecimals(1)
                layout.addWidget(self.widgets[opt_type][var_type], n, 0)
             
    def update_obj_fcn_settings(self, event=None):
        parent = self.parent()
        settings = self.settings['obj_fcn']
        
        settings['type'] = parent.obj_fcn_type_box.currentText()
        settings['scale'] = parent.obj_fcn_scale_box.currentText()

        settings['alpha'] = parent.loss_alpha_box.value()
        settings['c'] = parent.loss_c_box.value()

        settings['bayes_dist_type'] = parent.bayes_dist_type_box.currentText()
        settings['bayes_unc_sigma'] = parent.bayes_unc_sigma_box.value()

        self.save_settings(event)
         
    def update_opt_settings(self, event=None):
        if event is not None:
            box = self.sender()
            opt_type = box.info['opt_type']
            var_type = box.info['var']
            
            if var_type == 'run':
                self.settings[opt_type]['run'] = box.isChecked()
                for box in list(self.widgets[opt_type].values()) + self.labels[opt_type]:
                    if box is not self.sender():
                        box.setEnabled(self.settings[opt_type]['run'])
                return
            
            elif var_type == 'algorithm':
                if opt_type == 'global':
                    if box.currentText() == 'MLSL (Multi-Level Single-Linkage)':
                        self.widgets['local']['run'].setEnabled(False)
                        self.widgets['local']['run'].setChecked(True)
                    else: 
                        self.widgets['local']['run'].setEnabled(True)
        
        for opt_type, boxes in self.widgets.items():
            for var_type, box in self.widgets[opt_type].items():
                if isinstance(box, QtWidgets.QDoubleSpinBox) or isinstance(box, QtWidgets.QSpinBox):
                    self.settings[opt_type][var_type] = box.value()
                elif isinstance(box, QtWidgets.QComboBox):
                    self.settings[opt_type][var_type] = optAlgorithm[box.currentText()]
                elif isinstance(box, QtWidgets.QCheckBox):
                    self.settings[opt_type][var_type] = box.isChecked()
        
        self.save_settings(event)
    
    def save_settings(self, event=None):
        if event is None: return
        if not hasattr(self.parent(), 'user_settings'): return
        if 'path_file' not in self.parent().path: return

        self.parent().user_settings.save()

    def get(self, opt_type, var_type):
        return self.settings[opt_type][var_type]
        