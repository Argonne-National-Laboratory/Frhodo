# This file is part of Frhodo. Copyright © 2020, UChicago Argonne, LLC
# and licensed under BSD-3-Clause. See License.txt in the top-level 
# directory for license and copyright information.

import numpy as np
from qtpy.QtWidgets import *
from qtpy import QtWidgets, QtGui, QtCore
from copy import deepcopy

import misc_widget

class Observable_Widgets(QtCore.QObject):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        
        self.max_history = 1
        self.var_choice = []
        self.widget = {}
        self.updating_boxes = False
        
        # Limit observable options to subset of total
        keys = ['Temperature', 'Pressure', 'Density Gradient', 'Mole Fraction', 'Mass Fraction', 'Concentration']
        self.var_dict = {key:parent.SIM.all_var[key] for key in keys}
        
        self.create_choices()
    
    def create_choices(self):
        parent = self.parent
        
        spacer = QtWidgets.QSpacerItem(10, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        main_parameter_box = misc_widget.ItemSearchComboBox()
        sub_parameter_box = misc_widget.ItemSearchComboBox()
        # sub_parameter_box = misc_widget.CheckableSearchComboBox(parent)
        
        self.widget['main_parameter'] = main_parameter_box
        self.widget['sub_parameter'] = sub_parameter_box
        
        sub_parameter_box.hide()
        sub_parameter_box.checked = []
        
        self.populate_mainComboBox()
        
        layout = parent.observable_layout
        layout.addWidget(main_parameter_box, 0, 0)
        layout.addItem(spacer, 0, 1)
        layout.addWidget(sub_parameter_box, 0, 2)
        
        # connect signals
        main_parameter_box.currentIndexChanged[str].connect(self.main_parameter_changed)
        sub_parameter_box.currentIndexChanged[str].connect(self.sub_item_changed)
    
    def populate_mainComboBox(self):
        parent = self.parent
        choices = list(self.var_dict.keys())
        if 'name' in parent.var['reactor']: # remove density gradient as option for 0d reactors
            if parent.var['reactor']['name'] != 'Incident Shock Reactor':
                choices.remove('Density Gradient')
        
        comboBox = self.widget['main_parameter']
        prior_choice = comboBox.currentText()
        comboBox.blockSignals(True)
        comboBox.clear()
        comboBox.addItems(choices)
        comboBox.blockSignals(False)
        
        if 'name' not in parent.var['reactor']:
            comboBox.setCurrentIndex(2)       # TODO: MAKE SAVE, THIS SETS DENSITY GRADIENT AS DEFAULT
        else:
            if parent.var['reactor']['name'] == 'Incident Shock Reactor':
                comboBox.setCurrentIndex(2)       # TODO: MAKE SAVE, THIS SETS DENSITY GRADIENT AS DEFAULT
            elif prior_choice == 'Density Gradient':
                comboBox.setCurrentIndex(4)       # TODO: MAKE SAVE, SETS CONCENTRATION
            else:
                comboBox.setCurrentIndex(comboBox.findText(prior_choice))   # set choice to prior
    
    def populate_subComboBox(self, param):
        if not self.parent.mech_loaded: return # if mech isn't loaded successfully, exit
        
        comboBox = self.widget['sub_parameter']
        comboBox.blockSignals(True)
        comboBox.clear()
        comboBox.checked = []
        
        if (param == '-' or self.var_dict[param]['sub_type'] is None or not hasattr(self.parent, 'mech') or 
            self.widget['main_parameter'].currentText() == 'Density Gradient'):
            comboBox.hide()
        else:
            comboBox.show()
            
            if 'total' in self.var_dict[param]['sub_type']:
                comboBox.addItem('Total')
            
            if 'species' in self.var_dict[param]['sub_type']:
                for n, species in enumerate(self.parent.mech.gas.species_names):
                    comboBox.addItem(species)
                
            elif 'rxn' in self.var_dict[param]['sub_type']:
                for n, rxn in enumerate(self.parent.mech.gas.reaction_equations()):
                    comboBox.addItem('R{:d}:  {:s}'.format(n+1, rxn.replace('<=>', '=')))
        
        comboBox.blockSignals(False)
    
    def main_parameter_changed(self, event):
        if not self.parent.mech_loaded: return # if mech isn't loaded successfully, exit
        
        self.populate_subComboBox(event)
        self.update_observable()
        self.parent.run_single()    # rerun simulation and update plot
        
    def sub_item_changed(self, sender):
        self.update_observable()
        self.parent.run_single()    # rerun simulation and update plot
    
    def update_observable(self):
        mainComboBox = self.widget['main_parameter']
        subComboBox = self.widget['sub_parameter']
        # a nonvisible subComboBox returns -1
        self.parent.series.set('observable', [mainComboBox.currentText(), subComboBox.currentIndex()])
        
    def set_observable(self, selection):
        mainComboBox = self.widget['main_parameter']
        subComboBox = self.widget['sub_parameter']
                
        mainComboBox.setCurrentText(selection['main'])
        self.populate_subComboBox(selection['main'])
        subComboBox.setCurrentIndex(selection['sub'])