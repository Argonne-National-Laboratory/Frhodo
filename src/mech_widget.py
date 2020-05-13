# This file is part of Frhodo. Copyright © 2020, UChicago Argonne, LLC
# and licensed under BSD-3-Clause. See License.txt in the top-level 
# directory for license and copyright information.

import sys, ast, re
import misc_widget
import cantera as ct
import numpy as np
from copy import deepcopy
from functools import partial
from scipy.optimize import root_scalar
from qtpy.QtWidgets import *
from qtpy import QtWidgets, QtGui, QtCore

from timeit import default_timer as timer

def silentSetValue(obj, value):
    obj.blockSignals(True)           # stop changing text from signaling
    obj.setValue(value)
    obj.blockSignals(False)          # allow signals again
        
class Tree(QtCore.QObject):
    def __init__(self, parent):
        super().__init__(parent)
        self.run_sim_on_change = True
        self.copyRates = False
        self.convert = parent.convert_units
        
        self.color = {'variable_rxn': QtGui.QBrush(QtGui.QColor(188, 0, 188)),
                      'fixed_rxn': QtGui.QBrush(QtGui.QColor(0, 0, 0))}
        
        parent.mech_tree.setRootIsDecorated(False)
        parent.mech_tree.setIndentation(21)
        parent.mech_tree.itemClicked.connect(self.item_clicked)
    
    def item_clicked(self, event):
        if event.isExpanded():
            event.setExpanded(False)
        else:
            event.setExpanded(True)
    
    def set_trees(self, mech):
        parent = self.parent()
        parent.mech_tree.clear()
        if 'Chemkin' in parent.tab_select_comboBox.currentText():
            self.mech_tree_type = 'Chemkin'
        else:
            self.mech_tree_type = 'Bilbo'
        self.mech_tree_data = self._set_mech_tree_data(self.mech_tree_type, mech)
        self._set_mech_tree(self.mech_tree_data)

    def _set_mech_tree_data(self, selection, mech):
        parent = self.parent()
        data = []
        for i, rxn in enumerate(mech.gas.reactions()):
            if hasattr(rxn, 'rate'):
                attrs = [p for p in dir(rxn.rate) if not p.startswith('_')] # attributes not including __
                
                # Setup Coeffs for Mech
                temp = {}
                for attr in attrs:
                    temp[attr] = getattr(rxn.rate, attr)
                
                # Setup Coeffs for Tree
                coeffs = []
                coeffs_order = []
                for n, attr in enumerate(attrs):
                    if 'activation_energy' in attr:
                        attr_short = 'Ea'
                    elif 'pre_exponential_factor' in attr:
                        attr_short = 'A'
                    elif 'temperature_exponent' in attr:
                        attr_short = 'n'
                    else:
                        attr_short = attr_name
                    
                    coeffs.append([attr_short, attr, getattr(rxn.rate, attr)])
                    coeffs_order.append(n)
                      
                if type(rxn) is ct.ElementaryReaction or type(rxn) is ct.ThreeBodyReaction:
                    # Reorder coeffs into A, n, Ea
                    coeffs_order = [1, 2, 0]
                    
                    if 'Bilbo' in selection:
                        coeffs = self.convert._arrhenius(i, coeffs, 'Cantera2Bilbo')
                    elif 'Chemkin' in selection:
                        coeffs = self.convert._arrhenius(i, coeffs, 'Cantera2Chemkin')
                  
                    data.append({'num': i, 'eqn': rxn.equation, 'type': 'Arrhenius', 
                            'coeffs': coeffs, 'coeffs_order': coeffs_order})
                
            else:
                rxn_type = rxn.__class__.__name__.replace('Reaction', ' Reaction')
                data.append({'num': i, 'eqn': rxn.equation, 'type': rxn_type})
                # raise Exception("Equation type is not currently implemented for:\n{:s}".format(rxn.equation))
            
        return data                
        
    def _set_mech_tree(self, rxn_matrix):
        parent = self.parent()
        tree = parent.mech_tree           
        tree.setColumnCount(1)
        tree.setHeaderLabels(['Reaction'])
        
        # Set up right click popup menu and linked expand/collapse
        tree.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        try:    # This tries to disconnect the previous signal, will fail for the first loaded shock
            tree.customContextMenuRequested.disconnect()
            tree.itemExpanded.disconnect()
            tree.itemCollapsed.disconnect()
        except: pass
        tree.customContextMenuRequested.connect(self._popup_menu)
        tree.itemExpanded.connect(lambda sender: self._tabExpanded(sender, True))
        tree.itemCollapsed.connect(lambda sender: self._tabExpanded(sender, False))
                
        tree.setUpdatesEnabled(False)
        tree.rxn = []
        for rxn in rxn_matrix:
            L1 = QtWidgets.QTreeWidgetItem(tree)
            L1.setText(0, ' R{:d}:   {:s}'.format(rxn['num']+1, rxn['eqn'].replace('<=>', '=')))
            L1.setToolTip(0, rxn['type'])

            L2 = QtWidgets.QTreeWidgetItem(L1)
            
            widget = rxnRate(parent, rxnType=rxn['type'])
            L2.treeWidget().setItemWidget(L2, 0, widget)
            L1.addChild(L2)
            L1.info = {'tree': tree.objectName(), 'type': 'rxn tab', 'rxnNum': rxn['num'], 'rxnType': rxn['type'],
                       'hasExpanded': False}
            
            if rxn['type'] == 'Arrhenius':
                for box in [widget.uncValBox, widget.uncTypeBox]:
                    box.info = {'type': 'rateUnc', 'rxnNum': rxn['num']}   
                widget.uncValBox.valueChanged.connect(self.update_uncertainties)       # no update between F and %

                len_coef = len(rxn['coeffs_order'])
                tree.rxn.append({'item': L1, 'num': rxn['num'], 'rxnType': rxn['type'],
                                 'coef': rxn['coeffs'], 'dependent': False, 'rateBox': widget.valueBox, 
                                 'formulaBox': [None]*len_coef, 'valueBox': [None]*len_coef, 
                                 'uncBox': [widget.uncValBox]*(len_coef+1)})
                for coefNum in rxn['coeffs_order']:
                    coef = rxn['coeffs'][coefNum]
                    
                    L2 = QtWidgets.QTreeWidgetItem(L1)
                    widget = rateExpCoefficient(parent=parent, coef=coef)
                    widget.Label.setToolTip(coef[1].replace('_', ' ').title())
                    if self.mech_tree_type == 'Bilbo':
                        widget.valueBox.setSingleStep(0.01)
                    elif self.mech_tree_type == 'Chemkin':
                        widget.valueBox.setSingleStep(0.1)
                    
                    boxes_need_info = {'value': widget.valueBox, 'formula': widget.formulaBox, 
                                       'uncValue': widget.uncValBox, 'uncType': widget.uncTypeBox}
                    for type, box in boxes_need_info.items():
                        box.info = {'type': type, 'rxnNum': rxn['num'], 'coefNum': coefNum, 'label': widget.Label,
                                    'coef': coef[0:2], 'coefAbbr': coef[0], 'coefName': coef[1], 'coefVal': coef[2]}
                    
                    widget.formulaBox.setInitialFormula()
                    widget.formulaBox.valueChanged.connect(self.update_value)
                    widget.valueBox.valueChanged.connect(self.update_value)
                    widget.valueBox.resetValueChanged.connect(self.update_mech_reset_value)
                    widget.uncValBox.valueChanged.connect(self.update_uncertainties)
                    
                    tree.rxn[-1]['formulaBox'][coefNum] = widget.formulaBox
                    tree.rxn[-1]['valueBox'][coefNum] = widget.valueBox
                    tree.rxn[-1]['uncBox'][coefNum+1] = widget.uncValBox
                    
                    L2.treeWidget().setItemWidget(L2, 0, widget)
                    L1.addChild(L2)

            else:   # if not Arrhenius, show rate only
                tree.rxn.append({'item': L1, 'num': rxn['num'], 'rxnType': rxn['type'], 'coef': [], 
                                 'dependent': False, 'rateBox': widget.valueBox, 'formulaBox': [None], 
                                 'valueBox': [None], 'uncBox': [None]})
        
        tree.setUpdatesEnabled(True)
        self.update_box_reset_values()      # updates reset values, I don't know why this is needed now
        tree.header().setStretchLastSection(True)
        tree.header().setSectionResizeMode(0, QHeaderView.ResizeToContents) #QHeaderView.Stretch) #QHeaderView.Interactive)
        
        # Set Tab Order         TODO: FIX THIS
        '''
        last_arrhenius = 0
        for i, rxn in enumerate(rxn_matrix):
            print(rxn)
            if rxn['type'] != 'Arrhenius':
                if i > 0:
                    tree.setTabOrder(tree.rxn[i-1]['rateBox'], tree.rxn[i]['rateBox'])
            else:
                o = rxn['coeffs_order']                 # order
                
                if last_arrhenius > 0:
                    rxnPrior = rxn_matrix[last_arrhenius]
                    oPrior = rxnPrior['coeffs_order']
                    # link last reaction coef value/unc/rate to first of next
                    tree.setTabOrder(tree.rxn[i-1]['rateBox'], tree.rxn[i]['rateBox'])
                    tree.setTabOrder(tree.rxn[i-1]['valueBox'][oPrior[-1]], tree.rxn[i]['valueBox'][o[0]]) 
                    tree.setTabOrder(tree.rxn[i-1]['uncBox'][1+oPrior[-1]], tree.rxn[i]['uncBox'][0])
                
                tree.setTabOrder(tree.rxn[i]['uncBox'][0], tree.rxn[i]['uncBox'][1+o[0]])              # link rate unc to first coef unc
                for j in range(len(rxn['coeffs']) - 1):   # order coefficient unc and value boxes
                    tree.setTabOrder(tree.rxn[i]['valueBox'][o[j]], tree.rxn[i]['valueBox'][o[j+1]])
                    tree.setTabOrder(tree.rxn[i]['uncBox'][1+o[j]], tree.rxn[i]['uncBox'][1+o[j+1]])

                last_arrhenius = i
        '''
    
    def update_value(self, event):
        def getRateConst(parent, rxnNum, coefName, value):
            shock = parent.display_shock
            parent.mech.coeffs[rxnNum][coefName] = value
            parent.mech.modify_reactions(parent.mech.coeffs)
            mech_out = parent.mech.set_TPX(shock['T_reactor'], shock['P_reactor'], shock['thermo_mix'])
            if not mech_out['success']:
                parent.log.append(mech_out['message'])
                return
            return parent.mech.gas.forward_rate_constants[rxnNum]
        
        parent = self.parent()
        sender = self.sender()
        rxnNum, coefNum, coefName = sender.info['rxnNum'], sender.info['coefNum'], sender.info['coefName']
        
        # track changes to rxns
        if rxnNum in parent.rxn_change_history:
            parent.rxn_change_history.remove(rxnNum)
        parent.rxn_change_history.append(rxnNum)

        # Convert coeffs to cantera
        conv_type = self.mech_tree_type + '2Cantera'
        coeffs = [*sender.info['coef'], event]
        cantera_value = self.convert._arrhenius(rxnNum, deepcopy([coeffs]), conv_type)
                
        rateLimits = parent.display_shock['rate_bnds'][rxnNum]
        coefLimits = parent.mech.coeffs_bnds[rxnNum][coefName]['limits']
        outside_limits = True    
        if not np.isnan(coefLimits).all():  # if coef limits exist, default to using these
            if cantera_value[0][2] < coefLimits[0]:
                cantera_value[0][2] = coefLimits[0]
            elif cantera_value[0][2] > coefLimits[1]:
                cantera_value[0][2] = coefLimits[1]
            else:
                outside_limits = False
            
            if outside_limits:
                conv_type = 'Cantera2' + self.mech_tree_type
                coeffs = [*sender.info['coef'], cantera_value[0][2]]
                value = self.convert._arrhenius(rxnNum, deepcopy([coeffs]), conv_type)
                silentSetValue(sender, value[0][2])
        
        elif not np.isnan(rateLimits).all():  # if rate limits exist, and coef limits do not
            rateCon = getRateConst(parent, rxnNum, coefName, cantera_value[0][2])
            
            if rateCon < rateLimits[0]:
                limViolation = 0
            elif rateCon > rateLimits[1]:
                limViolation = 1
            else:
                outside_limits = False
            
            if outside_limits:
                # Calculate correct coef value
                fcn = lambda x: getRateConst(parent, rxnNum, coefName, x) - rateLimits[limViolation]
                x0 = parent.mech.coeffs_bnds[rxnNum][coefName]['resetVal']
                if x0 == 0:
                    x1 = 1E-9
                else:
                    x1 = x0*(1-1E-9)
                sol = root_scalar(fcn, x0=x0, x1=x1, method='secant')
                cantera_value[0][2] = sol.root
                
                # Update box
                conv_type = 'Cantera2' + self.mech_tree_type
                coeffs = [*sender.info['coef'], cantera_value[0][2]]
                value = self.convert._arrhenius(rxnNum, deepcopy([coeffs]), conv_type)
                silentSetValue(sender, value[0][2])
        
        # Update mech_coeffs
        parent.mech.coeffs[rxnNum][coefName] = cantera_value[0][2]
        
        self._updateDependents()             # update dependents and rates
        parent.mech.modify_reactions(parent.mech.coeffs)
        self.update_rates(rxnNum=rxnNum)     # Must happen after the mech is changed
        
        if self.run_sim_on_change:
            parent.run_single(rxn_changed=True)   
    
    def update_rates(self, rxnNum=None):
        parent = self.parent()
        shock = parent.display_shock
        
        if not parent.mech_loaded: return # if mech isn't loaded successfully, exit
        rxn_rate = parent.series.rates(shock)   # update rates from settings
        if rxn_rate is None: return
        
        num_reac_all = np.sum(parent.mech.gas.reactant_stoich_coeffs(), axis=0)
        
        if rxnNum is not None:
            if type(rxnNum) in [list, np.ndarray]:
                rxnNumRange = rxnNum
            else:
                rxnNumRange = [rxnNum]
        else:
            rxnNumRange = range(parent.mech.gas.n_reactions)
        
        for rxnNum in rxnNumRange:
            conv = np.power(1E3, num_reac_all[rxnNum]-1)
            rxn_rate_box = parent.mech_tree.rxn[rxnNum]['rateBox']
            rxn_rate_box.setValue(np.multiply(rxn_rate[rxnNum], conv))
            
        self._copy_expanded_tab_rates()
    
    def update_uncertainties(self, event=None, sender=None):
        parent = self.parent()
        mech = parent.mech
       
        # update uncertainty spinbox
        if event is not None:   # individual uncertainty is being updated
            if sender is None:
                sender = self.sender()
            
            rxnNum = sender.info['rxnNum']
            if 'coefName' in sender.info: # this means the coef unc was changed
                coefNum, coefName = sender.info['coefNum'], sender.info['coefName'], 
                coefUncDict = mech.coeffs_bnds[rxnNum][coefName]
                uncBox = parent.mech_tree.rxn[rxnNum]['uncBox'][coefNum+1]
                
                resetVal = mech.coeffs_bnds[rxnNum][coefName]['resetVal']
                coefUncDict['value'] = uncVal = uncBox.uncValue
                coefUncDict['limits'] = limits = uncBox.uncFcn(resetVal)
                coefUncDict['type'] = uncBox.uncType
                coefUncDict['opt'] = True
                if uncVal == uncBox.minimumBaseValue: # not optimized
                    coefUncDict['opt'] = False                    
                
                # update values if they now lie outside limits
                coefValue = mech.coeffs[rxnNum][coefName] 
                if coefValue < limits[0] or coefValue > limits[1]:
                    coefBox = parent.mech_tree.rxn[rxnNum]['valueBox'][coefNum]
                    coefBox.valueChanged.emit(coefBox.value())
                
                return
            else:
                rxnNumRange = [rxnNum]
        else:
            rxnNumRange = range(mech.gas.n_reactions)
        
        for rxnNum in rxnNumRange:  # update all rate uncertainties
            rxn = parent.mech_tree.rxn[rxnNum]
            if 'Arrhenius' not in rxn['rxnType']:   # skip if not Arrhenius
                mech.rate_bnds[rxnNum]['opt'] = False
                continue
            
            mech.rate_bnds[rxnNum]['value'] = uncVal = rxn['uncBox'][0].uncValue
            mech.rate_bnds[rxnNum]['type'] = rxn['uncBox'][0].uncType
            if np.isnan(uncVal) or uncVal == rxn['uncBox'][0].minimumBaseValue: # not optimized
                mech.rate_bnds[rxnNum]['opt'] = False
                rxn['item'].setForeground(0, self.color['fixed_rxn'])
            else:
                mech.rate_bnds[rxnNum]['opt'] = True
                rxn['item'].setForeground(0, self.color['variable_rxn'])
             
            # for coefNum, box in enumerate(parent.mech_tree.rxn[rxnNum]['uncBox'][1:]):
                # coefName = parent.mech_tree.rxn[rxnNum]['coef'][coefNum][1]
                
                # update values if they lie outside limits
                # coefValue = mech.coeffs[rxnNum][coefName] 
                # if rateVal < limits[0] or rateVal > limits[1]:
                    # coefBox = parent.mech_tree.rxn[rxnNum]['valueBox'][coefNum]
                    # coefBox.valueChanged.emit(coefBox.value())
        
        parent.series.rate_bnds(parent.display_shock) 
    
    def update_coef_rate_from_opt(self, coef_opt, x):
        parent = self.parent()

        conv_type = 'Cantera2' + self.mech_tree_type
        x0 = []
        for i, idxDict in enumerate(coef_opt):  # set changes to both spinboxes and backend coeffs
            rxnIdx, coefIdx = idxDict['rxnIdx'], idxDict['coefIdx']
            coefName = list(parent.mech.coeffs[rxnIdx].keys())[coefIdx]
            
            parent.mech.coeffs[rxnIdx][coefName] = x[i]
            coeffs = ['', coefName, x[i]]    # short name, long name, value
            value = self.convert._arrhenius(rxnIdx, [coeffs], conv_type)
            coefBox = parent.mech_tree.rxn[rxnIdx]['valueBox'][coefIdx]
            silentSetValue(coefBox, value[0][2])               
        
        parent.mech.modify_reactions(parent.mech.coeffs)
        self.update_rates()
    
    def update_mech_reset_value(self, event):
        parent = self.parent()
        sender = self.sender()
        rxnNum, coefNum, coefName = sender.info['rxnNum'], sender.info['coefNum'], sender.info['coefName']
        
        coeffs = [*sender.info['coef'], event]
        conv_type = self.mech_tree_type + '2Cantera'
        cantera_value = self.convert._arrhenius(rxnNum, deepcopy([coeffs]), conv_type)
        
        parent.mech.coeffs_bnds[rxnNum][coefName]['resetVal'] = cantera_value[0][2]
        self.update_uncertainties(event, sender)        # update uncertainties based on new reset value
    
    def update_box_reset_values(self):
        parent = self.parent()
        conv_type = 'Cantera2' + self.mech_tree_type
        for rxnNum, rxn in enumerate(self.mech_tree_data):
            if 'Arrhenius' not in parent.mech_tree.rxn[rxnNum]['rxnType']:  # skip if not arrhenius
                continue
                
            valBoxes = parent.mech_tree.rxn[rxnNum]['valueBox']
            for n, valBox in enumerate(valBoxes):    # update value boxes
                coefName = valBox.info['coefName']
                resetVal = parent.mech.coeffs_bnds[rxnNum][coefName]['resetVal']
                coeffs = [*valBox.info['coef'], deepcopy(resetVal)]
                value = self.convert._arrhenius(rxnNum, [coeffs], conv_type) 
                valBox.reset_value = value[0][2]
    
    def update_display_type(self):
        parent = self.parent()
        conv_type = 'Cantera2' + self.mech_tree_type
        self.mech_tree_data = self._set_mech_tree_data(self.mech_tree_type, parent.mech)   # recalculate mech tree data
        for rxnNum, rxn in enumerate(self.mech_tree_data):
            valBoxes = parent.mech_tree.rxn[rxnNum]['valueBox']
            uncBoxes = parent.mech_tree.rxn[rxnNum]['uncBox']
            for n, valBox in enumerate(valBoxes):    # update value boxes
                if valBox is None:  # in case there is no valbox because not arrhenius
                    continue
                
                coefNum = valBox.info['coefNum']
                coefName = valBox.info['coefName']
                silentSetValue(valBox, rxn['coeffs'][coefNum][2])  # update value
                valBox.info['coefAbbr'] = rxn['coeffs'][coefNum][0]    # update abbreviation
                valBox.info['label'].setText(valBox.info['coefAbbr'])
                if self.mech_tree_type == 'Bilbo':              # update step size
                    valBox.setSingleStep(0.01)
                elif self.mech_tree_type == 'Chemkin':
                    valBox.setSingleStep(0.1)
                
                uncBox = uncBoxes[n+1]
                # TODO: THIS REPEATS FROM RESET VALUES, BETTER TO ABSTRACT
                if uncBox.uncType == '±':
                    if uncBox.info['coef'][1] == 'pre_exponential_factor': continue
                    uncVal = parent.mech.coeffs_bnds[rxnNum][coefName]['value']
                    coeffs = [*valBox.info['coef'], deepcopy(uncVal)]
                    uncVal = self.convert._arrhenius(rxnNum, [coeffs], conv_type)[0][2]
                    silentSetValue(uncBox, uncVal)  # update value
        
        self.update_box_reset_values()
        self._updateDependents()             # update dependents and rates
        
    def _tabExpanded(self, sender, expanded):             # set uncboxes to not set upon first expand
        parent = self.parent()
        if hasattr(sender, 'info') and 'Arrhenius' in sender.info['rxnType']:
            rxnNum = sender.info['rxnNum']
        else: return
        
        if expanded:
            if sender.info['hasExpanded']: return
            else: sender.info['hasExpanded'] = True
        
            for box in parent.mech_tree.rxn[rxnNum]['uncBox']:
                # box.blockSignals(True)
                box.setValue(-1)
                # box.blockSignals(False)
                box.valueChanged.emit(box.value())
            
            self._copy_expanded_tab_rates()
        else:
            if rxnNum in parent.rxn_change_history:
                parent.rxn_change_history.remove(rxnNum)
                if parent.num_sim_lines_box.value() > 1:     # update history only if history tracked
                    parent.plot.signal.update_history()      # update plot history lines
 
    def _popup_menu(self, event):
        def setCopyRates(self, event):
            self.copyRates = event
            self._copy_expanded_tab_rates()   
        
        sender = self.sender()
        if len(self.sender().selectedItems()) > 0:
            selected = self._find_mech_item(sender.selectedItems()[0])
        else:
            selected = None
        
        popup_menu = QMenu(sender)
        
        copyRatesAction = QAction('Auto Copy Rates', checkable=True)
        copyRatesAction.setChecked(self.copyRates)
        popup_menu.addAction(copyRatesAction)
        copyRatesAction.triggered.connect(lambda event: setCopyRates(self, event))
        
        popup_menu.addSeparator()
        popup_menu.addAction('Expand All', lambda: sender.expandAll())
        popup_menu.addAction('Collapse All', lambda: sender.collapseAll())
        popup_menu.addSeparator()
        popup_menu.addAction('Reset All', lambda: self._reset_all())
        
        # this causes independent/dependent to not show if right click is not on rxn
        if selected is not None and 'Arrhenius' in selected['rxnType']: 
            popup_menu.addSeparator()
            
            dependentAction = QAction('Set Dependent', checkable=True)
            dependentAction.setChecked(selected['dependent'])
            popup_menu.addAction(dependentAction)
            dependentAction.triggered.connect(lambda event: self._setDependence(selected, event))
            
        popup_menu.exec_(sender.mapToGlobal(event))
        # popup_menu.exec_(QtGui.QCursor.pos()) # don't use exec_ twice or it will cause a double popup
    
    def _reset_all(self):
        parent = self.parent()
        self.run_sim_on_change = False
        mech = parent.mech
        for rxn in parent.mech_tree.rxn:
            if 'Arrhenius' not in rxn['rxnType']: continue # only reset Arrhenius rxns
            for spinbox in rxn['valueBox']:
                rxnNum, coefName = spinbox.info['rxnNum'], spinbox.info['coefName']
                resetCoef = mech.coeffs_bnds[rxnNum][coefName]['resetVal']
                mech.coeffs[rxnNum][coefName] = resetCoef
                
                spinbox._reset(silent=True)
            
        mech.modify_reactions(mech.coeffs)
        self._updateDependents()
        self.update_rates(rxnNum=rxnNum)
        
        self.run_sim_on_change = True
        parent.run_single()
    
    def _copy_expanded_tab_rates(self):
        if not self.copyRates:
            return

        parent = self.parent()
        values = []
        for rxnNum, rxn in enumerate(parent.mech_tree.rxn):
            if rxn['item'].isExpanded():
                values.append(str(rxn['rateBox'].value))
        
        if np.shape(values)[0] > 0: # only clear clipboard and copy if values exist
            values.append(str(parent.display_shock['time_offset']))       # add time offset
            parent.clipboard.clear()
            # mime = parent.clipboard.mimeData()
            # print(mime.formats()) # Maybe use xml spreadsheet?
            # parent.clipboard.setMimeData(values)
            parent.clipboard.setText('\t'.join(values)) # tab for new column, new line for new row
    
    def _find_mech_item(self, item):
        if not hasattr(item, 'info'): return None
        parent = self.parent()
        rxnNum = item.info['rxnNum']
        tree = parent.mech_tree
        if item is tree.rxn[rxnNum]['item'] or item in tree.rxn[rxnNum]['valueBox']:
            return tree.rxn[rxnNum]
                    
        return None
                
    def _setDependence(self, rxn, isDependent):
        rxn['dependent'] = isDependent
        if isDependent:
            for box in rxn['uncBox']:   # remove uncertainties
                box.setValue(-1)
                
            for valueBox, formulaBox in zip(rxn['valueBox'], rxn['formulaBox']):
                valueBox.hide()
                formulaBox.show()

            self._updateDependents()
        else:
            for valueBox, formulaBox in zip(rxn['valueBox'], rxn['formulaBox']):
                silentSetValue(valueBox, float(formulaBox.text()))  # propogate changes to valuebox
                formulaBox.hide()
                valueBox.show()

    def _updateDependents(self):
        parent = self.parent()
        mech = parent.mech
        # Update All formulas (this doesn't pick only the rates being modified)
        updateRates = []
        for rxnNum, rxn in enumerate(parent.mech_tree.rxn):
            if rxn['dependent']:
                updateRates.append(rxnNum)
                for box in rxn['formulaBox']:
                    rxnNum, coefName = box.info['rxnNum'], box.info['coefName']
                    current_value = mech.coeffs[rxnNum][coefName]
                    reset_value = mech.coeffs_bnds[rxnNum][coefName]['resetVal']
                    if reset_value == 0.0:
                        eqn = '+' + str(current_value)
                    else:
                        eqn = '*' + str(current_value/reset_value)
                    
                    box.applyFormula(emit=False, adjustment=eqn)

        parent.mech.modify_reactions(parent.mech.coeffs)
        self.update_rates(rxnNum=updateRates)     # Must happen after the mech is changed


class rateExpCoefficient(QWidget):  # rate expression coefficient # this is very slow
    def __init__(self, parent, coef, *args, **kwargs):
        # start_time = timer()
        QWidget.__init__(self)
        
        self.Label = QLabel(self.tr('{:s}:'.format(coef[0])))
        self.valueBox = misc_widget.ScientificDoubleSpinBox(parent=parent, *args, **kwargs)
        self.valueBox.setValue(coef[2])
        self.valueBox.setMaximumWidth(75)   # This matches the coefficients
        self.valueBox.setToolTip('Coefficient Value')
        
        self.formulaBox = ScientificLineEdit(parent)
        self.formulaBox.setValue(coef[2])
        self.formulaBox.setMaximumWidth(75)   # This matches the coefficients
        self.formulaBox.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.formulaBox.hide()
        
        self.unc = Uncertainty(parent, 'coef')
        self.uncValBox = self.unc.valBox
        self.uncTypeBox = self.unc.typeBox
        if coef[1] == 'pre_exponential_factor':                         # Lazy way to remove +/-
            for uncType in ['±', '+', '-']:                             # Remove uncTypes from box
                self.uncTypeBox.removeItem(self.uncTypeBox.findText(uncType))   
        
        spacer = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        end_spacer = QtWidgets.QSpacerItem(15, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
    
        layout = QGridLayout(self)
        layout.setContentsMargins(0,2,4,2)
        
        # layout.addItem(spacer, 0, 0)
        layout.addWidget(self.Label, 0, 0)
        layout.addItem(spacer, 0, 1)
        layout.addWidget(self.formulaBox, 0, 2)
        layout.addWidget(self.valueBox, 0, 2)          
        layout.addItem(end_spacer, 0, 3)
        layout.addWidget(self.unc, 0, 4)
        
        # print('{:.3f} ms'.format((timer() - start_time)*1E3))
                
class rxnRate(QWidget):
    def __init__(self, parent=None, label='', rxnType='Arrhenius', *args, **kwargs):
        QWidget.__init__(self, parent)
        self.parent = parent
        
        self.Label = QLabel(self.tr('k'))
        self.Label.setToolTip('Reaction Rate [mol, cm, s]')
        
        self.valueBox = ScientificLineEditReadOnly(parent, *args, **kwargs)
        self.valueBox.setMaximumWidth(75)   # This matches the coefficients
        self.valueBox.setDecimals(4)
        self.valueBox.setReadOnly(True)
        if 'value' in kwargs:
            self.valueBox.setValue(kwargs['value'])
        # self.valueBox.setFocusPolicy(QtCore.Qt.ClickFocus)
        
        spacer = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)

        layout = QGridLayout(self)
        layout.setContentsMargins(0,2,4,2)
        
        layout.addWidget(self.Label, 0, 0)
        layout.addItem(spacer, 0, 1)
        layout.addWidget(self.valueBox, 0, 2)
        
        if 'Arrhenius' in rxnType:
            end_spacer = QtWidgets.QSpacerItem(15, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
            
            self.unc = Uncertainty(parent, 'rate')
            self.uncValBox = self.unc.valBox
            self.uncTypeBox = self.unc.typeBox
            # layout.addItem(spacer, 0, 0)
            
            layout.addItem(end_spacer, 0, 3)
            layout.addWidget(self.unc, 0, 4)
        else:
            end_spacer = QtWidgets.QSpacerItem(117, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
            layout.addItem(end_spacer, 0, 3)


class Uncertainty(QWidget):
    def __init__(self, parent, type, *args, **kwargs):
        super().__init__(parent)
        # QWidget.__init__(self, parent)
        self.parent = parent
                
        self.typeBox = misc_widget.SearchComboBox() # uncertainty type
        self.typeBox.lineEdit().setReadOnly(True)   # defeats point of widget, but I like the look
        if type == 'coef':
            self.typeBox.addItems(['F', '%', '±', '+', '-'])
        elif type == 'rate':
            self.typeBox.addItems(['F', '%'])
            
        # This isn't pretty but it works
        tooltipTxt = ['<html><table border="0" cellspacing="1" cellpadding="0">'
                      '<tr><td style="padding-top:0; padding-bottom:6; padding-left:0; padding-right:4;"><p>F</p></td>',
                      '<td style="padding-top:0; padding-bottom:6; padding-left:4; padding-right:0;"><p>Uncertainty Factor</p></td></tr>',
                      '<tr><td style="padding-top:6; padding-bottom:6; padding-left:0; padding-right:4;"><p>%</p></td>',
                      '<td style="padding-top:6; padding-bottom:6; padding-left:4; padding-right:0;"><p>Percent Uncertainty (%/100)</p></td></tr>',
                      '<tr><td style="padding-top:6; padding-bottom:0; padding-left:0; padding-right:4;"><p>±</p></td>',
                      '<td style="padding-top:6; padding-bottom:0; padding-left:4; padding-right:0;"><p>Plus or Minus</p></td></tr></table></html>']
        self.typeBox.setToolTip(''.join(tooltipTxt))
        self.priorUncType = self.typeBox.currentText()
        self.typeBox.currentIndexChanged[str].connect(self.uncTypeChanged)
        
        self.uncMax = 100
        self.valBox = UncertaintyBox(parent, self.uncMax)        # uncertainty value
        self.valBox.setUncType(self.priorUncType)   # initialize uncertainty type
        if 'value' in kwargs:
            self.valBox.setValue(kwargs['value'])
    
        layout = QGridLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(4)
        
        layout.addWidget(self.typeBox, 0, 0)
        layout.addWidget(self.valBox, 0, 1)
       
    def uncTypeChanged(self, event):
        def plus_minus_values():
            tree = self.parent.mech_tree
            rxnNum, coefNum = self.valBox.info['rxnNum'], self.valBox.info['coefNum']
            coefBox = tree.rxn[rxnNum]['valueBox'][coefNum]
            
            return coefBox.strDecimals, coefBox.singleStep()*10, sys.float_info.max
            
        self.valBox.setUncType(event)   # pass event change to uncValBox
        if self.priorUncType == '±':
            self.valBox.setValue(self.valBox.minimum())  # if prior uncertainty was +-, reset

        uncVal = self.valBox.uncValue
        
        if event == 'F':    # only happens on event change, must have been % or ±
            if not np.isnan(uncVal):
                self.valBox.setValue(uncVal+1)
            self.valBox.setMinimum(1)
        elif self.priorUncType == 'F':
            self.valBox.setMinimum(0)
            self.valBox.setValue(uncVal-1)
        
        if event in ['F', '%']:
            self.valBox.setDecimals(2)
            self.valBox.setSingleStep(0.25)
            self.valBox.setMaximum(self.uncMax)
        else:
            dec, step, maxval = plus_minus_values()
            self.valBox.setDecimals(dec)
            self.valBox.setSingleStep(step)
            self.valBox.setMaximum(maxval)
            self.valBox.setValue(-step)
        
        self.priorUncType = event
        
        self.parent.tree.update_uncertainties(event, self.sender())

  
class UncValidator(QtGui.QValidator):
    def _isNum(self, s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    def validate(self, string, position):
        if self._isNum(string) or string == '-':
            state = QtGui.QValidator.Acceptable
        elif string == "" or string[position-1].lower() in 'e.-+':
            state = QtGui.QValidator.Intermediate
        else:
            state = QtGui.QValidator.Invalid
        return (state, string, position)
    
    
def uncertainty_fcn(x, uncVal, uncType):
    if np.isnan(uncVal):
        return [np.nan, np.nan]
    elif uncType == 'F':
        return np.sort([x/uncVal, x*uncVal])
    elif uncType == '%':
        return np.sort([x/(1+uncVal), x*(1+uncVal)])
    elif uncType == '±':
        return np.sort([x-uncVal, x+uncVal])
    elif uncType == '+':
        return np.sort([x, x+uncVal])
    elif uncType == '-':
        return np.sort([x-uncVal, x])
    
class UncertaintyBox(misc_widget.ScientificDoubleSpinBox):
    def __init__(self, parent, maxUnc, *args, **kwargs):
        super().__init__(parent=parent, *args, **kwargs)
        
        self.validator = UncValidator()
        self.setKeyboardTracking(False)
        self.setAccelerated(True)
        
        self.setMinimumWidth(55)
        self.setMaximumWidth(55)
        self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        
        self.minimumBaseValue= 1
        self.maximumBaseValue = maxUnc
        self.setSingleStep(0.25)
        self.setValue(1)
        self.setDecimals(2)
        self.setToolTip('Coefficient Uncertainty\n\nBased on reset values\nReset values default to initial mechanism')
        
        self.uncValue = self.value()
        self.valueChanged.connect(self.setUncVal)  # update uncertainty value based on value of box
        self.lineEdit().installEventFilter(self)
    
    def eventFilter(self, obj, event):  # clear if text is -
        if (event.type() == QtCore.QEvent.MouseButtonPress and obj is self.lineEdit() and 
                event.button() == QtCore.Qt.LeftButton and self.text() == '-'):
            self.lineEdit().clear()
            return True
        else:
            return super().eventFilter(obj, event)
    
    def validate(self, text, position):
        return self.validator.validate(text, position)
    
    def setSingleStep(self, event):
        super(UncertaintyBox, self).setSingleStep(event)   # don't want to overwrite all functionality
        self.setMinimum(self.minimumBaseValue)
        self.setMaximum(self.maximumBaseValue)
     
    def setMinimum(self, event):
        self.minimumBaseValue = event
        super(UncertaintyBox, self).setMinimum(event - self.singleStep())   # don't want to overwrite all functionality
        
    def setMaximum(self, event):
        self.maximumBaseValue = event
        super(UncertaintyBox, self).setMaximum(event + self.singleStep())   # don't want to overwrite all functionality
        
    def valueFromText(self, text):
        if text == '-':
            value = self.maximum()
        else:
            value = float(text)
        return value

    def textFromValue(self, value):
        if value < self.minimumBaseValue or value > self.maximumBaseValue:
            string = '-'
        else:
            string = super(UncertaintyBox, self).textFromValue(value)   # don't want to overwrite all functionality
            # string = '{:.{d}f}'.format(value, d = self.decimals())
        return string
    
    def stepBy(self, steps):
        if self.specialValueText() and self.value() == self.minimum():
            text = self.textFromValue(self.minimum())
        else:    
            text = self.cleanText()
        
        old_val = self.value()
        if self.value() < self.minimumBaseValue or self.value() > self.maximumBaseValue :
            val = old_val + self.singleStep()*steps
        else:
            val = old_val + np.power(10, misc_widget.OoM(old_val))*self.singleStep()*steps
        
        if val < self.singleStep(): 
            val = 0
        
        # if self.numFormat == 'g' and misc_widget.OoM(val) <= self.strDecimals:    # my own custom g
            # new_string = "{:.{dec}f}".format(val, dec=self.strDecimals)
        # else:
            # new_string = "{:.{dec}e}".format(val, dec=self.strDecimals)

        new_string = "{:g}".format(val)
        self.lineEdit().setText(new_string)
        self.setValue(float(new_string))
    
    def setUncVal(self, event):
        if event < self.minimumBaseValue or event > self.maximumBaseValue:
            self.uncValue = np.nan
        elif self.uncType in ['F', '%']:
            self.uncValue = event
        else:                               # if +-, need to convert to cantera units
            tree = self.parent().parent.tree
            rxnNum = self.info['rxnNum']
            coeffs = [*self.info['coef'], event]
            conv_type = tree.mech_tree_type + '2Cantera'
            cantera_value = tree.convert._arrhenius(rxnNum, deepcopy([coeffs]), conv_type)
            self.uncValue = cantera_value[0][2]
    
    def setUncType(self, event):    self.uncType = event
    
    def uncFcn(self, x):    
        return uncertainty_fcn(x, self.uncValue, self.uncType)
       
   
class ScientificLineEditReadOnly(QLineEdit):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.parent = parent
        self.setDecimals(3)
        self.setValue(0)
        
        if 'setDecimals' in kwargs:
            self.setDecimals(kwargs['setDecimals'])
            
        if 'value' in kwargs:
            self.setValue(kwargs['value'])

    def mousePressEvent(self, event):   # Yes, basically I made this whole line edit for this. It's called efficiency
        self.selectAll()
        # self.copy()
        
    def setValue(self, value):
        self.value = value          # Full precision. For box value set below as float(self.text())
        self.setText('{:.{dec}g}'.format(value, dec = self.decimals))
        
    def setDecimals(self, value): self.decimals = int(value)
    
        
class ScientificLineEdit(QLineEdit):
    valueChanged = QtCore.Signal(float)
    def __init__(self, parent, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.parent = parent
        self.setValue(0)
        self.displayMode = 'value'
        self.setReadOnly(True)
        
        if 'setDecimals' in kwargs:
            self.setDecimals(kwargs['setDecimals'])
            
        if 'value' in kwargs:
            self.setValue(kwargs['value'])
            
        self.textEdited.connect(self.updateFormula)
        self.editingFinished.connect(lambda: self.applyFormula(emit=True))
        
    def mousePressEvent(self, event):
        if self.isReadOnly():
            if event.button() == QtCore.Qt.LeftButton:
                self.displayMode = 'formula'
                self.setReadOnly(False)
                
                if hasattr(self, 'formula'):        # if this has been clicked before and has a formula show it
                    self.setText('={:s}'.format(self.formula))

        else:
            super(ScientificLineEdit, self).mousePressEvent(event)   # don't want to overwrite all functionality
        
    def setValue(self, value, emit=False):
        self.value = value          # Full precision. For box value set below as float(self.text())
        self.setText('{:g}'.format(value))
        if emit:
            self.valueChanged.emit(value)
    
    def setInitialFormula(self):
        abbr = self.info['coefAbbr']
        if abbr == 'log(A)':
            abbr = 'A'
        self.formula = '{:s}{:d}'.format(abbr, self.info['rxnNum']+1)
    
    def updateFormula(self):
        self.formula = self.text().replace('=', '').replace('^', '**')
    
    def applyFormula(self, emit, adjustment='*1.0'):               
        parent = self.parent
        rxnNum, boxType = self.info['rxnNum'], self.info['type']
        coefName = self.info['coefName']
        tree = parent.mech_tree
        
        formula = self.formula
        if formula is None or formula.strip() == '':
            self.setInitialFormula()
            formula = self.formula.replace('=', '')
            
        names = [node.id for node in ast.walk(ast.parse(formula)) 
                 if isinstance(node, ast.Name)]
        
        var = {}
        for name in names:
            abbr, subRxnNum = re.split('(\d+)', name)[0:2]
            subRxnNum = int(subRxnNum)-1
            formula = formula.replace(name, "var['" + name + "']")
            if subRxnNum == rxnNum: # if rxnNum matches current value, set value to initial
                rxnNum, coefName = self.info['rxnNum'], self.info['coefName']
                var[name] = parent.mech.coeffs_bnds[rxnNum][coefName]['resetVal']
                if len(names) == 1 and formula[-1] == ']' and adjustment not in ['*1.0', '+0.0']:
                    formula += adjustment
                    self.formula += adjustment
            else:
                for box in tree.rxn[subRxnNum]['valueBox']:
                    if box.info['coefName'] == coefName:
                        var[name] = parent.mech.coeffs[subRxnNum][coefName]
        
        parent.mech.coeffs[rxnNum][coefName] = value = eval(formula)
        conv_type = 'Cantera2' + parent.tree.mech_tree_type
        coeffs = [*self.info['coef'], value]
        value = parent.convert_units._arrhenius(rxnNum, deepcopy([coeffs]), conv_type)[0][2]
        
        self.displayMode = 'value'
        self.setReadOnly(True)
        self.setValue(value, emit)