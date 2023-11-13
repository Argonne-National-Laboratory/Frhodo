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

from calculate.mech_fcns import arrhenius_coefNames


default_coef_abbreviation = {
            "pre_exponential_factor": "A",
            "temperature_exponent": "n",
            "activation_energy": "Ea"}

coef_abbreviation = {key: default_coef_abbreviation[key] for key in arrhenius_coefNames}


def silentSetValue(obj, value):
    obj.blockSignals(True)           # stop changing text from signaling
    obj.setValue(value)
    obj.blockSignals(False)          # allow signals again
    
def keysFromBox(box, mech):
    coefAbbr, rxnIdx = box.info['coefAbbr'], box.info['rxnNum']
    rxn = mech.gas.reactions()[rxnIdx]
    
    return mech.get_coeffs_keys(rxn, coefAbbr, rxnIdx=rxnIdx)

class Tree(QtCore.QObject):
    def __init__(self, parent):
        super().__init__(parent)
        self.run_sim_on_change = True
        self.copyRates = False
        self.convert = parent.convert_units

        self.timer = QtCore.QTimer()

        self.model = QtGui.QStandardItemModel(parent.mech_tree)
        self.proxy_model = QSortFilterProxyModel(parent.mech_tree)
        self.proxy_model.setSourceModel(self.model)
        parent.mech_tree.setModel(self.proxy_model)
        self.tree_filter = TreeFilter(parent, self.proxy_model, self.model, self._set_mech_widgets)

        self.color = {'variable_rxn': QtGui.QBrush(QtGui.QColor(188, 0, 188)),
                      'fixed_rxn': QtGui.QBrush(QtGui.QColor(0, 0, 0))}
        
        parent.mech_tree.setRootIsDecorated(False)
        parent.mech_tree.setIndentation(21)
        parent.mech_tree.setExpandsOnDoubleClick(False)
        parent.mech_tree.clicked.connect(self.item_clicked)

        # Set up right click popup menu and linked expand/collapse
        parent.mech_tree.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        parent.mech_tree.customContextMenuRequested.connect(self._popup_menu)
        parent.mech_tree.expanded.connect(lambda sender: self._tabExpanded(sender, True))
        parent.mech_tree.collapsed.connect(lambda sender: self._tabExpanded(sender, False))
    
    def item_clicked(self, event):
        ix = self.proxy_model.mapToSource(event)
        item = self.model.itemFromIndex(ix)
        if not hasattr(item, 'info'): return
        
        rxnNum = item.info['rxnNum']
        tree = self.parent().mech_tree
        if tree.isExpanded(event):
            tree.collapse(event)
            self._tabExpanded(event, expanded=False)
            item.info['isExpanded'] = False
        else:
            if not hasattr(item, 'info'): return # skip if not a reaction, hence no info
            if not item.info['hasExpanded']:     # forward event if tab has never expanded
                self._tabExpanded(event, expanded=True)
            else:
                self._set_mech_widgets(item)
            tree.expand(event)
            item.info['isExpanded'] = True
    
    def set_trees(self, mech):
        parent = self.parent()
        #parent.mech_tree.reset()
        self.model.removeRows(0, self.model.rowCount())
        if 'Chemkin' in parent.tab_select_comboBox.currentText():
            self.mech_tree_type = 'Chemkin'
        else:
            self.mech_tree_type = 'Bilbo'
        self.mech_tree_data = self._set_mech_tree_data(self.mech_tree_type, mech)
        self._set_mech_tree(self.mech_tree_data)

    def _set_mech_tree_data(self, selection, mech):
        def get_coef_abbreviation(coefName):
            if 'activation_energy' == coefName:
                return 'Ea'
            elif 'pre_exponential_factor' == coefName:
                return 'A'
            elif 'temperature_exponent' == coefName:
                return 'n'

        parent = self.parent()
        data = []
        for rxnIdx, rxn in enumerate(mech.gas.reactions()):
            rxn_type = mech.reaction_type(rxn)

            if type(rxn.rate) is ct.ArrheniusRate:
                coeffs = [] # Setup Coeffs for Tree
                for coefName, coefAbbr in coef_abbreviation.items():
                    coeffs.append([coefAbbr, coefName, mech.coeffs[rxnIdx][0]])
                
                coeffs_order = [1, 2, 0] # order coeffs into A, n, Ea
   
                data.append({'num': rxnIdx, 'eqn': rxn.equation, 'type': rxn_type, 
                             'coeffs': coeffs, 'coeffs_order': coeffs_order})
            elif type(rxn.rate) in [ct.PlogRate, ct.FalloffRate, ct.TroeRate, ct.SriRate]:
                coeffs = []
                for key in ['high', 'low']:
                    if type(rxn.rate) is ct.PlogRate:
                        if key == 'high':
                            n = len(mech.coeffs[rxnIdx]) - 1
                        else:
                            n = 0
                    else:
                        n = f'{key}_rate'

                    for coefName, coefAbbr in coef_abbreviation.items():
                        coeffs.append([f'{coefAbbr}_{key}', coefName, mech.coeffs[rxnIdx][n]])

                coeffs_order = [1, 2, 0, 4, 5, 3] # order coeffs into A_high, n_high, Ea_high, A_low

                data.append({'num': rxnIdx, 'eqn': rxn.equation, 'type': rxn_type, 
                             'coeffs': coeffs, 'coeffs_order': coeffs_order})
            else:
                data.append({'num': rxnIdx, 'eqn': rxn.equation, 'type': rxn_type})
                # raise Exception("Equation type is not currently implemented for:\n{:s}".format(rxn.equation))
            
        return data                
        
    def _set_mech_tree(self, rxn_matrix):
        parent = self.parent()
        tree = parent.mech_tree           
        #tree.setColumnCount(1)
        self.model.setHorizontalHeaderLabels(['Reaction'])
                
        tree.setUpdatesEnabled(False)
        tree.rxn = []
        for rxn in rxn_matrix:
            L1 = QtGui.QStandardItem(f" R{rxn['num']+1:d}:   {rxn['eqn'].replace('<=>', '=')}")
            L1.setEditable(False)
            L1.setToolTip(rxn['type'])
            L1.info = {'tree': tree.objectName(), 'type': 'rxn tab', 'rxnNum': rxn['num'], 'rxnType': rxn['type'],
                       'hasExpanded': False, 'isExpanded': False, 'rxn_details': rxn, 'row': []}
            self.model.appendRow(L1)
            tree.rxn.append({'item': L1, 'num': rxn['num'], 'rxnType': rxn['type'], 'dependent': False})
        
        tree.setUpdatesEnabled(True)
        tree.sortByColumn(0, QtCore.Qt.AscendingOrder)
        tree.header().setStretchLastSection(True)
        tree.header().setSectionResizeMode(0, QHeaderView.ResizeToContents) #QHeaderView.Stretch) #QHeaderView.Interactive)

        # Set Tab Order         TODO: FIX THIS
        '''
        last_arrhenius = 0
        for i, rxn in enumerate(rxn_matrix):
            print(rxn)
            if rxn['type'] != 'Arrhenius Reaction':
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
    
    def _set_mech_widgets(self, sender):
        def set_rate_widget(unc={}):
            info = {'type': 'rateUnc', 'rxnNum': rxn['num']} 
            if len(unc) == 0:
                widget = rxnRate(parent, info, rxnType=rxn['type'])
            else:
                widget = rxnRate(parent, info, rxnType=rxn['type'], unc_type=unc['type'], unc_value=unc['value'])
            L1.info['row'].append({'item': QtGui.QStandardItem(''), 'widget': widget})
            L1.appendRow([L1.info['row'][-1]['item']])
            mIndex = self.proxy_model.mapFromSource(L1.info['row'][-1]['item'].index())
            tree.setIndexWidget(mIndex, L1.info['row'][-1]['widget'])

            return widget

        L1 = sender
        rxn = L1.info['rxn_details']
        rxnNum = rxn['num']
        parent = self.parent()
        tree = parent.mech_tree  

        # clear rows of qstandarditem (L1)
        L1.removeRows(0, L1.rowCount())

        if rxn['type'] in ['Arrhenius Reaction', 'Plog Reaction', 'Falloff Reaction']:
            widget = set_rate_widget(unc={'type': parent.mech.rate_bnds[rxnNum]['type'],
                                          'value': parent.mech.rate_bnds[rxnNum]['value']})
            widget.uncValBox.valueChanged.connect(self.update_uncertainties)       # no update between F and %

            len_coef = len(rxn['coeffs_order'])
            tree.rxn[rxnNum].update({'coef': rxn['coeffs'], 'rateBox': widget.valueBox, 
                                     'formulaBox': [None]*len_coef, 'valueBox': [None]*len_coef, 
                                     'uncBox': [widget.uncValBox]*(len_coef+1)})
            
            for coefNum in rxn['coeffs_order']:
                # convert mech coeffs to display units
                coef = deepcopy(rxn['coeffs'][coefNum])
                coef[2] = coef[2][coef[1]]  # get current value
                conv_type = f'Cantera2{self.mech_tree_type}'
                coef = self.convert._arrhenius(rxnNum, [coef], conv_type)[0]

                if rxn['type'] == 'Arrhenius Reaction':
                    bnds_key = 'rate'
                elif rxn['type'] in ['Plog Reaction', 'Falloff Reaction']:
                    if 'high' in coef[0]:
                        bnds_key = 'high_rate'
                    elif 'low' in coef[0]:
                        bnds_key = 'low_rate'
                
                unc_type = parent.mech.coeffs_bnds[rxnNum][bnds_key][coef[1]]['type']
                unc_value = parent.mech.coeffs_bnds[rxnNum][bnds_key][coef[1]]['value']

                if unc_type not in ['F', '%']:
                    unc_value = self.convert._arrhenius(rxnNum, [[*coef[:2], unc_value]], conv_type)[0][2]

                info = {'type': rxn['type'], 'rxnNum': rxn['num'], 'coefNum': coefNum, 'label': '',
                        'coef': coef[0:2], 'coefAbbr': coef[0], 'coefName': coef[1], 'coefVal': coef[2]}

                widget = rateExpCoefficient(parent, coef, info, unc_type=unc_type, unc_value=unc_value)
                widget.Label.setToolTip(coef[1].replace('_', ' ').title())
                if self.mech_tree_type == 'Bilbo':
                    widget.valueBox.setSingleStep(0.01)
                elif self.mech_tree_type == 'Chemkin':
                    widget.valueBox.setSingleStep(0.1)
                    
                widget.formulaBox.setInitialFormula()
                widget.formulaBox.valueChanged.connect(self.update_value)
                widget.valueBox.valueChanged.connect(self.update_value)
                widget.valueBox.resetValueChanged.connect(self.update_mech_reset_value)
                widget.uncValBox.valueChanged.connect(self.update_uncertainties)
                    
                tree.rxn[rxnNum]['formulaBox'][coefNum] = widget.formulaBox
                tree.rxn[rxnNum]['valueBox'][coefNum] = widget.valueBox
                tree.rxn[rxnNum]['uncBox'][coefNum+1] = widget.uncValBox
                
                L1.info['row'].append({'item': QtGui.QStandardItem(''), 'widget': widget})
                L1.appendRow([L1.info['row'][-1]['item']])
                mIndex = self.proxy_model.mapFromSource(L1.info['row'][-1]['item'].index())
                tree.setIndexWidget(mIndex, L1.info['row'][-1]['widget'])

        else:   # if not Arrhenius, show rate only
            widget = set_rate_widget()
            tree.rxn[rxnNum].update({'coef': [], 'rateBox': widget.valueBox, 'formulaBox': [None], 
                                     'valueBox': [None], 'uncBox': [None]})
        
        # update rates and reset values of created boxes
        self.update_box_reset_values(rxnNum)
        self.update_rates(rxnNum)
    
    def currentRxn(self):
        tree = self.parent().mech_tree
        sender_idx = tree.selectedIndexes()[0]
        ix = self.proxy_model.mapToSource(sender_idx)
        selected = self.model.itemFromIndex(ix)
        if hasattr(selected, 'info'):
            rxnNum = selected.info['rxnNum']
            return tree.rxn[rxnNum]
        else:
            return None

    def update_value(self, event):
        def getRateConst(parent, rxnNum, coef_key, coefName, value):
            shock = parent.display_shock
            parent.mech.coeffs[rxnNum][coef_key][coefName] = value
            parent.mech.modify_reactions(parent.mech.coeffs)
            mech_out = parent.mech.set_TPX(shock['T_reactor'], shock['P_reactor'], shock['thermo_mix'])
            if not mech_out['success']:
                parent.log.append(mech_out['message'])
                return
            return parent.mech.gas.forward_rate_constants[rxnNum]
        
        parent = self.parent()
        sender = self.sender()
        mech = parent.mech
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
        coef_key, bnds_key = keysFromBox(sender, mech)

        coef_bnds_dict = mech.coeffs_bnds[rxnNum][bnds_key][coefName]
        coefLimits = coef_bnds_dict['limits']()
        
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
            rateCon = getRateConst(parent, rxnNum, coef_key, coefName, cantera_value[0][2])
            
            if rateCon < rateLimits[0]:
                limViolation = 0
            elif rateCon > rateLimits[1]:
                limViolation = 1
            else:
                outside_limits = False
            
            if outside_limits:
                # Calculate correct coef value
                fcn = lambda x: getRateConst(parent, rxnNum, coef_key, coefName, x) - rateLimits[limViolation]
                x0 = coef_bnds_dict['resetVal']
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
        mech.coeffs[rxnNum][coef_key][coefName] = cantera_value[0][2]
        
        self._updateDependents()             # update dependents and rates
        mech.modify_reactions(mech.coeffs)
        self.update_rates(rxnNum=rxnNum)     # Must happen after the mech is changed
        
        if self.run_sim_on_change:
            parent.run_single(rxn_changed=True)   
    
    def update_rates(self, rxnNum=None):
        parent = self.parent()
        shock = parent.display_shock
        
        if not parent.mech_loaded: return # if mech isn't loaded successfully, exit
        rxn_rate = parent.series.rates(shock)   # update rates from settings
        if rxn_rate is None: return
        
        num_reac_all = np.sum(parent.mech.gas.reactant_stoich_coeffs, axis=0)
        
        if rxnNum is not None:
            if type(rxnNum) in [list, np.ndarray]:
                rxnNumRange = rxnNum
            else:
                rxnNumRange = [rxnNum]
        else:
            rxnNumRange = range(parent.mech.gas.n_reactions)
        
        for rxnNum in rxnNumRange:
            if 'rateBox' not in parent.mech_tree.rxn[rxnNum]:
                continue
            
            rxn_rate_box = parent.mech_tree.rxn[rxnNum]['rateBox']
            if rxn_rate_box is None: 
                continue
            
            conv = np.power(1E3, num_reac_all[rxnNum]-1)
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
                coefNum, coefName, coefAbbr = sender.info['coefNum'], sender.info['coefName'], sender.info['coefAbbr']
                
                # get correct uncertainty diction based on reaction type
                coef_key, bnds_key = keysFromBox(sender, mech)

                coefUncDict = mech.coeffs_bnds[rxnNum][bnds_key][coefName]
                uncBox = parent.mech_tree.rxn[rxnNum]['uncBox'][coefNum+1]
                
                coefUncDict['value'] = uncVal = uncBox.uncValue
                coefUncDict['type'] = uncBox.uncType
                limits = coefUncDict['limits']()
                
                coefUncDict['opt'] = True
                if uncVal == uncBox.minimumBaseValue: # not optimized
                    coefUncDict['opt'] = False                    
                
                # update values if they now lie outside limits
                coefValue = mech.coeffs[rxnNum][coef_key][coefName]

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
            if rxn['rxnType'] not in ['Arrhenius Reaction', 'Plog Reaction', 'Falloff Reaction']:   # skip if not allowable type
                mech.rate_bnds[rxnNum]['opt'] = False
                continue
            if 'uncBox' not in rxn:
                continue
            
            mech.rate_bnds[rxnNum]['value'] = uncVal = rxn['uncBox'][0].uncValue
            mech.rate_bnds[rxnNum]['type'] = rxn['uncBox'][0].uncType
            if np.isnan(uncVal) or uncVal == rxn['uncBox'][0].minimumBaseValue: # not optimized
                mech.rate_bnds[rxnNum]['opt'] = False
                rxn['item'].setForeground(self.color['fixed_rxn'])
            else:
                mech.rate_bnds[rxnNum]['opt'] = True
                rxn['item'].setForeground(self.color['variable_rxn'])
             
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
        for i, idxDict in enumerate(coef_opt):  # set changes to both spinboxes and backend coeffs
            rxnIdx, coefIdx = idxDict['rxnIdx'], idxDict['coefIdx']
            coeffs_key = idxDict['key']['coeffs']

            if coeffs_key == 'falloff_parameters':
                if type(parent.mech.coeffs[rxnIdx][coeffs_key]) is tuple:
                    parent.mech.coeffs[rxnIdx][coeffs_key] = list(parent.mech.coeffs[rxnIdx][coeffs_key])
                parent.mech.coeffs[rxnIdx][coeffs_key][coefIdx] = x[i]
                continue # no falloff parameter boxes

            coefName = list(parent.mech.coeffs[rxnIdx][coeffs_key].keys())[coefIdx]
            
            parent.mech.coeffs[rxnIdx][coeffs_key][coefName] = x[i]
            coeffs = ['', coefName, x[i]]    # short name, long name, value
            value = self.convert._arrhenius(rxnIdx, [coeffs], conv_type)
            if coeffs_key == 'low_rate':
                coefIdx += 3
            coefBox = parent.mech_tree.rxn[rxnIdx]['valueBox'][coefIdx]
            silentSetValue(coefBox, value[0][2])               
        
        parent.mech.modify_reactions(parent.mech.coeffs)
        self.update_rates()
    
    def update_mech_reset_value(self, event):   # TODO WHEN BOXES ARE CREATED UNCERTAINTIES ARE BEING CALLED TWICE
        parent = self.parent()
        sender = self.sender()
        rxnNum, coefNum, coefName = sender.info['rxnNum'], sender.info['coefNum'], sender.info['coefName']
        
        coeffs = [*sender.info['coef'], event]
        conv_type = self.mech_tree_type + '2Cantera'
        cantera_value = self.convert._arrhenius(rxnNum, deepcopy([coeffs]), conv_type)
        
        parent.mech.coeffs_bnds[rxnNum][coefName]['resetVal'] = cantera_value[0][2]
        self.update_uncertainties(event, sender)        # update uncertainties based on new reset value
    
    def update_box_reset_values(self, rxnNum=None):
        parent = self.parent()
        mech = parent.mech
        conv_type = 'Cantera2' + self.mech_tree_type
        
        if rxnNum is not None:
            if type(rxnNum) in [list, np.ndarray]:
                rxnNumRange = rxnNum
            else:
                rxnNumRange = [rxnNum]
        else:
            rxnNumRange = range(mech.gas.n_reactions)

        for rxnNum in rxnNumRange:
            rxn = parent.mech_tree.rxn[rxnNum]
            if (rxn['rxnType'] not in ['Arrhenius Reaction', 'Plog Reaction', 'Falloff Reaction'] or 'valueBox' not in rxn): continue
            

            valBoxes = parent.mech_tree.rxn[rxnNum]['valueBox']
            for n, valBox in enumerate(valBoxes):    # update value boxes
                coefName = valBox.info['coefName']
                coef_key, bnds_key = keysFromBox(valBox, mech)
                
                resetVal = mech.coeffs_bnds[rxnNum][bnds_key][coefName]['resetVal']
                coeffs = [*valBox.info['coef'], deepcopy(resetVal)]
                value = self.convert._arrhenius(rxnNum, [coeffs], conv_type) 
                valBox.reset_value = value[0][2]
    
    def update_display_type(self):
        parent = self.parent()
        mech = parent.mech
        conv_type = f'Cantera2{self.mech_tree_type}'
        for rxnNum, rxn in enumerate(mech.coeffs):
            if 'valueBox' not in parent.mech_tree.rxn[rxnNum]: continue

            valBoxes = parent.mech_tree.rxn[rxnNum]['valueBox']
            uncBoxes = parent.mech_tree.rxn[rxnNum]['uncBox']
            for n, valBox in enumerate(valBoxes):    # update value boxes
                if valBox is None:  # in case there is no valbox because not arrhenius
                    continue
                
                coefNum = valBox.info['coefNum']
                coefName = valBox.info['coefName']
                coef_key, bnds_key = keysFromBox(valBox, mech)
                coeffs = [*valBox.info['coef'], rxn[coef_key][coefName]]
                coeffs = self.convert._arrhenius(rxnNum, [coeffs], conv_type)[0]
                
                silentSetValue(valBox, coeffs[2])  # update value
                valBox.info['coefAbbr'] = f'{coeffs[0]}:'    # update abbreviation
                valBox.info['label'].setText(valBox.info['coefAbbr'])
                if self.mech_tree_type == 'Bilbo':              # update step size
                    valBox.setSingleStep(0.01)
                elif self.mech_tree_type == 'Chemkin':
                    valBox.setSingleStep(0.1)
                
                uncBox = uncBoxes[n+1]
                # TODO: THIS REPEATS FROM RESET VALUES, BETTER TO ABSTRACT
                if uncBox.uncType == '±':
                    if uncBox.info['coef'][1] == 'pre_exponential_factor': continue
                    
                    uncVal = mech.coeffs_bnds[rxnNum][bnds_key][coefName]['value']
                    coeffs = [*valBox.info['coef'], deepcopy(uncVal)]
                    uncVal = self.convert._arrhenius(rxnNum, [coeffs], conv_type)[0][2]
                    silentSetValue(uncBox, uncVal)  # update value
        
        self.update_box_reset_values()
        self._updateDependents()             # update dependents and rates
        
    def _tabExpanded(self, sender_idx, expanded):             # set uncboxes to not set upon first expand
        parent = self.parent()
        ix = self.proxy_model.mapToSource(sender_idx)
        sender = self.model.itemFromIndex(ix)
        if hasattr(sender, 'info'):
            rxnNum = sender.info['rxnNum']
        else: return
        
        if expanded:
            if sender.info['hasExpanded']: return
            else: 
                sender.info['hasExpanded'] = True
                self._set_mech_widgets(sender)
        
            if sender.info['rxnType'] in ['Arrhenius Reaction', 'Plog Reaction', 'Falloff Reaction']:
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
        
        tree = self.parent().mech_tree
        rxn = self.currentRxn()
        
        popup_menu = QMenu(tree)
        
        copyRatesAction = QAction('Auto Copy Rates', checkable=True)
        copyRatesAction.setChecked(self.copyRates)
        popup_menu.addAction(copyRatesAction)
        copyRatesAction.triggered.connect(lambda event: setCopyRates(self, event))
        
        popup_menu.addSeparator()
        #popup_menu.addAction('Expand All', lambda: tree.expandAll()) # bad idea slow to create many widgets
        popup_menu.addAction('Collapse All', lambda: tree.collapseAll())
        popup_menu.addSeparator()
        popup_menu.addAction('Reset All', lambda: self._reset_all())
        
        # this causes independent/dependent to not show if right click is not on rxn
        if rxn is not None and 'Arrhenius Reaction' in rxn['rxnType']: 
            popup_menu.addSeparator()
            
            dependentAction = QAction('Set Dependent', checkable=True)
            dependentAction.setChecked(rxn['dependent'])
            popup_menu.addAction(dependentAction)
            dependentAction.triggered.connect(lambda event: self._setDependence(rxn, event))
            
        #popup_menu.exec_(tree.mapToGlobal(event))
        popup_menu.exec_(QtGui.QCursor.pos()) # don't use exec_ twice or it will cause a double popup
    
    def _reset_all(self):
        parent = self.parent()
        self.run_sim_on_change = False
        mech = parent.mech
        for rxn in parent.mech_tree.rxn:
            if (rxn['rxnType'] not in ['Arrhenius Reaction', 'Plog Reaction', 'Falloff Reaction']
                or 'valueBox' not in rxn): continue # only reset Arrhenius boxes

            for box in rxn['valueBox']:
                rxnNum, coefName = box.info['rxnNum'], box.info['coefName']
                coef_key, bnds_key = keysFromBox(box, mech)
                resetCoef = mech.coeffs_bnds[rxnNum][bnds_key][coefName]['resetVal']
                mech.coeffs[rxnNum][coef_key][coefName] = resetCoef
                
                box._reset(silent=True)
                self.update_rates(rxnNum=rxnNum)
            
        mech.modify_reactions(mech.coeffs)
        self._updateDependents()
        
        self.run_sim_on_change = True
        parent.run_single()
    
    def _copy_expanded_tab_rates(self):
        parent = self.parent()

        def copy_to_clipboard(values):
            parent.clipboard.clear()
            #data = parent.clipboard.mimeData()
            #data.setText('\t'.join(values))
            #parent.clipboard.setMimeData(data)
            parent.clipboard.setText('\t'.join(values)) # tab for new column, new line for new row
        
        if not self.copyRates: return
        elif parent.optimize_running: return
        
        values = []
        for rxnNum, rxn in enumerate(parent.mech_tree.rxn):
            mIndex = self.proxy_model.mapFromSource(rxn['item'].index())
            if parent.mech_tree.isExpanded(mIndex):
                values.append(str(rxn['rateBox'].value))

        t_unit_conv = parent.var['reactor']['t_unit_conv']
        values.append(str(parent.display_shock['time_offset']/t_unit_conv))       # add time offset
        if np.shape(values)[0] > 0: # only clear clipboard and copy if values exist
            self.timer.singleShot(50, lambda: copy_to_clipboard(values))    # 50 ms to prevent errors
            
    
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
                    coef_key, bnds_key = keysFromBox(box, mech)
                    current_value = mech.coeffs[rxnNum][coef_key][coefName]
                    reset_value = mech.coeffs_bnds[rxnNum][bnds_key][coefName]['resetVal']
                    if reset_value == 0.0:
                        eqn = '+' + str(current_value)
                    else:
                        eqn = '*' + str(current_value/reset_value)
                    
                    box.applyFormula(emit=False, adjustment=eqn)

        parent.mech.modify_reactions(parent.mech.coeffs)
        self.update_rates(rxnNum=updateRates)     # Must happen after the mech is changed


class QSortFilterProxyModel(QtCore.QSortFilterProxyModel):
    def __init__(self, parent=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
    
    def filterAcceptsRow(self, row, parent):
        model = self.sourceModel()
        if parent.isValid(): # Do not apply the filter to child elements
            return True
        else:
            return super().filterAcceptsRow(row, parent)
        
        # DELETE THIS IF NOT CREATING CUSTOM FILTER
        #idx = model.index(row, 0, parent)
        #print(model.item(row).text(), str(self.filterRegExp()))

        #if (self._suffix
        #    and isinstance(model, QtWidgets.QFileSystemModel)
        #    and source_parent == model.index(model.rootPath())
        #):
        #    index = model.index(source_row, 0, source_parent)
        #    name = index.data(QtWidgets.QFileSystemModel.FileNameRole)
        #    file_info = model.fileInfo(index)
        #    return name.split(".")[-1] == self._suffix and file_info.isDir()

    def lessThan(self, left, right):
        rxnNum = lambda text: int(text[2:].split(':')[0])
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

        #self.mytreeview.setModel(self.proxy_model)
        #self.mytreeview.clicked.connect(self.update_model)

        self.filter_input.textChanged.connect(self.textChanged)

    def textChanged(self, event):
        regexp_raw = event.strip().split(' ')
        regexp = ['^.*']    # create regeular expression based on filter text
        for txt in regexp_raw:
            if txt == '|':
                regexp.append(txt)
            elif txt == '&': continue
            elif len(txt.strip()) > 0:
                txt = txt.replace('*', '.*')
                regexp.append(fr'(?=.*\b{txt}\b)')
        regexp.append('.*$')
        regexp = ''.join(regexp)

        self.proxy_model.setFilterRegularExpression(regexp)
        self.update_match_tooltip(self.proxy_model.rowCount())
        self.expand_items()

    def update_match_tooltip(self, num, show=True):
        if num == 1:
            self.filter_input.setToolTip(f'{num:d} match')
        else:
            self.filter_input.setToolTip(f'{num:d} matches')
        
        if show:
            pos = self.filter_input.mapToGlobal(QtCore.QPoint(0, 0))
            width = self.filter_input.sizeHint().width()
            pos.setX(pos.x() + width*2)
            height = self.filter_input.sizeHint().height()
            pos.setY(pos.y() + int(height/4))

            QToolTip.showText(pos, self.filter_input.toolTip())

    def expand_items(self):
        tree = self.parent.mech_tree
        for row_idx in range(self.proxy_model.rowCount()):
            proxy_idx = self.proxy_model.index(row_idx, 0)
            idx = self.proxy_model.mapToSource(proxy_idx)
            item = self.model.itemFromIndex(idx)
            rxnNum = item.info['rxnNum']

            if item.info['isExpanded']:
                tree.expand(proxy_idx)
                self._set_mech_widgets(item)  


class rateExpCoefficient(QWidget):  # rate expression coefficient
    def __init__(self, parent, coef, info, *args, **kwargs):
        QWidget.__init__(self)
        
        self.Label = QLabel(self.tr('{:s}:'.format(coef[0])))
        info['label'] = self.Label

        exclude_keys = ['unc_value', 'unc_type', 'info']
        valueBox_kwargs = {k: kwargs[k] for k in set(list(kwargs.keys())) - set(exclude_keys)}
        self.valueBox = misc_widget.ScientificDoubleSpinBox(parent=parent, *args, **valueBox_kwargs)
        self.valueBox.info = info
        self.valueBox.setValue(coef[2])
        self.valueBox.setSingleIntStep(0.01)
        self.valueBox.setMaximumWidth(75)   # This matches the coefficients
        self.valueBox.setToolTip('Coefficient Value')
        
        self.formulaBox = ScientificLineEdit(parent)
        self.formulaBox.info = info
        self.formulaBox.setValue(coef[2])
        self.formulaBox.setMaximumWidth(75)   # This matches the coefficients
        self.formulaBox.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.formulaBox.hide()
        
        info['mainValueBox'] = self.valueBox

        if 'unc_value' in kwargs and 'unc_type' in kwargs:
            self.unc = Uncertainty(parent, 'coef', info, value=kwargs['unc_value'], unc_choice=kwargs['unc_type'])
        else:
            self.unc = Uncertainty(parent, 'coef', info)

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

                
class rxnRate(QWidget):
    def __init__(self, parent, info, rxnType='Arrhenius Reaction', label='', *args, **kwargs):
        QWidget.__init__(self, parent)
        self.parent = parent
        
        self.Label = QLabel(self.tr('k'))
        self.Label.setToolTip('Reaction Rate [mol, cm, s]')
        
        exclude_keys = ['unc_value', 'unc_type']
        valueBox_kwargs = {k: kwargs[k] for k in set(list(kwargs.keys())) - set(exclude_keys)}
        self.valueBox = ScientificLineEditReadOnly(parent, *args, **valueBox_kwargs)
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
        
        if rxnType in ['Arrhenius Reaction', 'Plog Reaction', 'Falloff Reaction']:
            info['mainValueBox'] = self.valueBox

            if 'unc_value' in kwargs and 'unc_type' in kwargs:
                self.unc = Uncertainty(parent, 'rate', info, value=kwargs['unc_value'], unc_choice=kwargs['unc_type'])
            else:
                self.unc = Uncertainty(parent, 'rate', info)

            self.uncValBox = self.unc.valBox
            self.uncTypeBox = self.unc.typeBox
            # layout.addItem(spacer, 0, 0)

            end_spacer = QtWidgets.QSpacerItem(15, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
            layout.addItem(end_spacer, 0, 3)
            layout.addWidget(self.unc, 0, 4)
        else:
            end_spacer = QtWidgets.QSpacerItem(117, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
            layout.addItem(end_spacer, 0, 3)


class Uncertainty(QWidget):
    def __init__(self, parent, type, info, *args, **kwargs):
        super().__init__(parent)
        # QWidget.__init__(self, parent)
        self.parent = parent
        self.info = info

        self.typeBox = misc_widget.SearchComboBox() # uncertainty type
        self.typeBox.info = info
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
        if 'value' in kwargs:
            self.valBox = UncertaintyBox(parent, self.uncMax, value=kwargs['value'])  # uncertainty value
        else:
            self.valBox = UncertaintyBox(parent, self.uncMax)  # uncertainty value
        self.valBox.info = info

        if 'unc_choice' in kwargs:
            self.typeBox.setCurrentText(kwargs['unc_choice'])
            self.uncTypeChanged(kwargs['unc_choice'], update=False)   # initialize uncertainty type
        else:
            self.valBox.setUncType(self.priorUncType)   # initialize uncertainty type

        if 'value' in kwargs:
            self.valBox.setValue(kwargs['value'])
            
        layout = QGridLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(4)
        
        layout.addWidget(self.typeBox, 0, 0)
        layout.addWidget(self.valBox, 0, 1)
       
    def uncTypeChanged(self, event, update=True):
        def plus_minus_values():
            coefBox = self.info['mainValueBox']
            return coefBox.strDecimals, coefBox.singleStep()*10, sys.float_info.max
            
        self.valBox.setUncType(event)   # pass event change to uncValBox
        if self.priorUncType == '±' and update:
            self.valBox.setValue(self.valBox.minimum())  # if prior uncertainty was +-, reset

        uncVal = self.valBox.uncValue
        
        if event == 'F':    # only happens on event change, must have been % or ±
            if not np.isnan(uncVal) and update:
                self.valBox.setValue(uncVal+1)
            self.valBox.setMinimum(1)
        elif self.priorUncType == 'F':
            self.valBox.setMinimum(0)
            if update:
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
        if self._isNum(string) or string == '-':
            state = QtGui.QValidator.Acceptable
        elif string == "" or string[position-1].lower() in 'e.-+':
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
        
        self.minimumBaseValue= 1
        self.maximumBaseValue = maxUnc
        self.setSingleIntStep(0.25)
        self.setDecimals(2)
        if value is None:
            self.setValue(-1)
        else:
            self.setValue(value)
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
            tree = self.tree
            rxnNum = self.info['rxnNum']
            coeffs = [*self.info['coef'], event]
            conv_type = tree.mech_tree_type + '2Cantera'
            cantera_value = tree.convert._arrhenius(rxnNum, deepcopy([coeffs]), conv_type)
            self.uncValue = cantera_value[0][2]
    
    def setUncType(self, event):    self.uncType = event
       
   
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
        mech = parent.mech
        rxnNum, boxType, coefName = self.info['rxnNum'], self.info['type'], self.info['coefName']
        coef_key, bnds_key = keysFromBox(self, mech)
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
                var[name] = mech.coeffs_bnds[rxnNum][bnds_key][coefName]['resetVal']
                if len(names) == 1 and formula[-1] == ']' and adjustment not in ['*1.0', '+0.0']:
                    formula += adjustment
                    self.formula += adjustment
            else:
                for box in tree.rxn[subRxnNum]['valueBox']:
                    if box.info['coefName'] == coefName:
                        var[name] = mech.coeffs[subRxnNum][coef_key][coefName]
        
        mech.coeffs[rxnNum][coef_key][coefName] = value = eval(formula)
        conv_type = 'Cantera2' + parent.tree.mech_tree_type
        coeffs = [*self.info['coef'], value]
        value = parent.convert_units._arrhenius(rxnNum, deepcopy([coeffs]), conv_type)[0][2]
        
        self.displayMode = 'value'
        self.setReadOnly(True)
        self.setValue(value, emit)