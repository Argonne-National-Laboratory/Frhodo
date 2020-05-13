# This file is part of Frhodo. Copyright © 2020, UChicago Argonne, LLC
# and licensed under BSD-3-Clause. See License.txt in the top-level 
# directory for license and copyright information.

import numpy as np
import cantera as ct
import sys

conv2ct = {'Torr': 101325/760, 'kPa': 1E3, 'atm': 101325, 'bar': 100000, 
           'psi': 4.4482216152605/0.00064516, 'cm/s': 1E-2, 'mm/μs': 1000, 
           'ft/s': 1/3.28084, 'in/s': 1/39.37007874, 'mph': 1609.344/60**2,
           'kcal': 1/4184, 'cal': 1/4.184}

class Convert_Units:
    def __init__(self, parent):
        self.parent = parent

    def __call__(self, value, units, unit_dir = 'in'):
        units = units.replace('[', '').replace(']', '')
        return self._convert_units(value, units, unit_dir)
    
    def _basic2Cantera(self, value, units):
        if 'K' == units:     return value
        elif '°C' == units:  return value + 273.15
        elif '°F' == units:  return (value - 32)*5/9 + 273.15
        elif '°R' == units:  return value*5/9
        elif 'Pa' == units:  return value
        elif 'm/s' == units: return value
        else: return value*conv2ct[units]
       
    def _basic2Display(self, value, units):
        if 'K' == units:     return value
        elif '°C' == units:  return value - 273.15
        elif '°F' == units:  return (value - 273.15)*9/5 + 32
        elif '°R' == units:  return value*9/5
        elif 'Pa' == units:  return value
        elif 'm/s' == units: return value
        else: return value/conv2ct[units]
    
    def _arrhenius(self, rxnIdx, coeffs, conv_type):
        # coef_sum_all = np.sum(self.parent.mech.gas.reactant_stoich_coeffs(), axis=0)
        # coef_sum = coef_sum_all[rxnIdx]
        coef_sum = sum(self.parent.mech.gas.reaction(rxnIdx).reactants.values())
        if type(self.parent.mech.gas.reactions()[rxnIdx]) is ct.ThreeBodyReaction:
            coef_sum += 1     
        
        conv_factor = {'Cantera2Bilbo':   {'A':  lambda x: np.log10(x*np.power(1E3,coef_sum-1)), 
                                           'Ea': lambda x: x/4.184E6},
                       'Cantera2Chemkin': {'A':  lambda x: x*np.power(1E3,coef_sum-1), 
                                           'Ea': lambda x: x/4.184E3},
                       'Bilbo2Cantera':   {'A':  lambda x: np.power(10,x)/np.power(1E3,coef_sum-1), 
                                           'Ea': lambda x: x*4.184E6},
                       'Chemkin2Cantera': {'A':  lambda x: x/np.power(1E3,coef_sum-1), 
                                           'Ea': lambda x: x*4.184E3}}
        
        for coef in coeffs: # coef of format [coef_name, coef_abbreviation, coef_value]
            if 'pre_exponential_factor' in coef:
                if 'Cantera2Bilbo' in conv_type:
                    coef[0] = 'log(A)'          # Corrects shorthand
                with np.errstate(over='raise'):
                    try:
                        coef[2] = conv_factor[conv_type]['A'](coef[2])
                    except Exception as e: # If fails switch to log
                        coef[2] = sys.float_info.max
                        self.parent.log.append(e)
            elif 'activation_energy' in coef:
                if coef[2] != 0:
                    coef[2] = conv_factor[conv_type]['Ea'](coef[2])
        return coeffs
    
    def _convert_units(self, value, units, unit_dir):
        if unit_dir in ['in', '2ct', 'to_ct', '2cantera', 'to_cantera']:
           return self._basic2Cantera(value, units)
        else:
           return self._basic2Display(value, units)
           
    