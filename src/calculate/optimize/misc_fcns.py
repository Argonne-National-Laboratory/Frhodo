# This file is part of Frhodo. Copyright © 2020, UChicago Argonne, LLC
# and licensed under BSD-3-Clause. See License.txt in the top-level 
# directory for license and copyright information.

import numpy as np
import cantera as ct


Ru = ct.gas_constant

min_pos_system_value = (np.finfo(float).tiny*(1E20))**(1/2)
max_pos_system_value = (np.finfo(float).max*(1E-20))**(1/2)
min_neg_system_value = -max_pos_system_value
T_min = 300
T_max = 6000

default_arrhenius_coefNames = ['activation_energy', 'pre_exponential_factor', 'temperature_exponent']


def rates(rxn_coef_opt, mech):
    output = []
    for rxn_coef in rxn_coef_opt:
        rxnIdx = rxn_coef['rxnIdx']
        for n, (T, P) in enumerate(zip(rxn_coef['T'], rxn_coef['P'])):
            mech.set_TPX(T, P)
            output.append(mech.gas.forward_rate_constants[rxnIdx])
            
    return np.log(output)


def set_bnds(mech, rxnIdx, keys, coefNames):
    rxn = mech.gas.reaction(rxnIdx)
    coef_bnds = {'lower': [], 'upper': [], 'exist': []}
            
    for coefNum, (key, coefName) in enumerate(zip(keys, coefNames)):
        if coefName not in default_arrhenius_coefNames: continue    # skip anything not Arrhenius. Falloff follows this

        coef_x0 = mech.coeffs_bnds[rxnIdx][key['coeffs_bnds']][coefName]['resetVal']
        coef_limits = mech.coeffs_bnds[rxnIdx][key['coeffs_bnds']][coefName]['limits']()

        if np.isnan(coef_limits).any():
            coef_bnds['exist'].append([False, False])
            # set lower bnds
            if coefName == 'activation_energy':
                if coef_x0 > 0:
                    coef_bnds['lower'].append(0)                                # Ea shouldn't change sign
                else:
                    coef_bnds['lower'].append(-Ru*T_min*np.log(max_pos_system_value))
            elif coefName == 'pre_exponential_factor':
                coef_bnds['lower'].append(min_pos_system_value)             # A should be positive
            elif coefName == 'temperature_exponent':
                coef_bnds['lower'].append(np.log(min_pos_system_value)/np.log(T_max))
            elif not isinstance(coefName, int):     # ints will be falloff, they will be taken care of below
                coef_bnds['lower'].append(min_neg_system_value)
                    
            # set upper bnds
            if coefName == 'activation_energy':
                if coef_x0 < 0:   # Ea shouldn't change sign
                    coef_bnds['upper'].append(0)
                else:
                    coef_bnds['upper'].append(-Ru*T_max*np.log(min_pos_system_value))
            elif coefName == 'temperature_exponent':
                coef_bnds['upper'].append(np.log(max_pos_system_value)/np.log(T_max))
            elif not isinstance(coefName, int):
                coef_bnds['upper'].append(max_pos_system_value)
        else:
            coef_bnds['lower'].append(coef_limits[0])
            coef_bnds['upper'].append(coef_limits[1])
            coef_bnds['exist'].append([True, True])
            
    if type(rxn.rate) in [ct.PlogRate, ct.FalloffRate, ct.TroeRate, ct.SriRate]:
        for coef in ['A', 'T3', 'T1', 'T2']:
            coef_bnds['exist'].append([False, False])
            coef_bnds['lower'].append(min_neg_system_value)      
            coef_bnds['upper'].append(max_pos_system_value)

    coef_bnds['exist'] = np.array(coef_bnds['exist'])
    coef_bnds['lower'] = np.array(coef_bnds['lower'])
    coef_bnds['upper'] = np.array(coef_bnds['upper'])

    return coef_bnds


def set_arrhenius_bnds(x0, coefNames):
    bnds = [[], []]
    for n, coefName in enumerate(coefNames):
        # lower bnds
        if coefName == 'activation_energy':
            if x0[n] > 0:
                bnds[0].append(0)                            # Ea shouldn't change sign
            else:
                bnds[0].append(-Ru*T_min*np.log(max_pos_system_value))
        elif coefName == 'pre_exponential_factor':
            bnds[0].append(min_pos_system_value)             # A should be positive
        elif coefName == 'temperature_exponent':
            bnds[0].append(np.log(min_pos_system_value)/np.log(T_max))
        
        # set upper bnds
        if coefName == 'activation_energy':
            if x0[n] < 0:   # Ea shouldn't change sign
                bnds[1].append(0)
            else:
                bnds[1].append(-Ru*T_max*np.log(min_pos_system_value))
        elif coefName == 'pre_exponential_factor':
            bnds[1].append(max_pos_system_value)
        elif coefName == 'temperature_exponent':
            bnds[1].append(np.log(max_pos_system_value)/np.log(T_max))         

    return bnds