# This file is part of Frhodo. Copyright © 2020, UChicago Argonne, LLC
# and licensed under BSD-3-Clause. See License.txt in the top-level 
# directory for license and copyright information.

import numpy as np
from scipy import interpolate
from numba import jit
import cantera as ct
import pathlib, sys

import frhodo

Ru = ct.gas_constant

min_pos_system_value = (np.finfo(float).tiny*(1E20))**(1/2)
max_pos_system_value = (np.finfo(float).max*(1E-20))**(1/2)
min_neg_system_value = -max_pos_system_value
T_min = 300
T_max = 6000

default_arrhenius_coefNames = ['activation_energy', 'pre_exponential_factor', 'temperature_exponent']


# interpolation function for Z from loss function
path = {'main': pathlib.Path(frhodo.__file__).parents[0].resolve()}
path['Z_tck_spline.dat'] = path['main'] / 'data/loss_partition_fcn_tck_spline.dat'

tck = []
with open(path['Z_tck_spline.dat']) as f:
    for i in range(5):
        tck.append(np.array(f.readline().split(','), dtype=float))        

ln_Z = interpolate.RectBivariateSpline._from_tck(tck)

def rates(rxn_coef_opt, mech):
    output = []
    for rxn_coef in rxn_coef_opt:
        rxnIdx = rxn_coef['rxnIdx']
        for n, (T, P) in enumerate(zip(rxn_coef['T'], rxn_coef['P'])):
            mech.set_TPX(T, P)
            output.append(mech.gas.forward_rate_constants[rxnIdx])
            
    return np.log(output)

def weighted_quantile(values, quantiles, weights=None, values_sorted=False, old_style=False):
    """ https://stackoverflow.com/questions/21844024/weighted-percentile-using-numpy
    Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    finite_idx = np.where(np.isfinite(values))
    values = np.array(values)[finite_idx]
    quantiles = np.array(quantiles)
    if weights is None or len(weights) == 0:
        weights = np.ones_like(values)
    else:
        weights = np.array(weights)[finite_idx]

    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        weights = weights[sorter]

    weighted_quantiles = np.cumsum(weights) - 0.5 * weights
    if old_style: # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(weights)

    return np.interp(quantiles, weighted_quantiles, values)

def outlier(x, c=1, weights=[], max_iter=25, percentile=0.25):
    def diff(x_outlier):
        if len(x_outlier) < 2: 
            return 1
        else:
            return np.diff(x_outlier)[0]

    x = np.abs(x.copy())
    percentiles = [percentile, 1-percentile]
    x_outlier = []
    # define outlier with 1.5 IQR rule
    for n in range(max_iter):
        if diff(x_outlier) == 0:   # iterate until res_outlier is the same as prior iteration
            break
                
        if len(x_outlier) > 0:
            x = x[x < x_outlier[-1]] 
            
        [q1, q3] = weighted_quantile(x, percentiles, weights=weights)
        iqr = q3 - q1       # interquartile range      
            
        if len(x_outlier) == 2:
            del x_outlier[0]
            
        x_outlier.append(q3 + iqr*1.5)
        
    x_outlier = x_outlier[-1]

    return x_outlier*c # decreasing outliers increases outlier rejection

@jit(nopython=True, error_model='numpy') 
def generalized_loss_fcn(x, mu=0, a=2, c=1):    # defaults to sum of squared error
    x_c_2 = ((x-mu)/c)**2
    
    if a == 1:          # generalized function reproduces
        loss = (x_c_2 + 1)**(0.5) - 1
    if a == 2:
        loss = 0.5*x_c_2
    elif a == 0:
        loss = np.log(0.5*x_c_2+1)
    elif a == -2:       # generalized function reproduces
        loss = 2*x_c_2/(x_c_2 + 4)
    elif a <= -100:    # supposed to be negative infinity
        loss = 1 - np.exp(-0.5*x_c_2)
    else:
        loss = np.abs(a-2)/a*((x_c_2/np.abs(a-2) + 1)**(a/2) - 1)
    
    #loss = np.exp(np.log(loss) + a*np.log(c)) + mu  # multiplying by c^a is not necessary, but makes order appropriate
    #loss = loss*c**a + mu  # multiplying by c^a is not necessary, but makes order appropriate
    loss = loss + mu

    return loss

# penalize the loss function using approximate partition function
tau_min = 1.0
tau_max = 250.0
def penalized_loss_fcn(x, mu=0, a=2, c=1, use_penalty=True): # defaults to sum of squared error
    loss = generalized_loss_fcn(x, mu, a, c)

    if use_penalty:
        tau = 10.0*c
        if tau < tau_min:
            tau = tau_min
        elif tau > tau_max:
            tau = tau_max

        penalty = np.log(c) + ln_Z(tau, a)[0][0]        # approximate partition function
        loss += penalty

        if not np.isfinite(loss).any():
            print(mu, a, c, penalty)
            print(x)

    #non_zero_idx = np.where(loss > 0.0)
    #ln_loss = np.log(loss[non_zero_idx])
    #loss[non_zero_idx] = np.exp(ln_loss + a*np.log(c)) + mu

    return loss


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
            
    if type(rxn) in [ct.FalloffReaction, ct.PlogReaction]:
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