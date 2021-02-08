# This file is part of Frhodo. Copyright © 2020, UChicago Argonne, LLC
# and licensed under BSD-3-Clause. See License.txt in the top-level 
# directory for license and copyright information.

import numpy as np
import cantera as ct
import warnings
from copy import deepcopy
from scipy.optimize import curve_fit, OptimizeWarning, approx_fprime
from timeit import default_timer as timer

Ru = ct.gas_constant
# Ru = 1.98720425864083

default_arrhenius_coefNames = ['activation_energy', 'pre_exponential_factor', 'temperature_exponent']

def fit_arrhenius(rates, T, x0=[], coefNames=default_arrhenius_coefNames, bnds=[]):
    def fit_fcn_decorator(x0, alter_idx, jac=False):               
        def set_coeffs(*args):
            coeffs = x0
            for n, idx in enumerate(alter_idx):
                coeffs[idx] = args[n]
            return coeffs
        
        def ln_arrhenius(T, *args):
            [Ea, ln_A, n] = set_coeffs(*args)
            return ln_A + n*np.log(T) - Ea/(Ru*T)

        def ln_arrhenius_jac(T, *args):
            [Ea, ln_A, n] = set_coeffs(*args)
            jac = np.array([-1/(Ru*T), np.ones_like(T), np.log(T)]).T
            return jac[:, alter_idx]

        if not jac:
            return ln_arrhenius
        else:
            return ln_arrhenius_jac

    ln_k = np.log(rates)
    if len(x0) == 0:
        x0 = np.polyfit(np.reciprocal(T), ln_k, 1)
        x0 = np.array([-x0[0]*Ru, x0[1], 0]) # Ea, ln(A), n
    else:
        x0 = np.array(x0)
        x0[1] = np.log(x0[1])
    
    idx = []
    for n, coefName in enumerate(default_arrhenius_coefNames):
        if coefName in coefNames:
            idx.append(n)
    
    A_idx = None
    if 'pre_exponential_factor' in coefNames:
        A_idx = coefNames.index('pre_exponential_factor')
    
    fit_func = fit_fcn_decorator(x0, idx)
    fit_func_jac = fit_fcn_decorator(x0, idx, jac=True)
    p0 = x0[idx]

    if bnds:
        if A_idx is not None:
            bnds[0][A_idx] = np.log(bnds[0][A_idx])
            bnds[1][A_idx] = np.log(bnds[1][A_idx])

        # only valid initial guesses
        for n, val in enumerate(p0):
            if val < bnds[0][n]:
                p0[n] = bnds[0][n]
            elif val > bnds[1][n]:
                p0[n] = bnds[1][n]

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', OptimizeWarning)
            try:
                popt, _ = curve_fit(fit_func, T, ln_k, p0=p0, method='dogbox', bounds=bnds,
                                    jac=fit_func_jac, x_scale='jac', max_nfev=len(p0)*1000)
            except:
                return
    else:           
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', OptimizeWarning)
            try:
                popt, _ = curve_fit(fit_func, T, ln_k, p0=p0, method='dogbox',
                                    jac=fit_func_jac, x_scale='jac', max_nfev=len(p0)*1000)
            except:
                return
    
    if A_idx is not None:
        popt[A_idx] = np.exp(popt[A_idx])

    return popt

def fit_generic(rates, T, P, X, coefNames, rxnIdx, mech, x0, bnds):
    def fit_rate_eqn(P, X, mech, coefNames, rxnIdx):
        rxn = mech.gas.reaction(rxnIdx)
        def inner(invT, *coeffs):
            if type(rxn) is ct.ElementaryReaction or type(rxn) is ct.ThreeBodyReaction: # if 2 coeffs for Arrhenius
                if len(coeffs) == 2:                                                    # assume n = 0
                    coeffs = np.append(coeffs, 0)

            for coefName, coefVal in zip(coefNames, coeffs):   # updates reaction mechanism for specific reaction
                mech.coeffs[rxnIdx][coefName] = coefVal
            mech.modify_reactions(mech.coeffs, rxnNums=rxnIdx)

            rate = []
            temperatures = np.divide(10000, invT)
            for n, T in enumerate(temperatures):  # TODO FOR PRESSURE DEPENDENT REACTIONS ZIP PRESSURE
                mech.set_TPX(T, P[n], X[n])
                rate.append(mech.gas.forward_rate_constants[rxnIdx])
            return np.log10(rate)
        return inner
    
    def scale_fcn(coefVals, coefNames, rxn, dir='forward'):
        coefVals = np.copy(coefVals)
        # TODO: UPDATE THIS FOR OTHER TYPES OF EXPRESSIONS, Works only for Arrhenius
        if type(rxn) is ct.ElementaryReaction or type(rxn) is ct.ThreeBodyReaction: 
            for n, coefVal in enumerate(coefVals):   # updates reaction mechanism for specific reaction
                if coefNames[n] == 'pre_exponential_factor':
                    if dir == 'forward':
                        coefVals[n] = 10**coefVal
                    else:
                        coefVals[n] = np.log10(coefVal)
        return coefVals
    
    rxn = mech.gas.reaction(rxnIdx)
    
    # Faster and works for extreme values like n = -70
    if type(rxn) is ct.ElementaryReaction or type(rxn) is ct.ThreeBodyReaction:  
        x0 = [mech.coeffs_bnds[rxnIdx][coefName]['resetVal'] for coefName in mech.coeffs_bnds[rxnIdx]]
        coeffs = fit_arrhenius(rates, T, x0=x0, coefNames=coefNames, bnds=bnds)

        if type(rxn) is ct.ThreeBodyReaction and 'pre_exponential_factor' in coefNames:
            A_idx = coefNames.index('pre_exponential_factor')

            if not rxn.efficiencies:
                M = 1/mech.gas.density_mole
            else:
                M = 0
                for (s, conc) in zip(mech.gas.species_names, mech.gas.concentrations):
                    if s in rxn.efficiencies:
                        M += conc*rxn.efficiencies[s]
                    else:
                        M += conc
            
            coeffs[A_idx] = coeffs[A_idx]/M

        return coeffs
    
    # Generic fit if explicit case not specified
    invT = np.divide(10000, T)
    logk = np.log10(rates)
    x0s = scale_fcn(x0, coefNames, rxn, dir='inverse')
    
    if not isinstance(X, (list, np.ndarray)):   # if only a single composition is given, duplicate
        X = [X]*len(invT)
    
    eqn = lambda invT, *x: fit_rate_eqn(P, X, mech, coefNames, rxnIdx)(invT, *scale_fcn(x, coefNames, rxn))
    s = np.abs(approx_fprime(x0s, lambda x: eqn([np.mean(invT)], *x), 1E-9))
    s[s==0] = 1E-9  # TODO: MAKE THIS BETTER running into problem when Ea is zero, this is a janky workaround
    # s /= np.max(s)  # to prevent overflow if s_i is > 1 and unbounded
    scaled_eqn = lambda invT, *x: eqn(invT, *(x/s + x0s))
    bnds[0] = (scale_fcn(bnds[0], coefNames, rxn, dir='inverse') - x0s)*s
    bnds[1] = (scale_fcn(bnds[1], coefNames, rxn, dir='inverse') - x0s)*s
    p0 = np.zeros_like(x0s)
    
    with warnings.catch_warnings():
       warnings.simplefilter('ignore', OptimizeWarning)
       try:
           popt, _ = curve_fit(scaled_eqn, invT, logk, p0=p0, bounds=bnds,
                               method='dogbox', jac='2-point')
       except:
           return

    coeffs = scale_fcn(popt/s + x0s, coefNames, rxn)
    
    return coeffs


def fit_coeffs(rates, T, P, X, coefNames, rxnIdx, x0, bnds, mech):
    if len(coefNames) == 0: return # if not coefs being optimized in rxn, return 
    
    x0 = deepcopy(x0)
    bnds = deepcopy(bnds)

    return fit_generic(rates, T, P, X, coefNames, rxnIdx, mech, x0, bnds)
    

def debug(mech):
    import matplotlib.pyplot as plt
    from timeit import default_timer as timer
    start = timer()
    # rates = [1529339.05689338, 1548270.86688399, 1567437.0352583]
    rates = [1529339.05689338, 1548270.86688399, 1567437.0352583]*np.array([1.000002, 1.00002, 1])
    T = [2387.10188629, 2389.48898818, 2391.88086905]
    P = [16136.20900077, 16136.20900077, 16136.20900077]
    X = {'Kr': 0.99, 'C8H8': 0.01}
        
    coefNames = ['activation_energy', 'pre_exponential_factor', 'temperature_exponent']
    rxnIdx = 0
    coeffs = fit_coeffs(rates, T, P, X, coefNames, rxnIdx, mech)
    print(timer() - start)
    # print(coeffs)
    # print(np.array([2.4442928e+08, 3.4120000e+11, 0.0000000e+00]))
    
    rate_fit = []
    for n, T_val in enumerate(T):
        mech.set_TPX(T_val, P[0], X)
        rate_fit.append(mech.gas.forward_rate_constants[rxnIdx])
    
    print(np.sqrt(np.mean((rates - rate_fit)**2)))

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    start = timer()
    
    rates = np.array([9.13674578])/200
    T = [1513.8026716]
    x0 = [1439225332.24, 5.8499038e+276, -71.113552]
    coefNames = ['pre_exponential_factor']
    bnds = [[2.4424906541753446e-16], [1.7976931348623155e+288]]

    #bnds = [[0, 2.4424906541753446e-16, -1.7976931348623155e+288], 
    #        [1.7976931348623155e+288, 1.7976931348623155e+288, 1.7976931348623155e+288]]
    
    # rates = np.array([9.74253640e-01, 8.74004054e+02, 1.41896847e+05])
    # rates = np.array([1.54283654e-02, 3.89226810e+02, 1.65380781e+04])
    # rates = np.array([4.73813308e+00, 1.39405144e+03, 1.14981010e+05])
    #rates = np.array([6.17844122e-02, 9.74149806e+01, 2.01630443e+04])
    # rates = np.array([2.43094099e-02, 4.02305872e+01, 3.95740585e+03])

    # rates = rates*np.array([1, 1.1, 0.9])
    # rates = [1529339.05689338, 1548270.86688399, 1567437.0352583]*np.array([1, 1.00002, 1])
    #T = [1359.55345014, 1725.11257135, 2359.55345014]

    #print(fit_coeffs(rates, T, P, X, coefNames, rxnIdx, mech))
    [A] = fit_arrhenius(rates, T, x0=x0, coefNames=coefNames, bnds=bnds)
    Ea, n = x0[0], x0[2]
    print(timer() - start)
    print(x0)
    print([Ea, A, n])
    print(A/x0[1])
    
    T_fit = np.linspace(T[0], T[-1], 100)
    rate_fit = A*T_fit**n*np.exp(-Ea/(Ru*T_fit))
    
    plt.plot(10000*np.reciprocal(T), np.log10(rates), 'o')
    plt.plot(10000/T_fit, np.log10(rate_fit))
    plt.show()