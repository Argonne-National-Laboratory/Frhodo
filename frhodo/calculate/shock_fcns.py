# This file is part of Frhodo. Copyright © 2020, UChicago Argonne, LLC
# and licensed under BSD-3-Clause. See License.txt in the top-level 
# directory for license and copyright information.

"""
Shock Tube Reactor

Implementation of shock tube governing equations as derived by Franklin
Goldsmith.

--------------------------------------------------------------------------------

Copyright (c) 2016 Raymond L. Speth

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from __future__ import division

import cantera as ct
import numpy as np
from scipy.optimize import root
from timeit import default_timer as timer

# Ru = 8314.4598 # J/kmol K
Ru = ct.gas_constant

# Main ODE solver
class ReactorOde(object):
    def __init__(self, gas, t_lab_end, rhoI, L=0.1, As=0.2, A1=0.2, area_change=False):
        # Parameters of the ODE system and auxiliary data are stored in the
        # ReactorOde object.
        self.gas = gas
        self.N = gas.n_species + 6
        self.L = L
        self.As = As
        self.A1 = A1
        self.n = 0.5
        self.Wk = self.gas.molecular_weights
        self.rho1 = gas.density
        self.rhoI = rhoI        #initial density
        self.delta_dA = area_change
        self.t_lab_end = t_lab_end

    def __call__(self, t, y):
        '''
        ODE function, y' = f(t,y)
        State vector is [z, A, rho, v, T, tlab, Y_1, Y_2, ... Y_K]
                         0  1   2   3  4   5    6 ...
        '''
        z, A, rho, v, T, tlab = y[:6]
        if np.isnan(T) or T <= 0:   # return error to stop solver
            raise Exception('ODE Error: Temperature is invalid')
        if np.isnan(rho) or rho <= 0:
            raise Exception('ODE Error: Density is invalid')
            
        self.gas.set_unnormalized_mass_fractions(y[6:])
        self.gas.TD = T, rho
        cp = self.gas.cp_mass
        Wmix = self.gas.mean_molecular_weight
        hk = self.gas.partial_molar_enthalpies
        wdot = self.gas.net_production_rates

        if self.delta_dA:
            xi = max(z / self.L, 1e-10)
            dA_dt = v * self.As*self.n/self.L * xi**(self.n-1.0)/(1.0-xi**self.n)**2.0
        else:
            dA_dt = 0.0
        
        beta = v**2 * (1.0/(cp*T) - Wmix / (Ru * T))

        ydot = np.zeros(self.N)
        ydot[0] = v # dz/dt
        ydot[1] = dA_dt # dA/dt
        ydot[2] = 1/(1+beta) * (sum((hk/(cp*T) - Wmix) * wdot) - rho*beta/A * ydot[1]) # drho/dt
        ydot[3] = -v * (ydot[2]/rho + ydot[1]/A) # dv/dt
        ydot[4] = -(np.dot(wdot, hk)/rho + v*ydot[3]) / cp # dT/dt
        ydot[5] = 1 # dt_shock/dt_lab (converted below)
        ydot[6:] = wdot * self.Wk / rho # dYk/dt
        
        ydot = ydot*rho*A/(self.rhoI*self.A1) # convert from d/dt to d/dt_lab
        
        return ydot

    def jacobian(self, t, y):   # Might increase speed to include this, not currently set up
        '''
        ODE function, y' = f(t,y)
        State vector is [z, A, rho, v, T, tlab, Y_1, Y_2, ... Y_K]
                         0  1   2   3  4   5    6 ...
        '''
        z, A, rho, v, T, tlab = y[:6]
        if np.isnan(T) or T <= 0:   # return error to stop solver
            raise Exception('ODE Error: Temperature is invalid')
        if np.isnan(rho) or rho <= 0:
            raise Exception('ODE Error: Density is invalid')
            
        self.gas.set_unnormalized_mass_fractions(y[6:])
        self.gas.TD = T, rho
        cp = self.gas.cp_mass
        Wmix = self.gas.mean_molecular_weight
        hk = self.gas.partial_molar_enthalpies
        wdot = self.gas.net_production_rates

        if self.delta_dA:
            xi = max(z / self.L, 1e-10)
            dA_dt = v * self.As*self.n/self.L * xi**(self.n-1.0)/(1.0-xi**self.n)**2.0
        else:
            dA_dt = 0.0
        
        beta = v**2 * (1.0/(cp*T) - Wmix / (Ru * T))

        jac = np.zeros(self.N, self.N)
        jac[0] = v # dz/dt
        jac[1] = dA_dt # dA/dt
        jac[2] = 1/(1+beta) * (sum((hk/(cp*T) - Wmix) * wdot) - rho*beta/A * jac[1]) # drho/dt
        jac[3] = -v * (jac[2]/rho + jac[1]/A) # dv/dt
        jac[4] = -(np.dot(wdot, hk)/rho + v*jac[3]) / cp # dT/dt
        jac[5] = 1 # dt_shock/dt_lab (converted below)
        jac[6:] = wdot * self.Wk / rho # dYk/dt
        
        jac = jac*rho*A/(self.rhoI*self.A1) # convert from d/dt to d/dt_lab
        
        return jac


# compute the density gradient from the solution
def drhodz(states, L=0.1, As=0.2, A1=0.2, area_change=False):
    n = 0.5

    z = states.z
    A = states.A
    rho = states.density
    vel = states.vel
    T = states.T
    cp = states.cp_mass
    Wmix = states.mean_molecular_weight
    hk = states.partial_molar_enthalpies
    wdot = states.net_production_rates

    beta = vel**2 * (1.0/(cp*T) - Wmix / (Ru * T))
    outer = 1/vel/(1+beta)

    species_term = np.sum((hk/(cp*T)[:,None] - Wmix[:,None]) * wdot, axis=1)

    if area_change:
        xi = max(z / L, 1e-10)
        dAdt = vel * As*n/L * xi**(n-1.0)/(1.0-xi**n)**2.0 # dA/dt
        area_change_term = rho*beta/A*dAdt
    else:
        area_change_term = 0.0
    
    return outer*(species_term - area_change_term)

# compute the contribution of each reaction to the density gradient
def drhodz_per_rxn(states, L=0.1, As=0.2, A1=0.2, area_change=False, rxnNum=None):
    n = 0.5

    z = states.z
    A = states.A
    rho = states.density
    vel = states.vel
    T = states.T
    cp = states.cp_mass
    Wmix = states.mean_molecular_weight
    nu_fwd = states.product_stoich_coeffs()
    nu_rev = states.reactant_stoich_coeffs()
    delta_N = np.sum(nu_fwd, axis=0) - np.sum(nu_rev, axis=0)

    if rxnNum is None:
        rxns = range(states.n_reactions)
    elif isinstance(rxnNum, list):
        rxns = rxnNum
    else:
        rxns = [rxnNum]

    # per reaction properties
    rj = states.net_rates_of_progress[:,rxns]
    hj = states.delta_enthalpy[:,rxns]

    beta = vel**2 * (1.0/(cp*T) - Wmix/(Ru*T))
    outer = 1/vel/(1+beta)

    if area_change:
        xi = max(z/L, 1e-10)
        dAdt = vel * As*n/L * xi**(n - 1.0)/(1.0 - xi**n)**2.0 # dA/dt
        area_change_term = rho*beta/A*dAdt
    else:
        area_change_term = 0.0

    species_term = rj*(hj/(cp*T)[:,None] - Wmix[:, None]*delta_N)

    return outer[:,None]*(species_term - area_change_term)
       
"""
Written by Travis Sikes
"""

# Normal Shock Solver
class Properties():
    def __init__(self, gas, shock_vars, parent = None):
        self.gas = gas
        self.X_driven = shock_vars.pop('mix') # return value and remove from shock_vars
        if 'mix_driver' in shock_vars:  # if I one day want to include P4   
            self.X_driver  = shock_vars.pop('mix_driver') 
        
        self.success = True
        try:
            self.res = self.solve(shock_vars)
        except Exception as e: # If fails switch to log
            if parent:
                # parent.log.append('Error in loading Shock {:d}:'.format(parent.var['shock_choice']))
                parent.log.append(e)
            else:
                print(e)
            self.success = False

    def _create_zone(self, shock_vars):
        self.zone = []
        for i in range(6):
            if i in [1, 2, 5]:  # Create gas object for each shock tube zone
                self.zone.append({'X': self.X_driven})
            elif i == 4 and hasattr(self, 'X_driver'):
                self.zone.append({'X': self.X_driver})
            else:
                self.zone.append([])
        
        for var, val in shock_vars.items():
            self.zone[int(var[1])][var[0]] = val
    
    def _set_gas(self, T, P, X):
        if T <= 0:
            raise Exception('Shock Solver Error:\nTemperature is negative Kelvin')
        if P <= 0:
            raise Exception('Shock Solver Error:\nPressure is negative')
        
        self.gas.TPX = T, P, X
    
    def _shock_variables(self, knownVars, unknownVars=[], x=[]):
        var_dict = {}   # temporary holder for all variables, both known and iterated upon. Room for improvement here
        for var in knownVars:
            var_dict[var] = self.zone[int(var[1])][var[0]]
        
        for n, var in enumerate(unknownVars):
            var_dict[var] = x[n]
                
        return var_dict
    
    def _mach(self, zone_num=1):
        zone, n = self.zone, zone_num
        
        self._set_gas(zone[n]['T'], zone[n]['P'], zone[n]['X']) # this might be slightly slower
        gamma = self.gas.cp/self.gas.cv
        a1 = np.sqrt(gamma*Ru/self.gas.mean_molecular_weight*self.zone[n]['T'])
        M1 = self.zone[n]['u']/a1
        
        return M1
    
    def _perfect_gas_shock(self):
        zone = self.zone
        gas = self.gas
        
        T1 = zone[1]['T']
        P1 = zone[1]['P']
        self._set_gas(zone[1]['T'], zone[1]['P'], zone[1]['X'])
        # self.gas.equilibrate('TP', rtol=1.0e-6, maxiter=5000)    # If needed add to _Frosh too
        
        M1 = zone[1]['Mach'] = self._mach()

        # Used often so save some computations!
        gamma = self.gas.cp/self.gas.cv
        gp = gamma + 1
        gp_gm = gp/(gamma - 1)
        
        # Solve perfect gas shock equations for zone 2
        P2_P1 = 1+2*gamma/gp*(M1**2-1)
        zone[2]['P'] = P1*P2_P1
        zone[2]['T'] = T1*(P2_P1*(gp_gm+P2_P1)/(1+gp_gm*P2_P1))
                
        # Solve perfect gas shock equations for zone 5
        P5_P2 = ((gp_gm+2)*P2_P1-1)/(P2_P1+gp_gm)
        zone[5]['P'] = P5_P2*zone[2]['P']
        T5_T2 = P5_P2*(gp_gm+P5_P2)/(1+gp_gm*P5_P2)
        zone[5]['T'] = T5_T2*zone[2]['T']  

    def _perfect_gas_shock_zero(self, vars, knownVals, x):
        self.zone[1]['P'] = x[0]
        self.zone[1]['u'] = x[1]
        self._perfect_gas_shock()
        shockVarDict = self._shock_variables(vars['known'])
        
        zero = []
        for var, val in zip(vars['known'], knownVals):
            if var != 'T1':
                zero.append(shockVarDict[var] - val)
        
        return zero
        
    def _Frosh(self, vars, x, type):    # should work for any 4 missing variables given that 1 known is a pressure
        shockVarDict = self._shock_variables(vars['known'], vars['unknown'], x)
        zone = self.zone
        
        u1 = zone[1]['u'] = shockVarDict['u1']
        T1 = zone[1]['T'] = shockVarDict['T1']
        P1 = zone[1]['P'] = shockVarDict['P1']
        self._set_gas(zone[1]['T'], zone[1]['P'], zone[1]['X'])
        h1 = self.gas.enthalpy_mass
        zone[1]['rho'] = self.gas.density
        v1 = 1/zone[1]['rho']
        
        T2 = zone[2]['T'] = shockVarDict['T2']
        P2 = zone[2]['P'] = shockVarDict['P2']
        self._set_gas(zone[2]['T'], zone[2]['P'], zone[2]['X'])
        h2 = self.gas.enthalpy_mass
        zone[2]['rho'] = self.gas.density
        v2 = 1/zone[2]['rho']
        u2 = zone[2]['u'] = u1*v2/v1    # Conservation of Mass across shock
        
        T5 = zone[5]['T'] = shockVarDict['T5']
        P5 = zone[5]['P'] = shockVarDict['P5']
        self._set_gas(zone[5]['T'], zone[5]['P'], zone[5]['X'])
        h5 = self.gas.enthalpy_mass
        zone[5]['rho'] = self.gas.density
        v5 = 1/zone[5]['rho']
        zone[5]['u'] = u2*v5/v2         # Conservation of Mass across shock
        
        R = Ru/self.gas.mean_molecular_weight   # Since we're assuming frozen chemistry, it doesn't change

        # Computation savers
        u1s = u1**2
        a = T2*P1/(T1*P2)
        b = T5*P2/(T2*P5)

        if type == 'fcn':
            zero = [(P2/P1-1) + (u1s*(a-1)/(R*T1)),                    # Conservation of Momentum, inc shock
                    (2/u1s*(h2-h1)) + (a**2-1),                          # Conservation of Energy, inc shock
                    (P5/P2-1) + (u1s/(R*T2)*(1-a)**2/(b-1)),          # Conservation of Momentum, ref shock
                    (2*(h5-h2)/(u1s*(1-a)**2)) + 2/(b-1) + 1]        # Conservation of Energy, ref shock

        elif type == 'jac':
            for i, var in enumerate(vars['unknown']):   # Sets partials in jacobian according to unknowns
                f1, f2, f3, f4 = 0, 0, 0, 0
                if var == 'T1':
                    f1 = u1s/(R*T1**2)*(1 - 2*a)
                    f2 = -2/T1*(h1/u1s + a**2)
                    
                elif var == 'P1':
                    f1 = u1s*a/(R*T1*P1) - P2/P1**2
                    f2 = 2*a**2/P1
                    
                elif var == 'u1':
                    f1 = 2*u1/(R*T1)*(a-1)
                    f2 = 4/u1**3*(h1-h2)
                    f3 = 2*u1/(R*T2)*(1-a)**2/(b-1)
                    f4 = 4/u1**3*(h2-h5)/(1-a)**2
                   
                elif var == 'T2':
                    f1 = u1s*a/(R*T1*T2)
                    f2 = 2/T2*(h2/u1s + a**2)
                    f3 = u1s/(R*T2**2)*(1-a)/(b-1)**2*(1+a*(1-2*b))
                    f4 = 2/T2*(1/u1s*(a*(2*h5-h2)-h2)/(1-a)**3 + b/(b-1)**2)
                
                elif var == 'P2':
                    f1 = 1/P1 - u1s*a/(R*T1*P2)
                    f2 = -2/P2*a**2
                    f3 = -P5/P2**2 + u1s/(R*T2*P2)*(1-a)/(b-1)**2*(a*(3*b-2)-b)
                    f4 = -1/P2*2*b/(b-1)**2 
                    
                elif var == 'T5':
                    f3 = -u1s/(R*T2*T5)*b*((1-a)/(b-1))**2
                    f4 = 2/T5*(h5/(u1s*(1-a)**2) - b/(b**2-1))
                
                elif var == 'P5':
                    f3 = 1/P2 + u1s/(R*T2*P5)*b*((1-a)/(b-1))**2
                    f4 = 1/P5*2*b/(b**2-1)
                
                temp = np.array([f1, f2, f3, f4]).reshape(-1,1)
                if i == 0:
                    zero = temp
                else:
                    zero = np.concatenate((zero, temp), axis = 1)
        
        return zero
    
    def _P4_eqn(self):
        if not hasattr(self, 'X_driver'):
            return np.nan
            
        zone = self.zone
        for i in [1, 4]:
            self._set_gas(zone[1]['T'], zone[1]['P'], zone[i]['X'])
            zone[i]['MW'] = self.gas.mean_molecular_weight
            zone[i]['gamma'] = self.gas.cp/self.gas.cv
    
        gam1 = zone[1]['gamma']
        gam4 = zone[4]['gamma']

        P2_P1 = zone[2]['P']/zone[1]['P']
        a1_a4 = np.sqrt((gam1/zone[1]['MW'])/(gam4/zone[4]['MW']))  # assume T4 = T1
        P4_P1 = P2_P1*np.power(1-((gam4-1)*a1_a4*P2_P1)/(np.sqrt(2*gam1*(2*gam1+(gam1+1)*(P2_P1-1)))), 2*gam4/(1-gam4))
        
        # M1 = zone[1]['Mach']
        # a = 2*gam1/(gam1+1)
        # b = (gam4-1)/(gam1+1)*np.sqrt((gam1/zone[1]['MW'])/(gam4/zone[4]['MW']))
        # c = (2*gam4/(1-gam4))
        
        # P4_P1 = (1+a*(M1**2-1))*np.power(1-b*(M1**2-1)/M1, c)
        
        return P4_P1*zone[1]['P']
    
    def solve(self, shock_vars, tol = 1E-10):
        self._create_zone(shock_vars)
        
        vars = {'known': [], 'unknown': ['T1', 'P1', 'u1', 'T2', 'P2', 'T5', 'P5']}
        for var in shock_vars:
            vars['known'].append(var)
            vars['unknown'].remove(var)
        
        if set(['T1', 'P1', 'u1']) == set(vars['known']):
            self._perfect_gas_shock()               # initial guess is based on perfect gas shock
        else:       # find a good initial guess based on finding root of given parameters and perfect gas shock
            x0 = [1000, 1000]   # initial guesses of [P1, u1] for solving perfect gas shock 
            knownVarDict = self._shock_variables(vars['known'])
            knownVals = knownVarDict.values()
            root(lambda x, knownVals=knownVals: self._perfect_gas_shock_zero(vars, knownVals, x), x0, 
                 method='hybr')
        
        x0 = []
        for var in vars['unknown']:
            x0.append(self.zone[int(var[1])][var[0]])
            
        result = root(lambda x: self._Frosh(vars, x, type = 'fcn'), x0, method='hybr',
                      jac = lambda x: self._Frosh(vars, x, type = 'jac'), tol = tol,
                      options={'xtol': tol})
        
        # Post-Solution clean up and output
        zone = self.zone
        zone[1]['Mach'] = self._mach()
        output = {'u1': zone[1]['u'], 'rho1': zone[1]['rho'], 'M': zone[1]['Mach'],
                  'T1': zone[1]['T'], 'P1': zone[1]['P'],
                  'T2': zone[2]['T'], 'P2': zone[2]['P'], 'u2': zone[2]['u'],
                  'P4': self._P4_eqn(),
                  'T5': zone[5]['T'], 'P5': zone[5]['P']}
        
        return output