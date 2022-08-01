# This file is part of Frhodo. Copyright © 2020, UChicago Argonne, LLC
# and licensed under BSD-3-Clause. See License.txt in the top-level 
# directory for license and copyright information.

import numpy as np
from numba import jit
import cantera as ct
import nlopt, pygmo, rbfopt    # add rbfopt, supposed to be quite good
import warnings, sys, platform, pathlib, io, contextlib
from copy import deepcopy
from scipy.optimize import curve_fit, minimize, root_scalar, OptimizeWarning, least_squares, approx_fprime
from timeit import default_timer as timer
import itertools

import frhodo
from ..convert_units import OoM
from ..optimize.misc_fcns import penalized_loss_fcn, set_arrhenius_bnds

Ru = ct.gas_constant
# Ru = 1.98720425864083

min_pos_system_value = (np.finfo(float).tiny*(1E20))**(0.5)
max_pos_system_value = (np.finfo(float).max*(1E-20))**(0.5)
min_ln_val = np.log(min_pos_system_value)
max_ln_val = np.log(max_pos_system_value)
min_log_val = np.log10(min_pos_system_value)
max_log_val = np.log10(max_pos_system_value)
ln_k_max = np.log(1E60) # max_log_val

default_arrhenius_coefNames = ['activation_energy', 'pre_exponential_factor', 'temperature_exponent']
default_Troe_coefNames = ['activation_energy_0', 'pre_exponential_factor_0', 'temperature_exponent_0', 
                          'activation_energy_inf', 'pre_exponential_factor_inf', 'temperature_exponent_inf', 
                          'A', 'T3', 'T1', 'T2']

troe_falloff_0 = [[1.0,   1E-30,  1E-30,   1500],   # (0, 0, 0)
                  [0.6,     200,    600,   1200],   # (0, 0, 0)
                  [0.05,   1000,  -2000,   3000],   # (0, 1, 0)
                  [0.9,   -2000,    500,  10000]]   # (1, 0, 0)

#troe_bnds = [[-1E2, -1E8, -1E8, -1E4], [1.0, 1E9, 1E9, 1E6]]
troe_all_bnds = {'A':  {'-': [-1E2,  1.0],  '+': [-1E2, 1.0]},
                 'T3': {'-': [-1E8, -1E2],  '+': [ 1.0, 1E9]},
                 'T1': {'-': [-1E8, -1E2],  '+': [ 1.0, 1E9]},
                 'T2': {'-': [-1E4,  1E6],  '+': [-1E4, 1E6]}}


@jit(nopython=True, error_model='numpy')
def ln_arrhenius_k(T, Ea, ln_A, n):   # LPL, HPL, Fcent
    return ln_A + n*np.log(T) - Ea/(Ru*T)

def fit_arrhenius(rates, T, x0=[], coefNames=default_arrhenius_coefNames, bnds=[], loss='linear'):
    def fit_fcn_decorator(x0, alter_idx, jac=False):               
        def set_coeffs(*args):
            coeffs = x0git
            for n, idx in enumerate(alter_idx):
                coeffs[idx] = args[n]
            return coeffs
        
        def ln_arrhenius(T, *args):
            x = set_coeffs(*args)
            return ln_arrhenius_k(T, *x)

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
        if isinstance(coefNames, np.ndarray):
            A_idx = np.argwhere(coefNames == 'pre_exponential_factor')[0]
        else:
            A_idx = coefNames.index('pre_exponential_factor')
    
    fit_func = fit_fcn_decorator(x0, idx)
    fit_func_jac = fit_fcn_decorator(x0, idx, jac=True)
    p0 = x0[idx]

    if len(bnds) == 0:
        bnds = set_arrhenius_bnds(p0, coefNames)

    if A_idx is not None:
        bnds[0][A_idx] = np.log(bnds[0][A_idx])
        bnds[1][A_idx] = np.log(bnds[1][A_idx])

    # only valid initial guesses
    p0 = np.clip(p0, bnds[0], bnds[1])

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', OptimizeWarning)
        try:
            popt, _ = curve_fit(fit_func, T, ln_k, p0=p0, method='trf', bounds=bnds,
                                jac=fit_func_jac, x_scale='jac', max_nfev=len(p0)*1000,
                                loss=loss)
        except:
            return
    
    if A_idx is not None:
        popt[A_idx] = np.exp(popt[A_idx])

    return popt


path = {'main': pathlib.Path(frhodo.__file__).parents[0].resolve()}  # TODO (wardlt) - Make a globally-accessible Path
OS_type = platform.system()
if OS_type == 'Windows':
    path['bonmin'] = path['main'] / 'bonmin/bonmin-win64/bonmin.exe'
    path['ipopt'] = path['main'] / 'ipopt/ipopt-win64/ipopt.exe'
elif OS_type == 'Linux':
    path['bonmin'] = path['main'] / 'bonmin/bonmin-linux64/bonmin'
    path['ipopt'] = path['main'] / 'ipopt/ipopt-linux64/ipopt'
elif OS_type == 'Darwin':
    path['bonmin'] = path['main'] / 'bonmin/bonmin-osx/bonmin'
    path['ipopt'] = path['main'] / 'ipopt/ipopt-osx/ipopt'

@jit(nopython=True, error_model='numpy')
def exp_safe(num, den): # used to include divide by zero check, removed for numba
    x = num/den
    res = np.zeros_like(x) # since initialized to zero, values below min are already zero
    for i, x in enumerate(num/den):
        if x >= min_ln_val and x <= max_ln_val:
            res[i] = np.exp(x)
        elif x > max_ln_val:
            res[i] = max_pos_system_value

    return res

@jit(nopython=True)
def Fcent_calc(T, A, T3, T1, T2):
    exp_T3 = exp_safe(-T, T3)
    exp_T1 = exp_safe(-T, T1)
    exp_T2 = exp_safe(-T2, T)

    Fcent = (1-A)*exp_T3 + A*exp_T1 + exp_T2

    return Fcent

@jit(nopython=True, error_model='numpy')
def ln_Troe(T, M, *x):   # LPL, HPL, Fcent
    Ea_0, ln_A_0, n_0 = x[:3]
    Ea_inf, ln_A_inf, n_inf = x[3:6]
    Fcent_coeffs = x[-4:] # A, T3, T1, T2

    
    ln_k_0 = ln_arrhenius_k(T, Ea_0, ln_A_0, n_0)
    ln_k_inf = ln_arrhenius_k(T, Ea_inf, ln_A_inf, n_inf)

    for idx in np.argwhere(ln_k_0 < min_ln_val):
        ln_k_0[idx[0], idx[1]] = min_ln_val

    for idx in np.argwhere(ln_k_0 > max_ln_val):
        ln_k_0[idx[0], idx[1]] = max_ln_val

    for idx in np.argwhere(ln_k_inf < min_ln_val):
        ln_k_inf[idx[0], idx[1]] = min_ln_val

    for idx in np.argwhere(ln_k_inf > max_ln_val):
        ln_k_inf[idx[0], idx[1]] = max_ln_val

    k_0, k_inf = np.exp(ln_k_0), np.exp(ln_k_inf)

    Fcent = Fcent_calc(T[0,:], *Fcent_coeffs)
    for idx in np.argwhere(Fcent <= 0.0):   # to keep function continuous
        Fcent[idx] = min_pos_system_value

    P_r = k_0/k_inf*M
    for idx in np.argwhere(P_r <= 0.0): 
        P_r[idx] = min_pos_system_value

    log_P_r = np.log10(P_r)    
    log_Fcent = np.log10(Fcent)
    C = -0.4 - 0.67*log_Fcent
    N = 0.75 - 1.27*log_Fcent
    f1 = (log_P_r + C)/(N - 0.14*(log_P_r + C))
    ln_F = np.log(Fcent)/(1 + f1**2)

    log_interior = k_inf*P_r/(1 + P_r)
    for idx in np.argwhere(log_interior <= 0.0):
        log_interior[idx] = min_pos_system_value

    ln_k_calc = np.log(log_interior) + ln_F

    return ln_k_calc

bisymlog_C = 1/(np.exp(1)-1)
class falloff_parameters:   # based on ln_Fcent
    def __init__(self, T, M, ln_k, x0, algo_options):
        self.T = T
        self.M = M
        self.ln_k = ln_k
        self.x0 = x0
        self.s = np.ones_like(x0)

        self.penalty_vars = {'lambda': np.array([1,1,1,1]), 'rho': 1.0, # rho is penalty parameter rho > 0
                             'mu': 0.1}  # mu > 0

        self.Fcent_min = 1E-6
        self.Tmin = 273   # 100
        self.Tmax = 6000  # 20000

        self.algo = algo_options

        self.loss_alpha = self.algo['loss_fcn_param'][0]   # warning unknown how well this functions outside of alpha=2, C=1
        self.loss_scale = self.algo['loss_fcn_param'][1]
        
        # change all E-30 values to np.nan so only T2 is optimized
        if (self.x0[-3:-1] < 10).all() or np.isnan(self.x0).any():
            self.x0[-4:-1] = [1.0, 1.0E-30, 1.0E-30]
            self.Fcent_idx = [9]
        else:
            self.Fcent_idx = [6,7,8,9]

        if all(self.algo['is_P_limit']):
            self.alter_idx = self.Fcent_idx
        elif self.algo['is_P_limit'][0]:
            self.alter_idx = [3,4,5, *self.Fcent_idx]
        elif self.algo['is_P_limit'][1]:
            self.alter_idx = [0,1,2, *self.Fcent_idx]
        else:
            self.alter_idx = [0,1,2,3,4,5, *self.Fcent_idx]

        self.x = deepcopy(self.x0)

    def x_bnds(self, x0):
        bnds = []
        for n, coef in enumerate(['A', 'T3', 'T1', 'T2']):
            if x0[n] < 0:
                bnds.append(troe_all_bnds[coef]['-'])
            elif x0[n] > 0:
                bnds.append(troe_all_bnds[coef]['+'])
            else:   # if doesn't match either of the above then put nan as bounds
                bnds.append([np.nan, np.nan])

        return np.array(bnds).T

    def fit(self):
        T = self.T

        self.p0 = self.x0
        self.p0[-3:] = self.convert_Fcent(self.p0[-3:], 'base2opt')

        #s = np.array([4184, 1.0, 1E-2, 4184, 1.0, 1E-2, *(np.ones((1,4)).flatten()*1E-1)])

        p_bnds = set_arrhenius_bnds(self.p0[0:3], default_arrhenius_coefNames)
        p_bnds = np.concatenate((p_bnds, set_arrhenius_bnds(self.p0[3:6], default_arrhenius_coefNames)), axis=1)
        p_bnds[:,1] = np.log(p_bnds[:,1])   # setting bnds to ln(A), need to do this better
        p_bnds[:,4] = np.log(p_bnds[:,4])   # setting bnds to ln(A), need to do this better

        Fcent_bnds = self.x_bnds(self.p0[-4:])
        Fcent_bnds[-3:] = self.convert_Fcent(Fcent_bnds[-3:])

        self.p_bnds = np.concatenate((p_bnds, Fcent_bnds), axis=1)

        if len(self.p_bnds) > 0:
            self.p0 = np.clip(self.p0, self.p_bnds[0, :], self.p_bnds[1, :])

        self.p0 = self.p0[self.alter_idx]
        self.p_bnds = self.p_bnds[:,self.alter_idx]
        self.s = np.ones_like(self.p0)

        if self.algo['algorithm'] == 'scipy_curve_fit':
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', OptimizeWarning)
                x_fit, _ = curve_fit(self.ln_Troe, T, self.ln_k, p0=self.p0, method='trf', bounds=self.p_bnds, # dogbox
                                        #jac=fit_func_jac, x_scale='jac', max_nfev=len(self.p0)*1000)
                                        jac='2-point', x_scale='jac', max_nfev=len(self.p0)*1000, loss='huber')

        #print('scipy:', x_fit)
        #cmp = np.array([T, Fcent, np.exp(fit_func(T, *x_fit))]).T
        #for entry in cmp:
        #    print(*entry)
        #print('')
        #scipy_fit = np.exp(self.function(T, *x_fit))

        else: # maybe try pygmo with cstrs_self_adaptive or unconstrain or decompose
            p0_opt = np.zeros_like(self.p0)
            self.s = self.calc_s(p0_opt)
            bnds = (self.p_bnds-self.p0)/self.s
            
            '''
            opt = nlopt.opt(nlopt.AUGLAG, len(self.p0))
            #opt = nlopt.opt(self.algo['algorithm'], len(self.p0))

            opt.set_min_objective(self.objective)
            #opt.add_inequality_constraint(self.Fcent_constraint, 0.0)
            #opt.add_inequality_constraint(self.constraints, 1E-8)
            opt.set_maxeval(self.algo['max_eval'])
            #opt.set_maxtime(10)

            opt.set_xtol_rel(self.algo['xtol_rel'])
            opt.set_ftol_rel(self.algo['ftol_rel'])

            opt.set_lower_bounds(bnds[0])
            opt.set_upper_bounds(bnds[1])

            opt.set_initial_step(self.algo['initial_step'])
            #opt.set_population(int(np.rint(10*(len(idx)+1)*10)))
            
            sub_opt = nlopt.opt(self.algo['algorithm'], len(self.p0))
            sub_opt.set_initial_step(self.algo['initial_step'])
            sub_opt.set_xtol_rel(self.algo['xtol_rel'])
            sub_opt.set_ftol_rel(self.algo['ftol_rel'])
            opt.set_local_optimizer(sub_opt)

            x_fit = opt.optimize(p0_opt) # optimize!
            #print('Fcent_constraint: ', self.Fcent_constraint(x_fit))
            #print('Arrhe_constraint: ', self.Arrhenius_constraint(x_fit))

            '''
            class pygmo_objective_fcn:
                def __init__(self, obj_fcn, bnds):
                    self.obj_fcn = obj_fcn
                    self.bnds = bnds

                def fitness(self, x):
                    return [self.obj_fcn(x)]

                def get_bounds(self):
                    return self.bnds

                def gradient(self, x):
                    return pygmo.estimate_gradient_h(lambda x: self.fitness(x), x)

            pop_size = int(np.max([35, 5*(len(p0_opt)+1)]))
            num_gen = int(np.ceil(self.algo['max_eval']/pop_size))

            prob = pygmo.problem(pygmo_objective_fcn(self.objective, bnds))
            pop = pygmo.population(prob, pop_size)
            pop.push_back(x = p0_opt)   # puts initial guess into the initial population

            #F = (0.107 - 0.141)/(1 + (num_gen/225)**7.75)
            #F = 0.2
            #CR = 0.8032*np.exp(-1.165E-3*num_gen)
            #algo = pygmo.algorithm(pygmo.de(gen=num_gen, F=F, CR=CR, variant=6))
            algo = pygmo.algorithm(pygmo.sade(gen=num_gen, variant=6))
            #algo = pygmo.algorithm(pygmo.pso_gen(gen=num_gen))                
            #algo = pygmo.algorithm(pygmo.ipopt())
            #algo = pygmo.algorithm(pygmo.gwo(gen=num_gen))

            pop = algo.evolve(pop)

            x_fit = pop.champion_x

            x_fit = self.set_x_from_opt(x_fit)

        #print('nlopt:', x_fit)
        #cmp = np.array([T, Fcent, self.function(T, *x_fit)]).T
        ##cmp = np.array([T, Fcent, scipy_fit, np.exp(self.function(T, *x_fit))]).T
        #for entry in cmp:
        #    print(*entry)
        #print('')

        # change ln_A to A
        x_fit[1] = np.exp(x_fit[1])
        x_fit[4] = np.exp(x_fit[4])

        #res = {'x': x_fit, 'fval': opt.last_optimum_value(), 'nfev': opt.get_numevals()}
        res = {'x': x_fit, 'fval': pop.champion_f[0], 'nfev': pop.problem.get_fevals()}

        return res

    def set_x_from_opt(self, x):
        self.x[self.alter_idx] = x*self.s + self.p0
        x = np.array(self.x)
        x[-3:] = self.convert_Fcent(x[-3:], 'opt2base')

        for i in [7, 8]:
            if x[i] >= 0:
                if x[i] < 10:
                    x[i] = 1E-30
                elif x[i] > 1E8:
                    x[i] = 1E30

            elif np.isnan(x[i]):    # I don't actually know why it's nan sometimes
                x[i] = 1E-30

        return x

    def convert_Fcent(self, x, conv_type='base2opt'):
        #x = x*self.s + self.p0
        y = np.array(x)
        C = bisymlog_C

        flatten = False
        if y.ndim == 1:
            y = y[np.newaxis, :]
            flatten = True

        if conv_type == 'base2opt': # y = [A, T3, T1, T2]
            y = np.sign(y)*np.log(np.abs(y/C) + 1)

        else:
            y = np.sign(y)*C*(np.exp(np.abs(y)) - 1)

            #A = np.log(y[0])
            #T3, T1 = 1000/y[1], 1000/y[2]
            #T2 = y[3]*100

        if flatten: # if it's 1d it's the guess
            y = y.flatten()

        else:       # if it 2d it's the bnds, they need to be sorted
            y = np.sort(y, axis=0)

        return y

    def jacobian(self, T, *x):  # defunct
        [A, B, C, D] = x
        [A, T3, T1, T2] = self.convert_Fcent(x, 'opt2base') # A, B, C, D = x
        bC = bisymlog_C

        jac = []
        jac.append((np.exp(-T/T1) - np.exp(-T/T3)))                   # dFcent/dA
        jac.append(bC*np.exp(np.abs(B))*(1-A)*T/T3**2*np.exp(-T/T3))  # dFcent/dB
        jac.append(bC*np.exp(np.abs(C))*A*T/T1**2*np.exp(-T/T1))      # dFcent/dC
        jac.append(-bC*np.exp(np.abs(D))/T*np.exp(-T2/T))             # dFcent/dD

        jac = np.vstack(jac).T

        return jac

    def objective(self, x_fit, grad=np.array([]), obj_type='obj_sum', aug_lagrangian=True):
        x = self.set_x_from_opt(x_fit)
        T = self.T
        M = self.M

        resid = ln_Troe(T, M, *x) - self.ln_k
        #resid = self.ln_Troe(T, *x) - self.ln_k
        if obj_type == 'obj_sum':              
            obj_val = penalized_loss_fcn(resid, a=self.loss_alpha, c=self.loss_scale).sum()
        elif obj_type == 'obj':
            obj_val = penalized_loss_fcn(resid, a=self.loss_alpha, c=self.loss_scale)
        elif obj_type == 'resid':
            obj_val = resid

        if aug_lagrangian: # https://arxiv.org/pdf/2106.15044.pdf, https://www.him.uni-bonn.de/fileadmin/him/Section6_HIM_v1.pdf
            lamb = self.penalty_vars['lambda']
            mu = self.penalty_vars['mu']
            rho = self.penalty_vars['rho']

            Ea_0, ln_A_0, n_0 = x[:3]
            Ea_inf, ln_A_inf, n_inf = x[3:6]
            Fcent_coeffs = x[-4:] # A, T3, T1, T2

            con = []

            # Arrhenius constraints
            T_max = T[0,-1]

            ln_k_0 = ln_A_0 + n_0*np.log(T_max) - Ea_0/(Ru*T_max)
            ln_k_inf = ln_A_inf + n_inf*np.log(T_max) - Ea_inf/(Ru*T_max)
            con.append(np.max([0, ln_k_max - ln_k_0]))        # ln_k_0 <= ln_k_max
            con.append(np.max([0, ln_k_max - ln_k_inf]))      # ln_k_0 <= ln_k_max

            # Fcent constraints
            T_con = np.array([self.Tmin, *T[0,:], self.Tmax])
            Fcent = Fcent_calc(T_con, *Fcent_coeffs)
            con.append(np.max([0, np.min(Fcent - self.Fcent_min)]))   # Fcent >= Fcent_min
            con.append(np.max([0, np.min(1.0 - Fcent)]))              # Fcent <= 1.0

            con = np.array(con)

            z = 0.5/rho*(((lamb - rho*con)**2 + 4*rho*mu)**0.5 - (lamb - rho*con))
            penalty = 0.0
            for zi in z:
                if zi != 0.0 and zi > min_pos_system_value:
                    penalty += np.log(zi)

            penalty *= mu

            lamb = self.penalty_vars['lambda'] = lamb + rho*(z-con)

            err_norm = np.linalg.norm((z-con))
            if err_norm > 0.95*mu:
                rho_new = 2*rho
                mu_new = mu
            else:
                rho_new = np.max([rho, np.linalg.norm(lamb)])
                mu_new = 0.1*mu

            if rho_new > max_pos_system_value:
                self.penalty_vars['rho'] = max_pos_system_value
            else:
                self.penalty_vars['rho'] = rho_new

            if mu_new < min_pos_system_value:
                self.penalty_vars['mu'] = min_pos_system_value
            else:
                self.penalty_vars['mu'] = mu_new

            obj_val -= penalty

        #s[:] = np.abs(np.sum(loss*fit_func_jac(T, *x).T, axis=1))
        if grad.size > 0:
            grad[:] = self.objective_gradient(x, resid, numerical_gradient=True)
        #else:
        #    grad = self.objective_gradient(x, resid)

        #self.s = self.calc_s(x_fit, grad)

        #self.opt.set_lower_bounds((self.p_bnds[0] - self.p0)/self.s)
        #self.opt.set_upper_bounds((self.p_bnds[1] - self.p0)/self.s)

        return obj_val
    
    def constraints(self, x_fit, grad=np.array([])):
        x = self.set_x_from_opt(x_fit)
        T = self.T

        Ea_0, ln_A_0, n_0 = x[:3]
        Ea_inf, ln_A_inf, n_inf = x[3:6]
        Fcent_coeffs = x[-4:] # A, T3, T1, T2

        con = []

        # Arrhenius constraints
        T_max = T[0,-1]

        ln_k_0 = ln_arrhenius_k(T_max, Ea_0, ln_A_0, n_0)
        ln_k_inf = ln_arrhenius_k(T_max, Ea_inf, ln_A_inf, n_inf)

        con.append(np.max([0, ln_k_max - ln_k_0]))        # ln_k_0 <= ln_k_max
        con.append(np.max([0, ln_k_max - ln_k_inf]))      # ln_k_0 <= ln_k_max

        # Fcent constraints
        T_con = np.array([self.Tmin, *T[0,:], self.Tmax])
        Fcent = Fcent_calc(T_con, *Fcent_coeffs)
        con.append(np.max([0, np.min(Fcent - self.Fcent_min)]))   # Fcent >= Fcent_min
        con.append(np.max([0, np.min(1.0 - Fcent)]))              # Fcent <= 1.0

        con = np.array(con)

    def objective_gradient(self, x, resid=[], numerical_gradient=False):
        if numerical_gradient:
        #x = (x - self.p0)/self.s
            grad = approx_fprime(x, self.objective, 1E-10)
            
        else:
            if len(resid) == 0:
                resid = self.objective(x, obj_type='resid')

            x = x*self.s + self.p0
            T = self.T
            jac = self.jacobian(T, *x)
            if np.isfinite(jac).all():
                with np.errstate(all='ignore'):
                    grad = np.sum(jac.T*resid, axis=1)*self.s
                    grad[grad == np.inf] = max_pos_system_value
            else:
                grad = np.ones_like(self.p0)*max_pos_system_value

        return grad

    def calc_s(self, x, grad=[]):
        if len(grad) == 0:
            grad = self.objective_gradient(x, numerical_gradient=True)

        y = np.abs(grad)
        if (y < min_pos_system_value).all():
            y = np.ones_like(y)*1E-14
        else:
            y[y < min_pos_system_value] = 10**(OoM(np.min(y[y>=min_pos_system_value])) - 1)  # TODO: MAKE THIS BETTER running into problem when s is zero, this is a janky workaround
        
        s = 1/y
        #s = s/np.min(s)
        #s = s/np.max(s)

        return s

    def Fcent_constraint(self, x_fit, grad=np.array([])):
        def f_fp(T, A, T3, T1, T2, fprime=False, fprime2=False): # dFcent_dT 
            f = T2/T**2*np.exp(-T2/T) - (1-A)/T3*np.exp(-T/T3) - A/T1*np.exp(-T/T1)

            if not fprime and not fprime2:
                return f
            elif fprime and not fprime2:
                fp = T2*(T2 - 2*T)/T**4*np.exp(-T2/T) + (1-A)/T3**2*np.exp(-T/T3) +A/T1**2*np.exp(-T/T1)
                return f, fp

        x = self.set_x_from_opt(x_fit)
        [A, T3, T1, T2] = x[-4:]
        Tmin = self.Tmin
        Tmax = self.Tmax

        try:
            T_deriv_eq_0 = root_scalar(lambda T: f_fp(A, T3, T1, T2), 
                                        x0=(Tmax+Tmin)/4, x1=3*(Tmax+Tmin)/4, method='secant')
            T = np.array([Tmin, T_deriv_eq_0, Tmax])
        except:
            T = np.array([Tmin, Tmax])

        if len(T) == 3:
            print(T)

        Fcent = Fcent_calc(T, A, T3, T1, T2)   #TODO: OVERFLOW WARNING HERE
        min_con = np.max(self.Fcent_min - Fcent)
        max_con = np.max(Fcent - 1.0)
        con = np.max([max_con, min_con])*1E8

        if grad.size > 0:
            grad[:] = self.constraint_gradient(x, numerical_gradient=self.Fcent_constraint)

        return con

    def Arrhenius_constraint(self, x_fit, grad=np.array([])):
        x = self.set_x_from_opt(x_fit)

        T = self.T
        T_max = T[-1]

        Ea_0, ln_A_0, n_0 = x[:3]
        Ea_inf, ln_A_inf, n_inf = x[3:6]

        ln_k_0 = ln_A_0 + n_0*np.log(T_max) - Ea_0/(Ru*T_max)
        ln_k_inf = ln_A_inf + n_inf*np.log(T_max) - Ea_inf/(Ru*T_max)

        ln_k_limit_max = np.max([ln_k_0, ln_k_inf])

        con = ln_k_limit_max - ln_k_max

        if grad.size > 0:
            grad[:] = self.constraint_gradient(x, numerical_gradient=self.Arrhenius_constraint)

        return con

    def constraint_gradient(self, x, const_eval=[], numerical_gradient=None):
        if numerical_gradient is not None:
            grad = approx_fprime(x, numerical_gradient, 1E-10)
            
        else:   # I've not calculated the derivatives wrt coefficients for analytical
            if len(resid) == 0:
                const_eval = self.objective(x)

            T = self.T
            jac = self.jacobian(T, *x)
            if np.isfinite(jac).all():
                with np.errstate(all='ignore'):
                    grad = np.sum(jac.T*const_eval, axis=1)*self.s
                    grad[grad == np.inf] = max_pos_system_value
            else:
                grad = np.ones_like(self.x0)*max_pos_system_value

        return grad


def falloff_parameters_decorator(args_list):
    T, M, ln_k, x0, algorithm_options = args_list
    falloff = falloff_parameters(T, M, ln_k, x0, algorithm_options)
    return falloff.fit()


class Troe:
    def __init__(self, rates, T, P, M, x0=[], coefNames=default_Troe_coefNames, bnds=[], 
                 is_falloff_limit=None, accurate_fit=False, robust=False, mpPool=None):

        self.debug = True

        self.k = rates
        self.ln_k = np.log(rates)
        self.T = T
        self.P = P
        self.M = M

        self.x0 = x0
        if len(self.x0) != 10:
            self.x0[6:] = [1.0,   1E-30,  1E-30,   1500]
        self.x0 = np.array(self.x0)
        self.x = deepcopy(self.x0)
        self.coefNames = np.array(coefNames)

        idx = []
        for n, coefName in enumerate(default_Troe_coefNames):
            if coefName in coefNames:
                idx.append(n)

        self.p0 = self.x0[idx]

        # only valid initial guesses
        self.bnds = np.array(bnds)
        if len(self.bnds) > 0:
            self.p0 = np.clip(self.p0, self.bnds[0, :], self.bnds[1, :])

        self.pool = mpPool

        if robust:
            loss_fcn_param = [1, 1]  # huber-like
        else:
            loss_fcn_param = [2, 1]  # SSE

        self.alter_idx = {'arrhenius': [], 'low_rate': [], 'high_rate': [], 'falloff_parameters': [], 'all': []}
        for n, coefName in enumerate(default_Troe_coefNames):
            if coefName in coefNames:
                self.alter_idx['all'].append(n)
                if coefName.rsplit('_', 1)[0] in default_arrhenius_coefNames:
                    self.alter_idx['arrhenius'].append(n)
                    if '_0' == coefName[-2:]:
                        self.alter_idx['low_rate'].append(n)
                    elif '_inf' == coefName[-4:]:
                        self.alter_idx['high_rate'].append(n)
                else:
                    self.alter_idx['falloff_parameters'].append(n)
        
        idx = [-1]
        is_P_limit = [False, False]    # is low pressure limit, high pressure limit
        for i, arrhenius_type in enumerate(['low_rate', 'high_rate']):
            # set coefNames to be optimized based on GUI
            x_idx = np.array(self.alter_idx[arrhenius_type])
            idx = np.arange(idx[-1], idx[-1] + len(x_idx)) + 1

            if len(idx) < 3 or any(is_falloff_limit[idx]):
                is_P_limit[i] = True

        # scipy_curve_fit, GN_DIRECT_L, GN_DIRECT, GN_DIRECT_NOSCAL GN_CRS2_LM, LN_COBYLA, LN_SBPLX, LD_MMA 
        self.algorithm_options = [{'algorithm': nlopt.GN_DIRECT_L, 'xtol_rel': 1E-5, 'ftol_rel': 1E-5,
                                   'initial_step': 1E-1, 'max_eval': 2500,
                                   'loss_fcn_param': loss_fcn_param, 'is_P_limit': is_P_limit}, 

                                  {'algorithm': nlopt.LN_SBPLX, 'xtol_rel': 1E-5, 'ftol_rel': 1E-5,
                                   'initial_step': 1E-3, 'max_eval': 10000,  
                                   'loss_fcn_param': loss_fcn_param, 'is_P_limit': is_P_limit}]

        if accurate_fit:
            self.algorithm_options[0]['algorithm'] = nlopt.GN_CRS2_LM
            self.algorithm_options[0]['xtol_rel'] = 1E-6
            self.algorithm_options[0]['ftol_rel'] = 1E-6
            self.algorithm_options[0]['max_eval'] = 10000

    def fit(self):
        x = self.x
        start = timer()

        # fit arrhenius expression for each P
        arrhenius_coeffs, T, ln_k = self.fit_arrhenius_rates()

        # change initial guesses if arrhenius rates from before are pressure limits
        if self.algorithm_options[0]['is_P_limit'][0]:
            self.x0[0:3] = arrhenius_coeffs[0]
        
        if self.algorithm_options[0]['is_P_limit'][1]:
            self.x0[3:6] = arrhenius_coeffs[1]

        self.x0[[1,4]] = np.log(self.x0[[1,4]]) # convert to ln_A

        # set T, P, M as arrays
        P = np.unique(self.P)
        P[1:] = np.roll(P[1:], 1)
        T, P = np.meshgrid(T, P)

        M = [] # calculate M
        for i in range(0, np.shape(P)[0]):
            M.append(self.M(T[i,:], P[i,:]))

        M = np.array(M)

        # fit Troe parameters
        x0 = np.tile(self.x0[0:6], [np.shape(troe_falloff_0)[0], 1])
        x0 = np.concatenate([x0, troe_falloff_0], axis=1)
        if np.isnan(self.x0[-4:-1]).all():
            x0 = np.vstack([self.x0, x0[1:,:]])
        else:
            x0 = np.vstack([self.x0, x0])
        
        for n in range(50):
            if self.pool is not None:
                args_list = ((T, M, ln_k, p0, self.algorithm_options[0]) for p0 in x0)
                falloff_output = self.pool.map(falloff_parameters_decorator, args_list)
            else:
                falloff_output = []
                for i, p0 in enumerate(x0):
                    falloff = falloff_parameters(T, M, ln_k, p0, self.algorithm_options[0]) #GN_CRS2_LM LN_SBPLX
                    falloff_output.append(falloff.fit())

            HoF = {'obj_fcn': np.inf, 'coeffs': []}
            for i, res in enumerate(falloff_output):
                if res['fval'] < HoF['obj_fcn']:
                    HoF['obj_fcn'] = res['fval']
                    HoF['coeffs'] = res['x']
                    HoF['i'] = i

            print(n, HoF['obj_fcn'])

        # Run local optimizer
        x = HoF['coeffs']

        str_x = '\t'.join([f'{val:.3e}' for val in x])
        print(f'G {len(x)}\t{HoF["obj_fcn"]:.3e}\t', str_x)

        x0 = deepcopy(x)
        x0[1], x0[4] = np.log(x0[1]), np.log(x0[4])
        
        falloff = falloff_parameters(T, M, ln_k, x0, self.algorithm_options[1])
        res = falloff.fit()

        #use_str = 'using G'
        if res['fval'] < HoF['obj_fcn'] and len(res['x']) == len(x):    # this length check shouldn't be needed
            x = res['x']
            #use_str = 'using L'

        ##print(f'{HoF["obj_fcn"]:.3e}\t{res["fval"]:.3e}')
        #str_x = '\t'.join([f'{val:.3e}' for val in x])
        #print(f'L {len(x)}\t{res["fval"]:.3e}\t', str_x, use_str)

        #if self.debug:
        #    T = self.T
        #    P = self.P
        #    M = self.M
        #    ln_k = self.ln_k

        #    ln_k_0 = np.log(x[1]) + x[2]*np.log(T) - x[0]/(Ru*T)
        #    ln_k_inf = np.log(x[4]) + x[5]*np.log(T) - x[3]/(Ru*T)
        #    ln_Fcent = np.log((1-x[6])*np.exp(-T/x[7]) + x[6]*np.exp(-T/x[8]) + np.exp(-x[9]/T))

        #    cmp = np.array([T, P, M, ln_k, self.ln_Troe(M, ln_k_0, ln_k_inf, ln_Fcent)]).T
        #    for entry in cmp:
        #        print(*entry)
        #    print('')

        return x

    def fit_arrhenius_rates(self):
        T, P, M = self.T, self.P, self.M
        x0, bnds, x = self.x0, self.bnds, self.x
        ln_k, rates = self.ln_k, self.k
        alter_idx = self.alter_idx

        T_arrhenius = np.linspace(np.min(T), np.max(T), 50)

        # Fit HPL and LPL or low and high limits given
        f_ln_k = lambda T, A, n, Ea: list(np.log(A) + n*np.log(T) - Ea/(Ru*T))

        x_temp = deepcopy(x0)
        arrhenius_coeffs = []
        ln_k_Arrhenius = []
        idx = [-1]
        for i, (x0_idx, arrhenius_type) in enumerate(zip([[0, 1, 2], [3, 4, 5]], ['low_rate', 'high_rate'])):
            # set coefNames to be optimized based on GUI
            if self.algorithm_options[0]['is_P_limit'][i]:
                x_idx = np.array(alter_idx[arrhenius_type])
            else:
                x_idx = x0_idx

            idx = np.arange(idx[-1], idx[-1] + len(x_idx)) + 1
            coefNames = [coef.split('_0')[0].split('_inf')[0] for coef in self.coefNames[idx]]

            if len(idx) > 0:
                x_temp[x_idx] = fit_arrhenius(rates[idx], T[idx], x0=x0[x0_idx], coefNames=coefNames, bnds=[bnds[0][idx], bnds[1][idx]])
            
            Ea, A, n = x_temp[x0_idx]
            arrhenius_coeffs.append([Ea, A, n])
            ln_k_Arrhenius.append(f_ln_k(T_arrhenius, A, n, Ea))
        
        # Fit rates in falloff to Arrhenius and compute ln_k
        num_conditions = int(len(rates[idx[-1]+1:])/3)
        for i in range(0, num_conditions):
            idx = np.arange(idx[-1], idx[-1] + 3) + 1
            Ea, A, n = fit_arrhenius(rates[idx], T[idx])
            arrhenius_coeffs.append([Ea, A, n])
            ln_k_Arrhenius.append(f_ln_k(T_arrhenius, A, n, Ea))
        
        return arrhenius_coeffs, T_arrhenius, np.array(ln_k_Arrhenius)

    # accurate fit is for Troe fitting. Significant time savings
def fit_generic(rates, T, P, X, rxnIdx, coefKeys, coefNames, is_falloff_limit, mech, bnds, mpPool=None, accurate_fit=False):
    rxn = mech.gas.reaction(rxnIdx)
    rates = np.array(rates)
    T = np.array(T)
    P = np.array(P)
    coefNames = np.array(coefNames)
    bnds = np.array(bnds).copy()

    if type(rxn) in [ct.ElementaryReaction, ct.ThreeBodyReaction]:
        # set x0 for all parameters
        x0 = [mech.coeffs_bnds[rxnIdx]['rate'][coefName]['resetVal'] for coefName in mech.coeffs_bnds[rxnIdx]['rate']]
        coeffs = fit_arrhenius(rates, T, x0=x0, coefNames=coefNames, bnds=bnds)

        if type(rxn) is ct.ThreeBodyReaction and 'pre_exponential_factor' in coefNames:
            A_idx = np.argwhere(coefNames == 'pre_exponential_factor')[0]
            coeffs[A_idx] = coeffs[A_idx]/mech.M(rxn)
    
    elif type(rxn) in [ct.PlogReaction, ct.FalloffReaction]:
        M = lambda T, P: mech.M(rxn, [T, P, X])

        # get x0 for all parameters
        x0 = []
        for Initial_parameters in mech.coeffs_bnds[rxnIdx].values():
            for coef in Initial_parameters.values():
                x0.append(coef['resetVal'])

        # set coefNames to be optimized
        falloff_coefNames = []
        for key, coefName in zip(coefKeys, coefNames):
            if key['coeffs_bnds'] == 'low_rate':
                falloff_coefNames.append(f'{coefName}_0')
            elif key['coeffs_bnds'] == 'high_rate':
                falloff_coefNames.append(f'{coefName}_inf')

        falloff_coefNames.extend(['A', 'T3', 'T1', 'T2'])
        Troe_parameters = Troe(rates, T, P, M, x0=x0, coefNames=falloff_coefNames, bnds=bnds, 
                               is_falloff_limit=is_falloff_limit, accurate_fit=accurate_fit, mpPool=mpPool)
        coeffs = Troe_parameters.fit()

    return coeffs


def fit_coeffs(rates, T, P, X, rxnIdx, coefKeys, coefNames, is_falloff_limit, bnds, mech, mpPool=None): 
    if len(coefNames) == 0: return # if not coefs being optimized in rxn, return 

    return fit_generic(rates, T, P, X, rxnIdx, coefKeys, coefNames, is_falloff_limit, mech, bnds, mpPool)
    

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
    coefBndsKeys = {'coeffs': [0, 0, 0], 'coeffs_bnds': ['rate', 'rate', 'rate']}
    rxnIdx = 0
    coeffs = fit_coeffs(rates, T, P, X, rxnIdx, coefKeys, coefNames, mech)
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