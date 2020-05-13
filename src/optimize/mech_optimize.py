# This file is part of Frhodo. Copyright © 2020, UChicago Argonne, LLC
# and licensed under BSD-3-Clause. See License.txt in the top-level 
# directory for license and copyright information.

from qtpy.QtCore import QThreadPool, QObject, QRunnable, Signal, Slot
import numpy as np
import cantera as ct
import nlopt
import multiprocessing as mp
from scipy.optimize import minimize_scalar, curve_fit, OptimizeWarning
from scipy.interpolate import interp1d
from scipy import stats
import traceback, sys, re, io, contextlib, warnings
from copy import deepcopy
from timeit import default_timer as timer
import mech_fcns, mech_widget

class Multithread_Optimize:
    def __init__(self, parent):
        self.parent = parent
        
        # Initialize Threads
        parent.optimize_running = False
        # parent.threadpool = QThreadPool()
        # parent.threadpool.setMaxThreadCount(2) # Sets thread count to 1 (1 for gui - this is implicit, 1 for calc)
        # log_txt = 'Multithreading with maximum {:d} threads\n'.format(parent.threadpool.maxThreadCount())
        # parent.log.append(log_txt, alert=False)
        
        # Connect Toolbar Functions
        parent.action_Run.triggered.connect(self.start_threads)
        parent.action_Abort.triggered.connect(self.abort_workers)
        
    def start_threads(self):
        '''
        # rates = [1529339.05689338, 1548270.86688399, 1567437.0352583]
        rates = [1529339.05689338, 1548270.86688399, 1567437.0352583]*np.array([1, 1.001, 1])
        T = [2387.10188629, 2389.48898818, 2391.88086905]
        P = [16136.20900077, 16136.20900077, 16136.20900077]
        X = {'Kr': 0.99, 'C8H8': 0.01}
        
        coefNames = ['activation_energy', 'pre_exponential_factor', 'temperature_exponent']
        rxnIdx = 0
        mech = self.parent.mech
        fit_coeffs(rates, T, P, X, coefNames, rxnIdx, mech)
        print(np.array([2.4442928e+08, 3.4120000e+11, 0.0000000e+00]))
        return
        '''
        parent = self.parent
        parent.path_set.optimized_mech()
        if parent.directory.invalid: return
        if parent.optimize_running: return
        if len(parent.series_viewer.data_table) == 0: return
        if not parent.load_full_series_box.isChecked(): # TODO: May want to remove this limitation in future
            parent.log.append('"Load Full Series Into Memory" must be checked for optimization\n')
            return
        
        # Specify coefficients to be optimized
        self.coef_opt = coef_opt = self._set_coef_opt()
        if not coef_opt: return         # if nothing to optimize, don't!
                
        # Set shocks to be run
        self.shocks2run = []
        for series in parent.series.shock:
            for shock in series:
                # skip not included or exp_data not loaded from experiment
                if not shock['include'] or 'exp_data' in shock['err']: 
                    shock['SIM'] = None
                    continue
                
                # if weight variables aren't set, update
                weight_var = [shock[key] for key in ['weight_max', 'weight_min', 'weight_shift', 
                      'weight_k']]
                if np.isnan(np.hstack(weight_var)).any():
                    presize = np.shape(shock['exp_data'])[0]
                    parent.weight.update(shock=shock)
                    shock['weights'] = parent.series.weights(shock['exp_data'][:,0], shock)
                
                # if reactor temperature and pressure aren't set, update
                if np.isnan([shock['T_reactor'], shock['P_reactor']]).any():
                    parent.series.set('zone', shock['zone'])
                    
                parent.series.rate_bnds(shock)
                
                self.shocks2run.append(shock)
        
        if len(self.shocks2run) == 0: return    # if no shocks to run return
        
        shock_conditions = {'T2': [], 'P2': [], 'T5': [], 'P5': [], 'thermo_mix': []}
        for shock in self.shocks2run:
            for shock_condition in shock_conditions:
                shock_conditions[shock_condition].append(shock[shock_condition])
        
        # Set conditions of rates to be fit for each coefficient
        rxn_coef_opt = self._set_rxn_coef_opt(shock_conditions)
        
        parent.multiprocessing = parent.multiprocessing_box.isChecked()
        
        parent.update_user_settings()
        # parent.set_weights()
        
        parent.abort = False
        parent.optimize_running = True
        
        # Create mechs and duplicate mech variables
        if parent.multiprocessing == True:
            cpu_count = mp.cpu_count()
            # if cpu_count > 1: # leave open processor
                # cpu_count -= 1
            parent.max_processors = np.min([len(self.shocks2run), cpu_count])
            if parent.max_processors == 1:      # if only 1 shock, turn multiprocessing off
                parent.multiprocessing = False
            
            log_str = 'Number of processors: {:d}'.format(parent.max_processors)
            parent.log.append(log_str, alert=False)
        else:
            parent.max_processors = 1
        
        # Pass the function to execute
        self.worker = Worker(parent, self.shocks2run, parent.mech, coef_opt, rxn_coef_opt)
        self.worker.signals.result.connect(self.on_worker_done)
        self.worker.signals.finished.connect(self.thread_complete)
        self.worker.signals.update.connect(self.update)
        self.worker.signals.progress.connect(self.on_worker_progress)
        self.worker.signals.log.connect(parent.log.append)
        self.worker.signals.abort.connect(self.worker.abort)
        
        # Create Progress Bar
        # parent.create_progress_bar()
                
        if not parent.abort:
            s = 'Optimization starting\n\n   Iteration\t\t   Loss Function'
            parent.log.append(s, alert=False)
            parent.threadpool.start(self.worker)
    
    def _set_coef_opt(self):                   
        mech = self.parent.mech
        coef_opt = []
        for rxnIdx in range(mech.gas.n_reactions):      # searches all rxns
            if not mech.rate_bnds[rxnIdx]['opt']: continue        # ignore fixed reactions
            
            # check all coefficients
            for coefIdx, (coefName, coefDict) in enumerate(mech.coeffs_bnds[rxnIdx].items()):
                if coefDict['opt']:
                    coef_opt.append({'rxnIdx': rxnIdx, 'coefIdx': coefIdx, 'coefName': coefName})
        
        return coef_opt                    
    
    def _set_rxn_coef_opt(self, shock_conditions, min_T_range=1000):
        coef_opt = deepcopy(self.coef_opt)
        mech = self.parent.mech
        rxn_coef_opt = []
        for coef in coef_opt:
            if len(rxn_coef_opt) == 0 or coef['rxnIdx'] != rxn_coef_opt[-1]['rxnIdx']:
                rxn_coef_opt.append(coef)
                rxn_coef_opt[-1]['coefIdx'] = [rxn_coef_opt[-1]['coefIdx']]
                rxn_coef_opt[-1]['coefName'] = [rxn_coef_opt[-1]['coefName']]
            else:
                rxn_coef_opt[-1]['coefIdx'].append(coef['coefIdx'])
                rxn_coef_opt[-1]['coefName'].append(coef['coefName'])
        
        T_bnds = np.array([np.min(shock_conditions['T2']), np.max(shock_conditions['T2'])])
        if T_bnds[1] - T_bnds[0] < min_T_range:  # if T_range isn't large enough increase it
            T_mean = np.mean(T_bnds)
            T_bnds = np.array([T_mean-min_T_range/2, T_mean+min_T_range/2])
            # T_bnds = np.ones_like(T_bnds)*np.mean(T_bnds) + np.ones_like(T_bnds)*[-1, 1]*min_T_range/2
        invT_bnds = np.divide(10000, T_bnds)
        P_bnds = [np.min(shock_conditions['P2']), np.max(shock_conditions['P2'])]
        for rxn_coef in rxn_coef_opt:
            n_coef = len(rxn_coef['coefIdx'])
            rxn_coef['invT'] = np.linspace(*invT_bnds, n_coef)
            rxn_coef['T'] = np.divide(10000, rxn_coef['invT'])
            rxn_coef['P'] = np.linspace(*P_bnds, n_coef)
            rxn_coef['X'] = shock_conditions['thermo_mix'][0]   # TODO: IF MIXTURE COMPOSITION FOR DUMMY RATES MATTER CHANGE HERE
                      
        return rxn_coef_opt

    def update(self, result):
        loss_str = '{:.3e}'.format(result['loss']).replace('e+', 'e').replace('e-0', 'e-')
        self.parent.log.append('\t{:s} {:^5d}\t\t\t{:^s}'.format(
            result['type'][0].upper(), result['i'], loss_str), alert=False)
        self.parent.tree.update_coef_rate_from_opt(result['coef_opt'], result['x'])
        
        # if displayed shock isn't in shocks being optimized, calculate the new plot
        if result['ind_var'] is None and result['observable'] is None:
            self.parent.run_single()
        else:       # if displayed shock in list being optimized, show result
            self.parent.plot.signal.update_sim(result['ind_var'][:,0], result['observable'][:,0])
        
        self.parent.plot.opt.update(result['stat_plot'])
    
    def on_worker_progress(self, perc_completed, time_left):
        self.parent.update_progress(perc_completed, time_left)
    
    def thread_complete(self): pass
    
    def on_worker_done(self, result):
        parent = self.parent
        parent.optimize_running = False
        if result is None or len(result) == 0: return
        
        # update mech to optimized one
        if 'local' in result:
            update_mech_coef_opt(parent.mech, self.coef_opt, result['local']['x'])
        else:
            update_mech_coef_opt(parent.mech, self.coef_opt, result['global']['x'])
        
        for opt_type, res in result.items():
            total_shock_eval = (res['nfev']+1)*len(self.shocks2run)
            message = res['message'][:1].lower() + res['message'][1:]
            
            parent.log.append('\n{:s} {:s}'.format(opt_type.capitalize(), message))
            parent.log.append('\telapsed time:\t{:.2f}'.format(res['time']), alert=False)
            parent.log.append('\tloss function:\t{:.3e}'.format(res['fval']), alert=False)
            parent.log.append('\topt iters:\t\t{:.0f}'.format(res['nfev']+1), alert=False)
            parent.log.append('\tshock evals:\t{:.0f}'.format(total_shock_eval), alert=False)
            parent.log.append('\tsuccess:\t\t{:}'.format(res['success']), alert=False)
        
        parent.log.append('\n', alert=False)
        parent.save.chemkin_format(parent.mech.gas, parent.path_set.optimized_mech())
        parent.path_set.mech()  # update mech pulldown choices

    def abort_workers(self):
        if hasattr(self, 'worker'):
            self.worker.signals.abort.emit()
            self.parent.abort = True
            # self.parent.update_progress(100, '00:00:00') # This turns off the progress bar

def initialize_parallel_worker(path, coeffs, coeffs_bnds, rate_bnds):
    global mpMech
    mpMech = mech_fcns.Chemical_Mechanism()
    mech_load_output = mpMech.load_mechanism(path, silent=True)
    mpMech.coeffs = deepcopy(coeffs)
    mpMech.coeffs_bnds = deepcopy(coeffs_bnds)
    mpMech.rate_bnds = deepcopy(rate_bnds)

def fit_coeffs(rates, T, P, X, coefNames, rxnIdx, mech):
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
    
    def jacobian(func, x, h=np.finfo(float).eps): # central finite difference
        def OoM(x):
            x = np.copy(x)
            if not isinstance(x, np.ndarray):
                x = np.array(x)
            x[x==0] = 1                       # if zero, make OoM 0
            return np.floor(np.log10(np.abs(x)))
          
        len_x = len(x)
        h = np.ones((len_x, 1))*h
        h = (h.T * np.power(10, OoM(x))).T                  # scale h to OoM of variable
        x = np.tile(x, (2*len_x,1))
        x[1::2] = x[1::2] + (np.eye(len_x, len_x).T * h).T  # add h on odd rows
        x[::2] = x[::2] - (np.eye(len_x, len_x).T * h).T    # subtract h on even rows
        df = np.empty((len_x, 1))
        for i in range(len_x):
            df[i] = (func(x[2*i+1])[0] - func(x[2*i])[0])/(2*h[i])

        return df.T[0]
    
    def scale_fcn(coefVals, coefNames, rxn, dir='forward'):
        coefVals = np.copy(coefVals)
        if type(rxn) is ct.ElementaryReaction or type(rxn) is ct.ThreeBodyReaction: # NEED TO UPDATE THIS FOR OTHER TYPES OF EXPRESSIONS
            for n, coefVal in enumerate(coefVals):   # updates reaction mechanism for specific reaction
                if coefNames[n] == 'pre_exponential_factor':
                    if dir == 'forward':
                        coefVals[n] = 10**coefVal
                    else:
                        coefVals[n] = np.log10(coefVal)
        return coefVals
    
    if len(coefNames) == 0: return # if not coefs being optimized in rxn, return 
    
    min_neg_system_value = np.finfo(float).min*(1E-20) # Don't push the limits too hard
    min_pos_system_value = np.finfo(float).eps*(1.1)
    max_pos_system_value = np.finfo(float).max*(1E-20)
    
    rxn = mech.gas.reaction(rxnIdx)
    x0 = []
    lower_bnd = []
    upper_bnd = []
    for n, coefName in enumerate(coefNames):
        # if coef['rxnIdx'] != rxnIdx: continue   # ignore reaction not specified
        x0.append(mech.coeffs_bnds[rxnIdx][coefName]['resetVal'])
        if np.isnan(mech.coeffs_bnds[rxnIdx][coefName]['limits']).any():
            if coefName == 'pre_exponential_factor':
                lower_bnd.append(min_pos_system_value)             # A should be positive
            elif coefName == 'activation_energy' and x0[n] > 0:
                lower_bnd.append(0)                                # Ea shouldn't change sign
            else:
                lower_bnd.append(min_neg_system_value)
            
            if coefName == 'activation_energy' and x0[n] < 0:   # Ea shouldn't change sign
                upper_bnd.append(0)
            else:
                upper_bnd.append(max_pos_system_value)
        else:
            lower_bnd.append(mech.coeffs_bnds[rxnIdx][coefName]['limits'][0])
            upper_bnd.append(mech.coeffs_bnds[rxnIdx][coefName]['limits'][1])
        
    x0s = scale_fcn(x0, coefNames, rxn, dir='inverse')
        
    invT = np.divide(10000, T)
    logk = np.log10(rates)
    
    if not isinstance(X, (list, np.ndarray)):   # if only a single composition is given, duplicate
        X = [X]*len(invT)
    
    eqn = lambda invT, *x: fit_rate_eqn(P, X, mech, coefNames, rxnIdx)(invT, *scale_fcn(x, coefNames, rxn))
    s = np.abs(jacobian(lambda x: eqn([np.mean(invT)], *x), x0s, 1E-9))
    s[s==0] = 1E-9  # TODO: MAKE THIS BETTER running into problem when Ea is zero, this is a janky workaround
    # s /= np.max(s)  # to prevent overflow if s_i is > 1 and unbounded
    scaled_eqn = lambda invT, *x: eqn(invT, *(x/s + x0s))
    lower_bnd = (scale_fcn(lower_bnd, coefNames, rxn, dir='inverse') - x0s)*s
    upper_bnd = (scale_fcn(upper_bnd, coefNames, rxn, dir='inverse') - x0s)*s
    p0 = np.zeros_like(x0s)
      
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', OptimizeWarning)
        try:
            popt, _ = curve_fit(scaled_eqn, invT, logk, p0=p0, bounds=[lower_bnd, upper_bnd],
                            method='dogbox')
        except:
            return
    
    coeffs = scale_fcn(popt/s + x0s, coefNames, rxn)
    # print(coeffs)
    # print('')
    
    # T_test = np.linspace(np.min(T)*0.999, np.max(T)*1.001, 50)
    # rate_test = []
    # for T_t in T_test:
        # mech.set_TPX(T_t, P[0], X[0])
        # rate_test.append(mech.gas.forward_rate_constants[rxnIdx])
    
    # import matplotlib.pyplot as plt     
    # plt.plot(np.divide(10000, T), np.log10(rates), 'o')
    # plt.plot(np.divide(10000, T_test), np.log10(rate_test))
    # plt.show()
    
    return coeffs

def outlier(res, a=2, c=1, weights=[], iterate2convergence=True):
    def diff(res_outlier):
        if len(res_outlier) < 2: 
            return 1
        else:
            return np.diff(res_outlier)[0]
    
    if a != 2: # define outlier with 1.5 IQR rule
        trunc_res = np.abs(res.copy())
        percentile = 25
        if len(weights) == len(res):
            trunc_res = trunc_res[weights > 0.95*np.max(weights)]  # This computes the outlier threshold based on weights >= 0.95
        
        if iterate2convergence:
            res_outlier = []
            # while diff(res_outlier) != 0:   # iterate until res_outlier is the same as prior iteration
            for n in range(25): # maximum number of iterations
                if diff(res_outlier) == 0:   # iterate until res_outlier is the same as prior iteration
                    break
                    
                if len(res_outlier) > 0:
                    trunc_res = trunc_res[trunc_res < res_outlier[-1]] 
                    
                q1, q3 = np.nanpercentile(trunc_res, percentile), np.nanpercentile(trunc_res, 100-percentile)
                iqr = q3 - q1       # interquartile range      
                
                if len(res_outlier) == 2:
                    del res_outlier[0]
                
                res_outlier.append(q3 + iqr*1.5)
            
            res_outlier = res_outlier[-1]
        
        else:
            q1, q3 = np.nanpercentile(trunc_res, percentile), np.nanpercentile(trunc_res, 100-percentile)
            iqr = q3 - q1       # interquartile range      
            res_outlier = q3 + iqr*1.5
        
    else:
        res_outlier = 1
        
    return c*res_outlier
    
def generalized_loss_fcn(res, a=2, c=1, weights=[]):    # defaults to L2 loss
    x_c_2 = np.power(res/c, 2)
    if a == 2:
        loss = 0.5*x_c_2
    elif a == 0:
        loss = np.log(0.5*x_c_2+1)
    elif a <= -1000:  # supposed to be negative infinity
        loss = 1 - np.exp(-0.5*x_c_2)
    else:
        loss = np.abs(a-2)/a*(np.power(x_c_2/np.abs(a-2) + 1, a/2) - 1)
        
    if len(weights) == len(loss):
        loss = np.multiply(loss, weights)
        
    return loss*np.abs(c)   # multiplying by c is not necessary, but makes order appropriate

def update_mech_coef_opt(mech, coef_opt, x):
    mech_changed = False
    for i, idxDict in enumerate(coef_opt):
        rxnIdx, coefName = idxDict['rxnIdx'], idxDict['coefName']
        if mech.coeffs[rxnIdx][coefName] != x[i]:       # limits mech changes. Should increase speed a little
            mech_changed = True
            mech.coeffs[rxnIdx][coefName] = x[i]
    
    if mech_changed:
        mech.modify_reactions(mech.coeffs)  # Update mechanism with new coefficients
  
def calculate_residuals(args_list):   
    def calc_exp_bounds(t_sim, t_exp):
        t_bounds = [np.max([t_sim[0], t_exp[0]])]       # Largest initial time in SIM and Exp
        t_bounds.append(np.min([t_sim[-1], t_exp[-1]])) # Smallest final time in SIM and Exp
        # Values within t_bounds
        exp_bounds = np.where(np.logical_and((t_exp >= t_bounds[0]),(t_exp <= t_bounds[1])))[0]
        
        return exp_bounds
    
    def time_adjust_func(t_offset, t_adjust, t_sim, obs_sim, t_exp, obs_exp, weights, verbose=False):
        t_sim_shifted = t_sim + t_offset + t_adjust

        # Compare SIM Density Grad vs. Experimental
        exp_bounds = calc_exp_bounds(t_sim_shifted, t_exp)
        t_exp, obs_exp, weights = t_exp[exp_bounds], obs_exp[exp_bounds], weights[exp_bounds]
        
        f_interp = interp1d(t_sim_shifted.flatten(), obs_sim.flatten(), kind = 'cubic')
        obs_sim_interp = f_interp(t_exp)
        
        resid = np.subtract(obs_exp, obs_sim_interp)
        resid_outlier = outlier(resid, a=var['loss_alpha'], c=var['loss_c'], 
                                weights=weights)
        
        if verbose:
            output = {'resid': resid, 'resid_outlier': resid_outlier,
                      'weights': weights,
                      'obs_sim_interp': obs_sim_interp}
            
            return output
        else:   # needs to return single value for optimization
            return generalized_loss_fcn(resid, a=var['loss_alpha'], c=resid_outlier, 
                                        weights=weights).sum()
    
    def calc_density(x, data, dim=1):
        stdev = np.std(data)
        A = np.min([np.std(data), stats.iqr(data)/1.34])/stdev  # bandwidth is multiplied by std of sample
        bw = 0.9*A*len(data)**(-1./(dim+4))

        return stats.gaussian_kde(data, bw_method=bw)(x)
        
    def OoM(x):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        x[x==0] = 1                       # if zero, make OoM 0
        return np.floor(np.log10(np.abs(x)))
    
    var, coef_opt, x, shock = args_list
    mech = mpMech
    
    # Optimization Begins, update mechanism
    update_mech_coef_opt(mech, coef_opt, x)
    
    T_reac, P_reac, mix = shock['T_reactor'], shock['P_reactor'], shock['thermo_mix']
    
    SIM_kwargs = {'u_reac': shock['u2'], 'rho1': shock['rho1'], 'observable': shock['observable'], 
                  't_lab_save': None, 'sim_int_f': var['sim_interp_factor'], 
                  'ODE_solver': var['ode_solver'], 'rtol': var['ode_rtol'], 'atol': var['ode_atol']}
    
    if '0d Reactor' in var['name']:
        SIM_kwargs['solve_energy'] = var['solve_energy']
        SIM_kwargs['frozen_comp'] = var['frozen_comp']
    
    SIM, verbose = mech.run(var['name'], var['t_end'], T_reac, P_reac, mix, **SIM_kwargs)
        
    if SIM.success:
        shock['SIM'] = SIM
    else:
        shock['SIM'] = None
    
    ind_var, obs = SIM.independent_var[:,None], SIM.observable[:,None]
    
    weights = shock['norm_weights_trim']
    obs_exp = shock['exp_data_trim']
    
    if not np.any(var['t_unc']):
        t_unc = 0
    else:
        t_unc_OoM = np.mean(OoM(var['t_unc']))  # Do at higher level? (computationally efficient)
        time_adj_decorator = lambda t_adjust: time_adjust_func(shock['time_offset'], t_adjust*10**t_unc_OoM, 
                ind_var, obs, obs_exp[:,0], obs_exp[:,1], weights)
        
        res = minimize_scalar(time_adj_decorator, bounds=var['t_unc']/10**t_unc_OoM, method='bounded')
        t_unc = res.x*10**t_unc_OoM
    
    output = time_adjust_func(shock['time_offset'], t_unc, ind_var, obs, 
                obs_exp[:,0], obs_exp[:,1], weights, verbose=True)  
    
    output['shock'] = shock
    
    plot_stats = True
    if plot_stats:
        x = np.linspace(output['resid'].min(), output['resid'].max(), 300)
        density = calc_density(x, output['resid'], dim=1)   #kernel density estimation
        output['KDE'] = np.column_stack((x, density))

    return output

    
# Using optimization vs least squares curve fit because y_range's change if time_offset != 0
class Fit_Fun:
    def __init__(self, input_dict):
        self.parent = input_dict['parent']
        self.shocks2run = input_dict['shocks2run']
        self.data = self.parent.series.shock
        self.coef_opt = input_dict['coef_opt']
        self.rxn_coef_opt = input_dict['rxn_coef_opt']
        self.x0 = input_dict['x0']
        self.mech = input_dict['mech']
        self.var = self.parent.var
        self.t_unc = (-self.var['time_unc'], self.var['time_unc'])
        
        self.opt_type = 'local' # this is updated outside of the class
        
        self.loss_alpha = self.parent.optimization_settings.get('loss', 'alpha')
        self.loss_c = self.parent.optimization_settings.get('loss', 'c')
        
        if 'multiprocessing' in input_dict:
            self.multiprocessing = input_dict['multiprocessing']
        
        if 'pool' in input_dict:
            self.pool = input_dict['pool']
        else:
            self.multiprocessing = False
        
        self.signals = input_dict['signals']
        
        self.i = 0        
        self.__abort = False
    
    def __call__(self, s, optimizing=True):
        def append_output(output_dict, calc_resid_output):
            for key in calc_resid_output:
                if key not in output_dict:
                    output_dict[key] = []
                    
                output_dict[key].append(calc_resid_output[key])
            
            return output_dict
        
        if self.__abort: 
            raise Exception('Optimization terminated by user')
            self.signals.log.emit('\nOptimization aborted')
            return
        
        # Convert to mech values
        x = self.fit_all_coeffs(np.exp(s*self.x0))
        if x is None: 
            return np.inf
        
        # Run Simulations
        output_dict = {}
        
        var_dict = {key: val for key, val in self.var['reactor'].items()}
        var_dict['t_unc'] = self.t_unc
        var_dict.update({'loss_alpha': 2, 'loss_c': 1}) # loss function here is for finding t_unc, mse seems to work best.
        # var_dict.update({'loss_alpha': self.loss_alpha, 'loss_c': self.loss_c})
        
        if self.multiprocessing:
            args_list = ((var_dict, self.coef_opt, x, shock) for shock in self.shocks2run)
            calc_resid_outputs = self.pool.map(calculate_residuals, args_list)
            for calc_resid_output, shock in zip(calc_resid_outputs, self.shocks2run):
                shock['SIM'] = calc_resid_output['shock']['SIM']
                append_output(output_dict, calc_resid_output)

        else:
            global mpMech
            mpMech = self.mech
            
            for shock in self.shocks2run:
                args_list = (var_dict, self.coef_opt, x, shock)
                calc_resid_output = calculate_residuals(args_list)
                shock['SIM'] = calc_resid_output['shock']['SIM']
                append_output(output_dict, calc_resid_output)
        
        allResid = np.concatenate(output_dict['resid'], axis=0)
        weights = np.concatenate(output_dict['weights'], axis=0)
        resid_outlier = outlier(allResid, a=self.loss_alpha, c=self.loss_c, 
                                weights=weights)
        total_loss = generalized_loss_fcn(allResid, a=self.loss_alpha, c=resid_outlier, 
                                          weights=weights).sum()    
        # For updating
        self.i += 1
        if not optimizing or self.i % 1 == 0:#5 == 0: # updates plot every 5
            if total_loss == 0:
                total_loss = np.inf
                
            shock = self.parent.display_shock
            if shock['include']:
                ind_var, observable = shock['SIM'].independent_var[:,None], shock['SIM'].observable[:,None]
            else:
                ind_var, observable = None, None
            
            stat_plot = {'shocks2run': self.shocks2run, 'resid': output_dict['resid'], 
                        'resid_outlier': resid_outlier, 'weights': output_dict['weights']}
            
            if 'KDE' in output_dict:
                stat_plot['KDE'] = output_dict['KDE']
                
                stat_plot['fit_result'] = fitres = stats.gennorm.fit(allResid)
                stat_plot['QQ'] = []
                for resid in stat_plot['resid']:
                    QQ = stats.probplot(resid, sparams=fitres, dist=stats.gennorm, fit=False)
                    QQ = np.array(QQ).T
                    stat_plot['QQ'].append(QQ)
            
            update = {'type': self.opt_type, 'i': self.i, 
                      'loss': total_loss, 'stat_plot': stat_plot, 
                      'x': x, 'coef_opt': self.coef_opt, 'ind_var': ind_var, 'observable': observable}
            
            self.signals.update.emit(update)
                  
        if optimizing:
            return total_loss
        else:
            return total_loss, x, output_dict['shock']
            
    def fit_all_coeffs(self, all_rates):      
        coeffs = []
        i = 0
        for rxn_coef in self.rxn_coef_opt:
            rxnIdx = rxn_coef['rxnIdx']
            T, P, X = rxn_coef['T'], rxn_coef['P'], rxn_coef['X']
            rxn_rates = all_rates[i:i+len(T)]
            if len(coeffs) == 0:
                coeffs = fit_coeffs(rxn_rates, T, P, X, rxn_coef['coefName'], rxnIdx, self.mech)
                if coeffs is None:
                    return
            else:
                coeffs_append = fit_coeffs(rxn_rates, T, P, X, rxn_coef['coefName'], rxnIdx, self.mech)
                if coeffs_append is None:
                    return
                coeffs = np.append(coeffs, coeffs_append)
            
            i += len(T)
        
        return coeffs
    
class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.
    Supported signals are:
    finished
        No data
    error
        `tuple` (exctype, value, traceback.format_exc() )
    result
        `object` best returned from processing
    update
        `str` returns 'object' containing current best
    progress
        'float' returns % complete and estimated time left in s
    log
        'str' output to log
    abort
        No data
    '''
    finished = Signal()
    error = Signal(tuple)
    result = Signal(object)
    update = Signal(object)
    progress = Signal(int, str)
    log = Signal(str)
    abort = Signal()

pos_msg = ['Optimization terminated successfully.', 'Optimization terminated: Stop Value was reached.',
           'Optimization terminated: Function tolerance was reached.',
           'Optimization terminated: X tolerance was reached.',
           'Optimization terminated: Max number of evaluations was reached.',
           'Optimization terminated: Max time was reached.']
neg_msg = ['Optimization failed', 'Optimization failed: Invalid arguments given',
           'Optimization failed: Out of memory', 'Optimization failed: Roundoff errors limited progress',
           'Optimization failed: Forced termination']

class Worker(QRunnable):
    '''
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and 
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    It is computationally efficient to limit the amount of unnecessary information sent to the GUI
    
    '''

    def __init__(self, parent, shocks2run, mech, coef_opt, rxn_coef_opt, *args, **kwargs):
        super(Worker, self).__init__()
        # Store constructor arguments (re-used for processing)
        self.parent = parent
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        self.__abort = False
        self.err = False
        
        self.shocks2run = shocks2run
        self.coef_opt = coef_opt
        self.rxn_coef_opt = rxn_coef_opt
        self.mech = mech
        self._initialize()
    
    def _initialize(self):
        def rates():
            output = []
            for rxn_coef in self.rxn_coef_opt:
                for T, P in zip(rxn_coef['T'], rxn_coef['P']):
                    mech.set_TPX(T, P)
                    output.append(mech.gas.forward_rate_constants[rxn_coef['rxnIdx']])
            
            return np.log(output)
        
        mech = self.mech
        
        # reset mechanism
        initial_mech = deepcopy(mech.coeffs)
        for rxnIdx in range(mech.gas.n_reactions):
            for coefName in mech.coeffs[rxnIdx].keys():
                resetVal = mech.coeffs_bnds[rxnIdx][coefName]['resetVal']
                mech.coeffs[rxnIdx][coefName] = resetVal 
        
        mech.modify_reactions(mech.coeffs)
        
        # Calculate x0
        self.x0 = rates()
        
        # Determine bounds
        lb = []
        ub = []
        i = 0
        for rxn_coef in self.rxn_coef_opt:
            rxnIdx = rxn_coef['rxnIdx']
            rate_bnds_val = mech.rate_bnds[rxnIdx]['value']
            rate_bnds_type = mech.rate_bnds[rxnIdx]['type']
            for T, P in zip(rxn_coef['T'], rxn_coef['P']):
                mech.set_TPX(T, P)
                rate = self.mech.gas.forward_rate_constants[rxnIdx]
                bnds = mech_widget.uncertainty_fcn(rate, rate_bnds_val, rate_bnds_type)
                bnds = np.sort(np.log(bnds)/self.x0[i])  # operate on ln and scale
                lb.append(bnds[0])
                ub.append(bnds[1])
                
                i += 1
        
        # Calculate initial scalers
        mech.modify_reactions(initial_mech)
        self.s = np.divide(rates(), self.x0)
        
        # Correct initial guesses if outside bounds
        np.putmask(self.s, self.s < lb, lb)
        np.putmask(self.s, self.s > ub, ub)
        
        # Set opt option variables
        self.bnds = {'lower': np.array(lb), 'upper': np.array(ub)}       
    
    def trim_shocks(self): # trim shocks from zero weighted data
        for n, shock in enumerate(self.shocks2run):
            weights = shock['normalized_weights']
            
            exp_bounds = np.nonzero(weights)[0]
            shock['norm_weights_trim'] = weights[exp_bounds]
            shock['exp_data_trim'] = shock['exp_data'][exp_bounds,:]
    
    def optimize_coeffs(self):
        parent = self.parent
        pool = mp.Pool(processes=parent.max_processors,
                       initializer=initialize_parallel_worker,
                       initargs=(parent.path, parent.mech.coeffs, parent.mech.coeffs_bnds, 
                       parent.mech.rate_bnds,))
        
        self.trim_shocks()  # trim shock data from zero weighted data
        
        input_dict = {'parent': parent, 'pool': pool, 'shocks2run': self.shocks2run,
                      'coef_opt': self.coef_opt, 'rxn_coef_opt': self.rxn_coef_opt, 
                      'x0': self.x0, 'mech': self.mech, 
                      'multiprocessing': parent.multiprocessing, 
                      'signals': self.signals}
           
        Scaled_Fit_Fun = Fit_Fun(input_dict)
           
        def eval_fun(s, grad):            
            if self.__abort:
                raise Exception('Optimization terminated by user')
                self.signals.log.emit('\nOptimization aborted')
                # self.signals.result.emit(hof[0])
            else:
                return Scaled_Fit_Fun(s)

        try:
            opt_options = self.parent.optimization_settings.settings  

            s = self.s
            res = {}
            for n, opt_type in enumerate(['global', 'local']):
                timer_start = timer()
                Scaled_Fit_Fun.i = 0                     # reset iteration counter
                Scaled_Fit_Fun.opt_type = opt_type       # inform about optimization type
                options = opt_options[opt_type]
                if not options['run']: continue
                
                opt = nlopt.opt(options['algorithm'], np.size(self.x0))
                opt.set_min_objective(eval_fun)
                opt.set_xtol_rel(options['xtol_rel'])
                opt.set_ftol_rel(options['ftol_rel'])
                opt.set_lower_bounds(self.bnds['lower'])
                opt.set_upper_bounds(self.bnds['upper'])
                
                initial_step = (self.bnds['upper'] - self.bnds['lower'])*options['initial_step'] 
                np.putmask(initial_step, s < 1, -initial_step)  # first step in direction of more variable space
                opt.set_initial_step(initial_step)
                
                if options['algorithm'] is nlopt.GN_MLSL_LDS:   # if using multistart algorithm as global, set subopt
                    sub_opt = nlopt.opt(opt_options['local']['algorithm'], np.size(self.x0))
                    sub_opt.set_initial_step(initial_step)
                    sub_opt.set_xtol_rel(options['xtol_rel'])
                    sub_opt.set_ftol_rel(options['ftol_rel'])
                    opt.set_local_optimizer(sub_opt)
                
                s = opt.optimize(s) # optimize!
                
                loss, x, shock_output = Scaled_Fit_Fun(s, optimizing=False)
            
                if nlopt.SUCCESS > 0: 
                    success = True
                    msg = pos_msg[nlopt.SUCCESS-1]
                else:
                    success = False
                    msg = neg_msg[nlopt.SUCCESS-1]
                
                # opt.last_optimum_value() is the same as optimal loss
                res[opt_type] = {'coef_opt': self.coef_opt, 'x': x, 'shock': shock_output,
                                 'fval': loss, 'nfev': opt.get_numevals(),
                                 'success': success, 'message': msg, 'time': timer() - timer_start}
                
                if options['algorithm'] is nlopt.GN_MLSL_LDS:   # if using multistart algorithm, break upon finishing loop
                    break
                        
        except Exception as e:
            res = None
            if 'Optimization terminated by user' in str(e):
                self.signals.log.emit('\nOptimization aborted')
            else:
                self.err = True
                self.signals.log.emit('\n{:s}'.format(str(e)))
            
        pool.close()
        return res
        
        
        '''
        stdout = io.StringIO()
        stderr = io.StringIO()
        
        with contextlib.redirect_stderr(stderr):    
            with contextlib.redirect_stdout(stdout):
                try:
                    res = minimize(eval_fun, coef_norm, method='COBYLA')
                        # options={'rhobeg': 1e-02,})
                except Exception as e:
                    if 'Optimization terminated by user' in str(e):
                        self.signals.log.emit('\nOptimization aborted')
                        # self.signals.result.emit(hof[0])
                        return
                    else:
                        self.err = True
                        self.signals.log.emit('\n{:s}'.format(e))
        
        if self.err:
            out = stdout.getvalue()
            err = stderr.getvalue().replace('INFO:root:', 'Warning: ')
                
            for log_str in [out, err]:
                if log_str != '':
                    self.signals.log.append(log_str)  # Append output
        '''
        
    @Slot()
    def run(self):
        try:
            res = self.optimize_coeffs()
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            # self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(res)  # Return the result of the processing
        finally:
            pass
            # self.signals.finished.emit()  # Done
              
    def abort(self):
        self.__abort = True
        if hasattr(self, 'eval_fun'):
            self.eval_fun.__abort = True