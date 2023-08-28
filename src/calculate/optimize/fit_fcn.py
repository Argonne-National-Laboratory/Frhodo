# This file is part of Frhodo. Copyright © 2020, UChicago Argonne, LLC
# and licensed under BSD-3-Clause. See License.txt in the top-level 
# directory for license and copyright information.

import io, contextlib
import numpy as np
import nlopt
from scipy.optimize import minimize_scalar
from scipy.interpolate import CubicSpline
from scipy import stats
from copy import deepcopy

from timeit import default_timer as timer

from calculate.mech_fcns import Chemical_Mechanism
from calculate.convert_units import OoM, Bisymlog
from calculate.optimize.adaptive_loss import adaptive_weights, weighted_quantile
from calculate.optimize.fit_coeffs import fit_coeffs
from calculate.optimize.CheKiPEUQ_from_Frhodo import CheKiPEUQ_Frhodo_interface

mpMech = {}
def initialize_parallel_worker(mech_dict, species_dict, coeffs, coeffs_bnds, rate_bnds):
    mpMech['obj'] = mech = Chemical_Mechanism()

    # hide mechanism loading problems because they will already have been seen
    with contextlib.redirect_stderr(io.StringIO()):
        with contextlib.redirect_stdout(io.StringIO()):
            mech.set_mechanism(mech_dict, species_dict)    # load mechanism from yaml text in memory

    mech.coeffs = deepcopy(coeffs)
    mech.coeffs_bnds = deepcopy(coeffs_bnds)
    mech.rate_bnds = deepcopy(rate_bnds)

def rescale_loss_fcn(x, loss, x_outlier=None, weights=[]):
    x = x.copy()
    weights = weights.copy()

    if x_outlier is not None:
        trimmed_indices = np.argwhere(abs(x) < x_outlier)
        x = x[trimmed_indices]
        loss_trimmed = loss[trimmed_indices]
        weights = weights[trimmed_indices]
    else:
        loss_trimmed = loss

    if len(weights) == len(x):
        x_q1, x_q3 = weighted_quantile(x, np.array([0.0, 1.0]), weights=weights)
        loss_q1, loss_q3 = weighted_quantile(loss_trimmed, np.array([0.0, 1.0]), weights=weights)

    else:
        x_q1, x_q3 = x.min(), x.max()
        loss_q1, loss_q3 = loss_trimmed.min(), loss_trimmed.max()
    
    if x_q1 != x_q3 and loss_q1 != loss_q3: # prevent divide by zero if values end up the same
        loss_scaled = (x_q3 - x_q1)/(loss_q3 - loss_q1)*(loss - loss_q1) + x_q1

    else:
        loss_scaled = loss

    return loss_scaled

def update_mech_coef_opt(mech, coef_opt, x):
    mech_changed = False
    for i, idxDict in enumerate(coef_opt):
        rxnIdx, coefName = idxDict['rxnIdx'], idxDict['coefName']
        coeffs_key = idxDict['key']['coeffs']
        if mech.coeffs[rxnIdx][coeffs_key][coefName] != x[i]:       # limits mech changes. Should increase speed a little
            if type(mech.coeffs[rxnIdx][coeffs_key]) is tuple:      # don't know why but sometimes reverts to tuple
                    mech.coeffs[rxnIdx][coeffs_key] = list(mech.coeffs[rxnIdx][coeffs_key])
            
            mech_changed = True
            mech.coeffs[rxnIdx][coeffs_key][coefName] = x[i]

    if mech_changed:
        mech.modify_reactions(mech.coeffs)  # Update mechanism with new coefficients

def calculate_residuals(args_list):                                                                                                                                                                 
    def resid_func(t_offset, t_adjust, t_sim, obs_sim, t_exp, obs_exp, weights, obs_bounds=[],
                         loss_alpha=2, loss_c=1, loss_penalty=True, scale='Linear', 
                         bisymlog_scaling_factor=1.0, DoF=1, opt_type='Residual', verbose=False):

        def calc_exp_bounds(t_sim, t_exp):
            t_bounds = [max([t_sim[0], t_exp[0]])]       # Largest initial time in SIM and Exp
            t_bounds.append(min([t_sim[-1], t_exp[-1]])) # Smallest final time in SIM and Exp
            # Values within t_bounds
            exp_bounds = np.where(np.logical_and((t_exp >= t_bounds[0]),(t_exp <= t_bounds[1])))[0]
        
            return exp_bounds

        # Compare SIM Density Grad vs. Experimental
        t_sim_shifted = t_sim + t_offset + t_adjust
        exp_bounds = calc_exp_bounds(t_sim_shifted, t_exp)
        t_exp, obs_exp, weights = t_exp[exp_bounds], obs_exp[exp_bounds], weights[exp_bounds]
        if opt_type == 'Bayesian':
            obs_bounds = obs_bounds[exp_bounds]
        
        f_interp = CubicSpline(t_sim.flatten(), obs_sim.flatten())
        t_exp_shifted = t_exp - t_offset - t_adjust
        obs_sim_interp = f_interp(t_exp_shifted)
        
        if scale == 'Linear':
            resid = np.subtract(obs_exp, obs_sim_interp)                                                     
                                                    
        elif scale == 'Log':
            ind = np.argwhere(((obs_exp!=0.0)&(obs_sim_interp!=0.0)))
            exp_bounds = exp_bounds[ind]
            weights = weights[ind].flatten()
            
            m = np.ones_like(obs_exp[ind])
            i_g = obs_exp[ind] >= obs_sim_interp[ind]
            i_l = obs_exp[ind] < obs_sim_interp[ind]
            m[i_g] = np.divide(obs_exp[ind][i_g], obs_sim_interp[ind][i_g])
            m[i_l] = np.divide(obs_sim_interp[ind][i_l], obs_exp[ind][i_l])
            resid = np.log10(np.abs(m)).flatten()
            if verbose and opt_type == 'Bayesian':
                obs_exp = np.log10(np.abs(obs_exp[ind])).squeeze() # squeeze to remove extra dim
                obs_sim_interp = np.log10(np.abs(obs_sim_interp[ind])).squeeze()
                obs_bounds = np.log10(np.abs(obs_bounds[ind])).squeeze()   

        elif scale == 'Bisymlog':
            bisymlog = Bisymlog(C=None, scaling_factor=bisymlog_scaling_factor)
            bisymlog.set_C_heuristically(obs_exp)
            obs_exp_bisymlog = bisymlog.transform(obs_exp)
            obs_sim_interp_bisymlog = bisymlog.transform(obs_sim_interp)
            resid = np.subtract(obs_exp_bisymlog, obs_sim_interp_bisymlog) 
            if verbose and opt_type == 'Bayesian':
                obs_exp = obs_exp_bisymlog
                obs_sim_interp = obs_sim_interp_bisymlog
                obs_bounds = bisymlog.transform(obs_bounds)  # THIS NEEDS TO BE CHECKED
        
        loss_weights, C, alpha = adaptive_weights(resid, weights, C_scalar=loss_c, alpha=loss_alpha)
        agg_weights = weights*loss_weights

        loss_sqr = agg_weights*resid**2
        wgt_sum = weights.sum()
        N = wgt_sum - DoF
        if N <= 0:
            N = wgt_sum
        stderr_sqr = loss_sqr.sum()*wgt_sum/N
        chi_sqr = loss_sqr/stderr_sqr
        std_resid = chi_sqr**(0.5)
        #loss_scalar = (chi_sqr*weights).sum()
        loss_scalar = std_resid.sum()
        #loss_scalar = np.average(std_resid, weights=weights)
        #loss_scalar = weighted_quantile(std_resid, 0.5, weights=weights)    # median value
                                                  
        if verbose:                                                                                                           
            output = {'chi_sqr': chi_sqr, 'resid': resid, 'resid_outlier': C, 'loss': loss_scalar, 
                      'weights': loss_weights, 'aggregate_weights': weights*loss_weights, 
                      'obs_sim_interp': obs_sim_interp, 'obs_exp': obs_exp, 'obs_bounds': obs_bounds}

            return output
                                                                                                                            
        else:   # needs to return single value for optimization
            return loss_scalar
    
    def calc_density(x, data, dim=1):
        stdev = np.std(data)
        [q1, q3] = weighted_quantile(data, np.array([0.25, 0.75]))
        iqr = q3 - q1       # interquartile range   
        A = np.min([stdev, iqr/1.34])/stdev  # bandwidth is multiplied by std of sample
        bw = 0.9*A*len(data)**(-1./(dim+4))

        return stats.gaussian_kde(data, bw_method=bw)(x)
                                                                                                                                                                                                                                                                                                
    var, coef_opt, x, shock = args_list
    mech = mpMech['obj']
                                             
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
    ind_var, obs_sim = SIM.independent_var[:,None], SIM.observable[:,None]
    
    weights = shock['weights_trim']
    obs_exp = shock['exp_data_trim']
    obs_bounds = []
    if var['obj_fcn_type'] == 'Bayesian':
        obs_bounds = shock['abs_uncertainties_trim']
    
    if not np.any(var['t_unc']):
        t_unc = 0
    else:
        t_unc_OoM = np.mean(OoM(var['t_unc']))  # Do at higher level in code? (computationally efficient)
        # calculate time adjust with mse (loss_alpha = 2, loss_c =1)                                                                         
        time_adj_func = lambda t_adjust: resid_func(shock['opt_time_offset'], t_adjust*10**t_unc_OoM, 
                ind_var, obs_sim, obs_exp[:,0], obs_exp[:,1], weights, obs_bounds, scale=var['scale'], 
                bisymlog_scaling_factor= var['bisymlog_scaling_factor'], DoF=len(coef_opt), 
                opt_type=var['obj_fcn_type'])
        
        res = minimize_scalar(time_adj_func, bounds=var['t_unc']/10**t_unc_OoM, method='bounded')
        t_unc = res.x*10**t_unc_OoM

    # calculate loss shape function (alpha) if it is set to adaptive
    loss_alpha = var['loss_alpha']
    if loss_alpha == 3.0:
        loss_alpha_fcn = lambda alpha: resid_func(shock['opt_time_offset'], t_unc, 
                ind_var, obs_sim, obs_exp[:,0], obs_exp[:,1], weights, obs_bounds, 
                loss_alpha=alpha, loss_c=var['loss_c'], loss_penalty=True, scale=var['scale'], 
                bisymlog_scaling_factor= var['bisymlog_scaling_factor'], DoF=len(coef_opt), 
                opt_type=var['obj_fcn_type'])
        
        res = minimize_scalar(loss_alpha_fcn, bounds=[-100, 2], method='bounded')
        loss_alpha = res.x
    
    if var['obj_fcn_type'] == 'Residual':
        loss_penalty = True
    else:
        loss_penalty = False

    output = resid_func(shock['opt_time_offset'], t_unc, ind_var, obs_sim, obs_exp[:,0], obs_exp[:,1], 
                              weights, obs_bounds, loss_alpha=loss_alpha, loss_c=var['loss_c'], 
                              loss_penalty=loss_penalty, scale=var['scale'], 
                              bisymlog_scaling_factor= var['bisymlog_scaling_factor'], 
                              DoF=len(coef_opt), opt_type=var['obj_fcn_type'], verbose=True)  

    output['shock'] = shock
    output['independent_var'] = ind_var
    output['observable'] = obs_sim
    output['t_unc'] = t_unc
    output['loss_alpha'] = loss_alpha

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
        self.x0 = input_dict['rxn_rate_opt']['x0']
        self.mech = input_dict['mech']
        self.var = self.parent.var
        self.t_unc = (-self.var['time_unc'], self.var['time_unc'])
        
        self.opt_type = 'local' # this is updated outside of the class
        
        self.dist = self.parent.optimize.dist
        self.opt_settings = {'obj_fcn_type': self.parent.optimization_settings.get('obj_fcn', 'type'),
                             'scale': self.parent.optimization_settings.get('obj_fcn', 'scale'),
                             'bisymlog_scaling_factor': self.parent.plot.signal.bisymlog.scaling_factor,
                             'loss_alpha': self.parent.optimization_settings.get('obj_fcn', 'alpha'),
                             'loss_c': self.parent.optimization_settings.get('obj_fcn', 'c'),
                             'bayes_dist_type': self.parent.optimization_settings.get('obj_fcn', 'bayes_dist_type'),
                             'bayes_unc_sigma': self.parent.optimization_settings.get('obj_fcn', 'bayes_unc_sigma')}

        if 'multiprocessing' in input_dict:
            self.multiprocessing = input_dict['multiprocessing']
        
        if 'pool' in input_dict:
            self.pool = input_dict['pool']
        else:
            self.multiprocessing = False
        
        self.signals = input_dict['signals']
        
        self.i = 0        
        self.__abort = False

        if self.opt_settings['obj_fcn_type'] == 'Bayesian': # initialize Bayesian_dictionary if Bayesian selected
            input_dict['opt_settings'] = self.opt_settings
            self.CheKiPEUQ_Frhodo_interface = CheKiPEUQ_Frhodo_interface(input_dict)
    
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
            return np.nan
        
        # Convert to mech values
        log_opt_rates = s + self.x0
        x = self.fit_all_coeffs(np.exp(log_opt_rates))
        if x is None: 
            return np.inf

        # Run Simulations
        output_dict = {}
        
        var_dict = {key: val for key, val in self.var['reactor'].items()}
        var_dict['t_unc'] = self.t_unc
        var_dict.update(self.opt_settings)
        
        display_ind_var = None
        display_observable = None
                                                                          
        if self.multiprocessing and len(self.shocks2run) > 1:
            args_list = ((var_dict, self.coef_opt, x, shock) for shock in self.shocks2run)
            calc_resid_outputs = self.pool.map(calculate_residuals, args_list)
            for calc_resid_output, shock in zip(calc_resid_outputs, self.shocks2run):
                append_output(output_dict, calc_resid_output)
                if shock is self.parent.display_shock:
                    display_ind_var = calc_resid_output['independent_var'] 
                    display_observable = calc_resid_output['observable']

        else:
            mpMech['obj'] = self.mech
            
            for shock in self.shocks2run:
                args_list = (var_dict, self.coef_opt, x, shock)
                calc_resid_output = calculate_residuals(args_list)
                append_output(output_dict, calc_resid_output)
                if shock is self.parent.display_shock:
                    display_ind_var = calc_resid_output['independent_var'] 
                    display_observable = calc_resid_output['observable']
        
        loss_resid = np.array(output_dict['loss'])
        exp_loss_alpha = np.array(output_dict['loss_alpha'])

        loss_alpha = self.opt_settings['loss_alpha']
        if loss_alpha == 3.0:
            if np.size(loss_resid) <= 2:  # optimizing only a few experiments, use SSE
                loss_alpha = 2.0
            
            else: # alpha is based on residual loss function, not great, but it's super slow otherwise
                loss_alpha_fcn = lambda alpha: self.calculate_obj_fcn(x, loss_resid, alpha, log_opt_rates, output_dict, obj_fcn_type='Residual')
        
                res = minimize_scalar(loss_alpha_fcn, bounds=[-100, 2], method='bounded')
                loss_alpha = res.x

        # testing loss alphas
        # print([loss_alpha, *exp_loss_alpha])
        obj_fcn = self.calculate_obj_fcn(x, loss_resid, loss_alpha, log_opt_rates, output_dict, obj_fcn_type=self.opt_settings['obj_fcn_type'])

        # For updating
        self.i += 1
        if not optimizing or self.i % 1 == 0:#5 == 0: # updates plot every 5
            if obj_fcn == 0 and self.opt_settings['obj_fcn_type'] != 'Bayesian':
                obj_fcn = np.inf
            
            stat_plot = {'shocks2run': self.shocks2run, 'resid': output_dict['resid'], 
                        'resid_outlier': self.loss_outlier, 'weights': output_dict['weights']}
            
            if 'KDE' in output_dict:
                stat_plot['KDE'] = output_dict['KDE']
                allResid = np.concatenate(output_dict['resid'], axis=0)
                
                stat_plot['fit_result'] = fitres = self.dist.fit(allResid)
                stat_plot['QQ'] = []
                for resid in stat_plot['resid']:
                    QQ = stats.probplot(resid, sparams=fitres, dist=self.dist, fit=False)
                    QQ = np.array(QQ).T
                    stat_plot['QQ'].append(QQ)
            
            update = {'type': self.opt_type, 'i': self.i, 
                      'obj_fcn': obj_fcn, 'stat_plot': stat_plot, 
                      's': s, 'x': x, 'coef_opt': self.coef_opt, 
                      'ind_var': display_ind_var, 'observable': display_observable}
            
            self.signals.update.emit(update)

        if optimizing:
            return obj_fcn
        else:
            return obj_fcn, x, output_dict['shock']
       
    def calculate_obj_fcn(self, x, loss_resid, alpha, log_opt_rates, output_dict, obj_fcn_type='Residual', loss_outlier=0):
        C = self.opt_settings['loss_c']

        if np.size(loss_resid) == 1:  # optimizing single experiment
            loss_outlier = 0
            loss_exp = loss_resid
        else:                   # optimizing multiple experiments
            loss_min = loss_resid.min()
            exp_loss_weights, C, alpha = adaptive_weights(loss_resid-loss_min, weights=np.array([]), C_scalar=C, alpha=alpha)
            loss_exp = exp_loss_weights*(loss_resid-loss_min)**2
        
        self.loss_outlier = loss_outlier

        if obj_fcn_type == 'Residual':
            if np.size(loss_resid) == 1:  # optimizing single experiment
                obj_fcn = loss_exp[0]
            else:
                loss_exp = loss_exp - loss_exp.min() + loss_min
                #obj_fcn = np.median(loss_exp)
                obj_fcn = np.average(loss_exp)

        elif obj_fcn_type == 'Bayesian':
            if np.size(loss_resid) == 1:  # optimizing single experiment
                Bayesian_weights = np.array(output_dict['aggregate_weights'], dtype=object).flatten()
            else:
                loss_exp = rescale_loss_fcn(loss_resid, loss_exp)
                aggregate_weights = np.array(output_dict['aggregate_weights'], dtype=object)
                exp_loss_weights, C, alpha = adaptive_weights(loss_resid, weights=np.array([]), C_scalar=C, alpha=alpha)

                # SSE = penalized_loss_fcn(loss_resid, mu=loss_min, use_penalty=False)
                # SSE = rescale_loss_fcn(loss_resid, SSE)
                # exp_loss_weights = loss_exp/SSE # comparison is between selected loss fcn and SSE (L2 loss)

                Bayesian_weights = np.concatenate(aggregate_weights.T*exp_loss_weights, axis=0).flatten()
            
            # need to normalize weight values between iterations
            Bayesian_weights = Bayesian_weights/Bayesian_weights.sum()

            CheKiPEUQ_eval_dict = {'log_opt_rates': log_opt_rates, 'x': x, 'output_dict': output_dict, 
                                   'bayesian_weights': Bayesian_weights, 'iteration_num': self.i}

            obj_fcn = self.CheKiPEUQ_Frhodo_interface.evaluate(CheKiPEUQ_eval_dict)

        return obj_fcn

    def fit_all_coeffs(self, all_rates):      
        coeffs = []
        i = 0
        for rxn_coef in self.rxn_coef_opt:
            rxnIdx = rxn_coef['rxnIdx']
            T, P, X = rxn_coef['T'], rxn_coef['P'], rxn_coef['X']
            coef_bnds = [rxn_coef['coef_bnds']['lower'], rxn_coef['coef_bnds']['upper']]
            rxn_rates = all_rates[i:i+len(T)]
            if len(coeffs) == 0:
                coeffs = fit_coeffs(rxn_rates, T, P, X, rxnIdx, rxn_coef['key'], rxn_coef['coefName'], 
                                    rxn_coef['is_falloff_limit'], coef_bnds, self.mech, self.pool)
                if coeffs is None:
                    return
            else:
                coeffs_append = fit_coeffs(rxn_rates, T, P, X, rxnIdx, rxn_coef['key'], rxn_coef['coefName'], 
                                           rxn_coef['is_falloff_limit'], coef_bnds, self.mech, self.pool)
                if coeffs_append is None:
                    return
                coeffs = np.append(coeffs, coeffs_append)
            
            i += len(T)

        return coeffs