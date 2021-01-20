# This file is part of Frhodo. Copyright © 2020, UChicago Argonne, LLC
# and licensed under BSD-3-Clause. See License.txt in the top-level 
# directory for license and copyright information.

import io, contextlib
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.interpolate import CubicSpline
from scipy import stats
from copy import deepcopy

import mech_fcns
from optimize.fit_coeffs import fit_coeffs

mpMech = {}

def initialize_parallel_worker(mech_txt, coeffs, coeffs_bnds, rate_bnds):
    mpMech['obj'] = mech = mech_fcns.Chemical_Mechanism()

    # hide mechanism loading problems because they will already have been seen
    with contextlib.redirect_stderr(io.StringIO()):
        with contextlib.redirect_stdout(io.StringIO()):
            mech.set_mechanism(mech_txt)    # load mechanism from yaml text in memory

    mech.coeffs = deepcopy(coeffs)
    mech.coeffs_bnds = deepcopy(coeffs_bnds)
    mech.rate_bnds = deepcopy(rate_bnds)

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
    nonNan_idx = np.where(values!=np.nan)
    values = np.array(values[nonNan_idx])
    quantiles = np.array(quantiles)
    if weights is None or len(weights) == 0:
        weights = np.ones(len(values))
    weights = np.array(weights[nonNan_idx])
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

def outlier(res, a=2, c=1, weights=[], max_iter=25, percentile=0.25):
    def diff(res_outlier):
        if len(res_outlier) < 2: 
            return 1
        else:
            return np.diff(res_outlier)[0]

    trunc_res = np.abs(res.copy())
    percentiles = [percentile, 1-percentile]
    res_outlier = []
    if a != 2: # define outlier with 1.5 IQR rule
        for n in range(max_iter):
            if diff(res_outlier) == 0:   # iterate until res_outlier is the same as prior iteration
                break
                
            if len(res_outlier) > 0:
                trunc_res = trunc_res[trunc_res < res_outlier[-1]] 
            
            [q1, q3] = weighted_quantile(trunc_res, percentiles, weights=weights)
            iqr = q3 - q1       # interquartile range      
            
            if len(res_outlier) == 2:
                del res_outlier[0]
            
            res_outlier.append(q3 + iqr*1.5)
        
        res_outlier = res_outlier[-1]
    else:
        res_outlier = 1

    return c*res_outlier
    
def generalized_loss_fcn(res, a=2, c=1):    # defaults to L2 loss
    c_2 = c**2
    x_c_2 = res**2/c_2
    if a == 1:          # generalized function reproduces
        loss = (x_c_2 + 1)**(1/2) - 1
    if a == 2:
        loss = 0.5*x_c_2
    elif a == 0:
        loss = np.log(0.5*x_c_2+1)
    elif a == -2:       # generalized function reproduces
        loss = 2*x_c_2/(x_c_2 + 4)
    elif a <= -1000:    # supposed to be negative infinity
        loss = 1 - np.exp(-0.5*x_c_2)
    else:
        loss = np.abs(a-2)/a*((x_c_2/np.abs(a-2) + 1)**(a/2) - 1)
        
    return loss*c_2   # multiplying by c^2 is not necessary, but makes order appropriate

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
        t_bounds = [max([t_sim[0], t_exp[0]])]       # Largest initial time in SIM and Exp
        t_bounds.append(min([t_sim[-1], t_exp[-1]])) # Smallest final time in SIM and Exp
        # Values within t_bounds
        exp_bounds = np.where(np.logical_and((t_exp >= t_bounds[0]),(t_exp <= t_bounds[1])))[0]
        
        return exp_bounds
    
    def time_adjust_func(t_offset, t_adjust, t_sim, obs_sim, t_exp, obs_exp, weights, 
                         loss_alpha=2, loss_c=1, scale='Linear', DoF=1, verbose=False):

        t_sim_shifted = t_sim + t_offset + t_adjust

        # Compare SIM Density Grad vs. Experimental
        exp_bounds = calc_exp_bounds(t_sim_shifted, t_exp)
        t_exp, obs_exp, weights = t_exp[exp_bounds], obs_exp[exp_bounds], weights[exp_bounds]
        
        f_interp = CubicSpline(t_sim_shifted.flatten(), obs_sim.flatten())
        obs_sim_interp = f_interp(t_exp)
        
        if scale == 'Linear':
            resid = np.subtract(obs_exp, obs_sim_interp)
        elif scale == 'Log':
            ind = np.argwhere(((obs_exp!=0.0)&(obs_sim_interp!=0.0)))
            weights = weights[ind].flatten()
            m = np.divide(obs_exp[ind], obs_sim_interp[ind])
            resid = np.log10(np.abs(m)).flatten()
        
        resid_outlier = outlier(resid, a=loss_alpha, c=loss_c, weights=weights)
        loss = generalized_loss_fcn(resid, a=loss_alpha, c=resid_outlier)

        loss_sqr = loss**2
        wgt_sum = weights.sum()
        N = wgt_sum - DoF
        if N <= 0:
            N = wgt_sum
        stderr_sqr = (loss_sqr*weights).sum()/N
        chi_sqr = loss_sqr/stderr_sqr
        #loss_scalar = (chi_sqr*weights).sum()
        std_resid = chi_sqr**(1/2)
        loss_scalar = np.average(std_resid, weights=weights)
        
        if verbose:
            output = {'chi_sqr': chi_sqr, 'resid': resid, 'resid_outlier': resid_outlier,
                      'loss': loss_scalar, 'weights': weights, 'obs_sim_interp': obs_sim_interp}
            return output
        else:   # needs to return single value for optimization
            return loss_scalar
    
    def calc_density(x, data, dim=1):
        stdev = np.std(data)
        [q1, q3] = weighted_quantile(data, [0.25, 0.75])
        iqr = q3 - q1       # interquartile range   
        A = np.min([stdev, iqr/1.34])/stdev  # bandwidth is multiplied by std of sample
        bw = 0.9*A*len(data)**(-1./(dim+4))

        return stats.gaussian_kde(data, bw_method=bw)(x)
        
    def OoM(x):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        x[x==0] = 1                       # if zero, make OoM 0
        return np.floor(np.log10(np.abs(x)))
    
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
    ind_var, obs = SIM.independent_var[:,None], SIM.observable[:,None]
    
    weights = shock['weights_trim']
    obs_exp = shock['exp_data_trim']
    
    if not np.any(var['t_unc']):
        t_unc = 0
    else:
        t_unc_OoM = np.mean(OoM(var['t_unc']))  # Do at higher level? (computationally efficient)
        # calculate time adjust with mse (loss_alpha = 2, loss_c =1)
        time_adj_decorator = lambda t_adjust: time_adjust_func(shock['time_offset'], t_adjust*10**t_unc_OoM, 
                ind_var, obs, obs_exp[:,0], obs_exp[:,1], weights, scale=var['resid_scale'], 
                DoF=len(coef_opt))
        
        res = minimize_scalar(time_adj_decorator, bounds=var['t_unc']/10**t_unc_OoM, method='bounded')
        t_unc = res.x*10**t_unc_OoM
    
    output = time_adjust_func(shock['time_offset'], t_unc, ind_var, obs, obs_exp[:,0], obs_exp[:,1], 
                              weights, loss_alpha=var['loss_alpha'], loss_c=var['loss_c'], 
                              scale=var['resid_scale'], DoF=len(coef_opt), verbose=True)  
    
    output['shock'] = shock
    output['independent_var'] = ind_var
    output['observable'] = obs

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
        
        self.dist = self.parent.optimize.dist
        self.resid_scale = self.parent.optimization_settings.get('loss', 'resid_scale')
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
        var_dict['resid_scale'] = self.resid_scale
        var_dict.update({'loss_alpha': self.loss_alpha, 'loss_c': self.loss_c})
        
        display_ind_var = None
        display_observable = None
        if self.multiprocessing:
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
        
        # loss = np.concatenate(output_dict['loss'], axis=0)
        loss = np.array(output_dict['loss'])

        if np.size(loss) > 1:
            c = outlier(loss, a=self.loss_alpha, c=self.loss_c)
            loss = generalized_loss_fcn(loss, a=self.loss_alpha, c=c)
            total_loss = loss.mean()
        else:
            c = 0
            total_loss = loss[0]
        
        # For updating
        self.i += 1
        if not optimizing or self.i % 1 == 0:#5 == 0: # updates plot every 5
            if total_loss == 0:
                total_loss = np.inf
            
            stat_plot = {'shocks2run': self.shocks2run, 'resid': output_dict['resid'], 
                        'resid_outlier': c, 'weights': output_dict['weights']}
            
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
                      'loss': total_loss, 'stat_plot': stat_plot, 
                      'x': x, 'coef_opt': self.coef_opt, 
                      'ind_var': display_ind_var, 'observable': display_observable}
            
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