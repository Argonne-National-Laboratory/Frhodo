# This file is part of Frhodo. Copyright © 2020, UChicago Argonne, LLC
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

####THE BELOW LINE IS FOR BAYESIAN PARAMETER ESTIMATION DEVELOPMENT AND SHOULD BE REMOVED LATER####
forceBayesian = True

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
  
#below was formerly calculate_residuals  
def calculate_objective_function(args_list, objective_function_type='residual'):   
    if forceBayesian == True: objective_function_type = 'Bayesian'

    def calc_exp_bounds(t_sim, t_exp):
        t_bounds = [max([t_sim[0], t_exp[0]])]       # Largest initial time in SIM and Exp
        t_bounds.append(min([t_sim[-1], t_exp[-1]])) # Smallest final time in SIM and Exp
        # Values within t_bounds
        exp_bounds = np.where(np.logical_and((t_exp >= t_bounds[0]),(t_exp <= t_bounds[1])))[0]
        
        return exp_bounds
    
    #the below function has implied arguments of shock (from args_list) and also 
    #these for the case of bayesian objective_function_type: varying_rate_vals_indices, varying_rate_vals_initial_guess, varying_rate_vals_lower_bnds, varying_rate_vals_upper_bnds
    def time_adjust_func(t_offset, t_adjust, t_sim, obs_sim, t_exp, obs_exp, weights, 
                         loss_alpha=2, loss_c=1, scale='Linear', DoF=1, verbose=False, objective_function_type='residual'):

        t_sim_shifted = t_sim + t_offset + t_adjust

        # Compare SIM Density Grad vs. Experimental
        exp_bounds = calc_exp_bounds(t_sim_shifted, t_exp)
        t_exp, obs_exp, weights = t_exp[exp_bounds], obs_exp[exp_bounds], weights[exp_bounds]
        
        f_interp = CubicSpline(t_sim_shifted.flatten(), obs_sim.flatten())
        obs_sim_interp = f_interp(t_exp)
        
        if scale == 'Linear':
            resid = np.subtract(obs_exp, obs_sim_interp)
            if objective_function_type.lower() == 'bayesian':
                shock['last_obs_sim_interp'] = obs_sim_interp
        elif scale == 'Log':
            ind = np.argwhere(((obs_exp!=0.0)&(obs_sim_interp!=0.0)))
            weights = weights[ind].flatten()
            m = np.divide(obs_exp[ind], obs_sim_interp[ind])
            resid = np.log10(np.abs(m)).flatten()
            
        #There are two possible objective_function_types: 'residual' and 'Bayesian'.
        if objective_function_type.lower() == 'residual':
            #TODO: Ashi is not sure if some kind of trimming is happening to the experimental data in log scale.
            #For log scale, we would also need to change the PE_object creation to take in the log_scale data.
            if objective_function_type.lower() == 'bayesian':
                shock['last_obs_sim_interp'] = obs_sim_interp
            
        #There are two possible objective_function_types: 'residual' and 'Bayesian'.
        if objective_function_type.lower() == 'residual' or objective_function_type.lower() == 'bayesian': #FIXME: currently we go into this code for Bayesian also, but that is just to satisfy the QQ etc.
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
            objective_function_value = loss_scalar
            if verbose: 
                output = {'chi_sqr': chi_sqr, 'resid': resid, 'resid_outlier': resid_outlier,
                          'obj_fcn': loss_scalar, 'weights': weights, 'obs_sim_interp': obs_sim_interp}
            else:
                output = objective_function_value #normal case.
        elif objective_function_type.lower() == 'bayesian':
            objective_function_value = log_posterior_density
            #TODO: call CheKiPEUQ from here.
            if verbose: 
                output = objective_function_value #to be made a dictionary.
                print("line 183, about to fill output with the residual objective_function_value", objective_function_value)
                output = {'chi_sqr': chi_sqr, 'resid': resid, 'resid_outlier': resid_outlier,
                          'loss': loss_scalar, 'weights': weights, 'obs_sim_interp': obs_sim_interp}
            else:
                print("line 189, about to fill output with the residual objective_function_value", objective_function_value)
                output = objective_function_value #normal case for residuals based optimization.
        
        #FIXME: for Verbose currently we make most of the outputs above, and then override the 'loss_scalar' with the objective function from CheKiPEUQ.
        if objective_function_type.lower() == 'bayesian':
            #CheKiPEUQ requires a simulation_function based on only the paramters of interest.
            #Typically, we would us a wrapper.  However, here the structure is a bit different.
            #We have time_adjust_func and time_adj_decorator, so we need to create a function
            #that will get values from **inside** that function.
            #We could just "make" the PE_object again and again inside that function,
            #but doing so is not a good solution. Instead, we will use something like a global variable.
            #Here, the "shock" variable is a dictionary in the present namespace, a higher space than the time_adjust_func,
            #so we will just access a field from the shock variable. That field does not need to exist at the time of this function's creation,
            #just at the time that this function gets called.
            #We will only pass in the rate_val values that are being allowed to vary.
            #Note that the output does not actually depend on varying_rate_vals, so we rely upon only calling it
            #after last_obs_sim_interp has been changed.
            def get_last_obs_sim_interp(varying_rate_vals): 
                try:
                    last_obs_sim_interp = shock['last_obs_sim_interp']
                    last_obs_sim_interp = np.array(shock['last_obs_sim_interp']).T
                except:
                    print("this isline 207! There may be an error occurring!")
                    last_obs_sim_interp = None
                return last_obs_sim_interp
            import optimize.CheKiPEUQ_from_Frhodo    
            #now we make a PE_object from a wrapper function inside CheKiPEUQ_from_Frhodo. This PE_object can be accessed from inside time_adjust_func.
            #TODO: we should bring in x_bnds (the coefficent bounds) so that we can use the elementary step coefficients for Bayesian rather than the rate_val values.
            #Step 2 of Bayesian:  populate Bayesian_dict with any variables and uncertainties needed.
            Bayesian_dict = {}
            Bayesian_dict['simulation_function'] = get_last_obs_sim_interp #a wrapper that just returns the last_obs_sim_interp
            Bayesian_dict['observed_data'] = obs_exp
            Bayesian_dict['pars_initial_guess'] = varying_rate_vals_initial_guess
            Bayesian_dict['pars_lower_bnds'] = varying_rate_vals_lower_bnds
            Bayesian_dict['pars_upper_bnds'] = varying_rate_vals_upper_bnds
            Bayesian_dict['observed_data_lower_bounds'] = []
            Bayesian_dict['observed_data_upper_bounds'] = []
            Bayesian_dict['weights_data'] = weights
            Bayesian_dict['pars_uncertainty_distribution'] = 'gaussian' #A. Savara recommends 'uniform' for rate constants and 'gaussian' for things like "log(A)" and "Ea"
            
            #Step 3 of Bayesian:  create a CheKiPEUQ_PE_Object (this is a class object)
            CheKiPEUQ_PE_object = optimize.CheKiPEUQ_from_Frhodo.load_into_CheKiPUEQ(
                simulation_function=    Bayesian_dict['simulation_function'],
                observed_data=          Bayesian_dict['observed_data'],
                pars_initial_guess =    Bayesian_dict['pars_initial_guess'],
                pars_lower_bnds =       Bayesian_dict['pars_lower_bnds'],
                pars_upper_bnds =       Bayesian_dict['pars_upper_bnds'],
                observed_data_lower_bounds= Bayesian_dict['observed_data_lower_bounds'],
                observed_data_upper_bounds= Bayesian_dict['observed_data_upper_bounds'],
                weights_data=               Bayesian_dict['weights_data'],
                pars_uncertainty_distribution=  Bayesian_dict['pars_uncertainty_distribution'])
            #Step 4 of Bayesian:  call a function to get the posterior density which will be used as the objective function.
            #We need to provide the current values of the varying_rate_vals to feed into the function.
            varying_rate_vals = np.array(shock['rate_val'])[list(varying_rate_vals_indices)] #when extracting a list of multiple indices, instead of array[index] one use array[[indices]]
            log_posterior_density = optimize.CheKiPEUQ_from_Frhodo.get_log_posterior_density(CheKiPEUQ_PE_object, varying_rate_vals)
            #Step 5 of Bayesian:  return the objective function and any other metrics desired.
            objective_function_value = -1*log_posterior_density #need neg_logP because minimizing.
            if verbose: #FIXME: most of this dictionary is currently populated from values calculated for residuals.
                print("line 223, about to fill output with the Bayesian objective_function_value", objective_function_value)
                output = {'chi_sqr': chi_sqr, 'resid': resid, 'resid_outlier': resid_outlier,
                          'obj_fcn': objective_function_value, 'weights': weights, 'obs_sim_interp': obs_sim_interp}
            else:
                output = objective_function_value #normal case.

        return output
    
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
    
    #var is a dictionary of Connected Variables: {'t_unit_conv': 1e-06, 'name': 'Incident Shock Reactor', 'solve_energy': True, 'frozen_comp': False, 'ode_solver': 'BDF', 'ode_rtol': 1e-06, 'ode_atol': 1e-08, 't_end': 1.2e-05, 'sim_interp_factor': 1, 't_unc': (-0.0, 0.0), 'resid_scale': 'Linear', 'loss_alpha': -2.0, 'loss_c': 1.0}
    
    #coef_opt is a list of dictionaries containing reaction parameters to optimize (but not their bounds):  [{'rxnIdx': 2, 'coefIdx': 0, 'coefName': 'activation_energy'}, {'rxnIdx': 2, 'coefIdx': 1, 'coefName': 'pre_exponential_factor'}, {'rxnIdx': 2, 'coefIdx': 2, 'coefName': 'temperature_exponent'}] Note that the rxnIdx has array indexing, so rxnIdx of 2 is actually "R3" in the example reaction.
    
    #x is a small list of the coefficients which are being allowed to vary: [0.         4.16233447 3.04590318]
    
    #shock is a HUGE dictionary, like a global namespace. It includes all of the species % ratios, the rate_vals, the weightings, timeoffset, and many other things.  The rate boundaries are in absolute values in rate_bnds. The experimental dat is in exp_data, and the weights are in weights.
    var, coef_opt, x, shock = args_list
    mech = mpMech['obj']
    
    # print("line 202", var)
    # print("line 203", coef_opt)
    print("line 204", x)
    # print("line 205", shock);    sys.exit()
    # Optimization Begins, update mechanism
    update_mech_coef_opt(mech, coef_opt, x)

    T_reac, P_reac, mix = shock['T_reactor'], shock['P_reactor'], shock['thermo_mix']
    
    SIM_kwargs = {'u_reac': shock['u2'], 'rho1': shock['rho1'], 'observable': shock['observable'], 
                  't_lab_save': None, 'sim_int_f': var['sim_interp_factor'], 
                  'ODE_solver': var['ode_solver'], 'rtol': var['ode_rtol'], 'atol': var['ode_atol']}
    
    if '0d Reactor' in var['name']:
        SIM_kwargs['solve_energy'] = var['solve_energy']
        SIM_kwargs['frozen_comp'] = var['frozen_comp']
    
    #Below, SIM is a simulation result, and includes obs_sim
    SIM, verbose = mech.run(var['name'], var['t_end'], T_reac, P_reac, mix, **SIM_kwargs)    
    ind_var, obs_sim = SIM.independent_var[:,None], SIM.observable[:,None]
    
    weights = shock['weights_trim']
    obs_exp = shock['exp_data_trim']
    
    #TO CONSIDER: the CheKiPEUQ_PE_object creation can be created earlier (here or higher) if obs_exp would be 'constant' in size from here.
    #The whole block of code for CheKiPEUQ_PE_object creation has been moved into time_adjust_func because Frhodo seems to do one concentration at a time and to change the array size while doing so.
    #If we are doing a Bayesian parameter estimation, we need to create CheKiPEUQ_PE_object. This has to come between the above functions because we need to feed in the simulation_function, and it needs to come above the 'minimize' function that is below.
    
    if objective_function_type.lower() == 'bayesian':        
        import optimize.CheKiPEUQ_from_Frhodo    
        #Step 1 of Bayesian:  Prepare any variables that need to be passed into time_adjust_func.
        if 'original_rate_val' not in shock: #check if this is the first time being called, and store rate_vals and create PE_object if it is.
            shock['original_rate_val'] = deepcopy(shock['rate_val']) #TODO: Check if this is the correct place to make original_rate_val
            shock['newOptimization'] = True
            #TODO: need to make sure we get the **original** rate_vals and bounds and keep them as the prior.
        varying_rate_vals_indices, varying_rate_vals_initial_guess, varying_rate_vals_lower_bnds, varying_rate_vals_upper_bnds = optimize.CheKiPEUQ_from_Frhodo.get_varying_rate_vals_and_bnds(shock['original_rate_val'],shock['rate_bnds'])
        print("line 283, CREATING varying_rate_vals_initial_guess ", varying_rate_vals_initial_guess)
            #FIXME: #TODO: Need to add an "or" statement or flag to allow below to execute when somebody has changed their initial guesses intentionally or are doing a new optimization.
        # if ('newOptimization' in shock) and (shock['newOptimization'] == True):
    
    if not np.any(var['t_unc']):
        t_unc = 0
    else:        
        t_unc_OoM = np.mean(OoM(var['t_unc']))  # Do at higher level? (computationally efficient)
        # calculate time adjust with mse (loss_alpha = 2, loss_c =1)
        #comparing to time_adjust_func, arguments below are...: t_offset=shock['time_offset'], t_adjust=t_adjust*10**t_unc_OoM,
        #            t_sim=ind_var, obs_sim=obs_sim, t_exp=obs_exp[:,0], obs_exp=obs_exp[:,1], weights=weights
        time_adj_decorator = lambda t_adjust: time_adjust_func(shock['time_offset'], t_adjust*10**t_unc_OoM, 
                ind_var, obs_sim, obs_exp[:,0], obs_exp[:,1], weights, scale=var['scale'], 
                DoF=len(coef_opt), objective_function_type=objective_function_type) #objective_function_type is 'residual' or 'Bayesian'
        res = minimize_scalar(time_adj_decorator, bounds=var['t_unc']/10**t_unc_OoM, method='bounded')
        t_unc = res.x*10**t_unc_OoM    
    
    output = time_adjust_func(shock['time_offset'], t_unc, ind_var, obs_sim, obs_exp[:,0], obs_exp[:,1], 
                              weights, loss_alpha=var['loss_alpha'], loss_c=var['loss_c'], 
                              scale=var['scale'], DoF=len(coef_opt), verbose=True, objective_function_type=objective_function_type) #objective_function_type is 'residual' or 'Bayesian'                                
    
    output['shock'] = shock
    output['independent_var'] = ind_var
    output['observable'] = obs_sim

    plot_stats = True
    if plot_stats:
        if objective_function_type== 'residual':
            x = np.linspace(output['resid'].min(), output['resid'].max(), 300)
            density = calc_density(x, output['resid'], dim=1)   #kernel density estimation
            output['KDE'] = np.column_stack((x, density))
        if objective_function_type== 'Bayesian':
            pass #To be added.

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
        self.scale = self.parent.optimization_settings.get('obj_fcn', 'scale')
        self.loss_alpha = self.parent.optimization_settings.get('obj_fcn', 'alpha')
        self.loss_c = self.parent.optimization_settings.get('obj_fcn', 'c')
        
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
        # Below, calc_objective_function_output was formerly calc_resid_output 
        def append_output(output_dict, calc_objective_function_output):
            for key in calc_objective_function_output:
                if key not in output_dict:
                    output_dict[key] = []
                output_dict[key].append(calc_objective_function_output[key])
            
            return output_dict
        
        if self.__abort: 
            raise Exception('Optimization terminated by user')
            self.signals.log.emit('\nOptimization aborted')
            return
        
        # Convert to mech values, by putting in rate constants.
        x = self.fit_all_coeffs(np.exp(s*self.x0))
        if x is None: 
            return np.inf

        # Run Simulations
        output_dict = {}
        
        var_dict = {key: val for key, val in self.var['reactor'].items()}
        var_dict['t_unc'] = self.t_unc
        var_dict['scale'] = self.scale
        var_dict.update({'loss_alpha': self.loss_alpha, 'loss_c': self.loss_c})
        
        display_ind_var = None
        display_observable = None
        
        # Below, calc_objective_function_output was formerly calc_resid_output 
        if self.multiprocessing:
            args_list = ((var_dict, self.coef_opt, x, shock) for shock in self.shocks2run)
            calc_objective_function_outputs = self.pool.map(calculate_objective_function, args_list)
            for calc_objective_function_output, shock in zip(calc_objective_function_outputs, self.shocks2run):
                append_output(output_dict, calc_objective_function_output)
                if shock is self.parent.display_shock:
                    display_ind_var = calc_objective_function_output['independent_var'] 
                    display_observable = calc_objective_function_output['observable']
        else:
            mpMech['obj'] = self.mech
            for shock in self.shocks2run:
                args_list = (var_dict, self.coef_opt, x, shock)
                calc_objective_function_output = calculate_objective_function(args_list)
                append_output(output_dict, calc_objective_function_output)
                if shock is self.parent.display_shock:
                    display_ind_var = calc_objective_function_output['independent_var'] 
                    display_observable = calc_objective_function_output['observable']
        
        # obj_fcn = np.concatenate(output_dict['obj_fcn'], axis=0)
        obj_fcn = np.array(output_dict['obj_fcn'])

        if np.size(obj_fcn) > 1:
            c = outlier(obj_fcn, a=self.loss_alpha, c=self.loss_c)
            obj_fcn = generalized_loss_fcn(obj_fcn, a=self.loss_alpha, c=c)
            total_obj_fcn = obj_fcn.mean()
        else:
            c = 0
            total_obj_fcn = obj_fcn[0]
        
        # For updating
        self.i += 1
        if not optimizing or self.i % 1 == 0:#5 == 0: # updates plot every 5
            if total_obj_fcn == 0:
                total_obj_fcn = np.inf
            
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
                      'obj_fcn': total_obj_fcn, 'stat_plot': stat_plot, 
                      'x': x, 'coef_opt': self.coef_opt, 
                      'ind_var': display_ind_var, 'observable': display_observable}
            
            self.signals.update.emit(update)
                
        if optimizing:
            return total_obj_fcn
        else:
            return total_obj_fcn, x, output_dict['shock']
            
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
        #The coeffs are log(A), n, Ea
        return coeffs