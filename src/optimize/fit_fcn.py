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

def outlier(x, a=2, c=1, weights=[], max_iter=25, percentile=0.25):
    def diff(x_outlier):
        if len(x_outlier) < 2: 
            return 1
        else:
            return np.diff(x_outlier)[0]

    x = np.abs(x.copy())
    percentiles = [percentile, 1-percentile]
    x_outlier = []
    if a != 2: # define outlier with 1.5 IQR rule
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
    else:
        x_outlier = 1

    return c*x_outlier
    
def generalized_loss_fcn(x, a=2, c=1):    # defaults to L2 loss
    c_2 = c**2
    x_c_2 = x**2/c_2
    if a == 1:          # generalized function reproduces
        loss = (x_c_2 + 1)**(0.5) - 1
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
    def time_adjust_func(t_offset, t_adjust, t_sim, obs_sim, t_exp, obs_exp, weights, 
                         loss_alpha=2, loss_c=1, scale='Linear', DoF=1, opt_type='Residual', 
                         verbose=False):

        def calc_exp_bounds(t_sim, t_exp):
            t_bounds = [max([t_sim[0], t_exp[0]])]       # Largest initial time in SIM and Exp
            t_bounds.append(min([t_sim[-1], t_exp[-1]])) # Smallest final time in SIM and Exp
            # Values within t_bounds
            exp_bounds = np.where(np.logical_and((t_exp >= t_bounds[0]),(t_exp <= t_bounds[1])))[0]
        
            return exp_bounds

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
                      'loss': loss_scalar, 'weights': weights, 'obs_sim_interp': obs_sim_interp,
                      'obs_exp': obs_exp}

            if opt_type == 'Bayesian': # need to calculate aggregate weights to reduce outliers in bayesian
                loss_weights = loss/generalized_loss_fcn(resid) # comparison is between selected loss fcn and SSE (L2 loss)
                output['aggregate_weights'] = weights*loss_weights

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
    ind_var, obs_sim = SIM.independent_var[:,None], SIM.observable[:,None]
    
    weights = shock['weights_trim']
    obs_exp = shock['exp_data_trim']
    
    if not np.any(var['t_unc']):
        t_unc = 0
    else:
        t_unc_OoM = np.mean(OoM(var['t_unc']))  # Do at higher level in code? (computationally efficient)
        # calculate time adjust with mse (loss_alpha = 2, loss_c =1)                                                                         
        time_adj_decorator = lambda t_adjust: time_adjust_func(shock['time_offset'], t_adjust*10**t_unc_OoM, 
                ind_var, obs_sim, obs_exp[:,0], obs_exp[:,1], weights, scale=var['scale'], 
                DoF=len(coef_opt), opt_type=var['obj_fcn_type'])
        
        res = minimize_scalar(time_adj_decorator, bounds=var['t_unc']/10**t_unc_OoM, method='bounded')
        t_unc = res.x*10**t_unc_OoM
    
    output = time_adjust_func(shock['time_offset'], t_unc, ind_var, obs_sim, obs_exp[:,0], obs_exp[:,1], 
                              weights, loss_alpha=var['loss_alpha'], loss_c=var['loss_c'], 
                              scale=var['scale'], DoF=len(coef_opt), opt_type=var['obj_fcn_type'], 
                              verbose=True)  

    output['shock'] = shock
    output['independent_var'] = ind_var
    output['observable'] = obs_sim

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
        self.opt_settings = {'obj_fcn_type': self.parent.optimization_settings.get('obj_fcn', 'type'),
                             'scale': self.parent.optimization_settings.get('obj_fcn', 'scale'),
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
            #Step 1 of Bayesian:  Prepare any variables that need to be passed in for Bayesian PE_object creation.
            #Step 2 of Bayesian:  populate Bayesian_dict with any variables and uncertainties needed.
            self.Bayesian_dict = {}
            self.Bayesian_dict['pars_uncertainty_distribution'] = self.opt_settings['bayes_dist_type']  #options can be 'Auto', 'Gaussian' or 'Uniform'. 
            # T. Sikes: The options for self.opt_settings['bayes_dist_type'] is now Automatic, Gaussian, or Uniform
            
            #A. Savara recommends 'uniform' for rate constants and 'gaussian' for things like "log(A)" and "Ea"
            self.Bayesian_dict['rate_constants_initial_guess'] = deepcopy(self.x0)
            self.Bayesian_dict['rate_constants_lower_bnds'] = deepcopy(input_dict['bounds']['lower'])
            self.Bayesian_dict['rate_constants_upper_bnds'] = deepcopy(input_dict['bounds']['upper'])
            num_rate_consants = len(self.Bayesian_dict['rate_constants_initial_guess'])
            self.Bayesian_dict['rate_constants_bnds_exist'] =  np.array(np.ones((num_rate_consants,2)), dtype = bool) #From Jan 2021, we are setting [True True] for each rate_constant.

            self.Bayesian_dict['rate_constants_parameters_changing'] = deepcopy(self.coef_opt)
            self.Bayesian_dict['rate_constants_parameters_initial_guess'] = []
            self.Bayesian_dict['rate_constants_parameters_lower_bnds'] = []
            self.Bayesian_dict['rate_constants_parameters_upper_bnds'] = []
            self.Bayesian_dict['rate_constants_parameters_bnds_exist'] = []
            for rxn_coef in self.rxn_coef_opt:
                self.Bayesian_dict['rate_constants_parameters_initial_guess'].append(deepcopy(rxn_coef['coef_x0'])) 
                self.Bayesian_dict['rate_constants_parameters_lower_bnds'].append(deepcopy(rxn_coef['coef_bnds']['lower']))
                self.Bayesian_dict['rate_constants_parameters_upper_bnds'].append(deepcopy(rxn_coef['coef_bnds']['upper']))
                self.Bayesian_dict['rate_constants_parameters_bnds_exist'].extend(list(deepcopy(rxn_coef['coef_bnds']['exist']))) #we can't use append because this is a list of lists/array, and we want parallel construction.
            self.Bayesian_dict['rate_constants_parameters_initial_guess'] = np.array(self.Bayesian_dict['rate_constants_parameters_initial_guess']).flatten()
            self.Bayesian_dict['rate_constants_parameters_lower_bnds'] = np.array(self.Bayesian_dict['rate_constants_parameters_lower_bnds']).flatten()
            self.Bayesian_dict['rate_constants_parameters_upper_bnds'] = np.array(self.Bayesian_dict['rate_constants_parameters_upper_bnds']).flatten()
            #self.Bayesian_dict['rate_constants_parameters_bnds_exist'] does not get flattened because it is a list of list/arrays (so we don't want it flattened).

            Bayesian_dict = self.Bayesian_dict            
            import optimize.CheKiPEUQ_from_Frhodo    
            #concatenate all of the initial guesses and bounds. 
            Bayesian_dict['pars_initial_guess'], Bayesian_dict['pars_lower_bnds'],Bayesian_dict['pars_upper_bnds'], Bayesian_dict['pars_bnds_exist'], Bayesian_dict['unbounded_indices'] = optimize.CheKiPEUQ_from_Frhodo.get_consolidated_parameters_arrays( 
                Bayesian_dict['rate_constants_initial_guess'],
                Bayesian_dict['rate_constants_lower_bnds'],
                Bayesian_dict['rate_constants_upper_bnds'],                
                Bayesian_dict['rate_constants_bnds_exist'],
                Bayesian_dict['rate_constants_parameters_initial_guess'],
                Bayesian_dict['rate_constants_parameters_lower_bnds'],
                Bayesian_dict['rate_constants_parameters_upper_bnds'],
                Bayesian_dict['rate_constants_parameters_bnds_exist'],
                return_unbounded_indices=True
                )
            #remove the unbounded values.
            Bayesian_dict['pars_initial_guess_truncated'] = optimize.CheKiPEUQ_from_Frhodo.remove_unbounded_values(Bayesian_dict['pars_initial_guess'], Bayesian_dict['unbounded_indices'] )
            Bayesian_dict['pars_lower_bnds_truncated'] = optimize.CheKiPEUQ_from_Frhodo.remove_unbounded_values( Bayesian_dict['pars_lower_bnds'], Bayesian_dict['unbounded_indices'] )
            Bayesian_dict['pars_upper_bnds_truncated'] = optimize.CheKiPEUQ_from_Frhodo.remove_unbounded_values( Bayesian_dict['pars_upper_bnds'], Bayesian_dict['unbounded_indices'] )
            Bayesian_dict['pars_bnds_exist_truncated'] = optimize.CheKiPEUQ_from_Frhodo.remove_unbounded_values( Bayesian_dict['pars_bnds_exist'], Bayesian_dict['unbounded_indices'] )


            
    
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
        log_opt_rates = s*self.x0
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
        
        loss_resid = np.array(output_dict['loss'])

        if np.size(loss_resid) == 1:  # optimizing single experiment
            c = 0
            loss_exp = loss_resid
        else:                   # optimizing multiple experiments
            c = outlier(loss_resid, a=self.opt_settings['loss_alpha'], c=self.opt_settings['loss_c'])
            loss_exp = generalized_loss_fcn(loss_resid, a=self.opt_settings['loss_alpha'], c=c*0.1) # I find that the loss function doesn't do much unless c is reduced further
        
        if self.opt_settings['obj_fcn_type'] == 'Residual':
            obj_fcn = np.mean(loss_exp*loss_resid.max()/loss_exp.max())

        elif self.opt_settings['obj_fcn_type'] == 'Bayesian':
            import optimize.CheKiPEUQ_from_Frhodo
            Bayesian_dict = self.Bayesian_dict                
            Bayesian_dict['rate_constants_current_guess'] = deepcopy(log_opt_rates)
            Bayesian_dict['rate_constants_parameters_current_guess'] = deepcopy(x)
            print("line 397, rate_constants_current_guess", Bayesian_dict['rate_constants_current_guess'])
            print("line 397, rate_constants_parameters_current_guess", Bayesian_dict['rate_constants_parameters_current_guess'])
            Bayesian_dict['pars_current_guess'] = np.concatenate( (Bayesian_dict['rate_constants_current_guess'], Bayesian_dict['rate_constants_parameters_current_guess'] ) )
            Bayesian_dict['last_obs_sim_interp'] = np.concatenate(output_dict['obs_sim_interp'], axis=0)
            Bayesian_dict['observed_data'] = np.concatenate(output_dict['obs_exp'], axis=0)
            Bayesian_dict['observed_data_lower_bounds'] = []
            Bayesian_dict['observed_data_upper_bounds'] = []
            def get_last_obs_sim_interp(varying_rate_vals): #A. Savara added this. It needs to be here.
                try:
                    last_obs_sim_interp = Bayesian_dict['last_obs_sim_interp']
                    last_obs_sim_interp = np.array(last_obs_sim_interp).T
                except:
                    print("this is line 422! There may be an error occurring!")
                    last_obs_sim_interp = None
                return last_obs_sim_interp
            Bayesian_dict['simulation_function'] = get_last_obs_sim_interp #using wrapper that just returns the last_obs_sim_interp
            
            if np.size(loss_resid) == 1:  # optimizing single experiment
                Bayesian_dict['weights_data'] = np.array(output_dict['aggregate_weights'], dtype=object)
            else:
                aggregate_weights = np.array(output_dict['aggregate_weights'], dtype=object)
                exp_loss_weights = loss_exp/generalized_loss_fcn(loss_resid) # comparison is between selected loss fcn and SSE (L2 loss)
                Bayesian_dict['weights_data'] = np.concatenate(aggregate_weights*exp_loss_weights, axis=0)
            
            #Bayesian_dict['weights_data'] /= np.max(Bayesian_dict['weights_data'])  # if we want to normalize by maximum

            #for val in Bayesian_dict['weights_data']:
            #    print(val)

           
            #Step 3 of Bayesian:  create a CheKiPEUQ_PE_Object (this is a class object)
            #NOTE: normally, the Bayesian object would be created earlier. However, we are using a non-standard application
            #where the observed_data uncertainties might change with each simulation.
            #To gain efficiency, we could cheat and se the feature get_responses_simulation_uncertainties function of CheKiPEUQ, 
            #or could create a new get_responses_observed_uncertainties function in CheKiPEUQ
            #for now we will just create a new PE_object each time.
            print("line 443", np.shape(Bayesian_dict['observed_data']))
            print("line 443", np.shape(Bayesian_dict['weights_data']))
            if np.shape(Bayesian_dict['weights_data']) != np.shape(Bayesian_dict['observed_data']):
                sys.exit()
            
            CheKiPEUQ_PE_object = optimize.CheKiPEUQ_from_Frhodo.load_into_CheKiPUEQ(
                simulation_function=    Bayesian_dict['simulation_function'],
                observed_data=          Bayesian_dict['observed_data'],
                pars_initial_guess =    Bayesian_dict['pars_initial_guess_truncated'],
                pars_lower_bnds =       Bayesian_dict['pars_lower_bnds_truncated'],   
                pars_upper_bnds =       Bayesian_dict['pars_upper_bnds_truncated'],   
                pars_bnds_exist =       Bayesian_dict['pars_bnds_exist_truncated'],
                observed_data_lower_bounds= Bayesian_dict['observed_data_lower_bounds'],
                observed_data_upper_bounds= Bayesian_dict['observed_data_upper_bounds'],
                weights_data=               Bayesian_dict['weights_data'],
                pars_uncertainty_distribution=  Bayesian_dict['pars_uncertainty_distribution'],
                num_rate_constants_and_rate_constant_parameters = [len(Bayesian_dict['rate_constants_initial_guess']), len(Bayesian_dict['rate_constants_parameters_initial_guess'])]
                ) #this is assigned in the "__init__" function up above.
            #Step 4 of Bayesian:  call a function to get the posterior density which will be used as the objective function.
            #We need to provide the current values of the varying_rate_vals to feed into the function.
            #print("line 406", varying_rate_vals_indices)
            
            
            #varying_rate_vals = np.array(output_dict['shock']['rate_val'])[list(varying_rate_vals_indices)] #when extracting a list of multiple indices, instead of array[index] one use array[[indices]]
            Bayesian_dict['pars_current_guess_truncated'] = optimize.CheKiPEUQ_from_Frhodo.remove_unbounded_values(Bayesian_dict['pars_current_guess'], Bayesian_dict['unbounded_indices'] ) 
            
            log_posterior_density = optimize.CheKiPEUQ_from_Frhodo.get_log_posterior_density(CheKiPEUQ_PE_object, Bayesian_dict['pars_current_guess_truncated'])
            #Step 5 of Bayesian:  return the objective function and any other metrics desired.
            obj_fcn = -1*log_posterior_density #need neg_logP because minimizing.
            print("line 481 of fit_fcn, Bayesian obj_fcn", obj_fcn)
           
        # For updating
        self.i += 1
        if not optimizing or self.i % 1 == 0:#5 == 0: # updates plot every 5
            if obj_fcn == 0:
                obj_fcn = np.inf
            
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
                      'obj_fcn': obj_fcn, 'stat_plot': stat_plot, 
                      'x': x, 'coef_opt': self.coef_opt, 
                      'ind_var': display_ind_var, 'observable': display_observable}
            
            self.signals.update.emit(update)
                
        if optimizing:
            return obj_fcn
        else:
            return obj_fcn, x, output_dict['shock']
            
    def fit_all_coeffs(self, all_rates):      
        coeffs = []
        i = 0
        for rxn_coef in self.rxn_coef_opt:
            rxnIdx = rxn_coef['rxnIdx']
            T, P, X = rxn_coef['T'], rxn_coef['P'], rxn_coef['X']
            coef_x0 = rxn_coef['coef_x0']
            coef_bnds = [rxn_coef['coef_bnds']['lower'], rxn_coef['coef_bnds']['upper']]
            rxn_rates = all_rates[i:i+len(T)]
            if len(coeffs) == 0:
                coeffs = fit_coeffs(rxn_rates, T, P, X, rxn_coef['coefName'], rxnIdx, coef_x0, coef_bnds, self.mech)
                if coeffs is None:
                    return
            else:
                coeffs_append = fit_coeffs(rxn_rates, T, P, X, rxn_coef['coefName'], rxnIdx, coef_x0, coef_bnds, self.mech)
                if coeffs_append is None:
                    return
                coeffs = np.append(coeffs, coeffs_append)
            
            i += len(T)

        return coeffs