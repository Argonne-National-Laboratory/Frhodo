import numpy as np
from copy import deepcopy

try:
    import CheKiPEUQ as CKPQ
except:
    #import pathlib
    #sys.path.append(pathlib.Path(__file__).parent.absolute()) # add directory of **this** file to path
    import frhodo.calculate.optimize.CheKiPEUQ_local as CKPQ
    
try:
    import CheKiPEUQ.UserInput as UserInput
except:
    from .CheKiPEUQ_local import UserInput

import frhodo.CiteSoftLocal as CiteSoft

#get things ready for CiteSoft entry...
software_name = "CheKiPEUQ Bayesian Parameter Estimation"
software_version = "1.0.0"
software_unique_id = "https://doi.org/10.1002/cctc.202000953"
software_kwargs = {"version": software_version, "author": ["Aditya Savara", "Eric A. Walker"], "doi": "https://doi.org/10.1002/cctc.202000953", "cite": "Savara, A. and Walker, E.A. (2020), CheKiPEUQ Intro 1: Bayesian Parameter Estimation Considering Uncertainty or Error from both Experiments and Theory. ChemCatChem. Accepted. doi:10.1002/cctc.202000953"} 

##### DISABLING CITESOFT EXPORTATION FOR NOW BECAUSE IT WAS BECOMING EXPORTED TOO MANY TIMES WITH MULTI-PROCESSING EVEN WHEN ONLY AS AN IMPORT_CITE.
#CiteSoft.import_cite(unique_id=software_unique_id, software_name=software_name, write_immediately=True, **software_kwargs)
#decorator CiteSoft.after_call_compile_consolidated_log()
#decorator CiteSoft.module_call_cite(unique_id=software_unique_id, software_name=software_name, **software_kwargs)
def load_into_CheKiPUEQ(simulation_function, observed_data, pars_initial_guess = [], pars_lower_bnds=[], pars_upper_bnds =[], pars_bnds_exist = [], observed_data_lower_bounds=[], observed_data_upper_bounds=[], weights_data=[], pars_uncertainty_distribution='automatic', sigma_multiple = 3.0, num_rate_constants_and_rate_constant_parameters=[]):
    #observed_data is an array of values of observed data (can be nested if there is more than one observable)
    #pars_lower_bnds and pars_upper_bnds are the bounds of the parameters ('coefficents') in absolute values.
    #  for 'uniform' distribution the bounds are taken directly. For Gaussian, the larger of the 2 deltas is taken and divided by 3 for sigma.
    # rate_constants_parameters_bnds_exist is an array-like with values like [True False] where the Booleans are about whether the parameter has a lower bound and upper bound, respectively. So there is one pair of booleans per parameter.
    #pars_initial_guess is the initial guess for the parameters ('coefficients')
    #weights_data is an optional array of values that matches observed data in length.
    #sigma_multiple is how many sigma the bounds are equal to (relative to mean).
    #pars_uncertainty_distribution allows 3 choices: automatic, gaussian, uniform.  Automatic gives 'uniform' to the rate constants and 'gaussian' to the rate_constant_parameters.
    #num_rate_constants_and_rate_constant_parameters  allows the 'automatic setting of pars_uncertainty_distribution to assign distribution types based on the par index. If not supplied, everything is assumed to be a rate_constant.
    
    #TODO: put a "clear UserInput" type call here to UnitTesterSG_local
    
    if len(num_rate_constants_and_rate_constant_parameters) == 0: num_rate_constants_and_rate_constant_parameters = [len(pars_initial_guess), 0]
    
    UserInput.responses['responses_abscissa'] = []
    UserInput.responses['responses_observed'] = np.array(observed_data).T
    num_responses = np.shape(UserInput.responses['responses_observed'])[0]
    UserInput.responses['responses_observed_uncertainties'] = []
    if len(observed_data_lower_bounds) > 0: #assume that both lower and upper bounds exist on data if there is a lower bounds array provided.
        UserInput.responses['responses_observed_uncertainties'] = extract_larger_delta_and_make_sigma_values(UserInput.responses['responses_observed'], observed_data_lower_bounds, observed_data_upper_bounds, sigma_multiple)   
    try:
        #weights_data = np.atleast_2d(weights_data).T
        UserInput.responses['responses_observed_weighting'] = weights_data #(weights_data*np.ones(num_responses)).T
    except:
        print("There was an error in the weightings in CheKiPEUQ_from_Frhodo processing.")
    UserInput.model['InputParameterPriorValues'] = pars_initial_guess
    if pars_uncertainty_distribution.lower() == 'uniform': #make an array of -1 for uncertainties to signify a uniform distribution.
        UserInput.model['InputParametersPriorValuesUncertainties'] = -1*np.ones(len(pars_initial_guess))
    if pars_uncertainty_distribution.lower() == 'gaussian': 
        UserInput.model['InputParametersPriorValuesUncertainties'] = extract_larger_delta_and_make_sigma_values(pars_initial_guess, pars_lower_bnds, pars_upper_bnds, sigma_multiple)
    if pars_uncertainty_distribution.lower() == 'automatic' or pars_uncertainty_distribution.lower() == 'auto': 
        num_rate_constants = num_rate_constants_and_rate_constant_parameters[0] 
        num_rate_constants_parameters = num_rate_constants_and_rate_constant_parameters[1]
        rate_constant_uncertainties = -1*np.ones(num_rate_constants) #by default, use uniform for the rate_constant_uncertainties (signified by -1).
        rate_constant_parameters_uncertainties = extract_larger_delta_and_make_sigma_values(pars_initial_guess[num_rate_constants+0:], pars_lower_bnds[num_rate_constants+0:], pars_upper_bnds[num_rate_constants+0:], sigma_multiple)  #this returns a 1 sigma value for a gaussian, assuming that the range indicates a certain sigma_multiple in each direction. The "+0" is to start at next value with array indexing. Kind of like "-1 +1".
        UserInput.model['InputParametersPriorValuesUncertainties'] = np.concatenate( (rate_constant_uncertainties, rate_constant_parameters_uncertainties) )
    if len(pars_bnds_exist)> 1: #If this is not a blank list, we're going to check each entry. For anything which has a "False", we are going to set the InputParametersPriorValuesUncertainties value to "-1" to indicate uniform since that means it can't be a Gaussian.
        for exist_index, lower_upper_booleans in enumerate(pars_bnds_exist):
            #print("line 85", exist_index, lower_upper_booleans, np.sum(lower_upper_booleans))
            if np.sum(lower_upper_booleans) < 2: #True True will add to 2, anything else does not pass.
                UserInput.model['InputParametersPriorValuesUncertainties'][exist_index] = -1
    #print("line 86", UserInput.model['InputParametersPriorValuesUncertainties']) 
    
    #CheKiPEUQ cannot handle much larger than 1E100 for upper bounds.
    for upper_bound_index, upper_bound in enumerate(pars_upper_bnds):
        if upper_bound > 1.0E100:
            pars_upper_bnds[upper_bound_index] = 1.0E100

    #CheKiPEUQ cannot handle much more negative than -1E100 for lower bounds.
    for lower_bound_index, lower_bound in enumerate(pars_lower_bnds):
        if lower_bound < -1.0E100:
            pars_lower_bnds[lower_bound_index] = -1.0E100
    
    UserInput.model['InputParameterPriorValues_upperBounds'] = np.array(pars_upper_bnds)
    UserInput.model['InputParameterPriorValues_lowerBounds'] = np.array(pars_lower_bnds)
    UserInput.model['simulateByInputParametersOnlyFunction'] = simulation_function
    #print("line 61", CKPQ.frog)
    PE_object = CKPQ.parameter_estimation(UserInput)
    return PE_object

def get_log_posterior_density(PE_object, parameters):
    return PE_object.getLogP(parameters, runBoundsCheck=False)  # bounds are already checked prior to this. Differing methods were causing problems. No reason to make them match, just disable

#calculates delta between initial guess and bounds, takes the larger delta, and divides by sigma_multiple to return sigma.
def extract_larger_delta_and_make_sigma_values(initial_guess, lower_bound, upper_bound, sigma_multiple):
    #can probably be done faster with some kind of arrays and zipping, but for simple algorithm will use for loop and if statements.
    sigma_values = np.zeros(len(initial_guess)) #just initializing.
    for index in range(len(initial_guess)):
        upper_delta = np.abs(upper_bound[index]-initial_guess[index])
        lower_delta = np.abs(lower_bound[index]-initial_guess[index])
        max_delta = np.max([upper_delta,lower_delta])
        current_sigma = max_delta/np.float(sigma_multiple)
        sigma_values[index] = current_sigma
    return sigma_values
    
#This currently assumes all bounds are in pairs, but does not assume they are asymmetric.    
def get_varying_rate_vals_and_bnds(rate_vals, rate_bnds):
    #this takes in shock['rate_vals'] and and shock['rate_bnds'] returns the indices of the rate_vals that are varying, as well as their bounds.
    varying_rate_vals_indices = []
    varying_rate_vals_initial_guess = []
    varying_rate_vals_lower_bnds = []
    varying_rate_vals_upper_bnds = []
    for bounds_index, bounds_pair in enumerate(rate_bnds):
        if np.isnan(bounds_pair[0]) == False or np.isnan(bounds_pair[1]) == False: #this means there is a pair of bounds.
            varying_rate_vals_indices.append(bounds_index) #store this index.
            varying_rate_vals_initial_guess.append(rate_vals[bounds_index]) #append current rate_val
            varying_rate_vals_lower_bnds.append(rate_bnds[bounds_index][0]) #append current lower bound
            varying_rate_vals_upper_bnds.append(rate_bnds[bounds_index][1]) #append current upper bound
    return varying_rate_vals_indices, varying_rate_vals_initial_guess, varying_rate_vals_lower_bnds, varying_rate_vals_upper_bnds
    
def get_consolidated_parameters_arrays(rate_constants_values, rate_constants_lower_bnds, rate_constants_upper_bnds, rate_constants_bnds_exist, rate_constants_parameters_values, rate_constants_parameters_lower_bnds, rate_constants_parameters_upper_bnds, rate_constants_parameters_bnds_exist, return_unbounded_indices=True):
    #A. Savara recommends 'uniform' for rate constants and 'gaussian' for things like "log(A)" and "Ea"    
    #note that we use "pars_values" as the variable name because it can be pars_initial_guess or pars_current_guess and this makes the function more general.
    
    #we first start the arrays using the rate_constants arrays.
    pars_values = np.array(rate_constants_values).flatten()
    pars_lower_bnds = np.array(rate_constants_lower_bnds).flatten()
    pars_upper_bnds = np.array(rate_constants_upper_bnds).flatten()
    rate_constants_bnds_exist = np.array(rate_constants_bnds_exist, dtype = bool) #Can't flatten() because these have to be retained as pairs;
    
    #Now we concatenate those with the rate_constant_parameters arrays.
    pars_values = np.concatenate( (pars_values , np.array(rate_constants_parameters_values).flatten()) ) 
    pars_lower_bnds = np.concatenate( (pars_lower_bnds, np.array(rate_constants_parameters_lower_bnds).flatten()) )
    pars_upper_bnds = np.concatenate( (pars_upper_bnds, np.array(rate_constants_parameters_upper_bnds).flatten()) ) 
    pars_bnds_exist = np.concatenate( (rate_constants_bnds_exist, np.array(rate_constants_parameters_bnds_exist, dtype = bool) )) #Can't flatten() because these have to be retained as pairs; 
    
    unbounded_indices = []  #need to make this even if it will not be populated.
    if return_unbounded_indices==True:
        if len(pars_bnds_exist)> 1: #If this is not a blank list, we're going to check each entry. For anything which has a "False", we are going to set the InputParametersPriorValuesUncertainties value to "-1" to indicate uniform since that means it can't be a Gaussian.
            #pars_bnds_exist has values like [True False] for each parameter. [True False] would mean it has a lower bound but no upper bound.
            for exist_index, lower_upper_booleans in enumerate(pars_bnds_exist):
                if np.sum(lower_upper_booleans) < 2: #True True will add to 2, anything else does not pass.
                    unbounded_indices.append(exist_index)
        unbounded_indices = np.atleast_1d(np.array(unbounded_indices))
                
    return pars_values, pars_lower_bnds, pars_upper_bnds, pars_bnds_exist, unbounded_indices
    
def remove_unbounded_values(array_to_truncate, unbounded_indices):
    truncated_array = np.delete(array_to_truncate, unbounded_indices, axis=0)
    return truncated_array

class CheKiPEUQ_Frhodo_interface:
    def __init__(self, input_dict):
        #Step 1 of Bayesian:  Prepare any variables that need to be passed in for Bayesian PE_object creation.
        #Step 2 of Bayesian:  populate Bayesian_dict with any variables and uncertainties needed.
        self.Bayesian_dict = {}
        self.Bayesian_dict['pars_uncertainty_distribution'] = input_dict['opt_settings']['bayes_dist_type']  #options can be 'Auto', 'Gaussian' or 'Uniform'. 
        self.Bayesian_dict['rate_constants_initial_guess'] = deepcopy(input_dict['rxn_rate_opt']['x0'])
        self.Bayesian_dict['rate_constants_lower_bnds'] = deepcopy(input_dict['rxn_rate_opt']['bnds']['lower'])
        self.Bayesian_dict['rate_constants_upper_bnds'] = deepcopy(input_dict['rxn_rate_opt']['bnds']['upper'])
        num_rate_consants = len(self.Bayesian_dict['rate_constants_initial_guess'])
        self.Bayesian_dict['rate_constants_bnds_exist'] =  np.array(np.ones((num_rate_consants, 2)), dtype = bool) #From Jan 2021, we are setting [True True] for each rate_constant.

        self.Bayesian_dict['rate_constants_parameters_changing'] = deepcopy(input_dict['coef_opt'])
        self.Bayesian_dict['rate_constants_parameters_initial_guess'] = np.array([])
        self.Bayesian_dict['rate_constants_parameters_lower_bnds'] = np.array([])
        self.Bayesian_dict['rate_constants_parameters_upper_bnds'] = np.array([])
        self.Bayesian_dict['rate_constants_parameters_bnds_exist'] = []
        for rxn_coef in input_dict['rxn_coef_opt']:
            self.Bayesian_dict['rate_constants_parameters_initial_guess'] = np.concatenate((self.Bayesian_dict['rate_constants_parameters_initial_guess'], deepcopy(rxn_coef['coef_x0'])), axis=None)
            self.Bayesian_dict['rate_constants_parameters_lower_bnds'] = np.concatenate((self.Bayesian_dict['rate_constants_parameters_lower_bnds'], deepcopy(rxn_coef['coef_bnds']['lower'])), axis=None)
            self.Bayesian_dict['rate_constants_parameters_upper_bnds'] = np.concatenate((self.Bayesian_dict['rate_constants_parameters_upper_bnds'], deepcopy(rxn_coef['coef_bnds']['upper'])), axis=None)
            self.Bayesian_dict['rate_constants_parameters_bnds_exist'].extend(list(deepcopy(rxn_coef['coef_bnds']['exist']))) #we can't use append because this is a list of lists/array, and we want parallel construction.

        #self.Bayesian_dict['rate_constants_parameters_bnds_exist'] does not get flattened because it is a list of list/arrays (so we don't want it flattened).

        Bayesian_dict = self.Bayesian_dict            
        #concatenate all of the initial guesses and bounds. 
        Bayesian_dict['pars_initial_guess'], Bayesian_dict['pars_lower_bnds'],Bayesian_dict['pars_upper_bnds'], Bayesian_dict['pars_bnds_exist'], Bayesian_dict['unbounded_indices'] = get_consolidated_parameters_arrays( 
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
        # TODO: This might be improper way to handle all paramters having bounds
        if len(Bayesian_dict['unbounded_indices']) > 0:
            Bayesian_dict['pars_initial_guess_truncated'] = remove_unbounded_values(Bayesian_dict['pars_initial_guess'], Bayesian_dict['unbounded_indices'] )
            Bayesian_dict['pars_lower_bnds_truncated'] = remove_unbounded_values( Bayesian_dict['pars_lower_bnds'], Bayesian_dict['unbounded_indices'] )
            Bayesian_dict['pars_upper_bnds_truncated'] = remove_unbounded_values( Bayesian_dict['pars_upper_bnds'], Bayesian_dict['unbounded_indices'] )
            Bayesian_dict['pars_bnds_exist_truncated'] = remove_unbounded_values( Bayesian_dict['pars_bnds_exist'], Bayesian_dict['unbounded_indices'] )
        else:
            Bayesian_dict['pars_initial_guess_truncated'] = Bayesian_dict['pars_initial_guess']
            Bayesian_dict['pars_lower_bnds_truncated'] =  Bayesian_dict['pars_lower_bnds']
            Bayesian_dict['pars_upper_bnds_truncated'] =  Bayesian_dict['pars_upper_bnds']
            Bayesian_dict['pars_bnds_exist_truncated'] =  Bayesian_dict['pars_bnds_exist']

    def evaluate(self, CheKiPEUQ_eval_dict):
        Bayesian_dict = self.Bayesian_dict                
        Bayesian_dict['rate_constants_current_guess'] = deepcopy(CheKiPEUQ_eval_dict['log_opt_rates'])
        Bayesian_dict['rate_constants_parameters_current_guess'] = deepcopy(CheKiPEUQ_eval_dict['x'])
        Bayesian_dict['pars_current_guess'] = np.concatenate( (Bayesian_dict['rate_constants_current_guess'], Bayesian_dict['rate_constants_parameters_current_guess'] ) )
        Bayesian_dict['last_obs_sim_interp'] = np.concatenate(CheKiPEUQ_eval_dict['output_dict']['obs_sim_interp'], axis=0)
        Bayesian_dict['observed_data'] = np.concatenate(CheKiPEUQ_eval_dict['output_dict']['obs_exp'], axis=0)
        obs_data_bnds = np.concatenate(CheKiPEUQ_eval_dict['output_dict']['obs_bounds'], axis=0).T
        Bayesian_dict['observed_data_lower_bounds'] = obs_data_bnds[0]
        Bayesian_dict['observed_data_upper_bounds'] = obs_data_bnds[1]
        Bayesian_dict['weights_data'] = CheKiPEUQ_eval_dict['bayesian_weights']

        def get_last_obs_sim_interp(varying_rate_vals): #A. Savara added this. It needs to be here.
            try:
                last_obs_sim_interp = Bayesian_dict['last_obs_sim_interp']
                last_obs_sim_interp = np.array(last_obs_sim_interp).T
            except:
                print("this is line 422! There may be an error occurring!")
                last_obs_sim_interp = None
            return last_obs_sim_interp
            
        Bayesian_dict['simulation_function'] = get_last_obs_sim_interp #using wrapper that just returns the last_obs_sim_interp
           
        # Step 3 of Bayesian:  create a CheKiPEUQ_PE_Object (this is a class object)
        # NOTE: normally, the Bayesian object would be created earlier. However, we are using a non-standard application
        # where the observed_data uncertainties might change with each simulation.
        # To gain efficiency, we could cheat and se the feature get_responses_simulation_uncertainties function of CheKiPEUQ, 
        # or could create a new get_responses_observed_uncertainties function in CheKiPEUQ
        # for now we will just create a new PE_object each time.
        if np.shape(Bayesian_dict['weights_data']) != np.shape(Bayesian_dict['observed_data']):
            raise Exception('CheKiPEUQ Error: Shape of weights does not match observed data')
            
        num_rate_constants = len(Bayesian_dict['rate_constants_initial_guess'])
        num_rate_constant_parameters = len(Bayesian_dict['rate_constants_parameters_initial_guess'])

        CheKiPEUQ_PE_object = load_into_CheKiPUEQ(
            simulation_function =           Bayesian_dict['simulation_function'],
            observed_data =                 Bayesian_dict['observed_data'],
            pars_initial_guess =            Bayesian_dict['pars_initial_guess_truncated'],
            pars_lower_bnds =               Bayesian_dict['pars_lower_bnds_truncated'],   
            pars_upper_bnds =               Bayesian_dict['pars_upper_bnds_truncated'],   
            pars_bnds_exist =               Bayesian_dict['pars_bnds_exist_truncated'],
            observed_data_lower_bounds =    Bayesian_dict['observed_data_lower_bounds'],
            observed_data_upper_bounds =    Bayesian_dict['observed_data_upper_bounds'],
            weights_data =                  Bayesian_dict['weights_data'],
            pars_uncertainty_distribution = Bayesian_dict['pars_uncertainty_distribution'],
            num_rate_constants_and_rate_constant_parameters = [num_rate_constants, num_rate_constant_parameters]
            ) #this is assigned in the "__init__" function up above.
            
        # Step 4 of Bayesian:  call a function to get the posterior density which will be used as the objective function.
        # We need to provide the current values of the varying_rate_vals to feed into the function.
            
        #varying_rate_vals = np.array(output_dict['shock']['rate_val'])[list(varying_rate_vals_indices)] #when extracting a list of multiple indices, instead of array[index] one use array[[indices]]
        # TODO: This might be improper way to handle all paramters having bounds
        if len(Bayesian_dict['unbounded_indices']) > 0:
            Bayesian_dict['pars_current_guess_truncated'] = remove_unbounded_values(Bayesian_dict['pars_current_guess'], Bayesian_dict['unbounded_indices'] )
        else:
            Bayesian_dict['pars_current_guess_truncated'] = Bayesian_dict['pars_current_guess']
            
        log_posterior_density = get_log_posterior_density(CheKiPEUQ_PE_object, Bayesian_dict['pars_current_guess_truncated'])
        neg_log_posterior_density = -1*log_posterior_density # need neg_logP because minimizing.
            
        # Step 5 of Bayesian:  return the objective function and any other metrics desired.
        #obj_fcn = neg_log_posterior_density 
        if CheKiPEUQ_eval_dict['iteration_num'] == 0 and 'obj_fcn_initial' not in Bayesian_dict:
            Bayesian_dict['obj_fcn_initial'] = neg_log_posterior_density 
            obj_fcn = 0.0
        elif neg_log_posterior_density == np.inf:
            obj_fcn = np.inf
        else:
            obj_fcn = (neg_log_posterior_density - Bayesian_dict['obj_fcn_initial'])/np.abs(Bayesian_dict['obj_fcn_initial'])*100

        return obj_fcn