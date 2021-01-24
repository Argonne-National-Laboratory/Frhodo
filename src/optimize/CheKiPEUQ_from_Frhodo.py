import numpy as np
import sys; sys.path.append('../../'); import os
sys.path.append(os.path.join(os.path.dirname(__file__))) #This is so that CheKiPEUQ_local can be found properly. Otherwise there are problems importing CKPQ.parameter_estimation

# try:
    # import CheKiPEUQ as CKPQ
    # print("line 5 it is importing CheKiPEUQ!")
# except:
    # import os #The below lines are to allow CheKiPEUQ_local to be called regardless of user's working directory.
    # lenOfFileName = len(os.path.basename(__file__)) #This is the name of **this** file.
    # absPathWithoutFileName = os.path.abspath(__file__)[0:-1*lenOfFileName]
    # sys.path.append(absPathWithoutFileName)
    # import optimize.CheKiPEUQ_local as CKPQ #might need to put Frhodo.CheKiPEUQ_local or something like that.
    # # compare to C:\Users\fvs\Documents\GitHub\CheKiPEUQ\CheKiPEUQ\InverseProblem.py
    # print("line 14 it is in the except statement of importing CheKiPEUQ!")
    
lenOfFileName = len(os.path.basename(__file__)) #This is the name of **this** file.
absPathWithoutFileName = os.path.abspath(__file__)[0:-1*lenOfFileName]
sys.path.append(absPathWithoutFileName)
import optimize.CheKiPEUQ_local as CKPQ #might need to put Frhodo.CheKiPEUQ_local or something like that.
    
try:
    import CiteSoft
except:
    import os #The below lines are to allow CiteSoftLocal to be called regardless of user's working directory.
    lenOfFileName = len(os.path.basename(__file__)) #This is the name of **this** file.
    absPathWithoutFileName = os.path.abspath(__file__)[0:-1*lenOfFileName]
    sys.path.append(absPathWithoutFileName)
    import CiteSoftLocal as CiteSoft

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
    try:
        import CheKiPEUQ.UserInput as UserInput
        print("line 44, somehow imported CheKiPEUQ UserInput!!!")
    except:
        import optimize.CheKiPEUQ_local.UserInput as UserInput
    #TODO: put a "clear UserInput" type call here to UnitTesterSG_local
    
    if len(num_rate_constants_and_rate_constant_parameters) == 0: num_rate_constants_and_rate_constant_parameters = [len(pars_initial_guess), 0]
    
    UserInput.responses['responses_abscissa'] = []
    UserInput.responses['responses_observed'] = np.array(observed_data).T
    num_responses = np.shape(UserInput.responses['responses_observed'])[0]
    UserInput.responses['responses_observed_uncertainties'] = []
    if len(observed_data_lower_bounds) > 0: #assume that both lower and upper bounds exist on data if there is a lower bounds array provided.
        UserInput.responses['responses_observed_uncertainties'] = extract_larger_delta_and_make_sigma_values(UserInput.responses['responses_observed'], observed_data_lower_bounds, observed_data_upper_bounds, sigma_multiple)   
    try:
        weights_data = np.atleast_2d(weights_data).T
        UserInput.responses['responses_observed_weighting'] = (weights_data*np.ones(num_responses)).T
    except:
        print("There was an error in the weightings in CheKiPEUQ_from_Frhodo processing.")
    UserInput.model['InputParameterPriorValues'] = pars_initial_guess
    print("line 72 pars_uncertainty_distribution", pars_uncertainty_distribution)
    if pars_uncertainty_distribution.lower() == 'uniform': #make an array of -1 for uncertainties to signify a uniform distribution.
        UserInput.model['InputParametersPriorValuesUncertainties'] = -1*np.ones(len(pars_initial_guess))
    if pars_uncertainty_distribution.lower() == 'gaussian': 
        UserInput.model['InputParametersPriorValuesUncertainties'] = extract_larger_delta_and_make_sigma_values(pars_initial_guess, pars_lower_bnds, pars_upper_bnds, sigma_multiple)
    if pars_uncertainty_distribution.lower() == 'automatic' or pars_uncertainty_distribution.lower() == 'auto': 
        num_rate_constants = num_rate_constants_and_rate_constant_parameters[0] 
        num_rate_constants_parameters = num_rate_constants_and_rate_constant_parameters[1]
        rate_constant_uncertainties = -1*np.ones(num_rate_constants_parameters) #by default, use uniform for the rate_constant_uncertainties (signified by -1).
        print("line 81 pars_initial_guess", pars_initial_guess, num_rate_constants)
        rate_constant_parameters_uncertainties = extract_larger_delta_and_make_sigma_values(pars_initial_guess[num_rate_constants+0:], pars_lower_bnds[num_rate_constants+0:], pars_upper_bnds[num_rate_constants+0:], sigma_multiple)  #this returns a 1 sigma value for a gaussian, assuming that the range indicates a certain sigma_multiple in each direction. The "+0" is to start at next value with array indexing. Kind of like "-1 +1".
        UserInput.model['InputParametersPriorValuesUncertainties'] = np.concatenate( (rate_constant_uncertainties, rate_constant_parameters_uncertainties) )
        print("ilne 81 InputParametersPriorValuesUncertainties ",  UserInput.model['InputParametersPriorValuesUncertainties'])
    print("line 82", UserInput.model['InputParametersPriorValuesUncertainties'])
    if len(pars_bnds_exist)> 1: #If this is not a blank list, we're going to check each entry. For anything which has a "False", we are going to set the InputParametersPriorValuesUncertainties value to "-1" to indicate uniform since that means it can't be a Gaussian.
        for exist_index, lower_upper_booleans in enumerate(pars_bnds_exist):
            print("line 85", exist_index, lower_upper_booleans, np.sum(lower_upper_booleans))
            if np.sum(lower_upper_booleans) < 2: #True True will add to 2, anything else does not pass.
                UserInput.model['InputParametersPriorValuesUncertainties'][exist_index] = -1
    print("line 86", UserInput.model['InputParametersPriorValuesUncertainties']) 
    
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
    print("line 61", CKPQ.frog)
    PE_object = CKPQ.parameter_estimation(UserInput)
    return PE_object

def get_log_posterior_density(PE_object, parameters):
    return PE_object.getLogP(parameters)

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
    print("line 167, unbounded_indices", unbounded_indices, array_to_truncate)
    truncated_array = np.delete(array_to_truncate, unbounded_indices, axis=0)
    print("line 168, unbounded_indices", unbounded_indices, truncated_array)
    return truncated_array
