import sys; sys.path.append('../../'); 
try:
    import CheKiPEUQ as CKPQ
except:
    import CheKiPEUQ_local as CKPQ #might need to put Frhodo.CheKiPEUQ_local or something like that.
    # compare to C:\Users\fvs\Documents\GitHub\CheKiPEUQ\CheKiPEUQ\InverseProblem.py
    
    
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

@CiteSoft.after_call_compile_consolidated_log()
@CiteSoft.module_call_cite(unique_id=software_unique_id, software_name=software_name, **software_kwargs)    
def load_into_CheKiPUEQ(simulation_function, observed_data, pars_initial_guess = [], pars_bnds=[], observed_data_bounds=[], weights_data=[]):
    #observed_data is an array of values of observed data (can be nested if there is more than one observable)
    #pars_bnds is the bounds of the parmaeters ('coefficents') and is in the format of _______
    #pars_initial_guess is the initial guess for the parameters ('coefficients')
    #weights_data is an optional array of values that matches observed data in length.
    try:
        import CheKiPEUQ.UserInput as UserInput
    except:
        import CheKiPEUQ_local.UserInput as UserInput
    clear UserInput:
    UserInput.responses['responses_abscissa'] = []
    UserInput.responses['responses_observed'] = observed_data
    UserInput.responses['responses_observed_uncertainties'] = ...
    UserInput.responses['responses_observed_weighting'] = weights_data
    UserInput.model['InputParameterPriorValues'] = ...
    UserInput.model['InputParametersPriorValuesUncertainties'] = ... #Make it an array of many -1 if want uniform distribution.
    UserInput.model['InputParameterPriorValues_upperBounds'] = ...
    UserInput.model['InputParameterPriorValues_lowerBounds'] = ...
    UserInput.model['simulateByInputParametersOnlyFunction'] = simulation_function
    PE_object = CKPQ.parameter_estimation(UserInput)
    return PE_object

def get_log_posterior_density(PE_object):
    return PE_object.getLogP(parameters)