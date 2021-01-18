# Type 'pytest' in the current directory to run tests.
# EAW 2020/01/17
import plotting_functions
from plotting_functions import plotting_functions
import UserInput_ODE_KIN_BAYES_SG_EW as UserInput
import run_me
from run_me import ip

def test_mumpce_plots():
    plot_object = plotting_functions()
    assert plot_object.mumpce_plots(model_parameter_info = UserInput.model_parameter_info, active_parameters = UserInput.active_parameters, pairs_of_parameter_indices = UserInput.pairs_of_parameter_indices, posterior_mu_vector = UserInput.posterior_mu_vector, posterior_cov_matrix = UserInput.posterior_cov_matrix, prior_mu_vector = UserInput.prior_mu_vector, prior_cov_matrix = UserInput.prior_cov_matrix, contour_settings_custom = UserInput.contour_settings_custom) == None
    assert exec(open("mumpce_custom_plotting_example.py").read()) == None

def MH_then_scatterplot():
    parseUserInputParameters()
    ip_object = ip()
    [evidence, info_gain, samples, rate_tot_array, logP] = ip_object.MetropolisHastings()
    plotting_object = plotting_functions(samples=samples)
    assert plotting_object.seaborn_scatterplot_matrix() == None
