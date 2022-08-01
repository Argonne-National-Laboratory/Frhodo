# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 16:10:05 2019

@author: fvs
"""
import sys
sys.path.insert(0, '/mumpce/')
import mumpce.Project as mumpceProject
import mumpce.solution as mumpceSolution
import numpy as np
import matplotlib
from matplotlib import cm #EAW 2020/01/07
#matplotlib.use('Agg') #EAW 2020/01/07

'''
THIS FILE IS TO DEMONSTRATE CUSTOM PLOTTING WITH MUMPCE SOLUTION OBJECTS
The method for using the custom plotting features is to make a dictionary named contour_settings_custom
and to pass this dictionary as an optional argument into either plot_pdfs or _single_pdf_plot
which are functions in the Project class from Project.py of mumpce.

#Only one or more key:item pair needs to be added into contour_settings_custom to turn it on.
The item added does not even need to be a real option.
#So for example:; contour_settings_custom['on'] = True #would turn it on.

The options are:
contour_settings_custom['zoom_std_devs'] = 2.5 #how zoomed in the image is.
contour_settings_custom['center_on'] = 'all' #can be 'all', 'prior' or 'posterior'. 
contour_settings_custom['fontsize'] #sets the fontsize for everything except the colorbars. Can be an integer or the word 'auto', or the word "None". Should change space_between_subplots if fontsize is changed. 
contour_settings_custom['space_between_subplots'] #Typically a value between 0.20 and 5.0. Set to 4.0 by default. Should be changed when font size is changed. Fontsize 'auto' tends to make small fonts which needs smaller values like 0.20.
contour_settings_custom["colorbars"] = True #can be true or false.
contour_settings_custom['num_pts_per_axis'] = 500 #This sets the resolution of the contours.
contour_settings_custom["cmap_levels"] = 4   #This is the number of contours.
contour_settings_custom['cmap_prior'] #define a color map directly
contour_settings_custom['cmap_posterior']  #define a color map directly
contour_settings_custom['colormap_prior_customized'] #accepts a string (matplotlib colormap names, like 'Greens') or a list of tuples with 0-to-1 and colornames to interpolate between. For example, the default right now is:  [(0,    '#00FFFF'),(1,    '#0000FF')]. The tuple could have 0, 0.7, and 1, for example. #colors can be obtained from: https://www.htmlcsscolor.com/hex/244162  
contour_settings_custom['colormap_posterior_customized'] #accepts a string (matplotlib colormap names, like 'Oranges') or a list of tuples with 0-to-1 and colornames to interpolate between. For example, the default right now is:  [(0,    '#FFFF00'),(1,    '#FF0000')]. The tuple could have 0, 0.7, and 1, for example. #colors can be obtained from: https://www.htmlcsscolor.com/hex/244162  
contour_settings_custom['contours_normalized'] = True #Makes the maximum 1, with the contours at 0.2, 0.4, 0.6., 0.8
contour_settings_custom['num_y_ticks'] #adusts number of y ticks (actually sets a maximum number of them). #num_y_ticks and num_x_ticks must be either a string ('auto') or an integer (such as 4, either without string or with integer casting like int('5')). This feature is recommended.  #Note that this is a *request* When it's not fulfilled exactly, the user can play with the number.
contour_settings_custom['num_x_ticks'] #adjusts number of x ticks (actually sets a maximum number of them). #num_y_ticks and num_x_ticks must be either a string ('auto') or an integer (such as 4, either without string or with integer casting like int('5')).This feature is recommended. #Note that this is a *request* When it's not fulfilled exactly, the user can play with the number.
contour_settings_custom['x_ticks'] #feed in an array of directly. Not recommended.
contour_settings_custom['y_ticks'] #feed in an array of directly. Not recommended.
contour_settings_custom['axis_limits'] #Feed in list of [x_min, x_max, y_min, y_max]. This is recommended for use.

'''

###THE BELOW LINES OF CODE SETUP THE DATA IN THE MUMPCE PROJECT OBJECT####
mumpceProjectObject = mumpceProject.Project()
mumpceProjectObject.model_parameter_info = np.array([{'parameter_number': 0, 'parameter_name': 'Parameter 0', 'parameter_value': 1.0},
 {'parameter_number': 1, 'parameter_name': 'Parameter 1', 'parameter_value': 1.0}, #The parameter values are not used in making these contour plots.
 {'parameter_number': 2, 'parameter_name': 'Parameter 2', 'parameter_value': 1.0},
 {'parameter_number': 3, 'parameter_name': 'Parameter 3', 'parameter_value': 1.0},
 {'parameter_number': 4, 'parameter_name': 'Parameter 4', 'parameter_value': 1.0},
 {'parameter_number': 5, 'parameter_name': 'Parameter 5', 'parameter_value': 1.0},
 {'parameter_number': 6, 'parameter_name': 'Parameter 6', 'parameter_value': 1.0}]) #This must be made into a numpy array.
mumpceProjectObject.active_parameters = np.array([0, 1, 2, 4, 6]) #this must be made into a numpy array.
#mumpceProjectObject.set_active_parameters = [0, 1, 2, 4, 6]
Posterior_mu_vector = np.array([-0.58888733,1.1200355, 0.00704044, -1.62385888,0.80439847]) #this must be made into a numpy array. #This will become solution.x
Posterior_cov_vector = np.array([[ 0.0148872,-0.01894579, -0.01047339,0.01325883,0.04734254],
 [-0.01894579,0.04284732, -0.00131389, -0.04801795, -0.04545703],
 [-0.01047339, -0.00131389,0.02343653,0.01588293, -0.05618226],
 [ 0.01325883, -0.04801795,0.01588293,0.08171972,0.00875017],
 [ 0.04734254, -0.04545703, -0.05618226,0.00875017,0.20669273]]) #This will become solution.cov. It does not need to be a numpy array, but we make it one for consistency.

Prior_mu_vector = np.array([-0.98888733,0.8200355, 0.01204044, -7.02385888,0.40439847])
Prior_cov_vector = 10*Posterior_cov_vector

mumpceSolutionsObject = mumpceSolution.Solution(Posterior_mu_vector, Posterior_cov_vector, initial_x=Prior_mu_vector, initial_covariance=Prior_cov_vector)
mumpceProjectObject.solution = mumpceSolutionsObject
mumpceProjectObject.pairsOfParameterIndices = [[0, 1], [1, 2],[3, 4]]


###THE BELOW LINES OF CODE MAKE EXAMPLE PLOTS WITH THE DEFAULT AND WITH THE NEW REVISED WAY####

#This makes the figures as originally programmed in mumpce, which assumes/requries the cov to be normalized to 1.
mumpceProjectObject.plot_pdfs(mumpceProjectObject.pairsOfParameterIndices)

#I have expanded the code to allow more versatility an optional argument called contour_settings_custom.
#It does not assume/require that things be normalized to 1.


contour_settings_custom = {} 
contour_settings_custom['figure_name']='mumpce_plots_02'
mumpceProjectObject.plot_pdfs(mumpceProjectObject.pairsOfParameterIndices, contour_settings_custom = contour_settings_custom)


contour_settings_custom={}
contour_settings_custom['figure_name']='mumpce_plots_03'
contour_settings_custom['fontsize'] = 'auto'
contour_settings_custom['num_y_ticks'] = 'auto'
contour_settings_custom['num_x_ticks'] = 'auto'
contour_settings_custom['colorbars'] = True
contour_settings_custom['space_between_subplots'] = 0.20
mumpceProjectObject.plot_pdfs(mumpceProjectObject.pairsOfParameterIndices, contour_settings_custom = contour_settings_custom)

contour_settings_custom = {}
contour_settings_custom['figure_name']='mumpce_plots_04'
contour_settings_custom['fontsize'] = 'auto'
contour_settings_custom['num_y_ticks'] = 3 #Note that this is a *request* When it's not fulfilled exactly, the user can play with the number.
contour_settings_custom['num_x_ticks'] = 3 #Note that this is a *request* When it's not fulfilled exactly, the user can play with the number.
contour_settings_custom['colormap_posterior_customized'] = "Oranges"
contour_settings_custom['colormap_prior_customized'] = "Greens"
contour_settings_custom['contours_normalized'] = False
contour_settings_custom['center_on'] = 'prior'
contour_settings_custom['colorbars'] = True
mumpceProjectObject.plot_pdfs(mumpceProjectObject.pairsOfParameterIndices, contour_settings_custom = contour_settings_custom)

contour_settings_custom = {}
contour_settings_custom['figure_name']='mumpce_plots_05'
contour_settings_custom['space_between_subplots'] = 0.50
mumpceProjectObject.plot_pdfs(mumpceProjectObject.pairsOfParameterIndices, contour_settings_custom = contour_settings_custom)

contour_settings_custom = {}
contour_settings_custom['figure_name']='mumpce_plots_06'
contour_settings_custom['colorbars'] = True
contour_settings_custom['space_between_subplots'] = 0.50
contour_settings_custom['zoom_std_devs'] = 4.5 #These are in units of standard deviations of the prior.
mumpceProjectObject.plot_pdfs(mumpceProjectObject.pairsOfParameterIndices, contour_settings_custom = contour_settings_custom)
