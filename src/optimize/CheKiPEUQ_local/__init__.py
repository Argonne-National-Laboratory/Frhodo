#Try to do things a bit like https://github.com/python/cpython/blob/master/Lib/collections/__init__.py
#Except instead of putting things here directly in __init__, we'll impor them so they are accessible by importing the module.
print("this is line 3 of the _init of CheKiPEUQ_local!!!!!")
import sys; import os
#sys.path.append(os.path.join(os.path.dirname(__file__), '../')) #if the directory one above is added to sys.path, then InverseProblem can be found for importing from.
from CheKiPEUQ_local.InverseProblem import *
#from CheKiPEUQ.plotting_functions import *
#from CheKiPEUQ.mumpce import *
from InverseProblem import parameter_estimation
print("this is line 7 of the _init of CheKiPEUQ_local!!!!!")
frog = "This is the frog string!"