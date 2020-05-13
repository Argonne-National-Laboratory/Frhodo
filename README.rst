.. Frhodo

|Frhodo|

What does Frhodo do?
================

Frhodo is an open-source, GUI-based Python application to simulate 
experimental data and optimize chemical kinetics mechanisms using `Cantera <https://cantera.org>`_ 
as its chemistry solver. Among other things, it can be used to:

* Intuitive GUI eases user workload
* Simulate chemical kinetics experiments using:

  * 0D closed, homogeneous, constant-volume reactor
  * 0D closed, homogeneous, constant-pressure reactor
  * Custom incident shock reactor for reactions behind incident shock waves
* Import any Cantera-valid mechanism (except YAML input format, for now)
* Read an experimental directory to quickly switch between experimental conditions and measured data
* Display simulated observable over experimental data
* Alter mechanisms within memory and update simulation automatically
* Investigate non-observable variables of simulation using Sim Explorer
* Optimize mechanism based upon obervables (by hand or by machine learning routine)

Installation and Documentation
============

Installation instructions and documentation can be found in the provided `Manual <https://github.com/Argonne-National-Laboratory/Frhodo/blob/master/Doc/Manual.pdf>`_. 

Frhodo uses an Anaconda
environment and has been tested on Windows, macOS, and Linux.

.. |Frhodo| image:: https://github.com/Argonne-National-Laboratory/Frhodo/blob/master/Doc/Logo.png
    :target: https://github.com/Argonne-National-Laboratory/Frhodo/
    :alt: Frhodo logo
    :width: 100
    :align: middle

