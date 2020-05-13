.. Frhodo

|Frhodo|

What does Frhodo do?
================

Frhodo is an open-source, GUI-based Python application to simulate 
experimental data and optimize chemical kinetics mechanisms using `Cantera <https://cantera.org>`_ 
as its chemistry solver. 

Features include:

* Easing user workload through an intuitive and extensive GUI
* Simulating chemical kinetics experiments using:

  * 0D closed, homogeneous, constant-volume reactor
  * 0D closed, homogeneous, constant-pressure reactor
  * Custom incident shock reactor for reactions behind incident shock waves
* Importing Cantera-valid mechanisms (YAML input format is currently not supported)
* Reading an experimental directory to quickly switch between experimental conditions and measured data
* Displaying simulated observable over experimental data
* Altering mechanisms within memory and update simulation automatically
* Investigating non-observable variables of simulation using the Sim Explorer within Frhodo
* Optimizing mechanism based upon obervables (by hand or by machine learning routine)

  * Automatic routine requires bounds on reaction rate constants
  * Automatic routine can optimize all three Arrhenius parameters

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

