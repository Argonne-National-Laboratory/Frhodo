#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This file is part of Frhodo. Copyright © 2020, UChicago Argonne, LLC
# and licensed under BSD-3-Clause. See License.txt in the top-level 
# directory for license and copyright information.

import os, sys, platform, multiprocessing, pathlib, ctypes
# os.environ['QT_API'] = 'pyside2'        # forces pyside2
from typing import Tuple, Optional, List

from qtpy.QtWidgets import QMainWindow, QApplication, QMessageBox
from qtpy import uic, QtCore, QtGui
import numpy as np

from .plot.plot_main import All_Plots as plot
from .misc_widget import MessageWindow
from .calculate import mech_fcns, reactors, convert_units
from .version import __version__
from . import appdirs, options_panel_widgets, sim_explorer_widget
from . import settings, config_io, save_widget, error_window, help_menu

if os.environ['QT_API'] == 'pyside2': # Silence warning: "Qt WebEngine seems to be initialized from a plugin."
    QApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)

# Handle high resolution displays:  Minimum recommended resolution 1280 x 960
if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
    QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
    QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

# set main folder
path = {'main': pathlib.Path(__file__).parents[0].resolve()}

# set appdata folder using AppDirs library (but just using the source code file)
dirs = appdirs.AppDirs(appname='Frhodo', roaming=True, appauthor=False)
path['appdata'] = pathlib.Path(dirs.user_config_dir)
path['appdata'].mkdir(parents=True, exist_ok=True)  # Make path if it doesn't exist
shut_down = {'bool': False}


class Main(QMainWindow):
    def __init__(self, app: QApplication, path: dict, skip_config: bool = True):
        """Launch the Frhodo GUI application

        Args:
            app: Link the QTApplication
            path: Collection of useful paths
            skip_config: Skip loading in the default configuration. Primarily useful if running Frhodo from API
        """
        super().__init__()
        self.app = app

        # Create an object that will facilitate loading data from disk
        self.path_set = settings.Path(self, path)  # Also loads in mechanism data according to default configuration
        uic.loadUi(str(self.path['main']/'UI'/'main_window.ui'), self)  # ~0.4 sec
        self.splitter.moveSplitter(0, 1)    # moves splitter 0 as close to 1 as possible
        self.setWindowIcon(QtGui.QIcon(str(self.path['main']/'UI'/'graphics'/'main_icon.png')))

        # Start threadpools
        self.threadpool = QtCore.QThreadPool()
        self.threadpool.setMaxThreadCount(2)  # Sets thread count to 1 (1 for gui - this is implicit, 1 for calc)

        # Set selected tabs
        for tab_widget in [self.option_tab_widget, self.plot_tab_widget]:
            tab_widget.setCurrentIndex(0)

        # Set Clipboard
        self.clipboard = QApplication.clipboard()

        # Initialize the panels that manage experimental data
        self.var = {'reactor': {'t_unit_conv': 1}}  # `var`, like `parent`, is a catch-all variable storage
        self.SIM = reactors.Simulation_Result()
        self.mech_loaded = False
        self.run_block = True
        self.convert_units = convert_units.Convert_Units(self)
        self.series = settings.series(self)

        # Initialize the panels that display optimization results
        self.sim_explorer = sim_explorer_widget.SIM_Explorer_Widgets(self)
        self.plot = plot(self)

        # Create the options panel(s) and loads in the experimental data
        options_panel_widgets.Initialize(self)

        # Create storage for the chemical mechanism data
        self.mech = mech_fcns.Chemical_Mechanism()

        # Setup save sim
        self.save_sim = save_widget.Save_Dialog(self)
        self.save_sim_button.clicked.connect(self.save_sim.execute)
        self.action_Save.triggered.connect(self.save_sim.execute)

        if shut_down['bool']:
            sys.exit()
        else:
            self.show()
            self.app.processEvents()  # allow everything to draw properly

        # Initialize Settings
        self.initialize_settings(skip_config)  # ~ 4 sec

        # Setup help menu
        self.version = __version__
        help_menu.HelpMenu(self)

    def initialize_settings(self, skip_config: bool):  # TODO: Solving for loaded shock twice
        """Load in the user-defined settings from configuration files on disk

        Args:
            skip_config: Skip loading the default configuration
        """
        msgBox = MessageWindow(self, 'Loading...')
        self.app.processEvents()

        self.var['old_shock_choice'] = self.var['shock_choice'] = 1

        # Load in the UI selections for the last box
        self.user_settings = config_io.GUI_settings(self)
        if skip_config:
            self.path['default_config'] = None  # Will result in configuration loading to fail
        self.user_settings.load()

        # Whether to load >1 files from the
        self.load_full_series = self.load_full_series_box.isChecked()   # TODO: Move to somewhere else?

        # load previous paths if file in path, can be accessed, and is a file
        if ('path_file' in self.path and os.access(self.path['path_file'], os.R_OK) and
            self.path['path_file'].is_file()):

            self.path_set.load_dir_file(self.path['path_file']) # ~3.9 sec

        self.update_user_settings()
        self.run_block = False      # Block multiple simulations from running during initialization
        self.run_single()           # Attempt simulation after initialization completed
        msgBox.close()

    def load_mech(self, event = None):
        """Load in the mechanism data from disk"""
        def mechhasthermo(mech_path):
            f = open(mech_path, 'r')
            while True:
                line = f.readline()
                if '!' in line[0:2]:
                    continue
                if 'ther' in line.split('!')[0].strip().lower():
                    return True

                if not line:
                    break

            f.close()
            return False

        if self.mech_select_comboBox.count() == 0: return   # if no items return, unsure if this is needed now

        # Specify mech file path
        self.path['mech'] = self.path['mech_main'] / str(self.mech_select_comboBox.currentText())
        if not self.path['mech'].is_file(): # if it's not a file, then it was deleted
            self.path_set.mech()            # update mech pulldown choices
            return

        # Check use thermo box viability
        if mechhasthermo(self.path['mech']):
            if self.thermo_select_comboBox.count() == 0:
                self.use_thermo_file_box.setDisabled(True) # disable checkbox if no thermo in mech file
            else:
                self.use_thermo_file_box.setEnabled(True)
            # Autoselect checkbox off if thermo exists in mech
            if self.sender() is None or 'use_thermo_file_box' not in self.sender().objectName():
                self.use_thermo_file_box.blockSignals(True)           # stop set from sending signal, causing double load
                self.use_thermo_file_box.setChecked(False)
                self.use_thermo_file_box.blockSignals(False)          # allow signals again
        else:
            self.use_thermo_file_box.blockSignals(True)           # stop set from sending signal, causing double load
            self.use_thermo_file_box.setChecked(True)
            self.use_thermo_file_box.blockSignals(False)          # allow signals again
            self.use_thermo_file_box.setDisabled(True) # disable checkbox if no thermo in mech file

        # Enable thermo select based on use_thermo_file_box
        if self.use_thermo_file_box.isChecked():
            self.thermo_select_comboBox.setEnabled(True)
        else:
            self.thermo_select_comboBox.setDisabled(True)

        # Specify thermo file path        
        if self.use_thermo_file_box.isChecked():
            if self.thermo_select_comboBox.count() > 0:
                self.path['thermo'] = self.path['mech_main'] / str(self.thermo_select_comboBox.currentText())
            else:
                self.log.append('Error loading mech:\nNo thermodynamics given', alert=True)
                return
        else:
            self.path['thermo'] = None

        # Initialize Mechanism
        self.log.clear([])  # Clear log when mechanism changes to avoid log errors about prior mech
        mech_load_output = self.mech.load_mechanism(self.path)
        self.log.append(mech_load_output['message'], alert=not mech_load_output['success'])

        self.mech_loaded = mech_load_output['success']

        if not mech_load_output['success']:   # if error: update species and return
            self.mix.update_species()
            self.log._blink(True)   # updating_species is causing blink to stop due to successful shock calculation
            return

        # Initialize tables and trees
        self.tree.set_trees(self.mech)
        self.mix.update_species()       # this was commented out, could be because multiple calls to solver from update_mix / setItems

        # Update the appropriate display tab
        tabIdx = self.plot_tab_widget.currentIndex()
        tabText = self.plot_tab_widget.tabText(tabIdx)
        if tabText == 'Signal/Sim':
            # Force observable_widget to update
            observable = self.plot.observable_widget.widget['main_parameter'].currentText()
            self.plot.observable_widget.widget['main_parameter'].currentIndexChanged[str].emit(observable)
        elif tabText == 'Sim Explorer': # TODO: This gets called twice?
            self.sim_explorer.update_all_main_parameter()

    def shock_choice_changed(self, event):
        if 'exp_main' in self.directory.invalid:    # don't allow shock change if problem with exp directory
            return

        self.var['old_shock_choice'] = self.var['shock_choice']
        self.var['shock_choice'] = event

        self.shockRollingList = ['P1', 'u1']    # reset rolling list
        self.rxn_change_history = []  # reset tracking of rxn numbers changed

        if not self.optimize_running:
            self.log.clear([])
        self.series.change_shock()  # link display_shock to correct set and 

    def update_user_settings(self, event = None):
        # This is one is located on the Files tab
        shock = self.display_shock
        self.series.set('series_name', self.exp_series_name_box.text())

        t_unit_conv = self.var['reactor']['t_unit_conv']
        if self.time_offset_box.value()*t_unit_conv != shock['time_offset']: # if values are different
            self.series.set('time_offset', self.time_offset_box.value()*t_unit_conv)
            if hasattr(self.mech_tree, 'rxn'):              # checked if copy valid in function
                self.tree._copy_expanded_tab_rates()        # copy rates and time offset

        self.var['time_unc'] = self.time_unc_box.value()*t_unit_conv

        # self.user_settings.save()   # saves settings everytime a variable is changed

        if event is not None:
            sender = self.sender().objectName()
            if 'time_offset' in sender and hasattr(self, 'SIM'): # Don't rerun SIM if it exists
                if hasattr(self.SIM, 'independent_var') and hasattr(self.SIM, 'observable'):
                    self.plot.signal.update_sim(self.SIM.independent_var, self.SIM.observable)
            elif any(x in sender for x in ['end_time', 'sim_interp_factor', 'ODE_solver', 'rtol', 'atol']):
                self.run_single()
            elif self.display_shock['exp_data'].size > 0: # If exp_data exists
                self.plot.signal.update(update_lim=False)
                self.plot.signal.canvas.draw()
        '''
        # debug
        for i in self.var:
            print('key: {:<14s} value: {:<16s}'.format(i, str(self.var[i])))
        '''

    def keyPressEvent(self, event): pass
        # THIS IS NOT FULLY FUNCTIONING
        # http://ftp.ics.uci.edu/pub/centos0/ics-custom-build/BUILD/PyQt-x11-gpl-4.7.2/doc/html/qkeyevent.html
        # print(event.modifiers(),event.text())

    def run_single(self, event=None, t_save=None, rxn_changed=False):
        if self.run_block: return
        if not self.mech_loaded: return                 # if mech isn't loaded successfully, exit
        if not hasattr(self.mech_tree, 'rxn'): return   # if mech tree not set up, exit

        shock = self.display_shock

        # Get the conditions of the current reactor
        T_reac, P_reac, mix = shock['T_reactor'], shock['P_reactor'], shock['thermo_mix']

        # Make sure the rate constants are update-to-date with the conditions on the table and specified shock
        self.tree.update_rates()

        # calculate all properties or observable by sending t_save
        tabIdx = self.plot_tab_widget.currentIndex()
        tabText = self.plot_tab_widget.tabText(tabIdx)
        if tabText == 'Sim Explorer':
            t_save = np.array([0])

        # Formulate the output arguments for the
        SIM_kwargs = {'u_reac': shock['u2'], 'rho1': shock['rho1'], 'observable': self.display_shock['observable'],
            't_lab_save': t_save, 'sim_int_f': self.var['reactor']['sim_interp_factor'],
            'ODE_solver': self.var['reactor']['ode_solver'],
            'rtol': self.var['reactor']['ode_rtol'], 'atol': self.var['reactor']['ode_atol']}

        if '0d Reactor' in self.var['reactor']['name']:
            SIM_kwargs['solve_energy'] = self.var['reactor']['solve_energy']
            SIM_kwargs['frozen_comp'] = self.var['reactor']['frozen_comp']

        self.SIM, verbose = self.mech.run(self.var['reactor']['name'], self.var['reactor']['t_end'],
                                          T_reac, P_reac, mix, **SIM_kwargs)

        if verbose['success']:
            self.blink = self.log._blink(False)
        else:
            self.log.append(verbose['message'])

        if self.SIM is not None:
            self.plot.signal.update_sim(self.SIM.independent_var, self.SIM.observable, rxn_changed)
            if tabText == 'Sim Explorer':
                self.sim_explorer.populate_main_parameters()
                self.sim_explorer.update_plot(self.SIM) # sometimes duplicate updates
        else:
            nan = np.array([np.nan, np.nan])
            self.plot.signal.update_sim(nan, nan)   # make sim plot blank
            if tabText == 'Sim Explorer':
                self.sim_explorer.update_plot(None)
            return # If mech error exit function

    # def raise_error(self):
        # assert False


def launch_gui(args: Optional[List[str]] = None, fresh_gui: bool = False) -> Tuple[QApplication, Main]:
    """Launch the GUI

    Args:
        args: Arguments to pass to the QApplication. Use ``sys.argv`` by default
        fresh_gui: Whether to skip loading the previous configurations
    Returns:
        - The QApplication instance
        - Link to the main window
    """
    # Get the default argument if none specified
    if args is None:
        args = sys.argv.copy()

    if platform.system() == 'Windows':  # this is required for pyinstaller on windows
        multiprocessing.freeze_support()

        if getattr(sys, 'frozen', False):  # if frozen minimize console immediately
            ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 0)

    # Make a copy of the default paths, so that multiple launches of the Frhodo do not interfere
    #   WardLT: I added this while building unit tests, which involve launching Frhodo many times
    my_path = path.copy()

    # Launch the QT application
    app = QApplication(args)
    sys.excepthook = error_window.excepthookDecorator(app, my_path, shut_down)

    # Create the Frhodo main window
    main_window = Main(app, my_path, fresh_gui)
    return app, main_window


def main():
    """Launch the GUI and then block until it finishes"""
    # Launch the application
    app, main_window = launch_gui()

    # Pass the exit code forward
    sys.exit(app.exec_())
