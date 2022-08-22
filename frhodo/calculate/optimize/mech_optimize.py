# This file is part of Frhodo. Copyright © 2020, UChicago Argonne, LLC
# and licensed under BSD-3-Clause. See License.txt in the top-level 
# directory for license and copyright information.

import collections
from qtpy.QtCore import QThreadPool
import numpy as np
import cantera as ct
import multiprocessing as mp
from copy import deepcopy
from timeit import default_timer as timer

from scipy import stats

from ..optimize.optimize_worker import Worker
from ..optimize.fit_fcn import update_mech_coef_opt
from ..optimize.misc_fcns import rates, set_bnds
from ..optimize.fit_coeffs import fit_generic as Troe_fit

Ru = ct.gas_constant
default_arrhenius_coefNames = ['activation_energy', 'pre_exponential_factor', 'temperature_exponent']

class Multithread_Optimize:
    def __init__(self, parent):
        self.parent = parent

        # Initialize Threads
        parent.optimize_running = False
        parent.multiprocessing = True
        # parent.threadpool = QThreadPool()
        # parent.threadpool.setMaxThreadCount(2) # Sets thread count to 1 (1 for gui - this is implicit, 1 for calc)
        # log_txt = 'Multithreading with maximum {:d} threads\n'.format(parent.threadpool.maxThreadCount())
        # parent.log.append(log_txt, alert=False)

        # Set Distribution
        self.dist = stats.gennorm

        # Connect Toolbar Functions
        parent.action_Run.triggered.connect(self.start_threads)
        parent.action_Abort.triggered.connect(self.abort_workers)
     
    def start_threads(self):
        parent = self.parent

        # Prepare the name of the output file
        parent.path_set.optimized_mech()

        # Controls for the display
        self.last_plot_timer = 0.0
        self.time_between_plots = 0.0  # maximum update rate, updated based on time to plot

        # Check whether we should be using multiprocessing
        parent.multiprocessing = parent.multiprocessing_box.isChecked()

        ## Check fit_coeffs
        #from optimize.fit_coeffs import debug
        #debug(parent.mech)

        # Check if the optimization cannot start for some reason
        if parent.directory.invalid: 
            parent.log.append('Invalid directory found\n')
            return
        elif parent.optimize_running:
            parent.log.append('Optimize running flag already set to True\n')
            return
        elif not self._has_opt_rxns(): # if nothing to optimize, don't!
            parent.log.append('No reactions or coefficients set to be optimized\n')
            return         
                
        # Set shocks to be run
        self.shocks2run = []
        for series in parent.series.shock:
            for shock in series:
                # skip not included or exp_data not loaded from experiment
                if not shock['include'] or 'exp_data' in shock['err']: 
                    shock['SIM'] = None
                    continue

                self.shocks2run.append(shock)

        # Optimize current shock if nothing selected
        if len(self.shocks2run) == 0:
            self.shocks2run = [parent.display_shock]
        else:
            if not parent.load_full_series_box.isChecked(): # TODO: May want to remove this limitation in future
                parent.log.append('"Load Full Series Into Memory" must be checked for optimization of multiple experiments\n')
                return
            elif len(parent.series_viewer.data_table) == 0:
                parent.log.append('Set Series in Series Viewer and select experiments\n')
                return
        
        if len(self.shocks2run) == 0: return    # if no shocks to run return, not sure if necessary anymore

        # Initialize variables in shocks if need be
        opt_options = self.parent.optimization_settings.settings
        weight_fcn = parent.series.weights
        unc_fcn = parent.series.uncertainties
        for shock in self.shocks2run:      # TODO NEED TO UPDATE THIS FOR UNCERTAINTIES AND WEIGHTS
            # update opt_time_offset if it was changed while a prior optimization was running
            shock['opt_time_offset'] = shock['time_offset']
            
            # if weight variables aren't set, update
            if opt_options['obj_fcn']['type'] == 'Residual':
                weight_var = [shock[key] for key in ['weight_max', 'weight_min', 'weight_shift', 'weight_k']]
                if np.isnan(np.hstack(weight_var)).any():
                    parent.weight.update(shock=shock)
                    shock['weights'] = weight_fcn(shock['exp_data'][:,0], shock)
            else: # otherwise bayesian
                unc_var = [shock[key] for key in ['unc_max', 'unc_min', 'unc_shift', 'unc_k', 'unc_cutoff']]
                if np.isnan(np.hstack(unc_var)).any():
                    parent.exp_unc.update(shock=shock)
                    shock['uncertainties'] = unc_fcn(shock['exp_data'][:,0], shock, calcWeights=True)

            # if reactor temperature and pressure aren't set, update
            if np.isnan([shock['T_reactor'], shock['P_reactor']]).any():
                parent.series.set('zone', shock['zone'])
                    
            parent.series.rate_bnds(shock)
        
        # Set conditions of rates to be fit for each opt coefficient and rates/bnds
        self._initialize_opt()
        self._update_gas()
        
        parent.update_user_settings()
        # parent.set_weights()
        
        parent.abort = False
        parent.optimize_running = True
        
        # Create mechs and duplicate mech variables
        if parent.multiprocessing == True:
            cpu_count = mp.cpu_count() + 2
            # if cpu_count > 1: # leave open processor
                # cpu_count -= 1
            parent.max_processors = np.min([len(self.shocks2run), cpu_count])
            
            log_str = 'Number of processes: {:d}'.format(parent.max_processors)
            parent.log.append(log_str, alert=False)
        else:
            parent.max_processors = 1

        # Pass the function to execute
        self.worker = Worker(parent, self.shocks2run, parent.mech, self.coef_opt, self.rxn_coef_opt, self.rxn_rate_opt)
        self.worker.signals.result.connect(self.on_worker_done)
        self.worker.signals.finished.connect(self.thread_complete)
        self.worker.signals.update.connect(self.update)
        self.worker.signals.progress.connect(self.on_worker_progress)
        self.worker.signals.log.connect(parent.log.append)
        self.worker.signals.abort.connect(self.worker.abort)
        
        # Optimization plot
        parent.plot.opt.clear_plot()

        # Reset Hall of Fame
        self.HoF = []   # hall of fame for tracking the best result so far

        # Create Progress Bar
        # parent.create_progress_bar()

        # Invoke the optimization to begin in a separate thread
        if not parent.abort:
            s = 'Optimization starting\n\n   Iteration\t\t Objective Func\tBest Objetive Func'
            parent.log.append(s, alert=False)
            parent.threadpool.start(self.worker)
    
    def _has_opt_rxns(self):              
        mech = self.parent.mech
        for rxnIdx in range(mech.gas.n_reactions):      # searches all rxns
            if mech.rate_bnds[rxnIdx]['opt']: 
                return True
        return False    

    def _initialize_opt(self):
        """Initialize various dictionaries for optimization"""
        self.coef_opt = self._set_coef_opt()
        self.rxn_coef_opt = self._set_rxn_coef_opt()
        self.rxn_rate_opt = self._set_rxn_rate_opt()

    def _set_coef_opt(self):
        """Find which coefficients of kinetic models should be optimized"""
        mech = self.parent.mech
        coef_opt = []
        for rxnIdx, rxn in enumerate(mech.gas.reactions()):      # searches all rxns
            if not mech.rate_bnds[rxnIdx]['opt']: continue        # ignore fixed reactions

            # check all coefficients
            for bndsKey, subRxn in mech.coeffs_bnds[rxnIdx].items():
                for coefIdx, (coefName, coefDict) in enumerate(subRxn.items()):
                    if coefDict['opt']:
                        coefKey, bndsKey = mech.get_coeffs_keys(rxn, bndsKey, rxnIdx=rxnIdx)
                        coef_opt.append({'rxnIdx': rxnIdx, 'key': {'coeffs': coefKey, 'coeffs_bnds': bndsKey}, 
                                         'coefIdx': coefIdx, 'coefName': coefName})

        return coef_opt
    
    def _set_rxn_coef_opt(self, min_T_range=500, min_P_range_factor=2):
        """Get the initial value and bounds for the parameters being optimized"""
        mech = self.parent.mech
        rxn_coef_opt = []
        print(self.coef_opt)
        for coef in self.coef_opt:
            if len(rxn_coef_opt) == 0 or coef['rxnIdx'] != rxn_coef_opt[-1]['rxnIdx']:
                rxn_coef_opt.append(deepcopy(coef))
                rxn_coef_opt[-1]['key'] = [rxn_coef_opt[-1]['key']]
                rxn_coef_opt[-1]['coefIdx'] = [rxn_coef_opt[-1]['coefIdx']]
                rxn_coef_opt[-1]['coefName'] = [rxn_coef_opt[-1]['coefName']]
            else:
                rxn_coef_opt[-1]['key'].append(coef['key'])
                rxn_coef_opt[-1]['coefIdx'].append(coef['coefIdx'])
                rxn_coef_opt[-1]['coefName'].append(coef['coefName'])

        # Generate shock conditions to be optimized
        shock_conditions = {'T_reactor': [], 'P_reactor': [], 'thermo_mix': []}
        for shock in self.shocks2run:
            for shock_condition in shock_conditions:
                shock_conditions[shock_condition].append(shock[shock_condition])
        
        # Set evaluation rate temperature conditions
        T_min = np.min(shock_conditions['T_reactor'])
        T_max = np.max(shock_conditions['T_reactor'])

        T_bnds = np.array([T_min, T_max])
        if T_bnds[1] - T_bnds[0] < min_T_range:  # if T_range isn't large enough increase it
            T_median = np.median(shock_conditions['T_reactor'])
            T_bnds = np.array([T_median-min_T_range/2, T_median+min_T_range/2])
            if T_bnds[0] < 298.15:
                T_bnds[1] += 298.15 - T_bnds[0]
                T_bnds[0] = 298.15

        invT_bnds = np.divide(10000, T_bnds)

        # Set evaluation rate pressure conditions
        P_min = np.min(shock_conditions['P_reactor'])
        P_max = np.max(shock_conditions['P_reactor'])
        P_median = np.median(shock_conditions['P_reactor'])
        
        P_bnds = np.array([P_min, P_max])
        if P_bnds[1]/P_bnds[0] < min_P_range_factor:
            P_f_min = min_P_range_factor**0.5
            P_bnds = np.array([P_median/P_f_min, P_median*P_f_min])

        for rxn_coef in rxn_coef_opt:
            # Set coefficient initial values and bounds
            rxnIdx = rxn_coef['rxnIdx']
            rxn = mech.gas.reaction(rxnIdx)

            rxn_coef['coef_x0'] = []
            for coefNum, (key, coefName) in enumerate(zip(rxn_coef['key'], rxn_coef['coefName'])):
                coef_x0 = mech.coeffs_bnds[rxnIdx][key['coeffs_bnds']][coefName]['resetVal']
                rxn_coef['coef_x0'].append(coef_x0)

            rxn_coef['coef_bnds'] = set_bnds(mech, rxnIdx, rxn_coef['key'], rxn_coef['coefName'])

            if type(rxn) in [ct.ElementaryReaction, ct.ThreeBodyReaction]:
                P = P_median

            elif type(rxn) is ct.PlogReaction:
                P = []
                for PlogRxn in mech.coeffs[rxnIdx]:
                    P.append(PlogRxn['Pressure'])
                
                if len(P) < 4:
                    P = np.geomspace(np.min(P), np.max(P), 4)

            if type(rxn) is ct.FalloffReaction:
                P = np.linspace(P_bnds[0], P_bnds[1], 3)

            # set rxn_coef dict
            if type(rxn) in [ct.ElementaryReaction, ct.ThreeBodyReaction]:
                n_coef = len(rxn_coef['coefIdx'])
                rxn_coef['invT'] = np.linspace(*invT_bnds, n_coef)
                rxn_coef['T'] = np.divide(10000, rxn_coef['invT'])
                rxn_coef['P'] = np.ones_like(rxn_coef['T'])*P

            elif type(rxn) in [ct.PlogReaction, ct.FalloffReaction]:
                rxn_coef['invT'] = []
                rxn_coef['P'] = []

                # set conditions for upper and lower rates
                for coef_type in ['low_rate', 'high_rate']:
                    n_coef = 0
                    for coef in rxn_coef['key']:
                        if coef_type in coef['coeffs_bnds']:
                            n_coef += 1
                   
                    rxn_coef['invT'].append(np.linspace(*invT_bnds, n_coef))
                    if coef_type == 'low_rate':
                        rxn_coef['P'].append(np.ones(n_coef)*P[0]) # will evaluate LPL if LPL is constrained, else this value
                    elif coef_type == 'high_rate':
                        rxn_coef['P'].append(np.ones(n_coef)*P[-1]) # will evaluate HPL if HPL is constrained, else this value
                
                # set conditions for middle conditions (coefficients are always unbounded)
                if type(rxn) is ct.PlogReaction:
                    invT = np.linspace(*invT_bnds, 3)

                    P, invT = np.meshgrid(P[1:-1], invT)
                    rxn_coef['invT'].append(invT.T.flatten())
                    rxn_coef['P'].append(P.T.flatten())
                else:
                    rxn_coef['invT'].append(np.linspace(*invT_bnds, 3))
                    rxn_coef['P'].append(np.ones(3)*P_median) # Median P for falloff

                for key in ['invT', 'P']:
                    rxn_coef[key] = np.concatenate(rxn_coef[key], axis=0)
                rxn_coef['T'] = np.divide(10000, rxn_coef['invT'])

            rxn_coef['X'] = shock_conditions['thermo_mix'][0]   # TODO: IF MIXTURE COMPOSITION FOR DUMMY RATES MATTER CHANGE HERE
            rxn_coef['is_falloff_limit'] = np.array([False]*len(rxn_coef['T']))   # this only has meaning for falloff equations

        return rxn_coef_opt

    def _set_rxn_rate_opt(self):
        mech = self.parent.mech
        rxn_rate_opt = {}

        # Calculate x0 (initial rates)
        prior_coeffs = mech.reset()  # reset mechanism and get mech that it was
        rxn_rate_opt['x0'] = rates(self.rxn_coef_opt, mech)
        
        # Determine rate bounds
        bnds = np.array([[],[]])
        for i, rxn_coef in enumerate(self.rxn_coef_opt):
            rxnIdx = rxn_coef['rxnIdx']
            rxn = mech.gas.reaction(rxnIdx)
            rate_bnds_val = mech.rate_bnds[rxnIdx]['value']
            rate_bnds_type = mech.rate_bnds[rxnIdx]['type']
            if type(rxn) in [ct.PlogReaction, ct.FalloffReaction]: # if falloff, change arrhenius rates to LPL/HPL if they are not constrained
                key_list = np.array([x['coeffs_bnds'] for x in rxn_coef['key']])
                key_count = collections.Counter(key_list)

                for n, T in enumerate(rxn_coef['T']):
                    if n == len(key_list):
                        break

                    coef_type_key = key_list[n]
                    if 'rate' in coef_type_key:
                        idx_match = np.argwhere(coef_type_key == key_list)

                        if np.any(rxn_coef['coef_bnds']['exist'][idx_match]) or key_count[coef_type_key] < 3:
                            rxn_coef['is_falloff_limit'][n] = True

                            if type(rxn) is ct.FalloffReaction:
                                x = []
                                for ArrheniusCoefName in default_arrhenius_coefNames:
                                    x.append(mech.coeffs_bnds[rxnIdx][coef_type_key][ArrheniusCoefName]['resetVal'])

                                rxn_rate_opt['x0'][i+n] = np.log(x[1]) + x[2]*np.log(T) - x[0]/(Ru*T)                        

            ln_rate = rxn_rate_opt['x0'][i:i + len(rxn_coef['T'])]
            rxn_coef_bnds = mech.rate_bnds[rxnIdx]['limits'](np.exp(ln_rate))
            rxn_coef_bnds = np.sort(np.log(rxn_coef_bnds), axis=0)  # operate on ln and scale
            scaled_rxn_coef_bnds = rxn_coef_bnds - ln_rate

            bnds = np.concatenate((bnds, scaled_rxn_coef_bnds), axis=1)

        rxn_rate_opt['bnds'] = {'lower': bnds[0,:], 'upper': bnds[1,:]}

        # set mech to prior mech
        mech.coeffs = prior_coeffs
        mech.modify_reactions(mech.coeffs)

        return rxn_rate_opt

    def _update_gas(self): # TODO: What happens if a second optimization is run?
        parent = self.parent
        mech = parent.mech
        reset_mech = mech.reset_mech
        coef_opt = self.coef_opt
        
        # delete any falloff coefficients with 5 indices
        delete_idx = []
        for i, idxDict in enumerate(coef_opt):
            rxnIdx, coefName = idxDict['rxnIdx'], idxDict['coefName']
            if coefName == 4:
                delete_idx.append(i)

        for i in delete_idx[::-1]:
            del coef_opt[i]

        generate_new_mech = False
        rxns_changed = []
        i = 0
        for rxn_coef_idx, rxn_coef in enumerate(self.rxn_coef_opt):      # TODO: RXN_COEF_OPT INCORRECT FOR CHANGING RXN TYPES
            rxnIdx = rxn_coef['rxnIdx']
            rxn = mech.gas.reaction(rxnIdx)
            if type(rxn) in [ct.ElementaryReaction, ct.ThreeBodyReaction]:
                continue    # arrhenius type equations don't need to be converted

            T, P, X = rxn_coef['T'], rxn_coef['P'], rxn_coef['X']
            M = lambda T, P: mech.M(rxn, [T, P, X])
            rates = np.exp(self.rxn_rate_opt['x0'][i:i+len(T)])

            if type(rxn) is ct.FalloffReaction:
                lb = rxn_coef['coef_bnds']['lower']
                ub = rxn_coef['coef_bnds']['upper']
                if rxn.falloff.type == 'SRI':   
                    rxns_changed.append(rxn_coef['rxnIdx'])
                    rxn_coef['coef_x0'] = Troe_fit(rates, T, P, X, rxnIdx, rxn_coef['key'], [], 
                                               rxn_coef['is_falloff_limit'], mech, [lb, ub], accurate_fit=True)
                
                mech.coeffs[rxnIdx]['falloff_type'] = 'Troe'
                mech.coeffs[rxnIdx]['falloff_parameters'] = rxn_coef['coef_x0'][-4:]

            else:
                rxns_changed.append(rxn_coef['rxnIdx'])

                lb = rxn_coef['coef_bnds']['lower']
                ub = rxn_coef['coef_bnds']['upper']
                rxn_coef['coef_x0'] = Troe_fit(rates, T, P, X, rxnIdx, rxn_coef['key'], rxn_coef['coefName'], 
                                               rxn_coef['is_falloff_limit'], mech, [lb, ub], accurate_fit=True)

                rxn_coef['coefIdx'].extend(range(0, 4)) # extend to include falloff coefficients
                rxn_coef['coefName'].extend(range(0, 4)) # extend to include falloff coefficients
                rxn_coef['key'].extend([{'coeffs': 'falloff_parameters', 'coeffs_bnds': 'falloff_parameters'} for _ in range(0, 4)])

                # modify mech.coeffs from plog to falloff
                mech.coeffs[rxnIdx] = {'falloff_type': 'Troe', 'high_rate': {}, 'low_rate': {}, 
                                       'falloff_parameters': rxn_coef['coef_x0'][-4:], 'default_efficiency': 1.0, 'efficiencies': {}}

                n = 0
                for key in ['low_rate', 'high_rate']:
                    for coefName in default_arrhenius_coefNames:
                        rxn_coef['key'][n]['coeffs'] = key                          # change key value to match new reaction type
                        mech.coeffs[rxnIdx][key][coefName] = rxn_coef['coef_x0'][n] # updates new arrhenius values

                        n += 1
                
                rxn_coef['coef_bnds'] = set_bnds(mech, rxnIdx, rxn_coef['key'], rxn_coef['coefName'])

                # set reset_mech for new mechanism
                generate_new_mech = True
                reset_mech[rxnIdx]['rxnType'] = 'FalloffReaction'
                reset_mech[rxnIdx]['rxnCoeffs'] = mech.coeffs[rxnIdx]

            i += len(T)

        if generate_new_mech:
            # copy bounds to use in updated mechanism
            bnds = {'rate_bnds': [], 'coeffs_bnds': []}
            for rxnIdx in range(mech.gas.n_reactions):
                coeffs_bnds = {}
                for bndsKey, subRxn in mech.coeffs_bnds[rxnIdx].items():
                    coeffs_bnds[bndsKey] = {}
                    for coefIdx, (coefName, coefDict) in enumerate(subRxn.items()):
                        coeffs_bnds[bndsKey][coefName] = {'type': 'F', 'value': np.nan, 'opt': False}
                        if coefDict['opt'] and bndsKey != 'falloff_parameters':
                            coeffs_bnds[bndsKey][coefName] = {'type': coefDict['type'], 'value': coefDict['value'], 
                                                              'opt': coefDict['opt']}
                
                bnds['rate_bnds'].append({'value': mech.rate_bnds[rxnIdx]['value'], 'type': mech.rate_bnds[rxnIdx]['type'],
                                          'opt': mech.rate_bnds[rxnIdx]['opt']})
                bnds['coeffs_bnds'].append(coeffs_bnds)

            mech.set_mechanism(reset_mech, bnds=bnds)

            parent.save.chemkin_format(mech.gas, parent.path_set.optimized_mech(file_out='recast_mech'))

        if len(rxns_changed) > 0:
            self._initialize_opt()


    def update(self, result, writeLog=True):
        # Update Hall of Fame
        if not self.HoF:
            self.HoF = result
        elif result['obj_fcn'] < self.HoF['obj_fcn']:
            self.HoF = result

        # Update log
        obj_fcn_str = f"{result['obj_fcn']:.3e}"
        replace_strs = [['e+', 'e'], ['e0', 'e'], ['e-0', 'e-']]
        for pair in replace_strs:
            obj_fcn_str = obj_fcn_str.replace(pair[0], pair[1])
        result['obj_fcn_str'] = obj_fcn_str

        if writeLog:
            i = result['i']
            opt_type = result['type'][0].upper()

            if result['i'] > 999:
                obj_fcn_space = '\t\t'
            else:
                obj_fcn_space = '\t\t\t'
            
            if 'inf' in obj_fcn_str:
                obj_fcn_str_space = '\t\t\t\t\t'
            elif len(obj_fcn_str) < 6:
                obj_fcn_str_space = '\t\t\t\t'
            else:
                obj_fcn_str_space = '\t\t\t'
            
            log_str = (f'\t{opt_type.upper()} {i:^5d}{obj_fcn_space}{obj_fcn_str:^s}'
                       f'{obj_fcn_str_space}{self.HoF["obj_fcn_str"]:^s}')

            self.parent.log.append(log_str, alert=False)

        self.parent.tree.update_coef_rate_from_opt(self.coef_opt, result['x'])

        if timer() - self.last_plot_timer > self.time_between_plots:
            plot_start_time = timer()
            # if displayed shock isn't in shocks being optimized, calculate the new plot
            if result['ind_var'] is None and result['observable'] is None:
                self.parent.run_single()
            else:       # if displayed shock in list being optimized, show result
                self.parent.plot.signal.update_sim(result['ind_var'][:,0], result['observable'][:,0])
            self.parent.plot.opt.update(result['stat_plot'])

            # this keeps gui responsive, minimum time between plots = max time to plot*0.1
            current_time_to_plot = timer() - plot_start_time
            if current_time_to_plot*0.1 > self.time_between_plots:
                self.time_between_plots = current_time_to_plot*0.1

            self.last_plot_timer = timer()
    
    def on_worker_progress(self, perc_completed, time_left):
        self.parent.update_progress(perc_completed, time_left)
    
    def thread_complete(self): pass
    
    def on_worker_done(self, result):
        parent = self.parent
        parent.optimize_running = False
        if result is None or len(result) == 0: return
        
        # update mech to optimized one
        if 'local' in result:
            update_mech_coef_opt(parent.mech, self.coef_opt, result['local']['x'])
        else:
            update_mech_coef_opt(parent.mech, self.coef_opt, result['global']['x'])
        
        for opt_type, res in result.items():
            total_shock_eval = (res['nfev']+1)*len(self.shocks2run)
            message = res['message'][:1].lower() + res['message'][1:]
            
            parent.log.append('\n{:s} {:s}'.format(opt_type.capitalize(), message))
            parent.log.append('\telapsed time:\t{:.2f}'.format(res['time']), alert=False)
            parent.log.append('\tAvg Std Residual:\t{:.3e}'.format(res['fval']), alert=False)
            parent.log.append('\topt iters:\t\t{:.0f}'.format(res['nfev']+1), alert=False)
            parent.log.append('\tshock evals:\t{:.0f}'.format(total_shock_eval), alert=False)
            parent.log.append('\tsuccess:\t\t{:}'.format(res['success']), alert=False)
        
        parent.log.append('\n', alert=False)
        parent.save.chemkin_format(parent.mech.gas, parent.path_set.optimized_mech())
        parent.path_set.mech()  # update mech pulldown choices
        parent.tree._copy_expanded_tab_rates() # trigger copy rates

        parent.app.alert(parent, 5*1000) # make application alert for x millisec on completion

    def abort_workers(self):
        if hasattr(self, 'worker'):
            self.worker.signals.abort.emit()
            self.parent.abort = True
            if self.HoF:
                self.update(self.HoF, writeLog=False)
        
        self.parent.optimize_running = False
            # self.parent.update_progress(100, '00:00:00') # This turns off the progress bar
