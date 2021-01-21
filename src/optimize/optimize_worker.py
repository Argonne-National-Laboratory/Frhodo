# This file is part of Frhodo. Copyright © 2020, UChicago Argonne, LLC
# and licensed under BSD-3-Clause. See License.txt in the top-level 
# directory for license and copyright information.

from qtpy.QtCore import QObject, QRunnable, Signal, Slot
import multiprocessing as mp
import traceback, sys, io, contextlib
from copy import deepcopy

import nlopt
import numpy as np

from timeit import default_timer as timer

from optimize.fit_fcn import initialize_parallel_worker, Fit_Fun


class Worker(QRunnable):
    '''
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and 
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    It is computationally efficient to limit the amount of unnecessary information sent to the GUI
    
    '''

    def __init__(self, parent, shocks2run, mech, coef_opt, rxn_coef_opt, *args, **kwargs):
        super(Worker, self).__init__()
        # Store constructor arguments (re-used for processing)
        self.parent = parent
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        self.__abort = False
        self.err = False
        
        self.shocks2run = shocks2run
        self.coef_opt = coef_opt
        self.rxn_coef_opt = rxn_coef_opt
        self.mech = mech
        self._initialize()
    
    def _initialize(self):
        def rates():
            output = []
            for rxn_coef in self.rxn_coef_opt:
                for T, P in zip(rxn_coef['T'], rxn_coef['P']):
                    mech.set_TPX(T, P)
                    output.append(mech.gas.forward_rate_constants[rxn_coef['rxnIdx']])
            
            return np.log(output)
        
        mech = self.mech

        # reset mechanism
        initial_mech = deepcopy(mech.coeffs)
        for rxnIdx in range(mech.gas.n_reactions):
            for coefName in mech.coeffs[rxnIdx].keys():
                resetVal = mech.coeffs_bnds[rxnIdx][coefName]['resetVal']
                mech.coeffs[rxnIdx][coefName] = resetVal 
        
        mech.modify_reactions(mech.coeffs)
        
        # Calculate x0
        self.x0 = rates()
        
        # Determine rate bounds
        lb = []
        ub = []
        i = 0
        for rxn_coef in self.rxn_coef_opt:
            rxnIdx = rxn_coef['rxnIdx']
            rate_bnds_val = mech.rate_bnds[rxnIdx]['value']
            rate_bnds_type = mech.rate_bnds[rxnIdx]['type']
            for T, P in zip(rxn_coef['T'], rxn_coef['P']):
                mech.set_TPX(T, P)
                bnds = mech.rate_bnds[rxnIdx]['limits'](mech.gas.forward_rate_constants[rxnIdx])
                bnds = np.sort(np.log(bnds)/self.x0[i])  # operate on ln and scale
                lb.append(bnds[0])
                ub.append(bnds[1])
                
                i += 1
        
        self.bnds = {'lower': np.array(lb), 'upper': np.array(ub)}

        # Calculate coefficient x0 and bounds
        # TODO 

        # Calculate initial rate scalers
        mech.coeffs = initial_mech
        mech.modify_reactions(mech.coeffs)
        self.s = np.divide(rates(), self.x0)

        # Correct initial rate guesses if outside bounds
        np.putmask(self.s, self.s < lb, lb)
        np.putmask(self.s, self.s > ub, ub)       

    def trim_shocks(self): # trim shocks from zero weighted data
        for n, shock in enumerate(self.shocks2run):
            weights = shock['normalized_weights']
            #weights = shock['weights']
            
            exp_bounds = np.nonzero(weights)[0]
            shock['weights_trim'] = weights[exp_bounds]
            shock['exp_data_trim'] = shock['exp_data'][exp_bounds,:]
    
    def optimize_coeffs(self, debug=False):
        debug = True  # shows error message in command window. Does not close program
        parent = self.parent
        pool = mp.Pool(processes=parent.max_processors,
                       initializer=initialize_parallel_worker,
                       initargs=(parent.mech.yaml_txt, parent.mech.coeffs, parent.mech.coeffs_bnds, 
                       parent.mech.rate_bnds,))
        
        self.trim_shocks()  # trim shock data from zero weighted data
        
        input_dict = {'parent': parent, 'pool': pool, 'mech': self.mech, 'shocks2run': self.shocks2run,
                      'coef_opt': self.coef_opt, 'rxn_coef_opt': self.rxn_coef_opt,
                      'x0': self.x0, 'bounds': self.bnds,
                      'multiprocessing': parent.multiprocessing, 'signals': self.signals}
           
        Scaled_Fit_Fun = Fit_Fun(input_dict)
           
        def eval_fun(s, grad):            
            if self.__abort:
                parent.optimize_running = False
                raise Exception('Optimization terminated by user')
                self.signals.log.emit('\nOptimization aborted')
                # self.signals.result.emit(hof[0])
            else:
                return Scaled_Fit_Fun(s)

        try:
            opt_options = self.parent.optimization_settings.settings  

            s = self.s
            res = {}
            for n, opt_type in enumerate(['global', 'local']):
                timer_start = timer()
                Scaled_Fit_Fun.i = 0                     # reset iteration counter
                Scaled_Fit_Fun.opt_type = opt_type       # inform about optimization type
                options = opt_options[opt_type]
                if not options['run']: continue
                
                opt = nlopt.opt(options['algorithm'], np.size(self.x0))
                opt.set_min_objective(eval_fun)
                if options['stop_criteria_type'] == 'Iteration Maximum':
                    opt.set_maxeval(int(options['stop_criteria_val'])-1)
                elif options['stop_criteria_type'] == 'Maximum Time [min]':
                    opt.set_maxtime(options['stop_criteria_val']*60)

                opt.set_xtol_rel(options['xtol_rel'])
                opt.set_ftol_rel(options['ftol_rel'])
                opt.set_lower_bounds(self.bnds['lower'])
                opt.set_upper_bounds(self.bnds['upper'])
                
                initial_step = (self.bnds['upper'] - self.bnds['lower'])*options['initial_step'] 
                np.putmask(initial_step, s < 1, -initial_step)  # first step in direction of more variable space
                opt.set_initial_step(initial_step)

                # alter default size of population in relevant algorithms
                if options['algorithm'] in [nlopt.GN_CRS2_LM, nlopt.GN_MLSL_LDS, nlopt.GN_MLSL, nlopt.GN_ISRES]:
                    if options['algorithm'] is nlopt.GN_CRS2_LM:
                        default_pop_size = 10*(len(s)+1)
                    elif options['algorithm'] in [nlopt.GN_MLSL_LDS, nlopt.GN_MLSL]:
                        default_pop_size = 4
                    elif options['algorithm'] is nlopt.GN_ISRES:
                        default_pop_size = 20*(len(s)+1)

                    opt.set_population(int(np.rint(default_pop_size*options['initial_pop_multiplier'])))

                if options['algorithm'] is nlopt.GN_MLSL_LDS:   # if using multistart algorithm as global, set subopt
                    sub_opt = nlopt.opt(opt_options['local']['algorithm'], np.size(self.x0))
                    sub_opt.set_initial_step(initial_step)
                    sub_opt.set_xtol_rel(options['xtol_rel'])
                    sub_opt.set_ftol_rel(options['ftol_rel'])
                    opt.set_local_optimizer(sub_opt)
                
                s = opt.optimize(s) # optimize!
                
                obj_fcn, x, shock_output = Scaled_Fit_Fun(s, optimizing=False)
            
                if nlopt.SUCCESS > 0: 
                    success = True
                    msg = pos_msg[nlopt.SUCCESS-1]
                else:
                    success = False
                    msg = neg_msg[nlopt.SUCCESS-1]
                
                # opt.last_optimum_value() is the same as optimal obj_fcn
                res[opt_type] = {'coef_opt': self.coef_opt, 'x': x, 'shock': shock_output,
                                 'fval': obj_fcn, 'nfev': opt.get_numevals(),
                                 'success': success, 'message': msg, 'time': timer() - timer_start}
                
                if options['algorithm'] is nlopt.GN_MLSL_LDS:   # if using multistart algorithm, break upon finishing loop
                    break
                        
        except Exception as e:
            if debug:
                pool.close()
                raise

            res = None
            if 'Optimization terminated by user' in str(e):
                self.signals.log.emit('\nOptimization aborted')
            else:
                self.err = True
                self.signals.log.emit('\n{:s}'.format(str(e)))
            
        pool.close()
        return res
        
        
        '''
        stdout = io.StringIO()
        stderr = io.StringIO()
        
        with contextlib.redirect_stderr(stderr):    
            with contextlib.redirect_stdout(stdout):
                try:
                    res = minimize(eval_fun, coef_norm, method='COBYLA')
                        # options={'rhobeg': 1e-02,})
                except Exception as e:
                    if 'Optimization terminated by user' in str(e):
                        self.signals.log.emit('\nOptimization aborted')
                        # self.signals.result.emit(hof[0])
                        return
                    else:
                        self.err = True
                        self.signals.log.emit('\n{:s}'.format(e))
        
        if self.err:
            out = stdout.getvalue()
            err = stderr.getvalue().replace('INFO:root:', 'Warning: ')
                
            for log_str in [out, err]:
                if log_str != '':
                    self.signals.log.append(log_str)  # Append output
        '''
        
    @Slot()
    def run(self):
        try:
            res = self.optimize_coeffs()
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            # self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(res)  # Return the result of the processing
        finally:
            pass
            # self.signals.finished.emit()  # Done
              
    def abort(self):
        self.__abort = True
        if hasattr(self, 'eval_fun'):
            self.eval_fun.__abort = True
            
            
class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.
    Supported signals are:
    finished
        No data
    error
        `tuple` (exctype, value, traceback.format_exc() )
    result
        `object` best returned from processing
    update
        `str` returns 'object' containing current best
    progress
        'float' returns % complete and estimated time left in s
    log
        'str' output to log
    abort
        No data
    '''
    finished = Signal()
    error = Signal(tuple)
    result = Signal(object)
    update = Signal(object)
    progress = Signal(int, str)
    log = Signal(str)
    abort = Signal()

pos_msg = ['Optimization terminated successfully.', 'Optimization terminated: Stop Value was reached.',
           'Optimization terminated: Function tolerance was reached.',
           'Optimization terminated: X tolerance was reached.',
           'Optimization terminated: Max number of evaluations was reached.',
           'Optimization terminated: Max time was reached.']
neg_msg = ['Optimization failed', 'Optimization failed: Invalid arguments given',
           'Optimization failed: Out of memory', 'Optimization failed: Roundoff errors limited progress',
           'Optimization failed: Forced termination']