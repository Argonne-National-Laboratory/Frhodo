# This file is part of Frhodo. Copyright © 2020, UChicago Argonne, LLC
# and licensed under BSD-3-Clause. See License.txt in the top-level 
# directory for license and copyright information.

from qtpy.QtCore import QObject, QRunnable, Signal, Slot
import multiprocessing as mp
import sys, platform, io, traceback, contextlib, pathlib
from copy import deepcopy

import nlopt, pygmo, rbfopt
import numpy as np

from timeit import default_timer as timer

from calculate.optimize.fit_fcn import initialize_parallel_worker, Fit_Fun
from calculate.optimize.misc_fcns import rates


debug = True

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

    def __init__(self, parent, shocks2run, mech, coef_opt, rxn_coef_opt, rxn_rate_opt, *args, **kwargs):
        super(Worker, self).__init__()
        # Store constructor arguments (re-used for processing)
        self.parent = parent
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        self.__abort = False
        self.err = False

        self.debug = debug # shows error message in command window. Does not close program
        
        self.shocks2run = shocks2run
        self.coef_opt = coef_opt
        self.rxn_coef_opt = rxn_coef_opt
        self.rxn_rate_opt = rxn_rate_opt
        self.mech = mech
        self._initialize()
    
    def _initialize(self):        
        mech = self.mech

        # Calculate initial rate scalers
        lb, ub = self.rxn_rate_opt['bnds'].values()
        self.s = rates(self.rxn_coef_opt, mech) - self.rxn_rate_opt['x0'] # this initializes from current GUI settings

        # Correct initial rate guesses if outside bounds
        self.s = np.clip(self.s, lb*(1+1E-9), ub*(1-1E-9))

    def trim_shocks(self): # trim shocks from zero weighted data
        for n, shock in enumerate(self.shocks2run):
            weights = shock['normalized_weights']
            #weights = shock['weights']
            
            exp_bounds = np.nonzero(weights)[0]
            shock['weights_trim'] = weights[exp_bounds]
            shock['exp_data_trim'] = shock['exp_data'][exp_bounds,:]
            if 'abs_uncertainties' in shock:
                shock['abs_uncertainties_trim'] = shock['abs_uncertainties'][exp_bounds,:]
    
    def optimize_coeffs(self):
        parent = self.parent
        pool = mp.Pool(processes=parent.max_processors,
                       initializer=initialize_parallel_worker,
                       initargs=(parent.mech.reset_mech, parent.mech.thermo_coeffs, parent.mech.coeffs, parent.mech.coeffs_bnds, 
                       parent.mech.rate_bnds,))
        
        self.trim_shocks()  # trim shock data from zero weighted data
        
        input_dict = {'parent': parent, 'pool': pool, 'mech': self.mech, 'shocks2run': self.shocks2run,
                      'coef_opt': self.coef_opt, 'rxn_coef_opt': self.rxn_coef_opt, 'rxn_rate_opt': self.rxn_rate_opt,
                      'multiprocessing': parent.multiprocessing, 'signals': self.signals}
           
        Scaled_Fit_Fun = Fit_Fun(input_dict)
        def eval_fun(s, grad=None):            
            if self.__abort:
                parent.optimize_running = False
                self.signals.log.emit('\nOptimization aborted')
                raise Exception('Optimization terminated by user')
                # self.signals.result.emit(hof[0])
                return np.nan
            else:
                return Scaled_Fit_Fun(s)

        optimize = Optimize(eval_fun, self.s, self.rxn_rate_opt['bnds'], self.parent.optimization_settings.settings, Scaled_Fit_Fun)
        try:
            res = optimize.run()
                        
        except Exception as e:
            if self.debug:
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

nlopt_algorithms = [nlopt.GN_DIRECT, nlopt.GN_DIRECT_NOSCAL, nlopt.GN_DIRECT_L, nlopt.GN_DIRECT_L_RAND, 
                    nlopt.GN_DIRECT_L_NOSCAL, nlopt.GN_DIRECT_L_RAND_NOSCAL, nlopt.GN_ORIG_DIRECT, 
                    nlopt.GN_ORIG_DIRECT_L, nlopt.GN_CRS2_LM, nlopt.G_MLSL_LDS, nlopt.G_MLSL, nlopt.GD_STOGO,
                    nlopt.GD_STOGO_RAND, nlopt.GN_AGS, nlopt.GN_ISRES, nlopt.GN_ESCH, nlopt.LN_COBYLA,
                    nlopt.LN_BOBYQA, nlopt.LN_NEWUOA, nlopt.LN_NEWUOA_BOUND, nlopt.LN_PRAXIS, nlopt.LN_NELDERMEAD,
                    nlopt.LN_SBPLX, nlopt.LD_MMA, nlopt.LD_CCSAQ, nlopt.LD_SLSQP, nlopt.LD_LBFGS, nlopt.LD_TNEWTON,
                    nlopt.LD_TNEWTON_PRECOND, nlopt.LD_TNEWTON_RESTART, nlopt.LD_TNEWTON_PRECOND_RESTART, 
                    nlopt.LD_VAR1, nlopt.LD_VAR2]

pos_msg = ['Optimization terminated successfully.', 'Optimization terminated: Stop Value was reached.',
           'Optimization terminated: Function tolerance was reached.',
           'Optimization terminated: X tolerance was reached.',
           'Optimization terminated: Max number of evaluations was reached.',
           'Optimization terminated: Max time was reached.']
neg_msg = ['Optimization failed', 'Optimization failed: Invalid arguments given',
           'Optimization failed: Out of memory', 'Optimization failed: Roundoff errors limited progress',
           'Optimization failed: Forced termination']

path = {'main': pathlib.Path(sys.argv[0]).parents[0].resolve()}
OS_type = platform.system()
if OS_type == 'Windows':
    path['bonmin'] = path['main'] / 'bonmin/bonmin-win64/bonmin.exe'
    path['ipopt'] = path['main'] / 'ipopt/ipopt-win64/ipopt.exe'
elif OS_type == 'Linux':
    path['bonmin'] = path['main'] / 'bonmin/bonmin-linux64/bonmin'
    path['ipopt'] = path['main'] / 'ipopt/ipopt-linux64/ipopt'
elif OS_type == 'Darwin':
    path['bonmin'] = path['main'] / 'bonmin/bonmin-osx/bonmin'
    path['ipopt'] = path['main'] / 'ipopt/ipopt-osx/ipopt'

class Optimize:
    def __init__(self, obj_fcn, x0, bnds, opt_options, Scaled_Fit_Fun):
        self.obj_fcn = obj_fcn
        self.x0 = x0
        self.bnds = bnds
        self.opt_options = opt_options  
        self.Scaled_Fit_Fun = Scaled_Fit_Fun

    def run(self):
        x0 = self.x0
        bnds = list(self.bnds.values())
        opt_options = self.opt_options

        res = {}
        for n, opt_type in enumerate(['global', 'local']):
            self.Scaled_Fit_Fun.i = 0                     # reset iteration counter
            self.Scaled_Fit_Fun.opt_type = opt_type       # inform about optimization type

            options = opt_options[opt_type]
            if not options['run']: continue

            if options['algorithm'] in nlopt_algorithms:
                res[opt_type] = self.nlopt(x0, bnds, options)
            elif options['algorithm'] in ['pygmo_DE', 'pygmo_SaDE', 'pygmo_PSO', 'pygmo_GWO']:
                res[opt_type] = self.pygmo(x0, bnds, options)
            elif options['algorithm'] == 'RBFOpt':
                res[opt_type] = self.rbfopt(x0, bnds, options)

            if options['algorithm'] is nlopt.GN_MLSL_LDS:   # if using multistart algorithm, break upon finishing loop
                break

        return res

    def nlopt(self, x0, bnds, options):
        timer_start = timer()

        opt = nlopt.opt(options['algorithm'], np.size(x0))
        opt.set_min_objective(self.obj_fcn)
        if options['stop_criteria_type'] == 'Iteration Maximum':
            opt.set_maxeval(int(options['stop_criteria_val'])-1)
        elif options['stop_criteria_type'] == 'Maximum Time [min]':
            opt.set_maxtime(options['stop_criteria_val']*60)

        opt.set_xtol_rel(options['xtol_rel'])
        opt.set_ftol_rel(options['ftol_rel'])
        opt.set_lower_bounds(bnds[0])
        opt.set_upper_bounds(bnds[1])
                
        initial_step = (bnds[1] - bnds[0])*options['initial_step'] 
        np.putmask(initial_step, x0 < 1, -initial_step)  # first step in direction of more variable space
        opt.set_initial_step(initial_step)

        # alter default size of population in relevant algorithms
        if options['algorithm'] in [nlopt.GN_CRS2_LM, nlopt.GN_MLSL_LDS, nlopt.GN_MLSL, nlopt.GN_ISRES]:
            if options['algorithm'] is nlopt.GN_CRS2_LM:
                default_pop_size = 10*(len(x0)+1)
            elif options['algorithm'] in [nlopt.GN_MLSL_LDS, nlopt.GN_MLSL]:
                default_pop_size = 4
            elif options['algorithm'] is nlopt.GN_ISRES:
                default_pop_size = 20*(len(x0)+1)

            opt.set_population(int(np.rint(default_pop_size*options['initial_pop_multiplier'])))

        if options['algorithm'] is nlopt.GN_MLSL_LDS:   # if using multistart algorithm as global, set subopt
            sub_opt = nlopt.opt(opt_options['local']['algorithm'], np.size(x0))
            sub_opt.set_initial_step(initial_step)
            sub_opt.set_xtol_rel(options['xtol_rel'])
            sub_opt.set_ftol_rel(options['ftol_rel'])
            opt.set_local_optimizer(sub_opt)
                
        x = opt.optimize(x0) # optimize!
        #s = parent.optimize.HoF['s']
                
        obj_fcn, x, shock_output = self.Scaled_Fit_Fun(x, optimizing=False)
            
        if nlopt.SUCCESS > 0: 
            success = True
            msg = pos_msg[nlopt.SUCCESS-1]
        else:
            success = False
            msg = neg_msg[nlopt.SUCCESS-1]
                
        # opt.last_optimum_value() is the same as optimal obj_fcn
        res = {'x': x, 'shock': shock_output, 'fval': obj_fcn, 'nfev': opt.get_numevals(),
               'success': success, 'message': msg, 'time': timer() - timer_start}
                
        return res

    def pygmo(self, x0, bnds, options):
        class pygmo_objective_fcn:
            def __init__(self, obj_fcn, bnds):
                self.obj_fcn = obj_fcn
                self.bnds = bnds

            def fitness(self, x):
                return [self.obj_fcn(x)]

            def get_bounds(self):
                return self.bnds

            def gradient(self, x):
                return pygmo.estimate_gradient_h(lambda x: self.fitness(x), x)

        timer_start = timer()

        pop_size = int(np.max([35, 5*(len(x0)+1)]))
        if options['stop_criteria_type'] == 'Iteration Maximum':
            num_gen = int(np.ceil(options['stop_criteria_val']/pop_size))
        elif options['stop_criteria_type'] == 'Maximum Time [min]':
            num_gen = int(np.ceil(1E20/pop_size))

        prob = pygmo.problem(pygmo_objective_fcn(self.obj_fcn, tuple(bnds)))
        pop = pygmo.population(prob, pop_size)
        pop.push_back(x = x0)   # puts initial guess into the initial population

        # all coefficients/rules should be optimized if they're to be used
        if options['algorithm'] == 'pygmo_DE':  
            #F = (0.107 - 0.141)/(1 + (num_gen/225)**7.75)
            F = 0.2
            CR = 0.8032*np.exp(-1.165E-3*num_gen)
            algo = pygmo.algorithm(pygmo.de(gen=num_gen, F=F, CR=CR, variant=6))
        elif options['algorithm'] == 'pygmo_SaDE':
            algo = pygmo.algorithm(pygmo.sade(gen=num_gen, variant=6))
        elif options['algorithm'] == 'pygmo_PSO': # using generational version
            algo = pygmo.algorithm(pygmo.pso_gen(gen=num_gen))
        elif options['algorithm'] == 'pygmo_GWO':
            algo = pygmo.algorithm(pygmo.gwo(gen=num_gen))
        elif options['algorithm'] == 'pygmo_IPOPT':
            algo = pygmo.algorithm(pygmo.ipopt())

        pop = algo.evolve(pop)

        x = pop.champion_x

        obj_fcn, x, shock_output = self.Scaled_Fit_Fun(x, optimizing=False)

        msg = 'Optimization terminated successfully.'
        success = True

        res = {'x': x, 'shock': shock_output, 'fval': obj_fcn, 'nfev': pop.problem.get_fevals(),
               'success': success, 'message': msg, 'time': timer() - timer_start}
                
        return res

    def rbfopt(self, x0, bnds, options):  # noisy, cheap function option. supports discrete variables
        # https://rbfopt.readthedocs.io/en/latest/rbfopt_user_black_box.html
        # https://rbfopt.readthedocs.io/en/latest/rbfopt_settings.html
        # https://rbfopt.readthedocs.io/en/latest/rbfopt_algorithm.html
        
        timer_start = timer()

        if options['stop_criteria_type'] == 'Iteration Maximum':
            max_eval = int(options['stop_criteria_val'])
            max_time = 1E30
        elif options['stop_criteria_type'] == 'Maximum Time [min]':
            max_eval = 10000 # will need to check if rbfopt changes based on iteration
            max_time = options['stop_criteria_val']*60

        var_type = ['R']*np.size(x0)    # specifies that all variables are continious
        
        output = {'success': False, 'message': []}
        # Intialize and report any problems to log, not to console window
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with contextlib.redirect_stdout(stdout):
                bb = rbfopt.RbfoptUserBlackBox(np.size(x0), np.array(bnds[0]), np.array(bnds[1]),
                                    np.array(var_type), self.obj_fcn)
                settings = rbfopt.RbfoptSettings(max_iterations=max_eval,
                                                    max_evaluations=max_eval,
                                                    max_cycles=1E30,
                                                    max_clock_time=max_time,
                                                    minlp_solver_path=path['bonmin'], 
                                                    nlp_solver_path=path['ipopt'])
                algo = rbfopt.RbfoptAlgorithm(settings, bb, init_node_pos=x0)
                val, x, itercount, evalcount, fast_evalcount = algo.optimize()
                
                obj_fcn, x, shock_output = self.Scaled_Fit_Fun(x, optimizing=False)

                output['message'] = 'Optimization terminated successfully.'
                output['success'] = True
            
        ct_out = stdout.getvalue()
        ct_err = stderr.getvalue()

        print(ct_out)

        # opt.last_optimum_value() is the same as optimal obj_fcn
        res = {'x': x, 'shock': shock_output, 'fval': obj_fcn, 'nfev': evalcount + fast_evalcount,
               'success': output['success'], 'message': output['message'], 'time': timer() - timer_start}
                
        return res