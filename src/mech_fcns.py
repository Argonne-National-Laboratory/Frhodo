# This file is part of Frhodo. Copyright © 2020, UChicago Argonne, LLC
# and licensed under BSD-3-Clause. See License.txt in the top-level 
# directory for license and copyright information.

import os, io, contextlib
import cantera as ct
from cantera import interrupts, ck2yaml, cti2yaml#, ctml2yaml
import numpy as np
# import scipy.integrate
# import scipy.optimize
import integrate, shock_fcns

class Simulation_Result:
    def __init__(self, var_val_pairs=[]):
        self.units = 'SI'
        self.var_list = []
        self.reactor_var = {}
        # list of all possible variables
        self.all_var = {'Laboratory Time':                  {'SIM_name': 't_lab',        'sub_type': None},
                        'Shockwave Time':                   {'SIM_name': 't_shock',      'sub_type': None}, 
                        'Gas Velocity':                     {'SIM_name': 'v',            'sub_type': None}, 
                        'Temperature':                      {'SIM_name': 'T',            'sub_type': None}, 
                        'Pressure':                         {'SIM_name': 'P',            'sub_type': None},              
                        'Enthalpy':                         {'SIM_name': 'h',            'sub_type': ['total', 'species']},
                        'Entropy':                          {'SIM_name': 's',            'sub_type': ['total', 'species']}, 
                        'Density':                          {'SIM_name': 'rho',          'sub_type': None},       
                        'Density Gradient':                 {'SIM_name': 'drhodz',       'sub_type': ['total', 'rxn']},
                        '% Density Gradient':               {'SIM_name': 'perc_drhodz',  'sub_type': ['rxn']},
                        'Mole Fraction':                    {'SIM_name': 'X',            'sub_type': ['species']}, 
                        'Mass Fraction':                    {'SIM_name': 'Y',            'sub_type': ['species']}, 
                        'Concentration':                    {'SIM_name': 'conc',         'sub_type': ['species']}, 
                        'Net Production Rate':              {'SIM_name': 'wdot',         'sub_type': ['species']},
                        'Creation Rate':                    {'SIM_name': 'wdotfor',      'sub_type': ['species']},      
                        'Destruction Rate':                 {'SIM_name': 'wdotrev',      'sub_type': ['species']},      
                        'Delta Enthalpy (Heat of Reaction)':{'SIM_name': 'delta_h',      'sub_type': ['rxn']},
                        'Delta Entropy':                    {'SIM_name': 'delta_s',      'sub_type': ['rxn']},  
                        'Equilibrium Constant':             {'SIM_name': 'eq_con',       'sub_type': ['rxn']}, 
                        'Forward Rate Constant':            {'SIM_name': 'rate_con',     'sub_type': ['rxn']}, 
                        'Reverse Rate Constant':            {'SIM_name': 'rate_con_rev', 'sub_type': ['rxn']},          
                        'Net Rate of Progress':             {'SIM_name': 'net_ROP',      'sub_type': ['rxn']}, 
                        'Forward Rate of Progress':         {'SIM_name': 'for_ROP',      'sub_type': ['rxn']},        
                        'Reverse Rate of Progress':         {'SIM_name': 'rev_ROP',      'sub_type': ['rxn']}} 
        
        for var_val in var_val_pairs:
            setattr(self, var_val[0], [var_val[1]])
            self.var_list.append(var_val[0])
            
    def append_result(self, var_val_pairs): # This appends to old values
        for var_val in var_val_pairs:
            getattr(self, var_val[0]).append(var_val[1])
            
    def set_result(self, var_val_pairs):    # This overwrites old values, currently defunct
        for var_val in var_val_pairs:
            setattr(self, var_val[0], [var_val[1]])
            
    def finalize(self, success, ind_var, observable, units = 'CGS'):  # maybe transpose based on shape
        for var in self.var_list:     
            setattr(self, var, np.array(getattr(self, var)))
            if getattr(self, var).ndim > 1: # Transpose if matrix
                setattr(self, var, getattr(self, var).T)
        
        if hasattr(self, 'drhodz_tot') and hasattr(self, 'drhodz'):
            setattr(self, 'perc_drhodz', np.divide(self.drhodz, self.drhodz_tot)*100)
            self.var_list.append('perc_drhodz')
        
        # if hasattr(self, 't_shock'):
            # setattr(self, 't_shock', getattr(self, 't_shock')*1E6)
        # setattr(self, 't_lab', getattr(self, 't_lab')*1E6)
        
        if 'CGS' in units:
            self.convert_to(units)
        
        # assign independent variable and observable    ## TODO: refine this. Really ugly code
        self.independent_var = getattr(self, ind_var)
        k = observable['sub']
        if observable['main'] == 'Temperature':
            self.observable = self.T
        elif observable['main'] == 'Pressure':
            self.observable = self.P
        elif observable['main'] == 'Density Gradient':
            self.observable = self.drhodz_tot
        elif observable['main'] == 'Mole Fraction':
            self.observable = self.X
        elif observable['main'] == 'Mass Fraction':
            self.observable = self.Y
        elif observable['main'] == 'Concentration':
            self.observable = self.conc
        
        if self.observable.ndim > 1:                # reduce observable down to only plotted information
            self.observable = self.observable[k]
        
        self.success = success
                
    def convert_to(self, units):
        if units in self.units: return  # Skip function if the units already match
        
        conv = {'conc': 1E-3, 'wdot': 1E-3, 'T': 1, 'P': 760/101325, 'v': 1E2, 
                'rho': 1E-3, 'drhodz_tot': 1E-5,
                'delta_h': 1E-3/4184, 'h_tot': 1E-3/4184, 'h': 1E-3/4184, # to kcal
                'delta_s': 1/4184, 's_tot': 1/4184, 's': 1/4184, 
                'net_ROP':        1E-3/3.8,    # Don't understand 3.8 value
                'for_ROP':        1E-3/3.8,    # Don't understand 3.8 value
                'rev_ROP':        1E-3/3.8}    # Don't understand 3.8 value
        
        if hasattr(self, 'gas'):
            num_reac = np.sum(self.gas.reactant_stoich_coeffs(), axis=0)
            num_prod = np.sum(self.gas.product_stoich_coeffs(), axis=0)
            
            conv['drhodz']       = np.ones(self.gas.n_reactions)*1E-5
            conv['eq_con']       = 1E3**np.array(num_reac - num_prod)
            conv['rate_con']     = np.power(1E3,num_reac-1)
            conv['rate_con_rev'] = np.power(1E3,num_prod-1)
        
        for var, conversion in conv.items():
            if not hasattr(self, var): continue # skip loop if variable doesn't exist
            
            if 'CGS' in units and 'SI' in self.units:
                if type(conversion) is np.ndarray:
                    setattr(self, var, np.multiply(getattr(self, var), conversion[:, np.newaxis]))
                else:
                    setattr(self, var, getattr(self, var)*conversion)
            elif 'SI' in units and 'CGS' in self.units:
                if type(conversion) is np.ndarray:
                    setattr(self, var, np.divide(getattr(self, var), conversion[:, np.newaxis]))
                else:
                    setattr(self, var, getattr(self, var)/conversion)
                
        self.units = units
        
class Chemical_Mechanism:
    def __init__(self):
        self.isLoaded = False

    def load_mechanism(self, path, silent=False):
        def loader(self, path):
            # path is assumed to be the path dictionary
            surfaces = []
            if path['mech'].suffix in ['.yaml', '.yml']:    # check if it's a yaml cantera file
                mech_path = str(path['mech'])
            else:                                 # if not convert into yaml cantera file
                mech_path = str(path['Cantera_Mech'])
                
                if path['mech'].suffix == '.cti':
                    cti2yaml.convert(path['mech'], path['Cantera_Mech'])
                elif path['mech'].suffix in ['.ctml', '.xml']:
                    raise Exception('not implemented')
                    #ctml2yaml.convert(path['mech'], path['Cantera_Mech'])
                else:                             # if not a cantera file, assume chemkin
                    surfaces = self.chemkin2cantera(path)
                          
            print('Validating mechanism...', end='')    
            try:                                            # This test taken from ck2cti
                self.gas = ct.Solution(mech_path)
                for surfname in surfaces:
                    phase = ct.Interface(outName, surfname, [self.gas])
                print('PASSED.')
            except RuntimeError as e:
                print('FAILED.')
                print(e)
   
        output = {'success': False, 'message': []}
        # Intialize and report any problems to log, not to console window
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with contextlib.redirect_stdout(stdout):
                try:
                    loader(self, path)
                    output['success'] = True
                except Exception as e:
                    output['message'].append('Error in loading mech\n{:s}'.format(str(e)))
                except:
                    pass
                    # output['message'].append('Error when loading mech:\n')
        
        ct_out = stdout.getvalue()
        ct_err = stderr.getvalue().replace('INFO:root:', 'Warning: ')
        
        if 'FAILED' in ct_out:
            output['success'] = False
            self.isLoaded = False
        elif 'PASSED' in ct_out:
            output['success'] = True
            self.isLoaded = True
            
        for log_str in [ct_out, ct_err]:
            if log_str != '' and not silent:
                output['message'].append(log_str)
                output['message'].append('\n')
        
        if self.isLoaded:
            self.set_rate_expression_coeffs()   # set copy of coeffs
            self.set_thermo_expression_coeffs() # set copy of thermo coeffs
        
        return output
    
    def chemkin2cantera(self, path):
        if path['thermo'] is not None:
            surfaces = ck2yaml.convert_mech(path['mech'], thermo_file=path['thermo'], transport_file=None, surface_file=None,
                phase_name='gas', out_name=path['Cantera_Mech'], quiet=False, permissive=True)
        else:
            surfaces = ck2yaml.convert_mech(path['mech'], thermo_file=None, transport_file=None, surface_file=None,
                phase_name='gas', out_name=path['Cantera_Mech'], quiet=False, permissive=True)
            
        return surfaces
      
    def gas(self): return self.gas       
    
    def set_rate_expression_coeffs(self):
        coeffs = []
        coeffs_bnds = []
        rate_bnds = []
        for rxnNum, rxn in enumerate(self.gas.reactions()):
            if hasattr(rxn, 'rate'):
                attrs = [p for p in dir(rxn.rate) if not p.startswith('_')] # attributes not including __              
                coeffs.append({attr: getattr(rxn.rate, attr) for attr in attrs})
                coeffs_bnds.append({attr: {'resetVal': getattr(rxn.rate, attr), 'value': 1, 'type': 'F', 
                        'limits': [getattr(rxn.rate, attr), getattr(rxn.rate, attr)]} for attr in attrs})
                rate_bnds.append({'value': 1, 'type': 'F', 'opt': False})
            else:
                coeffs.append({})
                coeffs_bnds.append({})
                rate_bnds.append({})

        self.coeffs = coeffs
        self.coeffs_bnds = coeffs_bnds
        self.rate_bnds = rate_bnds
    
    def set_thermo_expression_coeffs(self):         # TODO Doesn't work with NASA 9
        self.thermo_coeffs = []
        for i in range(self.gas.n_species):
            S = self.gas.species(i)
            thermo_dict = {'name': S.name}
            thermo_dict['h_scaler'] = 1
            thermo_dict['s_scaler'] = 1
            try:
                thermo_dict['type'] = type(S.thermo)
                thermo_dict['coeffs'] = np.array(S.thermo.coeffs)
            except:
                thermo_dict['type'] = 'unknown'
                thermo_dict['coeffs'] = []
                
            self.thermo_coeffs.append(thermo_dict)   
    
    def modify_reactions(self, coeffs, rxnNums=[]):     # Only works for Arrhenius equations currently
        if not rxnNums:                     # if rxnNums does not exist, modify all
            rxnNums = range(len(coeffs))
        else:
            if isinstance(rxnNums, (float, int)):  # if single reaction given, run that one
                rxnNums = [rxnNums]
        
        for rxnNum in rxnNums:
            rxn = self.gas.reaction(rxnNum)
            if type(rxn) is ct.ElementaryReaction or type(rxn) is ct.ThreeBodyReaction:
                # Get current values 
                A = coeffs[rxnNum]['pre_exponential_factor']
                b = coeffs[rxnNum]['temperature_exponent']
                Ea = coeffs[rxnNum]['activation_energy']
                
                # Update reaction rate
                rxn.rate = ct.Arrhenius(A, b, Ea)
            # elif type(rxn) is ct.PlogReaction:
                # print(dir(rxn))
                # print(rxn.rates[rxn_num])
            # elif type(rxn) is ct.ChebyshevReaction: 
                # print(dir(rxn))
                # print(rxn.rates[rxn_num])
            else:
                continue
                
            self.gas.modify_reaction(rxnNum, rxn)
    
    def modify_thermo(self, multipliers):  # Only works for NasaPoly2 (NASA 7) currently
        for i in range(np.shape(self.gas.species_names)[0]):
            S_initial = self.gas.species(i)
            S = self.gas.species(i)
            if type(S.thermo) is ct.NasaPoly2:
                # Get current values 
                T_low = S_initial.thermo.min_temp
                T_high = S_initial.thermo.max_temp
                P_ref = S_initial.thermo.reference_pressure
                coeffs = S_initial.thermo.coeffs
                
                # Update thermo properties
                coeffs[1:] *= multipliers[i]
                S.thermo = ct.NasaPoly2(T_low, T_high, P_ref, coeffs)
            # elif type(S.thermo) is ct.ShomatePoly2: continue
            # elif type(S.thermo) is ct.NasaPoly1: continue
            # elif type(S.thermo) is ct.Nasa9PolyMultiTempRegion: continue
            # elif type(S.thermo) is ct.Nasa9Poly1: continue
            # elif type(S.thermo) is ct.ShomatePoly: continue
            else:
                print("{:.s}'s thermo is type: {:s}".format(self.gas.species_names[i], type(S.thermo)))
                continue

            self.gas.modify_species(i, S)
    
    def set_TPX(self, T, P, X=[]):
        output = {'success': False, 'message': []}
        if T <= 0 or np.isnan(T):
            output['message'].append('Error: Temperature is invalid')
            return output
        if P <= 0 or np.isnan(P):
            output['message'].append('Error: Pressure is invalid')
            return output
        if len(X) > 0:
            for species in X:
                if species not in self.gas.species_names:
                    output['message'].append('Species: {:s} is not in the mechanism'.format(species))
                    return output
            
            self.gas.TPX = T, P, X
        else:
            self.gas.TP = T, P
            
        output['success'] = True
        return output
    
    def checkRxnRates(self, gas):
        limit = [1E9, 1E15, 1E21]   # reaction limit [first order, second order, third order]
        checkRxn = []
        for rxnIdx in range(gas.n_reactions):
            coef_sum = int(sum(gas.reaction(rxnIdx).reactants.values()))
            if type(gas.reactions()[rxnIdx]) is ct.ThreeBodyReaction:
                coef_sum += 1
            if coef_sum > 0 and coef_sum-1 <= len(limit):   # check that the limit is specified
                rate = [gas.forward_rate_constants[rxnIdx], gas.reverse_rate_constants[rxnIdx]]
                if (np.array(rate) > limit[coef_sum-1]).any():  # if forward or reverse rate exceeds limit
                    checkRxn.append(rxnIdx+1)
        
        return checkRxn
    
    def run(self, reactor_choice, t_end, T_reac, P_reac, mix, **kwargs):
        def list2ct_mixture(mix):   # list in the form of [[species, mol_frac], [species, mol_frac],...]
            return ', '.join("{!s}:{!r}".format(species, mol_frac) for (species, mol_frac) in mix)
        
        details = {'success': False, 'message': []}
        
        if isinstance(mix, list):
            mix = list2ct_mixture(mix)
                
        gas = self.gas
        mech_out = self.set_TPX(T_reac, P_reac, mix)
        if not mech_out['success']:
            details['success'] = False
            details['message'] = mech_out['message']
            return None, mech_out
        
        if reactor_choice == 'Incident Shock Reactor':
            # SIM, details = self.incident_shock_reactor(details, t_end, obs_dict, **kwargs)
            SIM, details =  self.incident_shock_reactor(details, t_end, **kwargs)
        elif '0d Reactor' in reactor_choice:
            if reactor_choice == '0d Reactor - Constant Volume':
                reactor = ct.IdealGasReactor(gas)
            elif reactor_choice == '0d Reactor - Constant Pressure':
                reactor = ct.IdealGasConstPressureReactor(gas)
            
            SIM, details = self.zero_d_ideal_gas_reactor(reactor, details, t_end, **kwargs)
        
        return SIM, details
        
    def incident_shock_reactor(self, details, t_end, **kwargs):
        if 'u_reac' not in kwargs or 'rho1' not in kwargs:
            details['success'] = False
            details['message'] = 'velocity and rho1 not specified\n'
            return None, details
        
        # set default values
        var = {'sim_int_f': 1, 'observable': {'main': 'Density Gradient', 'sub': 0},
               'A1': 0.2, 'As': 0.2, 'L': 0.1, 't_lab_save': None, 
               'ODE_solver': 'BDF', 'rtol': 1E-4, 'atol': 1E-7}
        
        var.update(kwargs)
        
        gas = self.gas
        
        y0 = np.hstack((0.0, var['A1'], gas.density, var['u_reac'], gas.T, 0.0, gas.Y))   # Initial condition
        ode = shock_fcns.ReactorOde(gas, t_end, var['rho1'], var['L'], var['As'], var['A1'], False)
        
        # from timeit import default_timer as timer
        # start = timer()
        sol = integrate.solve_ivp(ode, [0, t_end], y0, method=var['ODE_solver'],
                dense_output=True, rtol=var['rtol'], atol=var['atol'])
        
        if sol.success:
            self.ODE_success = True # this is passed to SIM to inform saving output function
            details['success'] = True
        else:
            self.ODE_success = False # this is passed to SIM to inform saving output function
            details['success'] = False
            
            # Generate log output
            explanation = '\nCheck for: Fast rates or bad thermo data'
            checkRxns = self.checkRxnRates(gas)
            if len(checkRxns) > 0:
                explanation += '\nSuggested Reactions: ' + ', '.join([str(x) for x in checkRxns])
            details['message'] = '\nODE Error: {:s}\n{:s}\n'.format(sol.message, explanation)
        
        if var['sim_int_f'] > np.shape(sol.t)[0]:  # in case of integration failure
            var['sim_int_f'] = np.shape(sol.t)[0]
            
        if var['sim_int_f'] == 1:
            t_sim = sol.t
        else:              # perform interpolation if integrator sample factor > 1
            j = 0
            t_sim = np.zeros(var['sim_int_f']*(np.shape(sol.t)[0] - 1) + 1)    # preallocate array
            for i in range(np.shape(sol.t)[0]-1):
                t_interp = np.interp(np.linspace(i, i+1, var['sim_int_f']+1), [i, i+1], sol.t[i:i+2])
                t_sim[j:j+len(t_interp)] = t_interp
                j += len(t_interp) - 1
        
        # observable dictionary
        obs_dict = {'t_lab': 't', 't_shock': 't_shock', 'z': 'z', 'A': 'A', 'v': 'v', 'T': 'T', 'P': 'gas.P', 
                   'h_tot': 'gas.enthalpy_mole', 'h': 'gas.partial_molar_enthalpies', 
                   's_tot': 'gas.entropy_mole', 's': 'gas.partial_molar_entropies',
                   'rho': 'rho', 
                   'drhodz_tot': 'ode.drhodz(t, y)', 'drhodz': 'ode.drhodz_per_rxn(t, y)', 'perc_drhodz': '0',
                   'Y': 'Y', 'X': 'gas.X', 'conc': 'gas.concentrations', 'wdot': 'gas.net_production_rates', 
                   'wdotfor': 'gas.creation_rates', 'wdotrev': 'gas.destruction_rates', 
                   'delta_h': 'gas.delta_enthalpy', 'delta_s': 'gas.delta_entropy', 
                   'eq_con': 'gas.equilibrium_constants', 'rate_con': 'gas.forward_rate_constants', 
                   'rate_con_rev': 'gas.reverse_rate_constants', 'net_ROP': 'gas.net_rates_of_progress.tolist()',
                   'for_ROP': 'gas.forward_rates_of_progress.tolist()',
                   'rev_ROP': 'gas.reverse_rates_of_progress.tolist()'}
        
        k = var['observable']['sub']
        ind_var = 't_lab'       # INDEPENDENT VARIABLE CURRENTLY HARDCODED FOR t_lab
        if var['t_lab_save'] is None:  # if t_save is not being sent, only plotting variables are needed
            t_all = t_sim
            
            observables = {}
            observables[ind_var] = obs_dict[ind_var]    # add indendent variable to save list
            if var['observable']['main'] == 'Temperature':
                observables['T'] = obs_dict['T']
            elif var['observable']['main'] == 'Pressure':
                observables['P'] = obs_dict['P']
            elif var['observable']['main'] == 'Density Gradient':
                observables['drhodz_tot'] = obs_dict['drhodz_tot']
            elif var['observable']['main'] == 'Mole Fraction':
                observables['X'] = 'gas.X[k]'
            elif var['observable']['main'] == 'Mass Fraction':
                observables['Y'] = 'gas.Y[k]'
            elif var['observable']['main'] == 'Concentration':
                observables['conc'] = 'gas.concentrations[k]'
            
        else:
            t_all = np.sort(np.unique(np.concatenate((t_sim, var['t_lab_save'])))) # combine t_all and t_save, sort, only unique values
            observables = obs_dict
        
        for i, t in enumerate(t_all):   # calculate from solution
            y = sol.sol(t)  
            z, A, rho, v, T, t_shock = y[0:6]
            Y = y[6:]
            gas.TDY = T, rho, Y
            
            data = []
            for obs, observable_fcn in observables.items():
                data.append([obs, eval(observable_fcn)])
                        
            if i == 0:
                SIM = Simulation_Result(data)
            else:
                SIM.append_result(data)
                # if i == np.shape(sol.t)[0] - 1:
                    # break
        
        if var['t_lab_save'] is not None:
            SIM.gas = gas
        
        SIM.finalize(self.ODE_success, ind_var, var['observable'], units='CGS')
        
        # Get record of all available reactor variables
        SIM.reactor_var = {}
        for key, val in SIM.all_var.items():
            if val['SIM_name'] in obs_dict:
                SIM.reactor_var[key] = SIM.all_var[key]
                
        # print('{:0.1f} us'.format((timer() - start)*1E3))
        return SIM, details
     
    def zero_d_ideal_gas_reactor(self, reactor, details, t_end, **kwargs):
        # set default values
        var = {'sim_int_f': 1, 'observable': {'main': 'Concentration', 'sub': 0},
               't_lab_save': None, 'rtol': 1E-4, 'atol': 1E-7}
        
        var.update(kwargs)
        
        obs_dict = {'t_lab': 'states.t[-1]', 'T': 'states.T[-1]', 'P': 'states.P[-1]', 
                   'h_tot': 'states.enthalpy_mole[-1]', 'h': 'states.partial_molar_enthalpies[-1]', 
                   's_tot': 'states.entropy_mole[-1]', 's': 'states.partial_molar_entropies[-1]',
                   'rho': 'states.density[-1]', 
                   'Y': 'states.Y[-1]', 'X': 'states.X[-1]', 'conc': 'states.concentrations[-1]', 
                   'wdot': 'states.net_production_rates[-1]', 
                   'wdotfor': 'states.creation_rates[-1]', 'wdotrev': 'states.destruction_rates[-1]', 
                   'delta_h': 'states.delta_enthalpy[-1]', 'delta_s': 'states.delta_entropy[-1]', 
                   'eq_con': 'states.equilibrium_constants[-1]', 'rate_con': 'states.forward_rate_constants[-1]', 
                   'rate_con_rev': 'states.reverse_rate_constants[-1]', 'net_ROP': 'states.net_rates_of_progress[-1].tolist()',
                   'for_ROP': 'states.forward_rates_of_progress[-1].tolist()',
                   'rev_ROP': 'states.reverse_rates_of_progress[-1].tolist()'}
        
        # Modify reactor if necessary for frozen composition and isothermal
        reactor.energy_enabled = var['solve_energy']
        reactor.chemistry_enabled = not var['frozen_comp']
        
        # Create Sim
        sim = ct.ReactorNet([reactor])
        sim.atol = var['atol']
        sim.rtol = var['rtol']
        states = ct.SolutionArray(self.gas, extra=['t'])
        
        # set up times and observables
        ind_var = 't_lab'       # INDEPENDENT VARIABLE CURRENTLY HARDCODED FOR t_lab
        k = var['observable']['sub']
        if var['t_lab_save'] is None:
            t_all = [t_end]
            
            observables = {}
            observables[ind_var] = obs_dict[ind_var]    # add indendent variable to save list
            if var['observable']['main'] == 'Temperature':
                observables['T'] = obs_dict['T']
            elif var['observable']['main'] == 'Pressure':
                observables['P'] = obs_dict['P']
            elif var['observable']['main'] == 'Mole Fraction':
                observables['X'] = 'states.X[-1][k]'
            elif var['observable']['main'] == 'Mass Fraction':
                observables['Y'] = 'states.Y[-1][k]'
            elif var['observable']['main'] == 'Concentration':
                observables['conc'] = 'states.concentrations[-1][k]'
        else:
            t_all = np.sort(np.unique(np.concatenate(([t_end], var['t_lab_save'])))) # combine t_end and t_save, sort, only unique values
            observables = obs_dict
        
        states.append(reactor.thermo.state, t = 0.0)
        # --- TODO: Clean this up and turn into function since it's called twice
        data = []
        for obs, observable_fcn in observables.items():
            data.append([obs, eval(observable_fcn)])

        SIM = Simulation_Result(data)   # if 'SIM' not in locals(): create, else: append
        # print(sim.time, states.concentrations[-1][3])
        # ---
        for t in t_all:
            while sim.time < t:     # integrator step until time > target time
                sim.step()
                if sim.time > t:    # force interpolation to target time
                    sim.advance(t)
                states.append(reactor.thermo.state, t=sim.time)
                
                data = []
                for obs, observable_fcn in observables.items():
                    data.append([obs, eval(observable_fcn)])
                            
                SIM.append_result(data)
                # print(sim.time, states.concentrations[-1][3])
        
        self.ODE_success = True         # TODO: NEED REAL ERROR CHECKING OF REACTOR SUCCESS
        details['success'] = True
        
        if var['t_lab_save'] is not None:
            SIM.gas = self.gas
        
        SIM.finalize(self.ODE_success, ind_var, var['observable'], units='CGS')
        
        # Get record of all available reactor variables
        SIM.reactor_var = {}
        for key, val in SIM.all_var.items():
            if val['SIM_name'] in obs_dict:
                SIM.reactor_var[key] = SIM.all_var[key]

        return SIM, details
