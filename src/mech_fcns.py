# This file is part of Frhodo. Copyright © 2020, UChicago Argonne, LLC
# and licensed under BSD-3-Clause. See License.txt in the top-level 
# directory for license and copyright information.

import os, io, stat, contextlib, pathlib, time
import cantera as ct
from cantera import interrupts, cti2yaml#, ck2yaml, ctml2yaml
import numpy as np
import integrate, shock_fcns, ck2yaml
from timeit import default_timer as timer

# list of all possible variables
all_var = {'Laboratory Time':                  {'SIM_name': 't_lab',        'sub_type':  None},
           'Shockwave Time':                   {'SIM_name': 't_shock',      'sub_type':  None}, 
           'Gas Velocity':                     {'SIM_name': 'vel',          'sub_type':  None}, 
           'Temperature':                      {'SIM_name': 'T',            'sub_type':  None}, 
           'Pressure':                         {'SIM_name': 'P',            'sub_type':  None},              
           'Enthalpy':                         {'SIM_name': 'h',            'sub_type':  ['total', 'species']},
           'Entropy':                          {'SIM_name': 's',            'sub_type':  ['total', 'species']}, 
           'Density':                          {'SIM_name': 'rho',          'sub_type':  None},       
           'Density Gradient':                 {'SIM_name': 'drhodz',       'sub_type':  ['total', 'rxn']},
           '% Density Gradient':               {'SIM_name': 'perc_drhodz',  'sub_type':  ['rxn']},
           'Mole Fraction':                    {'SIM_name': 'X',            'sub_type':  ['species']}, 
           'Mass Fraction':                    {'SIM_name': 'Y',            'sub_type':  ['species']}, 
           'Concentration':                    {'SIM_name': 'conc',         'sub_type':  ['species']}, 
           'Net Production Rate':              {'SIM_name': 'wdot',         'sub_type':  ['species']},
           'Creation Rate':                    {'SIM_name': 'wdotfor',      'sub_type':  ['species']},      
           'Destruction Rate':                 {'SIM_name': 'wdotrev',      'sub_type':  ['species']},
           'Heat Release Rate':                {'SIM_name': 'HRR',          'sub_type':  ['total', 'rxn']},
           'Delta Enthalpy (Heat of Reaction)':{'SIM_name': 'delta_h',      'sub_type':  ['rxn']},
           'Delta Entropy':                    {'SIM_name': 'delta_s',      'sub_type':  ['rxn']},  
           'Equilibrium Constant':             {'SIM_name': 'eq_con',       'sub_type':  ['rxn']}, 
           'Forward Rate Constant':            {'SIM_name': 'rate_con',     'sub_type':  ['rxn']}, 
           'Reverse Rate Constant':            {'SIM_name': 'rate_con_rev', 'sub_type':  ['rxn']},          
           'Net Rate of Progress':             {'SIM_name': 'net_ROP',      'sub_type':  ['rxn']}, 
           'Forward Rate of Progress':         {'SIM_name': 'for_ROP',      'sub_type':  ['rxn']},        
           'Reverse Rate of Progress':         {'SIM_name': 'rev_ROP',      'sub_type':  ['rxn']}}

rev_all_var = {all_var[key]['SIM_name']: 
               {'name': key, 'sub_type': all_var[key]['sub_type']} for key in all_var.keys()}

# translation dictionary between SIM name and ct.SolutionArray name
SIM_Dict = {'t_lab': 't', 't_shock': 't_shock', 'z': 'z', 'A': 'A', 'vel': 'vel', 'T': 'T', 'P': 'P', 
            'h_tot': 'enthalpy_mole', 'h': 'partial_molar_enthalpies', 
            's_tot': 'entropy_mole', 's': 'partial_molar_entropies',
            'rho': 'density', 'drhodz_tot': 'drhodz_tot', 'drhodz': 'drhodz', 'perc_drhodz': 'perc_drhodz',
            'Y': 'Y', 'X': 'X', 'conc': 'concentrations', 'wdot': 'net_production_rates', 
            'wdotfor': 'creation_rates', 'wdotrev': 'destruction_rates', 
            'HRR_tot': 'heat_release_rate', 'HRR': 'heat_production_rates',
            'delta_h': 'delta_enthalpy', 'delta_s': 'delta_entropy', 
            'eq_con': 'equilibrium_constants', 'rate_con': 'forward_rate_constants', 
            'rate_con_rev': 'reverse_rate_constants', 'net_ROP': 'net_rates_of_progress',
            'for_ROP': 'forward_rates_of_progress', 'rev_ROP': 'reverse_rates_of_progress'}

class SIM_Property:
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.conversion = None  # this needs to be assigned per property
        self.value = {'SI': np.array([]), 'CGS': np.array([])}
        self.ndim = self.value['SI'].ndim

    def clear(self):
        self.value = {'SI': np.array([]), 'CGS': np.array([])}
        self.ndim = self.value['SI'].ndim

    def __call__(self, idx=None, units='CGS'): # units must be 'CGS' or 'SI'
        # assumes Sim data comes in as SI and is converted to CGS
        # values to be calculated post-simulation
        if len(self.value['SI']) == 0 or np.isnan(self.value['SI']).all():
            parent = self.parent
            if self.name == 'drhodz_tot':
                self.value['SI'] = shock_fcns.drhodz(parent.states)
            elif self.name == 'drhodz':
                self.value['SI'] = shock_fcns.drhodz_per_rxn(parent.states)
            elif self.name == 'perc_drhodz':
                self.value['SI'] = parent.drhodz(units='SI').T*100/parent.drhodz_tot(units='SI')[:,None]
            else:
                self.value['SI'] = getattr(parent.states, SIM_Dict[self.name])

            if self.value['SI'].ndim > 1: # Transpose if matrix
                self.value['SI'] = self.value['SI'].T

            self.ndim = self.value['SI'].ndim

        # currently converts entire list of properties rather than by index
        if units == 'CGS' and len(self.value['CGS']) == 0:
            if self.conversion is None:
                self.value['CGS'] = self.value['SI']
            else:
                self.value['CGS'] = self.conversion(self.value['SI'])

        return self.value[units]


class Simulation_Result:
    def __init__(self, num=None, states=None, reactor_vars=[]):
        self.states = states
        self.all_var = all_var
        self.rev_all_var = rev_all_var
        self.reactor_var = {}
        for var in reactor_vars:
            if var in self.rev_all_var:
                self.reactor_var[self.rev_all_var[var]['name']] = var

        if num is None: # if no simulation stop here
            self.reactor_var = {}
            return

        self.conv = {'conc': 1E-3, 'wdot': 1E-3, 'P': 760/101325, 'vel': 1E2, 
                     'rho': 1E-3, 'drhodz_tot': 1E-5, 'drhodz':  1E-5,  
                     'delta_h': 1E-3/4184, 'h_tot': 1E-3/4184, 'h': 1E-3/4184, # to kcal
                     'delta_s': 1/4184,    's_tot': 1/4184,    's': 1/4184, 
                     'eq_con': 1E3**np.array(num['reac'] - num['prod'])[:,None],
                     'rate_con': np.power(1E3,num['reac']-1)[:,None],
                     'rate_con_rev': np.power(1E3,num['prod']-1)[:,None],
                     'net_ROP':        1E-3/3.8,    # Don't understand 3.8 value
                     'for_ROP':        1E-3/3.8,    # Don't understand 3.8 value
                     'rev_ROP':        1E-3/3.8}    # Don't understand 3.8 value

        for name in reactor_vars:
            property = SIM_Property(name, parent=self)
            if name in self.conv:
                property.conversion = lambda x, s=self.conv[name]: x*s
            setattr(self, name, property)

    def set_independent_var(self, ind_var, units='CGS'):
        self.independent_var = getattr(self, ind_var)(units=units)

    def set_observable(self, observable, units='CGS'):
        k = observable['sub']
        if observable['main'] == 'Temperature':
            self.observable = self.T(units=units)
        elif observable['main'] == 'Pressure':
            self.observable = self.P(units=units)
        elif observable['main'] == 'Density Gradient':
            self.observable = self.drhodz_tot(units=units)
        elif observable['main'] == 'Heat Release Rate':
            self.observable = self.HRR_tot(units=units)
        elif observable['main'] == 'Mole Fraction':
            self.observable = self.X(units=units)
        elif observable['main'] == 'Mass Fraction':
            self.observable = self.Y(units=units)
        elif observable['main'] == 'Concentration':
            self.observable = self.conc(units=units)

        if self.observable.ndim > 1:                # reduce observable down to only plotted information
            self.observable = self.observable[k]

    def finalize(self, success, ind_var, observable, units='CGS'):  
        self.set_independent_var(ind_var, units)
        self.set_observable(observable, units)
        
        self.success = success
       

class Chemical_Mechanism:
    def __init__(self):
        self.isLoaded = False
        self.reactor = Reactor(self)

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
                self.yaml_txt = path['Cantera_Mech'].read_text()        # Storing full text could be bad if large
                self.gas = ct.Solution(yaml=self.yaml_txt)
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
                if (path['Cantera_Mech'], pathlib.WindowsPath): # reformat string to remove \\ making it unable to be copy paste
                    cantera_path = str(path['Cantera_Mech']).replace('\\', '\\\\')
                    log_str = log_str.replace(cantera_path, str(path['Cantera_Mech']))
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
    
    def set_mechanism(self, mech_txt):
        self.gas = ct.Solution(yaml=mech_txt)
        
        self.set_rate_expression_coeffs()   # set copy of coeffs
        self.set_thermo_expression_coeffs() # set copy of thermo coeffs
    
    def gas(self): return self.gas       
    
    def set_rate_expression_coeffs(self):
        coeffs = []
        coeffs_bnds = []
        rate_bnds = []
        for rxnNum, rxn in enumerate(self.gas.reactions()):
            if hasattr(rxn, 'rate'):
                attrs = [p for p in dir(rxn.rate) if not p.startswith('_')] # attributes not including __              
                coeffs.append({attr: getattr(rxn.rate, attr) for attr in attrs})
                coeffs_bnds.append({attr: {'resetVal': getattr(rxn.rate, attr), 
                                           'value': np.nan, 'type': 'F'} for attr in attrs})
                for coef_name in coeffs_bnds[-1].keys():
                    coeffs_bnds[-1][coef_name]['limits'] = Uncertainty('coef', rxnNum, 
                                                                       coef_name=coef_name, coeffs_bnds=coeffs_bnds)
                
                rate_bnds.append({'value': np.nan, 'limits': None, 'type': 'F', 'opt': False})
                rate_bnds[-1]['limits'] = Uncertainty('rate', rxnNum, rate_bnds=rate_bnds)
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
            rxnChanged = False
            if type(rxn) is ct.ElementaryReaction or type(rxn) is ct.ThreeBodyReaction:
                for coefName in ['activation_energy', 'pre_exponential_factor', 'temperature_exponent']:
                    if coeffs[rxnNum][coefName] != eval(f'rxn.rate.{coefName}'):
                        rxnChanged = True
                
                if rxnChanged:          # Update reaction rate
                    A = coeffs[rxnNum]['pre_exponential_factor']
                    b = coeffs[rxnNum]['temperature_exponent']
                    Ea = coeffs[rxnNum]['activation_energy']
                    rxn.rate = ct.Arrhenius(A, b, Ea)
            # elif type(rxn) is ct.PlogReaction:
                # print(dir(rxn))
                # print(rxn.rates[rxn_num])
            # elif type(rxn) is ct.ChebyshevReaction: 
                # print(dir(rxn))
                # print(rxn.rates[rxn_num])
            else:
                continue
            
            if rxnChanged:
                self.gas.modify_reaction(rxnNum, rxn)

        time.sleep(5E-3)        # Not sure if this is necessary, but it reduces strange behavior in incident shock reactor
    
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

    def run(self, reactor_choice, t_end, T_reac, P_reac, mix, **kwargs):
        return self.reactor.run(reactor_choice, t_end, T_reac, P_reac, mix, **kwargs)


class Uncertainty: # alternate name: why I hate pickle part 10
    def __init__(self, unc_type, rxnNum, **kwargs):
        # self.gas = gas
        self.unc_type = unc_type
        self.rxnNum = rxnNum
        self.unc_dict = kwargs
    
    def unc_fcn(self, x, uncVal, uncType): # uncertainty function
        if np.isnan(uncVal):
            return [np.nan, np.nan]
        elif uncType == 'F':
            return np.sort([x/uncVal, x*uncVal])
        elif uncType == '%':
            return np.sort([x/(1+uncVal), x*(1+uncVal)])
        elif uncType == '±':
            return np.sort([x-uncVal, x+uncVal])
        elif uncType == '+':
            return np.sort([x, x+uncVal])
        elif uncType == '-':
            return np.sort([x-uncVal, x])

    def __call__(self, x=None):
        if self.unc_type == 'rate':
            #if x is None:    # defaults to giving current rate bounds
            #    x = self.gas.forward_rate_constants[self.rxnNum]
            rate_bnds = self.unc_dict['rate_bnds']
            unc_value = rate_bnds[self.rxnNum]['value']
            unc_type = rate_bnds[self.rxnNum]['type']
            return self.unc_fcn(x, unc_value, unc_type)
        else:
            coeffs_bnds = self.unc_dict['coeffs_bnds']
            coefName = self.unc_dict['coef_name']
            coef_dict = coeffs_bnds[self.rxnNum][coefName]
            coef_val = coef_dict['resetVal']
            unc_value = coef_dict['value']
            unc_type = coef_dict['type']
            return self.unc_fcn(coef_val, unc_value, unc_type)
  

class Reactor:
    def __init__(self, mech):
        self.mech = mech
        self.ODE_success = False

    def run(self, reactor_choice, t_end, T_reac, P_reac, mix, **kwargs):
        def list2ct_mixture(mix):   # list in the form of [[species, mol_frac], [species, mol_frac],...]
            return ', '.join("{!s}:{!r}".format(species, mol_frac) for (species, mol_frac) in mix)
        
        details = {'success': False, 'message': []}
        
        if isinstance(mix, list):
            mix = list2ct_mixture(mix)
                
        mech_out = self.mech.set_TPX(T_reac, P_reac, mix)
        if not mech_out['success']:
            details['success'] = False
            details['message'] = mech_out['message']
            return None, mech_out
        
        #start = timer()
        if reactor_choice == 'Incident Shock Reactor':
            SIM, details =  self.incident_shock_reactor(self.mech.gas, details, t_end, **kwargs)
        elif '0d Reactor' in reactor_choice:
            if reactor_choice == '0d Reactor - Constant Volume':
                reactor = ct.IdealGasReactor(self.mech.gas)
            elif reactor_choice == '0d Reactor - Constant Pressure':
                reactor = ct.IdealGasConstPressureReactor(self.mech.gas)
            
            SIM, details = self.zero_d_ideal_gas_reactor(self.mech.gas, reactor, details, t_end, **kwargs)
        
        #print('{:0.1f} us'.format((timer() - start)*1E3))
        return SIM, details
    
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

    def incident_shock_reactor(self, gas, details, t_end, **kwargs):
        if 'u_reac' not in kwargs or 'rho1' not in kwargs:
            details['success'] = False
            details['message'] = 'velocity and rho1 not specified\n'
            return None, details
        
        # set default values
        var = {'sim_int_f': 1, 'observable': {'main': 'Density Gradient', 'sub': 0},
               'A1': 0.2, 'As': 0.2, 'L': 0.1, 't_lab_save': None, 
               'ODE_solver': 'BDF', 'rtol': 1E-4, 'atol': 1E-7}  
        var.update(kwargs)
        
        y0 = np.hstack((0.0, var['A1'], gas.density, var['u_reac'], gas.T, 0.0, gas.Y))   # Initial condition
        ode = shock_fcns.ReactorOde(gas, t_end, var['rho1'], var['L'], var['As'], var['A1'], False)

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
        
        ind_var = 't_lab'       # INDEPENDENT VARIABLE CURRENTLY HARDCODED FOR t_lab
        if var['t_lab_save'] is None:  # if t_save is not being sent, only plotting variables are needed
            t_all = t_sim
        else:
            t_all = np.sort(np.unique(np.concatenate((t_sim, var['t_lab_save'])))) # combine t_all and t_save, sort, only unique values
        
        states = ct.SolutionArray(gas, extra=['t', 't_shock', 'z', 'A', 'vel', 'drhodz_tot', 'drhodz', 'perc_drhodz'])
        for i, t in enumerate(t_all):   # calculate from solution
            y = sol.sol(t)  
            z, A, rho, v, T, t_shock = y[0:6]
            Y = y[6:]

            states.append(TDY=(T, rho, Y), t=t, t_shock=t_shock, z=z, A=A, vel=v, drhodz_tot=np.nan, drhodz=np.nan, perc_drhodz=np.nan)
        
        reactor_vars = ['t_lab', 't_shock', 'z', 'A', 'vel', 'T', 'P', 'h_tot', 'h', 
                        's_tot', 's', 'rho', 'drhodz_tot', 'drhodz', 'perc_drhodz',
                        'Y', 'X', 'conc', 'wdot', 'wdotfor', 'wdotrev', 
                        'HRR_tot', 'HRR', 'delta_h', 'delta_s', 
                        'eq_con', 'rate_con', 'rate_con_rev', 'net_ROP', 'for_ROP', 'rev_ROP']

        num = {'reac': np.sum(gas.reactant_stoich_coeffs(), axis=0),
               'prod': np.sum(gas.product_stoich_coeffs(), axis=0),
               'rxns': gas.n_reactions}
        
        SIM = Simulation_Result(num, states, reactor_vars)
        SIM.finalize(self.ODE_success, ind_var, var['observable'], units='CGS')
                
        return SIM, details
     
    def zero_d_ideal_gas_reactor(self, gas, reactor, details, t_end, **kwargs):
        # set default values
        var = {'sim_int_f': 1, 'observable': {'main': 'Concentration', 'sub': 0},
               't_lab_save': None, 'rtol': 1E-4, 'atol': 1E-7}
        
        var.update(kwargs)
        
        # Modify reactor if necessary for frozen composition and isothermal
        reactor.energy_enabled = var['solve_energy']
        reactor.chemistry_enabled = not var['frozen_comp']
        
        # Create Sim
        sim = ct.ReactorNet([reactor])
        sim.atol = var['atol']
        sim.rtol = var['rtol']
        
        # set up times and observables
        ind_var = 't_lab'       # INDEPENDENT VARIABLE CURRENTLY HARDCODED FOR t_lab
        if var['t_lab_save'] is None:
            t_all = [t_end]
        else:
            t_all = np.sort(np.unique(np.concatenate(([t_end], var['t_lab_save'])))) # combine t_end and t_save, sort, only unique values
        
        states = ct.SolutionArray(gas, extra=['t'])
        states.append(reactor.thermo.state, t = 0.0)
        for t in t_all:
            while sim.time < t:     # integrator step until time > target time
                sim.step()
                if sim.time > t:    # force interpolation to target time
                    sim.advance(t)
                states.append(reactor.thermo.state, t=sim.time)
        
        self.ODE_success = True         # TODO: NEED REAL ERROR CHECKING OF REACTOR SUCCESS
        details['success'] = True
        
        reactor_vars = ['t_lab', 'T', 'P', 'h_tot', 'h', 's_tot', 's', 'rho', 
                        'Y', 'X', 'conc', 'wdot', 'wdotfor', 'wdotrev', 'HRR_tot', 'HRR',
                        'delta_h', 'delta_s', 'eq_con', 'rate_con', 'rate_con_rev', 
                        'net_ROP', 'for_ROP', 'rev_ROP']

        num = {'reac': np.sum(gas.reactant_stoich_coeffs(), axis=0),
               'prod': np.sum(gas.product_stoich_coeffs(), axis=0),
               'rxns': gas.n_reactions}
        
        SIM = Simulation_Result(num, states, reactor_vars)
        SIM.finalize(self.ODE_success, ind_var, var['observable'], units='CGS')

        return SIM, details