# This file is part of Frhodo. Copyright © 2020, UChicago Argonne, LLC
# and licensed under BSD-3-Clause. See License.txt in the top-level 
# directory for license and copyright information.

import os, io, stat, contextlib, pathlib, time
from copy import deepcopy
import cantera as ct
from cantera import cti2yaml, ck2yaml
import numpy as np
from calculate import reactors, shock_fcns, integrate
from timeit import default_timer as timer


arrhenius_coefNames = ['activation_energy', 'pre_exponential_factor', 'temperature_exponent']


class Chemical_Mechanism:
    def __init__(self):
        self.isLoaded = False
        self.reactor = reactors.Reactor(self)

    def load_mechanism(self, path, silent=False):
        def chemkin2cantera(path):
            if path['thermo'] is not None:
                gas = ck2yaml.convert_mech(path['mech'], thermo_file=path['thermo'], transport_file=None, surface_file=None,
                    phase_name='gas', out_name=path['Cantera_Mech'], quiet=False, permissive=True)
            else:
                gas = ck2yaml.convert_mech(path['mech'], thermo_file=None, transport_file=None, surface_file=None,
                    phase_name='gas', out_name=path['Cantera_Mech'], quiet=False, permissive=True)
           
            return gas

        def loader(self, path):
            # path is assumed to be the path dictionary
            gas = []
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
                    gas = chemkin2cantera(path)
                          
            print('Validating mechanism...', end='')
            try:                                            # This test taken from ck2cti
                yaml_txt = path['Cantera_Mech'].read_text()
                self.gas = ct.Solution(yaml=yaml_txt)
                for gas_name in gas:
                    phase = ct.Interface(mech_path, gas_name, [self.gas])
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

            n_species = self.gas.n_species
            n_rxn = self.gas.n_reactions

            output['message'].append(f'Wrote YAML mechanism file to {path["Cantera_Mech"]}.')
            output['message'].append(f'Mechanism contains {n_species} species and {n_rxn} reactions.')
            
        for log_str in [ct_out, ct_err]:
            if log_str != '' and not silent:
                output['message'].append(log_str)
                output['message'].append('\n')
        
        if self.isLoaded:
            self.set_rate_expression_coeffs()   # set copy of coeffs
            self.set_thermo_expression_coeffs()                   # set copy of thermo coeffs

        return output
    
    def set_mechanism(self, mech_dict, species_dict={}, bnds=[]):
        def get_Arrhenius_parameters(entry):
            A = entry['pre_exponential_factor']
            b = entry['temperature_exponent']
            Ea = entry['activation_energy']

            return A, b, Ea

        if len(species_dict) == 0:
            species = self.gas.species()
        else:
            species = []
            for n in range(len(species_dict)):
                s_dict = species_dict[n]
                s = ct.Species(name=s_dict['name'], composition=s_dict['composition'],
                                  charge=s_dict['charge'], size=s_dict['size'])
                thermo = s_dict['type'](s_dict['T_low'], s_dict['T_high'], s_dict['P_ref'], s_dict['coeffs'])
                s.thermo = thermo

                species.append(s)
        
        # Set kinetics data
        rxns = []
        for rxnIdx in range(len(mech_dict)):
            if 'Arrhenius Reaction' == mech_dict[rxnIdx]['rxnType']:
                A, b, Ea = get_Arrhenius_parameters(mech_dict[rxnIdx]['rxnCoeffs'][0])
                rate = ct.ArrheniusRate(A, b, Ea)

                rxn = ct.Reaction(mech_dict[rxnIdx]['reactants'], mech_dict[rxnIdx]['products'], rate)

            elif 'Three Body Reaction' == mech_dict[rxnIdx]['rxnType']:
                A, b, Ea = get_Arrhenius_parameters(mech_dict[rxnIdx]['rxnCoeffs'][0])
                rate = ct.ArrheniusRate(A, b, Ea)
                
                rxn = ct.ThreeBodyReaction(mech_dict[rxnIdx]['reactants'], mech_dict[rxnIdx]['products'], rate)
                rxn.efficiencies = mech_dict[rxnIdx]['rxnCoeffs'][0]['efficiencies']

            elif 'Plog Reaction' == mech_dict[rxnIdx]['rxnType']:
                rates = []
                for plog in mech_dict[rxnIdx]['rxnCoeffs']:
                    pressure = plog['Pressure']
                    A, b, Ea = get_Arrhenius_parameters(plog)
                    rates.append((pressure, ct.Arrhenius(A, b, Ea)))

                rate = ct.PlogRate(rates)
                rxn = ct.Reaction(mech_dict[rxnIdx]['reactants'], mech_dict[rxnIdx]['products'], rate)

            elif 'Falloff Reaction' == mech_dict[rxnIdx]['rxnType']:
                # high pressure limit
                A, b, Ea = get_Arrhenius_parameters(mech_dict[rxnIdx]['rxnCoeffs']['high_rate'])
                high_rate = ct.Arrhenius(A, b, Ea)

                # low pressure limit
                A, b, Ea = get_Arrhenius_parameters(mech_dict[rxnIdx]['rxnCoeffs']['low_rate'])
                low_rate = ct.Arrhenius(A, b, Ea)

                # falloff parameters
                falloff_type = mech_dict[rxnIdx]['rxnCoeffs']['falloff_type']
                falloff_coeffs = mech_dict[rxnIdx]['rxnCoeffs']['falloff_parameters']

                if falloff_type == 'Lindemann':
                    rate = ct.LindemannRate(low_rate, high_rate, falloff_coeffs)

                elif falloff_type == 'Tsang':
                    rate = ct.TsangRate(low_rate, high_rate, falloff_coeffs)

                elif falloff_type == 'Troe':
                    if falloff_coeffs[-1] == 0.0:
                        falloff_coeffs = falloff_coeffs[0:-1]

                    rate = ct.TroeRate(low_rate, high_rate, falloff_coeffs)
                elif falloff_type == 'SRI':
                    rate = ct.SriRate(low_rate, high_rate, falloff_coeffs)

                rxn = ct.FalloffReaction(mech_dict[rxnIdx]['reactants'], mech_dict[rxnIdx]['products'], rate)

                rxn.efficiencies = mech_dict[rxnIdx]['rxnCoeffs']['efficiencies']

            elif 'Chebyshev Reaction' == mech_dict[rxnIdx]['rxnType']:
                rxn = ct.ChebyshevRate([mech_dict['Tmin'], mech_dict['Tmax']],
                                       [mech_dict['Pmin'], mech_dict['Pmax']],
                                       mech_dict['coeffs'])
                
                rxn = ct.Reaction(mech_dict[rxnIdx]['reactants'], mech_dict[rxnIdx]['products'], rate)
            
            rxn.duplicate = mech_dict[rxnIdx]['duplicate']
            rxn.reversible = mech_dict[rxnIdx]['reversible']
            rxn.allow_negative_orders = True
            rxn.allow_nonreactant_orders = True

            if hasattr(rxn, "allow_negative_pre_exponential_factor"):
                rxn.allow_negative_pre_exponential_factor = True

            rxns.append(rxn)
        
        self.gas = ct.Solution(thermo='IdealGas', kinetics='GasKinetics',
                               species=species, reactions=rxns)
        
        self.set_rate_expression_coeffs(bnds)   # set copy of coeffs
        self.set_thermo_expression_coeffs() # set copy of thermo coeffs
    
    def gas(self): return self.gas


    def reaction_type(self, rxn):
        if type(rxn.rate) is ct.ArrheniusRate:
            if rxn.reaction_type == "three-body":
                return "Three Body Reaction"
            else:
                return "Arrhenius Reaction"
        elif type(rxn.rate) is ct.PlogRate:
            return "Plog Reaction"
        elif type(rxn.rate) in [ct.FalloffRate, ct.LindemannRate, ct.TsangRate, ct.TroeRate, ct.SriRate]:
            return "Falloff Reaction"
        elif type(rxn.rate) is ct.ChebyshevRate:
            return "Chebyshev Reaction"        
        else:
            return str(type(rxn.rate))
        
    
    def set_rate_expression_coeffs(self, bnds=[]):
        def copy_bnds(new_bnds, bnds, rxnIdx, bnds_type, keys=[]):
            if len(bnds) == 0: return new_bnds

            if bnds_type == 'rate':
                for key in ['value', 'type', 'opt']:
                    new_bnds[rxnIdx][key] = bnds['rate_bnds'][rxnIdx][key]
            
            else:
                bndsKey, attrs = keys
                for coefName in attrs:
                    for key in ['value', 'type', 'opt']:
                        new_bnds[rxnIdx][bndsKey][coefName][key] = bnds['coeffs_bnds'][rxnIdx][bndsKey][coefName][key]

            return new_bnds

        self.coeffs = coeffs = []
        self.coeffs_bnds = coeffs_bnds = []
        self.rate_bnds = rate_bnds = []
        self.reset_mech = reset_mech = []

        for rxnIdx, rxn in enumerate(self.gas.reactions()):
            rate_bnds.append({'value': np.nan, 'limits': Uncertainty('rate', rxnIdx, rate_bnds=rate_bnds), 'type': 'F', 'opt': False})
            rate_bnds = copy_bnds(rate_bnds, bnds, rxnIdx, 'rate')

            rxn_type = self.reaction_type(rxn)

            if rxn_type in ["Arrhenius Reaction", "Three Body Reaction"]:
                coeffs.append([{attr: getattr(rxn.rate, attr) for attr in arrhenius_coefNames}])
                if rxn_type == "Three Body Reaction":
                    coeffs[-1][0]['efficiencies'] = rxn.third_body.efficiencies

                coeffs_bnds.append({'rate': {attr: {'resetVal': coeffs[-1][0][attr], 'value': np.nan, 
                                    'limits': Uncertainty('coef', rxnIdx, key='rate', coef_name=attr, coeffs_bnds=coeffs_bnds),
                                    'type': 'F', 'opt': False} for attr in arrhenius_coefNames}})
                
                coeffs_bnds = copy_bnds(coeffs_bnds, bnds, rxnIdx, 'coeffs', ['rate', arrhenius_coefNames])

                reset_mech.append({'reactants': rxn.reactants, 'products': rxn.products, 'rxnType': rxn_type,
                                    'duplicate': rxn.duplicate, 'reversible': rxn.reversible, 'orders': rxn.orders,
                                    'rxnCoeffs': deepcopy(coeffs[-1])})
                
            elif rxn_type == "Plog Reaction":
                coeffs.append([])
                coeffs_bnds.append({})
                for n, rate in enumerate(rxn.rate.rates):
                    coeffs[-1].append({'Pressure': rate[0]})
                    coeffs[-1][-1].update({attr: getattr(rate[1], attr) for attr in arrhenius_coefNames})
                    if n == 0 or n == len(rxn.rate.rates)-1: # only going to allow coefficient uncertainties to be placed on upper and lower pressures
                        if n == 0:
                            key = 'low_rate'
                        else:
                            key = 'high_rate'
                        coeffs_bnds[-1][key] = {attr: {'resetVal': coeffs[-1][-1][attr], 'value': np.nan, 
                                                        'limits': Uncertainty('coef', rxnIdx, key=key, coef_name=attr, coeffs_bnds=coeffs_bnds),
                                                        'type': 'F', 'opt': False} for attr in arrhenius_coefNames}

                        coeffs_bnds = copy_bnds(coeffs_bnds, bnds, rxnIdx, 'coeffs', [key, arrhenius_coefNames])

                reset_mech.append({'reactants': rxn.reactants, 'products': rxn.products, 'rxnType': rxn_type,
                                   'duplicate': rxn.duplicate, 'reversible': rxn.reversible, 'orders': rxn.orders,
                                   'rxnCoeffs': deepcopy(coeffs[-1])})

            elif rxn_type == "Falloff Reaction":
                coeffs_bnds.append({})
                fallof_type = rxn.reaction_type.split('-')[1]

                coeffs.append({'falloff_type': fallof_type, 'high_rate': [], 'low_rate': [], 'falloff_parameters': list(rxn.rate.falloff_coeffs), 
                               'default_efficiency': rxn.third_body.default_efficiency, 'efficiencies': rxn.third_body.efficiencies})
                for key in ['low_rate', 'high_rate']:
                    rate = getattr(rxn.rate, key)
                    coeffs[-1][key] = {attr: getattr(rate, attr) for attr in arrhenius_coefNames}

                    coeffs_bnds[-1][key] = {attr: {'resetVal': coeffs[-1][key][attr], 'value': np.nan, 
                                                    'limits': Uncertainty('coef', rxnIdx, key=key, coef_name=attr, coeffs_bnds=coeffs_bnds),
                                                    'type': 'F', 'opt': False} for attr in arrhenius_coefNames}

                    coeffs_bnds = copy_bnds(coeffs_bnds, bnds, rxnIdx, 'coeffs', [key, arrhenius_coefNames])

                key = 'falloff_parameters'
                n_coef = len(rxn.rate.falloff_coeffs)
                coeffs_bnds[-1][key] = {n: {'resetVal': coeffs[-1][key][n], 'value': np.nan, 
                                            'limits': Uncertainty('coef', rxnIdx, key=key, coef_name=n, coeffs_bnds=coeffs_bnds), 
                                            'type': 'F', 'opt': True} for n in range(0,n_coef)}

                reset_mech.append({'reactants': rxn.reactants, 'products': rxn.products, 'rxnType': rxn_type,
                                    'duplicate': rxn.duplicate, 'reversible': rxn.reversible, 'orders': rxn.orders,
                                    'falloffType': fallof_type, 'rxnCoeffs': deepcopy(coeffs[-1])})
            
            elif rxn_type == "Chebyshev Reaction":
                coeffs.append({})
                coeffs_bnds.append({})
                
                reset_coeffs = {'Pmin': rxn.rate.pressure_range[0], 'Pmax': rxn.rate.pressure_range[1], 
                                'Tmin': rxn.rate.temperature_range[0], 'Tmax': rxn.rate.temperature_range[1], 
                                'coeffs': rxn.rate.data}
                reset_mech.append({'reactants': rxn.reactants, 'products': rxn.products, 'rxnType': rxn_type,
                                    'duplicate': rxn.duplicate, 'reversible': rxn.reversible, 'orders': rxn.orders,
                                    'rxnCoeffs': reset_coeffs})

            else:
                coeffs.append({})
                coeffs_bnds.append({})
                reset_mech.append({'reactants': rxn.reactants, 'products': rxn.products, 'rxnType': rxn_type})
                msg = f'{rxn} is a {rxn_type} and is currently unsupported in Frhodo'
                raise(Exception(msg))
            

    def get_coeffs_keys(self, rxn, coefAbbr, rxnIdx=None):
        if type(rxn.rate) is ct.ArrheniusRate:
            bnds_key = 'rate'
            coef_key = 0

        elif type(rxn.rate) is ct.PlogRate:
            if 'high' in coefAbbr:
                if rxnIdx is None:  # get reaction index if not provided
                    for rxnIdx, mechRxn in enumerate(self.gas.reactions()):
                        if rxn is mechRxn:
                            break
                
                bnds_key = 'high_rate'
                coef_key = len(self.coeffs[rxnIdx]) - 1
            elif 'low' in coefAbbr:
                bnds_key = 'low_rate'
                coef_key = 0
                
        elif type(rxn.rate) in [ct.FalloffRate, ct.TroeRate, ct.SriRate]:
            if 'high' in coefAbbr:
                coef_key = bnds_key = 'high_rate'
            elif 'low' in coefAbbr:
                coef_key = bnds_key = 'low_rate'
            else:
                coef_key = bnds_key = 'falloff_parameters'

        return coef_key, bnds_key

    def set_thermo_expression_coeffs(self):         # TODO Doesn't work with NASA 9
        self.thermo_coeffs = []
        for i in range(self.gas.n_species):
            S = self.gas.species(i)
            thermo_dict = {'name': S.name, 'composition': S.composition, 'charge': S.charge,
                           'size': S.size, 'type': type(S.thermo), 'P_ref': S.thermo.reference_pressure,
                           'T_low': S.thermo.min_temp, 'T_high': S.thermo.max_temp,
                           'coeffs': np.array(S.thermo.coeffs),
                           'h_scaler': 1, 's_scaler': 1,}
            
            self.thermo_coeffs.append(thermo_dict)   
    
    def modify_reactions(self, coeffs, rxnIdxs=[]):     # Only works for Arrhenius equations currently
        if not rxnIdxs:                     # if rxnNums does not exist, modify all
            rxnIdxs = range(len(coeffs))
        else:
            if isinstance(rxnIdxs, (float, int)):  # if single reaction given, run that one
                rxnIdxs = [rxnIdxs]
        
        for rxnIdx in rxnIdxs:
            rxn = self.gas.reaction(rxnIdx)
            rxnChanged = False

            if type(rxn.rate) is ct.ArrheniusRate:
                for coefName in ['activation_energy', 'pre_exponential_factor', 'temperature_exponent']:
                    if coeffs[rxnIdx][0][coefName] != eval(f'rxn.rate.{coefName}'):
                        rxnChanged = True
                
                if rxnChanged:          # Update reaction rate
                    A = coeffs[rxnIdx][0]['pre_exponential_factor']
                    b = coeffs[rxnIdx][0]['temperature_exponent']
                    Ea = coeffs[rxnIdx][0]['activation_energy']
                    rxn.rate = ct.ArrheniusRate(A, b, Ea)

            elif type(rxn.rate) in [ct.FalloffRate, ct.TroeRate, ct.SriRate]:
                rate_dict = {'low_rate': None, 'high_rate': None, 'falloff_parameters': None}
                for key in rate_dict.keys():
                    if 'rate' in key:
                        for coefName in ['activation_energy', 'pre_exponential_factor', 'temperature_exponent']:
                            if coeffs[rxnIdx][key][coefName] != eval(f'rxn.rate.{key}.{coefName}'):
                                rxnChanged = True

                                A = coeffs[rxnIdx][key]['pre_exponential_factor']
                                b = coeffs[rxnIdx][key]['temperature_exponent']
                                Ea = coeffs[rxnIdx][key]['activation_energy']
                                rate_dict[key] = ct.Arrhenius(A, b, Ea)
                                break
                    else:
                        length_different = len(coeffs[rxnIdx][key]) != len(rxn.rate.falloff_coeffs)
                        if length_different or (coeffs[rxnIdx][key] != rxn.rate.falloff_coeffs).any():
                            rxnChanged = True

                            if coeffs[rxnIdx]['falloff_type'] == 'Troe':
                                if coeffs[rxnIdx][key][-1] == 0.0:
                                    rate_dict[key] = coeffs[rxnIdx][key][:-1]
                                else:
                                    rate_dict[key] = coeffs[rxnIdx][key]
                            else:   # could also be SRI. For optimization this would need to be cast as Troe
                                rate_dict[key] = ct.SriFalloff(coeffs[rxnIdx][key])
                
                if coeffs[rxnIdx]['falloff_type'] == 'Troe':
                    rate = ct.TroeRate(rate_dict['low_rate'], rate_dict['high_rate'], rate_dict['falloff_parameters'])
                else:
                    rate = ct.SriRate(rate_dict['low_rate'], rate_dict['high_rate'], rate_dict['falloff_parameters'])

                rxn.rate = rate

            elif type(rxn.rate) is ct.ChebyshevRate:
                 pass
            else:
                continue
            
            if rxnChanged:
                self.gas.modify_reaction(rxnIdx, rxn)

        # Not sure if this is necessary, but it reduces strange behavior in incident shock reactor
        #time.sleep(10E-3)        # TODO: if incident shock reactor is written in C++, this can likely be removed
    
    def rxn2Troe(self, rxnIdx, HPL, LPL, eff={}):
        reactants = self.gas.reaction(rxnIdx).reactants
        products = self.gas.reaction(rxnIdx).products
        r = ct.FalloffRate(reactants, products)
        print(r)
        #r.high_rate = ct.Arrhenius(7.4e10, -0.37, 0.0)
        #r.low_rate = ct.Arrhenius(2.3e12, -0.9, -1700*1000*4.184)
        #r.falloff = ct.TroeFalloff((0.7346, 94, 1756, 5182))
        #r.efficiencies = {'AR':0.7, 'H2':2.0, 'H2O':6.0}
        print(dir(self.gas))
        print(self.gas.thermo_model)
        print(self.gas.kinetics_model)

        start = timer()
        #self.gas.thermo_model
        #self.gas.kinetics_model
        self.gas = ct.Solution(thermo='IdealGas', kinetics='GasKinetics',
                               species=self.gas.species(), reactions=self.gas.reactions())
        print(timer() - start)

        start = timer()
        self.set_mechanism(self.reset_mech)
        print(timer() - start)

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
    
    def reset(self, rxnIdxs=None, coefNames=None):
        if rxnIdxs is None:
            rxnIdxs = range(self.gas.n_reactions)
        elif type(rxnIdxs) is not list: # if not list then assume given single rxnIdx
            rxnIdxs = [rxnIdxs]
        
        # if not list then assume given single coefficient
        # if Arrhenius, expects list of coefficients. If Plog or Falloff expected [['high_rate', 'activation_energy'], ...]
        if coefNames is not None and type(coefNames) is not list:
            coefNames = [coefNames]

        prior_coeffs = deepcopy(self.coeffs)
        for rxnIdx in rxnIdxs:
            if coefNames is None:   # resets all coefficients in rxn
                self.coeffs[rxnIdx] = self.reset_mech[rxnIdx]['rxnCoeffs']
            
            elif self.reset_mech[rxnIdx]['rxnType'] in ['Arrhenius Reaction', 'Three Body Reaction']:
                for coefName in coefNames:
                    self.coeffs[rxnIdx][coefName] = self.reset_mech[rxnIdx]['rxnCoeffs'][coefName]

            elif 'Plog Reaction' == self.reset_mech[rxnIdx]['rxnType']:
                for [limit_type, coefName] in coefNames:
                    if limit_type == 'low_rate':
                        self.coeffs[rxnIdx][0][coefName] = self.reset_mech[rxnIdx]['rxnCoeffs'][0][coefName]
                    elif limit_type == 'high_rate':
                        self.coeffs[rxnIdx][-1][coefName] = self.reset_mech[rxnIdx]['rxnCoeffs'][-1][coefName]

            elif 'Falloff Reaction' == self.reset_mech[rxnIdx]['rxnType']:
                self.coeffs[rxnIdx]['falloff_type'] = self.reset_mech[rxnIdx]['falloffType']
                for [limit_type, coefName] in coefNames:
                    self.coeffs[rxnIdx][limit_type][coefName] = self.reset_mech[rxnIdx]['rxnCoeffs'][limit_type][coefName]

        self.modify_reactions(self.coeffs)

        return prior_coeffs

    def set_TPX(self, T, P, X=[]):
        output = {'success': False, 'message': []}
        if T <= 0 or np.isnan(T):
            output['message'].append('Error: Temperature is invalid')
            return output

        elif P <= 0 or np.isnan(P):
            output['message'].append('Error: Pressure is invalid')
            return output

        elif len(X) > 0:
            for species in X:
                if species not in self.gas.species_names:
                    output['message'].append('Species: {:s} is not in the mechanism'.format(species))
                    return output
            
            self.gas.TPX = T, P, X

        else:
            self.gas.TP = T, P
            
        output['success'] = True
        return output

    def M(self, rxn, TPX=[]):   # kmol/m^3
        def get_M(rxn):
            M = self.gas.density_mole
            if rxn.third_body is not None:
                M *= rxn.third_body.default_efficiency
                for (s, conc) in zip(self.gas.species_names, self.gas.concentrations):
                    if s in rxn.third_body.efficiencies:
                        M += conc*(rxn.third_body.efficiencies[s] - 1.0)
                    else:
                        M += conc
            return M

        if len(TPX) == 0:
            M = get_M(rxn)
        else:
            [T, P, X] = TPX
            M = []
            for i in range(0, len(T)):
                self.set_TPX(T[i], P[i], X) # IF MIXTURE CHANGES THEN THIS NEEDS TO BE VARIABLE
                M.append(get_M(rxn))
            
            M = np.array(M)

        return M

    def run(self, reactor_choice, t_end, T_reac, P_reac, mix, **kwargs):
        return self.reactor.run(reactor_choice, t_end, T_reac, P_reac, mix, **kwargs)


class Uncertainty: # alternate name: why I hate pickle part 10
    def __init__(self, unc_type, rxnIdx, **kwargs):
        # self.gas = gas
        self.unc_type = unc_type
        self.rxnIdx = rxnIdx
        self.unc_dict = kwargs
    
    def _unc_fcn(self, x, uncVal, uncType): # uncertainty function
        if np.isnan(uncVal):
            return [np.nan, np.nan]
        elif uncType == 'F':
            return np.sort([x/uncVal, x*uncVal], axis=0)
        elif uncType == '%':
            return np.sort([x/(1 + uncVal), x*(1 + uncVal)], axis=0)
        elif uncType == '±':
            return np.sort([x - uncVal, x + uncVal], axis=0)
        elif uncType == '+':
            return np.sort([x, x + uncVal], axis=0)
        elif uncType == '-':
            return np.sort([x - uncVal, x], axis=0)

    def __call__(self, x=None):
        if self.unc_type == 'rate':
            #if x is None:    # defaults to giving current rate bounds
            #    x = self.gas.forward_rate_constants[self.rxnIdx]
            rate_bnds = self.unc_dict['rate_bnds']
            unc_value = rate_bnds[self.rxnIdx]['value']
            unc_type = rate_bnds[self.rxnIdx]['type']
            return self._unc_fcn(x, unc_value, unc_type)
        else:
            coeffs_bnds = self.unc_dict['coeffs_bnds']
            key = self.unc_dict['key']
            coefName = self.unc_dict['coef_name']

            if key == 'falloff_parameters': # falloff parameters have no limits
                return [np.nan, np.nan]

            coef_dict = coeffs_bnds[self.rxnIdx][key][coefName]
            coef_val = coef_dict['resetVal']
            unc_value = coef_dict['value']
            unc_type = coef_dict['type']
            return self._unc_fcn(coef_val, unc_value, unc_type)