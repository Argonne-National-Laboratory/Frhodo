# This file is part of Frhodo. Copyright © 2020, UChicago Argonne, LLC
# and licensed under BSD-3-Clause. See License.txt in the top-level 
# directory for license and copyright information.

import os, io, stat, contextlib, pathlib, time
from copy import deepcopy
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Tuple, Optional, Sequence, Union

import cantera as ct
from cantera import interrupts, cti2yaml  # , ck2yaml, ctml2yaml
import numpy as np
from timeit import default_timer as timer

from . import reactors, shock_fcns, integrate
from .reactors import Simulation_Result
from .. import ck2yaml, soln2ck


class Chemical_Mechanism:
    def __init__(self):
        self.isLoaded = False
        self.reactor = reactors.Reactor(self)

    def __getstate__(self):
        state = self.__dict__.copy()
        # Save the current state of the gas as Chemkin Mech file
        #  TODO: Figure out why we cannot save it to YML
        with NamedTemporaryFile() as fp:
            fp.close()
            soln2ck.write(self.gas, fp.name)
            with open(fp.name) as fp:
                state['gas'] = fp.read()
        return state

    def __setstate__(self, state):
        # Replace "gas" with a Cantera object
        with TemporaryDirectory() as tmp:
            # Convert the ChemKin output to YAML
            #  First write the mechanism to disk as a chemkin file
            tmp = Path(tmp)
            ck_file = tmp / 'gas.mech'
            with ck_file.open('w') as fp:
                print(state.pop('gas'), file=fp)

            #  Now convert it to a YAML
            ct_file = tmp / 'gas.yaml'
            ck2yaml.convert_mech(ck_file, thermo_file=None, transport_file=None, surface_file=None,
                                 phase_name='gas', out_name=ct_file, quiet=True,
                                 permissive=True)
            state['gas'] = ct.Solution(yaml=ct_file.read_text())

        self.__dict__.update(state)

    def load_mechanism(self, path, silent=False):
        def chemkin2cantera(path):
            if path['thermo'] is not None:
                surfaces = ck2yaml.convert_mech(path['mech'], thermo_file=path['thermo'], transport_file=None,
                                                surface_file=None,
                                                phase_name='gas', out_name=path['Cantera_Mech'], quiet=False,
                                                permissive=True)
            else:
                surfaces = ck2yaml.convert_mech(path['mech'], thermo_file=None, transport_file=None, surface_file=None,
                                                phase_name='gas', out_name=path['Cantera_Mech'], quiet=False,
                                                permissive=True)

            return surfaces

        def loader(self, path):
            # path is assumed to be the path dictionary
            surfaces = []
            if path['mech'].suffix in ['.yaml', '.yml']:  # check if it's a yaml cantera file
                mech_path = str(path['mech'])
            else:  # if not convert into yaml cantera file
                mech_path = str(path['Cantera_Mech'])

                if path['mech'].suffix == '.cti':
                    cti2yaml.convert(path['mech'], path['Cantera_Mech'])
                elif path['mech'].suffix in ['.ctml', '.xml']:
                    raise Exception('not implemented')
                    # ctml2yaml.convert(path['mech'], path['Cantera_Mech'])
                else:  # if not a cantera file, assume chemkin
                    surfaces = chemkin2cantera(path)

            print('Validating mechanism...', end='')
            try:  # This test taken from ck2cti
                yaml_txt = path['Cantera_Mech'].read_text()
                self.gas = ct.Solution(yaml=yaml_txt)
                for surfname in surfaces:
                    phase = ct.Interface(mech_path, surfname, [self.gas])
                print('PASSED.')
            except RuntimeError as e:
                print('FAILED.')
                print(e)

        output = {'success': False, 'message': []}
        # Initialize and report any problems to log, not to console window
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
                if (path['Cantera_Mech'],
                    pathlib.WindowsPath):  # reformat string to remove \\ making it unable to be copy paste
                    cantera_path = str(path['Cantera_Mech']).replace('\\', '\\\\')
                    log_str = log_str.replace(cantera_path, str(path['Cantera_Mech']))
                output['message'].append(log_str)
                output['message'].append('\n')

        if self.isLoaded:
            self.set_rate_expression_coeffs()  # set copy of coeffs
            self.set_thermo_expression_coeffs()  # set copy of thermo coeffs

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
            if 'ElementaryReaction' == mech_dict[rxnIdx]['rxnType']:
                rxn = ct.ElementaryReaction(mech_dict[rxnIdx]['reactants'], mech_dict[rxnIdx]['products'])
                rxn.allow_negative_pre_exponential_factor = True

                A, b, Ea = get_Arrhenius_parameters(mech_dict[rxnIdx]['rxnCoeffs'][0])
                rxn.rate = ct.Arrhenius(A, b, Ea)

            elif 'ThreeBodyReaction' == mech_dict[rxnIdx]['rxnType']:
                rxn = ct.ThreeBodyReaction(mech_dict[rxnIdx]['reactants'], mech_dict[rxnIdx]['products'])

                A, b, Ea = get_Arrhenius_parameters(mech_dict[rxnIdx]['rxnCoeffs'][0])
                rxn.rate = ct.Arrhenius(A, b, Ea)
                rxn.efficiencies = mech_dict[rxnIdx]['rxnCoeffs'][0]['efficiencies']

            elif 'PlogReaction' == mech_dict[rxnIdx]['rxnType']:
                rxn = ct.PlogReaction(mech_dict[rxnIdx]['reactants'], mech_dict[rxnIdx]['products'])

                rates = []
                for plog in mech_dict[rxnIdx]['rxnCoeffs']:
                    pressure = plog['Pressure']
                    A, b, Ea = get_Arrhenius_parameters(plog)
                    rates.append((pressure, ct.Arrhenius(A, b, Ea)))

                rxn.rates = rates

            elif 'FalloffReaction' == mech_dict[rxnIdx]['rxnType']:
                rxn = ct.FalloffReaction(mech_dict[rxnIdx]['reactants'], mech_dict[rxnIdx]['products'])

                # high pressure limit
                A, b, Ea = get_Arrhenius_parameters(mech_dict[rxnIdx]['rxnCoeffs']['high_rate'])
                rxn.high_rate = ct.Arrhenius(A, b, Ea)

                # low pressure limit
                A, b, Ea = get_Arrhenius_parameters(mech_dict[rxnIdx]['rxnCoeffs']['low_rate'])
                rxn.low_rate = ct.Arrhenius(A, b, Ea)

                # falloff parameters
                if mech_dict[rxnIdx]['rxnCoeffs']['falloff_type'] == 'Troe':
                    falloff_param = mech_dict[rxnIdx]['rxnCoeffs']['falloff_parameters']
                    if falloff_param[-1] == 0.0:
                        falloff_param = falloff_param[0:-1]

                    rxn.falloff = ct.TroeFalloff(falloff_param)
                else:
                    rxn.falloff = ct.SriFalloff(mech_dict[rxnIdx]['rxnCoeffs']['falloff_parameters'])

                rxn.efficiencies = mech_dict[rxnIdx]['rxnCoeffs']['efficiencies']

            elif 'ChebyshevReaction' == mech_dict[rxnIdx]['rxnType']:
                rxn = ct.ChebyshevReaction(mech_dict[rxnIdx]['reactants'], mech_dict[rxnIdx]['products'])
                rxn.set_parameters(Tmin=mech_dict['Tmin'], Tmax=mech_dict['Tmax'],
                                   Pmin=mech_dict['Pmin'], Pmax=mech_dict['Pmax'],
                                   coeffs=mech_dict['coeffs'])

            rxn.duplicate = mech_dict[rxnIdx]['duplicate']
            rxn.reversible = mech_dict[rxnIdx]['reversible']
            rxn.allow_negative_orders = True
            rxn.allow_nonreactant_orders = True

            rxns.append(rxn)

        self.gas = ct.Solution(thermo='IdealGas', kinetics='GasKinetics',
                               species=species, reactions=rxns)

        self.set_rate_expression_coeffs(bnds)  # set copy of coeffs
        self.set_thermo_expression_coeffs()  # set copy of thermo coeffs

    def gas(self):
        return self.gas

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
            rate_bnds.append({'value': np.nan, 'limits': Uncertainty('rate', rxnIdx, rate_bnds=rate_bnds), 'type': 'F',
                              'opt': False})
            rate_bnds = copy_bnds(rate_bnds, bnds, rxnIdx, 'rate')
            if type(rxn) in [ct.ElementaryReaction, ct.ThreeBodyReaction]:
                attrs = [p for p in dir(rxn.rate) if not p.startswith('_')]  # attributes not including __
                coeffs.append([{attr: getattr(rxn.rate, attr) for attr in attrs}])
                if type(rxn) is ct.ThreeBodyReaction:
                    coeffs[-1][0]['efficiencies'] = rxn.efficiencies

                coeffs_bnds.append({'rate': {attr: {'resetVal': coeffs[-1][0][attr], 'value': np.nan,
                                                    'limits': Uncertainty('coef', rxnIdx, key='rate', coef_name=attr,
                                                                          coeffs_bnds=coeffs_bnds),
                                                    'type': 'F', 'opt': False} for attr in attrs}})

                coeffs_bnds = copy_bnds(coeffs_bnds, bnds, rxnIdx, 'coeffs', ['rate', attrs])

                reset_mech.append(
                    {'reactants': rxn.reactants, 'products': rxn.products, 'rxnType': rxn.__class__.__name__,
                     'duplicate': rxn.duplicate, 'reversible': rxn.reversible, 'orders': rxn.orders,
                     'rxnCoeffs': deepcopy(coeffs[-1])})

            elif type(rxn) is ct.PlogReaction:
                coeffs.append([])
                coeffs_bnds.append({})
                for n, rate in enumerate(rxn.rates):
                    attrs = [p for p in dir(rate[1]) if not p.startswith('_')]  # attributes not including __
                    coeffs[-1].append({'Pressure': rate[0]})
                    coeffs[-1][-1].update({attr: getattr(rate[1], attr) for attr in attrs})
                    if n == 0 or n == len(
                            rxn.rates) - 1:  # only going to allow coefficient uncertainties to be placed on upper and lower pressures
                        if n == 0:
                            key = 'low_rate'
                        else:
                            key = 'high_rate'
                        coeffs_bnds[-1][key] = {attr: {'resetVal': coeffs[-1][-1][attr], 'value': np.nan,
                                                       'limits': Uncertainty('coef', rxnIdx, key=key, coef_name=attr,
                                                                             coeffs_bnds=coeffs_bnds),
                                                       'type': 'F', 'opt': False} for attr in attrs}

                        coeffs_bnds = copy_bnds(coeffs_bnds, bnds, rxnIdx, 'coeffs', [key, attrs])

                reset_mech.append(
                    {'reactants': rxn.reactants, 'products': rxn.products, 'rxnType': rxn.__class__.__name__,
                     'duplicate': rxn.duplicate, 'reversible': rxn.reversible, 'orders': rxn.orders,
                     'rxnCoeffs': deepcopy(coeffs[-1])})

            elif type(rxn) is ct.FalloffReaction:
                coeffs_bnds.append({})
                coeffs.append({'falloff_type': rxn.falloff.type, 'high_rate': [], 'low_rate': [],
                               'falloff_parameters': list(rxn.falloff.parameters),
                               'default_efficiency': rxn.default_efficiency, 'efficiencies': rxn.efficiencies})
                for key in ['low_rate', 'high_rate']:
                    rate = getattr(rxn, key)
                    attrs = [p for p in dir(rate) if not p.startswith('_')]  # attributes not including __
                    coeffs[-1][key] = {attr: getattr(rate, attr) for attr in attrs}

                    coeffs_bnds[-1][key] = {attr: {'resetVal': coeffs[-1][key][attr], 'value': np.nan,
                                                   'limits': Uncertainty('coef', rxnIdx, key=key, coef_name=attr,
                                                                         coeffs_bnds=coeffs_bnds),
                                                   'type': 'F', 'opt': False} for attr in attrs}

                    coeffs_bnds = copy_bnds(coeffs_bnds, bnds, rxnIdx, 'coeffs', [key, attrs])

                key = 'falloff_parameters'
                n_coef = len(rxn.falloff.parameters)
                coeffs_bnds[-1][key] = {n: {'resetVal': coeffs[-1][key][n], 'value': np.nan,
                                            'limits': Uncertainty('coef', rxnIdx, key=key, coef_name=n,
                                                                  coeffs_bnds=coeffs_bnds),
                                            'type': 'F', 'opt': True} for n in range(0, n_coef)}

                reset_mech.append(
                    {'reactants': rxn.reactants, 'products': rxn.products, 'rxnType': rxn.__class__.__name__,
                     'duplicate': rxn.duplicate, 'reversible': rxn.reversible, 'orders': rxn.orders,
                     'falloffType': rxn.falloff.type, 'rxnCoeffs': deepcopy(coeffs[-1])})

            elif type(rxn) is ct.ChebyshevReaction:
                coeffs.append({})
                coeffs_bnds.append({})
                if len(bnds) == 0:
                    rate_bnds.append({})

                reset_coeffs = {'Pmin': rxn.Pmin, 'Pmax': rxn.Pmax, 'Tmin': rxn.Tmin, 'Tmax': rxn.Tmax,
                                'coeffs': rxn.coeffs}
                reset_mech.append(
                    {'reactants': rxn.reactants, 'products': rxn.products, 'rxnType': rxn.__class__.__name__,
                     'duplicate': rxn.duplicate, 'reversible': rxn.reversible, 'orders': rxn.orders,
                     'rxnCoeffs': reset_coeffs})

            else:
                coeffs.append({})
                coeffs_bnds.append({})
                if len(bnds) == 0:
                    rate_bnds.append({})
                reset_mech.append(
                    {'reactants': rxn.reactants, 'products': rxn.products, 'rxnType': rxn.__class__.__name__})
                raise (
                    f'{rxn} is a {rxn.__class__.__name__} and is currently unsupported in Frhodo, but this error should never be seen')

    def get_coeffs_keys(self, rxn, coefAbbr, rxnIdx=None):
        """Get the keys in a reaction dictionary associated that define the par"""
        if type(rxn) in [ct.ElementaryReaction, ct.ThreeBodyReaction]:
            bnds_key = 'rate'
            coef_key = 0

        elif type(rxn) is ct.PlogReaction:
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

        elif type(rxn) is ct.FalloffReaction:
            if 'high' in coefAbbr:
                coef_key = bnds_key = 'high_rate'
            elif 'low' in coefAbbr:
                coef_key = bnds_key = 'low_rate'
            else:
                coef_key = bnds_key = 'falloff_parameters'

        else:
            raise ValueError(f'rxn type {rxn} not yet supported')

        return coef_key, bnds_key

    def set_thermo_expression_coeffs(self):  # TODO Doesn't work with NASA 9
        self.thermo_coeffs = []
        for i in range(self.gas.n_species):
            S = self.gas.species(i)
            thermo_dict = {'name': S.name, 'composition': S.composition, 'charge': S.charge,
                           'size': S.size, 'type': type(S.thermo), 'P_ref': S.thermo.reference_pressure,
                           'T_low': S.thermo.min_temp, 'T_high': S.thermo.max_temp,
                           'coeffs': np.array(S.thermo.coeffs),
                           'h_scaler': 1, 's_scaler': 1, }

            self.thermo_coeffs.append(thermo_dict)

    def modify_reactions(self, coeffs, rxnIdxs: Optional[Union[int, float, Sequence[int]]] = ()):
        """Update the reaction models in the underlying Cantera reaction models associated with this class

        Args:
            coeffs: New coefficient values
            rxnIdxs: IDs of reactions to be updated
        """
        # TODO (wardlt): This function is only called with `self.mech` as inputs
        # TODO (sikes): Only works for Arrhenius equations currently
        if not rxnIdxs:  # if rxnNums does not exist, modify all
            rxnIdxs = range(len(coeffs))
        else:
            if isinstance(rxnIdxs, (float, int)):  # if single reaction given, run that onee
                rxnIdxs = [rxnIdxs]

        for rxnIdx in rxnIdxs:
            rxn = self.gas.reaction(rxnIdx)
            rxnChanged = False
            if type(rxn) in [ct.ElementaryReaction, ct.ThreeBodyReaction]:
                for coefName in ['activation_energy', 'pre_exponential_factor', 'temperature_exponent']:
                    if coeffs[rxnIdx][0][coefName] != eval(f'rxn.rate.{coefName}'):
                        rxnChanged = True

                if rxnChanged:  # Update reaction rate
                    A = coeffs[rxnIdx][0]['pre_exponential_factor']
                    b = coeffs[rxnIdx][0]['temperature_exponent']
                    Ea = coeffs[rxnIdx][0]['activation_energy']
                    rxn.rate = ct.Arrhenius(A, b, Ea)

            elif type(rxn) is ct.FalloffReaction:
                for key in ['low_rate', 'high_rate', 'falloff_parameters']:
                    if 'rate' in key:
                        for coefName in ['activation_energy', 'pre_exponential_factor', 'temperature_exponent']:
                            if coeffs[rxnIdx][key][coefName] != eval(f'rxn.{key}.{coefName}'):
                                rxnChanged = True

                                A = coeffs[rxnIdx][key]['pre_exponential_factor']
                                b = coeffs[rxnIdx][key]['temperature_exponent']
                                Ea = coeffs[rxnIdx][key]['activation_energy']
                                setattr(rxn, key, ct.Arrhenius(A, b, Ea))
                                break
                    else:
                        length_different = len(coeffs[rxnIdx][key]) != len(rxn.falloff.parameters)
                        if length_different or (coeffs[rxnIdx][key] != rxn.falloff.parameters).any():
                            rxnChanged = True

                            if coeffs[rxnIdx]['falloff_type'] == 'Troe':
                                if coeffs[rxnIdx][key][-1] == 0.0:
                                    rxn.falloff = ct.TroeFalloff(coeffs[rxnIdx][key][:-1])
                                else:
                                    rxn.falloff = ct.TroeFalloff(coeffs[rxnIdx][key])
                            else:  # could also be SRI. For optimization this would need to be cast as Troe
                                rxn.falloff = ct.SriFalloff(coeffs[rxnIdx][key])

            elif type(rxn) is ct.ChebyshevReaction:
                pass
            else:
                continue

            if rxnChanged:
                self.gas.modify_reaction(rxnIdx, rxn)

        # Not sure if this is necessary, but it reduces strange behavior in incident shock reactor
        # time.sleep(10E-3)        # TODO: if incident shock reactor is written in C++, this can likely be removed

    def rxn2Troe(self, rxnIdx, HPL, LPL, eff={}):
        reactants = self.gas.reaction(rxnIdx).reactants
        products = self.gas.reaction(rxnIdx).products
        r = ct.FalloffReaction(reactants, products)
        print(r)
        # r.high_rate = ct.Arrhenius(7.4e10, -0.37, 0.0)
        # r.low_rate = ct.Arrhenius(2.3e12, -0.9, -1700*1000*4.184)
        # r.falloff = ct.TroeFalloff((0.7346, 94, 1756, 5182))
        # r.efficiencies = {'AR':0.7, 'H2':2.0, 'H2O':6.0}
        print(dir(self.gas))
        print(self.gas.thermo_model)
        print(self.gas.kinetics_model)

        start = timer()
        # self.gas.thermo_model
        # self.gas.kinetics_model
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
        elif type(rxnIdxs) is not list:  # if not list then assume given single rxnIdx
            rxnIdxs = [rxnIdxs]

        # if not list then assume given single coefficient
        # if Arrhenius, expects list of coefficients. If Plog or Falloff expected [['high_rate', 'activation_energy'], ...]
        if coefNames is not None and type(coefNames) is not list:
            coefNames = [coefNames]

        prior_coeffs = deepcopy(self.coeffs)
        for rxnIdx in rxnIdxs:
            if coefNames is None:  # resets all coefficients in rxn
                self.coeffs[rxnIdx] = self.reset_mech[rxnIdx]['rxnCoeffs']

            elif self.reset_mech[rxnIdx]['rxnType'] in ['ElementaryReaction', 'ThreeBodyReaction']:
                for coefName in coefNames:
                    self.coeffs[rxnIdx][coefName] = self.reset_mech[rxnIdx]['rxnCoeffs'][coefName]

            elif 'PlogReaction' == self.reset_mech[rxnIdx]['rxnType']:
                for [limit_type, coefName] in coefNames:
                    if limit_type == 'low_rate':
                        self.coeffs[rxnIdx][0][coefName] = self.reset_mech[rxnIdx]['rxnCoeffs'][0][coefName]
                    elif limit_type == 'high_rate':
                        self.coeffs[rxnIdx][-1][coefName] = self.reset_mech[rxnIdx]['rxnCoeffs'][-1][coefName]

            elif 'FalloffReaction' == self.reset_mech[rxnIdx]['rxnType']:
                self.coeffs[rxnIdx]['falloff_type'] = self.reset_mech[rxnIdx]['falloffType']
                for [limit_type, coefName] in coefNames:
                    self.coeffs[rxnIdx][limit_type][coefName] = self.reset_mech[rxnIdx]['rxnCoeffs'][limit_type][
                        coefName]

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

    def M(self, rxn, TPX=[]):  # kmol/m^3
        def get_M(rxn):
            M = self.gas.density_mole
            if hasattr(rxn, 'efficiencies') and rxn.efficiencies:
                M *= rxn.default_efficiency
                for (s, conc) in zip(self.gas.species_names, self.gas.concentrations):
                    if s in rxn.efficiencies:
                        M += conc * (rxn.efficiencies[s] - 1.0)
                    else:
                        M += conc
            return M

        if len(TPX) == 0:
            M = get_M(rxn)
        else:
            [T, P, X] = TPX
            M = []
            for i in range(0, len(T)):
                self.set_TPX(T[i], P[i], X)  # IF MIXTURE CHANGES THEN THIS NEEDS TO BE VARIABLE
                M.append(get_M(rxn))

            M = np.array(M)

        return M

    def run(self, reactor_choice, t_end, T_reac, P_reac, mix, **kwargs) -> Tuple[Simulation_Result, dict]:
        """Perform a simulation of the reaction output

        Args:
            reactor_choice: Choice of the reactor
            t_end: How long of a simulation to run
            T_reac: Temperature at which to perform the reaction
            P_reac: Pressure at which to perform the simulation
            mix: Initial concentrations of the reactants
            kwargs: Options that are passed forward to the specific type of reactor being used and ODE solver
        Returns:
            - Results of the simulation
            - Extensive details of the model performance
        """
        return self.reactor.run(reactor_choice, t_end, T_reac, P_reac, mix, **kwargs)


class Uncertainty:  # alternate name: why I hate pickle part 10
    """Computes the uncertainty bounds for parameters"""

    def __init__(self, unc_type, rxnIdx, **kwargs):
        """
        Args:
            unc_type: Type of the
            rxnIdx: Index of the reaction
            **kwargs:
        """
        # self.gas = gas
        self.unc_type = unc_type
        self.rxnIdx = rxnIdx
        self.unc_dict = kwargs

    def _unc_fcn(self, x, uncVal, uncType):  # TODO (wardlt): Can be static or
        """Compute the bounds given the value of a coefficient

        Args:
            x: Assembled value of the coefficient
            uncVal: Magnitude of the uncertainty
            uncType: Type of the uncertainty (e.g., %, +, F)
        Returns:
            Bounds for the value
        """
        if np.isnan(uncVal):
            return [np.nan, np.nan]
        elif uncType == 'F':
            return np.sort([x / uncVal, x * uncVal], axis=0)
        elif uncType == '%':
            return np.sort([x / (1 + uncVal), x * (1 + uncVal)], axis=0)
        elif uncType == '±':
            return np.sort([x - uncVal, x + uncVal], axis=0)
        elif uncType == '+':
            return np.sort([x, x + uncVal], axis=0)
        elif uncType == '-':
            return np.sort([x - uncVal, x], axis=0)
        else:
            raise ValueError(f'Unsupported type {uncType}')

    def __call__(self, x=None):
        if self.unc_type == 'rate':
            # if x is None:    # defaults to giving current rate bounds
            #    x = self.gas.forward_rate_constants[self.rxnIdx]
            rate_bnds = self.unc_dict['rate_bnds']
            unc_value = rate_bnds[self.rxnIdx]['value']
            unc_type = rate_bnds[self.rxnIdx]['type']
            return self._unc_fcn(x, unc_value, unc_type)
        else:
            coeffs_bnds = self.unc_dict['coeffs_bnds']
            key = self.unc_dict['key']
            coefName = self.unc_dict['coef_name']

            if key == 'falloff_parameters':  # falloff parameters have no limits
                return [np.nan, np.nan]

            coef_dict = coeffs_bnds[self.rxnIdx][key][coefName]
            coef_val = coef_dict['resetVal']
            unc_value = coef_dict['value']
            unc_type = coef_dict['type']
            return self._unc_fcn(coef_val, unc_value, unc_type)
