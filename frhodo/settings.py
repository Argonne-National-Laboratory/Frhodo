# This file is part of Frhodo. Copyright © 2020, UChicago Argonne, LLC
# and licensed under BSD-3-Clause. See License.txt in the top-level 
# directory for license and copyright information.

import numpy as np
import os, sys, platform, pathlib, shutil, configparser, re, csv
from copy import deepcopy
from dateutil.parser import parse
from scipy import integrate      # used to integrate weights numerically
from scipy.interpolate import CubicSpline
from qtpy import QtCore

from .calculate.convert_units import OoM, RoundToSigFigs
from .calculate.smooth_data import dual_tree_complex_wavelet_filter

min_pos_system_value = (np.finfo(float).tiny*(1E20))**(1/2)
max_pos_system_value = (np.finfo(float).max*(1E-20))**(1/2)


class Path:
    """Facilitates loading data from disk"""

    def __init__(self, parent, path):
        self.parent = parent
        self.loading_dir_file = False
        
        parent.path = path
        parent.path['graphics'] = parent.path['main'] / 'UI' / 'graphics'
        self.config = configparser.RawConfigParser()
        
        # Specify yaml files
        parent.path['default_config'] = parent.path['appdata'] / 'default_config.yaml'
        parent.path['Cantera_Mech'] = parent.path['appdata'] / 'generated_mech.yaml'
        for key in ['default_config', 'Cantera_Mech']:
            if parent.path[key].exists(): # Check that file is readable and writable
                if not os.access(parent.path[key], os.R_OK) or not os.access(parent.path[key], os.W_OK):
                    os.chmod(parent.path[key], stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP) # try to change if not

        # Create file watcher 
        self.fs_watcher = QtCore.QFileSystemWatcher()
        self.fs_watcher.directoryChanged.connect(self.mech)

    def mech(self):
        """Load in the chemical mechanism files from the "mech_main" directory"""
        parent = self.parent
        
        # Test for existence of mech folder and return if it doesn't
        #  TODO (wardlt): Remove after confirming this value is never used
        if parent.path['mech_main'].is_dir():
            self.mech_main_exists = True
        else:
            self.mech_main_exists = False
        
        unsorted_mech_files = []
        thermo_files = []
        trans_files = []
        for file in parent.path['mech_main'].glob('*'):
            if not file.is_file():
                continue
                
            name = file.name
            ext = file.suffix.lower()
            
            if ext == '.therm':
                thermo_files.append(name)
            if ext == '.tran':      # currently unused, but store transport files anyways
                trans_files.append(name)
            elif ext in ['.yaml', '.yml', '.cti','.ck', '.mech', '.inp']: #  '.ctml', '.xml', # TODO: enable ctml and xml format
                if 'generated_mech.yaml' == name: continue
                elif 'generated_mech.yml' == name: continue
                unsorted_mech_files.append(name)
        
        # Sort Mechs
        mech = {'base': [], 'suffix': [], 'num': [], 'ext': []}
        suffix = ' - Opt '
        for file in unsorted_mech_files:
            match_opt = re.findall('(.*){:s}(\d+).(.*)'.format(suffix), file)
            if match_opt is not None and len(match_opt) > 0:    # if match, then optimized mech with number
                mech['base'].append(match_opt[0][0])
                mech['suffix'].append(suffix)
                mech['num'].append(int(match_opt[0][1]))
                mech['ext'].append(match_opt[0][2])
            else:                                               # if no - Opt then it's a new mechanism
                match = re.findall('(.*)\.(.*)', file)
                mech['base'].append(match[0][0])
                mech['suffix'].append('')
                mech['num'].append(np.inf)          # Makes unedited mech first
                mech['ext'].append(match[0][1])

        # Sort by descending opt num and default sorting of mech name
        sort_idx = np.lexsort((-np.array(mech['num']), mech['base'])) 
        mech_files = []
        for n in sort_idx:
            if mech['suffix'][n] != '':
                num = str(mech['num'][n])
            else:
                num = ''
            name = mech['base'][n] + mech['suffix'][n] + num + '.' + mech['ext'][n]
            mech_files.append(name)
        
        # Add items to combobox
        for obj in [parent.mech_select_comboBox, parent.thermo_select_comboBox]:
            obj.blockSignals(True)
            oldText = obj.currentText()
            obj.clear()
            if obj is parent.mech_select_comboBox:
                obj.addItems(mech_files)
            else:
                obj.addItems(thermo_files)
            
            idx = obj.findText(oldText) # if the previous selection exists, reselect it
            obj.blockSignals(False)
            if idx >= 0:
                obj.setCurrentIndex(idx)
     
    def shock_paths(self, prefix, ext, max_depth=2):
        parent = self.parent
        path = parent.path
        path['shock'] = parent.path['exp_main']
        
        shock_num = np.array([]).astype(int)
        shock_path = []
        for file in parent.path['exp_main'].rglob('*'):
            if not file.is_file():  # only looking for files
                continue
            
            depth = len(file.relative_to(parent.path['exp_main']).parts)
            if depth > max_depth:   # skip if the depth is greater than allowed
                continue
            
            match = re.search(prefix + '.*\.' + ext + '$', file.name)   # search for prefix + ext
            if match:
                numMatch = re.search('(\d+)', file.name)   # search for numbers
                n = int(numMatch.group(0))
                
                # skip appending duplicates with larger path depth
                append_shock = True
                shock_match = np.argwhere(shock_num == n)
                if np.shape(shock_match)[0] > 0:
                    i = shock_match[0,0]
                    depth_old = len(shock_path[i].relative_to(parent.path['exp_main']).parts)
                    if depth > depth_old:
                        append_shock = False
                        
                if append_shock:
                    shock_num = np.append(shock_num, n)
                    shock_path.append(file.parents[0])  # appends root directory of shock
                
        if len(shock_num) == 0: # in case no shocks in directory
            return []
        
        # Sort based on shock number
        idx_sort = np.argsort(shock_num)
        shock_num = shock_num[idx_sort]
        shock_path = np.array(shock_path)[idx_sort]
        
        # Create sorted list of shock_num and shock_path
        return np.column_stack((shock_num, shock_path))
        
    def shock(self, shock_num):
        def find_nearest(array, value): # Finds the nearest value
            array = np.asarray(array)
            if value in array:
                idx = np.where(value == array)[0][0]
            elif np.max(array) < value:
                idx = np.argmax(array)
            elif np.min(array) > value:
                idx = np.argmin(array)
            else:
                idx = np.argmin(np.abs(array - value))
                
            return idx, array[idx]
            
        parent = self.parent
        
        change = parent.var['shock_choice'] - parent.var['old_shock_choice']
        if np.abs(change) == 1: # if stepping through experiments
            idx = np.where(parent.var['old_shock_choice'] == shock_num)[0] + change
            
            if np.shape(idx)[0] == 0:   # if shock cannot be found, find nearest
                idx, Shock_Choice = find_nearest(shock_num, parent.var['shock_choice'])
            else:                       # if shock can be found, step
                idx = idx[0]
                if idx >= len(shock_num):
                    idx = 0
                elif idx < 0:
                    idx = len(shock_num) - 1
                Shock_Choice = shock_num[idx]
        else:                   # if selecting experiment or loading a new exp directory
            idx, Shock_Choice = find_nearest(shock_num, parent.var['shock_choice'])
        
        # Update shock_choice_box and var['shock_choice']
        parent.var['shock_choice'] = Shock_Choice
        for box in parent.shock_choice_box.twin:
            box.blockSignals(True)
            box.setValue(Shock_Choice)
            box.blockSignals(False)
        
        return idx
    
    def shock_output(self):
        parent = self.parent
        log = parent.log
        
        # Add Exp_set_name if exists
        shock_num = str(parent.var['shock_choice'])
        if not parent.display_shock['series_name']:
            parent.path['output_dir'] = parent.path['sim_main'] / ('Shock ' + str(shock_num))
        else:
            parent.path['output_dir'] = ((parent.path['sim_main'] / parent.display_shock['series_name'])
                                         / ('Shock ' + str(shock_num)))
        
        # Create folders if needed
        if not parent.path['output_dir'].exists():
            try:
                parent.path['output_dir'].mkdir(exist_ok=True, parents=True)
            except (IOError, FileNotFoundError) as e:
                log.append('Error in saving:')
                log.append(e)  
                return
        
        parent.path['Sim log'] = parent.path['output_dir'] / 'Sim log.txt'
        
        # Find next sim number based on Sim log
        self.sim_num = 0
        if parent.path['Sim log'].exists():
            with open(parent.path['Sim log'], 'r') as f:
                for line in f:
                    if len(re.findall('Sim \d+:', line)) > 0:
                        self.sim_num += 1
        
        self.sim_num += 1
        
        # Assign sim directory
        if self.sim_num == 1:
            parent.path['sim_dir'] = parent.path['output_dir']
        elif self.sim_num > 1:
            parent.path['sim_dir'] = parent.path['output_dir'] / 'Sim {:d}'.format(self.sim_num)
            parent.path['sim_dir'].mkdir(exist_ok=True, parents=True)
            
            # Move files if second sim
            if self.sim_num == 2:
                sim_1_dir = parent.path['output_dir'] / 'Sim 1'
                sim_1_dir.mkdir(exist_ok=True, parents=True)
                for entry in parent.path['output_dir'].glob('*'):
                    if entry.is_file():
                        if len(re.findall('Sim \d+ - ', entry.name)) > 0:    # if files starts with Sim ####
                            shutil.move(entry, sim_1_dir / entry.name)
        
        for file in ['Mech.yaml', 'Mech.ck', 'Plot.png', 'Legend.txt']:
            parent.path[file] = parent.path['sim_dir'] / 'Sim {:d} - {:s}'.format(self.sim_num, file)
    
    def sim_output(self, var_name):  # takes variable name and creates path for it
        if var_name == '\u00B1 % |Density Gradient|':   # lots of invalid characters, replace
            var_name = 'signed % Abs Density Gradient'

        name = 'Sim {:d} - {:s}.txt'.format(self.sim_num, var_name)
        self.parent.path[var_name] = self.parent.path['sim_dir'] / name
        
        return self.parent.path[var_name]
    
    def optimized_mech(self, file_out='opt_mech'):
        """Define the filename of the optimized mechanism file"""
        parent = self.parent
        
        mech_name = parent.path['mech'].stem
        mech_name = re.sub(r' - Opt \d+$', '', str(mech_name))   # strip opt and trailing number
        mech_name += ' - Opt '                             # add opt back in
        
        num = [0]  
        for file in parent.path['mech_main'].glob('*'):
            if not file.is_file():
                continue
                
            num_found = re.findall(r'{:s}\s*(-?\d+(?:\.\d+)?)'.format(mech_name), file.name)
            if len(num_found) > 0:
                num.append(*[int(num) for num in num_found])
        
        opt_mech_file = '{:s}{:.0f}.mech'.format(mech_name, np.max(num)+1)
        recast_mech_file = opt_mech_file.replace('Opt', 'PreOpt')
        parent.path['Optimized_Mech.mech'] = parent.path['mech_main'] / opt_mech_file
        parent.path['Optimized_Mech_recast.mech'] = parent.path['mech_main'] / recast_mech_file
        
        if file_out == 'opt_mech':
            return parent.path['Optimized_Mech.mech']
        elif file_out == 'recast_mech':
            return parent.path['Optimized_Mech_recast.mech']
    
    def load_dir_file(self, file_path):
        """Load the directories specified by the user"""
        parent = self.parent
        self.loading_dir_file = True
        self.config.read(file_path)
        
        # loading exp_main creates a new series
        parent.exp_main_box.setPlainText(self.config['Directories']['exp_main'])
        
        if ('exp_main' not in parent.directory.invalid and 
            ': ' in self.config['Species Default Aliases']['aliases']):
            
            for pair in self.config['Species Default Aliases']['aliases'].split('; '):
                exp_name, thermo_name = pair.split(': ')
                parent.series.current['species_alias'][exp_name] = thermo_name
        
        parent.mech_main_box.setPlainText(self.config['Directories']['mech_main'])
        parent.sim_main_box.setPlainText(self.config['Directories']['sim_main'])
        if len(self.config['Experiment Set Name']['name']) > 0:
            parent.exp_series_name_box.setText(self.config['Experiment Set Name']['name'])
        
        # parent.option_tab_widget.setCurrentIndex(1)
        # if len(parent.series.current['species_alias']) > 0: # if species_alias exists
            # parent.mix.update_species()                     # update species

        self.mech()                         # This updates the mech and thermo combo boxes
        self.loading_dir_file = False
    
    def save_dir_file(self, file_path):
        self.config['Experiment Set Name'] = {'name':     self.parent.display_shock['series_name']}
                  
        self.config['Species Default Aliases'] = {'aliases': self._alias_str()}
            
        self.config['Directories'] = {'exp_main':        self.parent.path['exp_main'],
                                      'mech_main':       self.parent.path['mech_main'],
                                      'sim_main':        self.parent.path['sim_main']}
                                          
        with open(file_path, 'w') as configfile:
            self.config.write(configfile)
  
    def save_aliases(self, file_path):
        self.config.set('Species Default Aliases', 'aliases', self._alias_str())
        
        with open(file_path, 'w') as configfile:
            self.config.write(configfile)
            
    def _alias_str(self):
        # The path file isn't the ideal place based on name, but works functionally 
        species_aliases_str = []
        for alias, species in self.parent.series.current['species_alias'].items():
            species_aliases_str.append(alias + ': ' + species)
            
        return '; '.join(species_aliases_str)

    def set_watch_dir(self):
        if self.fs_watcher.directories():
            self.fs_watcher.removePaths(self.fs_watcher.directories())

        if self.parent.path['mech_main'].is_dir():
            self.fs_watcher.addPath(str(self.parent.path['mech_main']))


class experiment:
    def __init__(self, parent):
        self.parent = parent
        self.path = parent.path
        self.convert_units = parent.convert_units
        self.load_style = 'tranter_v1_0'
        # self.load_full_series_box = parent.load_full_series_box
        # self.set_load_full_set()
        # self.load_full_series_box.stateChanged.connect(lambda: self.set_load_full_set())

           
    def parameters(self, file_path):       
        with open(file_path) as f:
            lines = f.read().splitlines()

        if lines[0] == '[Date]':   # new Tranter style
            self.load_style = 'tranter_v1_0'
            parameters = self.read_tranter_exp_v1(lines)
        elif lines[0] == '"[Expt Parameters]"':   # old Tranter style
            self.load_style = 'tranter_v0_1'
            parameters = self.read_tranter_exp_v0(lines)
        else:
            self.load_style = 'tranter_v1_0'
            parameters = self.read_tranter_exp_v1(lines)

        # Units are assumed to be: T1 [°C], P1 [torr], u1 [mm/μs], P4 [psi]
        parameters['T1'] = self.convert_units(parameters['T1'], '°C', '2ct')
        parameters['P1'] = self.convert_units(parameters['P1'], 'torr', '2ct')
        parameters['u1'] = self.convert_units(parameters['u1'], 'mm/μs', '2ct')
        parameters['P4'] = self.convert_units(parameters['P4'], 'psi', '2ct')
        parameters['Sample_Rate'] *= 1E-6               # Hz      to MHz ?

        return parameters
      
    def read_tranter_exp_v1(self, lines):
        def get_config(section, key):
            val = self.config[section][key]
            for delimiter in ['"', "'"]:    # remove delimiters
                val = val.strip(delimiter)
            
            return val

        self.config = configparser.RawConfigParser()
        self.config.read_string('\n'.join(lines))

        # Get mixture composition
        mix = {}
        for key in [item[0] for item in self.config.items('Mixture')]:    # search all keys in section
            match_opt = re.findall('mol_(\d+)_formula', key) 
            if match_opt is not None and len(match_opt) > 0:    # if found, append species
                species_num = match_opt[0]
                
                # check that mol_frac exists
                if not self.config.has_option('Mixture', 'Mol_' + species_num + '_Mol frc'): 
                    continue
                
                species = get_config('Mixture', 'Mol_' + species_num + '_Formula')
                mol_frac = float(get_config('Mixture', 'Mol_' + species_num + '_Mol frc'))
                if mol_frac != 0:  # continue if mole fraction is not zero
                    mix[species] = mol_frac
        
        # Throw errors if P1, T1, or Sample Rate are zero
        if float(get_config('Expt Params', 'P1')) == 0:
            raise Exception('Exception in Experiment File: P1 is zero')
        elif float(get_config('Expt Params', 'T1')) == 0:
            raise Exception('Exception in Experiment File: T1 is zero')
        elif float(get_config('Expt Params', 'SampRate')) == 0:
            raise Exception('Exception in Experiment File: Sample Rate is zero')
             
        # Throw errors for u1 and calculate/get it from VelatObs or tOpt/PT Spacing
        if self.config.has_option('Expt Params', 'VelatObs'): # Tranter exp file v1.1 gives velocity
            u1 = float(get_config('Expt Params', 'VelatObs'))
            if u1 == 0:
                raise Exception('Exception in Experiment File: Velocity at observation is zero')
        else:
            tOpt = float(get_config('Expt Params', 'tOpt'))
            PT_spacing = float(get_config('Expt Params', 'PT Spacing'))
            u1 = PT_spacing/tOpt
            if tOpt == 0:
                raise Exception('Exception in Experiment File: tOpt is zero')
            elif PT_spacing == 0:
                raise Exception('Exception in Experiment File: PT Spacing is zero')

        parameters = {'T1': float(get_config('Expt Params', 'T1')),
                      'P1': float(get_config('Expt Params', 'P1')),
                      'u1': u1,
                      'P4': float(get_config('Expt Params', 'P4')),
                      'exp_mix': deepcopy(mix), 'thermo_mix': deepcopy(mix),
                      'Sample_Rate': float(get_config('Expt Params', 'SampRate'))}

        return parameters

    def read_tranter_exp_v0(self, lines):
        parameters = {'T1': None, 'P1': None, 'u1': None, 'exp_mix': {}, 'thermo_mix': {}}

        key = None
        processed = {'exp_mix': [], 'shock_conditions': []}
        for line in lines:
            if line == '"[Thermochemistry]"':
                key = 'exp_mix'
                continue
            elif line == '"[Start Conditions]"' or line == '"[Expt Times]"':
                key = 'shock_conditions'
                continue
            elif line.isspace() or len(line) == 0:
                key = None
                continue

            if key is not None:
                processed[key].append(line)

        parameters['P1'] = float(processed['shock_conditions'][1])
        parameters['T1'] = float(processed['shock_conditions'][2])
        parameters['u1'] = 120.0/float(processed['shock_conditions'][-1]) # assumes 120 mm spacing
        parameters['P4'] = 1.0 # Arbitrarily assign. Does not matter for general functionality

        for line in processed['exp_mix']:
            species, mol_frac = line[1:].split(';')[:2]
            parameters['exp_mix'][species] = float(mol_frac)

        parameters['thermo_mix'] = parameters['exp_mix']
        parameters['Sample_Rate'] = 50000000.0

        return parameters

    def csv(self, file):
        def is_numeric(strings):
            for str in strings:        # test for all strings
                try:
                    float(str)         # attempt to turn each item into float
                except ValueError:
                    return False
                    
            return True                 # if all items are floats, strings are numeric
            
        if file is None:
            return None
            
        data = []
        data_nonnumeric = []
        with open(file, newline='') as f:
            reader = csv.reader(f)
            try:
                for n, row in enumerate(reader):
                    if is_numeric(row):      # THIS SKIPS ALL NON NUMERIC LINES
                        data.append(row)
                    else:
                        data_nonnumeric.append([n, row[0]])
            except csv.Error as e:
                sys.exit('file {}, line {}: {}'.format(file.name, reader.line_num, e))
        
        data = np.array(data, float)
        
        if np.logical_not(np.isfinite(data)).any():
            raise Exception('Exception in {:s}: Nonfinite values found in data'.format(file.name))
        elif np.shape(data)[0] == 0:
            raise Exception('Exception in {:s}: No data found'.format(file.name))

        return data, data_nonnumeric
    
    def exp_data(self, file_path):
        exp_data, nonnumeric = self.csv(file_path)

        if self.load_style == 'tranter_v0_1':
            exp_data = np.array(exp_data)
            exp_data = exp_data[:,[0,2]]
            exp_data = exp_data[:-1,:]
        
        return exp_data
    
    def raw_signal(self, file_path):
        def is_date(string, fuzzy=False):
            try: 
                parse(string, fuzzy=fuzzy)
                return True

            except ValueError:
                return False
            
        raw_sig, nonnumeric = self.csv(file_path)
        if nonnumeric:  # if anything is nonnumeric
            # if first line is a date, assume format is Tranter's new format
            if is_date(nonnumeric[0][1].replace('-', ' '), fuzzy=True): 
                raw_sig = raw_sig[2:]
        
        return raw_sig
        
    def load_data(self, shock_num, main_path):       
        def try_load(fcn, path):
            try:
                return fcn(path)
            except Exception as e:
                # self.option_tabs.setCurrentWidget(self.option_tabs.findChild(QWidget, 'log_tab'))
                log.append('Error in loading Shock {:d}:'.format(shock_num))
                log.append(e)  
        
        log = self.parent.log
        shock_num = int(shock_num)
        # Search for exp, rho, and raw signal files. They default to None
        paths = {'Shock.exp': None, 'Shock.rho': None, 'ShockRaw1.sig': None}
        for item in main_path.glob('*'):
            if item.is_file():
                if 'Shock{:d}.exp'.format(shock_num) in item.name:
                    paths['Shock.exp'] = item
                elif 'Shock{:d}.rho'.format(shock_num) in item.name:
                    paths['Shock.rho'] = item
                elif 'Shock{:d}raw1.sig'.format(shock_num) in item.name:
                    paths['ShockRaw1.sig'] = item
                elif all(x in item.name for x in ['LS', 'L-R']):
                    paths['ShockRaw1.sig'] = item
        
        # Produce error messages for missing files
        data = {}
        load_fcn = {'Shock.exp': self.parameters, 'Shock.rho': self.exp_data, 
                     'ShockRaw1.sig': self.raw_signal}
        for file, path in paths.items():
            if path is None:
                log.append('Error in loading Shock {:d}:'.format(shock_num))
                log.append(file + ' is missing')
                data[file] = None
            else:
                data[file] = try_load(load_fcn[file], path)
                if file == 'Shock.rho' and data[file] is not None:
                    data[file][:,0] *= 1E-6
            
            if data[file] is None:
                if file == 'Shock.exp':
                    data[file] = {}
                else:
                    data[file] = np.array([])
        
        return data.values()


def double_sigmoid(x, A, k, x0):    # A = extrema, k = inverse growth rate, x0 = shifts
    def sig(x):     # Numerically stable sigmoid function
        # return np.where(x >= 0, 1/(1 + np.exp(-x)),   # should work, doesn't
                                # np.exp(x)/(1 + np.exp(x)))
        eval = np.empty_like(x)
        pos_val_f = np.exp(-x[x >= 0])
        eval[x >= 0] = 1/(1 + pos_val_f)
        neg_val_f = np.exp(x[x < 0])
        eval[x < 0] = neg_val_f/(1 + neg_val_f)

        # clip so they can be multiplied
        eval[eval > 0] = np.clip(eval[eval > 0], min_pos_system_value, max_pos_system_value)
        eval[eval < 0] = np.clip(eval[eval < 0], -max_pos_system_value, -min_pos_system_value)

        return eval
        
    def b_eval(x, k, x0):
        if k == 0:             # assign values if k = 0 aka infinite growth rate
            b = np.ones_like(x)*np.inf
            if isinstance(x,(list,np.ndarray)):
                b[x < x0] *= -1
            elif x <= x0:
                b *= -1             
        else:
            b = 1.5/k*(x - x0)
            
        return b

    b = [[],[]]
    for i in range(0,2):
        b[i] = b_eval(x, k[i], x0[i])
        
    if not np.isfinite(b).any():                                # if all infinity, don't use mean
        a = (A[2] - A[0])*sig(b[0]) + A[0]          # a is the changing minimum
    else:
        a = (A[2] - A[0])*sig(np.mean(b,0)) + A[0]  # a is the changing minimum
    
    #print((A[1] - a), sig(b[0]), sig(-b[1]), a)

    res = (A[1] - a)*sig(b[0])*sig(-b[1]) + a

    return res

class series:
    """Holds data from a collection of related shock experiments.

    Also provides functionality for switching between different shocks in displays"""
    def __init__(self, parent):
        self.parent = parent
        self.exp = experiment(parent)
        
        self.idx = 0        # series index number
        self.shock_idx = 0  # shock index number
        
        self.path = []
        self.name = []
        self.shock_num = []
        self.shock = []
        self.species_alias = []
        self.in_table = [False]
        
        self.initialize_shock()
        self.parent.display_shock = self.shock[self.idx][self.shock_idx]
    
    def initialize_shock(self):
        parent = self.parent
        
        self.path.append([])
        self.name.append([])
        self.shock_num.append([])
        self.species_alias.append({})
        self.in_table.append(False)
        
        self.update_current()
        
        shock = []
        shock.append(self._create_shock(1, []))
        
        # Shock Parameters: In case no experimental series is loaded
        for var in ['T1', 'P1', 'u1']:
            units = eval('str(parent.' + var + '_units_box.currentText())')
            value = float(eval('str(parent.' + var + '_value_box.value())'))
            shock[-1][var] = parent.convert_units(value, units, unit_dir = 'in')
 
        self.shock.append(shock)
    
    def _create_shock(self, num, shock_path):
        """Create a placeholder for data from a shock experiment"""
        parent = self.parent
        # TODO (wardlt): This complex dictionary is ripe for implementation as a dataclass
        shock = {'num': num, 'path': deepcopy(shock_path), 'include': False, 
                'series_name': self.name[-1], 'run_SIM': True,      # TODO: make run_SIM a trigger to calculate or not
                    
                # Shock Parameters
                'T1': np.nan, 'P1': np.nan, 'u1': np.nan,
                'rho1': np.nan, 'P4': np.nan,
                'exp_mix': {}, 'thermo_mix': {}, 'zone': 2,
                'T2': np.nan, 'P2': np.nan, 'u2': np.nan,
                'T5': np.nan, 'P5': np.nan,
                'T_reactor': np.nan, 'P_reactor': np.nan,
                'time_offset': parent.time_offset_box.value(),
                'opt_time_offset': parent.time_offset_box.value(),
                'Sample_Rate': np.nan,
                
                # Weight parameters
                'weight_max': [np.nan],
                'weight_k': [np.nan, np.nan],
                'weight_shift': [np.nan, np.nan],
                'weight_min': [np.nan, np.nan],

                # Uncertainty parameters
                'unc_max': [np.nan],
                'unc_k': [np.nan, np.nan],
                'unc_shift': [np.nan, np.nan],
                'unc_min': [np.nan, np.nan],
                'unc_cutoff': [np.nan, np.nan],
                
                # Mechanism parameters
                'species_alias': self.species_alias[-1],
                'rate_val': [],
                'rate_reset_val': [],
                'rate_bnds_type': [],
                'rate_bnds_val': [],
                'rate_bnds': [],
                
                # Data
                'observable': {'main': '', 'sub': None},
                'wavelet_lvls': np.nan,                    # number of wavelets for smoothing
                'raw_data': np.array([]),
                'exp_data': np.array([]),
                'exp_data_smoothed': np.array([]),
                'weights': np.array([]),
                'normalized_weights': np.array([]),
                'uncertainties': np.array([]),
                'SIM': np.array([]),
                
                # Load error
                'err': []}
        
        return shock
    
    def add_series(self):              # need to think about what to do when mech changes, anything?
        parent = self.parent           # how do I know when I'm looking at a new compound? base on mech change?
        
        if parent.path['exp_main'] in self.path:    # check if series already exists before adding
            self.change_shock()
            return
        
        parent.path['shock'] = parent.path_set.shock_paths(prefix='Shock', ext='exp')
        if len(parent.path['shock']) == 0:  # if there are no shocks in listed directory
            parent.directory.update_icons(invalid = 'exp_main')
            return
        
        if self.in_table and not self.in_table[-1]: # if list exists and last item not in table, clear it
            self.clear_series(-1)
        
        self.path.append(deepcopy(parent.path['exp_main']))
        self.name.append(parent.exp_series_name_box.text())
        self.shock_num.append(list(parent.path['shock'][:, 0].astype(int)))
        self.species_alias.append({})
        self.in_table.append(False)
        
        shock = []
        for (shock_num, shock_path) in parent.path['shock']:
            shock.append(self._create_shock(shock_num, shock_path))
        
        self.shock.append(shock)
        self.change_shock()
    
    def change_series(self):
        self.change_shock()
    
    def added_to_table(self, n):    # update if in table
        self.in_table[n] = True
    
    def clear_series(self, n):
        """Delete a series from the collection

        Args:
            n: Index of series to be deleted
        """
        del self.path[n], self.name[n], self.shock_num[n], self.species_alias[n]
        del self.shock[n], self.in_table[n]
    
    def clear_shocks(self):
        """Remove all shocks from a certain series"""
        if self.parent.load_full_series: return
        
        self.update_idx()   # in case series are changed and load full set not selected
        for shock in self.shock[self.idx]:
            shock['exp_data'] = np.array([])
            shock['weights'] = np.array([])
            shock['raw_data'] = np.array([])
            shock['SIM'] = np.array([])
    
    def update_current(self):
        # update series.current
        self.current = {'path': self.path[self.idx], 'name': self.name[self.idx], 
                        'shock_num': self.shock_num[self.idx], 'species_alias': self.species_alias[self.idx]}
    
    def update_idx(self):
        parent = self.parent
        
        self.idx = self.path.index(parent.path['exp_main'])
        self.shock_idx = parent.path_set.shock(self.shock_num[self.idx])   # correct shock num to valid
        self.update_current()
        

    def weights(self, time, shock=[], calcIntegral=True):
        if not shock:
            shock = self.shock[self.idx][self.shock_idx]    # sets parameters based on selected shock
            
        if len(shock['exp_data']) == 0: return np.array([])
        
        parameters = [shock[key] for key in ['weight_max', 'weight_min', 'weight_shift', 'weight_k']]
        if np.isnan(np.hstack(parameters)).any():  # if weight parameters aren't set, default to gui
            self.parent.weight.update()
        
        t_conv = self.parent.var['reactor']['t_unit_conv']
        t0 = shock['exp_data'][ 0, 0]
        tf = shock['exp_data'][-1, 0]

        shift     = np.array(shock['weight_shift'])/100*(tf-t0) + t0
        k         = np.array(shock['weight_k'])*t_conv
        w_min     = np.array(shock['weight_min'])/100
        w_max     = shock['weight_max'][0]/100
        A = np.insert(w_min, 1, w_max)

        weights = double_sigmoid(time, A, k, shift)

        if calcIntegral:    # using trapazoidal method for efficiency, no simple analytical integral
            integral = integrate.cumtrapz(weights, time)[-1] # based on weights at data points

            if integral == 0.0:
                shock['normalized_weights'] = np.zeros_like(weights)
            else:
                # normalize by the integral and then by the t_unit_conv
                # TODO: normalizing by the t_unit_conv could cause a problem if the scale changes?
                weights_norm = weights.copy()/(integral/t_conv)
                shock['normalized_weights'] = weights_norm
        
        return weights
    
    def _integrate_weights(self, shock):    # Defunct, but it works!
        weights = lambda time: self.weights(time, shock=shock, calcIntegral=False)
        t0 = shock['exp_data'][ 0, 0]
        tf = shock['exp_data'][-1, 0]
        integral, err = integrate.quad(weights, t0, tf)   # integrate from t0 to tf using gaussian quad
        # integral = integrate.romberg(weights, t0, tf, divmax=15)
        
        return integral
    
    def uncertainties(self, time, shock=[], calcWeights=False):
        if not shock:
            shock = self.shock[self.idx][self.shock_idx]    # sets parameters based on selected shock
            
        if len(shock['exp_data']) == 0: return np.array([])
        
        parameters = [shock[key] for key in ['unc_max', 'unc_min', 'unc_shift', 'unc_k']]
        if np.isnan(np.hstack(parameters)).any():  # if weight parameters aren't set, default to gui
            self.parent.exp_unc.update()
        
        t_conv = self.parent.var['reactor']['t_unit_conv']
        t0 = shock['exp_data'][ 0, 0]
        tf = shock['exp_data'][-1, 0]

        shift     = np.array(shock['unc_shift'])/100*(tf-t0) + t0
        k         = np.array(shock['unc_k'])*t_conv
        unc_min     = np.array(shock['unc_min'])
        unc_max     = np.array(shock['unc_max'])
        A = np.insert(unc_max, 1, unc_min)
        unc_cutoff  = np.array(shock['unc_cutoff'])/100*(tf-t0) + t0

        if self.parent.exp_unc.unc_type == '%':
            A /= 100

        unc = double_sigmoid(time, A, k, shift)
        
        if calcWeights:
            weights = shock['weights'] = double_sigmoid(time, [0, 1, 0], [0, 0], unc_cutoff)
            integral = integrate.cumtrapz(weights, time)[-1] # based on weights at data points

            if integral == 0.0:
                shock['normalized_weights'] = np.zeros_like(weights)
            else:
                # normalize by the integral and then by the t_unit_conv
                # TODO: normalizing by the t_unit_conv could cause a problem if the scale changes?
                weights_norm = weights.copy()/(integral/t_conv)
                shock['normalized_weights'] = weights_norm

            # also calculate absolute uncertainties
            obs_data = shock['exp_data'][:,1]
            
            if self.parent.exp_unc.unc_type == '%':
                shock['abs_uncertainties'] = np.sort([obs_data/(1+unc), obs_data*(1+unc)], axis=0).T
            else:
                shock['abs_uncertainties'] = np.sort([obs_data - unc, obs_data + unc], axis=0).T

        return unc

    def smoothed_data(self, signal):
        def calculate_C(signal):
            finite_signal = np.array(signal)[np.isfinite(signal)] # ignore nan and inf
            min_signal = finite_signal.min()  
            max_signal = finite_signal.max()
                
            # if zero is within total range, find largest pos or neg range
            if np.sign(max_signal) != np.sign(min_signal):  
                processed_signal = [finite_signal[finite_signal>=0], finite_signal[finite_signal<=0]]
                C = 0
                for signal in processed_signal:
                    range = np.abs(signal.max() - signal.min())
                    if range > C:
                        C = range
                        max_signal = signal.max()
            else:
                C = np.abs(max_signal-min_signal)

            C *= 10**(OoM(max_signal) + 2)  # scaling factor TODO: + 1 looks loglike, + 2 linear like
            C = RoundToSigFigs(C, 1)    # round to 1 significant figure

            return C

        t = signal[:,0]
        obs = signal[:,1]

        lvls = self.shock[self.idx][self.shock_idx]['wavelet_lvls']

        C = calculate_C(obs)

        # if data is not uniformly sampled, resample it
        dt = np.diff(t)
        min_dt = np.min(dt)
        max_dt = np.max(dt)
        if not np.isclose(min_dt, max_dt):
            f_interp = CubicSpline(t.flatten(), obs.flatten())

            N = np.ceil((np.max(t) - np.min(t))/min_dt)
            t = np.linspace(np.min(t), np.max(t), int(N))
            obs = f_interp(t).flatten()

        obs = np.sign(obs)*np.log10(1 + np.abs(obs/C))      # apply bisymlog prior to smoothing
        obs = dual_tree_complex_wavelet_filter(obs, lvls=lvls) # smooth transformed data
        obs = np.sign(obs)*C*(np.power(10, np.abs(obs)) - 1)  # inverse transform back to original scale

        signal = np.array([t, obs])

        return signal.T

    def set(self, key, val=[], **kwargs):
        parent = self.parent
        if key == 'exp_data':
            if parent.load_full_series:
                shocks = self.shock[self.idx]
            else:
                self.clear_shocks()
                shocks = [self.shock[self.idx][self.shock_idx]]
            
            for shock in shocks:
                parameters, exp_data, raw_signal = self.exp.load_data(shock['num'], shock['path'])
                shock.update(parameters)
                shock['exp_data'] = exp_data
                shock['raw_data'] = raw_signal
                
                for key in ['exp_data', 'raw_data']:
                    if shock[key].size == 0:
                        shock['err'].append(key)
        
        elif key == 'exp_data_smoothed':
            opt_type = parent.optimization_settings.get('obj_fcn', 'type')
            shock = self.shock[self.idx][self.shock_idx]
            lvls = self.parent.plot.signal.wavelet_levels

            if (opt_type != 'Bayesian' or parent.plot.signal.unc_shading != 'Smoothed Signal' 
                or len(shock['exp_data']) == 0 or shock['wavelet_lvls'] == lvls):
                return
            
            shock['wavelet_lvls'] = lvls
            if lvls == 1:
                shock['exp_data_smoothed'] = shock['exp_data'].copy()
            else:
                shock['exp_data_smoothed'] = self.smoothed_data(shock['exp_data'].copy())

        elif key == 'series_name':          # being called many times when weights changing, don't know why yet
            self.name[self.idx] = val
            for shock in self.shock[self.idx]:
                shock[key] = self.name[self.idx]
        
        elif key == 'observable':
            for shock in self.shock[self.idx]:
                shock[key]['main'] = val[0]
                shock[key]['sub'] = val[1]
        
        elif key == 'time_offset':
            for shock in self.shock[self.idx]:
                shock[key] = val

                # only updated if optimize isn't running so it doesn't interfere with optimization
                if not parent.optimize_running: 
                    shock['opt_time_offset'] = val
                
        elif key == 'zone':
            for shock in self.shock[self.idx]:
                shock[key] = val
                shock['T_reactor'] = shock['T{:d}'.format(val)]
                shock['P_reactor'] = shock['P{:d}'.format(val)]
    
    def thermo_mix(self, shock=[]):
        parent = self.parent
        alias = self.current['species_alias']
        
        if len(shock) == 0: # if no shock given, assume shock is display_shock
            exp_mix = parent.display_shock['exp_mix']
            parent.display_shock['thermo_mix'] = mix = deepcopy(parent.display_shock['exp_mix'])
        else:
            exp_mix = shock['exp_mix']
            shock['thermo_mix'] = mix = deepcopy(shock['exp_mix'])
        
        for species in exp_mix:
            if species in alias:
                mix[alias[species]] = mix.pop(species)
            # don't run if a species isn't in the mech or mech doesn't exist
            # elif not hasattr(parent.mech.gas, 'species_names'): return
            # elif species not in parent.mech.gas.species_names: return  

    def rates(self, shock, rxnIdx=()):
        """Resets and updates all rates in shock

        Args:
            shock: Parameters of the shock to update rates for
            rxnIdx: List of reactions to update
        """
        if not self.parent.mech.isLoaded:
            return
        mech = self.parent.mech
        
        mech_out = mech.set_TPX(shock['T_reactor'], shock['P_reactor'], shock['thermo_mix'])
        if not mech_out['success']:
            self.parent.log.append(mech_out['message'])
            return
        
        if not rxnIdx:
            rxnIdxRange = range(mech.gas.n_reactions)
        elif not isinstance(rxn, list):    # does not function right now
            rxnIdxRange = [rxnIdx]
        else:
            rxnIdxRange = rxnIdx
        
        shock['rate_val'] = []
        for rxnIdx in rxnIdxRange: # TODO: an improvement would be to update given rxns
            shock['rate_val'].append(mech.gas.forward_rate_constants[rxnIdx])
        
        # print(shock['rate_val'])
        
        return shock['rate_val']
    
    def rate_bnds(self, shock: dict):
        """Update the bonds on the reaction rates based on a certain shock experiment

        Args:
            shock: Dictionary contain the shock information
        """
        if not self.parent.mech.isLoaded: return
        mech = self.parent.mech
        
        mech.set_TPX(shock['T_reactor'], shock['P_reactor'], shock['thermo_mix'])
            
        # reset mech back to reset values
        prior_mech = mech.reset()  # reset mechanism and get mech that it was
                
        # Get reset rates and rate bounds
        shock['rate_reset_val'] = []
        shock['rate_bnds'] = []
        for rxnIdx in range(mech.gas.n_reactions):
            if self.parent.mech_tree.rxn[rxnIdx]['rxnType'] in ['Arrhenius', 'Plog Reaction', 'Falloff Reaction']:
                resetVal = mech.gas.forward_rate_constants[rxnIdx]
                shock['rate_reset_val'].append(resetVal)
                rate_bnds = mech.rate_bnds[rxnIdx]['limits'](resetVal)
                shock['rate_bnds'].append(rate_bnds)
                
            else:   # skip if not Arrhenius type
                shock['rate_reset_val'].append(np.nan)
                shock['rate_bnds'].append([np.nan, np.nan])
        
        # set mech to prior mech
        mech.coeffs = prior_mech
        mech.modify_reactions(mech.coeffs)
    
    def set_coef_reset(self, rxnIdx, coefName):
        mech = self.parent.mech
        reset_val = self.parent.mech.coeffs[rxnIdx][coefName]
        mech.coeffs_bnds[rxnIdx][coefName]['resetVal'] = reset_val  
    
    def load_full_series(self):
        self.set('exp_data')
        self.parent.series_viewer.update()

    def change_shock(self):
        """Charge the selected shock based on the choice in the GUI"""
        parent = self.parent
        # parent.user_settings.save(save_all = False)
        
        self.update_idx()
        # if current exp path doesn't match the current loaded path
        if parent.path['exp_main'] != self.path[self.idx]:
            parent.exp_main_box.setText(self.path[self.idx])
            self.update_idx()
        
        # Assign shock to main's display_shock
        shock = self.shock[self.idx][self.shock_idx]
        parent.display_shock = shock
        
        # load data and set if not already loaded
        if 'exp_data' not in shock['err']:
            if shock['exp_data'].size == 0:
                self.set('exp_data')
        
        if np.isnan(shock['T_reactor']):
            self.set('zone', shock['zone'])
        
        # if weights not set, set them otherwise load
        parameters = [shock[key] for key in ['weight_max', 'weight_min', 'weight_shift', 'weight_k']]
        if np.isnan(np.hstack(parameters)).any():  # if weight parameters aren't set, create from gui
            parent.weight.update()
        else:                           # set weights if they're set upon reloading shock
            parent.weight.set_boxes()

        # if uncertainties not set, set them otherwise load
        parameters = [shock[key] for key in ['unc_max', 'unc_min', 'unc_shift', 'unc_k', 'unc_cutoff']]
        if np.isnan(np.hstack(parameters)).any():  # if weight parameters aren't set, create from gui
            parent.exp_unc.update()
        else:                           # set weights if they're set upon reloading shock
            parent.exp_unc.set_boxes()
        
        # Set rate bnds if unassigned
        if not shock['rate_bnds']:
            shock['rate_bnds'] = deepcopy(parent.display_shock['rate_bnds'])
        
        # Set observable if unassigned
        if shock['observable'] == {'main': '', 'sub': None}:
            parent.plot.observable_widget.update_observable()
        else:
            parent.plot.observable_widget.set_observable(shock['observable'])
        
        # Update exp parameters (this assumes units are K, Pa, m/s)
        for var_type in ['T1', 'P1', 'u1']:
            parent.shock_widgets.set_shock_value_box(var_type)

        # Update mix
        parent.plot.signal.clear_sim()
        parent.mix.update_species()   # post-shock conditions called within, which runs SIM
        
        # Update rate bnds
        # self.rates(shock)
        self.rate_bnds(shock)
        
        # Update set (if exp path matches)
        parent.series_viewer.update(self.shock_idx) # update only the current shock
        
        # Update signal and raw_signal plots
        if parent.display_shock['exp_data'].size > 0:
            parent.plot.signal.update(update_lim=True)
            # create new shading plot
            # Background reset causes disappearing data on new shock load
            #parent.plot.signal.set_background()           # Reset background
        else:
            parent.plot.signal.clear_plot() 

        if parent.display_shock['raw_data'].size > 0:
            parent.plot.raw_sig.update(update_lim=True)
            #parent.plot.raw_sig.canvas.draw()
        else:
            parent.plot.raw_sig.clear_plot()
