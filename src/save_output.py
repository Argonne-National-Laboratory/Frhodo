# This file is part of Frhodo. Copyright © 2020, UChicago Argonne, LLC
# and licensed under BSD-3-Clause. See License.txt in the top-level 
# directory for license and copyright information.

import numpy as np
from tabulate import tabulate
import pathlib

from cantera.yaml2ck import convert as soln2ck
# from cantera import ck2cti        # Maybe later use ck2cti.Parser.writeCTI to write cti file

class Save:
    def __init__(self, parent):
        self.parent = parent
        self.path = parent.path
    
    def make_table(self, labels, data, name='', sig_fig = 3):
        labels = [label.replace(' ', ' ') for label in labels] # replace regular spaces with no-break space for excel
        num_col = np.shape(data)[0]
        if isinstance(sig_fig, list):   # Checks if sig_fig is a list
            if np.shape(sig_fig)[0] < num_col:  # if number of values is less than number of inputs, duplicate last entry
                sig_fig.append(np.ones(num_col - np.shape(sig_fig)[0])*sig_fig[-1]) 
            sig_fig = np.array(sig_fig).astype(int)
        else:
            sig_fig = (np.ones(num_col)*sig_fig).astype(int)    # if not a list, set all sig_fig as input sig_fig
            
        # ind = []
        # for i in range(0, int(np.floor(len(labels)/num_col) + 1)):
            # if i < int(np.floor(len(labels)/num_col)):
                # ind.append([num_col*i, num_col*i + num_col])
            # elif num_col*i != len(labels):
                # ind.append([num_col*i, len(labels)])
        
        ind = [[0, np.shape(data)[0]]]
        
        table_fill = []
        for i in range(0, len(ind)):
            widths = [len(i) for i in labels[ind[i][0]:ind[i][1]]]
            table_fill.append(labels[ind[i][0]:ind[i][1]])
            table_fill.append('-'*w for w in widths)
            if isinstance(data[ind[i][0]], np.ndarray):
                data = np.vstack(data)
                for n in range(0, np.shape(data)[1]):
                    line = data[ind[i][0]:ind[i][1],n]
                    formatted_line = []
                    for k, val in enumerate(line):
                        formatted_line.append('{:.{s}e}'.format(val, s = sig_fig[k]))
                    table_fill.append(formatted_line)
                pass
            else:
                table_fill.append(data[ind[i][0]:ind[i][1]])
            
            if i < len(ind) - 1:
                table_fill.append('')
            
        table = tabulate(table_fill, stralign='center', numalign='center', tablefmt='plain')
        table = table.splitlines()
        
        for n, row in enumerate(table): # Replace leading spaces with no-break space for excel
            leading_space_count = len(row) - len(row.lstrip(' '))
            if leading_space_count > 0:
                table[n] = ' '*leading_space_count + table[n][leading_space_count:]
            
        if len(name) > 0:
            table.insert(0, '='*len(table[0]))
            table.insert(0, '{0:<{w}s}'.format(name, w = len(table[0])))
            table.insert(0, '='*len(table[0]))
        # table.append('='*len(table[0]))
        
        return table
    
    def write_table(self, path, table, line_start = 0): # Notepad has a maximum line length of 1024 characters
        with open(path, 'w') as f:
            for i, line in enumerate(table):
                if i >= line_start:
                    f.write(line + '\n')  
    
    def all(self, SIM, save_var, units = 'CGS'):
        def find_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return idx
        
        parent = self.parent
        SIM = self.parent.SIM
        self.gas = self.parent.mech.gas
        parent.path_set.shock_output()
        path_set = parent.path_set.sim_output
        
        # if save integrator_time is checked or SIM failed, save all
        if save_var['integrator_time'] or not SIM.success:
            self.save_indices = range(np.shape(SIM.t_lab(units=units))[0])
        else:
            self.save_indices = []   # Find indices of closest values to the save values
            for t_save in save_var['output_time']:
                self.save_indices.append(find_nearest(SIM.t_lab(units=units), t_save))
        
        for parameter in save_var['parameters']:
            sim_var_path = path_set(parameter)
            self.sim_parameter(SIM, save_var, parameter, sim_var_path, units='CGS')
            if parameter == 'Density Gradient': # if Density Gradient is being saved, save Total Density Gradient
                sim_var_path = path_set('Total Density Gradient')
                self.sim_density_gradient(SIM, save_var, sim_var_path, units = 'CGS')
        
        # 
        # self.sim_species_time_histories(SIM, units = units)
        # self.sim_kinetic_time_histories(SIM, units = units)
        self.chemkin_format(self.gas)
        self.sim_log_txt(save_var)
        if save_var['save_plot']:
            # Setting dpi to current screen's dpi. This could be set explicitly as 150, 300 etc instead
            parent.plot.signal.fig.savefig(parent.path['Plot.png'], dpi='figure', format='png')
    
    def sim_parameter(self, SIM, save_var, parameter, path, units='CGS'):
        idx = self.save_indices       
        data = np.array(SIM.t_lab(units=units)[idx]).T + save_var['output_time_offset'] # add time offset to all
        
        header = ['time [s]']
        sub_type = SIM.all_var[parameter]['sub_type']
        SIM_name = SIM.all_var[parameter]['SIM_name']
        if sub_type is None:  # TODO: save in one file?
            header.append(parameter)
            data = np.vstack((data, getattr(SIM, SIM_name)(units=units)[idx]))
        elif 'species' in sub_type:
            if 'total' in sub_type:
                header.append('Total')
                data = np.vstack((data, getattr(SIM, SIM_name+'_tot')(units=units)[idx]))
                
            if len(save_var['species']) > 0:
                for species_idx, species_name in save_var['species'].items():
                    header.extend([species_name])
                    data = np.vstack((data, getattr(SIM, SIM_name)(units=units)[:,idx][species_idx]))
        elif 'rxn' in sub_type:
            if 'total' in sub_type:
                header.append('Total')
                data = np.vstack((data, getattr(SIM, SIM_name+'_tot')(units=units)[idx]))
                
            if len(save_var['reactions']) > 0:
                for rxn_idx, rxn_eqn in save_var['reactions'].items():
                    header.extend(['R{:.0f}'.format(rxn_idx+1)])
                    data = np.vstack((data, getattr(SIM, SIM_name)(units=units)[:,idx][rxn_idx]))
        
        self.write_table(path, self.make_table(header, data, sig_fig=3))
        
    def sim_density_gradient(self, SIM, save_var, path, units = 'CGS'): 
        name = 'Density Gradient Time History'
        if 'CGS' in units:
            labels = ['t_lab [s]', 'GRHO [g/cm4]']
        elif 'SI' in units:
            labels = ['t_lab [s]', 'GRHO [kg/m4]']
        
        data = np.array(SIM.t_lab(units=units)).T + save_var['output_time_offset'] # add time offset to all
        data = np.vstack((data, SIM.drhodz_tot(units=units)))
        self.write_table(path, self.make_table(labels, data, sig_fig = 6))
       
    def sim_log_txt(self, save_var): 
        name = self.path['Mech.ck'].parents[0].name
        if 'Sim' in name:    # Check for Sim in string (this should work beyond Sim 1)
            Sim_num = name
        else:               # if no Sim found in string, it must be Sim 1
            Sim_num = 'Sim 1'
            
        comment = save_var['comment']
        if save_var['output_time_offset'] != 0.0:
            addComment = 'Added Time Offset to output files'
            if len(comment) == 0 or comment.isspace():      # if no comment exists, make this comment
                comment = addComment
            else:
                comment = comment + '\n' + addComment
        
        comment = '\n\t'.join(comment.split('\n'))    # split and replace with tabs for formatting
        
        with open(self.path['Sim log'], 'a') as f: # open log in append mode
            f.write(Sim_num + ':\t' + comment + '\n')
        
    def chemkin_format(self, gas=[], path=[]):
        if not gas:
            gas = self.gas
        if not path:
            path = self.path['Mech.ck']
        
        soln2ck(gas, mechanism_path=path, sort_species="molar-mass", overwrite=True)