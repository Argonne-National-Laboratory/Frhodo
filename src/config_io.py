#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This file is part of Frhodo. Copyright © 2020, UChicago Argonne, LLC
# and licensed under BSD-3-Clause. See License.txt in the top-level 
# directory for license and copyright information.

try:
    import ruamel_yaml as yaml
except ImportError:
    from ruamel import yaml

import numpy as np

import sys, io, pathlib, collections

def FlowMap(*args, **kwargs):
    m = yaml.comments.CommentedMap(*args, **kwargs)
    m.fa.set_flow_style()
    return m

def FlowList(*args, **kwargs):
    lst = yaml.comments.CommentedSeq(*args, **kwargs)
    lst.fa.set_flow_style()
    return lst

# Improved float formatting requires Numpy >= 1.14
if hasattr(np, 'format_float_positional'):
    def float2string(data):
        if data == 0:
            return '0.0'
        elif 0.01 <= abs(data) < 10000:
            return np.format_float_positional(data, trim='0')
        else:
            return np.format_float_scientific(data, trim='0')
else:
    def float2string(data):
        return repr(data)

def represent_float(self, data):
    # type: (Any) -> Any
    if data != data:
        value = '.nan'
    elif data == self.inf_value:
        value = '.inf'
    elif data == -self.inf_value:
        value = '-.inf'
    else:
        value = float2string(data)

    return self.represent_scalar(u'tag:yaml.org,2002:float', value)

yaml.RoundTripRepresenter.add_representer(float, represent_float)

def deep_convert_dict(layer):   # convert all OrderedDict into dict to remove comments
    to_ret = layer              # they add a space each time prrogram is opened
    if isinstance(layer, collections.OrderedDict):
        to_ret = dict(layer)

    try:
        for key, value in to_ret.items():
            to_ret[key] = deep_convert_dict(value)
    except AttributeError:
        pass

    return to_ret

def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

class GUI_Config(yaml.YAML):
    def __init__(self):
        super().__init__()
        self.default_flow_style = False
        self.block_seq_indent = 2
        # self.indent = 4
        self.encoding = 'utf-8'
        self.allow_unicode = True
        self.width = 80
        
        self.loader = yaml.RoundTripLoader
        
        self.setDefault()
        
    def setDefault(self):
        self.settings = {'Directory Settings': {
                            'directory file': '',
                            },
                         'Experiment Settings': {
                            'temperature units': {'zone 1': 'K',    'zone 2': 'K',    'zone 5': 'K'},
                            'pressure units':    {'zone 1': 'Torr', 'zone 2': 'Torr', 'zone 5': 'atm'},
                            'velocity units': 'm/s',
                            },
                         'Reactor Settings': {
                            'reactor': 'Incident Shock Reactor',
                            'solve energy': True,
                            'frozen composition': False,
                            'simulation end time': {'value': 12.0, 'units': 'μs'},
                            'ODE solver': 'BDF',
                            'simulation interpolation factor': 1,
                            'ODE tolerance': {'relative': 1E-6, 'absolute': 1E-8},
                            },
                         'Optimization Settings': {
                            'time uncertainty': 0.0,
                            'objective function type': 'Residual',
                            'objective function scale': 'Linear',
                            'loss function alpha': -2.00,
                            'loss function c': 1.00,
                            'Bayesian distribution type': 'Automatic',
                            'Bayesian uncertainty sigma': 2.0,
                            'multiprocessing': True,
                            'enabled':                       {'global': True,                             'local': True},
                            'algorithm':                     {'global': 'CRS (Controlled Random Search)', 'local': 'Subplex'},
                            'initial step':                  {'global': 5.0E-2,                           'local': 1.0E-3},
                            'stop criteria type':            {'global': 'Iteration Maximum',              'local': 'Iteration Maximum'},
                            'stop criteria value':           {'global': 2500,                             'local': 2500},
                            'relative x tolerance':          {'global': 1.0E-3,                           'local': 1.0E-4},
                            'relative fcn tolerance':        {'global': 5.0E-2,                           'local': 1.0E-3},
                            'initial population multiplier': {'global': 1.0},
                            'weight function': {
                                'max': 100.0,
                                'min': [0.0, 0.0],
                                'time location': [4.5, 35.0],
                                'inverse growth rate': [0, 0.7],
                                },
                            'uncertainty function': {
                                'max': [5.0, 20.0],
                                'min': 5.0,
                                'time location': [4.5, 35.0],
                                'inverse growth rate': [0, 0.7],
                                'cutoff location': [0.0, 100],
                                'shading': True,
                                },
                            },
                         'Plot Settings': {
                            'x-scale': 'linear',
                            'y-scale': 'abslog',
                            },
                        }
                         
    def to_yaml(self, dest=None):
        settings = self.settings
        out = yaml.comments.CommentedMap(settings)
        
        # reformat certain sections 
        toFlowMap = [['Experiment Settings', 'temperature units'],
                     ['Experiment Settings', 'pressure units'],
                    ]
        toFlowList = [['Optimization Settings', 'weight function', 'min'],
                      ['Optimization Settings', 'weight function', 'time location'],
                      ['Optimization Settings', 'weight function', 'inverse growth rate'],
                      ['Optimization Settings', 'uncertainty function', 'max'],
                      ['Optimization Settings', 'uncertainty function', 'time location'],
                      ['Optimization Settings', 'uncertainty function', 'inverse growth rate'],
                      ['Optimization Settings', 'uncertainty function', 'cutoff location'],
                     ]
        for FlowType, toFlow in {'Map': toFlowMap, 'List': toFlowList}.items():
            for keys in toFlow:
                out_element = out
                settings_element = settings
                for key in keys[:-1]:
                    out_element = out_element[key]
                    settings_element = settings_element[key]
                
                if FlowType == 'Map':
                    out_element[keys[-1]] = FlowMap(settings_element[keys[-1]])
                elif FlowType == 'List':
                    out_element[keys[-1]] = FlowList(settings_element[keys[-1]])
        
        # add spacing between main sections
        for key in list(self.settings.keys())[1:]:
            out.yaml_set_comment_before_after_key(key, before='\n')
        
        # if node.note:
            # note = textwrap.dedent(node.note.rstrip())
            # if '\n' in note:
                # note = yaml.scalarstring.PreservedScalarString(note)
            # out['note'] = note

        # self.dump(representer.represent_dict(out), dest)
        if dest is None: 
            self.dump(out, sys.stdout)
        else:
            with io.open(dest, 'w', encoding='utf-8') as configFile:
                self.dump(out, configFile)

    def from_yaml(self, src=None):
        if src is None: return
        if not src.exists(): return
        
        with io.open(src, 'r', encoding='utf-8') as configFile:
            data = deep_convert_dict(self.load(configFile))

        self.settings = update(self.settings, data)


class GUI_settings:
    def __init__(self, parent):
        self.parent = parent    # Need to find a better solution than passing parent
        self.cfg_io = GUI_Config()
        self.cfg = self.cfg_io.settings
    
    def load(self):
        parent = self.parent
        
        self.cfg_io.from_yaml(parent.path['default_config'])
        settings = {'directory': self.cfg['Directory Settings'],
                    'exp': self.cfg['Experiment Settings'],
                    'reactor': self.cfg['Reactor Settings'],
                    'opt': self.cfg['Optimization Settings'],
                    'plot': self.cfg['Plot Settings'],
                   }

        ## Set Experiment Settings ##
        # Set Temperature Units
        parent.T1_units_box.setCurrentText(f"[{settings['exp']['temperature units']['zone 1']}]")
        parent.T2_units_box.setCurrentText(f"[{settings['exp']['temperature units']['zone 2']}]")
        parent.T5_units_box.setCurrentText(f"[{settings['exp']['temperature units']['zone 5']}]")
        
        # Set Pressure Units
        parent.P1_units_box.setCurrentText(f"[{settings['exp']['pressure units']['zone 1']}]")
        parent.P2_units_box.setCurrentText(f"[{settings['exp']['pressure units']['zone 2']}]")
        parent.P5_units_box.setCurrentText(f"[{settings['exp']['pressure units']['zone 5']}]")
        
        # Set Incident Velocity Units
        parent.u1_units_box.setCurrentText(f"[{settings['exp']['velocity units']}]")
        
        
        ## Set Reactor Settings ##
        parent.reactor_select_box.setCurrentText(settings['reactor']['reactor'])
        parent.solve_energy_box.setChecked(settings['reactor']['solve energy'])
        parent.frozen_comp_box.setChecked(settings['reactor']['frozen composition'])
        parent.end_time_value_box.setValue(settings['reactor']['simulation end time']['value'])
        parent.end_time_units_box.setCurrentText(f"[{settings['reactor']['simulation end time']['units']}]")
        parent.ODE_solver_box.setCurrentText(settings['reactor']['ODE solver'])
        parent.sim_interp_factor_box.setValue(settings['reactor']['simulation interpolation factor'])
        # parent.ODE_rtol_box.setValue(settings['reactor']['ODE tolerance']['relative'])    # TODO: Temporarily disabled until box is changed
        # parent.ODE_atol_box.setValue(settings['reactor']['ODE tolerance']['absolute'])    # TODO: Temporarily disabled until box is changed

        ## Set Optimization Settings ##
        parent.time_unc_box.setValue(settings['opt']['time uncertainty'])
        parent.obj_fcn_type_box.setCurrentIndex(parent.obj_fcn_type_box.findText(settings['opt']['objective function type']))
        parent.obj_fcn_scale_box.setCurrentIndex(parent.obj_fcn_scale_box.findText(settings['opt']['objective function scale']))
        parent.loss_alpha_box.setValue(settings['opt']['loss function alpha'])
        parent.loss_c_box.setValue(settings['opt']['loss function c'])
        parent.bayes_dist_type_box.setCurrentIndex(parent.bayes_dist_type_box.findText(settings['opt']['Bayesian distribution type']))
        parent.bayes_unc_sigma_box.setValue(settings['opt']['Bayesian uncertainty sigma'])
        parent.multiprocessing_box.setChecked(settings['opt']['multiprocessing'])
        
        # Update Global and Local Settings
        for opt_type in ['global', 'local']:
            widget = parent.optimization_settings.widgets[opt_type]

            if opt_type == 'global':
                parent.global_opt_enable_box.setChecked(settings['opt']['enabled'][opt_type])
                parent.global_opt_choice_box.setCurrentText(settings['opt']['algorithm'][opt_type])
                widget['initial_pop_multiplier'].setValue(settings['opt']['initial population multiplier'][opt_type])
            else:
                parent.local_opt_enable_box.setChecked(settings['opt']['enabled'][opt_type])
                parent.local_opt_choice_box.setCurrentText(settings['opt']['algorithm'][opt_type])
                
            widget['initial_step'].setValue(settings['opt']['initial step'][opt_type])
            widget['stop_criteria_type'].setCurrentText(settings['opt']['stop criteria type'][opt_type])
            widget['stop_criteria_val'].setValue(settings['opt']['stop criteria value'][opt_type])
            widget['xtol_rel'].setValue(settings['opt']['relative x tolerance'][opt_type])
            widget['ftol_rel'].setValue(settings['opt']['relative fcn tolerance'][opt_type])
        
        # Update weight function
        shock = parent.display_shock
        shock['weight_max'] = [settings['opt']['weight function']['max']]
        shock['weight_min'] = settings['opt']['weight function']['min']
        shock['weight_shift'] = settings['opt']['weight function']['time location']
        shock['weight_k'] = settings['opt']['weight function']['inverse growth rate']

        parent.weight.set_boxes()

        # Update uncertainty function
        shock['unc_max'] = settings['opt']['uncertainty function']['max']
        shock['unc_min'] = [settings['opt']['uncertainty function']['min']]
        shock['unc_shift'] = settings['opt']['uncertainty function']['time location']
        shock['unc_k'] = settings['opt']['uncertainty function']['inverse growth rate']
        shock['unc_cutoff'] = settings['opt']['uncertainty function']['cutoff location']

        parent.exp_unc.set_boxes()

        parent.unc_shading_box.setChecked(settings['opt']['uncertainty function']['shading'])

        ## Set Plot Settings ##
        parent.plot.signal._set_scale('x', settings['plot']['x-scale'], parent.plot.signal.ax[1], True)
        parent.plot.signal._set_scale('y', settings['plot']['y-scale'], parent.plot.signal.ax[1], True)

        ## Set Shock and Directory File
        parent.shock_choice_box.setValue(1)
        parent.path_file_box.setPlainText(str(settings['directory']['directory file']))
    
    def save(self, save_all=False):
        parent = self.parent
        
        settings = {'directory': self.cfg['Directory Settings'],
                    'exp': self.cfg['Experiment Settings'],
                    'reactor': self.cfg['Reactor Settings'],
                    'opt': self.cfg['Optimization Settings'],
                    'plot': self.cfg['Plot Settings'],
                   }
        
        settings['directory']['directory file'] = str(parent.path['path_file'])
        
        ## Set Experiment Settings ##
        # Set Temperature/Pressure Units
        for i in [1, 2, 5]:
            T_unit = eval(f'parent.T{i}_units_box.currentText()').lstrip('[').rstrip(']')
            P_unit = eval(f'parent.P{i}_units_box.currentText()').lstrip('[').rstrip(']')
            settings['exp']['temperature units'][f'zone {i}'] = T_unit
            settings['exp']['pressure units'][f'zone {i}'] = P_unit
        
        # Set Incident Velocity Units
        settings['exp']['velocity units'] = parent.u1_units_box.currentText().lstrip('[').rstrip(']')
        
        ## Set Reactor Settings ##
        settings['reactor']['reactor'] = parent.reactor_select_box.currentText()
        settings['reactor']['solve energy'] = parent.solve_energy_box.isChecked()
        settings['reactor']['frozen composition'] = parent.frozen_comp_box.isChecked()
        settings['reactor']['simulation end time']['value'] = parent.end_time_value_box.value()
        sim_time_units = parent.end_time_units_box.currentText().lstrip('[').rstrip(']')
        settings['reactor']['simulation end time']['units'] = sim_time_units
        settings['reactor']['ODE solver'] = parent.ODE_solver_box.currentText()
        settings['reactor']['simulation interpolation factor'] = parent.sim_interp_factor_box.value()
        # parent.ODE_rtol_box.setValue(settings['reactor']['ODE tolerance']['relative'])    # TODO: Temporarily disabled until box is changed
        # parent.ODE_atol_box.setValue(settings['reactor']['ODE tolerance']['absolute'])    # TODO: Temporarily disabled until box is changed

        ## Set Optimization Settings ##
        settings['opt']['time uncertainty'] = parent.time_unc_box.value()
        settings['opt']['objective function type'] = parent.obj_fcn_type_box.currentText()
        settings['opt']['objective function scale'] = parent.obj_fcn_scale_box.currentText()
        settings['opt']['loss function alpha'] = parent.loss_alpha_box.value()
        settings['opt']['loss function c'] = parent.loss_c_box.value()
        settings['opt']['Bayesian distribution type'] = parent.bayes_dist_type_box.currentText()
        settings['opt']['Bayesian uncertainty sigma'] = parent.bayes_unc_sigma_box.value()
        settings['opt']['multiprocessing'] = parent.multiprocessing_box.isChecked()

        # Update Global and Local Settings
        for opt_type in ['global', 'local']:
            widget = parent.optimization_settings.widgets[opt_type]

            if opt_type == 'global':
                settings['opt']['enabled'][opt_type] = parent.global_opt_enable_box.isChecked()
                settings['opt']['algorithm'][opt_type] = parent.global_opt_choice_box.currentText()
                settings['opt']['initial population multiplier'][opt_type] = widget['initial_pop_multiplier'].value()
            else:
                settings['opt']['enabled'][opt_type] = parent.local_opt_enable_box.isChecked()
                settings['opt']['algorithm'][opt_type] = parent.local_opt_choice_box.currentText()
                
            settings['opt']['initial step'][opt_type] = widget['initial_step'].value()
            settings['opt']['stop criteria type'][opt_type] = widget['stop_criteria_type'].currentText()
            settings['opt']['stop criteria value'][opt_type] = widget['stop_criteria_val'].value()
            settings['opt']['relative x tolerance'][opt_type] = widget['xtol_rel'].value()
            settings['opt']['relative fcn tolerance'][opt_type] = widget['ftol_rel'].value()
        
        # Update weight function
        shock = parent.display_shock
        settings['opt']['weight function']['max'] = shock['weight_max'][0]
        settings['opt']['weight function']['min'] = shock['weight_min']
        settings['opt']['weight function']['time location'] = shock['weight_shift']
        settings['opt']['weight function']['inverse growth rate'] = shock['weight_k']
        
        # Update uncertainty function
        settings['opt']['uncertainty function']['max'] = shock['unc_max']
        settings['opt']['uncertainty function']['min'] = shock['unc_min'][0]
        settings['opt']['uncertainty function']['time location'] = shock['unc_shift']
        settings['opt']['uncertainty function']['inverse growth rate'] = shock['unc_k']
        settings['opt']['uncertainty function']['cutoff location'] = shock['unc_cutoff']
        settings['opt']['uncertainty function']['shading'] = parent.unc_shading_box.isChecked()

        ## Set Plot Settings ##
        settings['plot']['x-scale'] = parent.plot.signal.ax[1].get_xscale()
        settings['plot']['y-scale'] = parent.plot.signal.ax[1].get_yscale()
        
        self.cfg_io.to_yaml(parent.path['default_config'])
        

if __name__ == '__main__':
    gui_cfg = GUI_Config()

    path = {}
    path['default_config'] = pathlib.Path('default_config.yaml')

    # gui_cfg.dump(gui_cfg.settings, sys.stdout)
    gui_cfg.to_yaml(path['default_config'])
    gui_cfg.from_yaml(path['default_config'])