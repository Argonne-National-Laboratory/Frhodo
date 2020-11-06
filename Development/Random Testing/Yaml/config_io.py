#!/usr/bin/env python3
# -*- coding: utf-8 -*-

try:
    import ruamel_yaml as yaml
except ImportError:
    from ruamel import yaml

import numpy as np

import pathlib, sys

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

class GUI_Config(yaml.YAML):
    def __init__(self):
        super().__init__()
        self.default_flow_style = False
        self.block_seq_indent = 2
        # self.indent = 4
        self.allow_unicode = True
        self.encoding = 'utf-8'
        self.width = 80
        
        self.loader = yaml.RoundTripLoader
        
        self.setDefault()
        
    def setDefault(self):
        self.settings = {'Directory Settings': {
                            'directory file': '',
                            'load full series': False,
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
                            'simulation end time': {'value': 12.0, 'units': 'us'},
                            'ODE solver': 'BDF',
                            'simulation interpolation factor': 1,
                            'ODE tolerance': {'relative': 1E-6, 'absolute': 1E-8},
                            },
                         'Optimization Settings': {
                            'time uncertainty': 0.0,
                            'loss function alpha': -2.00,
                            'loss function c': 1.00,
                            'multiprocessing': True,
                            'enabled':                  {'global': True,     'local': True},
                            'algorithm':                {'global': 'DIRECT', 'local': 'Subplex'},
                            'initial step':             {'global': 1.0E-2,   'local': 1.0E-2},
                            'relative tolerance x':     {'global': 1.0E-4,   'local': 1.0E-4},
                            'relative tolerance fcn':   {'global': 5.0E-4,   'local': 1.0E-3},
                            'weight function': {
                                'max': 100,
                                'min': [0, 0],
                                'time location': [0.5, 3.7],
                                'inverse growth rate': [0, 0.3],
                                },
                            },
                         'Plot Settings': {
                            'x-scale': 'linear',
                            'y-scale': 'abslog',
                            },
                        }
                         
    def to_yaml(self, dest=None):
        settings = self.settings
        # out = settings.copy()
        out = yaml.comments.CommentedMap(settings)
        
        # add spacing between main sections
        for key in list(self.settings.keys())[1:]:
            out.yaml_set_comment_before_after_key(key, before='\n')
        
        # reformat certain sections 
        toFlowMap = [['Experiment Settings', 'temperature units'],
                     ['Experiment Settings', 'pressure units'],
                    ]
        toFlowList = [['Optimization Settings', 'weight function', 'min'],
                      ['Optimization Settings', 'weight function', 'time location'],
                      ['Optimization Settings', 'weight function', 'inverse growth rate'],
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
        
        # if node.note:
            # note = textwrap.dedent(node.note.rstrip())
            # if '\n' in note:
                # note = yaml.scalarstring.PreservedScalarString(note)
            # out['note'] = note

        # self.dump(representer.represent_dict(out), dest)
        if dest is None: 
            self.dump(out, sys.stdout)
        else:
            with open(dest, 'w') as configFile:
                self.dump(out, configFile)

    def from_yaml(self, src=None):
        if src is None: return
        if not src.exists(): return
        
        with open(src, 'r') as configFile:
            self.settings.update(self.load(configFile))

if __name__ == '__main__':
    gui_cfg = GUI_Config()

    path = {}
    path['default_config'] = pathlib.Path('default_config.yaml')

    # gui_cfg.dump(gui_cfg.settings, sys.stdout)
    gui_cfg.to_yaml(path['default_config'])
    gui_cfg.from_yaml(path['default_config'])