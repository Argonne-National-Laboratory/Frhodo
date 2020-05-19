import sys
import numpy as np
import cantera as ct
from cantera import ck2yaml
from pathlib import Path

import soln2ck


mech_name = 'Mech_test.inp'

path = {'main': Path(sys.argv[0]).parents[0].resolve()}
path['mech'] = path['main'] / mech_name
path['yaml_mech'] = path['main'] / 'generated_mech.yaml'
path['out'] = path['main'] / 'mech.ck'


# Define a gas mixture at a high temperature that will undergo a reaction:
surfaces = ck2yaml.convert_mech(path['mech'], phase_name='gas', out_name=path['yaml_mech'], 
                quiet=False, permissive=True)

try:                                            # This test taken from ck2cti
    print('Validating mechanism...', end='')
    gas = ct.Solution(str(path['yaml_mech']))
    for surfname in surfaces:
        phase = ct.Interface(outName, surfname, [gas])
    print('PASSED.')
except RuntimeError as e:
    print('FAILED.')
    print(e)                

gas = ct.Solution(str(path['yaml_mech']))
print(gas.reaction(0).high_rate)
print(gas.reaction(0).low_rate)
soln2ck.write(gas, path['out'], path['yaml_mech'])