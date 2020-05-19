import io, re, subprocess, sys, pathlib
import numpy as np
import cantera as ct
# from cantera import ck2yaml
import ck2yaml
import matplotlib.pyplot as plt
from pathlib import Path

# from reduce_model import trim as model_trim
import soln2ck

path = {'main': pathlib.Path(sys.argv[0]).parents[0].resolve()}
path['mech'] = path['main'] / 'Styrene v0.3.0 - 120 torr.mech'
path['thermo'] = path['main'] / 'Styrene mech.therm'
# path['mech'] = path['main'] / 'nasa9-test.inp'
# path['thermo'] = path['main'] / 'nasa9-test-therm.dat'
path['Cantera_Mech'] = path['main'] / 'generated_mech.yaml'


# Define a gas mixture at a high temperature that will undergo a reaction:
surfaces = ck2yaml.convert_mech(path['mech'], thermo_file=path['thermo'], transport_file=None, surface_file=None,
                phase_name='gas', out_name=path['Cantera_Mech'], quiet=False, permissive=True)

try:                                            # This test taken from ck2yaml
    print('Validating mechanism...', end='')
    gas = ct.Solution(str(path['Cantera_Mech']))
    for surfname in surfaces:
        phase = ct.Interface(outName, surfname, [gas])
    print('PASSED.')
except RuntimeError as e:
    print('FAILED.')
    print(e)                



gas = ct.Solution(str(path['Cantera_Mech']))
soln2ck.write(gas, str(Path.cwd() / 'mech.ck'), path['Cantera_Mech'])
quit()
remove_elements = ['O', 'N']

remove_species = []
for remove_element in remove_elements:
    try:
        ele_num = gas.element_index(remove_element)
        for n in range(gas.n_species):
            if gas.n_atoms(n, ele_num) > 0.0:
                remove_species.append(gas.species_name(n))
    except:
        pass
    
# gas.n_atoms(0, element)
# remove_species = np.setdiff1d(gas.species_names, key_species)
trimmed_gas = model_trim('generated_mech.yaml', remove_species, 'reduced_mech.yaml')
soln2ck.write(trimmed_gas, Path.cwd() / 'reduced_mech.ck')