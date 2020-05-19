# I wanted to make a temp file in memory to write/read converted cantera mechs to and from. Best I can do
# is to write to the temp directory and delete it immediately.  

import cantera as ct
from cantera import ck2yaml
import tempfile, os

tf = tempfile.NamedTemporaryFile(suffix='.yaml', prefix='generated_mech', delete=False)
ck2yaml.convert_mech('Styrene.mech', thermo_file='Styrene.therm', transport_file=None, surface_file=None,
                 phase_name='gas', out_name=tf.name, quiet=False, permissive=True)
gas = ct.Solution(tf.name)

tf.close()
os.remove(tf.name)