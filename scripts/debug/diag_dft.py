"""Quick DFT diagnostic - test if QE is working."""
import sys
import os
import tempfile

from qe_interface.calculator import QECalculator
from qe_interface.structures import make_seed_structure

atoms = make_seed_structure("Si")
print(f"Silicon: {atoms.get_chemical_symbols()}, cell volume: {atoms.get_volume():.1f}")

# Run from temp directory
work_dir = tempfile.mkdtemp()
os.chdir(work_dir)
print(f"Working in: {work_dir}")

from qe_interface.calculator import QEConfig
cfg = QEConfig(
    pseudo_dir="/workspace/data/pseudos",
    pw_command="mpirun --allow-run-as-root -np 2 pw.x",
)
calc = QECalculator(config=cfg)

try:
    result = calc.run_scf(atoms)
    print(f"SCF SUCCESS: energy = {result['energy']:.4f} eV")
except Exception as e:
    print(f"SCF FAILED: {e}")
    # Check for QE output files
    for f in os.listdir(work_dir):
        if f.endswith('.pwo') or f.endswith('.out'):
            print(f"\n--- {f} ---")
            with open(os.path.join(work_dir, f)) as fh:
                print(fh.read()[-2000:])
