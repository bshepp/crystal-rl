"""Quick verification of new fingerprint size."""
from qe_interface.structures import structure_to_fingerprint, make_seed_structure
import numpy as np

for name in ["Si", "Ge", "GaAs", "AlN", "InAs"]:
    atoms = make_seed_structure(name)
    fp = structure_to_fingerprint(atoms)
    print(f"{name:6s} FP size={len(fp)}  first5={fp[:5].round(3)}")

# Verify discriminability
fps = {}
for name in ["Si", "Ge", "GaAs", "AlN", "InAs"]:
    fps[name] = structure_to_fingerprint(make_seed_structure(name))

pairs = [("Si","Ge"), ("Si","GaAs"), ("Ge","GaAs"), ("AlN","GaAs"), ("Si","AlN")]
for a, b in pairs:
    d = np.linalg.norm(fps[a] - fps[b])
    print(f"  L2({a},{b}) = {d:.3f}")
