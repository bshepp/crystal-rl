#!/usr/bin/env python3
"""Debug script to inspect raw QE bands output."""
import os
import subprocess
import tempfile

import numpy as np
from ase.build import bulk
from ase.calculators.espresso import Espresso, EspressoProfile

si = bulk("Si", "diamond", a=5.43)
scratch = tempfile.mkdtemp()
work_dir = os.path.join(scratch, "bands_test")
outdir = os.path.join(work_dir, "out")
os.makedirs(outdir, exist_ok=True)

pseudos = {"Si": "Si.pbe-n-rrkjus_psl.1.0.0.UPF"}
profile = EspressoProfile(
    command="mpirun --allow-run-as-root -np 2 pw.x",
    pseudo_dir="/opt/pseudopotentials",
)

# Step 1: SCF
scf_input = {
    "control": {
        "calculation": "scf",
        "outdir": outdir,
        "prefix": "bands",
        "pseudo_dir": "/opt/pseudopotentials",
        "verbosity": "high",
        "tprnfor": True,
        "tstress": True,
    },
    "system": {
        "ecutwfc": 30.0,
        "ecutrho": 240.0,
        "occupations": "smearing",
        "smearing": "cold",
        "degauss": 0.02,
    },
    "electrons": {"conv_thr": 1e-6, "mixing_beta": 0.7},
}
calc = Espresso(
    profile=profile,
    input_data=scf_input,
    pseudopotentials=pseudos,
    kpts=(4, 4, 4),
    directory=work_dir,
)
atoms = si.copy()
atoms.calc = calc
e = atoms.get_potential_energy()
print(f"SCF done: E = {e:.6f} eV")

# Step 2: Write bands input manually
path_obj = si.cell.bandpath(npoints=10)
kpts = path_obj.kpts
print(f"k-path: {len(kpts)} points")

lines = [
    "&CONTROL",
    "  calculation = 'bands'",
    f"  outdir = '{outdir}'",
    "  prefix = 'bands'",
    "  pseudo_dir = '/opt/pseudopotentials'",
    "  verbosity = 'high'",
    "/",
    "&SYSTEM",
    "  ecutwfc = 30.0",
    "  ecutrho = 240.0",
    "  occupations = 'smearing'",
    "  smearing = 'cold'",
    "  degauss = 0.02",
    "  ntyp = 1",
    "  nat = 2",
    "  ibrav = 0",
    "  nbnd = 22",
    "/",
    "&ELECTRONS",
    "  conv_thr = 1e-6",
    "/",
    "ATOMIC_SPECIES",
    "Si 28.085 Si.pbe-n-rrkjus_psl.1.0.0.UPF",
    "K_POINTS {crystal}",
    str(len(kpts)),
]
w = 1.0 / len(kpts)
for k in kpts:
    lines.append(f"{k[0]:.14f} {k[1]:.14f} {k[2]:.14f} {w:.14f}")
lines.append("CELL_PARAMETERS angstrom")
for r in si.cell[:]:
    lines.append(f"{r[0]:.14f} {r[1]:.14f} {r[2]:.14f}")
lines.append("ATOMIC_POSITIONS angstrom")
for s, p in zip(si.get_chemical_symbols(), si.get_positions()):
    lines.append(f"{s} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}")

pwi = os.path.join(work_dir, "bands.pwi")
with open(pwi, "w") as f:
    f.write("\n".join(lines) + "\n")

print("Bands input written")

pwo = os.path.join(work_dir, "bands.pwo")
cmd = "mpirun --allow-run-as-root -np 2 pw.x -in bands.pwi".split()
with open(pwo, "w") as f:
    proc = subprocess.run(cmd, cwd=work_dir, stdout=f, stderr=subprocess.PIPE, timeout=300)
print(f"pw.x return code: {proc.returncode}")

# Analyze the output
with open(pwo) as f:
    content = f.readlines()
print(f"Total output lines: {len(content)}")

# Find lines with k-points or eigenvalues
print("\n=== Key lines ===")
for i, line in enumerate(content):
    stripped = line.strip()
    if any(kw in stripped.lower() for kw in ["k =", "bands (ev)", "band energ", "eigenval", "end of band"]):
        # Print context: this line plus next 5
        for j in range(i, min(i + 6, len(content))):
            print(f"  L{j}: {content[j].rstrip()}")
        print()
