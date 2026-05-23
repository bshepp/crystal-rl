#!/usr/bin/env python3
"""Analyze JARVIS dft_3d dataset for effective mass and band gap data."""

from jarvis.db.figshare import data as jdata
import numpy as np

d = jdata('dft_3d')

# Count records with effective mass data
has_emass = [r for r in d if r.get('effective_masses_300K') not in [None, '', 'na', 'None', {}]]
has_avg_emass = [r for r in d if isinstance(r.get('avg_elec_mass'), (int, float)) and r['avg_elec_mass'] > 0]
has_avg_hmass = [r for r in d if isinstance(r.get('avg_hole_mass'), (int, float)) and r['avg_hole_mass'] > 0]
has_gap = [r for r in d if isinstance(r.get('optb88vdw_bandgap'), (int, float))]
has_mbj = [r for r in d if isinstance(r.get('mbj_bandgap'), (int, float))]

print(f"Total: {len(d)}")
print(f"Has effective_masses_300K: {len(has_emass)}")
print(f"Has avg_elec_mass > 0: {len(has_avg_emass)}")
print(f"Has avg_hole_mass > 0: {len(has_avg_hmass)}")
print(f"Has optb88vdw_bandgap: {len(has_gap)}")
print(f"Has mbj_bandgap: {len(has_mbj)}")

# Show sample
for r in d[:2000]:
    em = r.get('avg_elec_mass')
    hm = r.get('avg_hole_mass')
    if isinstance(em, (int, float)) and em > 0 and isinstance(hm, (int, float)) and hm > 0:
        print(f"\nSample: {r['jid']} {r['formula']}")
        print(f"  avg_elec_mass: {em}")
        print(f"  avg_hole_mass: {hm}")
        print(f"  optb88vdw_bandgap: {r.get('optb88vdw_bandgap')}")
        print(f"  effective_masses_300K: {str(r.get('effective_masses_300K'))[:300]}")
        atoms = r.get('atoms')
        print(f"  atoms type: {type(atoms).__name__}")
        if isinstance(atoms, dict):
            print(f"  atoms keys: {list(atoms.keys())}")
        break

# How many have BOTH m* and gap?
both = [r for r in d 
        if isinstance(r.get('avg_elec_mass'), (int, float)) and r['avg_elec_mass'] > 0
        and isinstance(r.get('optb88vdw_bandgap'), (int, float))]
print(f"\nHas BOTH avg_elec_mass>0 AND gap: {len(both)}")

# Distribution of gap values
gap_vals = [r['optb88vdw_bandgap'] for r in has_gap]
print(f"\nBand gap stats:")
print(f"  Mean: {np.mean(gap_vals):.3f} eV")
print(f"  Median: {np.median(gap_vals):.3f} eV")
print(f"  Semiconductors (gap > 0): {sum(1 for g in gap_vals if g > 0)}")
print(f"  Metals (gap = 0): {sum(1 for g in gap_vals if g == 0)}")

# What crystal systems are present?
from collections import Counter
spgs = Counter(r.get('spg_symbol', 'unknown') for r in d[:5000])
crys = Counter(r.get('crys', 'unknown') for r in d)
print(f"\nCrystal systems:")
for c, n in crys.most_common(10):
    print(f"  {c}: {n}")

# Check atoms format
sample = d[0]['atoms']
if isinstance(sample, dict):
    print(f"\nAtoms dict keys: {sorted(sample.keys())}")
    for k, v in sample.items():
        print(f"  {k}: {str(v)[:100]}")
