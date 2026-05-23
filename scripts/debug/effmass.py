#!/usr/bin/env python3
"""Debug effective mass computation on saved Si band data."""

import numpy as np
from ase.build import bulk
from scipy.constants import electron_mass, eV, hbar, angstrom

# Load the band data by running the calculation
import tempfile, logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

from qe_interface.calculator import QECalculator, QEConfig

si = bulk("Si", "diamond", a=5.43)

scratch = tempfile.mkdtemp(prefix="qe_debug_")
config = QEConfig(
    pw_command="mpirun --allow-run-as-root -np 2 pw.x",
    pseudo_dir="/opt/pseudopotentials",
    scratch_dir=scratch,
    ecutwfc=30.0,
    ecutrho=240.0,
    conv_thr=1e-6,
    kpoints=(4, 4, 4),
)

qe = QECalculator(config)
result = qe.run_bands(si, npoints=30)

kpoints = result.kpoints  # Nx3 crystal coords
energies = result.band_energies  # (nk, nbands)
lattice = si.cell[:]

print(f"\nkpoints shape: {kpoints.shape}")
print(f"energies shape: {energies.shape}")
print(f"lattice:\n{lattice}")

# Reproduce what compute_effective_mass does
n_valence = 8
n_occupied = n_valence // 2  # = 4

valence_band = energies[:, n_occupied - 1]  # band index 3
conduction_band = energies[:, n_occupied]  # band index 4

vbm_k = np.argmax(valence_band)
cbm_k = np.argmin(conduction_band)

print(f"\nVBM at k-index {vbm_k}: E = {valence_band[vbm_k]:.4f} eV")
print(f"CBM at k-index {cbm_k}: E = {conduction_band[cbm_k]:.4f} eV")
print(f"Band gap: {conduction_band[cbm_k] - valence_band[vbm_k]:.4f} eV")

# Convert k-points to Cartesian
recip = 2 * np.pi * np.linalg.inv(lattice).T
k_cart = kpoints @ recip
print(f"\nReciprocal lattice:\n{recip}")

# Cumulative distance
dk = np.linalg.norm(np.diff(k_cart, axis=0), axis=1)
k_dist = np.concatenate([[0], np.cumsum(dk)])

print(f"\nFirst 10 k-points (crystal coords):")
for i in range(min(10, len(kpoints))):
    print(f"  k[{i:2d}] = {kpoints[i]} | cart = {k_cart[i]} | dist = {k_dist[i]:.6f}")

print(f"\nValence band (idx 3) near VBM (idx {vbm_k}):")
for i in range(max(0, vbm_k - 4), min(len(valence_band), vbm_k + 5)):
    print(f"  k[{i:2d}] dist={k_dist[i]:.6f} E={valence_band[i]:.4f}")

print(f"\nConduction band (idx 4) near CBM (idx {cbm_k}):")
for i in range(max(0, cbm_k - 4), min(len(conduction_band), cbm_k + 5)):
    print(f"  k[{i:2d}] dist={k_dist[i]:.6f} E={conduction_band[i]:.4f}")

# Now do the polynomial fit at VBM
hw = 3
lo = max(0, vbm_k - hw)
hi = min(len(valence_band), vbm_k + hw + 1)
k_local = k_dist[lo:hi] - k_dist[vbm_k]
e_local = valence_band[lo:hi]
print(f"\nFit window for VBM (idx {vbm_k}): [{lo}:{hi}]")
print(f"  k_local = {k_local}")
print(f"  e_local = {e_local}")
coeffs = np.polyfit(k_local, e_local, 2)
d2E = 2.0 * coeffs[0]
print(f"  polyfit coeffs = {coeffs}")
print(f"  d²E/dk² = {d2E}")
d2E_SI = d2E * eV * (angstrom ** 2)
if abs(d2E_SI) > 1e-30:
    m_star = (hbar ** 2 / d2E_SI) / electron_mass
    print(f"  m* (hole) = {m_star:.4f} m_e")
else:
    print(f"  d2E_SI too small: {d2E_SI}")

# Same for CBM
lo = max(0, cbm_k - hw)
hi = min(len(conduction_band), cbm_k + hw + 1)
k_local = k_dist[lo:hi] - k_dist[cbm_k]
e_local = conduction_band[lo:hi]
print(f"\nFit window for CBM (idx {cbm_k}): [{lo}:{hi}]")
print(f"  k_local = {k_local}")
print(f"  e_local = {e_local}")
coeffs = np.polyfit(k_local, e_local, 2)
d2E = 2.0 * coeffs[0]
print(f"  polyfit coeffs = {coeffs}")
print(f"  d²E/dk² = {d2E}")
d2E_SI = d2E * eV * (angstrom ** 2)
if abs(d2E_SI) > 1e-30:
    m_star = (hbar ** 2 / d2E_SI) / electron_mass
    print(f"  m* (electron) = {m_star:.4f} m_e")
else:
    print(f"  d2E_SI too small: {d2E_SI}")
