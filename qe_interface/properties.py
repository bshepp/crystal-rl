"""Property extraction from QE results.

Computes derived quantities like effective mass and band gap
from raw DFT band structure output.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from ase import Atoms

logger = logging.getLogger(__name__)


@dataclass
class BandProperties:
    """Extracted band structure properties."""

    band_gap: float  # eV (0 = metallic)
    is_direct: bool
    vbm: float  # Valence band maximum (eV)
    cbm: float  # Conduction band minimum (eV)
    effective_mass_electron: Optional[float] = None  # in units of m_e
    effective_mass_hole: Optional[float] = None  # in units of m_e
    min_effective_mass: Optional[float] = None  # most extreme curvature found


def compute_effective_mass(
    kpoints: np.ndarray,
    energies: np.ndarray,
    band_index: int,
    lattice_vectors: np.ndarray,
    kpoint_index: Optional[int] = None,
) -> float:
    """Compute effective mass from band curvature via polynomial fitting.

    Fits a quadratic E(k) = a·k² + b·k + c to a local window around
    the target k-point, then derives m* = ℏ² / (2a).

    Args:
        kpoints: k-point coordinates in crystal units, shape (nk, 3).
        energies: Band energies in eV, shape (nk, nbands).
        band_index: Which band to analyze.
        lattice_vectors: Real-space lattice vectors (3x3) in Angstrom.
        kpoint_index: If given, compute at this k-point. Otherwise, find
                      the extremum with maximum |curvature|.

    Returns:
        Effective mass in units of electron mass (m_e).
        Negative values indicate inverted curvature (holes).
    """
    from scipy.constants import electron_mass, eV, hbar, angstrom

    band = energies[:, band_index]  # shape (nk,)
    nk = len(band)

    if nk < 3:
        return float("inf")

    # Convert k-points from crystal to Cartesian (1/Angstrom)
    recip = 2 * np.pi * np.linalg.inv(lattice_vectors).T
    k_cart = kpoints @ recip  # shape (nk, 3)

    # Cumulative distance along the k-path
    dk = np.linalg.norm(np.diff(k_cart, axis=0), axis=1)
    k_dist = np.concatenate([[0], np.cumsum(dk)])

    def _fit_curvature(center_idx: int, half_window: int = 3) -> float:
        """Fit quadratic to a window around center_idx, return d²E/dk²."""
        lo = max(0, center_idx - half_window)
        hi = min(nk, center_idx + half_window + 1)
        if hi - lo < 3:
            return 0.0

        k_local = k_dist[lo:hi] - k_dist[center_idx]
        e_local = band[lo:hi]

        # Fit quadratic: E = a*k² + b*k + c  →  d²E/dk² = 2a
        coeffs = np.polyfit(k_local, e_local, 2)
        return 2.0 * coeffs[0]  # 2a

    # Unit conversions: E in eV, k in 1/Å → SI
    eV_to_J = eV
    A_to_m = angstrom

    if kpoint_index is not None:
        d2E = _fit_curvature(kpoint_index)
        d2E_SI = d2E * eV_to_J * (A_to_m ** 2)
        if abs(d2E_SI) < 1e-50:
            return float("inf")
        m_star = hbar ** 2 / d2E_SI
        return m_star / electron_mass

    # No specific k-point: find the largest |curvature| along the path
    best_m = float("inf")
    for i in range(nk):
        d2E = _fit_curvature(i)
        d2E_SI = d2E * eV_to_J * (A_to_m ** 2)
        if abs(d2E_SI) < 1e-50:
            continue
        m = (hbar ** 2 / d2E_SI) / electron_mass
        if abs(m) < abs(best_m):
            best_m = m
    return best_m


def analyze_bands(
    kpoints: np.ndarray,
    energies: np.ndarray,
    n_electrons: int,
    lattice_vectors: np.ndarray,
) -> BandProperties:
    """Full band structure analysis.

    Args:
        kpoints: k-points in crystal coordinates, shape (nk, 3).
        energies: Band energies in eV, shape (nk, nbands).
        n_electrons: Number of electrons (for finding Fermi level).
        lattice_vectors: Real-space lattice, shape (3, 3) in Angstrom.

    Returns:
        BandProperties with gap, effective masses, etc.
    """
    n_occupied = n_electrons // 2  # assuming non-spin-polarized

    if energies.shape[1] <= n_occupied:
        logger.warning("Not enough bands to determine gap")
        return BandProperties(band_gap=0.0, is_direct=False, vbm=0.0, cbm=0.0)

    valence = energies[:, n_occupied - 1]  # highest occupied band
    conduction = energies[:, n_occupied]  # lowest unoccupied band

    vbm = np.max(valence)
    cbm = np.min(conduction)
    band_gap = cbm - vbm

    vbm_k = np.argmax(valence)
    cbm_k = np.argmin(conduction)
    is_direct = vbm_k == cbm_k

    props = BandProperties(
        band_gap=max(0.0, band_gap),
        is_direct=is_direct,
        vbm=vbm,
        cbm=cbm,
    )

    # Compute effective masses if we have enough k-points
    if len(kpoints) >= 5:
        try:
            props.effective_mass_hole = compute_effective_mass(
                kpoints, energies, n_occupied - 1, lattice_vectors, kpoint_index=vbm_k
            )
        except Exception as e:
            logger.warning(f"Could not compute hole effective mass: {e}")

        try:
            props.effective_mass_electron = compute_effective_mass(
                kpoints, energies, n_occupied, lattice_vectors, kpoint_index=cbm_k
            )
        except Exception as e:
            logger.warning(f"Could not compute electron effective mass: {e}")

        # Find the most extreme curvature across all bands near the Fermi level
        try:
            extreme = float("inf")
            for bi in range(max(0, n_occupied - 2), min(energies.shape[1], n_occupied + 2)):
                m = compute_effective_mass(kpoints, energies, bi, lattice_vectors)
                if abs(m) < abs(extreme):
                    extreme = m
            props.min_effective_mass = extreme
        except Exception as e:
            logger.warning(f"Could not compute extreme effective mass: {e}")

    return props


def compute_formation_energy(
    total_energy: float,
    atoms: Atoms,
    reference_energies: dict[str, float],
) -> float:
    """Compute formation energy per atom.

    Args:
        total_energy: DFT total energy in eV.
        atoms: The structure.
        reference_energies: Energy per atom of each element in its
            standard reference state (eV/atom).

    Returns:
        Formation energy in eV/atom.
    """
    symbols = atoms.get_chemical_symbols()
    ref_sum = sum(reference_energies.get(s, 0.0) for s in symbols)
    return (total_energy - ref_sum) / len(atoms)
