"""Structure generation and manipulation utilities.

Provides tools for creating seed structures, perturbing them,
and validating physical reasonableness.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.data import covalent_radii, atomic_numbers

logger = logging.getLogger(__name__)


def make_seed_structure(name: str) -> Atoms:
    """Create a known crystal structure by name.

    Args:
        name: One of the predefined structure names.

    Returns:
        ASE Atoms object.
    """
    structures = {
        "Si": lambda: bulk("Si", "diamond", a=5.431),
        "Ge": lambda: bulk("Ge", "diamond", a=5.658),
        "C-diamond": lambda: bulk("C", "diamond", a=3.567),
        "GaAs": lambda: bulk("GaAs", "zincblende", a=5.653),
        "AlAs": lambda: bulk("AlAs", "zincblende", a=5.661),
        "InAs": lambda: bulk("InAs", "zincblende", a=6.058),
        "GaP": lambda: bulk("GaP", "zincblende", a=5.451),
        "SiC-3C": lambda: bulk("SiC", "zincblende", a=4.360),
        "InP": lambda: bulk("InP", "zincblende", a=5.869),
        "AlN": lambda: bulk("AlN", "zincblende", a=4.380),
    }

    if name not in structures:
        available = ", ".join(structures.keys())
        raise ValueError(f"Unknown structure '{name}'. Available: {available}")

    atoms = structures[name]()
    logger.info(f"Created seed structure: {name} ({atoms.get_chemical_formula()})")
    return atoms


def perturb_positions(atoms: Atoms, amplitude: float = 0.02, rng: Optional[np.random.Generator] = None) -> Atoms:
    """Randomly perturb atomic positions in fractional coordinates.

    Args:
        atoms: Input structure.
        amplitude: Maximum fractional coordinate perturbation.
        rng: Random number generator (for reproducibility).

    Returns:
        New Atoms with perturbed positions.
    """
    if rng is None:
        rng = np.random.default_rng()

    new_atoms = atoms.copy()
    scaled = new_atoms.get_scaled_positions()
    perturbation = rng.uniform(-amplitude, amplitude, scaled.shape)
    scaled += perturbation
    # Wrap back into unit cell
    scaled = scaled % 1.0
    new_atoms.set_scaled_positions(scaled)
    return new_atoms


def perturb_lattice(atoms: Atoms, amplitude: float = 0.05, rng: Optional[np.random.Generator] = None) -> Atoms:
    """Randomly perturb lattice vectors.

    Args:
        atoms: Input structure.
        amplitude: Maximum fractional change to each lattice parameter.
        rng: Random number generator.

    Returns:
        New Atoms with perturbed cell.
    """
    if rng is None:
        rng = np.random.default_rng()

    new_atoms = atoms.copy()
    cell = new_atoms.get_cell()

    # Symmetric strain tensor (preserves lattice symmetry better)
    strain = rng.uniform(-amplitude, amplitude, (3, 3))
    strain = (strain + strain.T) / 2  # symmetrize
    np.fill_diagonal(strain, strain.diagonal() + 1.0)

    new_cell = cell @ strain
    new_atoms.set_cell(new_cell, scale_atoms=True)
    return new_atoms


def swap_species(atoms: Atoms, species_palette: list[str], rng: Optional[np.random.Generator] = None) -> Atoms:
    """Randomly swap one atom's species.

    Args:
        atoms: Input structure.
        species_palette: List of allowed element symbols.
        rng: Random number generator.

    Returns:
        New Atoms with one species swapped.
    """
    if rng is None:
        rng = np.random.default_rng()

    new_atoms = atoms.copy()
    symbols = list(new_atoms.get_chemical_symbols())

    # Pick a random atom
    idx = rng.integers(len(symbols))
    old = symbols[idx]

    # Pick a new species (different from current)
    candidates = [s for s in species_palette if s != old]
    if not candidates:
        return new_atoms

    new_species = rng.choice(candidates)
    symbols[idx] = new_species
    new_atoms.set_chemical_symbols(symbols)

    logger.debug(f"Swapped atom {idx}: {old} -> {new_species}")
    return new_atoms


def validate_structure(atoms: Atoms, min_distance_fraction: float = 0.5) -> bool:
    """Check if a structure is physically reasonable.

    Validates:
      - No overlapping atoms (interatomic distance > fraction of sum of covalent radii)
      - Cell volume is positive and reasonable
      - No NaN/inf in positions or cell

    Args:
        atoms: Structure to check.
        min_distance_fraction: Minimum allowed distance as fraction of
            sum of covalent radii.

    Returns:
        True if the structure passes all checks.
    """
    # Check for NaN/inf
    if np.any(~np.isfinite(atoms.positions)):
        logger.warning("Structure has non-finite positions")
        return False

    if np.any(~np.isfinite(atoms.cell)):
        logger.warning("Structure has non-finite cell")
        return False

    # Check volume
    vol = atoms.get_volume()
    if vol <= 0 or vol > 1e6:
        logger.warning(f"Unreasonable cell volume: {vol:.1f} A^3")
        return False

    # Volume per atom sanity check (typical range: 5-200 A^3/atom)
    vol_per_atom = vol / len(atoms)
    if vol_per_atom < 3.0 or vol_per_atom > 500.0:
        logger.warning(f"Unreasonable volume/atom: {vol_per_atom:.1f} A^3")
        return False

    # Check interatomic distances
    numbers = atoms.get_atomic_numbers()
    positions = atoms.get_positions()

    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            d = atoms.get_distance(i, j, mic=True)
            r_sum = covalent_radii[numbers[i]] + covalent_radii[numbers[j]]
            if d < min_distance_fraction * r_sum:
                logger.warning(
                    f"Atoms {i}-{j} too close: {d:.3f} A "
                    f"(min: {min_distance_fraction * r_sum:.3f} A)"
                )
                return False

    return True


def structure_to_fingerprint(atoms: Atoms, n_bins: int = 64) -> np.ndarray:
    """Create a fixed-length fingerprint of a crystal structure.

    Produces a composition-aware descriptor that encodes:
      1. Element composition (12 dims): fraction of each element in palette
      2. Element properties (4 dims): mean/std atomic number, mean/std covalent radius
      3. Lattice features (8 dims): volume/atom, lattice lengths, angles, density
      4. RDF histogram (n_bins dims): proper radial distribution function via neighbor lists
      5. Partial RDF (n_bins dims): element-pair-weighted RDF for chemical sensitivity

    Total output size: 2 * n_bins + 24

    Args:
        atoms: Crystal structure.
        n_bins: Number of bins in the RDF histogram.

    Returns:
        Fingerprint vector of length 2 * n_bins + 24.
    """
    from ase.neighborlist import neighbor_list as ase_neighbor_list

    features = []

    # ---- 1. Element composition (12 dims) ----
    palette = ["H", "C", "N", "O", "Si", "P", "Ge", "Ga", "As", "In", "Sn", "Al"]
    symbols = atoms.get_chemical_symbols()
    n_atoms = len(symbols)
    comp = np.zeros(len(palette), dtype=np.float32)
    for sym in symbols:
        if sym in palette:
            comp[palette.index(sym)] += 1.0 / n_atoms
    features.append(comp)

    # ---- 2. Element properties (4 dims) ----
    atomic_nums = atoms.get_atomic_numbers().astype(np.float32)
    cov_radii = np.array([covalent_radii[z] for z in atoms.get_atomic_numbers()],
                         dtype=np.float32)
    props = np.array([
        atomic_nums.mean(),
        atomic_nums.std() + 1e-6,
        cov_radii.mean(),
        cov_radii.std() + 1e-6,
    ], dtype=np.float32)
    # Normalize to reasonable range
    props[0] /= 50.0  # atomic number ~ [1, 100]
    props[1] /= 20.0
    props[2] /= 2.0   # cov radius ~ [0.3, 2.5] Å
    props[3] /= 1.0
    features.append(props)

    # ---- 3. Lattice features (8 dims) ----
    cell = atoms.cell
    vol = atoms.get_volume()
    lengths = cell.lengths()
    angles = cell.angles()
    lattice = np.array([
        vol / max(n_atoms, 1) / 30.0,  # vol/atom normalized (~20-40 Å³)
        lengths[0] / lengths[1] if lengths[1] > 0 else 1.0,  # a/b ratio
        lengths[1] / lengths[2] if lengths[2] > 0 else 1.0,  # b/c ratio
        lengths[0] / lengths[2] if lengths[2] > 0 else 1.0,  # a/c ratio
        angles[0] / 180.0,  # alpha normalized
        angles[1] / 180.0,  # beta normalized
        angles[2] / 180.0,  # gamma normalized
        sum(atoms.get_masses()) / vol if vol > 0 else 0.0,  # density (amu/Å³)
    ], dtype=np.float32)
    features.append(lattice)

    # ---- 4. Proper RDF via neighbor lists (n_bins dims) ----
    r_max = 8.0  # Angstrom
    bins = np.linspace(0, r_max, n_bins + 1)
    bin_width = bins[1] - bins[0]
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # Use ASE neighbor list — correctly enumerates ALL pairs within cutoff
    # including periodic images (typically ~50-200 pairs for r_max=8Å)
    i_idx, j_idx, d_arr = ase_neighbor_list('ijd', atoms, cutoff=r_max)

    if len(d_arr) > 0:
        # Proper RDF normalization: g(r) = hist / (n_atoms * 4π r² dr * ρ)
        rho = n_atoms / vol  # number density
        hist_raw, _ = np.histogram(d_arr, bins=bins)
        # Each pair (i,j) is counted once by neighbor_list; the shell volume
        # normalization converts raw counts to g(r)
        shell_vol = 4.0 * np.pi * bin_centers**2 * bin_width
        shell_vol = np.where(shell_vol > 0, shell_vol, 1.0)
        rdf = hist_raw.astype(np.float32) / (n_atoms * rho * shell_vol)
    else:
        rdf = np.zeros(n_bins, dtype=np.float32)
    features.append(rdf)

    # ---- 5. Partial RDF: electronegativity-weighted (n_bins dims) ----
    # Weight each pair distance by |Z_i - Z_j| to encode chemical contrast
    if len(d_arr) > 0:
        z = atoms.get_atomic_numbers()
        weights = np.abs(z[i_idx].astype(np.float32) - z[j_idx].astype(np.float32))
        weights /= max(weights.max(), 1.0)  # normalize to [0, 1]
        partial_hist, _ = np.histogram(d_arr, bins=bins, weights=weights)
        partial_rdf = partial_hist.astype(np.float32) / (n_atoms * rho * shell_vol)
    else:
        partial_rdf = np.zeros(n_bins, dtype=np.float32)
    features.append(partial_rdf)

    return np.concatenate(features)
