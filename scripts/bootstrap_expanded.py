#!/usr/bin/env python3
"""Expanded bootstrap DFT data collection with multiprocessing.

Generates ~200 structures per round using wider perturbation amplitudes,
species swaps, and parallel QE execution via multiprocessing.Pool.

Usage (inside Docker):
    python -m scripts.bootstrap_expanded --rounds 4 --workers 8 --seed 2024
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
import time
from multiprocessing import Pool, current_process
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(processName)s]: %(message)s",
)
logger = logging.getLogger(__name__)

# Species palette for swaps
SPECIES_PALETTE = [
    "Si", "Ge", "C", "Sn", "N", "P", "As", "Ga", "In", "Al",
    "Sb", "Bi", "Se", "Te",
]

ALL_SEEDS = [
    "Si", "Ge", "C-diamond", "GaAs", "AlAs",
    "InAs", "GaP", "SiC-3C", "InP", "AlN",
    "InSb", "GaSb",
    "GaAs-4", "InAs-4", "InSb-4", "GaSb-4", "Si-4", "Ge-4",
]

# Perturbation schedule: (label, pos_amplitude, lat_amplitude, n_species_swaps)
PERTURBATION_SCHEDULE = [
    ("small",  0.01, 0.02, 0),
    ("medium", 0.03, 0.04, 0),
    ("large",  0.06, 0.06, 0),
    ("swap1",  0.02, 0.02, 1),
    ("swap2",  0.02, 0.02, 2),
]

N_VARIANTS_PER_LEVEL = 4  # 5 levels × 4 variants × 10 seeds = 200 per round


def compute_one_structure(args):
    """Worker function: run a single DFT calculation and return results."""
    seed_name, label, atoms_dict, pseudo_dir, np_procs = args

    # Imports inside worker so multiprocessing fork works cleanly
    from ase import Atoms

    from qe_interface.calculator import QECalculator, QEConfig
    from qe_interface.properties import analyze_bands
    from qe_interface.structures import structure_to_fingerprint

    # Reconstruct Atoms from serialisable dict
    atoms = Atoms(
        symbols=atoms_dict["symbols"],
        positions=atoms_dict["positions"],
        cell=atoms_dict["cell"],
        pbc=True,
    )

    tag = f"{seed_name}/{label}"
    scratch = tempfile.mkdtemp(prefix=f"qe_exp_{current_process().name}_")

    config = QEConfig(
        pw_command=f"mpirun --allow-run-as-root -np {np_procs} pw.x",
        pseudo_dir=pseudo_dir,
        scratch_dir=scratch,
        ecutwfc=30.0,
        ecutrho=240.0,
        conv_thr=1e-6,
        kpoints=(4, 4, 4),
    )
    qe = QECalculator(config)

    t0 = time.time()
    try:
        result = qe.run_bands(atoms, npoints=20)
    except Exception as e:
        dt = time.time() - t0
        logger.error(f"  {tag}: bands failed ({dt:.0f}s): {e}")
        return None

    dt = time.time() - t0

    if not result.converged or result.band_energies is None:
        logger.warning(f"  {tag}: not converged or no bands ({dt:.0f}s)")
        return None

    n_electrons = qe.get_n_valence_electrons(atoms)
    props = analyze_bands(
        result.kpoints,
        result.band_energies,
        n_electrons,
        atoms.cell[:],
    )

    if props.min_effective_mass is None or not np.isfinite(props.min_effective_mass):
        logger.warning(f"  {tag}: no valid effective mass ({dt:.0f}s)")
        return None

    fp = structure_to_fingerprint(atoms)

    entry = {
        "seed": seed_name,
        "label": label,
        "formula": atoms.get_chemical_formula(),
        "band_gap": props.band_gap,
        "is_direct": props.is_direct,
        "vbm": props.vbm,
        "cbm": props.cbm,
        "m_electron": props.effective_mass_electron,
        "m_hole": props.effective_mass_hole,
        "m_min": props.min_effective_mass,
        "time_s": dt,
        "fingerprint": fp.tolist(),
    }

    logger.info(
        f"  {tag}: gap={props.band_gap:.3f} eV, "
        f"m*={props.min_effective_mass:.4f}, {dt:.0f}s"
    )
    return entry


def generate_work_items(round_idx, base_seed, pseudo_dir, np_procs):
    """Build a list of (seed, label, atoms_dict, ...) work items for one round."""
    from qe_interface.structures import (
        make_seed_structure,
        perturb_lattice,
        perturb_positions,
        swap_species,
        validate_structure,
    )

    rng = np.random.default_rng(base_seed + round_idx * 10_000)
    items = []
    skipped = 0

    for seed_name in ALL_SEEDS:
        try:
            base = make_seed_structure(seed_name)
        except Exception as e:
            logger.error(f"Could not create seed '{seed_name}': {e}")
            continue

        for level_label, pos_amp, lat_amp, n_swaps in PERTURBATION_SCHEDULE:
            for j in range(N_VARIANTS_PER_LEVEL):
                s = base.copy()

                # Apply species swaps first (if any)
                for _ in range(n_swaps):
                    s = swap_species(s, SPECIES_PALETTE, rng=rng)

                # Apply geometric perturbations
                s = perturb_positions(s, amplitude=pos_amp, rng=rng)
                s = perturb_lattice(s, amplitude=lat_amp, rng=rng)

                # Validate structure before queueing expensive DFT
                if not validate_structure(s):
                    logger.debug(
                        f"  {seed_name}/{level_label}_{j}: invalid, skipping"
                    )
                    skipped += 1
                    continue

                label = f"r{round_idx}_{level_label}_{j}"

                # Serialise Atoms for pickling across processes
                atoms_dict = {
                    "symbols": s.get_chemical_symbols(),
                    "positions": s.positions.tolist(),
                    "cell": s.cell.tolist(),
                }
                items.append(
                    (seed_name, label, atoms_dict, pseudo_dir, np_procs)
                )

    logger.info(
        f"Round {round_idx}: generated {len(items)} work items "
        f"({skipped} skipped as invalid)"
    )
    return items


def main():
    parser = argparse.ArgumentParser(
        description="Expanded bootstrap DFT data collection"
    )
    parser.add_argument(
        "--rounds", type=int, default=1,
        help="Number of rounds (each ~200 structures)",
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Parallel worker processes",
    )
    parser.add_argument(
        "--np-procs", type=int, default=2,
        help="MPI processes per QE run",
    )
    parser.add_argument(
        "--seed", type=int, default=2024,
        help="Base RNG seed",
    )
    parser.add_argument(
        "--pseudo-dir", type=str,
        default="/workspace/pseudopotentials",
        help="Pseudopotential directory",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="data/bootstrap",
        help="Output directory for results",
    )
    args = parser.parse_args()

    # Import here so arg parsing works even without torch installed
    from models.surrogate import SurrogatePredictor

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load existing data if present ----
    surrogate = SurrogatePredictor(input_dim=84, hidden_dim=128, n_layers=3)
    existing_path = out_dir / "surrogate_bootstrap"
    if (existing_path / "surrogate_weights.pt").exists():
        data = np.load(existing_path / "surrogate_data.npz")
        for fp, tgt in zip(data["fingerprints"], data["targets"]):
            surrogate.add_data(fp, float(tgt))
        logger.info(f"Loaded {surrogate.dataset_size} existing data points")
    else:
        logger.info("Starting from scratch (no existing bootstrap)")

    all_results: list[dict] = []
    total_start = time.time()
    n_success = 0
    n_fail = 0

    for round_idx in range(args.rounds):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"ROUND {round_idx + 1}/{args.rounds}")
        logger.info(f"{'=' * 60}")

        items = generate_work_items(
            round_idx, args.seed, args.pseudo_dir, args.np_procs
        )

        round_start = time.time()

        with Pool(processes=args.workers) as pool:
            results = pool.map(compute_one_structure, items)

        round_time = time.time() - round_start
        round_success = 0

        for r in results:
            if r is not None:
                all_results.append(r)
                fp = np.array(r["fingerprint"], dtype=np.float32)
                surrogate.add_data(fp, r["m_min"])
                round_success += 1
                n_success += 1
            else:
                n_fail += 1

        logger.info(
            f"Round {round_idx + 1}: {round_success}/{len(items)} "
            f"succeeded in {round_time:.0f}s ({round_time / 60:.1f} min)"
        )

        # Save per-round checkpoint
        round_file = out_dir / f"bootstrap_expanded_round{round_idx}.json"
        with open(round_file, "w") as f:
            json.dump(
                [r for r in results if r is not None], f, indent=2, default=str
            )
        logger.info(f"Round {round_idx + 1} saved to {round_file}")

    total_time = time.time() - total_start

    # ---- Summary ----
    logger.info(f"\n{'=' * 60}")
    logger.info("EXPANDED BOOTSTRAP SUMMARY")
    logger.info(f"{'=' * 60}")
    logger.info(f"Total successful: {n_success}/{n_success + n_fail}")
    logger.info(f"Total failed:     {n_fail}/{n_success + n_fail}")
    logger.info(f"Total time:       {total_time:.0f}s ({total_time / 60:.1f} min)")
    logger.info(f"Surrogate dataset: {surrogate.dataset_size} points (incl. prior)")

    # ---- Train surrogate on full dataset ----
    if surrogate.dataset_size >= 10:
        logger.info("\nTraining surrogate on full dataset...")
        surrogate.train(epochs=1000, verbose=True)
        surrogate.save(str(out_dir / "surrogate_bootstrap"))
        logger.info(f"Surrogate saved to {out_dir}/surrogate_bootstrap")

    # ---- Save combined results ----
    combined_file = out_dir / "bootstrap_expanded_all.json"
    with open(combined_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"All results saved to {combined_file}")
    logger.info(f"Final dataset size: {surrogate.dataset_size} points")


if __name__ == "__main__":
    main()
