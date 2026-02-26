#!/usr/bin/env python3
"""Full bootstrap DFT data collection — all seed structures + perturbations.

Covers all 10 available seed structures with 2 perturbation variants each,
totaling ~30 structures for a robust surrogate training set.

Usage (inside Docker):
    python -m scripts.bootstrap_full
"""

from __future__ import annotations

import json
import logging
import tempfile
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    from qe_interface.calculator import QECalculator, QEConfig
    from qe_interface.properties import analyze_bands
    from qe_interface.structures import (
        make_seed_structure,
        perturb_lattice,
        perturb_positions,
        swap_species,
        structure_to_fingerprint,
    )
    from models.surrogate import SurrogatePredictor

    scratch = tempfile.mkdtemp(prefix="qe_bootstrap_full_")
    config = QEConfig(
        pw_command="mpirun --allow-run-as-root -np 2 pw.x",
        pseudo_dir="/workspace/pseudopotentials",
        scratch_dir=scratch,
        ecutwfc=30.0,
        ecutrho=240.0,
        conv_thr=1e-6,
        kpoints=(4, 4, 4),
    )
    qe = QECalculator(config)

    # All available seeds — 10 base structures + 2 perturbations each = 30 total
    all_seeds = ["Si", "Ge", "C-diamond", "GaAs", "AlAs", "InAs",
                 "GaP", "SiC-3C", "InP", "AlN"]
    n_perturbations = 2
    rng = np.random.default_rng(2024)

    # Load existing surrogate data if available, to extend it
    surrogate = SurrogatePredictor(input_dim=84, hidden_dim=128, n_layers=3)
    existing_path = Path("data/bootstrap/surrogate_bootstrap")
    if (existing_path / "surrogate_weights.pt").exists():
        # Load just the data, we'll retrain on the full set
        data = np.load(existing_path / "surrogate_data.npz")
        for fp, tgt in zip(data["fingerprints"], data["targets"]):
            surrogate.add_data(fp, float(tgt))
        logger.info(f"Loaded {surrogate.dataset_size} existing data points")
    else:
        logger.info("Starting from scratch (no existing bootstrap)")

    results_log = []
    total_start = time.time()
    n_success = 0
    n_fail = 0

    for name in all_seeds:
        logger.info(f"\n{'='*60}")
        logger.info(f"Seed: {name}")
        logger.info(f"{'='*60}")

        try:
            base = make_seed_structure(name)
        except Exception as e:
            logger.error(f"  Could not create seed '{name}': {e}")
            n_fail += 1
            continue

        # Build list: base + perturbations
        structures = [("base", base)]
        for j in range(n_perturbations):
            s = base.copy()
            s = perturb_positions(s, amplitude=0.02, rng=rng)
            s = perturb_lattice(s, amplitude=0.03, rng=rng)
            structures.append((f"perturb_{j}", s))

        for label, atoms in structures:
            tag = f"{name}/{label}"
            t0 = time.time()
            logger.info(f"\n--- {tag}: {atoms.get_chemical_formula()} ---")

            try:
                result = qe.run_bands(atoms, npoints=20)
            except Exception as e:
                logger.error(f"  {tag}: bands failed: {e}")
                n_fail += 1
                continue

            dt = time.time() - t0

            if not result.converged or result.band_energies is None:
                logger.warning(f"  {tag}: not converged or no bands ({dt:.0f}s)")
                n_fail += 1
                continue

            n_electrons = qe.get_n_valence_electrons(atoms)
            props = analyze_bands(
                result.kpoints,
                result.band_energies,
                n_electrons,
                atoms.cell[:],
            )

            entry = {
                "seed": name,
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
            }
            results_log.append(entry)

            m_str = (f"{props.min_effective_mass:.4f}"
                     if props.min_effective_mass and np.isfinite(props.min_effective_mass)
                     else "N/A")
            logger.info(f"  {tag}: gap={props.band_gap:.3f} eV, m*_min={m_str}, {dt:.0f}s")

            # Feed into surrogate
            if props.min_effective_mass is not None and np.isfinite(props.min_effective_mass):
                fp = structure_to_fingerprint(atoms)
                surrogate.add_data(fp, props.min_effective_mass)
                n_success += 1
            else:
                n_fail += 1

    total_time = time.time() - total_start

    logger.info(f"\n{'='*60}")
    logger.info("FULL BOOTSTRAP SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Successful: {n_success}/{n_success + n_fail}")
    logger.info(f"Failed: {n_fail}/{n_success + n_fail}")
    logger.info(f"Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    logger.info(f"Surrogate dataset: {surrogate.dataset_size} points (incl. prior)")

    # Train the surrogate
    if surrogate.dataset_size >= 5:
        logger.info("\nTraining surrogate model on full dataset...")
        surrogate.train(epochs=500, verbose=True)

        out_dir = Path("data/bootstrap")
        out_dir.mkdir(parents=True, exist_ok=True)
        surrogate.save(str(out_dir / "surrogate_bootstrap"))
        logger.info(f"Surrogate saved to {out_dir}/surrogate_bootstrap")

        with open(out_dir / "bootstrap_full_results.json", "w") as f:
            json.dump(results_log, f, indent=2, default=str)
        logger.info(f"Results log saved to {out_dir}/bootstrap_full_results.json")

    # Summary table
    logger.info("\n{:>15s} {:>8s} {:>8s} {:>6s} {:>10s} {:>10s} {:>5s}".format(
        "Structure", "Gap(eV)", "VBM", "Dir?", "m*_e", "m*_h", "Time"
    ))
    for r in results_log:
        me = f"{r['m_electron']:.3f}" if r["m_electron"] and np.isfinite(r["m_electron"]) else "N/A"
        mh = f"{r['m_hole']:.3f}" if r["m_hole"] and np.isfinite(r["m_hole"]) else "N/A"
        tag = f"{r['seed']}/{r['label']}"
        logger.info(
            f"  {tag:>15s} "
            f"{r['band_gap']:8.3f} {r['vbm']:8.3f} "
            f"{'Y' if r['is_direct'] else 'N':>6s} "
            f"{me:>10s} {mh:>10s} "
            f"{r['time_s']:5.0f}s"
        )


if __name__ == "__main__":
    main()
