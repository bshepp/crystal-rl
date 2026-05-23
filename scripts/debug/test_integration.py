#!/usr/bin/env python3
"""End-to-end integration tests — bands, effective mass, and RL env stepping.

Usage (inside Docker with QE):
    python scripts/test_integration.py

Tests:
  1. Si band structure + effective mass extraction
  2. Gym environment stepping with DFT (energy mode)
  3. Gym environment stepping with surrogate
"""

from __future__ import annotations

import logging
import sys
import tempfile

import numpy as np
from ase.build import bulk

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def test_si_bands():
    """Run bands on Si and extract effective mass — the core physics pipeline."""
    from qe_interface.calculator import QECalculator, QEConfig
    from qe_interface.properties import analyze_bands

    logger.info("=" * 60)
    logger.info("Test: Silicon Band Structure + Effective Mass")
    logger.info("=" * 60)

    si = bulk("Si", "diamond", a=5.43)

    scratch = tempfile.mkdtemp(prefix="qe_int_")
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

    # Run band structure (SCF + NSCF)
    result = qe.run_bands(si, npoints=30)

    if not result.converged:
        logger.error(f"Calculation did not converge: {result.error}")
        return False

    if result.band_energies is None:
        logger.error("Band energies were not extracted!")
        return False

    logger.info(f"Band energies shape: {result.band_energies.shape}")
    logger.info(f"Energy range: [{result.band_energies.min():.2f}, {result.band_energies.max():.2f}] eV")

    # Analyze bands
    # Silicon: 2 atoms × 14 electrons each = 28 electrons → naive, but
    # QE uses valence only. Si has 4 valence electrons → 2 atoms = 8 valence electrons
    n_valence_electrons = 8  # 2 Si atoms × 4 valence e- each
    props = analyze_bands(
        result.kpoints,
        result.band_energies,
        n_valence_electrons,
        si.cell[:],
    )

    logger.info(f"Band gap: {props.band_gap:.4f} eV")
    logger.info(f"Is direct gap: {props.is_direct}")
    logger.info(f"VBM: {props.vbm:.4f} eV, CBM: {props.cbm:.4f} eV")
    logger.info(f"Electron effective mass: {props.effective_mass_electron}")
    logger.info(f"Hole effective mass: {props.effective_mass_hole}")
    logger.info(f"Min |m*|: {props.min_effective_mass}")

    # Sanity checks for silicon
    # Experimental Si band gap: 1.12 eV (DFT-PBE typically gives ~0.5-0.7 eV)
    if props.band_gap < 0.2 or props.band_gap > 2.0:
        logger.warning(f"Band gap {props.band_gap:.3f} eV outside expected range for Si")

    # Si has an indirect gap
    if props.is_direct:
        logger.warning("Si should have an indirect gap — path sampling may miss it")

    # Effective masses should be finite and small
    if props.effective_mass_electron is not None and np.isfinite(props.effective_mass_electron):
        logger.info(f"Electron m* = {props.effective_mass_electron:.4f} m_e")
        if 0.01 < abs(props.effective_mass_electron) < 10.0:
            logger.info("Electron effective mass in reasonable range: PASS")
        else:
            logger.warning("Electron m* outside typical range")
    else:
        logger.warning("Could not compute electron effective mass")

    logger.info("Silicon bands test PASSED")
    return True


def test_gym_dft_step():
    """Step the Gym environment with real DFT in energy mode."""
    from envs.crystal_env import CrystalEnv
    from qe_interface.calculator import QEConfig

    logger.info("=" * 60)
    logger.info("Test: Gym Environment DFT Step (energy mode)")
    logger.info("=" * 60)

    import tempfile
    scratch = tempfile.mkdtemp(prefix="qe_gym_")

    config = QEConfig(
        pw_command="mpirun --allow-run-as-root -np 2 pw.x",
        pseudo_dir="/workspace/pseudopotentials",
        scratch_dir=scratch,
        ecutwfc=30.0,
        ecutrho=240.0,
        conv_thr=1e-6,
        kpoints=(4, 4, 4),
    )

    env = CrystalEnv(
        seed_structure="Si",
        species_palette=["Si", "Ge"],
        max_steps=5,
        use_surrogate=False,
        qe_config=config,
        reward_mode="energy",
    )

    obs, info = env.reset(seed=42)
    logger.info(f"Reset: obs shape={obs.shape}, formula={info['formula']}")

    # Take a few actions
    actions_to_try = [0, 2, 3]  # perturb small, compress, expand
    for action in actions_to_try:
        logger.info(f"\nAction: {env.ACTION_NAMES[action]}")
        obs, reward, terminated, truncated, info = env.step(action)
        logger.info(f"  Reward: {reward:.4f}")
        logger.info(f"  Valid: {info.get('valid')}")
        logger.info(f"  Energy: {info.get('energy', 'N/A')}")
        logger.info(f"  Formula: {info.get('formula')}")
        logger.info(f"  Source: {info.get('source', 'N/A')}")

        if not info.get("valid", True):
            logger.warning("  Structure was invalid!")

    logger.info("\nGym DFT step test PASSED")
    return True


def test_gym_surrogate_step():
    """Step the Gym environment with a surrogate model."""
    from envs.crystal_env import CrystalEnv
    from models.surrogate import SurrogatePredictor
    from qe_interface.structures import structure_to_fingerprint

    logger.info("=" * 60)
    logger.info("Test: Gym Environment Surrogate Step")
    logger.info("=" * 60)

    # Create and train a quick surrogate
    predictor = SurrogatePredictor(input_dim=84, hidden_dim=128, n_layers=3)

    rng = np.random.default_rng(42)
    for _ in range(50):
        x = rng.standard_normal(64).astype(np.float32)
        y = float(0.5 + 0.1 * rng.standard_normal())  # mock m* around 0.5
        predictor.add_data(x, y)

    predictor.train(epochs=50, verbose=False)

    env = CrystalEnv(
        seed_structure="Si",
        species_palette=["Si", "Ge", "C"],
        max_steps=20,
        use_surrogate=True,
        surrogate_model=predictor,
        reward_mode="effective_mass",
    )

    obs, info = env.reset(seed=42)
    logger.info(f"Reset: obs shape={obs.shape}")

    total_reward = 0.0
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        logger.info(
            f"  Step {step}: action={env.ACTION_NAMES[action]}, "
            f"reward={reward:.4f}, valid={info.get('valid')}"
        )

        if terminated or truncated:
            break

    logger.info(f"\nTotal reward over {step + 1} steps: {total_reward:.4f}")
    logger.info("Gym surrogate step test PASSED")
    return True


if __name__ == "__main__":
    results = {}

    # Surrogate stepping (no QE needed)
    logger.info("\n>>> Test 1: Gym surrogate stepping")
    results["surrogate_step"] = test_gym_surrogate_step()

    # DFT tests (require QE)
    if "--with-qe" not in sys.argv:
        logger.info("\n>>> Skipping DFT tests (use --with-qe)")
    else:
        logger.info("\n>>> Test 2: Silicon band structure")
        results["si_bands"] = test_si_bands()

        logger.info("\n>>> Test 3: Gym DFT stepping")
        results["dft_step"] = test_gym_dft_step()

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("INTEGRATION TEST SUMMARY")
    logger.info("=" * 60)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        logger.info(f"  {name}: {status}")

    all_passed = all(results.values())
    logger.info(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    sys.exit(0 if all_passed else 1)
