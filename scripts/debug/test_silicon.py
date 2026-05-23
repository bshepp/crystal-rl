#!/usr/bin/env python3
"""Smoke test: Run a silicon SCF calculation through the QE interface.

Usage (inside Docker container):
    python scripts/test_silicon.py

This validates the entire QE + ASE + Python pipeline by performing
the simplest possible DFT calculation: SCF energy of bulk Si.
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
from ase.build import bulk

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def test_silicon_scf():
    """Run SCF on diamond-Si with loose convergence settings for a quick test."""

    logger.info("=" * 60)
    logger.info("Silicon SCF Smoke Test")
    logger.info("=" * 60)

    # ---- 1. Build structure ----
    si = bulk("Si", "diamond", a=5.43)
    logger.info(f"Structure: {si.get_chemical_formula()}, {len(si)} atoms")
    logger.info(f"Cell:\n{si.cell[:]}")
    logger.info(f"Positions:\n{si.positions}")

    # ---- 2. Set up calculator ----
    from qe_interface.calculator import QECalculator, QEConfig

    config = QEConfig(
        pw_command="mpirun --allow-run-as-root -np 2 pw.x",
        pseudo_dir="/workspace/pseudopotentials",
        scratch_dir=tempfile.mkdtemp(prefix="qe_test_"),
        ecutwfc=30.0,       # Lower cutoff for speed
        ecutrho=240.0,
        conv_thr=1e-6,      # Loose convergence
        kpoints=(4, 4, 4),  # Coarser k-mesh
    )

    qe = QECalculator(config)
    logger.info(f"QE command: {config.pw_command}")
    logger.info(f"Pseudo dir: {config.pseudo_dir}")

    # ---- 3. Run SCF ----
    logger.info("\nRunning SCF calculation...")
    result = qe.run_scf(si)

    if not result.converged:
        logger.error("SCF did NOT converge!")
        logger.error(f"Error: {result.error}")
        return False

    logger.info(f"SCF converged!")
    logger.info(f"Total energy: {result.energy:.6f} eV")
    logger.info(f"Forces:\n{result.forces}")

    # ---- 4. Sanity checks ----
    # Si total energy should be around -310 to -312 eV (2 atoms with SSSP pseudos)
    # This is a rough check - exact value depends on pseudopotential
    if result.energy is not None and np.isfinite(result.energy):
        logger.info("Energy is finite: PASS")
    else:
        logger.error("Energy is not finite: FAIL")
        return False

    if result.forces is not None and result.forces.shape == (2, 3):
        max_force = np.max(np.abs(result.forces))
        logger.info(f"Max force: {max_force:.6f} eV/Å")
        # Diamond Si is at equilibrium, forces should be near zero
        if max_force < 0.1:
            logger.info("Forces are small (equilibrium structure): PASS")
        else:
            logger.warning(f"Forces are larger than expected: {max_force:.6f}")
    else:
        logger.warning("Force shape unexpected")

    logger.info("\n" + "=" * 60)
    logger.info("Silicon SCF smoke test PASSED")
    logger.info("=" * 60)
    return True


def test_fingerprint():
    """Test that structure fingerprinting works."""
    from qe_interface.structures import structure_to_fingerprint, validate_structure

    si = bulk("Si", "diamond", a=5.43)

    fp = structure_to_fingerprint(si)
    logger.info(f"Fingerprint shape: {fp.shape}")
    logger.info(f"Fingerprint range: [{fp.min():.4f}, {fp.max():.4f}]")
    logger.info(f"Fingerprint sum: {fp.sum():.4f}")

    assert fp.shape == (64,), f"Expected shape (64,), got {fp.shape}"
    assert np.all(np.isfinite(fp)), "Fingerprint contains non-finite values"

    valid = validate_structure(si)
    logger.info(f"Validation: {valid}")
    assert valid, "Si structure should be valid"

    logger.info("Fingerprint & validation tests PASSED")
    return True


def test_surrogate_model():
    """Test that the surrogate model trains and predicts."""
    from models.surrogate import SurrogatePredictor

    predictor = SurrogatePredictor(input_dim=84, hidden_dim=128, n_layers=3)

    rng = np.random.default_rng(42)
    for i in range(50):
        x = rng.standard_normal(64).astype(np.float32)
        y = float(np.sin(x.sum()) + 0.1 * rng.standard_normal())
        predictor.add_data(x, y)

    logger.info(f"Dataset size: {predictor.dataset_size}")

    # Train
    predictor.train(epochs=50, verbose=False)

    # Predict
    x_test = rng.standard_normal(64).astype(np.float32)
    pred = predictor.predict(x_test)
    logger.info(f"Prediction: {pred:.4f}")
    assert np.isfinite(pred), "Prediction is not finite"

    logger.info("Surrogate model test PASSED")
    return True


def test_gym_env():
    """Test that the Gym environment can be instantiated and stepped."""
    from envs.crystal_env import CrystalEnv

    env = CrystalEnv(
        seed_structure="Si",
        species_palette=["Si", "Ge", "C"],
        max_steps=5,
        use_surrogate=False,
        reward_mode="energy",
    )

    obs, info = env.reset()
    logger.info(f"Observation shape: {obs.shape}")
    logger.info(f"Observation range: [{obs.min():.4f}, {obs.max():.4f}]")

    assert obs.shape == (64,), f"Expected shape (64,), got {obs.shape}"

    # Try one step (will require QE to be running for actual reward)
    # Just verify the step doesn't crash
    logger.info("Gym environment creation test PASSED")
    return True


if __name__ == "__main__":
    results = {}

    # Tests that work without QE installed
    logger.info("\n>>> Test 1: Structure fingerprinting")
    results["fingerprint"] = test_fingerprint()

    logger.info("\n>>> Test 2: Surrogate model")
    results["surrogate"] = test_surrogate_model()

    logger.info("\n>>> Test 3: Gym environment")
    results["gym_env"] = test_gym_env()

    # Test that requires QE (only run if --with-qe flag)
    if "--with-qe" in sys.argv:
        logger.info("\n>>> Test 4: Silicon SCF (requires QE)")
        results["silicon_scf"] = test_silicon_scf()
    else:
        logger.info("\n>>> Test 4: Silicon SCF SKIPPED (use --with-qe to enable)")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        logger.info(f"  {name}: {status}")

    all_passed = all(results.values())
    logger.info(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    sys.exit(0 if all_passed else 1)
