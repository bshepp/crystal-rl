#!/usr/bin/env python3
"""DFT validation of RL agent's best discovered structures.

Loads the trained PPO agent, generates structures via deterministic rollouts,
then validates the top candidates through real QE band structure calculations.
Compares surrogate predictions with DFT reality and augments the training set.

Usage (inside Docker):
    python -m scripts.validate_dft
"""

from __future__ import annotations

import json
import logging
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ase import Atoms

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class Candidate:
    """A candidate structure discovered by the agent."""
    seed: str
    formula: str
    surrogate_reward: float
    surrogate_m_star: float
    actions_taken: list[str]
    fingerprint: np.ndarray  # stored separately

    # Filled after DFT validation
    dft_band_gap: float | None = None
    dft_m_min: float | None = None
    dft_m_electron: float | None = None
    dft_m_hole: float | None = None
    dft_converged: bool = False
    dft_time_s: float = 0.0


def generate_candidates(model, env, surrogate, n_episodes: int = 30) -> list[tuple[Candidate, "Atoms"]]:
    """Run trained agent episodes, collect all final structures with atoms."""
    from qe_interface.structures import structure_to_fingerprint

    results = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        total_r = 0.0
        actions = []

        for step in range(env.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_r += reward
            actions.append(env.ACTION_NAMES[int(action)])
            if terminated or truncated:
                break

        # Get the surrogate's prediction for the final structure
        fp = structure_to_fingerprint(env.atoms)
        m_pred = surrogate.predict(fp)

        candidate = Candidate(
            seed=info.get("structure", "?"),
            formula=env.atoms.get_chemical_formula(),
            surrogate_reward=total_r,
            surrogate_m_star=float(m_pred),
            actions_taken=actions,
            fingerprint=fp,
        )
        results.append((candidate, env.atoms.copy()))

    return results


def deduplicate_candidates(results: list[tuple[Candidate, "Atoms"]]) -> list[tuple[Candidate, "Atoms"]]:
    """Keep unique formulas, preferring highest reward."""
    best_by_formula: dict[str, tuple[Candidate, "Atoms"]] = {}
    for c, atoms in results:
        if c.formula not in best_by_formula or c.surrogate_reward > best_by_formula[c.formula][0].surrogate_reward:
            best_by_formula[c.formula] = (c, atoms)
    deduped = sorted(best_by_formula.values(), key=lambda x: x[0].surrogate_reward, reverse=True)
    return deduped


def validate_with_dft(candidate: Candidate, qe, atoms) -> Candidate:
    """Run real DFT bands calculation and fill in results."""
    from qe_interface.properties import analyze_bands

    t0 = time.time()
    try:
        result = qe.run_bands(atoms, npoints=20)
    except Exception as e:
        logger.error(f"  DFT failed for {candidate.formula}: {e}")
        candidate.dft_time_s = time.time() - t0
        return candidate

    candidate.dft_time_s = time.time() - t0
    candidate.dft_converged = result.converged

    if result.converged and result.band_energies is not None:
        n_electrons = qe.get_n_valence_electrons(atoms)
        props = analyze_bands(
            result.kpoints, result.band_energies, n_electrons, atoms.cell[:]
        )
        candidate.dft_band_gap = props.band_gap
        candidate.dft_m_min = props.min_effective_mass
        candidate.dft_m_electron = props.effective_mass_electron
        candidate.dft_m_hole = props.effective_mass_hole

    return candidate


def main():
    from stable_baselines3 import PPO

    from envs.crystal_env import CrystalEnv
    from models.surrogate import SurrogatePredictor
    from qe_interface.calculator import QECalculator, QEConfig
    from qe_interface.structures import structure_to_fingerprint

    # ---- Load surrogate ----
    # Try full retrained surrogate first, fall back to original bootstrap
    full_path = Path("data/bootstrap/surrogate_full")
    bootstrap_path = Path("data/bootstrap/surrogate_bootstrap")
    if (full_path / "surrogate_weights.pt").exists():
        surrogate = SurrogatePredictor(input_dim=84, hidden_dim=192, n_layers=4)
        surrogate.load(str(full_path))
        logger.info(f"Loaded FULL surrogate: {surrogate.dataset_size} samples")
    elif (bootstrap_path / "surrogate_weights.pt").exists():
        surrogate = SurrogatePredictor(input_dim=84, hidden_dim=128, n_layers=3)
        surrogate.load(str(bootstrap_path))
        logger.info(f"Loaded bootstrap surrogate: {surrogate.dataset_size} samples")
    else:
        raise FileNotFoundError("No surrogate. Run bootstrap first.")
    prior_dataset_size = surrogate.dataset_size

    # ---- Load trained agent ----
    all_seeds = ["Si", "Ge", "C-diamond", "GaAs", "AlAs", "InAs",
                 "GaP", "SiC-3C", "InP", "AlN"]

    env = CrystalEnv(
        seed_structure=all_seeds,
        species_palette=["Si", "Ge", "C", "Ga", "As", "Al", "In", "N", "P"],
        max_steps=40,
        use_surrogate=True,
        surrogate_model=surrogate,
        reward_mode="effective_mass",
    )

    # Try retrained model first, fall back to train_medium
    ckpt_retrain = Path("data/checkpoints/retrain_full/ppo_final.zip")
    ckpt_medium = Path("data/checkpoints/train_medium/ppo_final.zip")
    ckpt = ckpt_retrain if ckpt_retrain.exists() else ckpt_medium
    if not ckpt.exists():
        raise FileNotFoundError(f"No trained model found")
    model = PPO.load(str(ckpt), env=env)
    logger.info(f"Loaded PPO agent from {ckpt}")

    # ---- Generate candidates ----
    logger.info("\nGenerating 30 candidate structures from agent rollouts...")
    results = generate_candidates(model, env, surrogate, n_episodes=30)
    unique = deduplicate_candidates(results)
    logger.info(f"Generated {len(results)} → {len(unique)} unique formulas")

    for c, _ in unique[:10]:
        logger.info(f"  {c.formula:>8s}  surr_reward={c.surrogate_reward:7.1f}  "
                     f"surr_m*={c.surrogate_m_star:+.4f}")

    # ---- DFT validation on top candidates ----
    n_validate = min(8, len(unique))  # Validate top 8
    logger.info(f"\nValidating top {n_validate} candidates with DFT...")

    scratch = tempfile.mkdtemp(prefix="qe_validate_")
    qe_config = QEConfig(
        pw_command="mpirun --allow-run-as-root -np 2 pw.x",
        pseudo_dir="/workspace/pseudopotentials",
        scratch_dir=scratch,
        ecutwfc=30.0,
        ecutrho=240.0,
        conv_thr=1e-6,
        kpoints=(4, 4, 4),
    )
    qe = QECalculator(qe_config)

    validated = []
    for i, (candidate, atoms) in enumerate(unique[:n_validate]):
        logger.info(f"\n--- DFT {i+1}/{n_validate}: {candidate.formula} ---")

        candidate = validate_with_dft(candidate, qe, atoms.copy())
        validated.append(candidate)

        # Log comparison
        if candidate.dft_m_min is not None and np.isfinite(candidate.dft_m_min):
            logger.info(
                f"  Surrogate m*={candidate.surrogate_m_star:+.4f} vs "
                f"DFT m*={candidate.dft_m_min:+.4f}  "
                f"(gap={candidate.dft_band_gap:.3f} eV)  "
                f"[{candidate.dft_time_s:.0f}s]"
            )

            # Add to surrogate training set
            surrogate.add_data(candidate.fingerprint, candidate.dft_m_min)
        elif candidate.dft_converged:
            logger.warning(f"  DFT converged but no valid m* [{candidate.dft_time_s:.0f}s]")
        else:
            logger.warning(f"  DFT did not converge [{candidate.dft_time_s:.0f}s]")

    # ---- Retrain surrogate with augmented data ----
    n_new = surrogate.dataset_size - prior_dataset_size
    if n_new > 0:
        logger.info(f"\nRetraining surrogate with {n_new} new DFT points "
                     f"({surrogate.dataset_size} total)...")
        surrogate.train(epochs=500, verbose=False)
        surrogate.save(str(bootstrap_path))
        logger.info("Updated surrogate saved")

    # ---- Summary ----
    logger.info(f"\n{'='*60}")
    logger.info("DFT VALIDATION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"{'Formula':>10s} {'Surr m*':>10s} {'DFT m*':>10s} "
                f"{'Gap(eV)':>10s} {'Conv?':>6s} {'Time':>6s}")

    for c in validated:
        dft_m = f"{c.dft_m_min:+.4f}" if c.dft_m_min and np.isfinite(c.dft_m_min) else "N/A"
        gap = f"{c.dft_band_gap:.3f}" if c.dft_band_gap is not None else "N/A"
        logger.info(
            f"  {c.formula:>10s} {c.surrogate_m_star:+10.4f} "
            f"{dft_m:>10s} {gap:>10s} "
            f"{'Y' if c.dft_converged else 'N':>6s} "
            f"{c.dft_time_s:5.0f}s"
        )

    # Compute surrogate accuracy
    pairs = [(c.surrogate_m_star, c.dft_m_min) for c in validated
             if c.dft_m_min is not None and np.isfinite(c.dft_m_min)]
    if pairs:
        surr_vals = np.array([p[0] for p in pairs])
        dft_vals = np.array([p[1] for p in pairs])
        mae = np.mean(np.abs(surr_vals - dft_vals))
        corr = np.corrcoef(surr_vals, dft_vals)[0, 1] if len(pairs) > 2 else float("nan")
        logger.info(f"\nSurrogate accuracy: MAE={mae:.3f} m_e, correlation={corr:.3f}")
        logger.info(f"DFT-validated points added: {len(pairs)}")
        logger.info(f"Total surrogate dataset: {surrogate.dataset_size}")

    # Save validation report
    out = Path("data/validation")
    out.mkdir(parents=True, exist_ok=True)
    report = []
    for c in validated:
        entry = {
            "formula": c.formula,
            "seed": c.seed,
            "surrogate_reward": c.surrogate_reward,
            "surrogate_m_star": c.surrogate_m_star,
            "dft_band_gap": c.dft_band_gap,
            "dft_m_min": c.dft_m_min,
            "dft_m_electron": c.dft_m_electron,
            "dft_m_hole": c.dft_m_hole,
            "dft_converged": c.dft_converged,
            "dft_time_s": c.dft_time_s,
            "actions": c.actions_taken,
        }
        report.append(entry)
    with open(out / "validation_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"\nReport saved to {out}/validation_report.json")


if __name__ == "__main__":
    main()
