#!/usr/bin/env python3
"""Retrain with improved fingerprint (neighbor-list RDF) + multi-task surrogate.

Key improvements over retrain_full.py:
  1. Recomputes fingerprints from reconstructed structures (old 84-dim → new 152-dim)
  2. Multi-task surrogate: predicts BOTH m* AND band_gap jointly
  3. RL reward penalises predicted metals (gap ≈ 0) to steer toward semiconductors
  4. Reports per-task validation metrics

Steps:
  1. Load 794-point bootstrap data
  2. Reconstruct approximate structures from seed + formula
  3. Compute new 152-dim fingerprints with proper neighbor-list RDF
  4. Train multi-task surrogate (m* + band_gap) with early stopping
  5. Train PPO 250k steps with metal penalty
  6. Save everything

Usage:
    cd rl-materials
    $env:PYTHONPATH = "."
    python scripts/retrain_multitask.py
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
BOOTSTRAP_FILE = DATA_DIR / "bootstrap" / "s3-full" / "results" / "bootstrap" / "bootstrap_expanded_all.json"
SURROGATE_OUT = DATA_DIR / "bootstrap" / "surrogate_multitask"
CHECKPOINT_DIR = DATA_DIR / "checkpoints" / "retrain_multitask"


# ────────────────────────────────────────────────────────────
# 1. Reconstruct structures and recompute fingerprints
# ────────────────────────────────────────────────────────────

def reconstruct_structure(seed_name: str, formula: str):
    """Reconstruct an approximate ASE Atoms from seed template + formula.

    For zincblende/diamond 2-atom cells we know the crystal template from the
    seed name, and the chemical symbols from the formula.  Position perturbations
    (<5%) are lost, but composition + lattice → RDF is preserved.
    """
    from ase import Atoms
    from qe_interface.structures import make_seed_structure

    template = make_seed_structure(seed_name)

    # Parse formula into atom symbols
    # Formula is e.g. "GaSi", "Si2", "AsIn", "AlN"
    import re
    parts = re.findall(r'([A-Z][a-z]?)(\d*)', formula)
    symbols = []
    for elem, count in parts:
        if elem:
            symbols.extend([elem] * max(1, int(count) if count else 1))

    if len(symbols) != len(template):
        # Mismatch — pad or trim to match template length
        while len(symbols) < len(template):
            symbols.append(symbols[-1])
        symbols = symbols[:len(template)]

    # Use template positions + cell, just swap species
    atoms = template.copy()
    atoms.set_chemical_symbols(symbols)
    return atoms


def load_and_recompute(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load bootstrap data, reconstruct structures, compute new fingerprints.

    Returns:
        fingerprints: (N, 152) array — new neighbor-list-based RDF
        targets_mstar: (N,) array of m_min values
        targets_gap: (N,) array of band_gap values
    """
    from qe_interface.structures import structure_to_fingerprint

    logger.info(f"Loading bootstrap data from {path}")
    with open(path) as f:
        records = json.load(f)
    logger.info(f"  Loaded {len(records)} records")

    fingerprints, targets_m, targets_g = [], [], []
    skipped = 0

    for rec in records:
        m_min = rec.get("m_min")
        gap = rec.get("band_gap", 0.0)
        seed = rec.get("seed", "Si")
        formula = rec.get("formula", "Si2")

        if m_min is None or not np.isfinite(m_min):
            skipped += 1
            continue
        if gap is None or not np.isfinite(gap):
            gap = 0.0

        try:
            atoms = reconstruct_structure(seed, formula)
            fp = structure_to_fingerprint(atoms)
        except Exception as e:
            logger.warning(f"  Skipping {formula}: {e}")
            skipped += 1
            continue

        fingerprints.append(fp)
        targets_m.append(float(m_min))
        targets_g.append(float(gap))

    logger.info(f"  Recomputed {len(fingerprints)} fingerprints ({skipped} skipped)")
    logger.info(f"  Fingerprint dim: {fingerprints[0].shape[0]}")

    X = np.array(fingerprints, dtype=np.float32)
    ym = np.array(targets_m, dtype=np.float32)
    yg = np.array(targets_g, dtype=np.float32)
    return X, ym, yg


# ────────────────────────────────────────────────────────────
# 2. Train multi-task surrogate
# ────────────────────────────────────────────────────────────

def train_multitask_surrogate(
    X: np.ndarray,
    ym: np.ndarray,
    yg: np.ndarray,
    epochs: int = 2000,
    val_fraction: float = 0.1,
):
    """Train multi-task surrogate with train/val split and early stopping."""
    from models.surrogate import MultiTaskSurrogatePredictor

    fp_dim = X.shape[1]
    logger.info(f"\n{'='*60}")
    logger.info("MULTI-TASK SURROGATE TRAINING")
    logger.info(f"{'='*60}")
    logger.info(f"  Fingerprint dim: {fp_dim}")
    logger.info(f"  m* stats: mean={ym.mean():.4f} std={ym.std():.4f} "
                f"[{ym.min():.4f}, {ym.max():.4f}]")
    logger.info(f"  gap stats: mean={yg.mean():.4f} std={yg.std():.4f} "
                f"[{yg.min():.4f}, {yg.max():.4f}]")
    logger.info(f"  gap > 0: {(yg > 0).sum()}/{len(yg)} ({100*(yg>0).mean():.1f}%)")

    surrogate = MultiTaskSurrogatePredictor(
        input_dim=fp_dim,
        hidden_dim=192,
        n_layers=4,
        lr=5e-4,
        gap_weight=0.3,
    )

    surrogate.add_batch(X, ym, yg)
    metrics = surrogate.train(
        epochs=epochs,
        batch_size=64,
        verbose=True,
        val_frac=val_fraction,
    )

    logger.info(f"  Training complete: {metrics}")

    # Detailed validation
    n = len(ym)
    n_val = max(10, int(n * val_fraction))
    perm = np.random.permutation(n)
    val_idx = perm[:n_val]

    val_preds_m = surrogate.predict_batch(X[val_idx])
    val_preds_g = surrogate.predict_gap_batch(X[val_idx])

    mse_m = float(np.mean((val_preds_m - ym[val_idx])**2))
    mae_m = float(np.mean(np.abs(val_preds_m - ym[val_idx])))
    corr_m = float(np.corrcoef(val_preds_m, ym[val_idx])[0, 1]) if n_val > 2 else 0

    mse_g = float(np.mean((val_preds_g - yg[val_idx])**2))
    mae_g = float(np.mean(np.abs(val_preds_g - yg[val_idx])))
    corr_g = float(np.corrcoef(val_preds_g, yg[val_idx])[0, 1]) if n_val > 2 else 0

    # Gap classification: is it a semiconductor (gap > 0)?
    actual_semi = yg[val_idx] > 0.01
    pred_semi = val_preds_g > 0.05
    gap_accuracy = float(np.mean(actual_semi == pred_semi))

    logger.info(f"\n  m* validation: MSE={mse_m:.4f} MAE={mae_m:.4f} corr={corr_m:.4f}")
    logger.info(f"  gap validation: MSE={mse_g:.4f} MAE={mae_g:.4f} corr={corr_g:.4f}")
    logger.info(f"  gap classifier accuracy: {gap_accuracy:.2%}")

    return surrogate


# ────────────────────────────────────────────────────────────
# 3. Train PPO with metal penalty
# ────────────────────────────────────────────────────────────

def train_ppo(surrogate, total_timesteps: int = 250_000):
    """Train PPO agent against the multi-task surrogate."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback

    from envs.crystal_env import CrystalEnv

    all_seeds = [
        "Si", "Ge", "C-diamond", "GaAs", "AlAs",
        "InAs", "GaP", "SiC-3C", "InP", "AlN",
        "InSb", "GaSb",
        "GaAs-4", "InAs-4", "InSb-4", "GaSb-4", "Si-4", "Ge-4",
    ]

    env = CrystalEnv(
        seed_structure=all_seeds,
        species_palette=["Si", "Ge", "C", "Sn", "N", "P", "As", "Ga", "In", "Al",
                         "Sb", "Bi", "Se", "Te"],
        max_steps=40,
        use_surrogate=True,
        surrogate_model=surrogate,
        reward_mode="effective_mass",
    )

    logger.info(f"\n{'='*60}")
    logger.info("PPO TRAINING")
    logger.info(f"{'='*60}")
    logger.info(f"  Seeds: {len(all_seeds)}, obs_dim={env.observation_space.shape[0]}")
    logger.info(f"  Metal penalty active: True (gap < 0.05 → -3.0)")
    logger.info(f"  Steps: {total_timesteps}")

    class TrainCallback(BaseCallback):
        def __init__(self, ckpt_dir: Path, ckpt_every: int = 50_000):
            super().__init__()
            self.episode_rewards = []
            self.current_reward = 0.0
            self.n_episodes = 0
            self.metal_count = 0
            self.ckpt_dir = ckpt_dir
            self.ckpt_every = ckpt_every
            self._last_ckpt = 0

        def _on_step(self) -> bool:
            self.current_reward += self.locals["rewards"][0]
            infos = self.locals.get("infos", [{}])
            if infos and infos[0].get("metal_penalty"):
                self.metal_count += 1
            done = self.locals["dones"][0]
            if done:
                self.episode_rewards.append(self.current_reward)
                self.n_episodes += 1
                if self.n_episodes % 200 == 0:
                    recent = self.episode_rewards[-200:]
                    avg = np.mean(recent)
                    best = np.max(recent)
                    metal_frac = self.metal_count / max(1, self.num_timesteps)
                    logger.info(
                        f"  Ep {self.n_episodes}: avg200={avg:.1f} best={best:.1f} "
                        f"metal_frac={metal_frac:.2%} steps={self.num_timesteps}"
                    )
                self.current_reward = 0.0

            if self.num_timesteps - self._last_ckpt >= self.ckpt_every:
                self._last_ckpt = self.num_timesteps
                path = self.ckpt_dir / f"ppo_{self.num_timesteps}"
                self.model.save(str(path))
                logger.info(f"  [Checkpoint: {path}]")

            return True

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        batch_size=128,
        n_steps=512,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,
        clip_range=0.2,
        verbose=0,
    )

    callback = TrainCallback(CHECKPOINT_DIR, ckpt_every=50_000)

    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callback)
    train_time = time.time() - t0

    model.save(str(CHECKPOINT_DIR / "ppo_final"))

    rewards = callback.episode_rewards
    logger.info(f"\n{'='*60}")
    logger.info("PPO TRAINING SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"  Episodes: {callback.n_episodes}")
    logger.info(f"  Time: {train_time:.0f}s ({train_time/60:.1f} min)")

    if len(rewards) >= 50:
        logger.info(f"  First 50 avg: {np.mean(rewards[:50]):.2f}")
        logger.info(f"  Last 50 avg:  {np.mean(rewards[-50:]):.2f}")
    logger.info(f"  Overall mean: {np.mean(rewards):.2f}")
    logger.info(f"  Best: {np.max(rewards):.2f}")

    q1 = np.mean(rewards[:len(rewards)//4])
    q4 = np.mean(rewards[-len(rewards)//4:])
    logger.info(f"  Q1→Q4: {q1:.1f} → {q4:.1f} (Δ = {q4-q1:+.1f})")

    # Evaluation
    logger.info(f"\n{'='*60}")
    logger.info("EVALUATION: 10 deterministic episodes")
    logger.info(f"{'='*60}")

    eval_rewards = []
    eval_formulas = []
    for ep in range(10):
        obs, info = env.reset()
        total_r = 0.0
        for step in range(40):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_r += reward
            if terminated or truncated:
                break

        formula = info.get("formula", "?")
        pred_gap = info.get("predicted_gap", None)
        pred_m = info.get("predicted_value", None)
        eval_rewards.append(total_r)
        eval_formulas.append(formula)

        gap_str = f"gap={pred_gap:.3f}" if pred_gap is not None else ""
        m_str = f"m*={pred_m:.3f}" if pred_m is not None else ""
        logger.info(f"  Eval {ep}: reward={total_r:.1f} {formula} {m_str} {gap_str}")

    logger.info(f"\n  Eval mean: {np.mean(eval_rewards):.1f} ± {np.std(eval_rewards):.1f}")
    logger.info(f"  Unique formulas: {len(set(eval_formulas))}")

    return model


# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("MULTI-TASK RETRAIN: fixed RDF + joint m*+gap surrogate")
    logger.info("=" * 60)

    t_start = time.time()

    # 1. Load + recompute fingerprints
    X, ym, yg = load_and_recompute(BOOTSTRAP_FILE)
    logger.info(f"\nData: {len(ym)} pts, fingerprint dim={X.shape[1]}")

    # 2. Train multi-task surrogate
    surrogate = train_multitask_surrogate(X, ym, yg, epochs=2000)

    SURROGATE_OUT.mkdir(parents=True, exist_ok=True)
    surrogate.save(str(SURROGATE_OUT))
    logger.info(f"Surrogate saved to {SURROGATE_OUT}")

    # 3. Train PPO
    model = train_ppo(surrogate, total_timesteps=250_000)

    total_time = time.time() - t_start
    logger.info(f"\n{'='*60}")
    logger.info(f"ALL DONE in {total_time:.0f}s ({total_time/60:.1f} min)")
    logger.info(f"  Surrogate: {SURROGATE_OUT}")
    logger.info(f"  PPO agent: {CHECKPOINT_DIR / 'ppo_final.zip'}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
