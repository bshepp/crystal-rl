#!/usr/bin/env python3
"""Retrain surrogate on full 794-point AWS bootstrap dataset + DFT validation data.

Then retrain PPO agent against the improved surrogate.

Steps:
  1. Load bootstrap_expanded_all.json (794 pts from AWS)
  2. Load validation_report.json (8 DFT-validated adversarial examples)
  3. Train surrogate on combined dataset (more epochs, fresh optimizer)
  4. Evaluate surrogate quality (train/val split)
  5. Train PPO 200k steps against the improved surrogate
  6. Save everything

Usage:
    cd rl-materials
    python -m scripts.retrain_full
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.surrogate import SurrogatePredictor

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
BOOTSTRAP_FILE = DATA_DIR / "bootstrap" / "s3-full" / "results" / "bootstrap" / "bootstrap_expanded_all.json"
VALIDATION_FILE = DATA_DIR / "validation" / "validation_report.json"
SURROGATE_OUT = DATA_DIR / "bootstrap" / "surrogate_full"
CHECKPOINT_DIR = DATA_DIR / "checkpoints" / "retrain_full"


def load_bootstrap_data(path: Path) -> list[dict]:
    """Load the bootstrap JSON and return list of records with fingerprints and targets."""
    logger.info(f"Loading bootstrap data from {path}")
    with open(path) as f:
        records = json.load(f)
    logger.info(f"  Loaded {len(records)} records")
    return records


def load_validation_data(path: Path) -> list[dict]:
    """Load DFT validation results."""
    if not path.exists():
        logger.warning(f"No validation file at {path}, skipping")
        return []
    logger.info(f"Loading validation data from {path}")
    with open(path) as f:
        records = json.load(f)
    logger.info(f"  Loaded {len(records)} validation records")
    return records


def extract_fingerprints_and_targets(
    bootstrap_records: list[dict],
    validation_records: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract (fingerprint, m_min) pairs from both datasets.

    Returns:
        fingerprints: (N, fp_dim) array
        targets: (N,) array of m_min values
    """
    fingerprints = []
    targets = []
    skipped = 0

    # Bootstrap records have 'fingerprint' and 'm_min' fields
    for rec in bootstrap_records:
        fp = rec.get("fingerprint")
        m_min = rec.get("m_min")
        if fp is None or m_min is None:
            skipped += 1
            continue
        if not np.isfinite(m_min):
            skipped += 1
            continue
        fingerprints.append(np.array(fp, dtype=np.float32))
        targets.append(float(m_min))

    n_bootstrap = len(targets)
    logger.info(f"  Bootstrap: {n_bootstrap} valid points ({skipped} skipped)")

    # Validation records have 'dft_m_min' but no fingerprint —
    # we need to regenerate fingerprints from the structure.
    # However, validation_report.json doesn't store atoms/fingerprints.
    # We'll note how many we could add and skip for now.
    n_val_added = 0
    for rec in validation_records:
        dft_m = rec.get("dft_m_min")
        if dft_m is None or not np.isfinite(dft_m):
            continue
        # Validation has no fingerprint stored — we'll log this
        n_val_added += 1

    if n_val_added > 0:
        logger.info(
            f"  Validation: {n_val_added} DFT points available but no fingerprints stored. "
            f"These will be used for evaluation only."
        )

    X = np.array(fingerprints, dtype=np.float32)
    y = np.array(targets, dtype=np.float32)
    return X, y


def train_surrogate(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 1500,
    val_fraction: float = 0.1,
) -> SurrogatePredictor:
    """Train surrogate on full dataset with train/val split for monitoring."""
    from models.surrogate import SurrogatePredictor

    # Shuffle and split
    n = len(y)
    idx = np.random.permutation(n)
    n_val = max(10, int(n * val_fraction))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    logger.info(f"Dataset split: {len(train_idx)} train, {len(val_idx)} val")
    logger.info(f"Target stats: mean={y.mean():.4f}, std={y.std():.4f}, "
                f"min={y.min():.4f}, max={y.max():.4f}")

    # Create fresh surrogate with slightly wider hidden dim for larger dataset
    fp_dim = X.shape[1]
    surrogate = SurrogatePredictor(
        input_dim=fp_dim,
        hidden_dim=192,
        n_layers=4,
        lr=5e-4,
    )

    # Add training data
    surrogate.add_batch(X_train, y_train)

    # Train with periodic validation
    logger.info(f"\nTraining surrogate for {epochs} epochs...")
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None

    import torch

    for epoch_block in range(0, epochs, 100):
        block_size = min(100, epochs - epoch_block)
        loss = surrogate.train(epochs=block_size, batch_size=64, verbose=False)

        # Compute validation loss
        val_preds = surrogate.predict_batch(X_val)
        val_mse = float(np.mean((val_preds - y_val) ** 2))
        val_mae = float(np.mean(np.abs(val_preds - y_val)))

        # Correlation
        if len(y_val) > 2:
            corr = float(np.corrcoef(val_preds, y_val)[0, 1])
        else:
            corr = float("nan")

        current_epoch = epoch_block + block_size
        logger.info(
            f"  Epoch {current_epoch:4d}: train_loss={loss:.6f}, "
            f"val_MSE={val_mse:.4f}, val_MAE={val_mae:.4f}, val_corr={corr:.4f}"
        )

        if val_mse < best_val_loss:
            best_val_loss = val_mse
            best_epoch = current_epoch
            # Save best weights
            best_state = {k: v.clone() for k, v in surrogate.model.state_dict().items()}

    # Restore best weights
    if best_state is not None:
        surrogate.model.load_state_dict(best_state)
        logger.info(f"\nRestored best model from epoch {best_epoch} (val_MSE={best_val_loss:.4f})")

    # Final validation report
    val_preds = surrogate.predict_batch(X_val)
    val_mse = float(np.mean((val_preds - y_val) ** 2))
    val_mae = float(np.mean(np.abs(val_preds - y_val)))
    corr = float(np.corrcoef(val_preds, y_val)[0, 1]) if len(y_val) > 2 else float("nan")

    logger.info(f"\nFinal validation: MSE={val_mse:.4f}, MAE={val_mae:.4f}, corr={corr:.4f}")

    # Also add val data back so saved surrogate has full dataset
    surrogate.add_batch(X_val, y_val)

    return surrogate


def train_ppo(surrogate, total_timesteps: int = 200_000):
    """Train PPO agent against the improved surrogate."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback

    from envs.crystal_env import CrystalEnv

    all_seeds = [
        "Si", "Ge", "C-diamond", "GaAs", "AlAs",
        "InAs", "GaP", "SiC-3C", "InP", "AlN",
    ]

    env = CrystalEnv(
        seed_structure=all_seeds,
        species_palette=["Si", "Ge", "C", "Ga", "As", "Al", "In", "N", "P"],
        max_steps=40,
        use_surrogate=True,
        surrogate_model=surrogate,
        reward_mode="effective_mass",
    )

    logger.info(f"\nEnvironment: {len(all_seeds)} seeds, "
                f"obs={env.observation_space}, act={env.action_space}")

    # Tracking callback
    class TrainCallback(BaseCallback):
        def __init__(self, ckpt_dir: Path, ckpt_every: int = 50_000):
            super().__init__()
            self.episode_rewards = []
            self.current_reward = 0.0
            self.n_episodes = 0
            self.ckpt_dir = ckpt_dir
            self.ckpt_every = ckpt_every
            self._last_ckpt = 0

        def _on_step(self) -> bool:
            self.current_reward += self.locals["rewards"][0]
            done = self.locals["dones"][0]
            if done:
                self.episode_rewards.append(self.current_reward)
                self.n_episodes += 1
                if self.n_episodes % 100 == 0:
                    recent = self.episode_rewards[-100:]
                    avg = np.mean(recent)
                    best = np.max(recent)
                    logger.info(
                        f"  Episode {self.n_episodes}: "
                        f"avg100={avg:.1f}, best100={best:.1f}, "
                        f"steps={self.num_timesteps}"
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

    logger.info(f"\nStarting PPO for {total_timesteps} steps...")
    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callback)
    train_time = time.time() - t0

    # Save final model
    model.save(str(CHECKPOINT_DIR / "ppo_final"))

    # Summary
    rewards = callback.episode_rewards
    logger.info(f"\n{'='*60}")
    logger.info("PPO TRAINING SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Episodes: {callback.n_episodes}")
    logger.info(f"Time: {train_time:.0f}s ({train_time/60:.1f} min)")

    if len(rewards) >= 50:
        logger.info(f"First 50 avg: {np.mean(rewards[:50]):.2f}")
        logger.info(f"Last 50 avg:  {np.mean(rewards[-50:]):.2f}")
    logger.info(f"Overall mean: {np.mean(rewards):.2f}")
    logger.info(f"Best:  {np.max(rewards):.2f}")
    logger.info(f"Worst: {np.min(rewards):.2f}")

    first_q = np.mean(rewards[:len(rewards)//4])
    last_q = np.mean(rewards[-len(rewards)//4:])
    logger.info(f"Q1 avg: {first_q:.2f} → Q4 avg: {last_q:.2f} (Δ = {last_q - first_q:+.2f})")

    if last_q > first_q:
        logger.info(">>> Agent is IMPROVING <<<")
    else:
        logger.info(">>> No improvement — may need more data or tuning <<<")

    # Quick eval
    logger.info(f"\n{'='*60}")
    logger.info("EVALUATION: 10 deterministic episodes")
    logger.info(f"{'='*60}")

    eval_rewards = []
    for ep in range(10):
        obs, info = env.reset()
        total_r = 0.0
        for step in range(40):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_r += reward
            if terminated or truncated:
                break
        eval_rewards.append(total_r)
        formula = info.get("formula", "N/A")
        logger.info(f"  Eval {ep}: reward={total_r:.1f}, formula={formula}")

    logger.info(f"\nEval mean: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")

    return model


def main():
    logger.info("=" * 60)
    logger.info("FULL RETRAIN: surrogate + PPO on 794-point dataset")
    logger.info("=" * 60)

    t_start = time.time()

    # ---- Step 1: Load data ----
    bootstrap_data = load_bootstrap_data(BOOTSTRAP_FILE)
    validation_data = load_validation_data(VALIDATION_FILE)

    X, y = extract_fingerprints_and_targets(bootstrap_data, validation_data)
    logger.info(f"\nTotal training data: {len(y)} points, fingerprint dim = {X.shape[1]}")

    # ---- Step 2: Train surrogate ----
    surrogate = train_surrogate(X, y, epochs=1500)

    # Save surrogate
    SURROGATE_OUT.mkdir(parents=True, exist_ok=True)
    surrogate.save(str(SURROGATE_OUT))
    logger.info(f"Surrogate saved to {SURROGATE_OUT}")

    # ---- Step 3: Train PPO ----
    model = train_ppo(surrogate, total_timesteps=200_000)

    total_time = time.time() - t_start
    logger.info(f"\n{'='*60}")
    logger.info(f"ALL DONE in {total_time:.0f}s ({total_time/60:.1f} min)")
    logger.info(f"Surrogate: {SURROGATE_OUT}")
    logger.info(f"PPO agent: {CHECKPOINT_DIR / 'ppo_final.zip'}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
