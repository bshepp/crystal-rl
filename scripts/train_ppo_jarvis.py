#!/usr/bin/env python3
"""Train PPO agent using the JARVIS-trained multi-task surrogate.

Loads the pre-trained surrogate from data/checkpoints/jarvis_surrogate/
and trains a PPO agent to explore crystal structures. Best surrogate
achieved m* corr=0.886, gap acc=95.2% via two-phase training.

Usage:
    python scripts/train_ppo_jarvis.py [--timesteps 250000] [--surrogate-dir ...]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


CHECKPOINT_DIR = Path("data/checkpoints/ppo_jarvis")


def train_ppo(surrogate, total_timesteps: int = 250_000):
    """Train PPO agent against the JARVIS multi-task surrogate."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from envs.crystal_env import CrystalEnv

    all_seeds = [
        "Si", "Ge", "C-diamond", "GaAs", "AlAs",
        "InAs", "GaP", "SiC-3C", "InP", "AlN",
        "InSb", "GaSb",
        # 4-atom supercells for ternary exploration
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
    logger.info("PPO TRAINING (JARVIS surrogate)")
    logger.info(f"{'='*60}")
    logger.info(f"  Seeds: {len(all_seeds)}, obs_dim={env.observation_space.shape[0]}")
    logger.info(f"  Surrogate: {surrogate.dataset_size} training points")
    logger.info(f"  Metal penalty: gap < 0.05 → -3.0")
    logger.info(f"  Steps: {total_timesteps:,}")

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

    # Evaluation: 20 deterministic episodes
    logger.info(f"\n{'='*60}")
    logger.info("EVALUATION: 20 deterministic episodes")
    logger.info(f"{'='*60}")

    eval_rewards = []
    eval_formulas = []
    for ep in range(20):
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
        logger.info(f"  Eval {ep:2d}: reward={total_r:7.1f} {formula:12s} {m_str} {gap_str}")

    logger.info(f"\n  Eval mean: {np.mean(eval_rewards):.1f} ± {np.std(eval_rewards):.1f}")
    logger.info(f"  Unique formulas: {len(set(eval_formulas))}/{len(eval_formulas)}")
    logger.info(f"  Semiconductors: {sum(1 for f in eval_formulas if 'gap' not in f)}/{len(eval_formulas)}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train PPO with JARVIS surrogate")
    parser.add_argument("--timesteps", type=int, default=250_000)
    parser.add_argument("--surrogate-dir", type=str, default="data/checkpoints/jarvis_surrogate")
    args = parser.parse_args()

    from models.surrogate import MultiTaskSurrogatePredictor

    surrogate_dir = Path(args.surrogate_dir)
    logger.info(f"Loading JARVIS surrogate from {surrogate_dir}...")
    surrogate = MultiTaskSurrogatePredictor(input_dim=152, hidden_dim=192)
    surrogate.load(surrogate_dir)
    logger.info(f"  Loaded: {surrogate.dataset_size} samples")

    # Quick sanity check
    test_fp = np.array(surrogate.fingerprints[0])
    m, g = surrogate.predict_both(test_fp)
    logger.info(f"  Sanity: m*={m:.3f}, gap={g:.3f}")

    train_ppo(surrogate, total_timesteps=args.timesteps)

    logger.info(f"\nDone! PPO model saved to {CHECKPOINT_DIR / 'ppo_final.zip'}")


if __name__ == "__main__":
    main()
