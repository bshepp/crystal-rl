#!/usr/bin/env python3
"""Medium-scale RL training — 200k PPO steps with full surrogate.

Uses the bootstrap-trained surrogate (794+ DFT points) covering 10 crystal families.
Trains PPO for 200k timesteps with periodic logging and checkpointing.

Usage (inside Docker):
    python -m scripts.train_medium
"""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback

    from envs.crystal_env import CrystalEnv
    from models.surrogate import SurrogatePredictor
    from qe_interface.calculator import QEConfig

    # ---- Load 42-point surrogate ----
    surrogate = SurrogatePredictor(input_dim=84, hidden_dim=128, n_layers=3)
    bootstrap_path = Path("data/bootstrap/surrogate_bootstrap")

    if (bootstrap_path / "surrogate_weights.pt").exists():
        surrogate.load(str(bootstrap_path))
        logger.info(f"Loaded surrogate with {surrogate.dataset_size} samples")
    else:
        raise FileNotFoundError(
            "No bootstrapped surrogate found. Run bootstrap_full.py first."
        )

    # ---- Environment: all 10 seed families ----
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

    logger.info(f"Environment: {len(all_seeds)} seed structures, "
                f"obs={env.observation_space}, act={env.action_space}")

    # ---- Checkpoint + logging callback ----
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

            # Periodic checkpoints
            if self.num_timesteps - self._last_ckpt >= self.ckpt_every:
                self._last_ckpt = self.num_timesteps
                path = self.ckpt_dir / f"ppo_{self.num_timesteps}"
                self.model.save(str(path))
                logger.info(f"  [Checkpoint saved: {path}]")

            return True

    # ---- PPO setup ----
    total_timesteps = 200_000

    ckpt_dir = Path("data/checkpoints/train_medium")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        batch_size=128,
        n_steps=512,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,   # encourage exploration
        clip_range=0.2,
        verbose=0,
    )

    callback = TrainCallback(ckpt_dir, ckpt_every=50_000)

    logger.info(f"\nStarting PPO for {total_timesteps} steps...")
    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callback)
    train_time = time.time() - t0

    # ---- Final save ----
    model.save(str(ckpt_dir / "ppo_final"))

    # ---- Summary ----
    rewards = callback.episode_rewards
    logger.info(f"\n{'='*60}")
    logger.info("TRAINING SUMMARY (200k steps)")
    logger.info(f"{'='*60}")
    logger.info(f"Episodes: {callback.n_episodes}")
    logger.info(f"Time: {train_time:.0f}s ({train_time/60:.1f} min)")
    logger.info(f"First 50 avg: {np.mean(rewards[:50]):.2f}")
    logger.info(f"Last 50 avg:  {np.mean(rewards[-50:]):.2f}")
    logger.info(f"Overall mean: {np.mean(rewards):.2f}")
    logger.info(f"Best:  {np.max(rewards):.2f}")
    logger.info(f"Worst: {np.min(rewards):.2f}")

    first_q = np.mean(rewards[:len(rewards)//4])
    last_q = np.mean(rewards[-len(rewards)//4:])
    logger.info(f"\nQ1 avg: {first_q:.2f} → Q4 avg: {last_q:.2f} "
                f"(Δ = {last_q - first_q:+.2f})")

    if last_q > first_q:
        logger.info(">>> Agent is IMPROVING <<<")
    else:
        logger.info(">>> No improvement — may need more data or tuning <<<")

    # ---- Evaluation ----
    logger.info(f"\n{'='*60}")
    logger.info("EVALUATION: 10 episodes (deterministic)")
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
        seed = info.get("structure", "?")
        logger.info(f"  Eval {ep}: reward={total_r:.1f}, formula={formula}")

    logger.info(f"\nEval mean: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")


if __name__ == "__main__":
    main()
