#!/usr/bin/env python3
"""First RL training run — short PPO against bootstrapped surrogate.

Loads the bootstrapped surrogate model and runs PPO for a small number
of steps to validate the full training loop.

Usage (inside Docker):
    python scripts/train_short.py
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

    # ---- Load bootstrapped surrogate ----
    surrogate = SurrogatePredictor(input_dim=84, hidden_dim=128, n_layers=3)
    bootstrap_path = Path("data/bootstrap/surrogate_bootstrap")

    if (bootstrap_path / "surrogate_weights.pt").exists():
        surrogate.load(str(bootstrap_path))
        logger.info(f"Loaded bootstrapped surrogate ({surrogate.dataset_size} samples)")
    else:
        logger.warning("No bootstrapped surrogate found — training with random surrogate")
        # Seed with some random data so the surrogate is functional
        rng = np.random.default_rng(42)
        for _ in range(50):
            x = rng.standard_normal(84).astype(np.float32)
            y = float(-1.0 + 0.5 * rng.standard_normal())
            surrogate.add_data(x, y)
        surrogate.train(epochs=100, verbose=False)

    # ---- QE config (for periodic DFT validation) ----
    scratch = tempfile.mkdtemp(prefix="qe_train_")
    qe_config = QEConfig(
        pw_command="mpirun --allow-run-as-root -np 2 pw.x",
        pseudo_dir="/workspace/pseudopotentials",
        scratch_dir=scratch,
        ecutwfc=30.0,
        ecutrho=240.0,
        conv_thr=1e-6,
        kpoints=(4, 4, 4),
    )

    # ---- Create environment ----
    env = CrystalEnv(
        seed_structure=["Si", "Ge", "GaAs"],
        species_palette=["Si", "Ge", "Ga", "As"],
        max_steps=30,
        use_surrogate=True,
        surrogate_model=surrogate,
        reward_mode="effective_mass",
    )

    logger.info("Environment created:")
    logger.info(f"  Obs space: {env.observation_space}")
    logger.info(f"  Action space: {env.action_space}")
    logger.info(f"  Actions: {env.ACTION_NAMES}")

    # ---- Reward tracking callback ----
    class RewardLogger(BaseCallback):
        def __init__(self):
            super().__init__()
            self.episode_rewards = []
            self.current_reward = 0.0
            self.n_episodes = 0

        def _on_step(self) -> bool:
            self.current_reward += self.locals["rewards"][0]
            done = self.locals["dones"][0]
            if done:
                self.episode_rewards.append(self.current_reward)
                self.n_episodes += 1
                if self.n_episodes % 10 == 0:
                    recent = self.episode_rewards[-10:]
                    avg = np.mean(recent)
                    logger.info(
                        f"  Episode {self.n_episodes}: "
                        f"avg reward (last 10) = {avg:.3f}, "
                        f"total steps = {self.num_timesteps}"
                    )
                self.current_reward = 0.0
            return True

    # ---- Train PPO ----
    total_timesteps = 50_000  # Medium run with real surrogate
    logger.info(f"\nStarting PPO training for {total_timesteps} timesteps...")

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        batch_size=64,
        n_steps=256,
        n_epochs=5,
        gamma=0.99,
        verbose=0,
    )

    callback = RewardLogger()
    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callback)
    train_time = time.time() - t0

    # ---- Results ----
    logger.info(f"\n{'='*60}")
    logger.info("TRAINING SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total timesteps: {total_timesteps}")
    logger.info(f"Episodes completed: {callback.n_episodes}")
    logger.info(f"Training time: {train_time:.1f}s")

    if callback.episode_rewards:
        rewards = callback.episode_rewards
        logger.info(f"Reward stats:")
        logger.info(f"  First 10 episodes avg: {np.mean(rewards[:10]):.3f}")
        if len(rewards) > 10:
            logger.info(f"  Last 10 episodes avg: {np.mean(rewards[-10:]):.3f}")
        logger.info(f"  Overall mean: {np.mean(rewards):.3f}")
        logger.info(f"  Best episode: {np.max(rewards):.3f}")
        logger.info(f"  Worst episode: {np.min(rewards):.3f}")

        # Check for learning signal
        if len(rewards) >= 20:
            first_half = np.mean(rewards[: len(rewards) // 2])
            second_half = np.mean(rewards[len(rewards) // 2 :])
            improvement = second_half - first_half
            logger.info(f"\nLearning signal:")
            logger.info(f"  First half avg: {first_half:.3f}")
            logger.info(f"  Second half avg: {second_half:.3f}")
            logger.info(f"  Improvement: {improvement:+.3f}")
            if improvement > 0:
                logger.info("  >>> Agent is IMPROVING! <<<")
            else:
                logger.info("  >>> No improvement yet (may need more steps) <<<")

    # Save checkpoint
    ckpt_dir = Path("data/checkpoints/train_short")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(ckpt_dir / "ppo_agent"))
    logger.info(f"\nModel saved to {ckpt_dir}/ppo_agent")

    # ---- Test the trained agent ----
    logger.info(f"\n{'='*60}")
    logger.info("EVALUATION: 5 episodes with trained agent")
    logger.info(f"{'='*60}")

    for ep in range(5):
        obs, info = env.reset()
        total_r = 0.0
        for step in range(20):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_r += reward
            if terminated or truncated:
                break
        logger.info(
            f"  Eval episode {ep}: steps={step+1}, reward={total_r:.3f}, "
            f"formula={info.get('formula', 'N/A')}"
        )


if __name__ == "__main__":
    main()
