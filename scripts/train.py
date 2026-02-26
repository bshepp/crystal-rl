"""RL Training loop using Stable-Baselines3.

Orchestrates the two-tier training:
  1. Collect DFT data via random/initial policy
  2. Train surrogate model
  3. Train RL agent against surrogate
  4. Periodically validate with real DFT
  5. Retrain surrogate with new DFT data
  6. Repeat
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from envs.crystal_env import CrystalEnv
from models.surrogate import SurrogatePredictor
from qe_interface.calculator import QEConfig
from qe_interface.structures import make_seed_structure, structure_to_fingerprint

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class DFTValidationCallback(BaseCallback):
    """Periodically run real DFT on the agent's proposed structures."""

    def __init__(
        self,
        surrogate: SurrogatePredictor,
        env: CrystalEnv,
        validate_every: int = 1000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.surrogate = surrogate
        self.env = env
        self.validate_every = validate_every
        self.n_dft_calls = 0

    def _on_step(self) -> bool:
        if self.num_timesteps % self.validate_every == 0 and self.env.atoms is not None:
            logger.info(f"Running DFT validation at timestep {self.num_timesteps}")

            # Temporarily switch to DFT mode
            old_mode = self.env.use_surrogate
            self.env.use_surrogate = False

            # Evaluate current structure with real DFT
            reward, info = self.env._compute_reward()
            self.n_dft_calls += 1

            # Add result to surrogate training data
            fp = structure_to_fingerprint(self.env.atoms)
            if info.get("min_effective_mass") is not None:
                self.surrogate.add_data(fp, info["min_effective_mass"])
                logger.info(
                    f"DFT validation #{self.n_dft_calls}: "
                    f"m* = {info['min_effective_mass']:.4f}, "
                    f"E = {info.get('energy', 'N/A')}"
                )

            # Restore surrogate mode
            self.env.use_surrogate = old_mode

        return True


def collect_initial_dft_data(
    surrogate: SurrogatePredictor,
    qe_config: QEConfig,
    seed_structures: list[str],
    n_perturbations: int = 20,
) -> None:
    """Bootstrap the surrogate model with DFT calculations on known structures.

    Args:
        surrogate: The surrogate model to populate with data.
        qe_config: QE calculation settings.
        seed_structures: Names of seed structures to evaluate.
        n_perturbations: Number of random perturbations of each seed.
    """
    from qe_interface.calculator import QECalculator
    from qe_interface.properties import analyze_bands
    from qe_interface.structures import perturb_lattice, perturb_positions

    qe = QECalculator(qe_config)
    rng = np.random.default_rng(42)

    for name in seed_structures:
        logger.info(f"Collecting DFT data for {name}...")
        base = make_seed_structure(name)

        # Evaluate the base structure + perturbations
        structures = [base]
        for _ in range(n_perturbations):
            s = base.copy()
            s = perturb_positions(s, amplitude=0.02, rng=rng)
            s = perturb_lattice(s, amplitude=0.03, rng=rng)
            structures.append(s)

        for i, atoms in enumerate(structures):
            logger.info(f"  {name} variant {i}/{len(structures)}")

            result = qe.run_bands(atoms)
            if not result.converged or result.band_energies is None:
                logger.warning(f"  Skipping {name} variant {i} (not converged)")
                continue

            n_electrons = qe.get_n_valence_electrons(atoms)
            props = analyze_bands(
                result.kpoints, result.band_energies, n_electrons, atoms.cell[:]
            )

            if props.min_effective_mass is not None and np.isfinite(props.min_effective_mass):
                fp = structure_to_fingerprint(atoms)
                surrogate.add_data(fp, props.min_effective_mass)
                logger.info(f"  m* = {props.min_effective_mass:.4f}")

    logger.info(f"Collected {surrogate.dataset_size} DFT data points")


def train(config_path: str = "configs/default.yaml") -> None:
    """Main training loop."""
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    rl_cfg = config["rl"]
    qe_cfg_dict = config["qe"]

    qe_config = QEConfig(
        pw_command=qe_cfg_dict["pw_command"],
        pseudo_dir=qe_cfg_dict["pseudo_dir"],
        scratch_dir=qe_cfg_dict["scratch_dir"],
        ecutwfc=qe_cfg_dict["defaults"]["ecutwfc"],
        ecutrho=qe_cfg_dict["defaults"]["ecutrho"],
        conv_thr=qe_cfg_dict["defaults"]["conv_thr"],
        mixing_beta=qe_cfg_dict["defaults"]["mixing_beta"],
        kpoints=tuple(qe_cfg_dict["defaults"]["kpoints"]),
    )

    # Initialize surrogate model
    surrogate = SurrogatePredictor(
        input_dim=84,
        hidden_dim=rl_cfg["surrogate"]["hidden_dim"],
        n_layers=rl_cfg["surrogate"]["n_layers"],
    )

    # Phase 1: Collect initial DFT data
    seed_names = [s["name"].split("-")[0].capitalize() for s in config["seed_structures"]]
    # Map to our seed structure names
    seed_map = {"Silicon": "Si", "Germanium": "Ge", "Gaas": "GaAs"}
    seed_structures = [seed_map.get(n, n) for n in seed_names]

    min_dataset = rl_cfg["surrogate"]["min_dataset_size"]
    logger.info(f"Phase 1: Collecting {min_dataset} initial DFT data points...")

    n_per_structure = max(1, min_dataset // len(seed_structures))
    collect_initial_dft_data(surrogate, qe_config, seed_structures, n_perturbations=n_per_structure)

    # Train initial surrogate
    if surrogate.dataset_size >= 10:
        logger.info("Training initial surrogate model...")
        surrogate.train(epochs=200, verbose=True)

    # Phase 2: RL training with surrogate
    logger.info("Phase 2: Starting RL training...")

    env = CrystalEnv(
        seed_structure="Si",
        species_palette=config["species_palette"],
        max_steps=rl_cfg["max_steps_per_episode"],
        use_surrogate=(surrogate.dataset_size >= min_dataset),
        surrogate_model=surrogate if surrogate.dataset_size >= min_dataset else None,
        qe_config=qe_config,
        reward_mode="effective_mass",
    )

    # Create RL agent
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=rl_cfg["learning_rate"],
        batch_size=rl_cfg["batch_size"],
        n_steps=rl_cfg["n_steps"],
        n_epochs=rl_cfg["n_epochs"],
        gamma=rl_cfg["gamma"],
        verbose=1,
        tensorboard_log="./data/tb_logs/",
    )

    # Callback for periodic DFT validation
    dft_callback = DFTValidationCallback(
        surrogate=surrogate,
        env=env,
        validate_every=int(1 / rl_cfg["surrogate"]["dft_validation_rate"]),
    )

    total_timesteps = rl_cfg["total_timesteps"]
    retrain_every = rl_cfg["surrogate"]["train_every"]

    # Training loop with periodic surrogate retraining
    steps_done = 0
    while steps_done < total_timesteps:
        chunk = min(retrain_every, total_timesteps - steps_done)
        logger.info(f"Training RL agent for {chunk} steps (total: {steps_done}/{total_timesteps})")

        model.learn(
            total_timesteps=chunk,
            callback=dft_callback,
            reset_num_timesteps=False,
        )
        steps_done += chunk

        # Retrain surrogate with new DFT data
        if surrogate.dataset_size > 10:
            logger.info(f"Retraining surrogate ({surrogate.dataset_size} samples)...")
            surrogate.train(epochs=100)

        # Save checkpoint
        checkpoint_dir = Path("data/checkpoints") / f"step_{steps_done}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(checkpoint_dir / "ppo_agent"))
        surrogate.save(str(checkpoint_dir / "surrogate"))
        logger.info(f"Checkpoint saved: {checkpoint_dir}")

    logger.info("Training complete!")
    logger.info(f"Total DFT evaluations: {dft_callback.n_dft_calls}")
    logger.info(f"Surrogate dataset size: {surrogate.dataset_size}")


if __name__ == "__main__":
    train()
