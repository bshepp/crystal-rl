"""Crystal Structure RL Environment.

A Gymnasium environment where an RL agent modifies crystal structures
to optimize material properties (e.g., minimize effective mass).

The environment operates in two modes:
  1. DFT mode: runs real QE calculations (slow, accurate)
  2. Surrogate mode: uses a trained MultiTaskMLP (fast, approximate)

Reward shaping:
  - Primary reward: 1/m* (lower effective mass → higher reward)
  - Metal penalty: -3.0 if predicted band gap < 0.05 eV
  - Soft stability discount: reward *= exp(-alpha * instability_score)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from ase.data import atomic_numbers
from gymnasium import spaces

from qe_interface.calculator import QECalculator, QEConfig, QEResult
from qe_interface.properties import BandProperties, analyze_bands
from qe_interface.structures import (
    make_seed_structure,
    perturb_lattice,
    perturb_positions,
    swap_species,
    structure_to_fingerprint,
    validate_structure,
)

logger = logging.getLogger(__name__)

# Pauling electronegativities for stability estimation
_ELECTRONEGATIVITY = {
    "Si": 1.90, "Ge": 2.01, "C": 2.55, "Sn": 1.96, "N": 3.04,
    "P": 2.19, "As": 2.18, "Ga": 1.81, "In": 1.78, "Al": 1.61,
    "Sb": 2.05, "Bi": 2.02, "Se": 2.55, "Te": 2.10,
    "B": 2.04, "O": 3.44,
}

# Covalent radii (Angstrom) for size mismatch estimation
_COVALENT_RADIUS = {
    "Si": 1.11, "Ge": 1.20, "C": 0.77, "Sn": 1.39, "N": 0.71,
    "P": 1.07, "As": 1.19, "Ga": 1.22, "In": 1.42, "Al": 1.21,
    "Sb": 1.39, "Bi": 1.48, "Se": 1.20, "Te": 1.38,
    "B": 0.82, "O": 0.66,
}


def estimate_instability(symbols: list[str]) -> float:
    """Lightweight instability score based on composition.

    Uses electronegativity difference and size mismatch as proxies
    for formation energy / hull distance.  Returns a dimensionless
    score ≥ 0 where 0 = perfectly matched (e.g. elemental crystals)
    and higher = more speculative compositions.

    The score is NOT a hard filter — it is used as a soft discount
    on the reward so the agent can still propose metastable things.
    """
    unique = list(set(symbols))
    if len(unique) <= 1:
        return 0.0  # elemental — trivially stable

    # Average pairwise electronegativity difference
    en_diffs = []
    rad_diffs = []
    for i, a in enumerate(unique):
        for b in unique[i + 1:]:
            en_a = _ELECTRONEGATIVITY.get(a, 2.0)
            en_b = _ELECTRONEGATIVITY.get(b, 2.0)
            en_diffs.append(abs(en_a - en_b))
            rad_a = _COVALENT_RADIUS.get(a, 1.2)
            rad_b = _COVALENT_RADIUS.get(b, 1.2)
            rad_diffs.append(abs(rad_a - rad_b) / max(rad_a, rad_b))

    # Combine: large EN diff (ionic) + large size mismatch = less likely to
    # form a stable tetrahedral semiconductor.  Weight EN more heavily.
    en_score = np.mean(en_diffs) if en_diffs else 0.0
    rad_score = np.mean(rad_diffs) if rad_diffs else 0.0

    # Penalize very unusual element counts (e.g. 4 different species in 4 atoms)
    n_species = len(unique)
    n_atoms = len(symbols)
    diversity_penalty = max(0.0, (n_species / n_atoms) - 0.5) * 0.5

    return float(en_score + 0.5 * rad_score + diversity_penalty)


class CrystalEnv(gym.Env):
    """RL environment for crystal structure optimization.

    Observation: fixed-length structural fingerprint (RDF-based).
    Actions: discrete set of structure modifications.

    Action space (Discrete(6)):
        0: Perturb positions (small)
        1: Perturb positions (large)
        2: Compress lattice
        3: Expand lattice
        4: Shear lattice
        5: Swap a species
    """

    metadata = {"render_modes": ["human"]}

    # Action descriptions
    ACTION_NAMES = [
        "perturb_pos_small",
        "perturb_pos_large",
        "compress_lattice",
        "expand_lattice",
        "shear_lattice",
        "swap_species",
    ]

    def __init__(
        self,
        seed_structure: str | list[str] = "Si",
        species_palette: Optional[list[str]] = None,
        max_steps: int = 50,
        use_surrogate: bool = False,
        surrogate_model: Optional[Any] = None,
        qe_config: Optional[QEConfig] = None,
        fingerprint_size: int = 64,
        reward_mode: str = "effective_mass",
        render_mode: Optional[str] = None,
        stability_alpha: float = 0.3,
    ):
        """Initialize the crystal RL environment.

        Args:
            seed_structure: Name of the starting crystal structure.
            species_palette: Allowed element types for swapping.
            max_steps: Maximum steps per episode.
            use_surrogate: If True, use surrogate model instead of DFT.
            surrogate_model: Trained surrogate model (required if use_surrogate=True).
            qe_config: Configuration for QE calculations.
            fingerprint_size: Size of the structural fingerprint vector.
            reward_mode: What property to optimize.
            render_mode: Gymnasium render mode.
            stability_alpha: Strength of soft stability discount (0 = off).
                Reward is multiplied by exp(-alpha * instability_score).
        """
        super().__init__()

        # Support multiple seed structures via comma-separated string or list
        if isinstance(seed_structure, list):
            self.seed_structure_names = seed_structure
        elif "," in seed_structure:
            self.seed_structure_names = [s.strip() for s in seed_structure.split(",")]
        else:
            self.seed_structure_names = [seed_structure]
        self.seed_structure_name = self.seed_structure_names[0]  # backward compat
        self.species_palette = species_palette or [
            "Si", "Ge", "C", "Sn", "N", "P", "As", "Ga", "In", "Al",
            "Sb", "Bi", "Se", "Te",
        ]
        self.max_steps = max_steps
        self.use_surrogate = use_surrogate
        self.surrogate_model = surrogate_model
        self.fingerprint_size = fingerprint_size
        self.reward_mode = reward_mode
        self.render_mode = render_mode
        self.stability_alpha = stability_alpha

        # QE calculator
        self.qe = QECalculator(qe_config or QEConfig())

        # Compute actual fingerprint size from a dummy structure
        # (fingerprint_size hint is used for n_bins in RDF, total is larger)
        self._n_rdf_bins = fingerprint_size
        dummy = make_seed_structure(self.seed_structure_names[0])
        actual_fp = structure_to_fingerprint(dummy, n_bins=self._n_rdf_bins)
        self._fp_size = len(actual_fp)

        # Spaces
        self.action_space = spaces.Discrete(len(self.ACTION_NAMES))
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._fp_size,),
            dtype=np.float32,
        )

        # Episode state
        self.atoms: Optional["Atoms"] = None
        self.step_count = 0
        self.best_reward = -np.inf
        self.episode_history = []
        self._cached_obs = None  # cache fingerprint to avoid recomputation

        # Random state
        self._rng = np.random.default_rng()

    def _get_obs(self) -> np.ndarray:
        """Compute observation from current structure (cached)."""
        assert self.atoms is not None, "call reset() before _get_obs()"
        if self._cached_obs is None:
            self._cached_obs = structure_to_fingerprint(self.atoms, n_bins=self._n_rdf_bins)
        return self._cached_obs

    def _invalidate_obs_cache(self) -> None:
        """Invalidate observation cache after structure modification."""
        self._cached_obs = None

    def _compute_reward(self) -> tuple[float, dict]:
        """Evaluate the current structure and compute reward.

        Applies soft stability discount: reward *= exp(-alpha * instability).

        Returns:
            Tuple of (reward, info_dict).
        """
        info = {}
        reward = self._compute_raw_reward(info)

        # Soft stability discount — penalizes speculative compositions
        # without hard-blocking them (metastable things can still score well)
        assert self.atoms is not None
        if self.stability_alpha > 0:
            symbols = list(self.atoms.get_chemical_symbols())
            instability = estimate_instability(symbols)
            stability_factor = float(np.exp(-self.stability_alpha * instability))
            info["instability_score"] = instability
            info["stability_factor"] = stability_factor
            if reward > 0:
                reward *= stability_factor
            # Don't amplify penalties — only discount positive rewards

        return reward, info

    def _compute_raw_reward(self, info: dict) -> float:
        """Compute the raw reward before stability discount."""

        if self.use_surrogate and self.surrogate_model is not None:
            # Fast surrogate evaluation — same shaping as DFT mode
            fp = self._get_obs()
            info["source"] = "surrogate"

            # Multi-task surrogate: predict m* AND band gap
            if hasattr(self.surrogate_model, 'predict_both'):
                prediction, predicted_gap = self.surrogate_model.predict_both(fp)
                info["predicted_value"] = prediction
                info["predicted_gap"] = predicted_gap
            else:
                prediction = self.surrogate_model.predict(fp)
                info["predicted_value"] = prediction
                predicted_gap = None

            # Apply reward shaping
            m = float(prediction)
            if m < 0:
                reward = 2.0 / max(abs(m), 0.01)  # reward for negative mass, clamp denom
            else:
                reward = -abs(m)  # penalize positive mass

            # Penalize predicted metals (band gap ≈ 0) — they are useless semiconductors
            if predicted_gap is not None and predicted_gap < 0.05:
                reward -= 3.0  # strong penalty for metals
                info["metal_penalty"] = True

            return reward

        # Real DFT evaluation
        scf_result = self.qe.run_scf(self.atoms)
        info["source"] = "dft"
        info["converged"] = scf_result.converged

        if not scf_result.converged:
            # Penalty for non-converging structures
            return -10.0

        info["energy"] = scf_result.energy
        info["max_force"] = float(np.max(np.abs(scf_result.forces)))

        if self.reward_mode == "effective_mass":
            # Run bands to get effective mass
            bands_result = self.qe.run_bands(self.atoms)
            if bands_result.band_energies is not None:
                n_electrons = self.qe.get_n_valence_electrons(self.atoms)
                props = analyze_bands(
                    bands_result.kpoints,
                    bands_result.band_energies,
                    n_electrons,
                    np.array(self.atoms.cell),
                )
                info["band_gap"] = props.band_gap
                info["effective_mass_e"] = props.effective_mass_electron
                info["effective_mass_h"] = props.effective_mass_hole

                # Reward: we want small |m*| — materials where carriers are light
                # or where curvature is inverted (negative m*)
                if props.min_effective_mass is not None and np.isfinite(props.min_effective_mass):
                    # Reward = -|m*| so smaller mass = higher reward
                    # Bonus for negative mass (inverted curvature)
                    m = props.min_effective_mass
                    if m < 0:
                        reward = 2.0 / abs(m)  # large reward for negative mass
                    else:
                        reward = -abs(m)
                    info["min_effective_mass"] = m
                    return reward

            # Bands didn't work; fall back to energy-based reward
            reward = -abs(scf_result.energy) / len(self.atoms)

        elif self.reward_mode == "energy":
            # Simple: minimize energy per atom
            reward = -scf_result.energy / len(self.atoms)

        else:
            raise ValueError(f"Unknown reward mode: {self.reward_mode}")

        return reward

    def _apply_action(self, action: int) -> None:
        """Apply the chosen action to modify the structure."""
        assert self.atoms is not None
        rng = self._rng

        if action == 0:  # perturb_pos_small
            self.atoms = perturb_positions(self.atoms, amplitude=0.01, rng=rng)

        elif action == 1:  # perturb_pos_large
            self.atoms = perturb_positions(self.atoms, amplitude=0.05, rng=rng)

        elif action == 2:  # compress_lattice
            scale = 1.0 - rng.uniform(0.01, 0.05)
            cell = np.array(self.atoms.get_cell())
            self.atoms.set_cell(cell * scale, scale_atoms=True)

        elif action == 3:  # expand_lattice
            scale = 1.0 + rng.uniform(0.01, 0.05)
            cell = np.array(self.atoms.get_cell())
            self.atoms.set_cell(cell * scale, scale_atoms=True)

        elif action == 4:  # shear_lattice
            self.atoms = perturb_lattice(self.atoms, amplitude=0.03, rng=rng)

        elif action == 5:  # swap_species
            self.atoms = swap_species(self.atoms, self.species_palette, rng=rng)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment to a seed structure."""
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)

        # Randomly pick a seed structure from the configured set
        structure_name = self._rng.choice(self.seed_structure_names)
        if options and "structure" in options:
            structure_name = options["structure"]

        self.atoms = make_seed_structure(structure_name)
        self.step_count = 0
        self.best_reward = -np.inf
        self.episode_history = []
        self._invalidate_obs_cache()

        obs = self._get_obs()
        info = {"structure": structure_name, "formula": self.atoms.get_chemical_formula()}
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Take an action and return (obs, reward, terminated, truncated, info)."""
        assert self.atoms is not None, "call reset() before step()"
        self.step_count += 1

        # Apply the structural modification
        old_atoms = self.atoms.copy()
        old_obs = self._cached_obs  # save in case we revert
        self._apply_action(action)
        self._invalidate_obs_cache()

        # Validate the new structure
        if not validate_structure(self.atoms):
            # Revert and penalize
            self.atoms = old_atoms
            self._cached_obs = old_obs  # restore cached obs
            obs = self._get_obs()
            info = {"action": self.ACTION_NAMES[action], "valid": False}
            return obs, -5.0, False, False, info

        # Compute reward
        reward, info = self._compute_reward()
        info["action"] = self.ACTION_NAMES[action]
        info["valid"] = True
        info["step"] = self.step_count
        info["formula"] = self.atoms.get_chemical_formula()

        # Track best
        if reward > self.best_reward:
            self.best_reward = reward
            info["new_best"] = True

        self.episode_history.append(info)

        # Episode termination
        terminated = False  # no natural terminal state
        truncated = self.step_count >= self.max_steps

        obs = self._get_obs()
        return obs, reward, terminated, truncated, info

    def render(self):
        """Print current state."""
        if self.render_mode == "human" and self.atoms is not None:
            print(
                f"Step {self.step_count}: {self.atoms.get_chemical_formula()} "
                f"vol={self.atoms.get_volume():.1f} A^3 "
                f"best_reward={self.best_reward:.4f}"
            )
