"""Surrogate model for predicting DFT properties from structural fingerprints.

This is the "fast" model that the RL agent trains against.
It is periodically retrained on accumulated DFT results.

Three architectures:
  1. MLP: simple feedforward network on RDF fingerprints (baseline)
  2. MultiTaskMLP: joint prediction of effective mass + band gap (current default)
  3. GNN: graph neural network on atomic graphs (planned for future)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class SurrogateMLP(nn.Module):
    """Simple MLP surrogate model.

    Maps structural fingerprints -> predicted property (e.g., effective mass).
    """

    def __init__(self, input_dim: int = 152, hidden_dim: int = 128, n_layers: int = 4):
        super().__init__()

        layers = []
        in_dim = input_dim
        for _ in range(n_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.1),
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MultiTaskMLP(nn.Module):
    """Multi-task MLP: shared trunk with separate heads for m* and band gap.

    Predicting both properties jointly forces the learned representation
    to capture more physical information than single-task m* prediction.
    """

    def __init__(self, input_dim: int = 152, hidden_dim: int = 192, n_layers: int = 4):
        super().__init__()

        # Shared trunk
        trunk = []
        in_dim = input_dim
        for _ in range(n_layers - 1):
            trunk.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.1),
            ])
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*trunk)

        # Task-specific heads (small separate networks)
        self.head_mstar = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.head_gap = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (m_star_pred, gap_pred) both shape (batch,)."""
        h = self.trunk(x)
        m_star = self.head_mstar(h).squeeze(-1)
        gap = self.head_gap(h).squeeze(-1)
        return m_star, gap


class SurrogatePredictor:
    """Wrapper around the surrogate model for training and inference.

    Maintains a dataset of (fingerprint, property) pairs from DFT,
    and retrains the model when new data arrives.
    """

    def __init__(
        self,
        input_dim: int = 84,
        hidden_dim: int = 128,
        n_layers: int = 4,
        lr: float = 1e-3,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.model = SurrogateMLP(input_dim, hidden_dim, n_layers).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        # Accumulated training data
        self.fingerprints: list[np.ndarray] = []
        self.targets: list[float] = []

        self._trained = False

    def add_data(self, fingerprint: np.ndarray, target: float) -> None:
        """Add a single DFT result to the training set."""
        self.fingerprints.append(fingerprint.copy())
        self.targets.append(target)

    def add_batch(self, fingerprints: np.ndarray, targets: np.ndarray) -> None:
        """Add a batch of DFT results."""
        for fp, t in zip(fingerprints, targets):
            self.add_data(fp, float(t))

    @property
    def dataset_size(self) -> int:
        return len(self.targets)

    def train(self, epochs: int = 100, batch_size: int = 32, verbose: bool = False) -> float:
        """Retrain the surrogate model on all accumulated data.

        Returns:
            Final training loss.
        """
        if self.dataset_size < 10:
            logger.warning(f"Only {self.dataset_size} samples — skipping training")
            return float("inf")

        X = torch.tensor(np.array(self.fingerprints), dtype=torch.float32).to(self.device)
        y = torch.tensor(np.array(self.targets), dtype=torch.float32).to(self.device)

        # Normalize targets
        self._y_mean = y.mean().item()
        self._y_std = y.std().item() + 1e-8
        y_norm = (y - self._y_mean) / self._y_std

        dataset = TensorDataset(X, y_norm)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        final_loss = float("inf")

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            for batch_X, batch_y in loader:
                self.optimizer.zero_grad()
                pred = self.model(batch_X)
                loss = self.criterion(pred, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            epoch_loss /= max(n_batches, 1)
            if verbose and epoch % 20 == 0:
                logger.info(f"Surrogate epoch {epoch}: loss = {epoch_loss:.6f}")
            final_loss = epoch_loss

        self._trained = True
        logger.info(
            f"Surrogate trained: {self.dataset_size} samples, "
            f"final loss = {final_loss:.6f}"
        )
        return final_loss

    def predict(self, fingerprint: np.ndarray) -> float:
        """Predict property from a structural fingerprint.

        Args:
            fingerprint: RDF fingerprint vector.

        Returns:
            Predicted property value (denormalized).
        """
        if not self._trained:
            raise RuntimeError("Surrogate model has not been trained yet")

        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(fingerprint, dtype=torch.float32).unsqueeze(0).to(self.device)
            pred_norm = self.model(x).item()
            return pred_norm * self._y_std + self._y_mean

    def predict_batch(self, fingerprints: np.ndarray) -> np.ndarray:
        """Predict properties for a batch of fingerprints."""
        if not self._trained:
            raise RuntimeError("Surrogate model has not been trained yet")

        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(fingerprints, dtype=torch.float32).to(self.device)
            pred_norm = self.model(x).cpu().numpy()
            return pred_norm * self._y_std + self._y_mean

    def save(self, path: str | Path) -> None:
        """Save model weights and training data."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), path / "surrogate_weights.pt")
        np.savez(
            path / "surrogate_data.npz",
            fingerprints=np.array(self.fingerprints),
            targets=np.array(self.targets),
        )
        logger.info(f"Surrogate saved to {path}")

    def load(self, path: str | Path) -> None:
        """Load model weights and training data."""
        path = Path(path)

        self.model.load_state_dict(torch.load(path / "surrogate_weights.pt", weights_only=True))
        data = np.load(path / "surrogate_data.npz")
        self.fingerprints = list(data["fingerprints"])
        self.targets = list(data["targets"])

        # Recompute normalization stats from loaded data
        y = np.array(self.targets, dtype=np.float32)
        self._y_mean = float(y.mean())
        self._y_std = float(y.std()) + 1e-8

        self._trained = True
        logger.info(f"Surrogate loaded from {path} ({self.dataset_size} samples)")


class MultiTaskSurrogatePredictor:
    """Multi-task surrogate: predicts BOTH effective mass and band gap.

    Training on both targets jointly forces the shared representation to
    learn more physics.  At inference the RL agent uses m* prediction,
    while the gap prediction lets us filter out metallic structures.
    """

    def __init__(
        self,
        input_dim: int = 152,
        hidden_dim: int = 192,
        n_layers: int = 4,
        lr: float = 1e-3,
        gap_weight: float = 0.3,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.model = MultiTaskMLP(input_dim, hidden_dim, n_layers).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.gap_weight = gap_weight  # relative weight for gap loss

        # Accumulated training data
        self.fingerprints: list[np.ndarray] = []
        self.targets_mstar: list[float] = []
        self.targets_gap: list[float] = []

        self._trained = False

    def add_data(self, fingerprint: np.ndarray, m_star: float, band_gap: float = 0.0) -> None:
        """Add a single DFT result."""
        self.fingerprints.append(fingerprint.copy())
        self.targets_mstar.append(m_star)
        self.targets_gap.append(band_gap)

    def add_batch(
        self,
        fingerprints: np.ndarray,
        targets_mstar: np.ndarray,
        targets_gap: np.ndarray,
    ) -> None:
        """Add a batch of DFT results."""
        for fp, m, g in zip(fingerprints, targets_mstar, targets_gap):
            self.add_data(fp, float(m), float(g))

    @property
    def dataset_size(self) -> int:
        return len(self.targets_mstar)

    def train(
        self,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: bool = False,
        val_frac: float = 0.0,
    ) -> dict:
        """Retrain the multi-task surrogate on all accumulated data.

        Returns:
            Dict with train/val losses and per-task metrics.
        """
        if self.dataset_size < 10:
            logger.warning(f"Only {self.dataset_size} samples — skipping training")
            return {"loss": float("inf")}

        X = torch.tensor(np.array(self.fingerprints), dtype=torch.float32).to(self.device)
        y_m = torch.tensor(np.array(self.targets_mstar), dtype=torch.float32).to(self.device)
        y_g = torch.tensor(np.array(self.targets_gap), dtype=torch.float32).to(self.device)

        # Normalize each target independently
        self._m_mean = y_m.mean().item()
        self._m_std = y_m.std().item() + 1e-8
        self._g_mean = y_g.mean().item()
        self._g_std = y_g.std().item() + 1e-8

        y_m_norm = (y_m - self._m_mean) / self._m_std
        y_g_norm = (y_g - self._g_mean) / self._g_std

        # Optional train/val split
        n = len(X)
        if val_frac > 0 and n > 20:
            n_val = max(int(n * val_frac), 5)
            perm = torch.randperm(n)
            val_idx, train_idx = perm[:n_val], perm[n_val:]
            X_val, ym_val, yg_val = X[val_idx], y_m_norm[val_idx], y_g_norm[val_idx]
            X, y_m_norm, y_g_norm = X[train_idx], y_m_norm[train_idx], y_g_norm[train_idx]
        else:
            X_val = None

        dataset = TensorDataset(X, y_m_norm, y_g_norm)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        best_val_loss = float("inf")
        best_state: dict | None = None
        patience_counter = 0
        patience = 100
        epoch_loss_m = 0.0
        epoch_loss_g = 0.0
        ym_val = torch.tensor([])
        yg_val = torch.tensor([])

        self.model.train()
        metrics = {}

        for epoch in range(epochs):
            epoch_loss_m, epoch_loss_g, n_batches = 0.0, 0.0, 0
            for batch_X, batch_ym, batch_yg in loader:
                self.optimizer.zero_grad()
                pred_m, pred_g = self.model(batch_X)
                loss_m = self.criterion(pred_m, batch_ym)
                loss_g = self.criterion(pred_g, batch_yg)
                loss = loss_m + self.gap_weight * loss_g
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                epoch_loss_m += loss_m.item()
                epoch_loss_g += loss_g.item()
                n_batches += 1

            epoch_loss_m /= max(n_batches, 1)
            epoch_loss_g /= max(n_batches, 1)

            # Validation
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    vp_m, vp_g = self.model(X_val)
                    vl_m = self.criterion(vp_m, ym_val).item()
                    vl_g = self.criterion(vp_g, yg_val).item()
                    val_loss = vl_m + self.gap_weight * vl_g
                self.model.train()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                if verbose and epoch % 50 == 0:
                    logger.info(
                        f"Epoch {epoch}: train_m={epoch_loss_m:.4f} train_g={epoch_loss_g:.4f} "
                        f"val_m={vl_m:.4f} val_g={vl_g:.4f} patience={patience_counter}"
                    )

                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            else:
                if verbose and epoch % 50 == 0:
                    logger.info(f"Epoch {epoch}: loss_m={epoch_loss_m:.4f} loss_g={epoch_loss_g:.4f}")

        # Restore best if early stopping was used
        if best_state is not None:
            self.model.load_state_dict(best_state)

        self._trained = True
        metrics = {
            "train_loss_mstar": epoch_loss_m,
            "train_loss_gap": epoch_loss_g,
        }
        if X_val is not None:
            metrics["val_loss"] = best_val_loss
        logger.info(f"Multi-task surrogate trained: {self.dataset_size} samples, metrics={metrics}")
        return metrics

    def predict(self, fingerprint: np.ndarray) -> float:
        """Predict m* (effective mass) from a fingerprint — for RL compatibility."""
        if not self._trained:
            raise RuntimeError("Multi-task surrogate has not been trained yet")
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(fingerprint, dtype=torch.float32).unsqueeze(0).to(self.device)
            pred_m_norm, _ = self.model(x)
            return pred_m_norm.item() * self._m_std + self._m_mean

    def predict_both(self, fingerprint: np.ndarray) -> tuple[float, float]:
        """Predict both m* and band_gap."""
        if not self._trained:
            raise RuntimeError("Multi-task surrogate has not been trained yet")
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(fingerprint, dtype=torch.float32).unsqueeze(0).to(self.device)
            pred_m_norm, pred_g_norm = self.model(x)
            m = pred_m_norm.item() * self._m_std + self._m_mean
            g = pred_g_norm.item() * self._g_std + self._g_mean
            return m, g

    def predict_batch(self, fingerprints: np.ndarray) -> np.ndarray:
        """Predict m* for a batch — for RL compatibility."""
        if not self._trained:
            raise RuntimeError("Multi-task surrogate has not been trained yet")
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(fingerprints, dtype=torch.float32).to(self.device)
            pred_m_norm, _ = self.model(x)
            return pred_m_norm.cpu().numpy() * self._m_std + self._m_mean

    def predict_gap_batch(self, fingerprints: np.ndarray) -> np.ndarray:
        """Predict band gap for a batch."""
        if not self._trained:
            raise RuntimeError("Multi-task surrogate has not been trained yet")
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(fingerprints, dtype=torch.float32).to(self.device)
            _, pred_g_norm = self.model(x)
            return pred_g_norm.cpu().numpy() * self._g_std + self._g_mean

    def save(self, path: str | Path) -> None:
        """Save model weights and training data."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path / "surrogate_weights.pt")
        np.savez(
            path / "surrogate_data.npz",
            fingerprints=np.array(self.fingerprints),
            targets_mstar=np.array(self.targets_mstar),
            targets_gap=np.array(self.targets_gap),
        )
        # Save normalization stats for loading
        np.savez(
            path / "surrogate_norm.npz",
            m_mean=self._m_mean, m_std=self._m_std,
            g_mean=self._g_mean, g_std=self._g_std,
        )
        logger.info(f"Multi-task surrogate saved to {path}")

    def load(self, path: str | Path) -> None:
        """Load model weights and training data."""
        path = Path(path)
        self.model.load_state_dict(torch.load(path / "surrogate_weights.pt", weights_only=True))

        data = np.load(path / "surrogate_data.npz")
        self.fingerprints = list(data["fingerprints"])
        self.targets_mstar = list(data["targets_mstar"])
        self.targets_gap = list(data["targets_gap"])

        norm = np.load(path / "surrogate_norm.npz")
        self._m_mean = float(norm["m_mean"])
        self._m_std = float(norm["m_std"])
        self._g_mean = float(norm["g_mean"])
        self._g_std = float(norm["g_std"])

        self._trained = True
        logger.info(f"Multi-task surrogate loaded from {path} ({self.dataset_size} samples)")
