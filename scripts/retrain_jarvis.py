#!/usr/bin/env python3
"""Retrain multi-task surrogate on JARVIS-DFT 3D dataset.

This script:
  1. Downloads/loads JARVIS dft_3d (75,993 structures)
  2. Filters for records with effective mass AND band gap
  3. Converts JARVIS atoms dicts → ASE Atoms → 152-dim fingerprints
  4. Optionally loads our bootstrap DFT data and combines
  5. Optionally adds Materials Project gap-only data (single-pass or two-phase)
  6. Trains MultiTaskMLP with train/val split + early stopping
  7. Reports comprehensive metrics (correlation, MAE, gap classification)
  8. Saves model to data/checkpoints/jarvis_surrogate/

Usage:
    python scripts/retrain_jarvis.py --include-bootstrap [--include-mp] [--two-phase]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import pearsonr

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qe_interface.structures import structure_to_fingerprint
from models.surrogate import MultiTaskMLP, MultiTaskSurrogatePredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JARVIS atoms dict → ASE Atoms
# ---------------------------------------------------------------------------
def jarvis_to_ase(atoms_dict: dict):
    """Convert a JARVIS atoms dictionary to an ASE Atoms object."""
    from ase import Atoms

    return Atoms(
        symbols=atoms_dict["elements"],
        positions=atoms_dict["coords"],
        cell=atoms_dict["lattice_mat"],
        pbc=True,
    )


# ---------------------------------------------------------------------------
# Load and filter JARVIS data
# ---------------------------------------------------------------------------
def load_jarvis_data(
    require_mstar: bool = True,
    max_atoms: int = 50,
    max_gap: float = 20.0,
    max_mstar: float = 50.0,
) -> list[dict]:
    """Load JARVIS dft_3d and filter for usable records.

    Args:
        require_mstar: If True, only keep records with avg_elec_mass > 0
        max_atoms: Skip structures with more atoms (fingerprint speed)
        max_gap: Skip records with unreasonably large gap
        max_mstar: Skip records with unreasonably large m*

    Returns:
        List of filtered JARVIS records
    """
    from jarvis.db.figshare import data as jdata

    log.info("Loading JARVIS dft_3d dataset...")
    all_data = jdata("dft_3d")
    log.info(f"  Total records: {len(all_data)}")

    filtered = []
    skip_reasons = {"no_mstar": 0, "too_many_atoms": 0, "bad_gap": 0, "bad_mstar": 0, "no_atoms": 0, "error": 0}

    for rec in all_data:
        try:
            # Check atoms dict exists and is valid
            atoms_d = rec.get("atoms")
            if not atoms_d or not isinstance(atoms_d, dict) or "elements" not in atoms_d:
                skip_reasons["no_atoms"] += 1
                continue

            # Check effective mass
            m_star = rec.get("avg_elec_mass", 0)
            if m_star is None or not isinstance(m_star, (int, float)):
                m_star = 0
            if require_mstar and m_star <= 0:
                skip_reasons["no_mstar"] += 1
                continue

            # Check band gap
            gap = rec.get("optb88vdw_bandgap", None)
            if gap is None or not isinstance(gap, (int, float)) or gap < 0 or gap > max_gap:
                skip_reasons["bad_gap"] += 1
                continue

            # Check m* range (if present)
            if m_star > 0 and m_star > max_mstar:
                skip_reasons["bad_mstar"] += 1
                continue

            # Check atom count
            n_atoms = len(atoms_d["elements"])
            if n_atoms > max_atoms:
                skip_reasons["too_many_atoms"] += 1
                continue

            filtered.append(rec)
        except Exception as e:
            skip_reasons["error"] += 1

    log.info(f"  Filtered: {len(filtered)} records")
    log.info(f"  Skip reasons: {skip_reasons}")
    return filtered


def load_gap_only_data(
    max_atoms: int = 50,
    max_gap: float = 20.0,
) -> list[dict]:
    """Load JARVIS records that have band gap but no effective mass.

    These are used for gap-head augmentation only.
    """
    from jarvis.db.figshare import data as jdata

    log.info("Loading JARVIS gap-only records for augmentation...")
    all_data = jdata("dft_3d")

    filtered = []
    for rec in all_data:
        try:
            m_star = rec.get("avg_elec_mass", 0)
            if m_star is not None and m_star > 0:
                continue  # skip — these are already in the main dataset

            gap = rec.get("optb88vdw_bandgap", None)
            if gap is None or gap < 0 or gap > max_gap:
                continue

            # Only semiconductors for augmentation
            if gap == 0:
                continue

            n_atoms = len(rec["atoms"]["elements"])
            if n_atoms > max_atoms:
                continue

            filtered.append(rec)
        except Exception:
            pass

    log.info(f"  Gap-only augmentation records: {len(filtered)}")
    return filtered


# ---------------------------------------------------------------------------
# Compute fingerprints in batch
# ---------------------------------------------------------------------------
def compute_fingerprints(records: list[dict], label: str = "") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert JARVIS records → (fingerprints, mstar, gap) arrays.

    Returns:
        X: (N, 152) fingerprints
        y_mstar: (N,) effective mass values
        y_gap: (N,) band gap values
    """
    fps, mstars, gaps = [], [], []
    n_fail = 0
    t0 = time.time()

    for i, rec in enumerate(records):
        try:
            atoms = jarvis_to_ase(rec["atoms"])
            fp = structure_to_fingerprint(atoms)

            m_star = rec.get("avg_elec_mass", 0)
            if m_star is None or m_star <= 0:
                m_star = 0.0  # placeholder for gap-only records

            gap = rec.get("optb88vdw_bandgap", 0)

            fps.append(fp)
            mstars.append(m_star)
            gaps.append(gap)
        except Exception as e:
            n_fail += 1
            if n_fail <= 5:
                log.warning(f"  Failed to fingerprint {rec.get('jid', '?')}: {e}")

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            log.info(f"  {label} Fingerprinted {i+1}/{len(records)} ({rate:.0f}/s, {n_fail} failed)")

    elapsed = time.time() - t0
    log.info(f"  {label} Done: {len(fps)} fingerprints in {elapsed:.1f}s ({n_fail} failed)")

    X = np.array(fps, dtype=np.float32)
    y_m = np.array(mstars, dtype=np.float32)
    y_g = np.array(gaps, dtype=np.float32)
    return X, y_m, y_g


# ---------------------------------------------------------------------------
# Load bootstrap data
# ---------------------------------------------------------------------------
def load_bootstrap_data(
    data_dir: Path,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Load our DFT bootstrap data if available.

    Checks multiple locations for 152-dim fingerprints:
      1. surrogate_multitask/surrogate_data.npz  (from retrain_multitask.py)
      2. surrogate_full/surrogate_data.npz
      3. all_results.npz
    """
    boot_dir = data_dir / "bootstrap"

    # Try paths in preference order
    candidates = [
        boot_dir / "surrogate_multitask" / "surrogate_data.npz",
        boot_dir / "surrogate_full" / "surrogate_data.npz",
        boot_dir / "all_results.npz",
    ]

    for path in candidates:
        if path.exists():
            data = np.load(path)
            fps = data.get("fingerprints")
            if fps is not None and fps.shape[1] == 152:
                mstar_key = "targets_mstar" if "targets_mstar" in data else "targets"
                gap_key = "targets_gap" if "targets_gap" in data else None
                y_m = np.abs(data[mstar_key])  # absolute value to match JARVIS convention
                y_g = data[gap_key] if gap_key else np.zeros_like(y_m)
                # Filter: only keep records with reasonable m* and semiconductor gap
                valid = (y_m > 0.001) & (y_m < 50.0) & np.isfinite(y_m) & np.isfinite(y_g)
                log.info(f"  Loaded bootstrap from {path}: {fps.shape[0]} records ({valid.sum()} valid), {fps.shape[1]}d")
                return fps[valid], y_m[valid], y_g[valid]

    log.warning(f"No 152-dim bootstrap data found in {boot_dir}")
    return None, None, None


# ---------------------------------------------------------------------------
# NaN/Inf sanitization
# ---------------------------------------------------------------------------
def sanitize_arrays(X, y_m, y_g):
    """Remove rows with NaN or Inf values."""
    mask = np.all(np.isfinite(X), axis=1) & np.isfinite(y_m) & np.isfinite(y_g)
    n_bad = (~mask).sum()
    if n_bad > 0:
        log.warning(f"Removing {n_bad} rows with NaN/Inf values")
    return X[mask], y_m[mask], y_g[mask]


# ---------------------------------------------------------------------------
# Training with masked losses (for gap-only augmentation)
# ---------------------------------------------------------------------------
def finetune_gap_head(
    model: MultiTaskMLP,
    X_train: np.ndarray,
    y_g_train: np.ndarray,
    X_val: np.ndarray,
    y_g_val: np.ndarray,
    y_m_val: np.ndarray,
    g_mean: float,
    g_std: float,
    m_mean: float,
    m_std: float,
    lr: float = 1e-3,
    epochs: int = 200,
    batch_size: int = 128,
    patience: int = 50,
) -> dict:
    """Phase 2: freeze trunk + m* head, fine-tune gap head only with all data.

    Returns updated metrics dict.
    """
    device = torch.device("cpu")
    model = model.to(device)

    # Freeze everything except gap head
    for param in model.trunk.parameters():
        param.requires_grad = False
    for param in model.head_mstar.parameters():
        param.requires_grad = False
    for param in model.head_gap.parameters():
        param.requires_grad = True

    gap_params = [p for p in model.head_gap.parameters() if p.requires_grad]
    n_gap_params = sum(p.numel() for p in gap_params)
    log.info(f"Phase 2: fine-tuning gap head only ({n_gap_params} params)")

    optimizer = torch.optim.Adam(gap_params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20, min_lr=1e-6
    )
    criterion = nn.MSELoss()

    X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    yg_t = torch.tensor(y_g_train, dtype=torch.float32, device=device)
    X_v = torch.tensor(X_val, dtype=torch.float32, device=device)
    yg_v = torch.tensor(y_g_val, dtype=torch.float32, device=device)
    ym_v = torch.tensor(y_m_val, dtype=torch.float32, device=device)

    yg_t_norm = (yg_t - g_mean) / g_std
    yg_v_norm = (yg_v - g_mean) / g_std

    dataset = TensorDataset(X_t, yg_t_norm)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    log.info(f"  Phase 2: {len(X_t)} train, {len(X_v)} val, {epochs} max epochs")

    for epoch in range(epochs):
        model.train()
        for batch_X, batch_yg in loader:
            optimizer.zero_grad()
            _, pred_g = model(batch_X)
            loss = criterion(pred_g, batch_yg)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            _, vp_g = model(X_v)
            val_loss = criterion(vp_g, yg_v_norm).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 25 == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            log.info(f"    Phase2 Epoch {epoch:4d}: val_g_loss={val_loss:.4f} lr={lr_now:.2e}")

        if patience_counter >= patience:
            log.info(f"    Phase 2 early stopping at epoch {epoch}")
            break

    if best_state:
        model.load_state_dict(best_state)

    # Unfreeze all for saving (state_dict is fine either way, but be clean)
    for param in model.parameters():
        param.requires_grad = True

    # Compute final metrics
    model.eval()
    with torch.no_grad():
        vp_m, vp_g = model(X_v)

    pred_m = vp_m.cpu().numpy() * m_std + m_mean
    pred_g = vp_g.cpu().numpy() * g_std + g_mean
    true_m = y_m_val
    true_g = y_g_val

    metrics = {
        "m_mean": m_mean, "m_std": m_std,
        "g_mean": g_mean, "g_std": g_std,
    }

    if len(true_m) > 2:
        try:
            corr_m, _ = pearsonr(true_m[true_m > 0], pred_m[true_m > 0])
            metrics["mstar_correlation"] = corr_m
        except:
            metrics["mstar_correlation"] = float("nan")
        metrics["mstar_mae"] = np.mean(np.abs(true_m[true_m > 0] - pred_m[true_m > 0]))

    if len(true_g) > 2:
        try:
            corr_g, _ = pearsonr(true_g, pred_g)
            metrics["gap_correlation"] = corr_g
        except:
            metrics["gap_correlation"] = float("nan")
        metrics["gap_mae"] = np.mean(np.abs(true_g - pred_g))

    gap_threshold = 0.1
    true_semi = true_g > gap_threshold
    pred_semi = pred_g > gap_threshold
    if true_semi.sum() > 0:
        metrics["gap_accuracy"] = (true_semi == pred_semi).mean()
        metrics["gap_precision"] = (true_semi & pred_semi).sum() / max(pred_semi.sum(), 1)
        metrics["gap_recall"] = (true_semi & pred_semi).sum() / max(true_semi.sum(), 1)

    return metrics


def train_multitask(
    X_train: np.ndarray,
    y_m_train: np.ndarray,
    y_g_train: np.ndarray,
    X_val: np.ndarray,
    y_m_val: np.ndarray,
    y_g_val: np.ndarray,
    input_dim: int = 152,
    hidden_dim: int = 192,
    n_layers: int = 4,
    lr: float = 5e-4,
    gap_weight: float = 0.3,
    epochs: int = 500,
    batch_size: int = 128,
    patience: int = 80,
    has_mstar_mask_train: np.ndarray | None = None,
) -> tuple[MultiTaskMLP, dict]:
    """Train MultiTaskMLP with optional masked m* loss.

    Args:
        has_mstar_mask_train: Boolean mask. Where False, m* loss is not applied.
                              Allows gap-only records to contribute to gap head.
    """
    device = torch.device("cpu")
    model = MultiTaskMLP(input_dim, hidden_dim, n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=30, min_lr=1e-6
    )
    criterion = nn.MSELoss(reduction="none")

    # Convert to tensors
    X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    ym_t = torch.tensor(y_m_train, dtype=torch.float32, device=device)
    yg_t = torch.tensor(y_g_train, dtype=torch.float32, device=device)

    X_v = torch.tensor(X_val, dtype=torch.float32, device=device)
    ym_v = torch.tensor(y_m_val, dtype=torch.float32, device=device)
    yg_v = torch.tensor(y_g_val, dtype=torch.float32, device=device)

    # Normalize targets
    # For m*: only use records that actually have m* data
    if has_mstar_mask_train is not None:
        mstar_mask_t = torch.tensor(has_mstar_mask_train, dtype=torch.bool, device=device)
        m_vals = ym_t[mstar_mask_t]
    else:
        mstar_mask_t = torch.ones(len(ym_t), dtype=torch.bool, device=device)
        m_vals = ym_t

    m_mean = m_vals.mean().item()
    m_std = m_vals.std().item() + 1e-8
    g_mean = yg_t.mean().item()
    g_std = yg_t.std().item() + 1e-8

    ym_t_norm = (ym_t - m_mean) / m_std
    yg_t_norm = (yg_t - g_mean) / g_std
    ym_v_norm = (ym_v - m_mean) / m_std
    yg_v_norm = (yg_v - g_mean) / g_std

    # Build dataset with mask
    if has_mstar_mask_train is not None:
        mask_float = mstar_mask_t.float()
        dataset = TensorDataset(X_t, ym_t_norm, yg_t_norm, mask_float)
    else:
        mask_float = torch.ones(len(X_t), dtype=torch.float32, device=device)
        dataset = TensorDataset(X_t, ym_t_norm, yg_t_norm, mask_float)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    log.info(f"Training: {len(X_t)} train, {len(X_v)} val, {epochs} max epochs")
    log.info(f"  Normalization: m*={m_mean:.3f}±{m_std:.3f}, gap={g_mean:.3f}±{g_std:.3f}")

    for epoch in range(epochs):
        model.train()
        epoch_loss_m, epoch_loss_g, n_batches = 0.0, 0.0, 0

        for batch_X, batch_ym, batch_yg, batch_mask in loader:
            optimizer.zero_grad()
            pred_m, pred_g = model(batch_X)

            # Masked m* loss
            loss_m_per = criterion(pred_m, batch_ym) * batch_mask
            n_mstar = batch_mask.sum().clamp(min=1)
            loss_m = loss_m_per.sum() / n_mstar

            loss_g = criterion(pred_g, batch_yg).mean()
            loss = loss_m + gap_weight * loss_g
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss_m += loss_m.item()
            epoch_loss_g += loss_g.item()
            n_batches += 1

        epoch_loss_m /= max(n_batches, 1)
        epoch_loss_g /= max(n_batches, 1)

        # Validation
        model.eval()
        with torch.no_grad():
            vp_m, vp_g = model(X_v)
            vl_m = criterion(vp_m, ym_v_norm).mean().item()
            vl_g = criterion(vp_g, yg_v_norm).mean().item()
            val_loss = vl_m + gap_weight * vl_g

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 25 == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            log.info(
                f"  Epoch {epoch:4d}: train_m={epoch_loss_m:.4f} train_g={epoch_loss_g:.4f} "
                f"val_m={vl_m:.4f} val_g={vl_g:.4f} lr={lr_now:.2e} patience={patience_counter}"
            )

        if patience_counter >= patience:
            log.info(f"  Early stopping at epoch {epoch}")
            break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # --- Compute final metrics ---
    model.eval()
    with torch.no_grad():
        vp_m, vp_g = model(X_v)

    # Denormalize predictions
    pred_m = vp_m.cpu().numpy() * m_std + m_mean
    pred_g = vp_g.cpu().numpy() * g_std + g_mean
    true_m = y_m_val
    true_g = y_g_val

    metrics = {
        "m_mean": m_mean, "m_std": m_std,
        "g_mean": g_mean, "g_std": g_std,
        "best_val_loss": best_val_loss,
        "epochs_trained": epoch + 1,
    }

    # m* correlation and MAE
    if len(true_m) > 2:
        try:
            corr_m, _ = pearsonr(true_m, pred_m)
            metrics["mstar_correlation"] = corr_m
        except:
            metrics["mstar_correlation"] = float("nan")
        metrics["mstar_mae"] = np.mean(np.abs(true_m - pred_m))

    # gap correlation and MAE
    if len(true_g) > 2:
        try:
            corr_g, _ = pearsonr(true_g, pred_g)
            metrics["gap_correlation"] = corr_g
        except:
            metrics["gap_correlation"] = float("nan")
        metrics["gap_mae"] = np.mean(np.abs(true_g - pred_g))

    # Gap classification: semiconductor (gap > 0.1) vs metal
    gap_threshold = 0.1
    true_semi = true_g > gap_threshold
    pred_semi = pred_g > gap_threshold
    if true_semi.sum() > 0:
        metrics["gap_accuracy"] = (true_semi == pred_semi).mean()
        metrics["gap_precision"] = (true_semi & pred_semi).sum() / max(pred_semi.sum(), 1)
        metrics["gap_recall"] = (true_semi & pred_semi).sum() / max(true_semi.sum(), 1)

    return model, metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Retrain surrogate on JARVIS data")
    parser.add_argument("--include-bootstrap", action="store_true",
                        help="Also include our 794 DFT bootstrap records")
    parser.add_argument("--gap-only-augment", action="store_true",
                        help="Add semiconductor gap-only records for gap head training")
    parser.add_argument("--include-mp", action="store_true",
                        help="Add Materials Project semiconductor gap data (12k+ records)")
    parser.add_argument("--max-mp", type=int, default=12000,
                        help="Max MP augmentation records (default: 12000)")
    parser.add_argument("--two-phase", action="store_true",
                        help="Two-phase training: Phase 1 trains full model on JARVIS+bootstrap, "
                             "Phase 2 freezes trunk+m* head and fine-tunes gap head with MP data")
    parser.add_argument("--max-gap-only", type=int, default=5000,
                        help="Max gap-only augmentation records")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gap-weight", type=float, default=0.3,
                        help="Weight of gap loss vs m* loss (lower = prioritize m*)")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--save-dir", type=str, default="data/checkpoints/jarvis_surrogate")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    data_dir = Path("data")
    cache_file = data_dir / "jarvis_fingerprints.npz"

    # ===== Step 1: Load/compute JARVIS fingerprints =====
    if cache_file.exists():
        log.info(f"Loading cached fingerprints from {cache_file}")
        cached = np.load(cache_file)
        X_jarvis = cached["X"]
        y_m_jarvis = cached["y_mstar"]
        y_g_jarvis = cached["y_gap"]
        log.info(f"  Loaded: {X_jarvis.shape[0]} records, {X_jarvis.shape[1]} features")
    else:
        # Load and filter
        records = load_jarvis_data(require_mstar=True)

        # Compute fingerprints
        log.info(f"Computing fingerprints for {len(records)} JARVIS records...")
        X_jarvis, y_m_jarvis, y_g_jarvis = compute_fingerprints(records, label="JARVIS")

        # Sanitize
        X_jarvis, y_m_jarvis, y_g_jarvis = sanitize_arrays(X_jarvis, y_m_jarvis, y_g_jarvis)

        # Cache
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_file, X=X_jarvis, y_mstar=y_m_jarvis, y_gap=y_g_jarvis)
        log.info(f"  Cached fingerprints to {cache_file}")

    log.info(f"JARVIS data: {X_jarvis.shape[0]} records")
    log.info(f"  m* range: [{y_m_jarvis.min():.3f}, {y_m_jarvis.max():.3f}], mean={y_m_jarvis.mean():.3f}")
    log.info(f"  gap range: [{y_g_jarvis.min():.3f}, {y_g_jarvis.max():.3f}], mean={y_g_jarvis.mean():.3f}")

    # Track which records have m* (for masked loss)
    has_mstar = y_m_jarvis > 0

    # ===== Step 2: Optionally load bootstrap data =====
    if args.include_bootstrap:
        X_boot, ym_boot, yg_boot = load_bootstrap_data(data_dir)
        if X_boot is not None:
            # Ensure matching feature dims
            if X_boot.shape[1] != X_jarvis.shape[1]:
                log.warning(
                    f"Bootstrap dim {X_boot.shape[1]} != JARVIS dim {X_jarvis.shape[1]}, "
                    f"skipping bootstrap data"
                )
            else:
                log.info(f"Adding {X_boot.shape[0]} bootstrap records")
                X_jarvis = np.vstack([X_jarvis, X_boot])
                y_m_jarvis = np.concatenate([y_m_jarvis, ym_boot])
                y_g_jarvis = np.concatenate([y_g_jarvis, yg_boot])
                has_mstar = np.concatenate([has_mstar, ym_boot > 0])

    # ===== Step 3a: Optionally add Materials Project gap data =====
    # When two-phase, MP data is added AFTER Phase 1 training (gap head fine-tune)
    if args.include_mp and not args.two_phase:
        mp_cache = data_dir / "mp_fingerprints.npz"
        if mp_cache.exists():
            log.info(f"Loading Materials Project fingerprints from {mp_cache}")
            mp_data = np.load(mp_cache, allow_pickle=True)
            X_mp = mp_data["X"]
            yg_mp = mp_data["y_gap"]

            # Subsample if needed
            if len(X_mp) > args.max_mp:
                rng_mp = np.random.default_rng(123)
                idx = rng_mp.choice(len(X_mp), args.max_mp, replace=False)
                X_mp = X_mp[idx]
                yg_mp = yg_mp[idx]

            log.info(f"  Adding {len(X_mp)} MP records (gap-only augmentation)")
            log.info(f"  MP gap range: [{yg_mp.min():.3f}, {yg_mp.max():.3f}], mean={yg_mp.mean():.3f}")
            X_jarvis = np.vstack([X_jarvis, X_mp])
            y_m_jarvis = np.concatenate([y_m_jarvis, np.zeros(len(X_mp))])
            y_g_jarvis = np.concatenate([y_g_jarvis, yg_mp])
            has_mstar = np.concatenate([has_mstar, np.zeros(len(X_mp), dtype=bool)])
        else:
            log.warning(f"MP cache not found at {mp_cache}. Run scripts/download_mp.py first.")

    # ===== Step 3b: Optionally add gap-only augmentation =====
    if args.gap_only_augment:
        gap_cache = data_dir / "jarvis_gap_only_fingerprints.npz"
        if gap_cache.exists():
            log.info(f"Loading cached gap-only fingerprints from {gap_cache}")
            cached = np.load(gap_cache)
            X_gap = cached["X"][:args.max_gap_only]
            yg_gap = cached["y_gap"][:args.max_gap_only]
        else:
            gap_records = load_gap_only_data()
            # Subsample
            if len(gap_records) > args.max_gap_only:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(gap_records), args.max_gap_only, replace=False)
                gap_records = [gap_records[i] for i in idx]

            log.info(f"Computing fingerprints for {len(gap_records)} gap-only records...")
            X_gap, _, yg_gap = compute_fingerprints(gap_records, label="GapOnly")

            gap_cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(gap_cache, X=X_gap, y_gap=yg_gap)

        log.info(f"Adding {len(X_gap)} gap-only augmentation records")
        X_jarvis = np.vstack([X_jarvis, X_gap])
        y_m_jarvis = np.concatenate([y_m_jarvis, np.zeros(len(X_gap))])
        y_g_jarvis = np.concatenate([y_g_jarvis, yg_gap])
        has_mstar = np.concatenate([has_mstar, np.zeros(len(X_gap), dtype=bool)])

    # Final sanitize
    X_jarvis, y_m_jarvis, y_g_jarvis = sanitize_arrays(X_jarvis, y_m_jarvis, y_g_jarvis)
    # has_mstar might have been invalidated by sanitize — but it tracks by index
    # Recompute
    has_mstar = y_m_jarvis > 0

    # ===== Step 4: Train/val split =====
    n = len(X_jarvis)
    n_val = int(n * args.val_frac)
    rng = np.random.default_rng(42)
    perm = rng.permutation(n)

    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    X_train, X_val = X_jarvis[train_idx], X_jarvis[val_idx]
    ym_train, ym_val = y_m_jarvis[train_idx], y_m_jarvis[val_idx]
    yg_train, yg_val = y_g_jarvis[train_idx], y_g_jarvis[val_idx]
    mask_train = has_mstar[train_idx]

    log.info(f"\nTrain: {len(X_train)} ({mask_train.sum()} with m*)")
    log.info(f"Val:   {len(X_val)} ({has_mstar[val_idx].sum()} with m*)")

    # ===== Step 5: Train =====
    input_dim = X_train.shape[1]
    log.info(f"\n{'='*60}")
    log.info(f"Training MultiTaskMLP ({input_dim} → {args.hidden_dim} × 4 → m* + gap)")
    log.info(f"{'='*60}")

    t0 = time.time()
    model, metrics = train_multitask(
        X_train, ym_train, yg_train,
        X_val, ym_val, yg_val,
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        n_layers=4,
        lr=args.lr,
        gap_weight=args.gap_weight,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=80,
        has_mstar_mask_train=mask_train if not np.all(mask_train) else None,
    )
    elapsed = time.time() - t0
    log.info(f"\nPhase 1 training completed in {elapsed:.1f}s")

    # ===== Step 5b: Two-phase training — fine-tune gap head with MP data =====
    if args.two_phase and args.include_mp:
        log.info(f"\n{'='*60}")
        log.info("PHASE 1 METRICS (before gap fine-tuning)")
        log.info(f"{'='*60}")
        for key, label in [
            ("mstar_correlation", "m* correlation"),
            ("gap_accuracy",      "gap accuracy"),
        ]:
            v = metrics.get(key)
            if v is not None:
                log.info(f"  {label:20s} {float(v):.4f}")

        # Load MP data
        mp_cache = data_dir / "mp_fingerprints.npz"
        if mp_cache.exists():
            mp_data = np.load(mp_cache, allow_pickle=True)
            X_mp = mp_data["X"]
            yg_mp = mp_data["y_gap"]
            if len(X_mp) > args.max_mp:
                rng_mp = np.random.default_rng(123)
                idx = rng_mp.choice(len(X_mp), args.max_mp, replace=False)
                X_mp = X_mp[idx]
                yg_mp = yg_mp[idx]
            log.info(f"\n{'='*60}")
            log.info(f"PHASE 2: Fine-tuning gap head with {len(X_mp)} MP records")
            log.info(f"{'='*60}")

            # Combine all data for gap training (but only val on ALL-data val split)
            X_all = np.vstack([X_jarvis, X_mp])
            yg_all = np.concatenate([y_g_jarvis, yg_mp])
            ym_all = np.concatenate([y_m_jarvis, np.zeros(len(X_mp))])

            n_all = len(X_all)
            n_val2 = int(n_all * args.val_frac)
            rng2 = np.random.default_rng(99)
            perm2 = rng2.permutation(n_all)
            val_idx2 = perm2[:n_val2]
            train_idx2 = perm2[n_val2:]

            t1 = time.time()
            phase2_metrics = finetune_gap_head(
                model,
                X_all[train_idx2], yg_all[train_idx2],
                X_all[val_idx2], yg_all[val_idx2],
                ym_all[val_idx2],
                g_mean=metrics["g_mean"], g_std=metrics["g_std"],
                m_mean=metrics["m_mean"], m_std=metrics["m_std"],
                lr=1e-3, epochs=200, batch_size=128, patience=50,
            )
            elapsed2 = time.time() - t1
            log.info(f"Phase 2 completed in {elapsed2:.1f}s")

            # Merge metrics (keep m_mean/m_std/g_mean/g_std from phase 1)
            metrics.update(phase2_metrics)
            # Update the combined data for saving
            X_jarvis = X_all
            y_m_jarvis = ym_all
            y_g_jarvis = yg_all
        else:
            log.warning(f"MP cache not found at {mp_cache} — skipping Phase 2")

    # ===== Step 6: Report metrics =====
    log.info(f"\n{'='*60}")
    log.info("VALIDATION METRICS")
    log.info(f"{'='*60}")
    for key, label in [
        ("mstar_correlation", "m* correlation"),
        ("mstar_mae",         "m* MAE"),
        ("gap_correlation",   "gap correlation"),
        ("gap_mae",           "gap MAE"),
        ("gap_accuracy",      "gap accuracy"),
        ("gap_precision",     "gap precision"),
        ("gap_recall",        "gap recall"),
    ]:
        v = metrics.get(key)
        if v is not None:
            log.info(f"  {label:20s} {float(v):.4f}")
        else:
            log.info(f"  {label:20s} N/A")
    log.info(f"  best val loss:      {metrics.get('best_val_loss', 'N/A'):.6f}")
    log.info(f"  epochs trained:     {metrics.get('epochs_trained', 'N/A')}")

    # ===== Step 7: Save =====
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save model weights
    torch.save(model.state_dict(), save_dir / "surrogate_weights.pt")

    # Save normalization stats (needed by MultiTaskSurrogatePredictor)
    np.savez(
        save_dir / "surrogate_norm.npz",
        m_mean=metrics["m_mean"],
        m_std=metrics["m_std"],
        g_mean=metrics["g_mean"],
        g_std=metrics["g_std"],
    )

    # Save training data reference (for loading into predictor)
    np.savez(
        save_dir / "surrogate_data.npz",
        fingerprints=X_jarvis,
        targets_mstar=y_m_jarvis,
        targets_gap=y_g_jarvis,
    )

    # Save metrics
    import json
    with open(save_dir / "metrics.json", "w") as f:
        json.dump({k: float(v) if isinstance(v, (float, np.floating)) else v
                    for k, v in metrics.items()}, f, indent=2)

    log.info(f"\nModel saved to {save_dir}/")
    log.info(f"  surrogate_weights.pt  ({sum(p.numel() for p in model.parameters()):,} params)")
    log.info(f"  surrogate_data.npz    ({X_jarvis.shape[0]} records × {X_jarvis.shape[1]} features)")
    log.info(f"  surrogate_norm.npz    (normalization stats)")
    log.info(f"  metrics.json          (validation results)")

    # ===== Step 8: Verification — load into predictor =====
    log.info(f"\nVerification: loading saved model into MultiTaskSurrogatePredictor...")
    predictor = MultiTaskSurrogatePredictor(input_dim=input_dim, hidden_dim=args.hidden_dim)
    predictor.load(save_dir)
    log.info(f"  Loaded OK: {predictor.dataset_size} samples")

    # Quick sanity check
    test_fp = X_val[0]
    m_pred, g_pred = predictor.predict_both(test_fp)
    log.info(f"  Sample prediction: m*={m_pred:.4f} (true={ym_val[0]:.4f}), gap={g_pred:.4f} (true={yg_val[0]:.4f})")

    log.info("\n✓ JARVIS surrogate training complete!")
    log.info(f"  To use: predictor.load('{save_dir}')")


if __name__ == "__main__":
    main()
