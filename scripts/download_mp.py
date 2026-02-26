#!/usr/bin/env python3
"""Download Materials Project semiconductor data and compute fingerprints.

Downloads stable semiconductor structures with band gaps from the Materials
Project API, converts pymatgen Structure → ASE Atoms → 152-dim fingerprints,
and caches the result for use as gap-only augmentation in surrogate training.

This massively expands gap-head training data (12k+ vs prior 757 records),
improving metal/semiconductor classification that drives the RL reward.

Usage:
    python scripts/download_mp.py [--api-key KEY] [--max-atoms 50]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qe_interface.structures import structure_to_fingerprint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def pymatgen_to_ase(structure):
    """Convert a pymatgen Structure to an ASE Atoms object."""
    from ase import Atoms

    return Atoms(
        symbols=[str(s) for s in structure.species],
        positions=structure.cart_coords,
        cell=structure.lattice.matrix,
        pbc=True,
    )


def download_mp_semiconductors(
    api_key: str,
    gap_min: float = 0.05,
    gap_max: float = 15.0,
    max_atoms: int = 50,
    stable_only: bool = True,
) -> list[dict]:
    """Download semiconductor entries from Materials Project.

    Args:
        api_key: Materials Project API key
        gap_min: Minimum band gap (eV) — excludes metals
        gap_max: Maximum band gap (eV) — excludes insulators with unrealistic gaps
        max_atoms: Maximum number of atoms per unit cell
        stable_only: Only include thermodynamically stable entries

    Returns:
        List of dicts with keys: material_id, formula, band_gap, structure (ASE Atoms)
    """
    from mp_api.client import MPRester

    log.info(f"Downloading MP semiconductors (gap={gap_min}–{gap_max} eV, ≤{max_atoms} atoms)...")

    t0 = time.time()
    with MPRester(api_key) as mpr:
        search_kwargs = dict(
            band_gap=(gap_min, gap_max),
            fields=[
                "material_id",
                "band_gap",
                "formula_pretty",
                "structure",
                "nsites",
                "is_gap_direct",
            ],
        )
        if stable_only:
            search_kwargs["is_stable"] = True

        docs = mpr.materials.summary.search(**search_kwargs)

    elapsed = time.time() - t0
    log.info(f"  Downloaded {len(docs)} entries in {elapsed:.1f}s")

    # Filter by atom count and convert
    records = []
    skipped = 0
    for d in docs:
        if d.nsites > max_atoms:
            skipped += 1
            continue
        try:
            atoms = pymatgen_to_ase(d.structure)
            records.append({
                "material_id": str(d.material_id),
                "formula": d.formula_pretty,
                "band_gap": float(d.band_gap),
                "is_direct": bool(d.is_gap_direct) if d.is_gap_direct is not None else None,
                "nsites": int(d.nsites),
                "atoms": atoms,
            })
        except Exception as e:
            skipped += 1
            if skipped <= 5:
                log.warning(f"  Skipped {d.material_id}: {e}")

    log.info(f"  Usable: {len(records)} (skipped {skipped})")
    return records


def compute_fingerprints(
    records: list[dict],
    batch_log_interval: int = 500,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Compute 152-dim fingerprints for MP records.

    Returns:
        X: (N, 152) fingerprints
        y_gap: (N,) band gap values
        ids: (N,) material IDs
    """
    fps, gaps, ids = [], [], []
    n_fail = 0
    t0 = time.time()

    for i, rec in enumerate(records):
        try:
            fp = structure_to_fingerprint(rec["atoms"])
            fps.append(fp)
            gaps.append(rec["band_gap"])
            ids.append(rec["material_id"])
        except Exception as e:
            n_fail += 1
            if n_fail <= 5:
                log.warning(f"  FP failed for {rec['material_id']} ({rec['formula']}): {e}")

        if (i + 1) % batch_log_interval == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            log.info(f"  Fingerprinted {i+1}/{len(records)} ({rate:.0f}/s, {n_fail} failed)")

    elapsed = time.time() - t0
    log.info(f"  Done: {len(fps)} fingerprints in {elapsed:.1f}s ({n_fail} failed)")

    X = np.array(fps, dtype=np.float32)
    y_gap = np.array(gaps, dtype=np.float32)
    return X, y_gap, ids


def main():
    parser = argparse.ArgumentParser(description="Download MP data")
    parser.add_argument("--api-key", type=str, required=True, help="Materials Project API key")
    parser.add_argument("--gap-min", type=float, default=0.05)
    parser.add_argument("--gap-max", type=float, default=15.0)
    parser.add_argument("--max-atoms", type=int, default=50)
    parser.add_argument("--include-unstable", action="store_true",
                        help="Include non-stable entries (larger but noisier)")
    parser.add_argument("--cache-file", type=str, default="data/mp_fingerprints.npz")
    args = parser.parse_args()

    cache_file = Path(args.cache_file)

    if cache_file.exists():
        log.info(f"Cache exists at {cache_file}. Loading...")
        cached = np.load(cache_file, allow_pickle=True)
        X = cached["X"]
        y_gap = cached["y_gap"]
        ids = cached["ids"]
        log.info(f"  Loaded: {X.shape[0]} records, {X.shape[1]} features")
        log.info(f"  Gap range: [{y_gap.min():.3f}, {y_gap.max():.3f}], mean={y_gap.mean():.3f}")
        return

    # === Step 1: Download ===
    records = download_mp_semiconductors(
        api_key=args.api_key,
        gap_min=args.gap_min,
        gap_max=args.gap_max,
        max_atoms=args.max_atoms,
        stable_only=not args.include_unstable,
    )

    if not records:
        log.error("No records downloaded!")
        sys.exit(1)

    # Stats
    gaps = [r["band_gap"] for r in records]
    sites = [r["nsites"] for r in records]
    log.info(f"\n  Band gap stats:")
    log.info(f"    Range: [{min(gaps):.3f}, {max(gaps):.3f}] eV")
    log.info(f"    Mean:  {np.mean(gaps):.3f} eV")
    log.info(f"    Median: {np.median(gaps):.3f} eV")
    log.info(f"  Atom count stats:")
    log.info(f"    Range: [{min(sites)}, {max(sites)}]")
    log.info(f"    Mean:  {np.mean(sites):.1f}")

    # === Step 2: Compute fingerprints ===
    log.info(f"\nComputing 152-dim fingerprints for {len(records)} structures...")
    X, y_gap, ids = compute_fingerprints(records)

    # Sanitize
    valid = np.all(np.isfinite(X), axis=1) & np.isfinite(y_gap)
    n_bad = (~valid).sum()
    if n_bad > 0:
        log.warning(f"Removing {n_bad} rows with NaN/Inf")
        X = X[valid]
        y_gap = y_gap[valid]
        ids = [ids[i] for i in range(len(ids)) if valid[i]]

    # === Step 3: Cache ===
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_file,
        X=X,
        y_gap=y_gap,
        ids=np.array(ids),
    )
    log.info(f"\nCached to {cache_file}")
    log.info(f"  {X.shape[0]} records × {X.shape[1]} features")
    log.info(f"  Gap range: [{y_gap.min():.3f}, {y_gap.max():.3f}], mean={y_gap.mean():.3f}")
    log.info(f"\n✓ MP data ready for integration!")
    log.info(f"  Use with: python scripts/retrain_jarvis.py --include-bootstrap --include-mp")


if __name__ == "__main__":
    main()
