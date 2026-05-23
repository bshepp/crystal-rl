#!/usr/bin/env python3
"""Pad the cached bootstrap fingerprints from the old 12-element palette
(152-dim total) to the current 16-element palette (156-dim total).

The fingerprint layout is:
  [12 or 16] composition  +  4 elemental  +  8 lattice  +  64 RDF  +  64 partial-RDF

The only structural change going from the 12-element to the 16-element palette
is in the composition section: 4 extra fractional slots appended for
Sb / Bi / Se / Te. Every bootstrap structure was generated from the original
10-element seed set (Si, Ge, C, Ga, As, Al, In, P, N, Sn — plus rare oxide/H
records pulled in by ASE) and contains zero atoms of the new elements, so the
4 extra composition slots are correctly 0.0 for every bootstrap record.

Zero-padding the cached 152-dim fingerprints at indices 12..15 therefore
produces an exact match for what `structure_to_fingerprint` would compute
today — no need to reconstruct ASE Atoms or rerun the fingerprint code.

Usage:
    python -m scripts.regen_bootstrap_for_palette
        [--in  data/bootstrap/surrogate_multitask/surrogate_data.npz]
        [--out data/bootstrap/surrogate_multitask/surrogate_data.npz]
        [--backup-suffix _152dim]

By default it operates in-place on the canonical loader path, moving the
old file to a `_152dim` suffix before writing the new one. Safe to re-run:
if the input is already 156-dim, the script reports and exits.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

# --- Layout constants (must match qe_interface/structures.py) ---
OLD_COMPOSITION_DIM = 12
NEW_COMPOSITION_DIM = 16
PAD_INSERT_POSITION = OLD_COMPOSITION_DIM  # insert 4 zeros at idx 12..15
PAD_AMOUNT = NEW_COMPOSITION_DIM - OLD_COMPOSITION_DIM  # 4
NON_COMPOSITION_DIM = 4 + 8 + 64 + 64  # elemental + lattice + RDF + partial-RDF
EXPECTED_OLD_DIM = OLD_COMPOSITION_DIM + NON_COMPOSITION_DIM  # 152
EXPECTED_NEW_DIM = NEW_COMPOSITION_DIM + NON_COMPOSITION_DIM  # 156


def pad_fingerprints(fps: np.ndarray) -> np.ndarray:
    """Insert 4 zero columns at index 12 to convert 152-dim → 156-dim."""
    if fps.shape[1] != EXPECTED_OLD_DIM:
        raise ValueError(
            f"Expected input fingerprint dim {EXPECTED_OLD_DIM}, "
            f"got {fps.shape[1]}"
        )
    zeros = np.zeros((fps.shape[0], PAD_AMOUNT), dtype=fps.dtype)
    return np.hstack([
        fps[:, :PAD_INSERT_POSITION],   # 12 old composition cols
        zeros,                           # 4 new Sb/Bi/Se/Te cols (all 0)
        fps[:, PAD_INSERT_POSITION:],   # rest unchanged
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--in", dest="input_path",
        default="data/bootstrap/surrogate_multitask/surrogate_data.npz",
        help="Path to the 152-dim cached bootstrap data",
    )
    parser.add_argument(
        "--out", dest="output_path",
        default="data/bootstrap/surrogate_multitask/surrogate_data.npz",
        help="Path to write the 156-dim padded bootstrap data",
    )
    parser.add_argument(
        "--backup-suffix", default="_152dim",
        help="Suffix for the backup of the original 152-dim file",
    )
    args = parser.parse_args()

    in_path = Path(args.input_path)
    out_path = Path(args.output_path)

    if not in_path.exists():
        print(f"INPUT NOT FOUND: {in_path}", file=sys.stderr)
        sys.exit(1)

    data = np.load(in_path)
    fps = data["fingerprints"]
    print(f"Loaded {in_path}")
    print(f"  fingerprints: shape={fps.shape}, dtype={fps.dtype}")

    if fps.shape[1] == EXPECTED_NEW_DIM:
        print(f"  Already {EXPECTED_NEW_DIM}-dim; nothing to do.")
        return

    if fps.shape[1] != EXPECTED_OLD_DIM:
        print(f"  Unexpected dim {fps.shape[1]}; expected "
              f"{EXPECTED_OLD_DIM} or {EXPECTED_NEW_DIM}", file=sys.stderr)
        sys.exit(2)

    fps_padded = pad_fingerprints(fps)
    print(f"  Padded to: shape={fps_padded.shape}")

    # Surface basic stats so the operator can sanity-check signed m\*
    if "targets_mstar" in data:
        y_m = data["targets_mstar"]
        n_neg = int((y_m < 0).sum())
        n_pos = int((y_m > 0).sum())
        print(f"  targets_mstar: {len(y_m)} records "
              f"({n_neg} negative, {n_pos} positive), range "
              f"[{y_m.min():.3f}, {y_m.max():.3f}]")

    # Back up the original before overwriting (only if in_path == out_path)
    if in_path.resolve() == out_path.resolve():
        backup_path = in_path.with_name(in_path.stem + args.backup_suffix + ".npz")
        if not backup_path.exists():
            print(f"  Backing up original -> {backup_path}")
            shutil.copy2(in_path, backup_path)
        else:
            print(f"  Backup already exists at {backup_path} (leaving as-is)")

    # Build the output dict (preserve every key from the input)
    out_dict = {k: data[k] for k in data.keys()}
    out_dict["fingerprints"] = fps_padded

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **out_dict)
    print(f"Wrote {out_path} ({EXPECTED_NEW_DIM}-dim, {fps_padded.shape[0]} records)")


if __name__ == "__main__":
    main()
