#!/usr/bin/env python3
"""Assemble the rl-surrogate-dft-gap Hugging Face dataset.

Produces three artifacts in --output-dir, ready for `hf upload`:

  data/bootstrap_signed_mstar.parquet
      The 794-record bootstrap DFT dataset with SIGNED effective mass
      (681 negative, 113 positive). Each row carries the raw QE outputs
      (m_electron, m_hole, m_min, band_gap, vbm, cbm, is_direct) plus
      the 156-dim structural fingerprint produced by the current
      pipeline (16-element composition + RDF + lattice features).

      Signed-m* DFT data is rare — JARVIS, Materials Project, and AFLOW
      all store magnitudes only. This is the primary reusable artifact.

  data/surrogate_dft_pairs.parquet
      24 surrogate-prediction-vs-DFT-truth pairs across three independent
      RL+surrogate validation rounds (Phase 7 unsigned baseline, path-B
      run #1 without bootstrap loaded, path-B run #3 with bootstrap
      loaded). This is the dataset that grounds the negative-result
      methodology claim: surrogate-predicted "low m*" candidates
      consistently fail to validate as low-positive-m* semiconductors
      under DFT, across runs with very different surrogate qualities.

  README.md
      Dataset card with YAML frontmatter, written by hand into
      hf_dataset_card.md and copied alongside the parquet files.

Usage:
    python -m scripts.build_hf_dataset \
        --bootstrap-json data/bootstrap/bootstrap_expanded_all.json \
        --bootstrap-npz  data/bootstrap/surrogate_multitask/surrogate_data.npz \
        --validation phase7=data/validation/validation_report.json \
        --validation pathb_run1=/path/to/run1_validation.json \
        --validation pathb_run3=/path/to/run3_validation.json \
        --output-dir /tmp/hf-dataset-staging
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_bootstrap(json_path: Path, npz_path: Path) -> pd.DataFrame:
    """Merge the rich per-record JSON with the 156-dim fingerprints."""
    with open(json_path) as f:
        records = json.load(f)

    npz = np.load(npz_path)
    fps = npz["fingerprints"]
    targets_mstar = npz["targets_mstar"]
    targets_gap = npz["targets_gap"]

    if fps.shape[0] != len(records):
        raise SystemExit(
            f"Row count mismatch: JSON has {len(records)} records but "
            f"NPZ has {fps.shape[0]} fingerprints. Aborting — the two files "
            f"must align row-for-row."
        )
    if fps.shape[1] != 156:
        raise SystemExit(
            f"Fingerprint dim is {fps.shape[1]}, expected 156. "
            f"Run scripts/regen_bootstrap_for_palette.py first."
        )

    # Sanity-check: the m_min in the JSON should match targets_mstar in the NPZ
    # (within float precision). The NPZ targets ARE the canonical signed m* used
    # for surrogate training.
    json_m_min = np.array([r["m_min"] for r in records], dtype=float)
    if not np.allclose(json_m_min, targets_mstar, atol=1e-3):
        print(
            f"WARNING: JSON m_min and NPZ targets_mstar differ by up to "
            f"{np.max(np.abs(json_m_min - targets_mstar)):.4f}. Using NPZ "
            f"targets as canonical.",
            file=sys.stderr,
        )

    rows = []
    for i, rec in enumerate(records):
        rows.append({
            "row_id": i,
            "seed": rec.get("seed", ""),
            "label": rec.get("label", ""),
            "formula": rec.get("formula", ""),
            "dft_m_electron": float(rec["m_electron"]),
            "dft_m_hole": float(rec["m_hole"]),
            "dft_m_min_signed": float(targets_mstar[i]),
            "dft_band_gap_ev": float(rec["band_gap"]),
            "dft_vbm_ev": float(rec["vbm"]),
            "dft_cbm_ev": float(rec["cbm"]),
            "is_direct_gap": bool(rec["is_direct"]) if rec["is_direct"] is not None else None,
            "qe_runtime_seconds": float(rec["time_s"]),
            "fingerprint_156": fps[i].tolist(),
        })

    df = pd.DataFrame(rows)
    print(f"bootstrap_signed_mstar: {len(df)} records, "
          f"{int((df['dft_m_min_signed'] < 0).sum())} negative m*, "
          f"{int((df['dft_m_min_signed'] > 0).sum())} positive m*")
    return df


def load_validation_pairs(named_paths: dict[str, Path]) -> pd.DataFrame:
    """Concatenate the per-run validation_report.json files into one table."""
    all_rows = []
    for run_id, path in named_paths.items():
        with open(path) as f:
            records = json.load(f)
        for rec in records:
            all_rows.append({
                "run_id": run_id,
                "formula": rec["formula"],
                "seed": rec.get("seed", "?"),
                "surrogate_m_star": float(rec["surrogate_m_star"]),
                "surrogate_reward": float(rec["surrogate_reward"]),
                "dft_m_min_signed": float(rec["dft_m_min"]),
                "dft_m_electron": float(rec["dft_m_electron"]),
                "dft_m_hole": float(rec["dft_m_hole"]),
                "dft_band_gap_ev": float(rec["dft_band_gap"]),
                "dft_converged": bool(rec["dft_converged"]),
                "dft_runtime_seconds": float(rec["dft_time_s"]),
                "unusual_topology": bool(rec.get("unusual_topology", False)),
                # Computed convenience field for the negative-result narrative
                "is_low_positive_semiconductor": (
                    rec["dft_m_min"] > 0
                    and rec["dft_m_min"] < 0.5
                    and rec["dft_band_gap"] > 0.1
                ),
            })

    df = pd.DataFrame(all_rows)
    n_low = int(df["is_low_positive_semiconductor"].sum())
    n_topo = int(df["unusual_topology"].sum())
    print(f"surrogate_dft_pairs: {len(df)} records across "
          f"{df['run_id'].nunique()} runs")
    print(f"  -> {n_low} DFT-validated low-positive-m* semiconductors")
    print(f"  -> {n_topo} with band-inversion signature")
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bootstrap-json", required=True, type=Path)
    parser.add_argument("--bootstrap-npz", required=True, type=Path)
    parser.add_argument("--validation", action="append", required=True,
                        metavar="run_id=path",
                        help="Validation report; specify --validation NAME=PATH "
                             "for each run to include (repeatable).")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = args.output_dir / "data"
    data_dir.mkdir(exist_ok=True)

    # Parse the --validation key=path arguments
    named_paths = {}
    for v in args.validation:
        if "=" not in v:
            raise SystemExit(f"Bad --validation format (expected NAME=PATH): {v}")
        name, p = v.split("=", 1)
        named_paths[name] = Path(p)

    print("Building bootstrap_signed_mstar.parquet ...")
    boot_df = load_bootstrap(args.bootstrap_json, args.bootstrap_npz)
    boot_out = data_dir / "bootstrap_signed_mstar.parquet"
    boot_df.to_parquet(boot_out, index=False, compression="snappy")
    print(f"  -> {boot_out} ({boot_out.stat().st_size / 1024:.1f} KB)")

    print("\nBuilding surrogate_dft_pairs.parquet ...")
    pairs_df = load_validation_pairs(named_paths)
    pairs_out = data_dir / "surrogate_dft_pairs.parquet"
    pairs_df.to_parquet(pairs_out, index=False, compression="snappy")
    print(f"  -> {pairs_out} ({pairs_out.stat().st_size / 1024:.1f} KB)")

    print(f"\nDataset assembled at {args.output_dir}/")
    print("  data/bootstrap_signed_mstar.parquet")
    print("  data/surrogate_dft_pairs.parquet")
    print("Add README.md and run:")
    print(f"  hf upload bshepp/rl-surrogate-dft-gap {args.output_dir} . "
          "--repo-type dataset")


if __name__ == "__main__":
    main()
