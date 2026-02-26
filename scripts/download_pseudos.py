#!/usr/bin/env python3
"""Download SSSP Efficiency pseudopotentials for the target species palette.

Downloads from the SSSP library (Materials Cloud) into the pseudopotentials/ directory.
These are needed by QE for DFT calculations.

Usage:
    python scripts/download_pseudos.py
"""

from __future__ import annotations

import logging
import sys
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# SSSP Efficiency 1.3.0 pseudopotentials
# Source: https://www.materialscloud.org/discover/sssp/table/efficiency
PSEUDOS = {
    "Si.pbe-n-rrkjus_psl.1.0.0.UPF": "https://pseudopotentials.quantum-espresso.org/upf_files/Si.pbe-n-rrkjus_psl.1.0.0.UPF",
    "Ge.pbe-dn-kjpaw_psl.1.0.0.UPF": "https://pseudopotentials.quantum-espresso.org/upf_files/Ge.pbe-dn-kjpaw_psl.1.0.0.UPF",
    "C.pbe-n-kjpaw_psl.1.0.0.UPF": "https://pseudopotentials.quantum-espresso.org/upf_files/C.pbe-n-kjpaw_psl.1.0.0.UPF",
    "Sn.pbe-dn-kjpaw_psl.1.0.0.UPF": "https://pseudopotentials.quantum-espresso.org/upf_files/Sn.pbe-dn-kjpaw_psl.1.0.0.UPF",
    "N.pbe-n-kjpaw_psl.1.0.0.UPF": "https://pseudopotentials.quantum-espresso.org/upf_files/N.pbe-n-kjpaw_psl.1.0.0.UPF",
    "P.pbe-n-rrkjus_psl.1.0.0.UPF": "https://pseudopotentials.quantum-espresso.org/upf_files/P.pbe-n-rrkjus_psl.1.0.0.UPF",
    "As.pbe-n-rrkjus_psl.0.2.UPF": "https://pseudopotentials.quantum-espresso.org/upf_files/As.pbe-n-rrkjus_psl.0.2.UPF",
    "Ga.pbe-dn-kjpaw_psl.1.0.0.UPF": "https://pseudopotentials.quantum-espresso.org/upf_files/Ga.pbe-dn-kjpaw_psl.1.0.0.UPF",
    "In.pbe-dn-rrkjus_psl.1.0.0.UPF": "https://pseudopotentials.quantum-espresso.org/upf_files/In.pbe-dn-rrkjus_psl.1.0.0.UPF",
    "Al.pbe-n-kjpaw_psl.1.0.0.UPF": "https://pseudopotentials.quantum-espresso.org/upf_files/Al.pbe-n-kjpaw_psl.1.0.0.UPF",
    "B.pbe-n-kjpaw_psl.1.0.0.UPF": "https://pseudopotentials.quantum-espresso.org/upf_files/B.pbe-n-kjpaw_psl.1.0.0.UPF",
    "O.pbe-n-kjpaw_psl.1.0.0.UPF": "https://pseudopotentials.quantum-espresso.org/upf_files/O.pbe-n-kjpaw_psl.1.0.0.UPF",
    # Extended palette: V-group pnictides and VI-group chalcogenides
    "Sb.pbe-n-kjpaw_psl.1.0.0.UPF": "https://pseudopotentials.quantum-espresso.org/upf_files/Sb.pbe-n-kjpaw_psl.1.0.0.UPF",
    "Bi.pbe-dn-kjpaw_psl.1.0.0.UPF": "https://pseudopotentials.quantum-espresso.org/upf_files/Bi.pbe-dn-kjpaw_psl.1.0.0.UPF",
    "Se.pbe-dn-kjpaw_psl.1.0.0.UPF": "https://pseudopotentials.quantum-espresso.org/upf_files/Se.pbe-dn-kjpaw_psl.1.0.0.UPF",
    "Te.pbe-n-rrkjus_psl.1.0.0.UPF": "https://pseudopotentials.quantum-espresso.org/upf_files/Te.pbe-n-rrkjus_psl.1.0.0.UPF",
}


def download_pseudos(output_dir: str | Path = "pseudopotentials") -> None:
    """Download all pseudopotentials to the specified directory."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    n_downloaded = 0
    n_skipped = 0
    n_failed = 0

    for filename, url in PSEUDOS.items():
        filepath = out / filename

        if filepath.exists() and filepath.stat().st_size > 0:
            logger.info(f"  Already exists: {filename}")
            n_skipped += 1
            continue

        logger.info(f"  Downloading: {filename}...")
        try:
            urllib.request.urlretrieve(url, filepath)
            size_kb = filepath.stat().st_size / 1024
            logger.info(f"    OK ({size_kb:.0f} KB)")
            n_downloaded += 1
        except Exception as e:
            logger.error(f"    FAILED: {e}")
            n_failed += 1

    logger.info(f"\nDone: {n_downloaded} downloaded, {n_skipped} skipped, {n_failed} failed")


if __name__ == "__main__":
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "pseudopotentials"
    logger.info(f"Downloading SSSP pseudopotentials to {target_dir}/")
    download_pseudos(target_dir)
