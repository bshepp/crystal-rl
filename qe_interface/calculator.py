"""QE Calculator — thin wrapper around ASE's Espresso interface.

Handles:
  - Building pw.x input from ASE Atoms
  - Running SCF, bands, DOS calculations
  - Parsing results into clean Python objects
  - Timeout and error recovery
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from ase import Atoms
from ase.calculators.espresso import Espresso, EspressoProfile
from ase.io import read as ase_read

logger = logging.getLogger(__name__)


@dataclass
class QEResult:
    """Container for a single QE calculation result."""

    energy: float  # Total energy in eV
    forces: np.ndarray  # Forces in eV/Angstrom, shape (natoms, 3)
    stress: Optional[np.ndarray] = None  # Stress tensor in eV/A^3, shape (3, 3)
    converged: bool = True
    n_scf_steps: int = 0
    fermi_energy: Optional[float] = None
    band_energies: Optional[np.ndarray] = None  # shape (nkpts, nbands)
    kpoints: Optional[np.ndarray] = None  # shape (nkpts, 3)
    error: Optional[str] = None


@dataclass
class QEConfig:
    """Configuration for QE calculations."""

    pw_command: str = "mpirun --allow-run-as-root -np 4 pw.x"
    pseudo_dir: str = "/workspace/pseudopotentials"
    scratch_dir: str = "/tmp/qe_scratch"
    ecutwfc: float = 40.0
    ecutrho: float = 320.0
    conv_thr: float = 1.0e-8
    mixing_beta: float = 0.7
    occupations: str = "smearing"
    smearing: str = "cold"
    degauss: float = 0.02
    kpoints: tuple[int, int, int] = (6, 6, 6)
    timeout: int = 3600  # seconds


class QECalculator:
    """Manages Quantum ESPRESSO calculations via ASE."""

    # Map element symbols to pseudopotential filenames (SSSP Efficiency 1.3)
    PSEUDO_MAP = {
        "H": "H.pbe-rrkjus_psl.1.0.0.UPF",
        "C": "C.pbe-n-kjpaw_psl.1.0.0.UPF",
        "N": "N.pbe-n-kjpaw_psl.1.0.0.UPF",
        "O": "O.pbe-n-kjpaw_psl.1.0.0.UPF",
        "Si": "Si.pbe-n-rrkjus_psl.1.0.0.UPF",
        "P": "P.pbe-n-rrkjus_psl.1.0.0.UPF",
        "Ge": "Ge.pbe-dn-kjpaw_psl.1.0.0.UPF",
        "Ga": "Ga.pbe-dn-kjpaw_psl.1.0.0.UPF",
        "As": "As.pbe-n-rrkjus_psl.0.2.UPF",
        "In": "In.pbe-dn-rrkjus_psl.1.0.0.UPF",
        "Sn": "Sn.pbe-dn-kjpaw_psl.1.0.0.UPF",
        "Al": "Al.pbe-n-kjpaw_psl.1.0.0.UPF",
    }

    # Valence electrons per element in the SSSP pseudopotentials above
    VALENCE_ELECTRONS = {
        "H": 1,
        "C": 4,
        "N": 5,
        "O": 6,
        "Si": 4,
        "P": 5,
        "Ge": 14,  # Ge.pbe-dn: 3d¹⁰ 4s² 4p² (d in valence)
        "Ga": 13,  # Ga.pbe-dn: 3d¹⁰ 4s² 4p¹
        "As": 5,
        "In": 13,  # In.pbe-dn: 4d¹⁰ 5s² 5p¹
        "Sn": 14,  # Sn.pbe-dn: 4d¹⁰ 5s² 5p²
        "Al": 3,
    }

    def get_n_valence_electrons(self, atoms: Atoms) -> int:
        """Return the total number of valence electrons for the pseudopotentials."""
        symbols = atoms.get_chemical_symbols()
        total = 0
        for s in symbols:
            if s in self.VALENCE_ELECTRONS:
                total += self.VALENCE_ELECTRONS[s]
            else:
                raise ValueError(
                    f"No valence electron count for element '{s}'. "
                    f"Add it to QECalculator.VALENCE_ELECTRONS"
                )
        return total

    def __init__(self, config: Optional[QEConfig] = None):
        self.config = config or QEConfig()
        os.makedirs(self.config.scratch_dir, exist_ok=True)

    def _get_pseudopotentials(self, atoms: Atoms) -> dict[str, str]:
        """Get pseudopotential file mapping for all species in the structure."""
        species = set(atoms.get_chemical_symbols())
        pseudos = {}
        for s in species:
            if s in self.PSEUDO_MAP:
                pseudos[s] = self.PSEUDO_MAP[s]
            else:
                raise ValueError(
                    f"No pseudopotential configured for element '{s}'. "
                    f"Add it to QECalculator.PSEUDO_MAP"
                )
        return pseudos

    def _make_input_data(self, calculation: str = "scf") -> dict:
        """Build the input_data dict for ASE's Espresso calculator."""
        cfg = self.config
        return {
            "control": {
                "calculation": calculation,
                "restart_mode": "from_scratch",
                "pseudo_dir": cfg.pseudo_dir,
                "outdir": cfg.scratch_dir,
                "tprnfor": True,
                "tstress": True,
                "verbosity": "high",
            },
            "system": {
                "ecutwfc": cfg.ecutwfc,
                "ecutrho": cfg.ecutrho,
                "occupations": cfg.occupations,
                "smearing": cfg.smearing,
                "degauss": cfg.degauss,
            },
            "electrons": {
                "conv_thr": cfg.conv_thr,
                "mixing_beta": cfg.mixing_beta,
                "electron_maxstep": 200,
            },
        }

    def _build_calculator(self, atoms: Atoms, calculation: str = "scf") -> Espresso:
        """Create an ASE Espresso calculator for the given structure."""
        pseudos = self._get_pseudopotentials(atoms)
        input_data = self._make_input_data(calculation)
        kpts = self.config.kpoints

        profile = EspressoProfile(
            command=self.config.pw_command,
            pseudo_dir=self.config.pseudo_dir,
        )

        calc = Espresso(
            profile=profile,
            input_data=input_data,
            pseudopotentials=pseudos,
            kpts=kpts,
        )
        return calc

    def run_scf(self, atoms: Atoms) -> QEResult:
        """Run a self-consistent field calculation.

        Args:
            atoms: ASE Atoms object with the structure to calculate.

        Returns:
            QEResult with energy, forces, stress.
        """
        logger.info(
            f"Running SCF: {atoms.get_chemical_formula()} "
            f"({len(atoms)} atoms, {atoms.cell.cellpar()[:3].round(2)} A)"
        )

        calc = self._build_calculator(atoms, "scf")
        atoms_copy = atoms.copy()
        atoms_copy.calc = calc

        try:
            energy = atoms_copy.get_potential_energy()
            forces = atoms_copy.get_forces()
            stress = atoms_copy.get_stress(voigt=False)

            result = QEResult(
                energy=energy,
                forces=forces,
                stress=stress,
                converged=True,
            )
            logger.info(f"SCF converged: E = {energy:.6f} eV")
            return result

        except Exception as e:
            logger.error(f"SCF failed: {e}")
            return QEResult(
                energy=0.0,
                forces=np.zeros((len(atoms), 3)),
                converged=False,
                error=str(e),
            )

    def run_relax(self, atoms: Atoms, fmax: float = 0.05) -> tuple[Atoms, QEResult]:
        """Run ionic relaxation.

        Args:
            atoms: Initial structure.
            fmax: Force convergence threshold in eV/A.

        Returns:
            Tuple of (relaxed Atoms, QEResult).
        """
        logger.info(f"Running relaxation: {atoms.get_chemical_formula()}")

        input_data = self._make_input_data("relax")
        input_data["control"]["forc_conv_thr"] = fmax * 0.0194469  # eV/A to Ry/bohr
        pseudos = self._get_pseudopotentials(atoms)

        profile = EspressoProfile(
            command=self.config.pw_command,
            pseudo_dir=self.config.pseudo_dir,
        )

        calc = Espresso(
            profile=profile,
            input_data=input_data,
            pseudopotentials=pseudos,
            kpts=self.config.kpoints,
        )

        atoms_copy = atoms.copy()
        atoms_copy.calc = calc

        try:
            energy = atoms_copy.get_potential_energy()
            forces = atoms_copy.get_forces()

            result = QEResult(
                energy=energy,
                forces=forces,
                converged=True,
            )
            logger.info(f"Relaxation done: E = {energy:.6f} eV")
            return atoms_copy, result

        except Exception as e:
            logger.error(f"Relaxation failed: {e}")
            result = QEResult(
                energy=0.0,
                forces=np.zeros((len(atoms), 3)),
                converged=False,
                error=str(e),
            )
            return atoms, result

    def run_bands(
        self, atoms: Atoms, kpath: Optional[np.ndarray] = None, npoints: int = 40
    ) -> QEResult:
        """Run SCF + band structure calculation.

        Two-step QE workflow:
          1. SCF on standard k-mesh → charge density
          2. NSCF bands on k-path → eigenvalues

        ASE doesn't natively chain these, so we manage the working directory
        manually and parse QE output files.

        Args:
            atoms: Structure (should be pre-relaxed).
            kpath: Custom k-point path. If None, uses ASE's default for the lattice.
            npoints: Points per segment of the band path.

        Returns:
            QEResult with band_energies and kpoints populated.
        """
        import subprocess

        # Create a temporary working directory for the two-step calc
        work_dir = tempfile.mkdtemp(prefix="qe_bands_", dir=self.config.scratch_dir)
        outdir = os.path.join(work_dir, "out")
        os.makedirs(outdir, exist_ok=True)

        pseudos = self._get_pseudopotentials(atoms)

        # ----- Step 1: SCF -----
        logger.info("Bands step 1/2: SCF on uniform k-mesh")
        scf_input = self._make_input_data("scf")
        scf_input["control"]["outdir"] = outdir
        scf_input["control"]["prefix"] = "bands"

        profile = EspressoProfile(
            command=self.config.pw_command,
            pseudo_dir=self.config.pseudo_dir,
        )

        scf_calc = Espresso(
            profile=profile,
            input_data=scf_input,
            pseudopotentials=pseudos,
            kpts=self.config.kpoints,
            directory=work_dir,
        )

        atoms_scf = atoms.copy()
        atoms_scf.calc = scf_calc

        try:
            scf_energy = atoms_scf.get_potential_energy()
            scf_forces = atoms_scf.get_forces()
        except Exception as e:
            logger.error(f"Bands SCF step failed: {e}")
            shutil.rmtree(work_dir, ignore_errors=True)
            return QEResult(
                energy=0.0,
                forces=np.zeros((len(atoms), 3)),
                converged=False,
                error=f"Bands SCF failed: {e}",
            )

        logger.info(f"SCF done: E = {scf_energy:.6f} eV")

        # ----- Step 2: NSCF bands -----
        logger.info("Bands step 2/2: NSCF on k-path")

        if kpath is None:
            path_obj = atoms.cell.bandpath(npoints=npoints)
            kpts_3 = path_obj.kpts  # shape (nk, 3) in crystal coords
        else:
            kpts_3 = np.asarray(kpath)

        # ASE Espresso only accepts Nx4 arrays (k + weight).
        # For band structures all k-points get equal weight.
        if kpts_3.ndim == 2 and kpts_3.shape[1] == 3:
            weights = np.ones((len(kpts_3), 1)) / len(kpts_3)
            kpts = np.hstack([kpts_3, weights])
        else:
            kpts = kpts_3  # already Nx4

        bands_input = self._make_input_data("bands")
        bands_input["control"]["outdir"] = outdir
        bands_input["control"]["prefix"] = "bands"
        bands_input["system"]["nbnd"] = int(max(
            len(atoms) * 4,  # enough bands above Fermi level
            sum(atoms.get_atomic_numbers()) // 2 + 8
        ))

        try:
            import subprocess

            # Write the QE bands input file manually.
            # ASE's Espresso calculator doesn't handle bands mode well.
            pwi_path = os.path.join(work_dir, "espresso.pwi")
            pwo_path = os.path.join(work_dir, "espresso.pwo")

            self._write_bands_input(
                pwi_path, atoms, pseudos, bands_input, kpts_3
            )

            cmd = self.config.pw_command.split() + ["-in", "espresso.pwi"]
            logger.info(f"Running bands NSCF: {' '.join(cmd)}")

            with open(pwo_path, "w") as pwo_file:
                proc = subprocess.run(
                    cmd,
                    cwd=work_dir,
                    stdout=pwo_file,
                    stderr=subprocess.PIPE,
                    timeout=600,
                )

            if proc.returncode != 0:
                stderr_text = proc.stderr.decode() if proc.stderr else ""
                logger.error(f"NSCF pw.x failed (rc={proc.returncode}): {stderr_text[:500]}")
                return QEResult(
                    energy=scf_energy,
                    forces=scf_forces,
                    converged=True,
                    error=f"NSCF pw.x failed: {stderr_text[:200]}",
                    kpoints=kpts_3,
                )

            # Extract band energies from QE output
            band_energies = self._parse_bands_output(pwo_path)

            result = QEResult(
                energy=scf_energy,
                forces=scf_forces,
                converged=True,
                kpoints=kpts_3,  # Nx3 crystal coords (no weights)
                band_energies=band_energies,
            )

            if band_energies is not None:
                logger.info(
                    f"Bands done: {band_energies.shape[0]} k-points, "
                    f"{band_energies.shape[1]} bands"
                )
            else:
                logger.warning("Bands calculation ran but eigenvalues not parsed")

            return result

        except Exception as e:
            logger.error(f"Bands NSCF step failed: {e}")
            # Try to log the QE output for debugging
            pwo = os.path.join(work_dir, "espresso.pwo")
            pwi = os.path.join(work_dir, "espresso.pwi")
            if os.path.exists(pwo):
                with open(pwo) as f:
                    logger.error(f"QE NSCF output (last 50 lines):\n{''.join(f.readlines()[-50:])}")
            if os.path.exists(pwi):
                with open(pwi) as f:
                    logger.error(f"QE NSCF input:\n{f.read()}")
            # Return SCF result at least
            return QEResult(
                energy=scf_energy,
                forces=scf_forces,
                converged=True,
                error=f"NSCF bands failed: {e}",
                kpoints=kpts_3,
            )
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    @staticmethod
    def _write_bands_input(
        filepath: str,
        atoms: Atoms,
        pseudos: dict,
        input_data: dict,
        kpoints: np.ndarray,
    ) -> None:
        """Write a QE pw.x input file for a bands calculation.

        Generates the file manually since ASE doesn't handle bands I/O well.

        Args:
            filepath: Path to write the .pwi file.
            atoms: Crystal structure.
            pseudos: Dict mapping element symbol → pseudo filename.
            input_data: Nested dict with control/system/electrons settings.
            kpoints: k-point array, shape (nk, 3) in crystal coordinates.
        """
        from ase.data import atomic_masses, atomic_numbers

        lines = []

        # Namelists
        for namelist in ["control", "system", "electrons", "ions", "cell"]:
            section = input_data.get(namelist, {})
            lines.append(f"&{namelist.upper()}")
            for key, val in section.items():
                if isinstance(val, bool):
                    fval = ".true." if val else ".false."
                elif isinstance(val, str):
                    fval = f"'{val}'"
                elif isinstance(val, float):
                    fval = f"{val}"
                elif isinstance(val, int):
                    fval = f"{val}"
                else:
                    fval = str(val)
                lines.append(f"   {key} = {fval}")

            # Add ntyp and nat to SYSTEM
            if namelist == "system":
                symbols = list(set(atoms.get_chemical_symbols()))
                if "ntyp" not in section:
                    lines.append(f"   ntyp = {len(symbols)}")
                if "nat" not in section:
                    lines.append(f"   nat = {len(atoms)}")
                if "ibrav" not in section:
                    lines.append("   ibrav = 0")

            lines.append("/")

        # Atomic species
        lines.append("ATOMIC_SPECIES")
        symbols = sorted(set(atoms.get_chemical_symbols()))
        for sym in symbols:
            mass = atomic_masses[atomic_numbers[sym]]
            pseudo_file = pseudos[sym]
            lines.append(f"{sym} {mass:.3f} {pseudo_file}")

        # K-points in crystal coordinates
        lines.append(f"K_POINTS {{crystal}}")
        lines.append(f"{len(kpoints)}")
        weight = 1.0 / len(kpoints)
        for kpt in kpoints:
            lines.append(f"{kpt[0]:.14f} {kpt[1]:.14f} {kpt[2]:.14f} {weight:.14f}")

        # Cell parameters
        lines.append("CELL_PARAMETERS angstrom")
        cell = atoms.cell[:]
        for row in cell:
            lines.append(f"{row[0]:.14f} {row[1]:.14f} {row[2]:.14f}")

        # Atomic positions
        lines.append("ATOMIC_POSITIONS angstrom")
        for sym, pos in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
            lines.append(f"{sym} {pos[0]:.10f} {pos[1]:.10f} {pos[2]:.10f}")

        with open(filepath, "w") as f:
            f.write("\n".join(lines) + "\n")

    @staticmethod
    def _parse_bands_output(pwo_path: str) -> Optional[np.ndarray]:
        """Parse eigenvalues from a QE .pwo output file.

        In QE 7.x bands output, after "End of band structure calculation",
        each k-point block looks like:

              k = 0.0000 0.0000 0.0000 (   749 PWs)   bands (ev):

          -5.7234   6.2498   6.2498   6.2498   8.7948  ...
          13.9603  13.9603  ...

        Returns:
            Array of shape (n_kpoints, n_bands) in eV, or None if parsing fails.
        """
        try:
            with open(pwo_path, "r") as f:
                lines = f.readlines()
        except FileNotFoundError:
            logger.error(f"QE output not found: {pwo_path}")
            return None

        # Find "End of band structure calculation" marker
        start_idx = 0
        for i, line in enumerate(lines):
            if "End of band structure calculation" in line:
                start_idx = i
                break

        kpoint_eigenvalues = []
        i = start_idx

        while i < len(lines):
            line = lines[i].strip()

            # QE 7.x format: k = ... bands (ev): on the SAME line
            if "k =" in line and "bands (ev)" in line:
                i += 1
                # Skip blank lines after the header
                while i < len(lines) and lines[i].strip() == "":
                    i += 1
                # Read eigenvalue lines until blank line or non-numeric
                eigenvals = []
                while i < len(lines) and lines[i].strip():
                    vals = lines[i].split()
                    try:
                        eigenvals.extend([float(v) for v in vals])
                    except ValueError:
                        break
                    i += 1
                if eigenvals:
                    kpoint_eigenvalues.append(eigenvals)
                continue

            i += 1

        if not kpoint_eigenvalues:
            logger.warning("No eigenvalues found in QE output")
            return None

        # Ensure all k-points have same number of bands
        n_bands = len(kpoint_eigenvalues[0])
        valid = [ev for ev in kpoint_eigenvalues if len(ev) == n_bands]

        if len(valid) != len(kpoint_eigenvalues):
            logger.warning(
                f"Inconsistent band counts: {len(valid)}/{len(kpoint_eigenvalues)} k-points"
            )

        if not valid:
            return None

        logger.info(f"Parsed {len(valid)} k-points × {n_bands} bands from QE output")
        return np.array(valid)
