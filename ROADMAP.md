# RL-Materials Project Roadmap

## Current Status (Feb 2026)

**Phase: Surrogate Expansion & PPO Optimization**

### Completed Milestones

#### Phase 1: Bootstrap DFT (✅ Complete)
- 794-point bootstrap dataset from AWS EC2 (c5.4xlarge, instance i-0d3ff917412bd0f51)
- Results uploaded to S3 bucket `rl-materials-bootstrap-290318879194`
- 10 crystal seed families: Si, Ge, C-diamond, GaAs, AlAs, InAs, GaP, SiC-3C, InP, AlN
- 152-dim structural fingerprints (12 comp + 4 elem + 8 lattice + 64 RDF + 64 partial_RDF)

#### Phase 2: JARVIS-DFT Integration (✅ Complete)
- Integrated JARVIS dft_3d dataset: 75,993 structures, 3,565 with effective mass
- Combined with 794 bootstrap records → 4,359 training samples
- Trained MultiTaskMLP (shared trunk 192×4 + m*/gap heads): m* corr=0.625, gap acc=90.4%

#### Phase 3: Materials Project Integration (✅ Complete)
- Downloaded 12,990 stable MP semiconductors (gap 0.05–11.69 eV, ≤50 atoms)
- Computed 152-dim fingerprints for all 12,990 with 0 failures (cached in `data/mp_fingerprints.npz`)
- MP and AFLOW confirmed to have NO pre-computed effective mass fields
- MP used as gap-only augmentation data

#### Phase 4: Two-Phase Surrogate Training (✅ Complete)
- **Phase 1:** Train full model on JARVIS+bootstrap only (preserves m* signal)
- **Phase 2:** Freeze trunk+m* head, fine-tune gap head with 4k MP records
- Result: **m* corr=0.886, gap acc=95.2%** (vs baseline 0.625 / 90.4%)
- Model saved: `data/checkpoints/jarvis_surrogate/` (141,890 params, 8,359 records)

#### Phase 5: PPO Training with Improved Surrogate (✅ Complete)
- 250k timesteps, 6,259 episodes in 105 min
- Q1→Q4 reward: 183.6 → 585.4 (+401.8)
- **Eval mean: 655.4** (vs prior 478.3 — 37% improvement)
- 20/20 evaluation episodes produced semiconductors
- Best single episode: 2873.5
- Model saved: `data/checkpoints/ppo_jarvis/ppo_final.zip`

### Next Steps

#### Phase 6: SNUMAT Integration (Planned)
- SNUMAT database has ~10,000 HSE06 effective mass records
- This is actual m* data (not gap-only like MP) — would directly improve m* head
- Requires web download from snumat.com, CIF → ASE conversion

#### Phase 7: DFT Validation of Discovered Candidates
- Run Quantum ESPRESSO on top PPO-discovered structures
- Compare surrogate predictions vs actual DFT band gap and effective mass
- Quantify prediction accuracy on novel (out-of-distribution) materials

---

## PLANNED: Quantum Validation via Amazon Braket

> **Why this matters:** Our DFT calculations use the PBE functional, which is an
> approximation. For some compound families (especially InN, AlN, and other
> nitrogen-containing III-V materials that our RL agent favors), PBE can have
> meaningful errors. A quantum computer can solve the *exact* electronic
> structure for small systems — no functional approximation needed.

### The Idea

Use VQE (Variational Quantum Eigensolver) on small molecular fragments of our
top RL-discovered compounds to cross-validate DFT accuracy:

1. Take each top compound family (InN, AlN, GaAs, SiC, etc.)
2. Extract a small representative cluster (dimer or 4-atom tetrahedron)
3. Build the molecular Hamiltonian (Jordan-Wigner / Bravyi-Kitaev transform)
4. Run VQE on Braket to get the exact ground-state energy
5. Compare VQE energy vs DFT energy for the same cluster geometry
6. Derive a **quantum confidence score** per compound family

If DFT and VQE agree → DFT is trustworthy for that chemistry.
If they disagree → PBE is unreliable, need hybrid functional or DFT+U correction.

### Qubit Requirements (minimal basis STO-3G)

| Fragment | Active electrons | Qubits needed | Feasible on             |
|----------|-----------------|---------------|-------------------------|
| Si₂      | 4-8             | 4-8           | SV1, any QPU            |
| InN      | 4-10            | 8-12          | SV1, Forte Enterprise   |
| GaAs     | 6-10            | 8-12          | SV1, Forte Enterprise   |
| AlN      | 4-8             | 6-10          | SV1, any QPU            |
| Si₂N₂    | 8-16            | 16-24         | SV1, Forte Enterprise   |

### Available Hardware (Feb 2026)

**Recommended path — cheapest first:**

| Device               | Provider | Qubits | Type              | Cost         | Best for                  |
|----------------------|----------|--------|-------------------|--------------|---------------------------|
| **SV1** (simulator)  | AWS      | 34     | Noiseless sim     | $0.075/min   | Development & baseline    |
| **Garnet**           | IQM      | 20     | Superconducting   | $0.00145/shot| Cheap real-QPU runs       |
| **Ankaa-3**          | Rigetti  | 82     | Superconducting   | $0.0009/shot | Largest qubit count       |
| **Forte Enterprise** | IonQ     | 36     | Trapped-ion (FC)  | $0.08/shot   | Best fidelity, full connectivity |
| **Aquila**           | QuEra    | 256    | Neutral-atom      | $0.01/shot   | Lattice Hamiltonian sims  |

**Note:** IonQ Forte Enterprise 1 is the most scientifically useful QPU for
chemistry (fully connected qubits = no SWAP overhead) but extremely expensive.
Start with SV1 simulator to develop and debug, then run on Garnet or Ankaa-3
for a real-QPU result at low cost.

### QuEra Aquila — Alternative Angle

Aquila is an analog quantum simulator (not gate-based). It physically simulates
Rydberg atom lattice Hamiltonians. Potential use:

- Encode a simplified Hubbard model of our crystal lattice
- Check whether electron correlations are weak (validating DFT's mean-field
  approach) or strong (suggesting DFT is unreliable for that lattice geometry)
- This is a more exotic/research-grade application but scientifically novel

### Implementation Plan

```
Phase 1: SV1 Simulator (est. ~$2-5)
  - Install amazon-braket-sdk + pennylane-braket
  - Build InN dimer Hamiltonian in minimal basis
  - Run VQE with UCCSD ansatz on SV1
  - Compare to DFT energy for same geometry
  - Repeat for Si₂, AlN, GaAs dimers

Phase 2: Real QPU validation (est. ~$20-50)
  - Port VQE circuits to IQM Garnet or Rigetti Ankaa-3
  - Run with error mitigation
  - Compare noisy QPU vs noiseless SV1 vs DFT
  - Quantify hardware noise impact on energy accuracy

Phase 3: Compound screening (est. ~$50-100)
  - Run VQE for all top-10 RL candidates
  - Generate quantum confidence scores
  - Flag compounds where PBE disagrees with VQE
  - Re-run flagged compounds with hybrid functional (HSE06)
```

### Key Insight

This won't replace DFT for band structure or effective mass (those are bulk
properties needing periodic boundary conditions and hundreds of electrons).
But it validates the **underlying electronic structure accuracy** — which is
the foundation everything else rests on. If PBE gets the bonding wrong for
InN, then band gaps and effective masses computed with PBE are also wrong.

---

## Pipeline Overview

```
Seeds (10 crystals) → Bootstrap (perturb/swap) → DFT (QE pw.x) → Dataset
    → Train Surrogate MLP → RL Agent (PPO) → Novel Candidates
    → DFT Validation → [Braket Quantum Validation] → Publish
```
