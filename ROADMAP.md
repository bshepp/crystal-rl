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
- **Phase 2:** Freeze trunk+m* head, fine-tune gap head with 12k MP records
- Result: **m* corr=0.928, gap acc=96.7%** (vs baseline 0.625 / 90.4%)
- Model saved: `data/checkpoints/jarvis_surrogate/` (141,890 params, 16,359 records)

#### Phase 5: PPO Training with Improved Surrogate (✅ Complete)
- 250k timesteps, ~88 min training
- Q1→Q4 reward: 156.3 → 454.4 (+298.2)
- **Eval mean: 657.2** ±386.5
- 20/20 evaluation episodes produced semiconductors (11 unique formulas)
- Best formulas: GaP, In₂Sb₂, InP, Si₂, CSi, As₂In₂, InSb, Ge₄, C₂, AsGa, AlAs
- Model saved: `data/checkpoints/ppo_jarvis/ppo_final.zip`

### Next Steps

#### Phase 10: DFT Validation of Signed-m\* Candidates (Next)
- Priority: DFT-validate CAlAsP (m\*=0.002), Sb₂Te₂ (0.067), AsGe (0.015)
- These are the top 3 by reward from the signed-m\* PPO agent
- 6/20 eval episodes produced negative m\* predictions — confirms surrogate
  learned sign direction but has limited accuracy (only 794 bootstrap records
  carry signed m\*, vs 3,565 JARVIS records that are always positive)
- Run on Docker QE container: coarse SCF → band structure → signed m\* extraction

#### Phase 11: SNUMAT Integration (Planned)
- SNUMAT database has ~10,000 HSE06 effective mass records
- This is actual m\* data (not gap-only like MP) — would directly improve m\* head
- Requires web download from snumat.com, CIF → ASE conversion
- Would significantly help signed m\* training (more data with curvature sign)

#### Phase 12: Quantum Validation via Amazon Braket (Planned)
- Use VQE on small molecular fragments to cross-validate DFT accuracy
- See dedicated section below

### Completed (Post-Pipeline)

#### Phase 6b: Feature Expansion (✅ Complete)
- Expanded element palette from 10 to 14 species (+Sn, Sb, Bi, Se, Te)
- Added 6 supercell seeds (Si₄, GaAs₄, InP₄, SiC₄, Ge₄, AlN₄) → 18 total seeds
- Added stability discount penalty for highly strained structures
- Added unusual topology logger for band-inverted candidates

#### Phase 7: DFT Validation of Discovered Candidates (✅ Complete)
- Ran Quantum ESPRESSO on 8 top PPO-discovered structures
- **All 8 candidates converged** in DFT (QE pw.x)
- **6 of 8 showed unusual band topology** (negative DFT effective mass → band inversion)
- Surrogate MAE vs DFT: 1.492 mₑ, correlation 0.057
- Candidates with inverted bands: As₂Ga₂, Ge₄, Ge₂, AsIn, AsGa, As₂In₂
- Normal candidates: GaP (DFT m*=+1.174, gap=0.229 eV), Si₄ (DFT m*=+1.750)
- Results saved: `data/validation/validation_report.json`, `data/validation/unusual_topology.json`

#### Phase 8: Pipeline Retrain with Expanded Chemistry (✅ Complete)
- Retrained surrogate on 16,359 records with expanded chemistry
- Retrained PPO with 18 seeds, 14-element palette, stability penalty
- Full DFT validation loop completed

#### Phase 9: Signed Effective Mass Fix (✅ Complete)

**Problem discovered:** The pipeline was stripping the sign of effective mass
at 10+ locations, causing the RL agent to exploit surrogate confusion about
band-inverted (negative curvature) materials for inflated rewards.

**Root causes fixed:**
- `retrain_jarvis.py`: `np.abs()` on bootstrap DFT data removed all negative
  curvature values; `m_star <= 0` filters rejected valid negative-m\* records;
  `has_mstar = y_m > 0` masks treated negative m\* as missing data
- `crystal_env.py`: Reward function gave `2.0/abs(m*)` for negative m\*
  (massive reward for band-inverted predictions) instead of penalizing

**Fixes applied (10+ edits across 2 files):**
- `retrain_jarvis.py`:
  - Bootstrap loader: `np.abs(data[mstar_key])` → `data[mstar_key].astype(float)`
  - JARVIS filter: `m_star <= 0` → `m_star == 0` (allow negatives)
  - Range filter: `m_star > max_mstar` → `abs(m_star) > max_mstar`
  - Gap-only filter: `m_star > 0` → `m_star != 0`
  - Fingerprint builder: only `None → 0.0` (preserve negatives)
  - Valid mask: `y_m > 0.001` → `np.abs(y_m) > 0.001`
  - All `has_mstar = y_m > 0` → `has_mstar = y_m != 0` (3 locations)
  - Metrics: `true_m > 0` → `true_m != 0` (2 locations)
- `crystal_env.py`:
  - Negative m\*: `2.0/abs(m)` → flat `-2.0` penalty
  - Positive m\*: `-abs(m)` → `1.0/max(m, 0.01)` (lighter carriers → higher reward)
  - DFT reward: same pattern

**Stale cache:** Renamed `jarvis_fingerprints.npz` → `jarvis_fingerprints_unsigned.npz`
to force regeneration with signed values.

**Surrogate retrain results:**
- m\* correlation: 0.928 → **0.817** (expected — signed targets are harder)
- Gap accuracy: 96.7% → **96.0%**
- Normalization: m_mean=0.333, m_std=1.089 (signed range, vs previous unsigned)
- 16,359 records, 141,890 params, early stopped Phase 1 epoch 115 + Phase 2 epoch 96
- Model now predicts negative m\* (sample: pred=-1.66, true=-3.44)

**PPO retrain results:**
- Eval mean: 657.2 → **512.8 ± 193.1** (expected — can't exploit negative m\*)
- Unique formulas: 11/20 → **20/20** (major improvement in diversity)
- Metal fraction: 1.88% → 8.84% (explores more broadly)
- Training: 250k timesteps, ~100 min, avg200 climbed 199→393 over 6,000+ episodes

**Notable discoveries from signed-m\* PPO:**

| Formula | Predicted m\* | Predicted gap (eV) | Reward | Notes |
|---------|--------------|-------------------|--------|-------|
| Sb₂Te₂ | 0.067 | 0.548 | 820.6 | Best reward — light carrier chalcogenide |
| CAlAsP | 0.002 | 1.190 | 581.1 | Near-zero m\* — extremely interesting |
| AsGe | 0.015 | 1.823 | 524.1 | IV-V binary, very light carrier |
| AsGa₂N | 0.036 | 1.516 | 506.5 | III-V variant |
| AlAsGaN | 0.130 | 1.429 | 413.9 | Quaternary III-V |
| BiInNSb | 0.084 | 1.351 | 392.1 | Bismide quaternary |
| Al₂PSn | 0.067 | 1.595 | 377.7 | Novel III-IV-V |
| CAlGeN | 0.125 | 1.791 | 369.2 | Carbon-III-IV-V |
| Si₂ | 0.250 | 1.162 | 247.7 | Rediscovered silicon |
| AsGaSn | 0.065 | 1.010 | 118.3 | III-IV-V ternary |

**Key insight:** CAlAsP with m\*=0.002 is a strong DFT validation candidate.
If confirmed, it would rival InSb (m\*=0.014) as an ultra-low effective mass
semiconductor — in a non-toxic composition.

Commit: `8d1ed96`

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
Seeds (18 crystals) → Bootstrap (perturb/swap) → DFT (QE pw.x) → Dataset
    → Train Surrogate MLP → RL Agent (PPO) → Novel Candidates
    → DFT Validation → [Braket Quantum Validation] → Publish
```
