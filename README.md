# RL-Materials: Reinforcement Learning for Semiconductor Discovery

An RL-driven materials discovery pipeline that uses Quantum ESPRESSO DFT
calculations, a neural network surrogate model, and PPO reinforcement learning
to search for novel crystal structures with optimal electronic properties
(low effective mass semiconductors).

## Architecture

```
Seeds (10 crystals) → Bootstrap DFT (QE pw.x) → Dataset (794+ records)
    ↓                                               ↓
JARVIS-DFT (3,565 m* records)  ──────────→  Train MultiTaskMLP Surrogate
Materials Project (12,990 gap records) ──→       ↓
                                          PPO Agent explores crystal space
                                               ↓
                                          Top candidates → DFT Validation
```

## Key Results

| Stage | Metric | Value |
|-------|--------|-------|
| Surrogate (two-phase) | m* correlation | 0.886 |
| Surrogate (two-phase) | gap accuracy | 95.2% |
| PPO (250k steps) | eval mean reward | 655.4 |
| PPO (250k steps) | semiconductor rate | 100% (20/20) |

## Project Structure

```
rl-materials/
├── configs/default.yaml        # QE and RL hyperparameters
├── envs/crystal_env.py         # Gymnasium RL environment
├── models/surrogate.py         # MultiTaskMLP surrogate model
├── qe_interface/               # Quantum ESPRESSO wrapper (ASE-based)
│   ├── calculator.py           # QE runner with timeout/error recovery
│   ├── properties.py           # Band gap + effective mass extraction
│   └── structures.py           # Seed structures + 152-dim fingerprints
├── scripts/
│   ├── retrain_jarvis.py       # Surrogate training (JARVIS + bootstrap + MP)
│   ├── train_ppo_jarvis.py     # PPO training against surrogate
│   ├── download_mp.py          # Materials Project data download
│   ├── validate_dft.py         # DFT validation of RL candidates
│   ├── bootstrap_*.py          # DFT bootstrap data collection
│   └── ...                     # Debug/analysis utilities
├── pseudopotentials/           # SSSP Efficiency PAW/RRKJUS pseudopotentials
├── data/
│   ├── jarvis_fingerprints.npz # 3,565 × 152 (m* + gap)
│   ├── mp_fingerprints.npz     # 12,990 × 152 (gap-only)
│   ├── bootstrap/              # 794 DFT results from AWS
│   └── checkpoints/            # Saved surrogate + PPO models
├── Dockerfile                  # QE 7.3.1 + Python ML stack
├── docker-compose.yml          # Dev container config
└── pyproject.toml              # Python dependencies
```

## Quick Start

### Local (surrogate + PPO only, no DFT)

```bash
# Create virtual environment
python -m venv .venv && .venv\Scripts\activate  # Windows
# python -m venv .venv && source .venv/bin/activate  # Linux/Mac

pip install -e .

# Train surrogate on JARVIS + bootstrap + MP (two-phase)
PYTHONPATH=. python scripts/retrain_jarvis.py \
    --include-bootstrap --include-mp --max-mp 4000 \
    --hidden-dim 192 --gap-weight 0.3 --two-phase

# Train PPO agent
PYTHONPATH=. python scripts/train_ppo_jarvis.py --timesteps 250000
```

### Docker (with DFT)

```bash
docker compose build
docker compose run --rm qe-rl python -m scripts.train_medium
```

## Surrogate Model

**MultiTaskMLP** — shared trunk with separate heads for effective mass (m\*)
and band gap prediction. Trained with two-phase approach:

1. **Phase 1:** Full model trained on JARVIS + bootstrap data (all records
   have both m\* and gap labels) — optimizes m\* correlation
2. **Phase 2:** Freeze trunk + m\* head, fine-tune gap head with 4,000
   Materials Project semiconductor records (gap-only labels) — boosts gap
   accuracy without degrading m\* performance

Architecture: `152-in → [192×SiLU+LN+Drop]×3 → head_m*(192→96→1) + head_gap(192→96→1)`
Total parameters: 141,890

## Data Sources

| Source | Records | Properties | Notes |
|--------|---------|------------|-------|
| Bootstrap DFT | 794 | m\* + gap | QE pw.x on AWS EC2 |
| JARVIS dft_3d | 3,565 | m\* + gap | `avg_elec_mass` + `optb88vdw_bandgap` |
| Materials Project | 12,990 | gap only | Stable semiconductors, 0.05–11.7 eV |

## Fingerprint

152-dimensional structural descriptor:
- 12 composition features (mean/std/min/max of Z, mass, radius)
- 4 elemental property features (electronegativity stats)
- 8 lattice features (a, b, c, α, β, γ, volume, density)
- 64 RDF features (radial distribution function, 0.5–8.0 Å)
- 64 partial RDF features (element-pair resolved)

## License

MIT
