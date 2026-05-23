# Debug / diagnostic scripts (not part of the canonical pipeline)

These scripts were written ad-hoc during development to inspect QE output,
validate the fingerprint code, probe individual pipeline stages, and smoke-
test the Docker integration. **They are not part of the canonical pipeline**
(which is `retrain_jarvis.py` → `train_ppo_jarvis.py` → `validate_dft.py`,
all at `scripts/`).

## Files

| Script | Purpose |
|---|---|
| `bands.py` | Run a Si band-structure calculation through QE and dump raw output for inspection |
| `effmass.py` | Standalone effective-mass extraction sanity check |
| `diag_dft.py` | Probe individual QE pipeline steps (SCF → NSCF → bands) |
| `verify_fp.py` | Round-trip check of the structural fingerprint code |
| `analyze_jarvis.py` | Inspect the JARVIS dataset (record counts, m\* distributions, etc.) |
| `check_aflow_schema.py` | Probe the AFLOW database schema (used while deciding between AFLOW / MP / JARVIS as augmentation source) |
| `test_silicon.py` | Smoke test: bulk-Si SCF inside Docker. Validates the QE + ASE + Python install end-to-end |
| `test_integration.py` | End-to-end integration test: bands + effective mass + env stepping with DFT and surrogate |

## How to run

Most of these import from `qe_interface` / `envs` / `models`, so invoke
them as modules from the repo root rather than as scripts:

```bash
python -m scripts.debug.effmass
python -m scripts.debug.verify_fp
```

The QE-dependent ones (`bands.py`, `diag_dft.py`, `test_silicon.py`,
`test_integration.py`) need Quantum ESPRESSO installed — they're intended
to run inside the project's Docker container:

```bash
docker compose run --rm qe-rl python -m scripts.debug.test_silicon
```
