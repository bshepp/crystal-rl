# AWS Braket scripts — Phase 12 scaffolding (not wired up)

These four scripts are **discovery / inventory tools** for Amazon Braket
QPUs. They were written in anticipation of **Phase 12 — Quantum Validation
via Amazon Braket** (see [`ROADMAP.md`](../../../ROADMAP.md)), where small
molecular fragments of top RL-discovered compounds would be cross-validated
against PBE DFT using VQE on a quantum simulator (SV1) or real QPU (IQM
Garnet, Rigetti Ankaa-3, IonQ Forte).

**Current state: not wired into any pipeline.** No VQE circuits are
constructed here, no jobs are submitted, no results are collected. The
scripts only query AWS for device availability and pricing.

## Files

| Script | Purpose |
|---|---|
| `inventory.py` | List available Braket QPUs and simulators across regions |
| `details.py` | Fetch detailed specs (qubits, cost-per-shot, status) for a chosen device |
| `missing.py` | Brute-force search for niche devices that don't show up in standard listings (AQT IBEX, IQM Garnet/Emerald, IonQ Forte-Enterprise) |
| `search_all.py` | Scan every AWS region for QPU inventory |

## How to run

Each script is standalone and requires only the AWS CLI configured with
credentials that have Braket read access:

```bash
python scripts/aws/braket/inventory.py
python scripts/aws/braket/details.py
# etc.
```

These do not run inside the project's Docker container — they call the
host's `aws` CLI directly.

## When Phase 12 actually happens

The validated path (see `ROADMAP.md` Phase 12) is:
1. Start on **SV1** (noiseless simulator) to develop the VQE circuit
2. Promote to **IQM Garnet** or **Rigetti Ankaa-3** for low-cost real-QPU runs
3. Compare VQE ground-state energy vs PBE DFT energy for each top RL candidate
4. Flag chemistry families where the two disagree (PBE error → use HSE06)

The scripts in this directory will be the starting point for picking which
device to target — but the circuit construction, job submission, error
mitigation, and analysis code is not yet written.
