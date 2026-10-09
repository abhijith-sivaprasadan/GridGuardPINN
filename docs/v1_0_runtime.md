# GridGuardPINN runtime guide (research demonstrator)

This is an IEEE-14/ANDES surrogate-routing research application, **not an operational grid controller**.

## What executes

- `load_checkpoint` verifies an **externally pinned SHA-256**, checks the experiment protocol, uses PyTorch `weights_only=True`, and loads model weights strictly.
- `compute_model_residual` evaluates the same 151-point fault-aware swing-equation residual as the frozen research protocol, using independently trusted machine inertia, damping, and frequency arrays.
- `run_model_case` computes the residual, applies the conservative bus/duration/residual policy, and runs `run_ieee14_fault` if routing rejects the learned model.
- `scripts/run_andes_runtime.py` provides a CLI that writes `routing.json` and `electromechanical.csv` with the source of each result.

**Important:** Only **one-hot** checkpoints are currently accepted by the runtime. The experimental network-aware v0.3 checkpoint must not be loaded into the one-hot feature encoder; the loader rejects that configuration. Machine constants must be supplied from a trusted IEEE-14 source and ordered according to ANDES's five GENROU machines. A SHA-256 computed from a downloaded untrusted checkpoint and accepted without independent verification does *not* establish provenance.

## CLI invocation template

Install PyTorch and ANDES (Python 3.11):

```bash
pip install torch
pip install -e ".[dev,andes]"
```

Download a compatible trained one-hot model checkpoint from the published experiment artifact; verify its digest independently. Then run:

```bash
python scripts/run_andes_runtime.py \
  --checkpoint PATH_TO_MODEL_PT \
  --sha256 TRUSTED_64_CHARACTER_SHA256 \
  --protocol EXPECTED_EXPERIMENT_PROTOCOL \
  --bus 3 --duration 0.08 \
  --inertia M1 M2 M3 M4 M5 \
  --damping D1 D2 D3 D4 D5 \
  --frequency F1 F2 F3 F4 F5
```

Do not substitute arbitrary machine constants or fabricated residual scores. Values above are explicit placeholders, not runnable numeric defaults.

## Verified integration and limitations

The `ANDES v1 runtime integration smoke` workflow executes a real IEEE-14 reference simulation, evaluates the physics residual of a deliberately **untrained** model, and verifies that the conservative path executes ANDES. It also exercises the unseen-bus reference path and checkpoint roundtrip. Passing this smoke confirms **runtime plumbing only**, not useful surrogate accuracy or safety.

The old ANDES experiments remain the evidence for model accuracy and failures, with only 7 fresh-ID cases per seed. The v0.3 network-aware residual-only gate produced 3–4 false accepts per seed for previously unseen fault locations. Therefore the runtime categorically refuses unseen buses. Even for eligible trained locations, a residual threshold calibrated on seven validation cases does not provide a statistical guarantee of zero false accepts.

Reference failures raise errors; the runtime does not replace failed reference calculations with untrusted predictions. Reference trajectories also contain bus voltages, which the 20-output learned surrogate does not predict.

## Outstanding work for v1.0 research release acceptance

- Demonstrate routing with a **frozen trained checkpoint** and independently documented SHA-256.
- Run an actual-ANDES benchmark with a source-labelled per-case decision ledger, error comparison and clear acceptance criteria on an untouched evaluation set.
- Extend provenance to machine constants and exact network/case fingerprint.
- Publish an immutable release tag and complete reproducibility instructions after all acceptance gates pass.
