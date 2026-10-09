# Pre-registered v1.0 independent-duration routing evaluation

**Freeze:** this document and the executable case manifest are committed before the benchmark trigger. No model retraining, trust-threshold changes, or score redesign are permitted after inspecting its outputs.

## Aim and independence

Evaluate the **existing three one-hot v0.3 trained checkpoints** (seeds 17, 29, 41), held at their original validation-only thresholds, on new fault durations not used anywhere in v0.1–v0.3 train/validation/ID/duration-OOD protocols. This tests interpolation across duration values, **not** new network topology, independent organizations, or a different simulator. The broader project has already inspected related IEEE-14 cases; this is a new case-list assessment, not an independent model-development history.

## Frozen scenario manifest

- Seven robust fault buses: `(3,6,7,8,9,11,14)`.
- New interpolation durations: `(0.05,0.07,0.09,0.11)` seconds; **28** unique cases per model.
- New longer-duration stress set: `0.13` seconds × seven buses; **7** unique cases per model.
- Reference: unchanged ANDES 2.0.0 IEEE-14 3-phase fault, at `t=1.0` s, fault reactance `1e-4`, 401-point time grid to `2.0` s.
- Accuracy screen unchanged: max machine rotor-angle RMSE ≤ 0.05 rad **and** speed RMSE ≤ 5e-4 pu.
- Three pinned model/checkpoint SHA-256 values and archive digest are those documented in `scripts/run_andes_trained_runtime_audit.py`.

## Frozen outcomes and interpretation

Report per seed and split: reference successes/failures, surrogate coverage, true false accepts among accepted predictions, false escalations, mean composite error of accepted predictions, and routed errors. Reference failures must be reported explicitly and excluded from accuracy denominators with the count visible. If any eligible surrogate result exceeds the research error screen, **v1 trust acceptance fails**.

Release candidate acceptance requires, for each of three seeds:

1. no unsafe surrogate accepts among successfully referenced interpolation cases;
2. at least 50% surrogate coverage on successfully referenced interpolation cases;
3. all duration-OOD cases must route to reference (success or explicitly surfaced simulator failure);
4. exact SHA-256 checkpoint verification and successful CI.

No threshold adaptation, selective case omission, or model updates are permitted after evaluation. Any failed condition is a negative result and blocks an unqualified v1.0 release. Statistical safety assurance is explicitly outside scope.
