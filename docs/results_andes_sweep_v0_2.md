# ANDES IEEE-14 expanded fault sweep result v0.2

Run: GitHub Actions `ANDES IEEE-14 expanded sweep`  
Frozen matrix: 14 buses × 6 fault durations = 84 cases  
Reference: ANDES 2.0.0, dynamic `ieee14/ieee14.json` + programmatic fault

## Result

- planned: **84**
- successful: **63**
- failed: **21**
- successful fraction: **75%**

| Fault bus | Successful durations | Count |
|---:|---|---:|
| 1 | 0.04, 0.06, 0.08, 0.10 | 4/6 |
| 2 | 0.04, 0.06, 0.08, 0.10 | 4/6 |
| 3 | all six | 6/6 |
| 4 | all except 0.10 | 5/6 |
| 5 | 0.04, 0.06, 0.08 | 3/6 |
| 6 | all six | 6/6 |
| 7 | all six | 6/6 |
| 8 | all six | 6/6 |
| 9 | all six | 6/6 |
| 10 | 0.06, 0.08, 0.10 | 3/6 |
| 11 | all six | 6/6 |
| 12 | none | 0/6 |
| 13 | 0.12, 0.14 | 2/6 |
| 14 | all six | 6/6 |

Seven buses—**3, 6, 7, 8, 9, 11, 14**—form a fully successful 42-case core
domain spanning all six durations.

## Why this changes the next experiment

The result is rich enough to freeze a real multi-machine surrogate protocol
without pretending the reference simulator is valid everywhere.

The first learned model will therefore:

- train/calibrate/test only on the seven fully robust buses for clean ID and
  duration-extrapolation experiments;
- reserve all successful cases from partially robust buses as a separate
  **unseen-location edge set**;
- retain the 21 failed reference cases as a documented simulator-feasibility
  boundary, not as missing data.

The protocol is frozen in
[`andes_surrogate_protocol_v0_1.md`](andes_surrogate_protocol_v0_1.md).

## Important interpretation

A failed ANDES run is **not automatically a physically unstable system**.
Workflow logs include both stability-criteria termination and numerical
time-step collapse. The feasibility map therefore describes the behaviour of
this specific reference workflow under fixed solver settings.

The result supports automated multi-machine reference generation and disciplined
dataset construction. It does not yet support any learned-surrogate accuracy or
trust-gate claim.
