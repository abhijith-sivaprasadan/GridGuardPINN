# ANDES multi-machine surrogate result v0.1

Run: GitHub Actions `ANDES multi-machine surrogate v0.1`  
Workflow run: **37899938971**  
Frozen model commit: **8cf0b3e56cd476a65a05640afbe653b0d752ccbe**  
Seed: **17**  
Reference: ANDES 2.0.0 IEEE-14 dynamic case with programmatic three-phase faults

This result was opened only after the v0.1 architecture, split, optimisation
schedule, accuracy screen, and validation-only gate rule had been frozen.

## Configuration

- train: 21 cases on buses 3, 6, 7, 8, 9, 11, 14 at 0.04/0.08/0.12 s;
- validation: 7 cases at 0.06 s;
- fresh ID test: 7 cases at 0.10 s;
- duration OOD: 7 cases at 0.14 s;
- unseen-location edge set: 21 successful cases on buses 1, 2, 4, 5, 10, 13;
- model: 20 outputs (five-machine delta, omega, tm, te);
- hidden network: five 96-unit tanh layers with event/Fourier encoding;
- physics loss: shared GENBase swing-equation residuals;
- epochs: 1000;
- physics weight: 0.02 after warm-up/ramp;
- best validation checkpoint: epoch 1000;
- reference generation wall time: 47.27 s;
- model-training wall time: 39.74 s.

The research-screening accuracy criterion remained:

- maximum machine rotor-angle RMSE <= 0.05 rad;
- maximum machine speed RMSE <= 5e-4 pu.

## Accuracy

| Split | Cases | Good | Mean composite ratio | Median | Max | Residual/error Spearman rho |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 7 | **7/7** | 0.146 | 0.147 | 0.168 | 0.714 |
| Fresh ID test | 7 | **7/7** | 0.152 | 0.138 | 0.206 | 0.786 |
| Duration OOD (0.14 s) | 7 | **7/7** | 0.420 | 0.407 | 0.582 | 0.643 |
| Unseen-location edge set | 21 | **0/21** | 6.079 | 3.470 | 17.301 | 0.412 |

The largest fresh-ID errors were still comfortably below the frozen screen:
max angle RMSE 0.00904 rad and max speed RMSE 1.03e-4 pu.

The duration-extrapolation cases were harder but remained within the screen:
max angle RMSE 0.02910 rad and max speed RMSE 2.16e-4 pu.

## Trust-gate result

The validation-calibrated residual threshold was **0.274355**.

| Split | Always-accept false accepts | Residual-only accepted | Residual-only false accepts | Combined strict accepted | Combined false accepts |
|---|---:|---:|---:|---:|---:|
| Validation | 0/7 | 7/7 | 0 | 7/7 | 0 |
| Fresh ID test | 0/7 | 4/7 | 0 | 4/7 | 0 |
| Duration OOD | 0/7 | 0/7 | 0 | 0/7 | 0 |
| Unseen-location edge set | **21/21** | 0/21 | 0 | 0/21 | 0 |

The gate therefore made no unsafe acceptance on this run, but the residual
threshold was substantially over-conservative outside its calibration slice.

Three fresh-ID cases were accurate but escalated because their residual scores
were above the 0.06 s validation threshold. More importantly, **all seven
0.14 s duration-OOD trajectories were accurate yet rejected**. Their residual
scores (about 0.36-0.57) increased with disturbance duration even though their
trajectory errors remained below the frozen tolerance.

## Interpretation

This first multi-machine result is materially stronger than the reduced-order
SMIB baselines:

1. The learned surrogate achieved **7/7 accuracy on the fresh ID set**.
2. It also achieved **7/7 accuracy on the frozen duration-extrapolation set**.
3. It failed **all 21 unseen-location cases**, which is consistent with the
   categorical one-hot fault-bus encoding: weights connected to unseen bus
   categories receive no supervised training signal.
4. Physics-residual magnitude retained a positive association with trajectory
   error, but a single validation slice at 0.06 s was not sufficient to produce
   a high-coverage duration-robust threshold.
5. Explicit location-OOD rejection remains necessary. Always accepting the
   surrogate on unseen locations would have produced 21/21 false accepts.

The central v0.1 result is therefore:

> **The first frozen IEEE-14 multi-machine surrogate is accurate across the
> held-out trained-location duration range and one frozen duration extrapolation,
> but it does not generalise to unseen fault locations. The present residual
> gate is safe on this small test but unnecessarily conservative as fault
> duration increases.**

## What is not claimed

- This is one random seed.
- The ID and duration test sets each contain only seven cases.
- The unseen-location set contains only reference simulations that completed
  successfully; separate reference-feasibility results retain the 21 failed
  simulations.
- The quality thresholds are transparent research screens, not grid-code,
  protection, or operational acceptance limits.
- The model enforces only the shared electromechanical swing equations, not the
  full GENROU/network DAE residual set.
- v0.1 did not record per-case surrogate inference timing. Timing instrumentation
  was added before the pre-registered v0.2 ablation.

## Reproducibility and timing check

After refactoring the experiment runner, workflow run **37912546737** repeated the
same seed-17 experiment. All reported trajectory metrics, residual/error
correlations, and the calibrated residual threshold matched the untouched
artifact exactly at the stored floating-point values.

The refactored run also separated pure ANDES solve timing from preprocessing:

- 63 ANDES solves: **50.01 s** total;
- median ANDES solve per case: **0.678 s**;
- median 401-point neural forward pass: about **0.00043 s** across splits;
- median reference-solve / forward-pass ratio: about **1.35e3 to 1.66e3x**
  depending on split.

This ratio compares the neural network forward pass with the full ANDES
case solve. It is a research timing measurement on the same GitHub Actions CPU
runner, not a production latency or end-to-end service benchmark.

## Frozen next step

Do not tune the v0.1 model or gate around these held-out observations.

The next experiment is the already pre-registered
[data-only vs physics-informed ablation](andes_surrogate_ablation_plan_v0_2.md)
using identical architecture, splits, optimiser schedule, and seeds 17/29/41.
Its purpose is to determine whether the physics term contributes to accuracy,
duration generalisation, residual/error association, or safe routing coverage.
