# Baseline result — v0.2

Run source: GitHub Actions, CPU PyTorch, seed 7, 400 Adam epochs.

This result is intentionally recorded before changing the surrogate architecture
or sampling strategy.

## Frozen protocol

- 24 training cases
- 16 validation cases
- 16 untouched interpolating ID-test cases
- 16 untouched OOD cases
- 61 uniform supervised anchors per training case
- 48 uniformly sampled physics-collocation points per training case
- provisional acceptable-case limits:
  - rotor-angle RMSE <= 0.05 rad
  - speed-deviation RMSE <= 5e-4 pu

## Result

The first parametric PINN **failed the accuracy requirement before the trust
gate became useful**.

| Split | Always-accept false accepts | Mean composite error ratio | Combined-gate coverage | Combined false accepts |
|---|---:|---:|---:|---:|
| Validation | 16 / 16 | 8.23 | 0% | 0 |
| ID test | 16 / 16 | 8.20 | 0% | 0 |
| OOD test | 16 / 16 | 941.31 | 0% | 0 |

The parameter-space OOD detector rejected all 16 OOD cases, but it accepted all
validation and ID-test cases, as expected. Because every validation trajectory
was outside the provisional accuracy tolerance, validation-only calibration of
the residual threshold selected a reject-all policy.

Physics residual still carried some diagnostic signal: Spearman correlation
between residual score and composite trajectory error was approximately 0.63
on validation, 0.63 on ID test, and 0.51 on the OOD set. That is not enough to
rescue an inaccurate surrogate.

## Interpretation

This is a useful negative baseline, not a gate success.

The likely first-order limitations are known *without changing the held-out
results*:

1. **Uniform temporal anchors under-resolve the short fault-on interval.** With
   61 points over 5 s, the dynamic event receives very few supervised anchors.
2. **Uniform collocation has the same problem.** Physics loss is dominated by
   the long post-fault interval.
3. **The hard initial-condition transform scales network corrections with
   t / t_end.** Very early transient outputs therefore require unnecessarily
   large raw network values.
4. **The smooth MLP has no explicit event-phase features** despite piecewise
   forcing at fault application and clearing.
5. 400 Adam epochs are a baseline budget, not evidence of convergence.

The next iteration will address those structural issues rather than changing
the provisional trajectory-quality limits.

## Research-integrity consequence

The v0.2 ID and OOD sets have now been observed. They remain useful regression
benchmarks, but they will not be described as blind holdouts for the tuned
v0.3 model. A new v0.3 held-out set is frozen before evaluating the improved
surrogate.
