# Result — v0.3 event-aware surrogate

Run commit: 3397df51029a21b9c5aec1678de75bb797caf1a2  
Seed: 7  
Training: 1,200 Adam epochs, 81 supervised anchors/case, 64 physics points/case

## Structural changes from v0.2

- event-aware supervised sampling;
- phase-stratified physics collocation;
- explicit fault/post-fault features;
- less restrictive hard initial-condition transform;
- data-first physics-loss warm-up/ramp;
- fresh held-out ID/OOD sets frozen before evaluation.

## Accuracy

| Split | Cases | Good cases | Mean composite error | Median | Maximum |
|---|---:|---:|---:|---:|---:|
| Validation | 16 | 2 | 1.97 | 1.84 | 3.21 |
| Fresh ID test | 24 | 2 | 1.81 | 1.62 | 3.36 |
| Fresh OOD test | 24 | 0 | 3.66 | 2.99 | 14.37 |

The provisional good-case definition remains rotor-angle RMSE <= 0.05 rad and
speed-deviation RMSE <= 5e-4 pu.

Relative to v0.2, the ID mean composite error fell from about 8.2 to 1.81. The
model is therefore substantially better, but still not accurate enough to act
as a generally accepted surrogate.

## Trust-signal result

Physics residual became strongly informative:

- validation residual/error Spearman rho = 0.921;
- fresh ID-test rho = 0.886;
- fresh OOD-test rho = 0.906.

This is the most important positive v0.3 result: residual magnitude is not proof
of correctness, but in this experiment it is strongly associated with actual
trajectory error against the reference solver.

## Frozen validation-calibrated gate

Residual threshold: 0.17569  
Parameter-space OOD threshold: 2.07814

### Fresh ID test

Always accepting the surrogate produced 22 false accepts in 24 cases.

The combined residual + OOD gate:

- accepted 1 / 24 cases (4.17% coverage);
- produced 0 false accepts;
- produced 1 false escalation (one genuinely acceptable case was rejected);
- accepted-case composite error = 0.879.

This is safe but not useful enough: a gate that falls back on 23/24 ID cases
does not yet deliver a meaningful surrogate-speed advantage.

### Fresh OOD test

All 24 OOD trajectories exceeded the provisional error tolerance.

The combined gate rejected all 24, producing zero false accepts. OOD-only
routing was not sufficient: it still accepted 8/24 OOD cases, all of which were
bad trajectories.

## OOD mechanism breakdown

Mean always-accept composite error:

- damping shift: 2.24;
- inertia shift: 4.19;
- fault-severity shift: 1.91;
- clearing-time shift: 6.29.

The parameter-space detector rejected all clearing-time cases, but only part of
the H/D/fault-ratio shifts. Residual gating rejected all OOD cases in this run.

## Conclusion

v0.3 supports two claims and does **not** support a third:

1. Event-aware training materially improves the parametric PINN.
2. Physics residual is a useful empirical trust signal in the frozen tests.
3. The surrogate is **not yet accurate enough for useful high-coverage
   routing** under the stated tolerance.

The next iteration should target surrogate accuracy while preserving the
unchanged tolerance and validation-only gate calibration. Because the v0.3
holdouts have now been observed, a fresh v0.4 holdout must be frozen before
evaluating the next architecture.
