# GridGuardPINN experiment specification v0.3

**Freeze point:** this protocol is committed after observing the failed v0.2
baseline and before evaluating the improved v0.3 surrogate on the new held-out
sets.

## Why a new holdout exists

The v0.2 ID and OOD results have already been inspected. They remain regression
benchmarks, but using them to claim blind post-tuning generalisation would be
methodologically weak. v0.3 therefore keeps the same training and validation
sets while freezing new held-out cases before the architecture and sampling
changes are evaluated.

## Training and calibration

Training remains the same 24-case factorial parameter box.

Validation remains the same 16 unseen interpolating cases. Architecture and
training choices may be assessed on training/validation behaviour only.

## Fresh v0.3 ID test

24 cases are generated from a fixed seed (20261009), sampling strictly inside
the training parameter ranges but away from the original factorial grid.

These cases are not used for training or trust-threshold calibration.

## Fresh v0.3 OOD test

24 cases are generated from the same fixed seed. Six cases isolate each of four
OOD mechanisms:

- inertia H outside the training interval;
- damping D outside the training interval;
- fault-clearing time outside the training interval;
- fault-on transfer ratio outside the training interval.

All other parameters remain sampled inside the nominal range. This makes the
OOD diagnosis more interpretable than the v0.2 all-extremes grid.

## Planned structural changes

The v0.3 surrogate changes are motivated by the v0.2 training design, not by
individual v0.3 holdout outputs:

1. event-aware supervised sampling with more anchors around fault application
   and clearing;
2. phase-stratified physics collocation so the short fault-on interval is not
   swamped by the long post-fault interval;
3. explicit event-phase features in the surrogate;
4. a less restrictive hard-initial-condition transform for early transients;
5. data-first physics-loss warm-up/ramp;
6. a larger but still bounded optimisation budget.

## Quality definition

The provisional acceptable-case limits remain unchanged from v0.2:

- rotor-angle RMSE <= 0.05 rad;
- speed-deviation RMSE <= 5e-4 pu.

Changing the model is allowed. Moving the accuracy goalposts after seeing the
result is not.

## Trust evaluation

As in v0.2, the trust gate is calibrated on validation only and is compared
against:

- always accept;
- OOD-only routing;
- residual-only routing;
- combined residual + OOD routing.

Primary safety metric: false accepts among accepted surrogate cases.

Primary usefulness metric: surrogate coverage subject to low false-accept rate.
