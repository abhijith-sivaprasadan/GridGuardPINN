# GridGuardPINN experiment specification v0.2

**Freeze point:** this protocol was defined before inspecting any parametric-PINN
validation, ID-test, or OOD-test trajectory-error result.

## Objective

Test whether a deterministic trust gate can reject unreliable PINN surrogate
trajectories while retaining useful surrogate coverage for SMIB dynamic cases.

This is not a claim that SMIB dynamics represent a production grid. The
scientific object is the verification and fallback protocol.

## Surrogate

The model predicts rotor angle and per-unit speed deviation from:

- physical time;
- inertia constant H;
- damping D;
- fault-clearing time;
- fault-on transfer ratio.

The initial equilibrium is enforced as a hard output constraint. Supervised
reference-trajectory anchors are combined with swing-equation residual loss, so
the network is a hybrid data/physics-informed surrogate rather than a pure
black-box regressor.

## Frozen split

- Train: 24 factorial cases inside the nominal parameter box.
- Validation: 16 unseen interpolating cases, used for gate calibration.
- ID test: 32 unseen interpolating combinations; never used for calibration.
- OOD test: 16 cases with one or more parameters outside the train box; never
  used for calibration.

The exact parameter values are defined in canonical_splits().

## Provisional trajectory-quality tolerances

A case is provisionally labelled acceptable only if both are true:

- rotor-angle RMSE <= 0.05 rad;
- speed-deviation RMSE <= 5e-4 pu.

The composite error ratio is the maximum of each RMSE divided by its tolerance.
These are transparent screening thresholds for this demonstrator, not industry
protection or transient-stability acceptance standards.

## Trust signals

1. Normalised PINN physics-residual score away from event discontinuities.
2. Mahalanobis distance in the four-dimensional scenario-parameter space.

The parameter-space detector is fitted on training scenarios only.

## Gate calibration

The OOD threshold is derived only from train + validation in-distribution
scores. The residual threshold is selected on validation cases to maximise
coverage subject to zero validation false accepts.

No ID-test or OOD-test case may affect either threshold.

## Evaluation

Report for validation, untouched ID test, and untouched OOD test:

- rotor-angle RMSE and maximum absolute error;
- speed RMSE and maximum absolute error;
- physics residuals;
- OOD score;
- residual/error Spearman association;
- surrogate coverage;
- false accepts among all cases and among accepted cases;
- false escalations;
- mean accepted-case error.

Compare four routing strategies:

- always accept surrogate;
- OOD gate only;
- residual gate only;
- combined residual + OOD gate.

A negative result is still a valid result. If the gate cannot reduce false
accepts without collapsing coverage, the repository should report that result
rather than changing the held-out sets.
