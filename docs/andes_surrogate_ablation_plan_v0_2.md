# Pre-registered multi-machine physics ablation plan v0.2

**Freeze point:** committed while the first v0.1 multi-machine experiment is
still running and before its held-out results are inspected.

## Purpose

Determine whether the swing-equation residual loss contributes measurable
generalisation or trust-signal value beyond the same neural architecture trained
from ANDES trajectories alone.

This is an ablation study, not a new hyperparameter search.

## Models

Two models will use the same:

- 20-output architecture;
- input/event encoding;
- 21 train / 7 validation / 7 ID-test / 7 duration-OOD / 21 location-OOD split;
- supervised anchor locations;
- optimiser schedule;
- hidden width/depth;
- output scaling;
- random seed(s);
- validation checkpointing rule.

The only planned difference is:

1. **data-only:** physics weight = 0;
2. **physics-informed:** the frozen v0.1 swing-equation residual loss and weight.

No architecture or error threshold may be changed after inspecting the v0.1
test results for the purpose of making either ablation arm look better.

## Required comparisons

For both arms report:

- validation and fresh ID trajectory errors;
- duration-OOD and location-OOD trajectory errors;
- rotor-angle and speed metrics separately;
- inference time for a 401-point trajectory;
- reference ANDES simulation time for the same successful cases;
- residual/error Spearman correlation;
- validation-calibrated residual-gate false accepts, coverage, and false escalations.

The data-only model is still evaluated with the same swing-equation residual at
inference time. This tests whether the residual is useful as a *diagnostic*
even when it was not part of training.

## Repeats

If compute permits, run three fixed seeds: **17, 29, 41**. Aggregate median and
range across seeds. If only one seed is completed initially, label it explicitly
as a single-seed result and do not generalise stochastic conclusions.

## Interpretation

Evidence for a useful physics-informed contribution requires more than a lower
training physics loss. At least one of the following should improve without a
material regression elsewhere:

- fresh-ID trajectory accuracy;
- duration-extrapolation accuracy;
- residual/error association;
- safe gate coverage at a fixed false-accept requirement.

A null or negative ablation result is reportable and should not be tuned away.
