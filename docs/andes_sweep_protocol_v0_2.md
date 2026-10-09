# ANDES IEEE-14 reference sweep protocol v0.2

**Freeze point:** this 84-case matrix is committed before its results are inspected.

## Purpose

Map reference-simulator feasibility over a broader IEEE-14 disturbance space and
produce a sufficiently rich candidate pool from which a later multi-machine
surrogate train/validation/test protocol can be frozen.

This remains a **reference-data experiment**, not a learned-surrogate result.

## Fixed simulator settings

- ANDES 2.0.0
- dynamic base case: `ieee14/ieee14.json`
- three-phase-to-ground `Fault`
- fault application: 1.0 s
- fault resistance: 0 pu
- fault reactance: 1e-4 pu
- simulation end: 2.0 s
- export grid: 401 points from 0–2 s

## Frozen scenario matrix

Fault location:

- every IEEE-14 bus, **1 through 14**

Fault duration:

- 0.04 s
- 0.06 s
- 0.08 s
- 0.10 s
- 0.12 s
- 0.14 s

Total planned cases: **84**.

## Exported trajectories

Every successful case exports:

- 5 GENROU rotor-angle trajectories;
- 5 GENROU speed trajectories;
- 14 bus-voltage trajectories;
- the deterministic 401-point time grid;
- case-level excursion metrics.

Every failed case remains in `case_summary.csv` with its fault location,
duration, exception type, and message.

## Why rotor angle is added

The v0.1 reference adapter exported speed and bus voltage only. v0.2 also
exports GENROU rotor angle so that the later surrogate can evaluate at least the
kinematic machine residual linking angle and speed, and can be extended toward
full symbolic GENROU residuals rather than becoming a purely supervised neural
network.

## Discipline for the next phase

The 84-case sweep is used only to determine where the reference simulator
returns usable trajectories.

After this sweep:

1. a separate multi-machine surrogate protocol will freeze train, validation,
   in-domain test, and OOD/edge-case partitions;
2. failed reference cases will remain part of the documented feasibility map;
3. no test trajectory may be used to tune network architecture or gate
   thresholds;
4. the first surrogate iteration will report negative results if the available
   reference domain is too sparse for meaningful held-out evaluation.

## Interpretation boundary

The sweep must not be described as:

- a stability-limit or critical-clearing-time calculation;
- an ML/PINN accuracy result;
- evidence that ANDES failure implies physical instability;
- evidence of equivalence to PowerFactory, PSCAD, EMTP, PSS/E, or TSAT.
