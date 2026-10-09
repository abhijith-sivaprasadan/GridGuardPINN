# ANDES IEEE-14 reference sweep protocol v0.1

**Freeze point:** this scenario matrix is committed before running the sweep.

## Purpose

Establish whether the programmatic ANDES IEEE-14 reference path is robust across
multiple fault locations and clearing durations, and generate the first
multi-case transient dataset for the next surrogate phase.

This is a **reference-simulator feasibility/data-generation experiment**. It is
not yet a machine-learning train/test split.

## Fixed settings

- ANDES 2.0.0
- dynamic base case: `ieee14/ieee14.json`
- three-phase-to-ground `Fault`
- fault application time: 1.0 s
- fault resistance: 0 pu
- fault reactance: 1e-4 pu
- simulation end: 2.0 s
- deterministic export grid: 401 points from 0–2 s

## Scenario matrix

Fault buses:

- 2
- 4
- 5
- 9
- 12
- 14

Fault durations:

- 0.06 s
- 0.10 s
- 0.14 s

Total planned cases: **18**.

The buses are deliberately distributed across the IEEE-14 network rather than
concentrated around the packaged bus-9 example.

## Per-case outputs

For every successful case:

- native TDS point count;
- all 5 GENROU speed trajectories;
- all 14 bus-voltage trajectories;
- maximum absolute generator-speed deviation;
- minimum and maximum bus voltage;
- deterministic 401-point resampled trajectory.

For every failed case:

- bus;
- duration;
- exception type/message.

A failed reference simulation remains in the case table and is **not silently
removed**.

## Sweep-level outputs

- `case_summary.csv`
- `trajectories.csv`
- `summary.json`

## Interpretation boundary

This sweep may be used to decide whether the ANDES scenario generator is
sufficiently robust to support a later surrogate dataset.

It must not be presented as:

- a PINN accuracy result;
- a stability-limit study;
- a critical-clearing-time calculation;
- a protection or planning assessment.

If the sweep succeeds broadly, a separate surrogate protocol will freeze
train/validation/held-out cases before any ANDES-trained neural model is
evaluated.
