# ANDES multi-machine surrogate protocol v0.1

**Freeze point:** committed after the 84-case reference-feasibility sweep and
before training or inspecting any learned multi-machine surrogate.

## Research question

Can a compact neural surrogate reproduce the generator electromechanical
response of successful IEEE-14 ANDES fault simulations while an explicit,
validation-calibrated trust layer refuses cases where the surrogate should not
be trusted?

The first model is intentionally narrower than a full network surrogate.

## Inputs

For each time point:

- physical time `t`;
- fault duration;
- fault-bus categorical encoding (14 buses).

Fault application remains fixed at 1.0 s.

## Outputs

The first multi-machine model predicts, for all five GENROU machines:

- rotor angle `delta`;
- rotor speed `omega`;
- mechanical torque `tm`;
- electrical torque `te`.

Bus-voltage prediction is retained as a later extension rather than making the
first experiment unnecessarily high-dimensional.

## Physics loss

The first physics-informed residuals are the shared GENBase electromechanical
equations used by ANDES:

```text
d(delta_i)/dt = 2*pi*fn_i*(omega_i - 1)

M_i*d(omega_i)/dt = tm_i - te_i - D_i*(omega_i - 1)
```

The machine constants `M`, `D`, and `fn` are taken directly from the
reference case. Derivatives of predicted angle and speed are obtained through
autograd.

This is a **swing-equation-informed GENROU surrogate**, not a claim that every
GENROU flux/electromagnetic DAE is enforced. A later version may derive the
full residual set from ANDES symbolic equations.

## Frozen split

The seven buses that completed all six v0.2 reference durations are:

`3, 6, 7, 8, 9, 11, 14`.

For each of those buses:

- **train:** 0.04, 0.08, 0.12 s — 21 cases;
- **validation:** 0.06 s — 7 cases;
- **fresh ID test:** 0.10 s — 7 cases;
- **duration OOD:** 0.14 s — 7 cases.

The **location OOD / edge set** contains the 21 successful reference cases on
buses 1, 2, 4, 5, 10, and 13. None of those bus categories appears in
training.

Bus 12 and the remaining failed cases stay in the separate reference-feasibility
record because they do not provide complete trajectories against which
surrogate error can be measured.

## Trust rules

The gate is not allowed to learn from either test set.

1. A categorical location not observed in training is an explicit OOD signal
   and is rejected by the v0.1 gate.
2. Duration extrapolation beyond the training maximum is an OOD signal but is
   still evaluated to determine whether physics residual can distinguish
   accurate from inaccurate trajectories.
3. The continuous physics-residual threshold is calibrated using validation
   trajectories only.
4. ID and duration-OOD test metrics are opened only after architecture,
   optimisation, and gate thresholds are frozen.

## Accuracy reporting

Primary trajectory metrics:

- maximum-over-machines rotor-angle RMSE;
- maximum-over-machines speed RMSE;
- torque RMSE as an auxiliary metric;
- worst-case absolute errors.

For continuity with the SMIB study, the initial demonstrator quality screen is:

- max machine angle RMSE <= 0.05 rad;
- max machine speed RMSE <= 5e-4 pu.

These remain research-screening tolerances, not grid-code, stability, or
protection criteria.

## Required comparisons

Report:

- always accept;
- physics-residual gate;
- OOD-only gate;
- combined deterministic gate.

Primary trust outcomes remain false accepts, accepted-case false-accept rate,
coverage, and false escalations.

## Stop rule

If the first multi-machine model is too inaccurate for non-trivial coverage,
report the negative result and improve the surrogate under a **new frozen test
set**. Do not relax the error thresholds after seeing test performance.
