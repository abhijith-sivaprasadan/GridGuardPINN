# Research protocol — v0.1

## Research question

Can an explicit deterministic trust gate identify when a PINN surrogate for
power-system dynamics should be accepted and when the workflow should fall
back to a physics-based reference simulation?

The initial system is a classical SMIB swing-equation model. The reduced model
is a controlled environment for testing the verification method, not an end in
itself.

## Primary hypotheses

**H1 — surrogate validity is local.** A PINN trained on a bounded operating
region will show materially larger trajectory error for at least some stressed
or out-of-distribution combinations of inertia, damping, fault severity, and
clearing time.

**H2 — trust signals contain useful information.** Physics-residual magnitude
and parameter-space OOD score will be associated with trajectory error, but
neither is assumed to be a proof of correctness.

**H3 — a frozen gate can reduce unsafe surrogate use.** A deterministic gate
calibrated only on train/validation cases can lower false accepts on untouched
test/OOD cases relative to always accepting the surrogate, while retaining
non-zero surrogate coverage.

## Frozen v0.1 scenario variables

The first parametric surrogate conditions on:

- inertia constant H;
- damping D;
- fault clearing time;
- fault-on transfer ratio Pmax_fault/Pmax_pre;
- physical time t.

Mechanical input power, pre/post-fault transfer limits, system frequency, and
fault application time are fixed in v0.1 to prevent the first experiment from
becoming unnecessarily high-dimensional.

## Split discipline

`canonical_splits()` defines train, validation, and OOD sets.

- Training cases fit the surrogate and the OOD feature distribution.
- Validation cases may tune gate thresholds and the acceptable trajectory-error tolerance.
- OOD cases are untouched until the gate is frozen.

No threshold may be selected by looking at OOD false-accept performance.

## Reference solver

The numerical reference integrates each smooth interval separately and restarts
exactly at the fault-application and fault-clearing boundaries. This avoids
silently stepping across a discontinuous transfer limit. Solver tolerances are
stricter than needed for the intended surrogate-error scale.

The SciPy solution is a **reference numerical solution**, not an exact solution.

## Surrogate error

The first trajectory-level error metrics will include:

- rotor-angle RMSE;
- speed-deviation RMSE;
- maximum absolute rotor-angle error;
- maximum absolute speed error.

A single aggregate metric will not be treated as sufficient evidence.

## Trust signals

v0.1 exposes:

1. RMS physics residual of the PINN trajectory;
2. Mahalanobis distance in scenario-parameter space;
3. an optional uncertainty signal for later ensemble/dropout experiments.

The first deterministic gate accepts only if all enabled signals are below
thresholds frozen on validation data.

## Primary gate metric

The main safety-oriented metric is **false accepts**:

> a case accepted by the gate even though its reference-solver trajectory error
> exceeds the predefined tolerance.

Report both:

- false accepts / all cases;
- false accepts / accepted cases.

Secondary metrics:

- surrogate coverage;
- false escalations;
- error distribution among accepted cases;
- reference-solver calls avoided.

## Stress tests

After the baseline model is stable:

- low/high inertia outside training range;
- low/high damping outside training range;
- longer fault clearing;
- unseen fault severity;
- combined shifts, not only one-factor-at-a-time tests.

## Upgrade path

The SMIB study is complete only when the gate is evaluated on untouched OOD
cases. The next scientific upgrade is to repeat the protocol with an
open-source transient-stability simulator (planned: ANDES or another suitable
reference implementation) and a standard multi-machine case.

An LLM may later orchestrate tool calls, but **must not replace the deterministic
trust decision** unless a separate study establishes that doing so is safe.

## Reporting rule

Negative results are valid results. If residual/OOD signals do not predict
trajectory failures well enough to support a useful gate, the repository should
report that directly rather than tune around the failure.
