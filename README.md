# GridGuardPINN

**Trust-aware physics-informed neural-network surrogates for power-system dynamic simulation.**

GridGuardPINN studies a narrower question than “can a PINN imitate a dynamic simulator?”:

> **When is a fast PINN surrogate accurate enough to use, and when should the workflow deterministically fall back to the reference simulator?**

The project uses a classical single-machine infinite-bus (SMIB) swing-equation system as a controlled first testbed, with explicit train/calibration/held-out protocols, physics-informed learning, out-of-distribution (OOD) stress tests, and a trust gate evaluated as its own research object.

## Why this project exists

Average surrogate accuracy is not enough for safety-relevant engineering workflows. A useful surrogate needs both:

1. **accuracy** inside a stated operating envelope; and
2. **a reliable refusal mechanism** when a case is outside that envelope or the surrogate is behaving poorly.

GridGuardPINN therefore separates:

```text
dynamic case
    |
    v
PINN surrogate
    |
    +--> physics-residual signal
    +--> parameter-space OOD signal
    |
    v
deterministic trust gate
    | accept                     | reject
    v                            v
surrogate trajectory       reference simulation
```

An LLM may eventually orchestrate this workflow, but it is deliberately **not** the safety gate.

## Current evidence

### v0.2 — uniform-sampling baseline

The first parametric PINN was intentionally evaluated before tuning around held-out results.

| Split | Cases | Mean composite error ratio | Good cases | Combined-gate coverage |
|---|---:|---:|---:|---:|
| Validation | 16 | 8.23 | 0 | 0% |
| ID test | 16 | 8.20 | 0 | 0% |
| OOD test | 16 | 941.31 | 0 | 0% |

This was a useful negative result: the surrogate was too inaccurate for the trust gate to do anything except reject every case.

See [v0.2 baseline result](docs/results_v0_2_baseline.md).

### v0.3 — event-aware surrogate

Before changing the model, a **fresh held-out set** was frozen. v0.3 then added event-aware supervised sampling, phase-stratified physics collocation, explicit fault/post-fault features, an improved hard initial-condition transform, and a data-first physics-loss curriculum.

| Split | Cases | Good cases | Mean composite error ratio | Residual/error Spearman ρ |
|---|---:|---:|---:|---:|
| Validation | 16 | 2 | 1.97 | 0.921 |
| Fresh ID test | 24 | 2 | 1.81 | 0.886 |
| Fresh OOD test | 24 | 0 | 3.66 | 0.906 |

On the **fresh ID test**, always accepting the surrogate produced 22 false accepts in 24 cases. The validation-calibrated combined gate accepted 1/24 cases with **zero false accepts**, but coverage was only **4.17%**.

On the fresh OOD set, all 24 surrogate trajectories violated the provisional accuracy tolerance and the combined gate rejected all 24. Parameter-space OOD detection by itself was insufficient: it accepted 8/24 OOD cases, all of which were bad trajectories.

The important v0.3 result is therefore not “the surrogate works.” It is:

> **Event-aware training materially improved the surrogate, and PINN physics-residual magnitude became strongly associated with true trajectory error, but the surrogate was still not accurate enough for useful high-coverage routing.**

See [v0.3 result](docs/results_v0_3.md).


### v0.4 — Fourier/event-aware surrogate

A third fresh holdout was frozen before increasing model capacity or adding
Fourier time features.

| Split | Cases | Good cases | Mean composite error ratio | Combined-gate coverage | Combined false accepts |
|---|---:|---:|---:|---:|---:|
| Validation | 16 | 12 | 0.795 | 75.0% | 0 |
| Fresh ID test | 32 | **23** | **0.904** | **65.6%** | **1** |
| Fresh OOD test | 32 | 12 | 48.07* | 18.8% | 1 |

\* OOD mean is dominated by extreme clearing-time failures; OOD median is 1.162.

On the fresh ID test, always accepting the surrogate would have produced
9/32 false accepts. The validation-calibrated combined gate reduced that to
**1/32** while still routing **21/32 cases through the surrogate**.

This is the first version with both majority-case ID accuracy and meaningful
safe-routing coverage. It is still not fail-safe: one fresh-ID false accept and
one OOD false accept remain.

See [v0.4 result](docs/results_v0_4.md).

## Accuracy definition

For the SMIB demonstrator, a trajectory is provisionally labelled acceptable only if both are satisfied:

- rotor-angle RMSE ≤ **0.05 rad**;
- speed-deviation RMSE ≤ **5×10⁻⁴ pu**.

The composite error ratio is:

```text
max(
    rotor-angle RMSE / 0.05 rad,
    speed RMSE / 5e-4 pu
)
```

A composite ratio ≤ 1 is “good” for this experiment.

These are **transparent research-screening thresholds**, not industry protection, stability, or grid-code acceptance standards.

## Governing model

For rotor angle `delta` and per-unit speed deviation `omega`:

```text
d(delta)/dt = omega_b * omega

d(omega)/dt = [Pm - Pe(delta, t) - D*omega] / (2H)

Pe(delta, t) = Pmax(t) * sin(delta)
```

`Pmax(t)` is piecewise-defined across pre-fault, fault-on, and post-fault intervals. The reference solver restarts exactly at event boundaries rather than stepping blindly across the discontinuity.

The default initial state is the pre-fault equilibrium:

```text
delta_0 = asin(Pm / Pmax_pre)
omega_0 = 0
```

## Experimental discipline

The project treats data-split discipline as part of the engineering result.

- Gate thresholds are calibrated on **validation only**.
- Held-out ID/OOD cases do not set thresholds.
- Once a held-out set has been inspected, it is never described as “blind” for a tuned successor model.
- New model iterations freeze a **new held-out set before evaluation**.
- Negative results are recorded rather than hidden by moving the acceptance threshold.

Protocols and results are versioned in `docs/`.

## Trust signals

### Physics residual

The PINN trajectory is differentiated with autograd and tested against the governing swing equations away from fault-switching discontinuities.

Residual magnitude is **not treated as proof of correctness**. It is evaluated empirically against actual trajectory error from the reference solver.

### Parameter-space OOD score

A Mahalanobis-distance detector is fit only on training-scenario parameters:

- inertia `H`;
- damping `D`;
- fault-clearing time;
- fault-on transfer ratio.

v0.3 and v0.4 show why this cannot be the sole safety mechanism: parameter-space proximity does not guarantee trajectory accuracy.

## Gate metrics

The primary safety failure is a **false accept**:

> the gate accepts a surrogate trajectory whose reference-solver error exceeds the frozen tolerance.

Reported metrics include:

- false accepts / all cases;
- false accepts / accepted cases;
- surrogate coverage;
- false escalations;
- accepted-case error;
- residual/error association;
- comparison against always-accept, OOD-only, and residual-only routing.

## Repository layout

```text
src/gridguardpinn/
  dynamics.py       # SMIB equations and scenario definition
  reference.py      # segmented high-accuracy numerical reference
  scenarios.py      # versioned train/validation/ID/OOD protocols
  dataset.py        # event-aware reference anchors + collocation points
  pinn.py           # parametric event-aware PINN
  training.py       # reproducible curriculum training
  evaluation.py     # trajectory/residual/OOD evaluation
  calibration.py    # validation-only trust-gate calibration
  trust.py          # trust signals and gate metrics

scripts/
  run_reference_sweep.py
  run_experiment.py

docs/
  research_protocol.md
  experiment_spec_v0_2.md
  results_v0_2_baseline.md
  experiment_spec_v0_3.md
  results_v0_3.md
  experiment_spec_v0_4.md

tests/
.github/workflows/
```

## Reproduce the core solver

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
python scripts/run_reference_sweep.py
```

## Run a PINN experiment

Install CPU/GPU PyTorch as appropriate, then:

```bash
pip install -e ".[dev]"
python scripts/run_experiment.py --protocol v0.4 --epochs 2500 --anchors 121 --collocation 96 --seed 7
```

The GitHub Actions experiment workflow records artifacts including:

- `summary.json`;
- `case_metrics.csv`;
- `training_history.json`;
- model checkpoint.

## Roadmap

- **M0 — reference dynamics:** complete.
- **M1 — parametric PINN baseline:** complete.
- **M2 — deterministic trust gate:** implemented and evaluated.
- **M3 — stress testing:** active; one-factor OOD mechanisms evaluated.
- **M4 — surrogate-accuracy refinement:** active under frozen successive holdouts.
- **M5 — simulator upgrade:** repeat the protocol using an open-source power-system transient-stability reference such as ANDES on a standard multi-machine case.
- **M6 — orchestration:** only after the trust mechanism is useful, add a tool-calling orchestration layer whose job is workflow routing, not safety judgement.

## Scientific boundaries

GridGuardPINN currently does **not** claim:

- operational, planning-grade, or protection-grade validation;
- equivalence to PowerFactory, PSCAD, EMTP, PSS/E, TSAT, or another commercial tool;
- that SMIB findings automatically transfer to multi-machine systems;
- that physics residual alone proves predictive correctness;
- that the current deterministic gate provides useful production-level coverage;
- that an LLM should decide whether a surrogate is safe.

The reduced-order SMIB phase is a method-development testbed. The planned simulator upgrade is required before making broader power-system claims.

## Implementation note

AI-assisted implementation is used in this repository. The project owner defines the research question, governing model, assumptions, experimental protocol, validation criteria, architecture decisions, and interpretation of results, and remains responsible for verifying the implementation.

## License

MIT.
