# GridGuardPINN

**Trust-aware physics-informed neural-network surrogates for power-system dynamic simulation.**

GridGuardPINN studies a narrower question than “can a PINN imitate a dynamic simulator?”:

> **When is a fast PINN surrogate accurate enough to use, and when should the workflow deterministically fall back to the reference simulator?**

The project uses a classical single-machine infinite-bus (SMIB) swing-equation system as a controlled first testbed, with explicit train/calibration/held-out protocols, physics-informed learning, out-of-distribution (OOD) stress tests, and a trust gate evaluated as its own research object.

> **v1.0 research demonstrator (October 2026):** This repository provides reproducible surrogate experiments and a fail-closed ANDES case-routing policy, **not** a certified safe grid-control application. See [v1.0 release scope](docs/v1_0_release_scope.md), [frozen v0.3 location-encoding results](docs/results_andes_location_encoding_v0_3.md), and [conservative router](src/gridguardpinn/andes_router.py). The network-aware model improves unseen-location error in 2/3 seeds but **residual-only gating falsely accepts inaccurate unseen-location predictions**, so the v1.0 default rejects all unseen fault buses.\n\n## Published v1.0.0 research release

The versioned [v1.0.0 GitHub Release](https://github.com/abhijith-sivaprasadan/GridGuardPINN/releases/tag/v1.0.0) contains the original frozen model archive, case-level validation and randomized timing ledgers, two vector figures, exact source commit, resolved dependency snapshot, and SHA-256 manifest. The released tag currently points to commit `839fd6e103c5b3ff249e3e550e08ffb5c690c54f`. **It is a research demonstrator, not operational grid software or a peer-reviewed publication.**

## Research working paper

The [GridGuardPINN v1.0 technical working paper](docs/technical_working_paper_v1_0.md) brings together the SMIB development sequence, matched physics-loss ablations, pre-registered network-aware encoding experiment, conservative trust routing, independent-duration checks and randomized end-to-end timing. It includes research questions, mathematical definitions, figures, negative results, uncertainty analysis, limitations and source-linked reproducibility evidence. **Working paper, not peer reviewed or formally published.**

## Pre-registered v1.0 release-gate evaluation (October 2026)

The [frozen new-duration release gate](docs/results_andes_v1_fresh_duration.md) **passed**: three previously trained one-hot models accepted **76/84 (90.5%)** new interpolation-duration evaluations with **zero observed inaccurate accepts**, and sent **21/21** new longer-duration cases to ANDES. All 35 unique reference scenarios succeeded. [CI and case-level artifacts](https://github.com/abhijith-sivaprasadan/GridGuardPINN/actions/runs/37925430499). These are previously unused duration combinations **on the same IEEE-14 grid**, not independent topologies or an operational safety guarantee.

## Repeated randomized full-route timing (October 2026)

A [pre-registered randomized paired replication](docs/results_andes_v1_randomized_performance.md) now supersedes the single-pass benchmark as the preferred performance estimate: with **three rounds, 14 fault cases per round and three frozen model seeds**, the fully routed hybrid achieved **1.57–1.95× speedup** against freshly timed ANDES-only solves on the fixed 50% in-envelope / 50% OOD workload. The **95% bus-cluster bootstrap intervals** were 1.24–2.02×, 1.43–2.08× and 1.87–2.03× across the seeds. No false accepts were observed. The comparison excludes training and installation, and does **not** establish safety or throughput on other grids. [Workflow and complete data](https://github.com/abhijith-sivaprasadan/GridGuardPINN/actions/runs/37939032862).

## End-to-end performance (October 2026)

The [actual ANDES/PINN/full-router benchmark](docs/results_andes_v1_performance.md) found a **1.99–2.95× total wall-time reduction** against reference-only execution on a fixed 50% eligible / 50% duration-OOD mixture across three frozen seeds. This includes physics-residual computation and **real ANDES fallback**; no false surrogate accepts were observed on these cases. **OOD-only routing was not faster** than using ANDES directly, and large in-envelope ratios are sensitive to reference startup and workload order. These are CI-host-specific research measurements, not production throughput or guaranteed gains. [Workflow and numerical artifacts](https://github.com/abhijith-sivaprasadan/GridGuardPINN/actions/runs/37928033388).

## Trained-model runtime audit (October 2026)

A [pinned-checkpoint, real-ANDES routing audit](docs/results_andes_trained_runtime_v1_candidate.md) has now passed CI using the three **original one-hot seed-17/29/41 checkpoints**. Of 105 retrospectively evaluated seed-case combinations, the strict route returned 13 surrogate trajectories, escalated 92 cases to ANDES, and observed zero inaccurate surrogate acceptances. All unseen locations and duration-OOD cases escalated to the independent simulator. **These are previously inspected cases, not a new blind test and not evidence of operational safety.** The [workflow and downloadable ledger](https://github.com/abhijith-sivaprasadan/GridGuardPINN/actions/runs/37924679301) allow inspection of every decision.

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

### ANDES multi-machine surrogate v0.1

The first frozen IEEE-14 multi-machine surrogate is a 20-output model covering
five GENROU machines' rotor angle, speed, mechanical torque, and electrical
torque. The untouched seed-17 result was substantially stronger than the SMIB
development sequence:

| Split | Cases | Good cases | Mean composite ratio | Residual/error Spearman rho |
|---|---:|---:|---:|---:|
| Validation | 7 | **7/7** | 0.146 | 0.714 |
| Fresh ID test | 7 | **7/7** | 0.152 | 0.786 |
| Duration OOD (0.14 s) | 7 | **7/7** | 0.420 | 0.643 |
| Unseen-location edge set | 21 | **0/21** | 6.079 | 0.412 |

The important trust result is mixed. The validation-calibrated residual gate
made zero false accepts, but accepted only 4/7 fresh-ID cases and rejected all
seven accurate duration-OOD cases. In contrast, always accepting unseen fault
locations would have produced 21/21 false accepts.

This shifts the research bottleneck from basic trained-location surrogate
accuracy to **whether physics improves generalisation and whether the trust
signal can remain selective without becoming over-conservative**. The
pre-registered data-only vs physics-informed ablation is now executable across
the frozen seeds 17/29/41.

See [ANDES surrogate v0.1 result](docs/results_andes_surrogate_v0_1.md).

### Pre-registered physics ablation v0.2

The three-seed data-only vs physics-informed comparison is now complete. Both
arms pass all validation, fresh-ID, and 0.14 s duration-OOD cases, while both
fail every unseen-location case with the current categorical bus encoding.

The physics-informed arm produces a consistent **11.9-19.8% reduction in
fresh-ID mean composite error across seeds 17/29/41**, but duration-OOD
mean-error benefit is mixed and residual/error ranking is not consistently
better. The validation-calibrated residual gate still rejects almost every
accurate duration-OOD case. Physics-informed training costs about **6.2x** more
than data-only training with essentially identical inference cost.

That is the current research result: swing-equation regularisation improves
trained-location precision, but the trust layer and fault-location
representation remain the real bottlenecks.

See the [v0.2 ablation result](docs/results_andes_surrogate_ablation_v0_2.md)
and its [pre-registration](docs/andes_surrogate_ablation_plan_v0_2.md).

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
  andes_reference.py # ANDES IEEE-14 transient reference adapter
  andes_surrogate_protocol.py # frozen multi-machine train/validation/test cases
  andes_surrogate.py # multi-machine swing-equation-informed neural surrogate
  andes_multimachine.py # reference-data assembly for ANDES experiments
  andes_training.py  # multi-machine training loop
  andes_evaluation.py # trajectory/residual evaluation + inference timing
  andes_gate.py      # validation-only multi-machine gate calibration
  andes_experiment.py # reusable experiment/artifact engine

scripts/
  run_reference_sweep.py
  run_experiment.py
  run_andes_sweep_v02.py
  run_andes_surrogate_experiment.py
  run_andes_ablation.py

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

## Run the first ANDES multi-machine surrogate experiment

The frozen v0.1 experiment uses the 63 successful IEEE-14 reference cases
partitioned before training into 21 train, 7 validation, 7 fresh ID test,
7 duration-OOD, and 21 unseen-location edge cases.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,andes]"
python scripts/run_andes_surrogate_experiment.py --epochs 1000 --anchors 101 --collocation 56 --seed 17
```

The first multi-machine model predicts five machines' rotor angle, speed,
mechanical torque, and electrical torque. Its physics loss enforces the shared
ANDES GENBase electromechanical equations. It is therefore described as a
**swing-equation-informed GENROU surrogate**, not as a full GENROU-DAE PINN.


## Run the pre-registered multi-machine physics ablation

The v0.2 ablation keeps the architecture, frozen case split, optimiser schedule,
output scaling, and validation checkpointing rule fixed. The only planned
training difference is the swing-equation physics-loss weight:

- data-only: `physics_weight = 0`;
- physics-informed: `physics_weight = 0.02`.

The default command runs the three frozen seeds (17, 29, 41) while generating
the 63-case ANDES reference batch only once:

```bash
python scripts/run_andes_ablation.py --epochs 1000 --anchors 101 --collocation 56
```

Each arm/seed writes its own model, case metrics, training history, and summary.
The top-level artifact directory additionally writes `seed_comparison.csv` and
`aggregate.json` with median/range results across seeds. Evaluation now records
per-case 401-point surrogate inference time and the wall-clock cost of the
corresponding ANDES reference simulation.

## Multi-machine reference milestone

The reduced-order SMIB phase is now complemented by a CI-verified **ANDES 2.0.0 IEEE-14 transient-stability reference path**.

The first frozen location/duration sweep evaluated 18 three-phase-fault cases across six buses and three clearing durations. **11/18 completed successfully and 7/18 were retained as explicit simulator failures**, rather than silently removed. The expanded frozen sweep then evaluated **84 cases across all 14 buses and six durations: 63/84 succeeded (75%)**. Seven buses (3, 6, 7, 8, 9, 11, 14) completed all six durations, creating a 42-case robust core domain for the first multi-machine surrogate protocol.

See [ANDES reference result](docs/results_andes_reference_v0_1.md), [18-case sweep result](docs/results_andes_sweep_v0_1.md), [84-case sweep result](docs/results_andes_sweep_v0_2.md), and the frozen [multi-machine surrogate protocol](docs/andes_surrogate_protocol_v0_1.md).

## Roadmap

- **M0 — reference dynamics:** complete.
- **M1 — parametric SMIB PINN baseline:** complete.
- **M2 — deterministic trust gate:** implemented and evaluated.
- **M3 — SMIB stress testing:** complete enough for method development; v0.4 preserved as regression evidence.
- **M4 — ANDES multi-machine reference:** active; IEEE-14 dynamic faults and explicit simulator-failure handling are CI verified.
- **M5 — multi-machine surrogate:** active; the frozen v0.1 20-output GENROU surrogate is implemented and evaluated through the reusable experiment engine.
- **M6 — multi-machine trust layer:** active; validation-only residual calibration, explicit location/duration OOD flags, timing metrics, and the pre-registered data-only vs physics-informed ablation are implemented.
- **M7 — orchestration:** only after the trust mechanism is useful, add a tool-calling orchestration layer whose job is workflow routing, not safety judgement.

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
