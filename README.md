# GridGuardPINN

**Trust-aware physics-informed neural-network surrogates for power-system dynamic simulation.**

GridGuardPINN is a research-oriented project for studying a narrow question:

> When is a fast PINN surrogate reliable enough to use, and when should a dynamic-simulation workflow fall back to a higher-fidelity reference solver?

The project starts with a classical single-machine infinite-bus (SMIB) swing-equation system and deliberately separates four layers:

1. **Reference dynamics** — reproducible numerical integration of the governing equations under fault/clearing disturbances.
2. **PINN surrogate** — a physics-informed neural network trained to reproduce dynamic trajectories while satisfying the swing-equation residual.
3. **Trust layer** — explicit physics-residual / out-of-distribution signals and a deterministic acceptance gate.
4. **Fallback path** — rejected cases are escalated to the reference simulator rather than silently trusted.

An LLM orchestration layer is intentionally **not** part of v0.1. The trust decision must first be independently measurable and auditable.

## Current scope

v0.1 establishes the scientific contract and testable software skeleton:

- SMIB swing-equation model with configurable inertia, damping, mechanical power, transfer limits, and fault/clearing times.
- SciPy reference simulation with strict tolerances and reproducible scenario grids.
- In-distribution and deliberately out-of-distribution scenario splits.
- PyTorch PINN baseline and physics-residual calculation.
- Deterministic trust-gate API.
- Gate metrics including coverage, false accepts, false escalations, and accepted-case error.
- Unit tests and CI.

## Governing model

For rotor-angle deviation `delta` and per-unit speed deviation `omega`:

```text
d(delta)/dt = omega_b * omega

d(omega)/dt = [Pm - Pe(delta, t) - D*omega] / (2H)

Pe(delta, t) = Pmax(t) * sin(delta)
```

`Pmax(t)` is piecewise-defined across pre-fault, fault-on, and post-fault intervals. The default initial rotor angle is the pre-fault equilibrium:

```text
delta_0 = asin(Pm / Pmax_pre)
omega_0 = 0
```

This is a deliberately reduced-order educational/research model. It is **not** a planning-grade or protection-grade transient-stability tool.

## Trust-gate concept

The intended decision path is:

```text
dynamic case
   |
   v
PINN surrogate
   |
   +--> physics-residual signal
   +--> OOD signal
   +--> optional uncertainty signal
   |
   v
deterministic trust gate
   | accepted                 | rejected
   v                          v
surrogate result       reference simulation
```

The gate is evaluated as its own research object. In particular, a **false accept** — trusting a surrogate case whose true trajectory error exceeds the allowed tolerance — is treated as the most important failure mode.

## Repository layout

```text
src/gridguardpinn/
  dynamics.py      # SMIB equations and scenario definition
  reference.py     # high-accuracy numerical reference simulation
  scenarios.py     # reproducible ID/OOD scenario generation
  trust.py         # OOD model, deterministic gate, gate metrics
  pinn.py          # PyTorch PINN model and physics residuals

scripts/
  run_reference_sweep.py
  train_baseline.py

tests/
docs/
  research_protocol.md
```

## Quick start

Core reference-solver functionality:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
python scripts/run_reference_sweep.py
```

For the PINN components:

```bash
pip install -e ".[dev,ml]"
python scripts/train_baseline.py --epochs 2000
```

The training script is a baseline, not a reported benchmark. Reproducible experiment results will be added only after the scenario protocol and trust metrics are frozen.

## Research roadmap

- **M0 — reference model:** equations, fault cases, tests, reproducible scenario splits.
- **M1 — PINN baseline:** trajectory + physics-residual learning on in-distribution SMIB cases.
- **M2 — trust gate:** calibrate residual/OOD thresholds on validation cases; report false accepts / false escalations on untouched test cases.
- **M3 — stress testing:** inertia, damping, fault severity, clearing time, and unseen combinations.
- **M4 — simulator upgrade:** connect the reference side to an open-source power-system dynamic simulator such as ANDES and repeat the trust experiment on a standard system.
- **M5 — orchestration:** only after M2–M4, add a tool-calling orchestration layer whose role is workflow routing, not the safety decision itself.

## Scientific boundaries

GridGuardPINN is currently a research demonstrator. It does not claim:

- operational or protection-grade validation;
- equivalence to commercial tools such as PowerFactory, PSCAD, EMTP, PSS/E, or TSAT;
- that physics residual alone proves predictive correctness;
- that a deterministic gate developed on SMIB cases transfers automatically to multi-machine grids;
- that an LLM should decide whether a surrogate is safe.

Those are precisely the boundaries the project is intended to investigate.

## Implementation note

AI-assisted implementation is used in this repository. The project owner defines the research question, model formulation, assumptions, verification criteria, experiment design, and interpretation of results, and remains responsible for validating the implementation.

## License

MIT.
