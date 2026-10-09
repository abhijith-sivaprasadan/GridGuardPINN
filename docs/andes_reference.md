# ANDES reference integration

GridGuardPINN's reduced-order SMIB work is a method-development phase. The next
reference layer uses ANDES, an open-source differential-algebraic power-system
simulation framework, so later learned surrogates can be evaluated against a
standard multi-machine transient-stability simulator.

## First reference case

The initial adapter uses ANDES's packaged Kundur system and programmatically
adds a three-phase fault:

- fault bus: 5;
- fault application: 1.0 s;
- fault clearing: 1.1 s;
- simulation end: 5.0 s.

The packaged Kundur Toggle is disabled so this trajectory isolates the added
fault.

## Extracted quantities

After ANDES power flow and time-domain simulation complete successfully:

- simulation time is read from `system.dae.ts.t`;
- all `GENROU.omega` state trajectories are read from
  `system.dae.ts.x[:, system.GENROU.omega.a]`;
- all bus-voltage trajectories are read from
  `system.dae.ts.y[:, system.Bus.v.a]`.

The adapter copies those arrays immediately and rejects non-finite outputs.

The native adaptive/variable TDS time stamps are then linearly resampled to a
fixed grid for later machine-learning use. Both native-run summary metrics and
the deterministic-grid trajectory are exported.

## Why Kundur

The objective is not to reproduce the earlier SMIB equations inside another
software package. Kundur introduces a genuinely multi-machine reference with a
DAE simulation stack and network response. This gives the project direct
exposure to:

- transient-stability software;
- multi-generator rotor-speed trajectories;
- network bus-voltage dynamics;
- explicit three-phase fault events;
- reference-simulator failure/status handling.

## Current boundary

This integration does not yet train a neural surrogate on ANDES output. The
first milestone is narrower: prove that the reference simulation is
programmatically reproducible, extractable and testable.

Once that passes in CI, the next experiment will create a controlled ANDES
scenario sweep over fault location/duration and selected dynamic parameters,
freeze train/validation/OOD splits, and only then train the multi-output
surrogate.

## Reproduce

ANDES 2.0.0 requires Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,andes]"
python scripts/run_andes_smoke.py
```

Outputs:

```text
artifacts/andes_kundur_smoke/
  summary.json
  trajectory.csv
```
