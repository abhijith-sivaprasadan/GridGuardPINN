# ANDES reference integration

GridGuardPINN's reduced-order SMIB work is a method-development phase. The next
reference layer uses ANDES, an open-source differential-algebraic power-system
simulation framework, so later learned surrogates can be evaluated against a
standard multi-machine transient-stability simulator.

## Verified first reference case

The first successful CI-backed reference uses ANDES 2.0.0 and its packaged
dynamic IEEE-14 three-phase-fault case:

- 14 buses;
- 5 GENROU synchronous machines;
- fault bus: 9;
- fault application: 1.0 s;
- fault clearing: 1.1 s;
- simulation end: 2.0 s.

The GitHub Actions run completed power flow and TDS successfully, extracted 66
native time points, and resampled them to a deterministic 401-point grid.

Observed summary metrics from the successful run:

- maximum absolute generator-speed deviation: ~0.004913 pu;
- minimum bus voltage during the fault: ~0.000444 pu;
- maximum bus voltage: ~1.07932 pu.

These are simulator outputs for this test case, not operational acceptance
criteria.

## Extracted quantities

After ANDES power flow and time-domain simulation complete successfully:

- simulation time is read from `system.dae.ts.t`;
- all `GENROU.omega` state trajectories are read from
  `system.dae.ts.x[:, system.GENROU.omega.a]`;
- all bus-voltage trajectories are read from
  `system.dae.ts.y[:, system.Bus.v.a]`.

The adapter copies those arrays immediately and rejects non-finite outputs.

The native TDS timestamps are then linearly resampled to a fixed grid for later
machine-learning use.

## Parameterized reference path

The adapter also supports adding a three-phase fault programmatically to the
packaged dynamic `ieee14/ieee14.json` base case before ANDES system setup.

This is the path intended for controlled scenario sweeps over:

- fault location;
- fault duration;
- eventually selected physical machine parameters.

Before using that path for a learned surrogate, CI cross-checks a programmatic
bus-9, 1.0–1.1 s, xf=1e-4 case against the packaged fixed fault case on the same
resampled grid.

## A useful failed robustness case

The first attempt used the Kundur system with a severe custom bus-5 fault and
xf=1e-6. ANDES installed and initialized, but TDS reduced its timestep to zero
at fault application and terminated with exit code 1.

That run is not presented as a successful reference. It is retained as useful
evidence that the adapter must explicitly handle simulator non-convergence
rather than silently dropping difficult cases.

## Why IEEE-14 is the next step beyond SMIB

The objective is not to reproduce the earlier swing equation inside another
software package. IEEE-14 introduces:

- a multi-machine DAE simulation;
- five synchronous-generator speed trajectories;
- network bus-voltage dynamics;
- explicit three-phase fault events;
- simulator convergence/failure status;
- a maintained external power-system simulation package.

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
artifacts/andes_ieee14_smoke/
  summary.json
  packaged_trajectory.csv
  programmatic_trajectory.csv
```

## Next milestone

After the programmatic fault cross-check is verified, freeze a small
simulator-backed scenario protocol over fault locations and clearing durations,
generate reference trajectories, and only then define/train the first
multi-output learned surrogate.

The SMIB trust-gate results remain a regression benchmark; they are not reused
as if they were evidence of multi-machine performance.
