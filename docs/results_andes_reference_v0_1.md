# ANDES reference result v0.1 — IEEE-14 fault integration

## Runtime

- ANDES: 2.0.0
- CI: GitHub Actions, Python 3.11
- packaged reference: `ieee14/ieee14_fault.xlsx`
- programmatic base: `ieee14/ieee14.json` + added `Fault`
- fault: bus 9, 1.0–1.1 s, rf=0, xf=1e-4
- simulation end: 2.0 s
- comparison grid: 401 points

## Packaged case

The packaged maintained fault case completed successfully.

- 5 GENROU synchronous machines
- 14 buses
- 66 native TDS time points
- maximum absolute generator-speed deviation: 0.00491308 pu
- minimum bus voltage: 0.000444015 pu
- maximum bus voltage: 1.079321 pu

## Programmatically generated case

The dynamic IEEE-14 JSON base with an added `Fault` also completed successfully.

- 5 GENROU synchronous machines
- 14 buses
- 68 native TDS time points
- maximum absolute generator-speed deviation: 0.00586273 pu
- minimum bus voltage: 0.000442570 pu
- maximum bus voltage: 1.089957 pu

## Cross-check

After independently resampling both successful simulations onto the same
401-point time grid:

- maximum absolute generator-speed difference: **0.00137038 pu**
- maximum absolute bus-voltage difference: **0.0297013 pu**

The two trajectories are therefore close enough to demonstrate that the
programmatic fault path produces the intended type of dynamic event, but they
are **not claimed to be numerically equivalent**.

That difference is retained rather than tuned away. The packaged XLSX and the
dynamic JSON base should be treated as separate source cases with overlapping,
not identical, model content.

## Robustness note

An earlier custom Kundur bus-5 fault with xf=1e-6 caused ANDES TDS to reduce its
time step to zero at fault application and terminate with exit code 1. The
adapter correctly surfaced that failure.

This is useful for the next dataset stage: simulator non-convergence is an
outcome that must be logged per scenario, not silently removed from the
experiment.

## Milestone status

The project now has direct, CI-verified experience with:

- ANDES 2.0.0 installation and scripting;
- power-flow + time-domain simulation;
- three-phase fault dynamics;
- five-machine rotor-speed extraction;
- fourteen-bus voltage extraction;
- TDS native-time handling and deterministic resampling;
- simulator failure/status handling;
- programmatic disturbance construction.

The next step is a fault-location / fault-duration reference sweep. It is a
simulator-feasibility/data-generation phase, not yet a learned-surrogate result.
