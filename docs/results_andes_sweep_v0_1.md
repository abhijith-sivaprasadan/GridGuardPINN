# ANDES IEEE-14 fault sweep result v0.1

Run: GitHub Actions `ANDES IEEE-14 sweep`  
Reference: ANDES 2.0.0, dynamic `ieee14/ieee14.json` + programmatic three-phase fault  
Scenario matrix: 6 fault buses × 3 durations = 18 cases

## Outcome

The frozen feasibility sweep completed and retained simulator failures explicitly.

- planned cases: **18**
- successful cases: **11**
- failed cases: **7**
- success fraction: **61.1%**
- successful max generator-speed deviation range: **0.003143–0.010237 pu**
- successful minimum bus-voltage range: **0.0002775–0.0006790 pu**

### Failure matrix

| Fault bus | 0.06 s | 0.10 s | 0.14 s |
|---:|:---:|:---:|:---:|
| 2 | success | success | failed |
| 4 | success | failed | success |
| 5 | success | failed | failed |
| 9 | success | success | success |
| 12 | failed | failed | failed |
| 14 | success | success | success |

The non-monotonic success pattern at bus 4 is a warning against interpreting
reference-simulator completion as a simple physical stability boundary.

## Failure behaviour

The workflow logs contained two broad failure modes:

- ANDES stability-criteria termination after the disturbed trajectory evolved;
- time-step collapse / convergence failure close to fault application or clearing.

The v0.1 case table records these as failed reference simulations. They are not
silently removed or relabelled as physical instability.

## Interpretation

This result establishes that the programmatic IEEE-14 fault generator is useful
but **not uniformly robust over the first scenario matrix**. Training a neural
surrogate only on the 11 successful survivors and calling it a general IEEE-14
surrogate would therefore overstate the supported domain.

The next reference protocol expands the feasibility map before freezing any
multi-machine ML split. It also exports GENROU rotor angle in addition to speed
and bus voltage, because angle/speed pairs are required for meaningful
machine-dynamics residual tests.

## What this supports

- reproducible multi-machine transient simulation in ANDES;
- automated location/duration disturbance sweeps;
- explicit simulator-success/failure accounting;
- deterministic trajectory resampling for learned-surrogate datasets.

## What this does not support

- a critical-clearing-time result;
- a protection or stability-limit study;
- a PINN accuracy result;
- treating every ANDES failure as physical system instability;
- a claim that the 11 successful cases define the full valid operating domain.
