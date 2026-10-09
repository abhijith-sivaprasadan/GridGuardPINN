# GridGuardPINN v1.0 — preregistered new-duration routing results

**Executed 9 October 2026:** [GitHub Actions run 37925430499](https://github.com/abhijith-sivaprasadan/GridGuardPINN/actions/runs/37925430499), original trigger commit `9800a5917830f519f2034f0fd0a6d81b4d545f0a`. [Full case ledger and summary artifact](https://github.com/abhijith-sivaprasadan/GridGuardPINN/actions/runs/37925430499/artifacts/11613913010).

**Protocol frozen before triggering:** [andes_v1_fresh_duration_protocol.md](andes_v1_fresh_duration_protocol.md), original freeze commit `c0883d288769c62ecf9c2dfe6fbc3acf0328f6d0`. The model weights, seeds, reference configuration, thresholds, scenario list and accuracy screen were unchanged after outcomes were observed. A lint-only change to the benchmark script was committed during execution; it made no scientific modifications.

## New cases

The same original IEEE-14 testbed and same frozen one-hot v0.3 trained models were used. New fault-duration combinations: 28 interpolation cases per seed (seven previously used buses × durations `0.05, 0.07, 0.09, 0.11` seconds) plus seven longer-duration `0.13`-second OOD cases per seed. These duration values were absent from the prior train/validation/ID/duration-OOD protocol. These are new *parameter combinations*, not an independent power grid or new fault mechanism.

ANDES successfully generated all **35/35** distinct reference scenarios. The integrated reference-fallback route re-ran ANDES when necessary, and reference/surrogate outputs were compared with independently generated trajectories. No issues were recorded.

## Frozen results

| Seed | Interpolation surrogate accepts | Interpolation coverage | Interpolation false accepts | Duration-OOD fallback | Mean accepted composite ratio | Max accepted ratio |
|---|---:|---:|---:|---:|---:|---:|
| 17 | 24/28 | 85.7% | 0 | 7/7 | 0.1331 | 0.2245 |
| 29 | 27/28 | 96.4% | 0 | 7/7 | 0.1462 | 0.2338 |
| 41 | 25/28 | 89.3% | 0 | 7/7 | 0.1353 | 0.2111 |
| **Total** | **76/84** | **90.5%** | **0** | **21/21** | — | — |

All four pre-registered acceptance categories held in each seed (no interpolation false accepts, ≥50% interpolation coverage, all duration-OOD rejected, immutable checkpoint/digest checks and CI). The workflow reports `release_gate_pass=true`, with **no reference or runtime issues**.

The 105 seed-case observations include only **35** unique scenario conditions, reused across three independently trained seeds, and should not be presented as 105 independent physical-system validations.

## Interpretation and remaining limits

These results satisfy the *specified research-release acceptance protocol*. In particular, they provide evidence of usable surrogate coverage at new intermediate fault durations. They are not an operational safety certification. Coverage on unseen buses is still deliberately zero; reference fallback is required there. A zero false-accept count among 76 accepted predictions does not guarantee zero future false accepts, particularly for changed operating conditions, contingencies, and topologies.

The model is a 20-output, five-machine swing-equation-informed GENROU surrogate, **not** a full GENROU-DAE model or protection system. This benchmark does not evaluate a different network, weather/load operating points, other fault types, or broad stochastic uncertainty.

## Reproduce and inspect

- Workflow: `.github/workflows/andes-v1-fresh-duration.yml`.
- Executable: `scripts/run_andes_v1_fresh_duration.py`.
- SHA-256-pinned trained models: same as `scripts/run_andes_trained_runtime_audit.py`.
- Case-level decisions and errors: [download the successful GitHub Actions artifact](https://github.com/abhijith-sivaprasadan/GridGuardPINN/actions/runs/37925430499/artifacts/11613913010).

**Release recommendation:** evidence now supports a **research-demonstrator v1.0** scoped explicitly to the tested IEEE-14 domain with fail-closed ANDES fallback. This does not establish production readiness. A fully published GitHub release and long-lived artifact availability are separate release-management tasks.
