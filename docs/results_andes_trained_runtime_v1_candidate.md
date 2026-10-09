# Frozen-checkpoint ANDES routing audit — v1 research-release candidate

**Executed:** 9 October 2026. [Successful workflow run 37924679301](https://github.com/abhijith-sivaprasadan/GridGuardPINN/actions/runs/37924679301). [Uploaded per-case ledger and summary](https://github.com/abhijith-sivaprasadan/GridGuardPINN/actions/runs/37924679301/artifacts/11612834465).

**Interpretation:** retrospective reproducibility and routing-integration audit, **not** a new, blind holdout or assurance of operational safety. All 35 test cases were already evaluated during earlier experiments.

## Pinned source and executable path

- Original pretrained artifact: [v0.3 run 37922112172](https://github.com/abhijith-sivaprasadan/GridGuardPINN/actions/runs/37922112172), artifact ID `11612523064`.
- Archived ZIP SHA-256 (verified against original GitHub Actions upload): `02a82e4d53544203376749e7054778b9b93a2a5d6af3c83ab7d2820fd24f6666`.
- One-hot seed-17 checkpoint: `c2e39a1b0618336bb4342037d51e957a866a8312afe887cdb3546de61e0ebfd5`.
- One-hot seed-29 checkpoint: `4a136596171930052b59dda7adeeb76bc8b998a4d6474e8fec2a910399d38ffa`.
- One-hot seed-41 checkpoint: `e3f8fe19a44edebb1616b741f797cd7057300627ff292e4c3ae7afae27fa0367`.
- The CI job downloads the original ZIP, checks its SHA-256, loads each strictly pinned checkpoint with `weights_only=True`, and re-runs an independent IEEE-14 ANDES reference as necessary.

## Actual executed route

For eligible fault buses/durations, the script computes the 151-point swing residual using the actual trained model and the trusted IEEE-14 machine constants from the reference. The residual is compared against that model's frozen validation-calibrated threshold. Rejected cases are executed through the ANDES fault simulator. The execution path records `source`, `reason`, and routed trajectory error, and re-runs ANDES rather than substituting a synthetic zero-error result.

| Seed | ID surrogate accepted | ID reference fallback | Duration OOD accepted | Location OOD accepted | Unsafe accepts |
|---|---:|---:|---:|---:|---:|
| 17 | 4/7 | 3/7 | 0/7 | 0/21 | 0 |
| 29 | 6/7 | 1/7 | 0/7 | 0/21 | 0 |
| 41 | 3/7 | 4/7 | 0/7 | 0/21 | 0 |
| **Total** | **13/21** | **8/21** | **0/21** | **0/63** | **0** |

Routed trajectory composite-error means on the seven ID cases: seed 17 `0.07340`, seed 29 `0.12176`, seed 41 `0.05424`. All duration-OOD and location-OOD cases fell back to ANDES; the recorded routed mean errors were 0 against separate recomputed ANDES references on this deterministic testbed.

Across the 105 individual seed-case evaluations, **13 were returned by the surrogate, 92 were returned by the reference**, and none of the accepted surrogate outputs exceeded the pre-established exploratory accuracy screen. The 105 observations are not statistically independent: the same 35 fault cases are evaluated under three seeds.

## Limits that prevent operational claims

- The evaluated cases are *not* untouched after the v0.3 investigation. The results are integration/reproducibility evidence only.
- Location OOD coverage is intentionally zero: the conservative location-hard gate refuses all unseen buses, even when some network-aware model outputs could have been accurate.
- Only one-hot checkpoints were routed; a network-aware feature encoder is not part of this production-style path.
- Zero observed false accepts on 13 accepted ID results is a very small sample and provides **no safety guarantee**.
- This is a 20-channel electromechanical surrogate, **not a full GENROU-DAE solver**, operational protection system, or grid-code-certified assessment.
- The experiment used the same frozen training/validation protocol and tolerances, not new risk calibration.
- An immutable v1 release/tag and a larger, genuinely fresh independent validation protocol remain outstanding.

## Reproduce

Workflow: `.github/workflows/andes-trained-runtime-audit.yml`. Python entry: `scripts/run_andes_trained_runtime_audit.py`. Run the workflow manually or reproduce its pinned download/verification steps and run the script with `--artifact-dir`.

Full results: `case_ledger.csv`, `summary.json` in the [GitHub Actions artifact](https://github.com/abhijith-sivaprasadan/GridGuardPINN/actions/runs/37924679301/artifacts/11612834465).
