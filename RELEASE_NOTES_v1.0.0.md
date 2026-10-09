# GridGuardPINN v1.0.0 — Research Demonstrator

**Status:** release documentation; publication status must be checked against the GitHub Releases page.

## Scope

A reproducible, trust-aware physics-informed neural surrogate research demonstrator for ANDES 2.0.0 IEEE-14 three-phase-fault transient simulation. The surrogate predicts 20 five-machine GENROU electromechanical channels. Unsupported disturbances and failed trust checks route to independent ANDES simulation.

**Not intended for:** real-time grid control, relay protection, certified stability decisions, deployment on other networks, or claims of statistically guaranteed zero routing errors.

## Research results

The pre-registered new-duration evaluation of three frozen, SHA-256-pinned trained models passed its frozen research release gates:

- Interpolation fault durations 0.05, 0.07, 0.09 and 0.11 seconds at seven familiar fault buses: 76 of 84 seed-case evaluations accepted the surrogate (90.5%).
- Zero observed false accepts among the 76 accepted interpolation outputs.
- All 21 seed-case evaluations with 0.13 s fault duration routed to ANDES reference.
- All 35 distinct reference scenarios solved; no benchmark runtime failures.
- Held-out duration values were predeclared; the same IEEE-14 network and fault mechanism were retained, so this is not cross-grid external validation.

**Frozen results:** [report](docs/results_andes_v1_fresh_duration.md), [protocol](docs/andes_v1_fresh_duration_protocol.md), [successful experimental run and downloadable case ledger](https://github.com/abhijith-sivaprasadan/GridGuardPINN/actions/runs/37925430499).

The earlier, previously inspected trained-checkpoint integration audit returned 13 surrogate predictions and 92 reference fallbacks across 105 seed-case evaluations with zero observed false accepts. It is integration/reproducibility evidence rather than new blind evaluation: [report](docs/results_andes_trained_runtime_v1_candidate.md).

## Reproduction and model provenance

- Python >=3.11 for ANDES integration, with `pip install torch` and `pip install -e ".[dev,andes]"`.
- `pytest` and `ruff check src tests scripts` for core checks.
- `python scripts/run_andes_runtime_smoke.py` for actual ANDES reference and routing integration; CI workflow `.github/workflows/andes-runtime-smoke.yml`.
- `scripts/run_andes_trained_runtime_audit.py` and `scripts/run_andes_v1_fresh_duration.py` for reproducible trained-model audits.
- Original v0.3 model artifact from [run 37922112172](https://github.com/abhijith-sivaprasadan/GridGuardPINN/actions/runs/37922112172), artifact ID 11612523064, ZIP SHA-256 `02a82e4d53544203376749e7054778b9b93a2a5d6af3c83ab7d2820fd24f6666`.
- Individually pinned model hashes, benchmark decisions, metrics and limitations are in the respective scripts and reports.
- Checkpoint loading fails on digest mismatch or incompatible input encoding.

## Release restrictions

- The research accuracy screen (max machine angle RMSE <=0.05 rad, speed RMSE <=0.0005 pu) is an experimental criterion, not a regulatory or grid-code tolerance.
- The trust gate **always rejects unseen fault buses and duration-OOD** for the runtime's restricted envelope, even if a model might happen to predict those cases well.
- Physics residuals and limited validation do not prove safety.
- Model inference omits bus voltage; fallback reference output contains additional observables.
- GitHub Actions artifacts have retention limits. Archive artifacts for long-lived reproducibility before relying on them as permanent release attachments.

## Publication verification

The publication workflow checks successful core and integrated CI on the **exact release commit** before creating the tag. It uploads the SHA-256-pinned training archive, validation and performance ledgers, vector figures, source-commit identifier, environment snapshot and checksum manifest. It then verifies the GitHub Release is published with the expected tag and minimum number of assets. If the workflow fails, **do not represent a release as published** until the GitHub Releases page and files are independently verified.

## QA follow-up and archival assets

The QA remediation includes strict router numeric validation, explicit one-hot checkpoint metadata, precise physics-residual fallback reasons, canonical ANDES v2.0.0 case hash and generator-order verification in the CLI, full-horizon time grids, and standard CI incorporating real ANDES plus PyTorch. CLI routing JSON additionally records the residual score, residual threshold and reference fallback solve time. These changes do not alter the frozen training data, model parameters or calibration thresholds.

See [the internal QA report](docs/qa_audit_v1_0_2026_10_09.md), [verified external literature context](docs/literature_context_v1_0.md), and [reproducible vector figures](docs/figures/).

A GitHub Release should include the four original evidence archives, two SVG figures, a resolved Python environment snapshot, the exact source commit, and a SHA-256 manifest. The environment snapshot is documentation of one successful build, **not** a universal cross-platform dependency lock. Historical model artifacts are authenticated by hashes but do not embed a contemporaneous case-file fingerprint; this limitation cannot be retroactively erased.

The package and report remain **research-only**. No other-grid validation or formal peer review has occurred.
