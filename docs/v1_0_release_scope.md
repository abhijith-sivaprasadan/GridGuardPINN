# GridGuardPINN v1.0 — research-demonstrator release scope

**Release claim:** reproducible, research-grade trust-aware surrogate experiments and a conservative routing-policy module. **Not** a production grid simulator, safety certification, or operational protection controller.

## What is available

- Segmented SMIB references and previously frozen v0.2–v0.4 experiments.
- ANDES 2.0.0 IEEE-14 reference path and five-machine, 20-output swing-equation-informed surrogate.
- Three-seed pre-registered physics ablation v0.2 and fault-location descriptor experiment v0.3, with explicit negative results.
- Audit of routing errors, including unsafe residual-only accepts for unseen fault locations.
- `route_andes_case`: deterministic fail-closed eligibility check that rejects unseen buses, unsupported fault durations, invalid inputs, invalid residual scores, and unready models.
- `andes_runtime`: digest-pinned checkpoint loading and an executable surrogate/reference routing adapter, with reference errors propagated rather than hidden. See [runtime limitations](v1_0_runtime.md).

## Scope boundaries

The `andes_router` function provides a policy decision, while `andes_runtime.run_routed_case` now executes either a prediction callback or a reference callback. The calling application must still provide a trusted physics-residual score, a validated model, and an operational reference solver; reference failures propagate as errors. The function does **not** claim risk bounds for apparently in-distribution cases. Validation calibration on a handful of cases cannot establish zero failure probability.

The v0.3 unseen-location improvement cannot be marketed as safe unseen-location coverage because residual-only gating produced false accepts. The conservative default therefore **always rejects unseen buses**. Duration shift is likewise rejected.

## Release acceptance checks

1. Core CI lint and tests, including the fail-closed routing tests, pass on the release commit.
2. Evidence and negative results remain publicly linked.
3. The package version is 1.0.0, clearly marked as a research demonstrator rather than a production safety solution.
4. Experimental claims remain tied to the original committed protocols and immutable workflow runs.

## Research-grade limitations and follow-up

- No held-out guarantee of zero false accepts; strict route has not been certified.
- Current evaluated accuracy tolerances are project-internal screening limits.
- A basic executable reference-fallback adapter and digest-pinned checkpoint loader now exist, but live physics-residual computation, a dedicated CLI, and an end-to-end actual-ANDES benchmark remain outstanding.
- ANDES convergence failures in reference sweeps must be surfaced, never silently classified as correct trajectories.
- Next scientific milestone: new pre-registered trust-policy evaluation with broader fault topologies, temporal ranges, more seeds, and uncertainty estimates. Do not tune to the inspected v0.3 holdout.
