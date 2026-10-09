# GridGuardPINN randomized repeated full-route performance replication

**Run:** [37939032862](https://github.com/abhijith-sivaprasadan/GridGuardPINN/actions/runs/37939032862), 9 October 2026, **passed**, no simulator or routing errors. [Paired per-case timing ledger and complete summary](https://github.com/abhijith-sivaprasadan/GridGuardPINN/actions/runs/37939032862/artifacts/11620885900).

**Protocol frozen before run:** [paired randomized protocol](andes_v1_randomized_performance_protocol.md). Original frozen one-hot checkpoints, thresholds and 14 IEEE-14 cases unchanged.

## Design

Three paired rounds per seed (17,29,41), each covering seven familiar fault buses at 0.07 s (within accepted envelope) and 0.13 s (duration OOD). Cases shuffled within rounds with deterministic RNG seed 20261009; order of reference-versus-routed execution randomized for each pair. Actual ANDES was run afresh on **both** sides of each comparison; no reuse of reference wall time. One untimed reference warm-up, plus model forward/physics-residual warm-up per seed. Model loads measured separately.

Total 126 paired observations from 14 **distinct** fault scenarios and three learned seeds. Ninety-five-percent intervals are percentile **bus-cluster** bootstrap (5,000 draws) retaining paired repeated observations. These intervals describe variation over seven buses on this CI host, not independent-grid performance, cross-host confidence, or certified error bounds.

## Full-route results: mixed 50/50 workload

| Seed | End-to-end speedup | 95% bus-cluster bootstrap interval | Surrogate accepted | False accepts |
|---|---:|---:|---:|---:|
| 17 | **1.755×** | 1.430–2.084× | 18/42 | 0 |
| 29 | **1.954×** | 1.867–2.035× | 21/42 | 0 |
| 41 | **1.568×** | 1.241–2.022× | 15/42 | 0 |

Each ratio is total reference-only wall time divided by total **routed** wall time. Full routing includes physics-residual computation and real ANDES execution on fallback. All 63 duration-OOD evaluations across seeds and rounds refused the learned surrogate. No incomplete case or runtime exception was logged.

**Per-round mixed-workload speedup:**

- Seed 17: 1.689×, 1.759×, 1.822×.
- Seed 29: 2.029×, 1.885×, 1.952×.
- Seed 41: 1.540×, 1.472×, 1.703×.

These show some order/run variation, but all three seed-level combined-workload 95% cluster intervals remain above 1.0 for these limited seven buses.

## Secondary split results

| Seed | In-envelope accepts | In-envelope end-to-end speedup | Duration-OOD speedup |
|---|---:|---:|---:|
| 17 | 18/21 | 6.658× | 1.014× |
| 29 | 21/21 | 81.249× | 1.008× |
| 41 | 15/21 | 3.476× | 1.018× |

In-envelope bootstrap intervals for seeds 17 and 41 are exceptionally wide (**2.36–81.58×** and **1.62–79.63×**) because some bootstrap bus samples contain no costly fallback while others contain several. Avoid advertising huge in-envelope ratios as stable benefits. OOD-only intervals all span 1.0, so **do not claim a meaningful OOD-only acceleration**.

Model load time per seed was 4.6–7.0 milliseconds in the running Python process; including one model load per seed does not materially change combined speedup at this benchmark batch size.

## Conclusion

On a fixed, measured 50% eligible / 50% duration-OOD IEEE-14 workload, the **full trust-aware route** was approximately **1.57–1.95× faster** than independently executing ANDES for every case, after warm-up, across three unchanged trained model seeds. This is lower and substantially more conservative than the earlier single-pass **1.99–2.95×** result, and better supported methodologically by paired randomized order and repeated reference measurements.

There were zero observed false surrogate accepts, but the sample comprises only seven eligible fault buses on one IEEE-14 configuration; the same bus scenarios are repeated. No claim of statistically guaranteed safe routing, other fault modes, other network sizes, industrial throughput or outside-topology transfer is justified.

## Evidence and reproducibility

- [Complete workflow, stdout and artifacts](https://github.com/abhijith-sivaprasadan/GridGuardPINN/actions/runs/37939032862)
- `scripts/run_andes_v1_randomized_performance.py`
- `.github/workflows/andes-v1-randomized-performance.yml`
- Frozen pretrained model hashes in `scripts/run_andes_trained_runtime_audit.py`
