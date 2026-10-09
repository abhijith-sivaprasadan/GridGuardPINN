# GridGuardPINN v1.0 end-to-end latency and reliability benchmark

**Executed:** 9 October 2026. [Successful GitHub Actions run 37928033388](https://github.com/abhijith-sivaprasadan/GridGuardPINN/actions/runs/37928033388), [case-level CSV and JSON artifact](https://github.com/abhijith-sivaprasadan/GridGuardPINN/actions/runs/37928033388/artifacts/11615455455). [Predeclared benchmark protocol](andes_v1_performance_protocol.md).

## Method and limits

Measured on a Linux x86-64 Azure GitHub Actions runner, Python 3.11.17. Three unchanged seed-17, 29, 41 **one-hot** trained checkpoints were SHA-256 verified. For each seed, tested seven familiar fault buses at fault duration 0.07 s (inside envelope) and seven at 0.13 s (duration OOD). The simulator was ANDES 2.0.0 IEEE-14. All 14 reference cases solved, and there were no recorded benchmark failures.

For each unique case the harness ran the reference solver to obtain reference-only duration and an accuracy target. For each seed and case, it then measured separate PINN forward latency, eligible-case physics-residual computation, and **complete** `run_model_case` time including another real ANDES solve on fallback. Thus routed wall time includes the trust decision, residual calculation and actual fallback execution.

Model loading times were measured separately (~3.5–5.6 ms) and included in a second amortized combined workload comparison, not disguised as per-case cost.

## Observed wall-clock results

| Seed | In-envelope accepted | In-envelope speedup | OOD accepted | OOD-only speedup | Combined speedup | Combined speedup including model load | False accepts |
|---|---:|---:|---:|---:|---:|---:|---:|
| 17 | 6/7 | 10.96× | 0/7 | 0.990× | 2.525× | 2.522× | 0 |
| 29 | 7/7 | 185.38× | 0/7 | 0.989× | 2.950× | 2.947× | 0 |
| 41 | 5/7 | 6.61× | 0/7 | 0.825× | 1.986× | 1.985× | 0 |

**Combined seven-in-envelope + seven-OOD wall times per seed:** reference-only total 11.745 s; routed totals 4.652 s (17), 3.982 s (29), and 5.914 s (41). The ratio of total reference time to total routed time is the benchmark's reported speedup—not the isolated neural-forward-pass speedup.

The median per-case PINN-only forward time was approximately 0.7–1.5 ms, versus median standalone eligible-case physics-residual computation about 5.1–5.25 ms. Forward-only timing is **not** a trustworthy speedup metric because it excludes routing cost and yields untrusted outputs.

## Cautions that affect interpretation

1. Reference-only times for the seven in-envelope cases totalled **7.847 s**, whereas the seven duration-OOD cases totalled **3.898 s**. These differences reflect the specific sequence, cases, process state, convergence and potential first-run/cold-start effects. The especially large 185× in-envelope speedup for seed 29 should not be generalized as a hardware-independent throughput ratio.
2. ANDES-only reference costs were measured once per distinct fault case and reused as the comparator for three model seeds; routed fallback invokes ANDES afresh. This is a same-run **sequential** comparison, not a repeated randomized cross-over study.
3. Each seed used 14 scenarios, only seven within the surrogate's evaluated domain. The 42 seed-case comparisons reuse 14 underlying physical scenarios, so they do not establish general statistical error bounds.
4. The OOD-only hybrid was **slightly slower than** reference-only for seeds 17 and 29, and substantially slower for seed 41: refusal does not inherently save time.
5. Accuracy failures were not observed among accepted routes here, but this is a previously investigated IEEE-14 domain and is **not** a new model-validation holdout.
6. The benchmark excludes package installation, model training, artifact downloads, and external infrastructure integration. ANDES initialization effects are not standardized across independent processes. Further CPU-pinned randomized repetitions would be required before quoting an industrial performance guarantee.

## Supported conclusion

On this **specific 50% in-envelope / 50% OOD mixed workload**, GridGuardPINN reduced total measured routing latency versus running ANDES for every case, by approximately **1.99× to 2.95×** across the three frozen model seeds, including full physics-residual gating and reference fallbacks. There were zero observed false accepts. The computational benefit depends strongly on workload mix, model seed, fraction of fallbacks and initialization conditions.

This is a credible **research-demonstrator performance result**, not proof of operational protection-system safety or real-world speed gains on other topologies.

## Reproduce

- Executable: `scripts/run_andes_v1_performance.py`
- Workflow: `.github/workflows/andes-v1-performance.yml`
- Frozen checkpoints: [original source artifact](https://github.com/abhijith-sivaprasadan/GridGuardPINN/actions/runs/37922112172)
- [Full decision/error/timing ledger](https://github.com/abhijith-sivaprasadan/GridGuardPINN/actions/runs/37928033388/artifacts/11615455455)
