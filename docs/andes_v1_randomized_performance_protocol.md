# Pre-registered v1.0 randomized repeated performance protocol

Frozen before execution. This is a **performance replication**, not an independent accuracy holdout. No checkpoint, gate, threshold, case, or model selection changes are allowed.

## Design

Compare exact same seven IEEE-14 GENROU fault buses (3,6,7,8,9,11,14), at 0.07 s within the supported duration envelope and 0.13 s outside it, for all three pinned one-hot seeds (17,29,41).

Use 3 paired measurement rounds, with deterministic RNG seed 20261009. Within each round and model seed, shuffle the 14 cases using the RNG and **randomize whether the independent reference solve or the full routed solve is timed first for each case**. Run one untimed reference warm-up and one untimed model physics-residual/forward warm-up per seed before recording times. Measure both branches freshly for each pair, not via a reference-time cache. Use actual ANDES on routed fallback. Record independent ANDES reference and routed trajectories to compute accepted-case error; propagate and record all errors. No replay of premeasured reference time as a benchmark observation.

Wall-clock uses perf_counter on one GitHub Actions CPU runner; library versions and machine metadata recorded. Report per-seed and split total reference/routed time, speedup by ratio of totals, accepted fraction, false accepts and exceptions. **Primary endpoint** mixed 50% in-envelope / 50% OOD workload, with full physics-residual gate included. In-envelope-only and OOD-only are secondary.

For uncertainty, use a *paired cluster bootstrap over seven bus IDs* with 5,000 deterministic draws, retaining all paired rounds and both durations of any sampled bus. Estimate 95% percentile intervals for ratio of summed reference / summed routed durations. Bus sampling is the only resampling unit; rounds and model seeds are dependent repeats and **must not be treated as independent scenarios**. Also report per-round speedups to expose time/order drift.

Warm-up and checkpoint load time are reported outside steady-state primary endpoint; provide amortized ratio including one model load for 42 routed cases per seed. Both systems execute as in-process Python; installation, training, and artifact download costs excluded. The performance test uses previously inspected IEEE-14 cases and cannot establish generalization or certification.

No performance threshold is set in advance. A result below 1.0× (or interval spanning 1.0) is an honest negative/inconclusive result, not reason to alter protocol.
