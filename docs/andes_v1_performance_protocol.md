# v1.0 reproducible end-to-end performance protocol

Frozen **before** executing benchmark results. No thresholds, checkpoints or selected scenarios may change after inspection.

Compare (A) actual ANDES-only wall time, (B) PINN-only forward pass (unsafe baseline), and (C) real routed GridGuardPINN latency, **including physics residual and fallback execution**. Report ratios of summed wall times, not ratios of isolated forward-pass medians.

Cases: seven robust IEEE-14 buses (3,6,7,8,9,11,14) at 0.07 seconds (new intermediate duration), plus seven at 0.13 seconds (duration OOD). Repeat across pinned one-hot seeds 17/29/41. Reference solves run once per unique case for ground truth and timing. Routed fallbacks **execute actual ANDES again**; never substitute zero elapsed time. No training, calibration or parameter search is permitted.

Record model loading time per seed, isolated model forward latency, physics-residual latency, reference-only solve latency, routed latency, routing source, acceptance error versus reference, exceptions and machine/processor description from the runner. Include cold-versus-warm caveats, shared initialisation/caching, solver stochasticity, CI host variation, and exclusion of package installation costs. Each case is run sequentially on the same runner. Report network-wide savings for different workloads separately: in-envelope only, OOD only, and combined. For each split/seed: **total reference seconds divided by total routed seconds**, plus arithmetic coverage; no throughput claims without workload definition.

Accuracy screens and checkpoints remain frozen as in the existing v1 protocol; a false accept is a routed surrogate with composite ratio >1. A benchmark failure, slow hybrid or no significant net acceleration is valid negative evidence. This test reuses already inspected topology/duration combinations; **performance benchmarking is not independent model accuracy validation**.

Model loading cost is one-time per seed, not hidden inside per-case latency. Report amortized totals separately from steady-state totals. The PINN-only route is not a safe replacement for ANDES, and comparisons must not imply that it is.
