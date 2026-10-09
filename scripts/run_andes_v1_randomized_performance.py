"""Paired randomized real-ANDES versus full-router wall-time replication."""
from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import numpy as np
from run_andes_trained_runtime_audit import TRAINED_SHA256
from run_andes_v1_fresh_duration import channels, reference_fn

from gridguardpinn.andes_reference import resample_trajectory
from gridguardpinn.andes_runtime import (
    compute_model_residual,
    load_checkpoint,
    model_prediction,
    run_model_case,
)
from gridguardpinn.andes_surrogate_protocol import ROBUST_BUSES

DURATIONS = (("in_envelope", 0.07), ("duration_ood", 0.13))
ROUNDS = 3
BOOTSTRAP_DRAWS = 5000
GRID = np.linspace(0.0, 2.0, 401)
RNG_SEED = 20261009


def composite(output, truth):
    error = output - truth
    return max(
        float(np.sqrt(np.mean(error[:, :5] ** 2, axis=0)).max()) / 0.05,
        float(np.sqrt(np.mean(error[:, 5:10] ** 2, axis=0)).max()) / 5e-4,
    )


def summarize(rows, *, load_seconds=0.0):
    if not rows:
        return None
    a = float(sum(row["reference_seconds"] for row in rows))
    b = float(sum(row["routed_seconds"] for row in rows))
    return {
        "n": len(rows),
        "accepted": sum(row["source"] == "surrogate" for row in rows),
        "false_accepts": sum(row["false_accept"] for row in rows),
        "coverage": sum(row["source"] == "surrogate" for row in rows) / len(rows),
        "reference_seconds": a,
        "routed_seconds": b,
        "speedup": a / b,
        "amortized_speedup": a / (b + load_seconds),
    }


def bootstrap_bus_clusters(rows, *, rng, draws=BOOTSTRAP_DRAWS):
    bus_groups = {bus: [r for r in rows if r["bus"] == bus] for bus in ROBUST_BUSES}
    totals = {
        bus: (sum(row["reference_seconds"] for row in group),
              sum(row["routed_seconds"] for row in group))
        for bus, group in bus_groups.items()
    }
    samples = []
    for _ in range(draws):
        selected = rng.choice(ROBUST_BUSES, size=len(ROBUST_BUSES), replace=True)
        ref = sum(totals[int(bus)][0] for bus in selected)
        routed = sum(totals[int(bus)][1] for bus in selected)
        samples.append(ref / routed)
    lo, hi = np.quantile(samples, (0.025, 0.975))
    return [float(lo), float(hi)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--output-dir", default="artifacts/andes_v1_randomized_performance")
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)
    issues = []
    rows = []
    loading = {}
    warmups = {}
    constants = None
    try:
        initial = reference_fn(3, 0.07)
        constants = {
            "inertia_M": initial.machine_inertia_M,
            "damping_D": initial.machine_damping_D,
            "frequency_hz": initial.machine_frequency_hz,
        }
        if any(np.shape(vector) != (5,) or
               not np.all(np.isfinite(vector)) for vector in constants.values()):
            raise RuntimeError("Invalid reference machine constants")
    except Exception as exc:  # noqa: BLE001 - preserve benchmark errors
        issues.append({"phase": "warmup", "error": repr(exc)})
    if constants is None:
        raise RuntimeError("Reference warm-up failed; cannot run performance test")

    cases = [(split, bus, duration)
             for split, duration in DURATIONS for bus in ROBUST_BUSES]
    for seed, digest in TRAINED_SHA256.items():
        started = time.perf_counter()
        model, threshold, _ = load_checkpoint(
            Path(args.artifact_dir) / "one_hot" / f"seed_{seed}" / "model.pt",
            expected_sha256=digest,
            expected_protocol="andes-location-encoding-v0.3",
        )
        loading[str(seed)] = time.perf_counter() - started
        warm = time.perf_counter()
        model_prediction(model)(GRID, 3, 0.07)
        compute_model_residual(
            model, fault_bus=3, fault_duration_s=0.07, **constants
        )
        warmups[str(seed)] = time.perf_counter() - warm
        for round_id in range(ROUNDS):
            case_order = list(rng.permutation(len(cases)))
            for index in case_order:
                split, bus, duration = cases[int(index)]
                first = "reference" if int(rng.integers(0, 2)) == 0 else "routed"
                try:
                    def execute_reference(bus=bus, duration=duration, constants=constants):
                        began = time.perf_counter()
                        native = reference_fn(bus, duration)
                        seconds = time.perf_counter() - began
                        sampled = resample_trajectory(native, time_grid_s=GRID)
                        if not all(np.allclose(a, b) for a, b in
                                   zip(constants.values(), (
                                       sampled.machine_inertia_M,
                                       sampled.machine_damping_D,
                                       sampled.machine_frequency_hz,
                                   ), strict=True)):
                            raise RuntimeError("Reference constants changed")
                        return channels(sampled), seconds

                    def execute_route(bus=bus, duration=duration, model=model, threshold=threshold, constants=constants):
                        began = time.perf_counter()
                        answer = run_model_case(
                            model=model, fault_bus=bus, fault_duration_s=duration,
                            residual_threshold=threshold, reference=reference_fn,
                            **constants,
                        )
                        return answer, time.perf_counter() - began

                    if first == "reference":
                        truth, ref_s = execute_reference()
                        routed, routed_s = execute_route()
                    else:
                        routed, routed_s = execute_route()
                        truth, ref_s = execute_reference()
                    ratio = composite(routed.electromechanical, truth)
                    if not np.isfinite(ratio):
                        raise ValueError("Non-finite trajectory error")
                    rows.append({
                        "seed": seed, "round": round_id + 1, "split": split,
                        "bus": bus, "duration_s": duration, "first": first,
                        "reference_seconds": ref_s, "routed_seconds": routed_s,
                        "source": routed.source, "reason": routed.decision.reason,
                        "composite_error": ratio,
                        "false_accept": int(routed.source == "surrogate" and ratio > 1),
                    })
                except Exception as exc:  # noqa: BLE001 - record all failures
                    issues.append({
                        "seed": seed, "round": round_id + 1,
                        "split": split, "bus": bus, "duration_s": duration,
                        "phase": "measurement", "error": repr(exc),
                    })

    with (out / "paired_timings.csv").open("w", newline="", encoding="utf-8") as fh:
        fieldnames = [
            "seed", "round", "split", "bus", "duration_s", "first",
            "reference_seconds", "routed_seconds", "source", "reason",
            "composite_error", "false_accept",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    result = {}
    bootstrap_rng = np.random.default_rng(RNG_SEED + 1)
    for seed in TRAINED_SHA256:
        cohort = [r for r in rows if r["seed"] == seed]
        result[str(seed)] = {}
        for split in ("in_envelope", "duration_ood", "combined"):
            subset = cohort if split == "combined" else [
                r for r in cohort if r["split"] == split
            ]
            stats = summarize(subset, load_seconds=loading[str(seed)])
            if stats is not None and len(subset) == ROUNDS * (
                len(ROBUST_BUSES) * (2 if split == "combined" else 1)
            ):
                stats["cluster_bootstrap_95pct_ci"] = bootstrap_bus_clusters(
                    subset, rng=bootstrap_rng
                )
            result[str(seed)][split] = stats
        result[str(seed)]["per_round"] = {
            str(round_id): summarize([
                r for r in cohort if r["round"] == round_id
            ]) for round_id in range(1, ROUNDS + 1)
        }
    summary = {
        "protocol": "docs/andes_v1_randomized_performance_protocol.md",
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "rng_seed": RNG_SEED,
        "rounds": ROUNDS,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "checkpoint_sha256": TRAINED_SHA256,
        "model_load_seconds": loading,
        "warmup_seconds": warmups,
        "groups": result,
        "issues": issues,
        "complete": not issues and len(rows) == len(TRAINED_SHA256) * ROUNDS * len(cases),
        "warning": "CI-runner-specific paired timing, not independent grid validation",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("RANDOMIZED_PERFORMANCE_SUMMARY_JSON=" + json.dumps(summary), flush=True)
    if not summary["complete"]:
        raise SystemExit("Incomplete benchmark; inspect recorded issues")


if __name__ == "__main__":
    main()
