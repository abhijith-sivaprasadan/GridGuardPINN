"""Run honest wall-clock ANDES/PINN/routed comparison using pinned checkpoints."""
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

DURATIONS = {"in_envelope": 0.07, "duration_ood": 0.13}
GRID = np.linspace(0.0, 2.0, 401)


def record_stats(rows):
    ref = sum(r["reference_seconds"] for r in rows)
    routed = sum(r["routed_seconds"] for r in rows)
    return {
        "cases": len(rows),
        "accepted": sum(r["source"] == "surrogate" for r in rows),
        "false_accepts": sum(r["false_accept"] for r in rows),
        "reference_total_seconds": ref,
        "routed_total_seconds": routed,
        "speedup_ratio": ref / routed if routed else None,
        "median_forward_seconds": float(np.median(
            [r["forward_seconds"] for r in rows])),
        "median_residual_seconds": float(np.median(
            [r["residual_seconds"] for r in rows if r["residual_seconds"] is not None]
        )) if any(r["residual_seconds"] is not None for r in rows) else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--output-dir", default="artifacts/andes_v1_performance")
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ground_truth = {}
    issues = []
    constants = None

    # Reference-only baseline: actual simulator executed for every unique case.
    for name, duration in DURATIONS.items():
        for bus in ROBUST_BUSES:
            started = time.perf_counter()
            try:
                native = reference_fn(bus, duration)
                elapsed = time.perf_counter() - started
                sampled = resample_trajectory(native, time_grid_s=GRID)
                vectors = (sampled.machine_inertia_M, sampled.machine_damping_D,
                           sampled.machine_frequency_hz)
                if any(v is None or np.shape(v) != (5,) for v in vectors):
                    raise RuntimeError("Missing machine constants")
                if constants is None:
                    constants = dict(zip(
                        ("inertia_M", "damping_D", "frequency_hz"),
                        vectors, strict=True,
                    ))
                elif not all(np.allclose(a, b) for a, b in
                             zip(constants.values(), vectors, strict=True)):
                    raise RuntimeError("Machine constants differ across cases")
                ground_truth[(name, bus)] = (channels(sampled), elapsed)
            except Exception as exc:  # noqa: BLE001 - record failed reference
                issues.append({"phase": "reference", "bus": bus, "duration": duration,
                               "error": repr(exc)})

    if constants is None:
        raise RuntimeError("No successful ANDES reference cases")
    rows, loading = [], {}
    for seed, digest in TRAINED_SHA256.items():
        started = time.perf_counter()
        model, threshold, summary = load_checkpoint(
            Path(args.artifact_dir) / "one_hot" / f"seed_{seed}" / "model.pt",
            expected_sha256=digest,
            expected_protocol="andes-location-encoding-v0.3",
        )
        loading[str(seed)] = time.perf_counter() - started
        predict = model_prediction(model)
        for name, duration in DURATIONS.items():
            for bus in ROBUST_BUSES:
                if (name, bus) not in ground_truth:
                    continue
                truth, reference_seconds = ground_truth[(name, bus)]
                try:
                    # PINN forward-only: measured separately, never used as a trusted answer.
                    started = time.perf_counter()
                    raw = predict(GRID, bus, duration)
                    forward_seconds = time.perf_counter() - started
                    raw_diff = raw - truth
                    raw_error = max(
                        float(np.sqrt(np.mean(raw_diff[:, :5] ** 2, axis=0)).max()) / 0.05,
                        float(np.sqrt(np.mean(raw_diff[:, 5:10] ** 2, axis=0)).max()) / 5e-4,
                    )
                    # Standalone residual time, measured for decomposition only;
                    # full routed time includes a separately executed residual.
                    residual_seconds = None
                    if name == "in_envelope":
                        started = time.perf_counter()
                        compute_model_residual(
                            model, fault_bus=bus, fault_duration_s=duration, **constants
                        )
                        residual_seconds = time.perf_counter() - started
                    started = time.perf_counter()
                    routed = run_model_case(
                        model=model, fault_bus=bus, fault_duration_s=duration,
                        residual_threshold=threshold,
                        reference=reference_fn, **constants,
                    )
                    routed_seconds = time.perf_counter() - started
                    delta = routed.electromechanical - truth
                    routed_error = max(
                        float(np.sqrt(np.mean(delta[:, :5] ** 2, axis=0)).max()) / 0.05,
                        float(np.sqrt(np.mean(delta[:, 5:10] ** 2, axis=0)).max()) / 5e-4,
                    )
                    rows.append({
                        "seed": seed, "split": name, "bus": bus,
                        "duration_s": duration, "source": routed.source,
                        "reason": routed.decision.reason,
                        "reference_seconds": reference_seconds,
                        "forward_seconds": forward_seconds,
                        "residual_seconds": residual_seconds,
                        "routed_seconds": routed_seconds,
                        "raw_pinn_composite_error": raw_error,
                        "routed_composite_error": routed_error,
                        "false_accept": int(routed.source == "surrogate" and routed_error > 1),
                        "checkpoint_protocol": summary["protocol"],
                    })
                except Exception as exc:  # noqa: BLE001 - keep complete failure ledger
                    issues.append({"phase": "routed", "seed": seed,
                                   "split": name, "bus": bus, "error": repr(exc)})
    if rows:
        with (out / "per_case.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    groups = {}
    for seed in TRAINED_SHA256:
        groups[str(seed)] = {}
        for name in DURATIONS:
            subset = [r for r in rows if r["seed"] == seed and r["split"] == name]
            if subset:
                groups[str(seed)][name] = record_stats(subset)
        subset = [r for r in rows if r["seed"] == seed]
        if subset:
            groups[str(seed)]["combined"] = record_stats(subset)
            group = groups[str(seed)]["combined"]
            group["with_model_load_speedup_ratio"] = (
                group["reference_total_seconds"] /
                (group["routed_total_seconds"] + loading[str(seed)])
            )
    summary = {
        "protocol": "docs/andes_v1_performance_protocol.md",
        "hardware": {"platform": platform.platform(), "processor": platform.processor(),
                     "python": platform.python_version()},
        "checkpoint_sha256": TRAINED_SHA256,
        "model_load_seconds": loading,
        "groups": groups,
        "issues": issues,
        "warning": "CI timing is environment-specific; premeasured reference time reused per seed.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("ANDES_PERFORMANCE_SUMMARY_JSON=" + json.dumps(summary), flush=True)
    if issues or len(rows) != len(TRAINED_SHA256) * len(ROBUST_BUSES) * len(DURATIONS):
        raise SystemExit("Performance audit incomplete; see preserved summary")


if __name__ == "__main__":
    main()
