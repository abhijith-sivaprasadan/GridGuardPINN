"""Pre-registered fresh-duration release-gate evaluation.

Do not edit this protocol or its seeds/checkpoints after seeing held-out results.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from run_andes_trained_runtime_audit import TRAINED_SHA256

from gridguardpinn.andes_reference import resample_trajectory, run_ieee14_fault
from gridguardpinn.andes_runtime import load_checkpoint, run_model_case
from gridguardpinn.andes_surrogate_protocol import ROBUST_BUSES

INTERPOLATION_DURATIONS = (0.05, 0.07, 0.09, 0.11)
STRESS_DURATIONS = (0.13,)
GRID = np.linspace(0.0, 2.0, 401)
SPLITS = {
    "fresh_interpolation": [(bus, duration) for bus in ROBUST_BUSES
                            for duration in INTERPOLATION_DURATIONS],
    "new_duration_ood": [(bus, duration) for bus in ROBUST_BUSES
                         for duration in STRESS_DURATIONS],
}


def reference_fn(bus, duration):
    return run_ieee14_fault(
        fault_bus=bus, fault_start_s=1.0, fault_clear_s=1.0 + duration,
        simulation_end_s=2.0, fault_reactance_pu=1e-4,
    )


def channels(trajectory):
    fields = (trajectory.generator_angle_rad, trajectory.generator_speed_pu,
              trajectory.generator_mechanical_torque_pu,
              trajectory.generator_electrical_torque_pu)
    if any(field is None for field in fields):
        raise ValueError("Reference is missing electromechanical channels.")
    return np.column_stack(fields)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--output-dir", default="artifacts/andes_v1_fresh_duration")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Only a fixed, independently executed reference may supply targets.
    references = {}
    issues = []
    constants = None
    for split, cases in SPLITS.items():
        for bus, duration in cases:
            started = time.perf_counter()
            try:
                native = reference_fn(bus, duration)
                trajectory = resample_trajectory(native, time_grid_s=GRID)
                truth = channels(trajectory)
                vectors = (trajectory.machine_inertia_M, trajectory.machine_damping_D,
                           trajectory.machine_frequency_hz)
                if any(v is None or len(v) != 5 or not np.all(np.isfinite(v))
                       for v in vectors):
                    raise RuntimeError("Missing trusted GENROU machine parameters.")
                if constants is None:
                    constants = {
                        "inertia_M": vectors[0], "damping_D": vectors[1],
                        "frequency_hz": vectors[2],
                    }
                else:
                    if not all(np.allclose(a, b) for a, b in zip(
                            constants.values(), vectors, strict=True)):
                        raise RuntimeError("Reference case machine constants changed.")
                references[(bus, duration)] = (truth, time.perf_counter() - started)
            except Exception as exc:  # noqa: BLE001 - record all ANDES failures
                issues.append({"split": split, "bus": bus, "duration_s": duration,
                               "error_type": type(exc).__name__, "message": str(exc)})
    if constants is None:
        raise RuntimeError("No reference solved; cannot assess this protocol.")

    rows = []
    for seed, sha in TRAINED_SHA256.items():
        model, threshold, summary = load_checkpoint(
            Path(args.artifact_dir) / "one_hot" / f"seed_{seed}" / "model.pt",
            expected_sha256=sha, expected_protocol="andes-location-encoding-v0.3",
        )
        if summary.get("experiment_metadata", {}).get("location_encoding") != "one_hot":
            raise ValueError("Checkpoint encoder mismatch.")
        for split, cases in SPLITS.items():
            for bus, duration in cases:
                if (bus, duration) not in references:
                    continue
                truth, reference_seconds = references[(bus, duration)]
                try:
                    started = time.perf_counter()
                    result = run_model_case(
                        model=model, fault_bus=bus, fault_duration_s=duration,
                        residual_threshold=threshold, reference=reference_fn, **constants,
                    )
                    routed_seconds = time.perf_counter() - started
                    error = result.electromechanical - truth
                    angle = float(np.sqrt(np.mean(error[:, :5] ** 2, axis=0)).max())
                    speed = float(np.sqrt(np.mean(error[:, 5:10] ** 2, axis=0)).max())
                    composite = max(angle / 0.05, speed / 0.0005)
                    if not np.isfinite(composite):
                        raise RuntimeError("Non-finite routed error.")
                    rows.append({
                        "seed": seed, "split": split, "bus": bus, "duration_s": duration,
                        "source": result.source, "reason": result.decision.reason,
                        "composite_error_ratio": composite, "angle_rmse_rad": angle,
                        "speed_rmse_pu": speed,
                        "false_accept": int(result.source == "surrogate" and composite > 1),
                        "false_escalation": int(result.source == "reference" and split == "fresh_interpolation"),
                        "reference_seconds": reference_seconds, "routed_seconds": routed_seconds,
                    })
                except Exception as exc:  # noqa: BLE001 - record runtime failures
                    issues.append({"split": split, "seed": seed, "bus": bus,
                                   "duration_s": duration,
                                   "error_type": type(exc).__name__, "message": str(exc)})
    reports = {}
    checks = {}
    for seed in TRAINED_SHA256:
        reports[str(seed)] = {}
        for split, cases in SPLITS.items():
            sample = [r for r in rows if r["seed"] == seed and r["split"] == split]
            accepted = [r for r in sample if r["source"] == "surrogate"]
            reports[str(seed)][split] = {
                "planned": len(cases), "evaluated": len(sample),
                "coverage": len(accepted) / len(sample) if sample else None,
                "surrogate_accepted": len(accepted),
                "reference_fallback": len(sample) - len(accepted),
                "false_accepts": sum(r["false_accept"] for r in sample),
                "mean_accepted_error": float(np.mean(
                    [r["composite_error_ratio"] for r in accepted])) if accepted else None,
                "max_accepted_error": max(
                    (r["composite_error_ratio"] for r in accepted), default=None),
            }
        inter = reports[str(seed)]["fresh_interpolation"]
        ood = reports[str(seed)]["new_duration_ood"]
        checks[str(seed)] = {
            "no_false_accepts": inter["evaluated"] > 0 and inter["false_accepts"] == 0,
            "interpolation_coverage_at_least_50pct": inter["coverage"] is not None
            and inter["coverage"] >= 0.5,
            "duration_ood_refused": ood["evaluated"] > 0
            and ood["surrogate_accepted"] == 0,
        }
    all_good = (
        not issues
        and all(all(result.values()) for result in checks.values())
        and all(reports[str(seed)][split]["evaluated"] == len(cases)
                for seed in TRAINED_SHA256 for split, cases in SPLITS.items())
    )
    with (output / "case_ledger.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else
                                ["seed", "split", "bus", "duration_s"])
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "protocol": "docs/andes_v1_fresh_duration_protocol.md",
        "note": "Fresh duration values; not independent topology, equipment, or simulator.",
        "checkpoint_sha256": TRAINED_SHA256,
        "splits": {name: len(cases) for name, cases in SPLITS.items()},
        "reports": reports, "acceptance_checks": checks,
        "issues": issues, "release_gate_pass": all_good,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("V1_FRESH_DURATION_SUMMARY_JSON=" + json.dumps(summary))
    if not all_good:
        raise SystemExit("Pre-registered v1 release gate did not pass; inspect artifact.")


if __name__ == "__main__":
    main()
