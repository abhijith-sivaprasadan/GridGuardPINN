"""Run the pre-registered ANDES multi-machine physics ablation v0.2."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from gridguardpinn.andes_experiment import (
    run_surrogate_experiment,
    write_experiment_artifacts,
)
from gridguardpinn.andes_multimachine import generate_reference_batch
from gridguardpinn.andes_surrogate_protocol import (
    assert_protocol_integrity,
    surrogate_splits_v01,
)
from gridguardpinn.andes_training import AndesTrainingConfig

ARMS = {
    "data_only": 0.0,
    "physics_informed": 0.02,
}


def _finite(values):
    return np.asarray(
        [float(value) for value in values if value is not None and np.isfinite(value)],
        dtype=float,
    )


def _aggregate(values) -> dict[str, float | None]:
    array = _finite(values)
    if array.size == 0:
        return {"median": None, "min": None, "max": None}
    return {
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _flat_row(arm, seed, split_name, summary):
    accuracy = summary["accuracy"][split_name]
    gate = summary["reports"][split_name]["combined_strict"]
    return {
        "arm": arm,
        "seed": seed,
        "split": split_name,
        "good_fraction": accuracy["good_fraction"],
        "composite_median": accuracy["composite_median"],
        "composite_mean": accuracy["composite_mean"],
        "residual_error_spearman": accuracy["residual_error_spearman"],
        "combined_coverage": gate["coverage"],
        "combined_false_accepts": gate["false_accepts"],
        "combined_false_escalations": gate["false_escalations"],
        "median_inference_seconds": accuracy["surrogate_inference"]["median_seconds"],
        "median_reference_seconds": accuracy["reference_simulation"]["median_seconds"],
        "median_speedup_vs_reference": accuracy["median_speedup_vs_reference"],
        "best_epoch": summary["training"]["best_epoch"],
        "training_seconds": summary["training"]["training_seconds"],
    }


def _aggregate_summaries(flat_rows):
    output = {}
    for arm in ARMS:
        arm_rows = [row for row in flat_rows if row["arm"] == arm]
        output[arm] = {}
        for split_name in ("validation", "test_id", "ood_duration", "ood_location"):
            rows = [row for row in arm_rows if row["split"] == split_name]
            output[arm][split_name] = {
                key: _aggregate([row[key] for row in rows])
                for key in (
                    "good_fraction",
                    "composite_median",
                    "composite_mean",
                    "residual_error_spearman",
                    "combined_coverage",
                    "combined_false_accepts",
                    "combined_false_escalations",
                    "median_inference_seconds",
                    "median_reference_seconds",
                    "median_speedup_vs_reference",
                )
            }
        output[arm]["training_seconds"] = _aggregate(
            [row["training_seconds"] for row in arm_rows if row["split"] == "validation"]
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--anchors", type=int, default=101)
    parser.add_argument("--collocation", type=int, default=56)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 41])
    parser.add_argument(
        "--output-dir",
        default="artifacts/andes_surrogate_ablation_v0_2",
    )
    args = parser.parse_args()

    if not args.seeds:
        parser.error("--seeds requires at least one integer seed.")

    assert_protocol_integrity()
    splits = surrogate_splits_v01()
    all_cases = list(
        dict.fromkeys(
            case
            for cases in splits.values()
            for case in cases
        )
    )
    reference_batch = generate_reference_batch(all_cases, samples=401)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    flat_rows = []
    summaries = {}

    for arm, physics_weight in ARMS.items():
        summaries[arm] = {}
        for seed in args.seeds:
            config = AndesTrainingConfig(
                epochs=args.epochs,
                anchors_per_case=args.anchors,
                collocation_per_case=args.collocation,
                physics_weight=physics_weight,
                seed=seed,
            )
            result = run_surrogate_experiment(
                reference_batch,
                splits,
                config=config,
                protocol="andes-surrogate-ablation-v0.2",
                run_label=f"{arm}-seed-{seed}",
            )
            arm_dir = output / arm / f"seed_{seed}"
            write_experiment_artifacts(
                result,
                splits,
                output_dir=arm_dir,
            )
            summaries[arm][str(seed)] = result.summary

            for split_name in (
                "validation",
                "test_id",
                "ood_duration",
                "ood_location",
            ):
                flat_rows.append(
                    _flat_row(
                        arm,
                        seed,
                        split_name,
                        result.summary,
                    )
                )

    with (output / "seed_comparison.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)

    aggregate = {
        "protocol": "andes-surrogate-ablation-v0.2",
        "seeds": args.seeds,
        "arms": ARMS,
        "reference_generation": {
            "andes_solve_total_seconds": reference_batch.total_seconds,
            "pipeline_total_seconds": reference_batch.pipeline_seconds,
        },
        "aggregate": _aggregate_summaries(flat_rows),
        "per_seed": summaries,
    }
    (output / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2),
        encoding="utf-8",
    )
    print("ANDES_ABLATION_SUMMARY_JSON=" + json.dumps(aggregate))


if __name__ == "__main__":
    main()
