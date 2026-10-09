"""Run the pre-registered ANDES fault-location encoding experiment v0.3."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from gridguardpinn.andes_experiment import run_surrogate_experiment, write_experiment_artifacts
from gridguardpinn.andes_location_features import build_ieee14_fault_location_features
from gridguardpinn.andes_multimachine import generate_reference_batch
from gridguardpinn.andes_surrogate_protocol import assert_protocol_integrity, surrogate_splits_v01
from gridguardpinn.andes_training import AndesTrainingConfig

ARMS = ("one_hot", "network_aware")
SPLITS = ("validation", "test_id", "ood_duration", "ood_location")


def _finite(values):
    return np.asarray(
        [
            float(value)
            for value in values
            if value is not None and np.isfinite(value)
        ],
        dtype=float,
    )


def _aggregate(values):
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
    reports = summary["reports"][split_name]
    residual = reports["residual_only"]
    location_hard = reports["combined_location_hard"]
    strict = reports["combined_strict"]
    return {
        "arm": arm,
        "seed": seed,
        "split": split_name,
        "good_cases": accuracy["good_cases"],
        "good_fraction": accuracy["good_fraction"],
        "composite_mean": accuracy["composite_mean"],
        "composite_median": accuracy["composite_median"],
        "composite_max": accuracy["composite_max"],
        "max_delta_rmse_rad": accuracy["max_delta_rmse_rad"],
        "max_omega_rmse_pu": accuracy["max_omega_rmse_pu"],
        "residual_error_spearman": accuracy["residual_error_spearman"],
        "residual_coverage": residual["coverage"],
        "residual_false_accepts": residual["false_accepts"],
        "residual_false_escalations": residual["false_escalations"],
        "location_hard_coverage": location_hard["coverage"],
        "location_hard_false_accepts": location_hard["false_accepts"],
        "strict_coverage": strict["coverage"],
        "strict_false_accepts": strict["false_accepts"],
        "median_inference_seconds": accuracy["surrogate_inference"][
            "median_seconds"
        ],
        "median_reference_seconds": accuracy["reference_simulation"][
            "median_seconds"
        ],
        "median_speedup_vs_reference": accuracy["median_speedup_vs_reference"],
        "best_epoch": summary["training"]["best_epoch"],
        "training_seconds": summary["training"]["training_seconds"],
    }


def _aggregate_rows(flat_rows):
    metrics = (
        "good_cases",
        "good_fraction",
        "composite_mean",
        "composite_median",
        "composite_max",
        "max_delta_rmse_rad",
        "max_omega_rmse_pu",
        "residual_error_spearman",
        "residual_coverage",
        "residual_false_accepts",
        "residual_false_escalations",
        "location_hard_coverage",
        "location_hard_false_accepts",
        "strict_coverage",
        "strict_false_accepts",
        "median_inference_seconds",
        "median_reference_seconds",
        "median_speedup_vs_reference",
    )
    output = {}
    for arm in ARMS:
        output[arm] = {}
        for split_name in SPLITS:
            rows = [
                row
                for row in flat_rows
                if row["arm"] == arm and row["split"] == split_name
            ]
            output[arm][split_name] = {
                metric: _aggregate([row[metric] for row in rows])
                for metric in metrics
            }
        train_rows = [
            row
            for row in flat_rows
            if row["arm"] == arm and row["split"] == "validation"
        ]
        output[arm]["training_seconds"] = _aggregate(
            [row["training_seconds"] for row in train_rows]
        )
    return output


def _paired_rows(flat_rows):
    lookup = {
        (row["arm"], row["seed"], row["split"]): row
        for row in flat_rows
    }
    rows = []
    for seed in sorted({row["seed"] for row in flat_rows}):
        for split_name in SPLITS:
            baseline = lookup[("one_hot", seed, split_name)]
            network = lookup[("network_aware", seed, split_name)]
            rows.append(
                {
                    "seed": seed,
                    "split": split_name,
                    "one_hot_good_cases": baseline["good_cases"],
                    "network_good_cases": network["good_cases"],
                    "delta_good_cases": (
                        network["good_cases"] - baseline["good_cases"]
                    ),
                    "one_hot_composite_mean": baseline["composite_mean"],
                    "network_composite_mean": network["composite_mean"],
                    "delta_composite_mean": (
                        network["composite_mean"]
                        - baseline["composite_mean"]
                    ),
                    "one_hot_composite_median": baseline[
                        "composite_median"
                    ],
                    "network_composite_median": network[
                        "composite_median"
                    ],
                    "delta_composite_median": (
                        network["composite_median"]
                        - baseline["composite_median"]
                    ),
                    "one_hot_residual_coverage": baseline[
                        "residual_coverage"
                    ],
                    "network_residual_coverage": network[
                        "residual_coverage"
                    ],
                }
            )
    return rows


def _success_check(paired_rows):
    location = [
        row for row in paired_rows if row["split"] == "ood_location"
    ]
    lower_median_seeds = sum(
        row["network_composite_median"]
        < row["one_hot_composite_median"]
        for row in location
    )
    passing_location_seeds = sum(
        row["network_good_cases"] > 0
        for row in location
    )
    id_rows = [row for row in paired_rows if row["split"] == "test_id"]
    duration_rows = [
        row for row in paired_rows if row["split"] == "ood_duration"
    ]
    id_non_regression = all(
        row["network_good_cases"] == 7
        for row in id_rows
    )
    duration_non_regression = all(
        row["network_good_cases"] == 7
        for row in duration_rows
    )
    return {
        "location_median_lower_in_at_least_2_of_3_seeds": (
            lower_median_seeds >= 2
        ),
        "location_has_passing_cases_in_at_least_2_of_3_seeds": (
            passing_location_seeds >= 2
        ),
        "fresh_id_remains_7_of_7_all_seeds": id_non_regression,
        "duration_ood_remains_7_of_7_all_seeds": duration_non_regression,
        "pre_registered_location_generalisation_success": (
            lower_median_seeds >= 2
            and passing_location_seeds >= 2
            and id_non_regression
            and duration_non_regression
        ),
        "seeds_with_lower_location_median": lower_median_seeds,
        "seeds_with_any_passing_location_case": passing_location_seeds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--anchors", type=int, default=101)
    parser.add_argument("--collocation", type=int, default=56)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 41])
    parser.add_argument(
        "--output-dir",
        default="artifacts/andes_location_encoding_v0_3",
    )
    args = parser.parse_args()

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
    features = build_ieee14_fault_location_features()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "fault_location_features.json").write_text(
        json.dumps(features.metadata(), indent=2),
        encoding="utf-8",
    )

    flat_rows = []
    summaries = {}
    for arm in ARMS:
        summaries[arm] = {}
        descriptors = (
            None if arm == "one_hot" else features.descriptors
        )
        for seed in args.seeds:
            config = AndesTrainingConfig(
                epochs=args.epochs,
                anchors_per_case=args.anchors,
                collocation_per_case=args.collocation,
                physics_weight=0.02,
                seed=seed,
            )
            metadata = {
                "location_encoding": arm,
                "feature_names": (
                    None
                    if arm == "one_hot"
                    else list(features.feature_names)
                ),
                "generator_buses": (
                    None
                    if arm == "one_hot"
                    else list(features.generator_buses)
                ),
            }
            result = run_surrogate_experiment(
                reference_batch,
                splits,
                config=config,
                protocol="andes-location-encoding-v0.3",
                run_label=f"{arm}-seed-{seed}",
                fault_descriptors=descriptors,
                experiment_metadata=metadata,
            )
            write_experiment_artifacts(
                result,
                splits,
                output_dir=output / arm / f"seed_{seed}",
            )
            summaries[arm][str(seed)] = result.summary
            for split_name in SPLITS:
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
        writer = csv.DictWriter(
            handle,
            fieldnames=list(flat_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(flat_rows)

    paired = _paired_rows(flat_rows)
    with (output / "paired_encoding_deltas.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(paired[0].keys()),
        )
        writer.writeheader()
        writer.writerows(paired)

    aggregate = {
        "protocol": "andes-location-encoding-v0.3",
        "seeds": args.seeds,
        "reference_generation": {
            "andes_solve_total_seconds": reference_batch.total_seconds,
            "pipeline_total_seconds": reference_batch.pipeline_seconds,
        },
        "aggregate": _aggregate_rows(flat_rows),
        "pre_registered_success": _success_check(paired),
        "per_seed": summaries,
    }
    (output / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2),
        encoding="utf-8",
    )
    print("ANDES_LOCATION_ENCODING_SUMMARY_JSON=" + json.dumps(aggregate))


if __name__ == "__main__":
    main()
