"""Run a frozen GridGuardPINN trust experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from gridguardpinn.calibration import baseline_reports, calibrate_gate
from gridguardpinn.evaluation import evaluate_split
from gridguardpinn.scenarios import (
    scenario_vector,
    splits_v02,
    splits_v03,
    splits_v04,
)
from gridguardpinn.training import TrainingConfig, train_parametric_pinn
from gridguardpinn.trust import MahalanobisOOD


def residual_error_correlation(rows: list[dict[str, float | str | int]]) -> float:
    if len(rows) < 3:
        return float("nan")
    residual = [float(row["residual_score"]) for row in rows]
    error = [float(row["composite_error_ratio"]) for row in rows]
    return float(spearmanr(residual, error).statistic)


def error_summary(rows: list[dict[str, float | str | int]]) -> dict[str, float | int]:
    def values(key: str) -> np.ndarray:
        return np.asarray([float(row[key]) for row in rows], dtype=float)

    composite = values("composite_error_ratio")
    delta = values("delta_rmse_rad")
    omega = values("omega_rmse_pu")
    return {
        "composite_mean": float(np.mean(composite)),
        "composite_median": float(np.median(composite)),
        "composite_max": float(np.max(composite)),
        "delta_rmse_mean_rad": float(np.mean(delta)),
        "delta_rmse_max_rad": float(np.max(delta)),
        "omega_rmse_mean_pu": float(np.mean(omega)),
        "omega_rmse_max_pu": float(np.max(omega)),
        "good_cases": int(np.sum(composite <= 1.0)),
    }


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def choose_splits(protocol: str):
    factories = {
        "v0.2": splits_v02,
        "v0.3": splits_v03,
        "v0.4": splits_v04,
    }
    return factories[protocol]()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        choices=("v0.2", "v0.3", "v0.4"),
        default="v0.4",
    )
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--anchors", type=int, default=121)
    parser.add_argument("--collocation", type=int, default=96)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    output = Path(
        args.output_dir or f"artifacts/experiment_{args.protocol.replace('.', '_')}"
    )
    output.mkdir(parents=True, exist_ok=True)

    splits = choose_splits(args.protocol)
    config = TrainingConfig(
        epochs=args.epochs,
        anchors_per_case=args.anchors,
        collocation_per_case=args.collocation,
        seed=args.seed,
    )
    training = train_parametric_pinn(
        splits["train"],
        config=config,
        validation_scenarios=splits["validation"] if args.protocol == "v0.4" else None,
    )

    train_vectors = np.vstack([scenario_vector(case) for case in splits["train"]])
    detector = MahalanobisOOD().fit(train_vectors)

    rows_by_split: dict[str, list[dict[str, float | str | int]]] = {}
    for split_name in ("validation", "test_id", "ood"):
        rows_by_split[split_name] = evaluate_split(
            training.model,
            splits[split_name],
            split=split_name,
            ood_detector=detector,
        )

    train_ood = detector.score(train_vectors)
    validation_ood = np.asarray(
        [float(row["ood_score"]) for row in rows_by_split["validation"]]
    )
    gate = calibrate_gate(
        rows_by_split["validation"],
        in_distribution_ood_scores=np.concatenate([train_ood, validation_ood]),
    )

    reports = {
        split_name: baseline_reports(rows, gate)
        for split_name, rows in rows_by_split.items()
    }
    correlations = {
        split_name: residual_error_correlation(rows)
        for split_name, rows in rows_by_split.items()
    }
    errors = {
        split_name: error_summary(rows)
        for split_name, rows in rows_by_split.items()
    }

    ood_factor_reports: dict[str, dict[str, dict[str, float | int]]] = {}
    factor_names = sorted({str(row["ood_factors"]) for row in rows_by_split["ood"]})
    for factor in factor_names:
        factor_rows = [
            row for row in rows_by_split["ood"] if str(row["ood_factors"]) == factor
        ]
        ood_factor_reports[factor] = baseline_reports(factor_rows, gate)

    all_rows = [row for rows in rows_by_split.values() for row in rows]
    with (output / "case_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    summary = {
        "protocol": args.protocol,
        "git_sha": os.environ.get("GITHUB_SHA"),
        "split_sizes": {name: len(cases) for name, cases in splits.items()},
        "training": training.metadata(),
        "final_training_state": training.history[-1],
        "gate": asdict(gate),
        "error_summary": errors,
        "reports": reports,
        "ood_factor_reports": ood_factor_reports,
        "residual_error_spearman": correlations,
        "provisional_good_case_definition": {
            "delta_rmse_rad_lte": gate.angle_tolerance_rad,
            "omega_rmse_pu_lte": gate.speed_tolerance_pu,
            "composite_error_ratio_lte": 1.0,
        },
    }
    summary = json_safe(summary)

    (output / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (output / "training_history.json").write_text(
        json.dumps(json_safe(training.history), indent=2),
        encoding="utf-8",
    )
    torch.save(
        {
            "state_dict": training.model.state_dict(),
            "metadata": training.metadata(),
        },
        output / "model.pt",
    )

    print("EXPERIMENT_SUMMARY_JSON=" + json.dumps(summary))


if __name__ == "__main__":
    main()
