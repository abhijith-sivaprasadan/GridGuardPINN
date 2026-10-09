"""Run the frozen v0.2 parametric-PINN trust experiment."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from gridguardpinn.calibration import baseline_reports, calibrate_gate
from gridguardpinn.evaluation import evaluate_split
from gridguardpinn.scenarios import canonical_splits, scenario_vector
from gridguardpinn.training import TrainingConfig, train_parametric_pinn
from gridguardpinn.trust import MahalanobisOOD


def residual_error_correlation(rows: list[dict[str, float | str | int]]) -> float:
    if len(rows) < 3:
        return float("nan")
    residual = [float(row["residual_score"]) for row in rows]
    error = [float(row["composite_error_ratio"]) for row in rows]
    result = spearmanr(residual, error)
    return float(result.statistic)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--anchors", type=int, default=61)
    parser.add_argument("--collocation", type=int, default=48)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default="artifacts/experiment_v0_2")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    splits = canonical_splits()
    config = TrainingConfig(
        epochs=args.epochs,
        anchors_per_case=args.anchors,
        collocation_per_case=args.collocation,
        seed=args.seed,
    )
    training = train_parametric_pinn(splits["train"], config=config)

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

    all_rows = [row for rows in rows_by_split.values() for row in rows]
    with (output / "case_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    summary = {
        "protocol": "v0.2-frozen-before-parametric-results",
        "split_sizes": {name: len(cases) for name, cases in splits.items()},
        "training": training.metadata(),
        "gate": asdict(gate),
        "reports": reports,
        "residual_error_spearman": correlations,
        "provisional_good_case_definition": {
            "delta_rmse_rad_lte": gate.angle_tolerance_rad,
            "omega_rmse_pu_lte": gate.speed_tolerance_pu,
            "composite_error_ratio_lte": 1.0,
        },
    }

    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8"
    )
    (output / "training_history.json").write_text(
        json.dumps(training.history, indent=2), encoding="utf-8"
    )
    torch.save(
        {
            "state_dict": training.model.state_dict(),
            "metadata": training.metadata(),
        },
        output / "model.pt",
    )

    print("EXPERIMENT_SUMMARY_JSON=" + json.dumps(summary, allow_nan=True))


if __name__ == "__main__":
    main()
