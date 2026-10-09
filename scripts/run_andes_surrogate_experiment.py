"""Run the frozen ANDES IEEE-14 multi-machine surrogate experiment v0.1."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from gridguardpinn.andes_evaluation import evaluate_split
from gridguardpinn.andes_gate import calibrate_gate, gate_reports
from gridguardpinn.andes_multimachine import generate_reference_map
from gridguardpinn.andes_surrogate_protocol import (
    assert_protocol_integrity,
    surrogate_splits_v01,
)
from gridguardpinn.andes_training import (
    AndesTrainingConfig,
    train_andes_surrogate,
)


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def split_summary(rows):
    composite = np.asarray(
        [float(row["composite_error_ratio"]) for row in rows]
    )
    delta = np.asarray(
        [float(row["max_machine_delta_rmse_rad"]) for row in rows]
    )
    omega = np.asarray(
        [float(row["max_machine_omega_rmse_pu"]) for row in rows]
    )
    residual = np.asarray([float(row["residual_score"]) for row in rows])
    correlation = float("nan")
    if (
        len(rows) >= 3
        and np.std(residual) > 0
        and np.std(composite) > 0
    ):
        correlation = float(spearmanr(residual, composite).statistic)
    return {
        "cases": len(rows),
        "good_cases": int(np.sum(composite <= 1.0)),
        "good_fraction": float(np.mean(composite <= 1.0)),
        "composite_mean": float(np.mean(composite)),
        "composite_median": float(np.median(composite)),
        "composite_max": float(np.max(composite)),
        "max_delta_rmse_rad": float(np.max(delta)),
        "max_omega_rmse_pu": float(np.max(omega)),
        "residual_error_spearman": correlation,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--anchors", type=int, default=101)
    parser.add_argument("--collocation", type=int, default=56)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--output-dir",
        default="artifacts/andes_surrogate_v0_1",
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

    reference_started = time.perf_counter()
    reference_map = generate_reference_map(all_cases, samples=401)
    reference_seconds = time.perf_counter() - reference_started

    config = AndesTrainingConfig(
        epochs=args.epochs,
        anchors_per_case=args.anchors,
        collocation_per_case=args.collocation,
        seed=args.seed,
    )
    training = train_andes_surrogate(
        reference_map,
        splits["train"],
        splits["validation"],
        config=config,
    )

    rows_by_split = {}
    for split_name in (
        "validation",
        "test_id",
        "ood_duration",
        "ood_location",
    ):
        rows_by_split[split_name] = evaluate_split(
            training.model,
            reference_map,
            splits[split_name],
            split=split_name,
        )

    gate = calibrate_gate(rows_by_split["validation"])
    reports = {
        name: gate_reports(rows, gate)
        for name, rows in rows_by_split.items()
    }
    summaries = {
        name: split_summary(rows)
        for name, rows in rows_by_split.items()
    }

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    all_rows = [
        row
        for rows in rows_by_split.values()
        for row in rows
    ]
    with (output / "case_metrics.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(all_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(all_rows)

    summary = {
        "protocol": "andes-surrogate-v0.1",
        "git_sha": os.environ.get("GITHUB_SHA"),
        "split_sizes": {
            name: len(cases)
            for name, cases in splits.items()
        },
        "reference_generation_seconds": reference_seconds,
        "training": training.metadata(),
        "gate": asdict(gate),
        "accuracy": summaries,
        "reports": reports,
        "quality_screen": {
            "max_machine_delta_rmse_rad_lte": 0.05,
            "max_machine_omega_rmse_pu_lte": 5e-4,
        },
    }
    summary = json_safe(summary)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (output / "training_history.json").write_text(
        json.dumps(
            json_safe(training.history),
            indent=2,
        ),
        encoding="utf-8",
    )
    (output / "case_manifest.json").write_text(
        json.dumps(
            {
                name: [asdict(case) for case in cases]
                for name, cases in splits.items()
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    torch.save(
        {
            "state_dict": training.model.state_dict(),
            "training": training.metadata(),
            "gate": asdict(gate),
        },
        output / "model.pt",
    )
    print(
        "ANDES_SURROGATE_SUMMARY_JSON="
        + json.dumps(summary)
    )


if __name__ == "__main__":
    main()
