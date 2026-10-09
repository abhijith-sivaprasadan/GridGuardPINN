"""Reusable execution and artifact handling for ANDES surrogate experiments."""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from .andes_evaluation import evaluate_split
from .andes_gate import AndesGate, calibrate_gate, gate_reports
from .andes_multimachine import ReferenceBatch
from .andes_training import AndesTrainingConfig, AndesTrainingResult, train_andes_surrogate


@dataclass
class AndesExperimentResult:
    """In-memory result for one frozen surrogate training/evaluation arm."""

    training: AndesTrainingResult
    rows_by_split: dict[str, list[dict[str, object]]]
    gate: AndesGate
    summary: dict[str, object]


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _timing_summary(values) -> dict[str, float]:
    values = np.asarray(list(values), dtype=float)
    if values.size == 0:
        return {
            "mean_seconds": float("nan"),
            "median_seconds": float("nan"),
            "max_seconds": float("nan"),
        }
    return {
        "mean_seconds": float(np.mean(values)),
        "median_seconds": float(np.median(values)),
        "max_seconds": float(np.max(values)),
    }


def split_summary(rows) -> dict[str, object]:
    composite = np.asarray([float(row["composite_error_ratio"]) for row in rows])
    delta = np.asarray([float(row["max_machine_delta_rmse_rad"]) for row in rows])
    omega = np.asarray([float(row["max_machine_omega_rmse_pu"]) for row in rows])
    residual = np.asarray([float(row["residual_score"]) for row in rows])
    inference = np.asarray(
        [float(row["inference_seconds_trajectory"]) for row in rows]
    )
    reference = np.asarray(
        [float(row["reference_simulation_seconds"]) for row in rows]
    )

    correlation = float("nan")
    if len(rows) >= 3 and np.std(residual) > 0 and np.std(composite) > 0:
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
        "surrogate_inference": _timing_summary(inference),
        "reference_simulation": _timing_summary(reference),
        "median_speedup_vs_reference": float(
            np.median(reference / np.maximum(inference, 1e-12))
        ),
    }


def run_surrogate_experiment(
    reference_batch: ReferenceBatch,
    splits,
    *,
    config: AndesTrainingConfig,
    protocol: str,
    run_label: str | None = None,
) -> AndesExperimentResult:
    """Train one arm, evaluate frozen splits, and calibrate the gate on validation only."""
    reference_map = reference_batch.trajectories
    training = train_andes_surrogate(
        reference_map,
        splits["train"],
        splits["validation"],
        config=config,
    )

    rows_by_split: dict[str, list[dict[str, object]]] = {}
    for split_name in (
        "validation",
        "test_id",
        "ood_duration",
        "ood_location",
    ):
        cases = splits[split_name]
        rows = evaluate_split(
            training.model,
            reference_map,
            cases,
            split=split_name,
        )
        for row, case in zip(rows, cases, strict=True):
            row["reference_simulation_seconds"] = float(
                reference_batch.case_seconds[case]
            )
        rows_by_split[split_name] = rows

    gate = calibrate_gate(rows_by_split["validation"])
    reports = {
        name: gate_reports(rows, gate)
        for name, rows in rows_by_split.items()
    }
    summaries = {
        name: split_summary(rows)
        for name, rows in rows_by_split.items()
    }

    summary = {
        "protocol": protocol,
        "run_label": run_label,
        "git_sha": os.environ.get("GITHUB_SHA"),
        "split_sizes": {
            name: len(cases)
            for name, cases in splits.items()
        },
        "reference_generation": {
            "total_seconds": reference_batch.total_seconds,
            **_timing_summary(reference_batch.case_seconds.values()),
        },
        "training": training.metadata(),
        "gate": asdict(gate),
        "accuracy": summaries,
        "reports": reports,
        "quality_screen": {
            "max_machine_delta_rmse_rad_lte": 0.05,
            "max_machine_omega_rmse_pu_lte": 5e-4,
        },
    }
    return AndesExperimentResult(
        training=training,
        rows_by_split=rows_by_split,
        gate=gate,
        summary=json_safe(summary),
    )


def write_experiment_artifacts(
    result: AndesExperimentResult,
    splits,
    *,
    output_dir: str | Path,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    all_rows = [
        row
        for rows in result.rows_by_split.values()
        for row in rows
    ]
    with (output / "case_metrics.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    (output / "summary.json").write_text(
        json.dumps(result.summary, indent=2),
        encoding="utf-8",
    )
    (output / "training_history.json").write_text(
        json.dumps(json_safe(result.training.history), indent=2),
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
            "state_dict": result.training.model.state_dict(),
            "training": result.training.metadata(),
            "gate": asdict(result.gate),
            "summary": result.summary,
        },
        output / "model.pt",
    )
