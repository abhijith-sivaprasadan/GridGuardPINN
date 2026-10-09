"""Validation-only calibration and gate comparisons."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .trust import evaluate_gate


@dataclass(frozen=True)
class CalibratedGate:
    residual_threshold: float
    ood_threshold: float
    angle_tolerance_rad: float = 0.05
    speed_tolerance_pu: float = 5e-4


def _arrays(rows: list[dict[str, float | str | int]]) -> tuple[np.ndarray, ...]:
    residual = np.asarray([float(r["residual_score"]) for r in rows])
    ood = np.asarray([float(r["ood_score"]) for r in rows])
    error = np.asarray([float(r["composite_error_ratio"]) for r in rows])
    return residual, ood, error


def calibrate_gate(
    validation_rows: list[dict[str, float | str | int]],
    *,
    in_distribution_ood_scores: np.ndarray,
) -> CalibratedGate:
    """Tune residual threshold on validation without inspecting held-out tests."""
    residual, ood, error = _arrays(validation_rows)
    reference = np.asarray(in_distribution_ood_scores, dtype=float)
    ood_threshold = float(np.quantile(reference, 0.99) * 1.001)

    candidates = np.unique(
        np.concatenate(
            [
                [np.nextafter(float(residual.min()), -np.inf)],
                residual,
                [float(residual.max()) * 1.001],
            ]
        )
    )

    best_threshold = float(candidates[0])
    best_accepted = -1
    for threshold in candidates:
        accepted = (residual <= threshold) & (ood <= ood_threshold)
        bad = error > 1.0
        false_accepts = int(np.sum(accepted & bad))
        accepted_count = int(accepted.sum())
        if false_accepts == 0 and accepted_count > best_accepted:
            best_threshold = float(threshold)
            best_accepted = accepted_count

    return CalibratedGate(
        residual_threshold=best_threshold,
        ood_threshold=ood_threshold,
    )


def apply_gate(
    rows: list[dict[str, float | str | int]],
    gate: CalibratedGate,
) -> np.ndarray:
    residual, ood, _ = _arrays(rows)
    return (residual <= gate.residual_threshold) & (ood <= gate.ood_threshold)


def gate_report(
    rows: list[dict[str, float | str | int]],
    gate: CalibratedGate,
) -> dict[str, float | int]:
    accepted = apply_gate(rows, gate)
    error = np.asarray([float(r["composite_error_ratio"]) for r in rows])
    return evaluate_gate(accepted, error, error_tolerance=1.0)


def baseline_reports(
    rows: list[dict[str, float | str | int]],
    gate: CalibratedGate,
) -> dict[str, dict[str, float | int]]:
    residual, ood, error = _arrays(rows)
    return {
        "always_accept": evaluate_gate(
            np.ones(len(rows), dtype=bool), error, error_tolerance=1.0
        ),
        "ood_only": evaluate_gate(
            ood <= gate.ood_threshold, error, error_tolerance=1.0
        ),
        "residual_only": evaluate_gate(
            residual <= gate.residual_threshold, error, error_tolerance=1.0
        ),
        "combined": gate_report(rows, gate),
    }
