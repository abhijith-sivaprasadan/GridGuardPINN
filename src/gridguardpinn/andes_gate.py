"""Validation-only residual calibration and deterministic gate comparisons."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AndesGate:
    residual_threshold: float


def _error(rows) -> np.ndarray:
    return np.asarray(
        [float(row["composite_error_ratio"]) for row in rows],
        dtype=float,
    )


def _residual(rows) -> np.ndarray:
    return np.asarray(
        [float(row["residual_score"]) for row in rows],
        dtype=float,
    )


def calibrate_gate(validation_rows) -> AndesGate:
    """Choose maximum validation coverage subject to zero validation false accepts."""
    residual = _residual(validation_rows)
    error = _error(validation_rows)
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
        accepted = residual <= threshold
        false_accepts = int(np.sum(accepted & (error > 1.0)))
        accepted_count = int(accepted.sum())
        if false_accepts == 0 and accepted_count > best_accepted:
            best_threshold = float(threshold)
            best_accepted = accepted_count
    return AndesGate(residual_threshold=best_threshold)


def _metrics(
    accepted: np.ndarray,
    error: np.ndarray,
) -> dict[str, float | int]:
    accepted = np.asarray(accepted, dtype=bool)
    error = np.asarray(error, dtype=float)
    good = error <= 1.0
    false_accept = accepted & ~good
    false_escalate = ~accepted & good
    n = len(error)
    accepted_count = int(accepted.sum())
    return {
        "n": n,
        "accepted": accepted_count,
        "coverage": accepted_count / n if n else float("nan"),
        "false_accepts": int(false_accept.sum()),
        "false_accept_rate_accepted": (
            int(false_accept.sum()) / accepted_count
            if accepted_count
            else float("nan")
        ),
        "false_escalations": int(false_escalate.sum()),
        "mean_error_accepted": (
            float(np.mean(error[accepted]))
            if accepted_count
            else float("nan")
        ),
    }


def gate_reports(rows, gate: AndesGate):
    residual = _residual(rows)
    error = _error(rows)
    location_ood = np.asarray(
        [bool(row["location_ood"]) for row in rows],
        dtype=bool,
    )
    duration_ood = np.asarray(
        [bool(row["duration_ood"]) for row in rows],
        dtype=bool,
    )
    residual_ok = residual <= gate.residual_threshold
    explicit_ood = location_ood | duration_ood
    return {
        "always_accept": _metrics(
            np.ones(len(rows), dtype=bool),
            error,
        ),
        "residual_only": _metrics(residual_ok, error),
        "ood_only_strict": _metrics(~explicit_ood, error),
        "combined_strict": _metrics(
            residual_ok & ~explicit_ood,
            error,
        ),
        "combined_location_hard": _metrics(
            residual_ok & ~location_ood,
            error,
        ),
    }
