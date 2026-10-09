"""Trust signals, OOD scoring, deterministic gating, and gate evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class MahalanobisOOD:
    """Simple parameter-space OOD score fitted only on training scenarios."""

    def __init__(self, ridge: float = 1e-6) -> None:
        self.ridge = ridge
        self.mean_: np.ndarray | None = None
        self.inv_cov_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "MahalanobisOOD":
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[0] < 2:
            raise ValueError("x must be a 2D array with at least two rows.")
        self.mean_ = x.mean(axis=0)
        covariance = np.cov(x, rowvar=False)
        covariance = np.atleast_2d(covariance)
        covariance += self.ridge * np.eye(covariance.shape[0])
        self.inv_cov_ = np.linalg.pinv(covariance)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.inv_cov_ is None:
            raise RuntimeError("Fit the OOD model before scoring.")
        x = np.atleast_2d(np.asarray(x, dtype=float))
        centered = x - self.mean_
        squared = np.einsum("ni,ij,nj->n", centered, self.inv_cov_, centered)
        return np.sqrt(np.maximum(squared, 0.0))


@dataclass(frozen=True)
class TrustSignals:
    residual_rms: float
    ood_score: float
    uncertainty: float | None = None


@dataclass(frozen=True)
class TrustDecision:
    accepted: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class TrustGate:
    residual_threshold: float
    ood_threshold: float
    uncertainty_threshold: float | None = None

    def decide(self, signals: TrustSignals) -> TrustDecision:
        reasons: list[str] = []
        if signals.residual_rms > self.residual_threshold:
            reasons.append("physics_residual")
        if signals.ood_score > self.ood_threshold:
            reasons.append("out_of_distribution")
        if (
            self.uncertainty_threshold is not None
            and signals.uncertainty is not None
            and signals.uncertainty > self.uncertainty_threshold
        ):
            reasons.append("uncertainty")
        return TrustDecision(accepted=not reasons, reasons=tuple(reasons))


def evaluate_gate(
    accepted: np.ndarray,
    true_errors: np.ndarray,
    *,
    error_tolerance: float,
) -> dict[str, float | int]:
    """Evaluate safety and usefulness of a frozen deterministic gate."""
    accepted = np.asarray(accepted, dtype=bool)
    true_errors = np.asarray(true_errors, dtype=float)
    if accepted.shape != true_errors.shape:
        raise ValueError("accepted and true_errors must have the same shape.")
    if error_tolerance <= 0:
        raise ValueError("error_tolerance must be positive.")

    good = true_errors <= error_tolerance
    false_accept = accepted & ~good
    false_escalate = ~accepted & good

    accepted_count = int(accepted.sum())
    rejected_count = int((~accepted).sum())
    n = int(accepted.size)

    accepted_error = float(np.mean(true_errors[accepted])) if accepted_count else float("nan")
    return {
        "n": n,
        "accepted": accepted_count,
        "rejected": rejected_count,
        "coverage": accepted_count / n if n else float("nan"),
        "false_accepts": int(false_accept.sum()),
        "false_accept_rate_total": float(false_accept.mean()) if n else float("nan"),
        "false_accept_rate_accepted": (
            int(false_accept.sum()) / accepted_count if accepted_count else float("nan")
        ),
        "false_escalations": int(false_escalate.sum()),
        "false_escalation_rate_total": float(false_escalate.mean()) if n else float("nan"),
        "mean_error_accepted": accepted_error,
    }
