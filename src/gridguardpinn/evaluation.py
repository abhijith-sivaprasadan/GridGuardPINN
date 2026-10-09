"""Case-level trajectory and physics-residual evaluation."""

from __future__ import annotations

import math

import numpy as np
import torch

from .dataset import feature_row
from .dynamics import SMIBScenario
from .pinn import physics_residuals
from .reference import simulate_reference
from .scenarios import classify_ood_factors, scenario_vector
from .trust import MahalanobisOOD


def trajectory_metrics(
    reference: np.ndarray,
    prediction: np.ndarray,
    *,
    angle_tolerance_rad: float = 0.05,
    speed_tolerance_pu: float = 5e-4,
) -> dict[str, float]:
    error = np.asarray(prediction) - np.asarray(reference)
    delta_rmse = float(np.sqrt(np.mean(error[:, 0] ** 2)))
    omega_rmse = float(np.sqrt(np.mean(error[:, 1] ** 2)))
    delta_max = float(np.max(np.abs(error[:, 0])))
    omega_max = float(np.max(np.abs(error[:, 1])))
    composite = max(delta_rmse / angle_tolerance_rad, omega_rmse / speed_tolerance_pu)
    return {
        "delta_rmse_rad": delta_rmse,
        "omega_rmse_pu": omega_rmse,
        "delta_max_abs_rad": delta_max,
        "omega_max_abs_pu": omega_max,
        "composite_error_ratio": float(composite),
    }


def evaluate_case(
    model: torch.nn.Module,
    scenario: SMIBScenario,
    *,
    ood_detector: MahalanobisOOD,
    samples: int = 301,
    event_exclusion_s: float = 0.005,
) -> dict[str, float | str]:
    reference = simulate_reference(scenario, samples=samples)
    x_np = feature_row(reference.t, scenario)
    x = torch.tensor(x_np, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        prediction = model(x).cpu().numpy()
    metrics: dict[str, float | str] = trajectory_metrics(reference.states, prediction)

    mask = (
        (np.abs(reference.t - scenario.t_fault) > event_exclusion_s)
        & (np.abs(reference.t - scenario.t_clear) > event_exclusion_s)
    )
    x_res = torch.tensor(x_np[mask], dtype=torch.float32, requires_grad=True)
    r_delta, r_omega = physics_residuals(model, x_res)
    delta_rms = float(torch.sqrt(torch.mean(r_delta**2)).detach())
    omega_rms = float(torch.sqrt(torch.mean(r_omega**2)).detach())
    residual_score = math.sqrt(delta_rms**2 + (omega_rms / 0.1) ** 2)

    metrics.update(
        {
            "residual_delta_rms": delta_rms,
            "residual_omega_rms": omega_rms,
            "residual_score": float(residual_score),
            "ood_score": float(ood_detector.score(scenario_vector(scenario))[0]),
            "ood_factors": classify_ood_factors(scenario),
            "H": scenario.H,
            "D": scenario.D,
            "t_clear": scenario.t_clear,
            "fault_ratio": scenario.Pmax_fault / scenario.Pmax_pre,
        }
    )
    return metrics


def evaluate_split(
    model: torch.nn.Module,
    scenarios: list[SMIBScenario],
    *,
    split: str,
    ood_detector: MahalanobisOOD,
) -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []
    for case_id, scenario in enumerate(scenarios):
        row: dict[str, float | str | int] = {"split": split, "case_id": case_id}
        row.update(evaluate_case(model, scenario, ood_detector=ood_detector))
        rows.append(row)
    return rows
