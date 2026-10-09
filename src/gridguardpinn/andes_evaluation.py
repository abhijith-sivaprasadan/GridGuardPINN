"""Case-level evaluation for the ANDES multi-machine surrogate."""

from __future__ import annotations

import math

import numpy as np
import torch

from .andes_multimachine import target_matrix
from .andes_surrogate import raw_input, swing_residuals

ANGLE_TOLERANCE_RAD = 0.05
SPEED_TOLERANCE_PU = 5e-4
TRAIN_DURATION_MIN_S = 0.04
TRAIN_DURATION_MAX_S = 0.12
TRAIN_BUSES = frozenset((3, 6, 7, 8, 9, 11, 14))


def trajectory_metrics(
    reference: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    error = np.asarray(prediction) - np.asarray(reference)
    delta_rmse = np.sqrt(np.mean(error[:, 0:5] ** 2, axis=0))
    omega_rmse = np.sqrt(np.mean(error[:, 5:10] ** 2, axis=0))
    tm_rmse = np.sqrt(np.mean(error[:, 10:15] ** 2, axis=0))
    te_rmse = np.sqrt(np.mean(error[:, 15:20] ** 2, axis=0))
    max_delta_rmse = float(np.max(delta_rmse))
    max_omega_rmse = float(np.max(omega_rmse))
    return {
        "max_machine_delta_rmse_rad": max_delta_rmse,
        "max_machine_omega_rmse_pu": max_omega_rmse,
        "mean_tm_rmse_pu": float(np.mean(tm_rmse)),
        "mean_te_rmse_pu": float(np.mean(te_rmse)),
        "max_abs_delta_error_rad": float(np.max(np.abs(error[:, 0:5]))),
        "max_abs_omega_error_pu": float(np.max(np.abs(error[:, 5:10]))),
        "composite_error_ratio": max(
            max_delta_rmse / ANGLE_TOLERANCE_RAD,
            max_omega_rmse / SPEED_TOLERANCE_PU,
        ),
    }


def evaluate_case(
    model,
    trajectory,
    case,
    *,
    residual_samples: int = 151,
):
    x_np = raw_input(
        trajectory.time_s,
        case.fault_duration_s,
        case.fault_bus,
    )
    x = torch.tensor(x_np, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        prediction = model(x).cpu().numpy()
    metrics = trajectory_metrics(target_matrix(trajectory), prediction)

    clear = 1.0 + case.fault_duration_s
    candidate = np.linspace(0.0, 2.0, residual_samples)
    mask = (
        (np.abs(candidate - 1.0) > 0.006)
        & (np.abs(candidate - clear) > 0.006)
    )
    residual_times = candidate[mask]
    x_res = torch.tensor(
        raw_input(
            residual_times,
            case.fault_duration_s,
            case.fault_bus,
        ),
        dtype=torch.float32,
        requires_grad=True,
    )
    inertia = torch.tensor(
        trajectory.machine_inertia_M,
        dtype=torch.float32,
    )
    damping = torch.tensor(
        trajectory.machine_damping_D,
        dtype=torch.float32,
    )
    frequency = torch.tensor(
        trajectory.machine_frequency_hz,
        dtype=torch.float32,
    )
    r_delta, r_omega = swing_residuals(
        model,
        x_res,
        inertia_M=inertia,
        damping_D=damping,
        frequency_hz=frequency,
    )
    r_delta_scaled = r_delta / 0.2
    r_omega_scaled = r_omega / 0.05
    residual_score = math.sqrt(
        float(torch.mean(r_delta_scaled**2).detach())
        + float(torch.mean(r_omega_scaled**2).detach())
    )
    per_machine = torch.sqrt(
        torch.mean(r_delta_scaled**2, dim=0)
        + torch.mean(r_omega_scaled**2, dim=0)
    )

    metrics.update(
        {
            "residual_score": residual_score,
            "max_machine_residual_score": float(
                torch.max(per_machine).detach()
            ),
            "location_ood": case.fault_bus not in TRAIN_BUSES,
            "duration_ood": not (
                TRAIN_DURATION_MIN_S
                <= case.fault_duration_s
                <= TRAIN_DURATION_MAX_S
            ),
            "fault_bus": case.fault_bus,
            "fault_duration_s": case.fault_duration_s,
        }
    )
    return metrics


def evaluate_split(
    model,
    reference_map,
    cases,
    *,
    split: str,
):
    rows = []
    for case_id, case in enumerate(cases):
        row = {"split": split, "case_id": case_id}
        row.update(evaluate_case(model, reference_map[case], case))
        rows.append(row)
    return rows
