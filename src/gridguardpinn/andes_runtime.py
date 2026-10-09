"""Auditable research inference with reference fallback for IEEE-14 faults.

This is a research demonstrator, not an operational power-grid controller.
Surrogate outputs are restricted to 20 GENROU electromechanical channels;
reference trajectories also include bus voltages, which the surrogate does not.
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .andes_router import Route, RoutingDecision, route_andes_case


@dataclass(frozen=True)
class RuntimeResult:
    source: str
    decision: RoutingDecision
    time_s: np.ndarray
    electromechanical: np.ndarray | None
    reference: object | None


def sha256_file(path: str | Path) -> str:
    """Stream a checkpoint digest before deserializing it."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(
    path: str | Path, *, expected_sha256: str, expected_protocol: str,
):
    """Reconstruct only an explicitly trusted model checkpoint.

    A caller-supplied SHA-256 pinned to an independently trusted source is
    mandatory. Merely storing the digest beside the checkpoint is insufficient.
    """
    import torch

    from .andes_surrogate import AndesMultiMachineSurrogate

    if len(expected_sha256) != 64 or any(c not in "0123456789abcdef" for c in expected_sha256):
        raise ValueError("A lowercase, independently verified SHA-256 is required.")
    if not expected_protocol:
        raise ValueError("Expected training protocol must be specified.")
    if sha256_file(path) != expected_sha256:
        raise ValueError("Checkpoint SHA-256 mismatch; refusing to load.")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError("Unsupported checkpoint structure.")
    summary = checkpoint["summary"]
    if summary["protocol"] != expected_protocol:
        raise ValueError("Checkpoint protocol does not match the pinned protocol.")
    encoding = summary.get("experiment_metadata", {}).get("location_encoding")
    if encoding != "one_hot":
        raise ValueError("Only one-hot checkpoints supported by this runtime.")
    state = checkpoint["state_dict"]
    config = checkpoint["training"]["config"]
    if state["output_shift"].shape != (20,) or state["output_scale"].shape != (20,):
        raise ValueError("Unexpected model output scaling.")
    model = AndesMultiMachineSurrogate(
        state["output_shift"].numpy(),
        state["output_scale"].numpy(),
        hidden_width=int(config["hidden_width"]),
        hidden_layers=int(config["hidden_layers"]),
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    threshold = float(checkpoint["gate"]["residual_threshold"])
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("Invalid checkpoint gate threshold.")
    return model, threshold, summary


def run_routed_case(
    *,
    fault_bus: int,
    fault_duration_s: float,
    residual_score: float,
    residual_threshold: float,
    predict: Callable[[np.ndarray, int, float], np.ndarray],
    reference: Callable[[int, float], object],
    time_grid_s: np.ndarray | None = None,
    model_ready: bool = True,
) -> RuntimeResult:
    """Route then execute; never invoke the surrogate for a rejected case.

    Callers must supply an independently operating reference function. A
    reference exception propagates rather than returning a surrogate answer.
    """
    if time_grid_s is None:
        time_grid_s = np.linspace(0.0, 2.0, 401)
    grid = np.asarray(time_grid_s, dtype=float)
    if (
        grid.ndim != 1
        or len(grid) < 2
        or not np.all(np.isfinite(grid))
        or np.any(np.diff(grid) <= 0)
        or not np.isclose(grid[0], 0.0, atol=1e-10)
        or not np.isclose(grid[-1], 2.0, atol=1e-10)
    ):
        raise ValueError("A finite ascending time grid in [0, 2] is required.")
    decision = route_andes_case(
        fault_bus=fault_bus,
        fault_duration_s=fault_duration_s,
        residual_score=residual_score,
        residual_threshold=residual_threshold,
        model_ready=model_ready,
    )
    if decision.route == Route.ACCEPT_SURROGATE:
        try:
            prediction = np.asarray(
                predict(grid, fault_bus, fault_duration_s), dtype=float
            )
            if prediction.shape != (len(grid), 20) or not np.all(np.isfinite(prediction)):
                raise ValueError("Non-finite or malformed surrogate trajectory.")
        except (ValueError, RuntimeError, FloatingPointError):
            decision = RoutingDecision(Route.RUN_REFERENCE, "surrogate_prediction_failed")
        else:
            return RuntimeResult("surrogate", decision, grid, prediction, None)
    # Reference failure is an explicit error; never silently return a surrogate.
    native = reference(fault_bus, fault_duration_s)
    from .andes_reference import resample_trajectory
    sampled = resample_trajectory(native, time_grid_s=grid)
    channels = (
        sampled.generator_angle_rad,
        sampled.generator_speed_pu,
        sampled.generator_mechanical_torque_pu,
        sampled.generator_electrical_torque_pu,
    )
    if any(channel is None for channel in channels):
        raise RuntimeError("Reference trajectory lacks electromechanical channels.")
    values = np.column_stack(channels)
    if not np.all(np.isfinite(values)):
        raise RuntimeError("Reference simulator returned non-finite outputs.")
    return RuntimeResult("reference", decision, grid, values, sampled)


def andes_reference_fault(fault_bus: int, fault_duration_s: float):
    """Actual ANDES reference callback for the frozen IEEE-14 experiment."""
    from .andes_reference import run_ieee14_fault

    return run_ieee14_fault(
        fault_bus=fault_bus,
        fault_start_s=1.0,
        fault_clear_s=1.0 + fault_duration_s,
        simulation_end_s=2.0,
        fault_reactance_pu=1e-4,
    )


def model_prediction(model):
    """Bind a model to the runtime's 20-channel prediction signature."""
    import torch

    from .andes_surrogate import raw_input

    def predict(grid: np.ndarray, bus: int, duration: float) -> np.ndarray:
        x = torch.tensor(raw_input(grid, duration, bus), dtype=torch.float32)
        with torch.no_grad():
            return model(x).detach().cpu().numpy()

    return predict


def compute_model_residual(
    model,
    *,
    fault_bus: int,
    fault_duration_s: float,
    inertia_M: np.ndarray,
    damping_D: np.ndarray,
    frequency_hz: np.ndarray,
    samples: int = 151,
) -> float:
    """Match the frozen evaluation's residual score without using target errors.

    Machine constants must be supplied from independently trusted IEEE-14
    configuration; they cannot be inferred from a learned prediction.
    """
    import torch

    from .andes_surrogate import raw_input, swing_residuals

    if samples < 3 or not math.isfinite(fault_duration_s):
        raise ValueError("Invalid collocation grid or fault duration.")
    values = (inertia_M, damping_D, frequency_hz)
    vectors = [np.asarray(v, dtype=float) for v in values]
    if any(v.shape != (5,) or not np.all(np.isfinite(v)) for v in vectors):
        raise ValueError("Five finite trusted machine constants required per family.")
    clear = 1.0 + fault_duration_s
    candidate = np.linspace(0.0, 2.0, samples)
    mask = (np.abs(candidate - 1.0) > 0.006) & (
        np.abs(candidate - clear) > 0.006
    )
    if not np.any(mask):
        raise ValueError("No physics collocation points remain.")
    x = torch.tensor(
        raw_input(candidate[mask], fault_duration_s, fault_bus),
        dtype=torch.float32,
        requires_grad=True,
    )
    delta, omega = swing_residuals(
        model,
        x,
        inertia_M=torch.tensor(vectors[0], dtype=torch.float32),
        damping_D=torch.tensor(vectors[1], dtype=torch.float32),
        frequency_hz=torch.tensor(vectors[2], dtype=torch.float32),
    )
    score = float(
        torch.sqrt(torch.mean((delta / 0.2) ** 2) +
                   torch.mean((omega / 0.05) ** 2)).detach()
    )
    if not math.isfinite(score):
        raise ValueError("Non-finite physics residual.")
    return score


def run_model_case(
    *,
    model,
    fault_bus: int,
    fault_duration_s: float,
    residual_threshold: float,
    inertia_M: np.ndarray,
    damping_D: np.ndarray,
    frequency_hz: np.ndarray,
    reference: Callable[[int, float], object] = andes_reference_fault,
) -> RuntimeResult:
    """End-to-end research route with the frozen residual definition.

    For out-of-envelope cases, refuse before evaluating model physics. Any
    residual-computation failure routes to the independent reference.
    """
    preliminary = route_andes_case(
        fault_bus=fault_bus,
        fault_duration_s=fault_duration_s,
        residual_score=0.0,
        residual_threshold=residual_threshold,
        model_ready=model is not None,
    )
    if not preliminary.use_surrogate:
        return run_routed_case(
            fault_bus=fault_bus,
            fault_duration_s=fault_duration_s,
            residual_score=0.0,
            residual_threshold=residual_threshold,
            predict=lambda *args: (_ for _ in ()).throw(
                RuntimeError("Refused case must not invoke surrogate")
            ),
            reference=reference,
            model_ready=model is not None,
        )
    try:
        residual = compute_model_residual(
            model,
            fault_bus=fault_bus,
            fault_duration_s=fault_duration_s,
            inertia_M=inertia_M,
            damping_D=damping_D,
            frequency_hz=frequency_hz,
        )
    except (ValueError, RuntimeError, FloatingPointError):
        # Preserve the underlying *category* in the route ledger rather than
        # mislabelling a physics-evaluation failure as a non-finite user input.
        fallback = run_routed_case(
            fault_bus=fault_bus,
            fault_duration_s=fault_duration_s,
            residual_score=float("inf"),
            residual_threshold=residual_threshold,
            predict=lambda *args: (_ for _ in ()).throw(
                RuntimeError("Residual failed; surrogate must not execute")
            ),
            reference=reference,
        )
        return RuntimeResult(
            source=fallback.source,
            decision=RoutingDecision(Route.RUN_REFERENCE, "residual_computation_failed"),
            time_s=fallback.time_s,
            electromechanical=fallback.electromechanical,
            reference=fallback.reference,
        )
    return run_routed_case(
        fault_bus=fault_bus,
        fault_duration_s=fault_duration_s,
        residual_score=residual,
        residual_threshold=residual_threshold,
        predict=model_prediction(model),
        reference=reference,
    )
