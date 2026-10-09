"""Auditable research inference with reference fallback for IEEE-14 faults.

This is a research demonstrator, not an operational power-grid controller.
Surrogate outputs are restricted to 20 GENROU electromechanical channels;
reference trajectories also include bus voltages, which the surrogate does not.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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
        raise ValueError("Unsupported checkpoint structure.")
    summary = checkpoint["summary"]
    if summary["protocol"] != expected_protocol:
        raise ValueError("Checkpoint protocol does not match the pinned protocol.")
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
        or grid[0] < 0
        or grid[-1] > 2.0
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
    from .andes_multimachine import target_matrix

    sampled = resample_trajectory(native, time_grid_s=grid)
    values = target_matrix(sampled)
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
