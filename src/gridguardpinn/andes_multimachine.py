"""Reference-data assembly for the frozen ANDES multi-machine experiment."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .andes_surrogate import raw_input


@dataclass(frozen=True)
class MachineConstants:
    inertia_M: np.ndarray
    damping_D: np.ndarray
    frequency_hz: np.ndarray


@dataclass(frozen=True)
class ReferenceBatch:
    """Reference trajectories plus measured wall-clock cost for each case."""

    trajectories: dict
    case_seconds: dict
    total_seconds: float


def event_aware_times(
    fault_duration_s: float,
    target_points: int = 101,
) -> np.ndarray:
    """Deterministic anchors with extra density around switching events."""
    if target_points < 50:
        raise ValueError("target_points must be at least 50.")
    clear = 1.0 + fault_duration_s
    uniform = np.linspace(0.0, 2.0, target_points - 32)
    fault_dense = np.linspace(0.96, 1.04, 17)
    clear_dense = np.linspace(clear - 0.04, clear + 0.04, 17)
    return np.unique(
        np.clip(
            np.concatenate([uniform, fault_dense, clear_dense]),
            0.0,
            2.0,
        )
    )


def collocation_times(
    fault_duration_s: float,
    *,
    points: int,
    rng: np.random.Generator,
    event_exclusion_s: float = 0.005,
) -> np.ndarray:
    """Sample physics points across pre-, during-, and post-fault phases."""
    if points < 12:
        raise ValueError("points must be at least 12.")
    clear = 1.0 + fault_duration_s
    eps = event_exclusion_s
    n_pre = points // 4
    n_fault = points // 4
    n_post = points - n_pre - n_fault
    pre = rng.uniform(0.0, 1.0 - eps, n_pre)
    fault = rng.uniform(1.0 + eps, clear - eps, n_fault)
    post = rng.uniform(clear + eps, 2.0, n_post)
    return np.sort(np.concatenate([pre, fault, post]))


def target_matrix(trajectory) -> np.ndarray:
    required = (
        trajectory.generator_angle_rad,
        trajectory.generator_speed_pu,
        trajectory.generator_mechanical_torque_pu,
        trajectory.generator_electrical_torque_pu,
    )
    if any(value is None for value in required):
        raise RuntimeError(
            "Reference trajectory is missing angle/speed/torque signals."
        )
    return np.column_stack(required)


def machine_constants(reference_map) -> MachineConstants:
    first = next(iter(reference_map.values()))
    arrays = (
        first.machine_inertia_M,
        first.machine_damping_D,
        first.machine_frequency_hz,
    )
    if any(value is None for value in arrays):
        raise RuntimeError("Reference trajectory is missing machine constants.")
    constants = MachineConstants(
        inertia_M=np.asarray(first.machine_inertia_M, dtype=float),
        damping_D=np.asarray(first.machine_damping_D, dtype=float),
        frequency_hz=np.asarray(first.machine_frequency_hz, dtype=float),
    )
    for trajectory in reference_map.values():
        if not np.allclose(trajectory.machine_inertia_M, constants.inertia_M):
            raise RuntimeError("Machine inertia changed across reference cases.")
        if not np.allclose(trajectory.machine_damping_D, constants.damping_D):
            raise RuntimeError("Machine damping changed across reference cases.")
        if not np.allclose(
            trajectory.machine_frequency_hz,
            constants.frequency_hz,
        ):
            raise RuntimeError("Machine frequency changed across reference cases.")
    return constants


def supervised_arrays(
    reference_map,
    cases,
    *,
    anchors_per_case: int,
) -> tuple[np.ndarray, np.ndarray]:
    from .andes_reference import resample_trajectory

    x_rows = []
    y_rows = []
    for case in cases:
        trajectory = reference_map[case]
        times = event_aware_times(case.fault_duration_s, anchors_per_case)
        sampled = resample_trajectory(trajectory, time_grid_s=times)
        x_rows.append(raw_input(times, case.fault_duration_s, case.fault_bus))
        y_rows.append(target_matrix(sampled))
    return np.vstack(x_rows), np.vstack(y_rows)


def collocation_array(
    cases,
    *,
    points_per_case: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = []
    for case in cases:
        times = collocation_times(
            case.fault_duration_s,
            points=points_per_case,
            rng=rng,
        )
        rows.append(raw_input(times, case.fault_duration_s, case.fault_bus))
    return np.vstack(rows)


def generate_reference_batch(cases, *, samples: int = 401) -> ReferenceBatch:
    """Run frozen cases and record per-case reference-simulation wall time."""
    from .andes_reference import resample_trajectory, run_ieee14_fault

    grid = np.linspace(0.0, 2.0, samples)
    reference_map = {}
    case_seconds = {}
    started = time.perf_counter()

    for case in cases:
        case_started = time.perf_counter()
        native = run_ieee14_fault(
            fault_bus=case.fault_bus,
            fault_start_s=1.0,
            fault_clear_s=1.0 + case.fault_duration_s,
            simulation_end_s=2.0,
            fault_reactance_pu=1e-4,
        )
        sampled = resample_trajectory(native, time_grid_s=grid)
        target_matrix(sampled)
        reference_map[case] = sampled
        case_seconds[case] = time.perf_counter() - case_started

    machine_constants(reference_map)
    return ReferenceBatch(
        trajectories=reference_map,
        case_seconds=case_seconds,
        total_seconds=time.perf_counter() - started,
    )


def generate_reference_map(cases, *, samples: int = 401):
    """Backward-compatible trajectory-only wrapper."""
    return generate_reference_batch(cases, samples=samples).trajectories
