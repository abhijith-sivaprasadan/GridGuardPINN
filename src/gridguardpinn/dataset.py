"""Reference-data and collocation-point generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dynamics import SMIBScenario
from .reference import simulate_reference


@dataclass(frozen=True)
class SupervisedDataset:
    x: np.ndarray
    y: np.ndarray
    case_index: np.ndarray


def feature_row(t: np.ndarray, scenario: SMIBScenario) -> np.ndarray:
    """Build x=[t,H,D,t_clear,fault_ratio] rows for one case."""
    t = np.asarray(t, dtype=float).reshape(-1)
    n = t.size
    return np.column_stack(
        [
            t,
            np.full(n, scenario.H),
            np.full(n, scenario.D),
            np.full(n, scenario.t_clear),
            np.full(n, scenario.Pmax_fault / scenario.Pmax_pre),
        ]
    )


def event_aware_time_grid(
    scenario: SMIBScenario,
    *,
    samples: int = 81,
    early_window_s: float = 0.5,
) -> np.ndarray:
    """Return an exact-size grid with dense transient/event coverage."""
    if samples < 15:
        raise ValueError("samples must be at least 15 for event-aware sampling.")
    early_end = min(early_window_s, scenario.t_end)
    early_count = max(9, int(round(samples * 0.45)))
    late_count = samples - early_count

    early = np.linspace(0.0, early_end, early_count, endpoint=False)
    late = np.linspace(early_end, scenario.t_end, late_count)
    times = np.concatenate([early, late])

    event_targets = np.asarray(
        [
            max(0.0, scenario.t_fault - 0.01),
            scenario.t_fault,
            min(scenario.t_end, scenario.t_fault + 0.01),
            max(0.0, scenario.t_clear - 0.01),
            scenario.t_clear,
            min(scenario.t_end, scenario.t_clear + 0.01),
        ]
    )
    used: set[int] = set()
    for target in event_targets:
        order = np.argsort(np.abs(times - target))
        index = next(int(i) for i in order if int(i) not in used)
        times[index] = target
        used.add(index)

    return np.sort(times)


def build_supervised_dataset(
    scenarios: list[SMIBScenario],
    *,
    samples_per_case: int = 81,
    event_aware: bool = True,
) -> SupervisedDataset:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    case_ids: list[np.ndarray] = []

    for case_id, scenario in enumerate(scenarios):
        if event_aware:
            times = event_aware_time_grid(scenario, samples=samples_per_case)
            result = simulate_reference(scenario, times=times)
        else:
            result = simulate_reference(scenario, samples=samples_per_case)

        xs.append(feature_row(result.t, scenario))
        ys.append(result.states)
        case_ids.append(np.full(result.t.size, case_id, dtype=int))

    return SupervisedDataset(
        x=np.vstack(xs),
        y=np.vstack(ys),
        case_index=np.concatenate(case_ids),
    )


def build_collocation_points(
    scenarios: list[SMIBScenario],
    *,
    points_per_case: int = 64,
    event_exclusion_s: float = 0.005,
    seed: int = 7,
    phase_stratified: bool = True,
) -> np.ndarray:
    """Sample fixed collocation points with explicit fault-phase coverage."""
    if points_per_case < 12:
        raise ValueError("points_per_case must be at least 12.")
    if event_exclusion_s < 0:
        raise ValueError("event_exclusion_s must be non-negative.")

    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []

    for scenario in scenarios:
        if not phase_stratified:
            accepted: list[float] = []
            while len(accepted) < points_per_case:
                candidates = rng.uniform(0.0, scenario.t_end, size=points_per_case * 3)
                mask = (
                    (np.abs(candidates - scenario.t_fault) > event_exclusion_s)
                    & (np.abs(candidates - scenario.t_clear) > event_exclusion_s)
                )
                accepted.extend(candidates[mask].tolist())
            rows.append(feature_row(np.asarray(accepted[:points_per_case]), scenario))
            continue

        n_pre = max(4, int(round(points_per_case * 0.20)))
        n_fault = max(6, int(round(points_per_case * 0.25)))
        n_post = points_per_case - n_pre - n_fault
        if n_post < 4:
            raise ValueError("points_per_case too small for phase-stratified sampling.")

        pre_hi = max(0.0, scenario.t_fault - event_exclusion_s)
        fault_lo = min(scenario.t_clear, scenario.t_fault + event_exclusion_s)
        fault_hi = max(fault_lo, scenario.t_clear - event_exclusion_s)
        post_lo = min(scenario.t_end, scenario.t_clear + event_exclusion_s)

        pre = rng.uniform(0.0, pre_hi, n_pre)
        fault = rng.uniform(fault_lo, fault_hi, n_fault)
        post = rng.uniform(post_lo, scenario.t_end, n_post)
        rows.append(feature_row(np.concatenate([pre, fault, post]), scenario))

    return np.vstack(rows)


def normalization_from_training(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-feature centre and scale derived from training inputs only."""
    x = np.asarray(x, dtype=float)
    centre = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return centre, scale
