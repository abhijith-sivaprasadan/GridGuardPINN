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


def build_supervised_dataset(
    scenarios: list[SMIBScenario],
    *,
    samples_per_case: int = 61,
) -> SupervisedDataset:
    if samples_per_case < 5:
        raise ValueError("samples_per_case must be at least 5.")

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    case_ids: list[np.ndarray] = []
    for case_id, scenario in enumerate(scenarios):
        result = simulate_reference(scenario, samples=samples_per_case)
        xs.append(feature_row(result.t, scenario))
        ys.append(result.states)
        case_ids.append(np.full(samples_per_case, case_id, dtype=int))
    return SupervisedDataset(
        x=np.vstack(xs),
        y=np.vstack(ys),
        case_index=np.concatenate(case_ids),
    )


def build_collocation_points(
    scenarios: list[SMIBScenario],
    *,
    points_per_case: int = 48,
    event_exclusion_s: float = 0.01,
    seed: int = 7,
) -> np.ndarray:
    """Sample fixed collocation points away from event discontinuities."""
    if points_per_case < 4:
        raise ValueError("points_per_case must be at least 4.")
    if event_exclusion_s < 0:
        raise ValueError("event_exclusion_s must be non-negative.")

    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    for scenario in scenarios:
        accepted: list[float] = []
        while len(accepted) < points_per_case:
            candidates = rng.uniform(0.0, scenario.t_end, size=points_per_case * 3)
            mask = (
                (np.abs(candidates - scenario.t_fault) > event_exclusion_s)
                & (np.abs(candidates - scenario.t_clear) > event_exclusion_s)
            )
            accepted.extend(candidates[mask].tolist())
        rows.append(feature_row(np.asarray(accepted[:points_per_case]), scenario))
    return np.vstack(rows)


def normalization_from_training(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-feature centre and scale derived from training inputs only."""
    x = np.asarray(x, dtype=float)
    centre = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return centre, scale
