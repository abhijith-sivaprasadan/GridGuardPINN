"""High-accuracy numerical reference integration for SMIB cases."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from scipy.integrate import solve_ivp

from .dynamics import SMIBScenario, state_derivative


@dataclass(frozen=True)
class SimulationResult:
    t: np.ndarray
    delta: np.ndarray
    omega: np.ndarray
    scenario: SMIBScenario

    @property
    def states(self) -> np.ndarray:
        return np.column_stack([self.delta, self.omega])


def _segment_pmax(scenario: SMIBScenario, start: float, end: float) -> float:
    midpoint = 0.5 * (start + end)
    return float(scenario.pmax_at(midpoint))


def simulate_reference(
    scenario: SMIBScenario,
    *,
    samples: int = 1001,
    rtol: float = 1e-9,
    atol: float = 1e-11,
    max_step: float | None = None,
) -> SimulationResult:
    """Integrate each piecewise-smooth interval separately.

    Segmenting exactly at fault application and clearing avoids asking an
    adaptive ODE solver to step blindly across the discontinuous transfer limit.
    """
    if samples < 3:
        raise ValueError("samples must be at least 3.")

    times = np.linspace(0.0, scenario.t_end, samples)
    states = np.empty((samples, 2), dtype=float)
    y0 = scenario.initial_state.copy()

    boundaries = [0.0, scenario.t_fault, scenario.t_clear, scenario.t_end]
    for index, (start, end) in enumerate(pairwise(boundaries)):
        if end <= start:
            continue

        pmax = _segment_pmax(scenario, start, end)
        segment_max_step = max_step
        if segment_max_step is None:
            segment_max_step = min(0.01, (end - start) / 20.0)

        solution = solve_ivp(
            lambda t, y, pmax=pmax: state_derivative(
                t, y, scenario, fixed_pmax=pmax
            ),
            (start, end),
            y0,
            method="RK45",
            rtol=rtol,
            atol=atol,
            max_step=segment_max_step,
            dense_output=True,
        )
        if not solution.success:
            raise RuntimeError(f"Reference integration failed: {solution.message}")

        if index == 0:
            mask = (times >= start) & (times <= end)
        else:
            mask = (times > start) & (times <= end)
        states[mask] = solution.sol(times[mask]).T
        y0 = solution.y[:, -1]

    if not np.all(np.isfinite(states)):
        raise RuntimeError("Reference integration produced non-finite states.")

    return SimulationResult(
        t=times,
        delta=states[:, 0],
        omega=states[:, 1],
        scenario=scenario,
    )
