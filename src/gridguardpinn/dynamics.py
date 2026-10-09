"""Classical single-machine infinite-bus (SMIB) dynamics.

The model is intentionally reduced-order. Speed is represented as per-unit
deviation from synchronous speed and rotor angle is in electrical radians.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SMIBScenario:
    """Parameters for a fault-disturbed classical SMIB swing-equation case."""

    H: float = 5.0
    D: float = 1.0
    Pm: float = 0.8
    Pmax_pre: float = 1.2
    Pmax_fault: float = 0.2
    Pmax_post: float = 1.0
    t_fault: float = 0.10
    t_clear: float = 0.20
    t_end: float = 5.0
    frequency_hz: float = 50.0
    delta0: float | None = None
    omega0: float = 0.0

    def __post_init__(self) -> None:
        if self.H <= 0:
            raise ValueError("H must be positive.")
        if self.D < 0:
            raise ValueError("D must be non-negative.")
        if self.frequency_hz <= 0:
            raise ValueError("frequency_hz must be positive.")
        if not (0 <= self.t_fault < self.t_clear < self.t_end):
            raise ValueError("Require 0 <= t_fault < t_clear < t_end.")
        if min(self.Pmax_pre, self.Pmax_fault, self.Pmax_post) < 0:
            raise ValueError("Transfer limits must be non-negative.")
        if self.delta0 is None and abs(self.Pm / self.Pmax_pre) >= 1:
            raise ValueError("Default equilibrium requires abs(Pm/Pmax_pre) < 1.")

    @property
    def omega_base(self) -> float:
        """Synchronous electrical angular speed in rad/s."""
        return 2.0 * math.pi * self.frequency_hz

    @property
    def initial_delta(self) -> float:
        """Initial rotor angle, defaulting to pre-fault equilibrium."""
        if self.delta0 is not None:
            return float(self.delta0)
        return float(math.asin(self.Pm / self.Pmax_pre))

    @property
    def initial_state(self) -> np.ndarray:
        return np.asarray([self.initial_delta, self.omega0], dtype=float)

    def pmax_at(self, t: float | np.ndarray) -> float | np.ndarray:
        """Piecewise transfer capability before, during, and after the fault."""
        if np.isscalar(t):
            t_scalar = float(t)
            if t_scalar < self.t_fault:
                return self.Pmax_pre
            if t_scalar < self.t_clear:
                return self.Pmax_fault
            return self.Pmax_post

        t_arr = np.asarray(t, dtype=float)
        return np.where(
            t_arr < self.t_fault,
            self.Pmax_pre,
            np.where(t_arr < self.t_clear, self.Pmax_fault, self.Pmax_post),
        )


def electrical_power(delta: float | np.ndarray, pmax: float | np.ndarray) -> float | np.ndarray:
    """Classical electrical air-gap power Pe = Pmax * sin(delta)."""
    return np.asarray(pmax) * np.sin(delta)


def state_derivative(
    t: float,
    state: np.ndarray,
    scenario: SMIBScenario,
    *,
    fixed_pmax: float | None = None,
) -> np.ndarray:
    """Return [d(delta)/dt, d(omega)/dt] for the SMIB swing equation."""
    delta, omega = np.asarray(state, dtype=float)
    pmax = scenario.pmax_at(t) if fixed_pmax is None else fixed_pmax
    pe = float(electrical_power(delta, pmax))

    d_delta = scenario.omega_base * omega
    d_omega = (scenario.Pm - pe - scenario.D * omega) / (2.0 * scenario.H)
    return np.asarray([d_delta, d_omega], dtype=float)
