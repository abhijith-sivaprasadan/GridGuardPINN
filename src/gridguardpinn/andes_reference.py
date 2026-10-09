"""ANDES transient-stability reference adapter.

This module intentionally contains no surrogate logic. It establishes a
reproducible, simulator-backed reference trajectory that later learned models
can be tested against.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AndesTrajectory:
    """Time-domain trajectory extracted from an ANDES simulation."""

    time_s: np.ndarray
    generator_speed_pu: np.ndarray
    bus_voltage_pu: np.ndarray
    fault_bus: int
    fault_start_s: float
    fault_clear_s: float
    andes_version: str

    def __post_init__(self) -> None:
        n = self.time_s.size
        if self.generator_speed_pu.ndim != 2:
            raise ValueError("generator_speed_pu must be two-dimensional.")
        if self.bus_voltage_pu.ndim != 2:
            raise ValueError("bus_voltage_pu must be two-dimensional.")
        if self.generator_speed_pu.shape[0] != n:
            raise ValueError("Generator-speed rows must align with time.")
        if self.bus_voltage_pu.shape[0] != n:
            raise ValueError("Bus-voltage rows must align with time.")

    @property
    def fault_duration_s(self) -> float:
        return self.fault_clear_s - self.fault_start_s

    def metrics(self) -> dict[str, float | int | str]:
        speed_deviation = np.abs(self.generator_speed_pu - 1.0)
        return {
            "andes_version": self.andes_version,
            "fault_bus": self.fault_bus,
            "fault_start_s": self.fault_start_s,
            "fault_clear_s": self.fault_clear_s,
            "fault_duration_s": self.fault_duration_s,
            "n_time_points": int(self.time_s.size),
            "n_generators": int(self.generator_speed_pu.shape[1]),
            "n_buses": int(self.bus_voltage_pu.shape[1]),
            "max_abs_speed_deviation_pu": float(np.max(speed_deviation)),
            "min_bus_voltage_pu": float(np.min(self.bus_voltage_pu)),
            "max_bus_voltage_pu": float(np.max(self.bus_voltage_pu)),
        }


def _import_andes():
    try:
        import andes
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            'ANDES integration requires Python >=3.11 and: pip install -e ".[andes]"'
        ) from exc
    return andes


def run_kundur_fault(
    *,
    fault_bus: int = 5,
    fault_start_s: float = 1.0,
    fault_clear_s: float = 1.1,
    simulation_end_s: float = 5.0,
    fault_resistance_pu: float = 0.0,
    fault_reactance_pu: float = 1e-6,
) -> AndesTrajectory:
    """Run the packaged Kundur case with one added three-phase bus fault.

    ANDES's packaged Kundur case includes a timed Toggle. It is disabled before
    the custom fault is added so the returned trajectory isolates the requested
    fault event.
    """
    if not 0.0 < fault_start_s < fault_clear_s < simulation_end_s:
        raise ValueError(
            "Require 0 < fault_start_s < fault_clear_s < simulation_end_s."
        )

    andes = _import_andes()
    case_path = andes.get_case("kundur/kundur_full.xlsx")
    system = andes.load(case_path, setup=False)

    if system.Toggle.n:
        first_toggle_idx = system.Toggle.idx.v[0]
        system.Toggle.set("u", first_toggle_idx, 0)

    system.add(
        "Fault",
        bus=fault_bus,
        tf=fault_start_s,
        tc=fault_clear_s,
        rf=fault_resistance_pu,
        xf=fault_reactance_pu,
    )
    system.setup()

    system.PFlow.run()
    if system.exit_code != 0:
        raise RuntimeError(f"ANDES power flow failed with exit_code={system.exit_code}.")

    system.TDS.config.tf = simulation_end_s
    system.TDS.config.no_tqdm = 1
    system.TDS.run()
    if system.exit_code != 0:
        raise RuntimeError(f"ANDES TDS failed with exit_code={system.exit_code}.")

    time_s = np.asarray(system.dae.ts.t, dtype=float).copy()
    generator_speed = np.asarray(
        system.dae.ts.x[:, system.GENROU.omega.a],
        dtype=float,
    ).copy()
    bus_voltage = np.asarray(
        system.dae.ts.y[:, system.Bus.v.a],
        dtype=float,
    ).copy()

    if not (
        np.all(np.isfinite(time_s))
        and np.all(np.isfinite(generator_speed))
        and np.all(np.isfinite(bus_voltage))
    ):
        raise RuntimeError("ANDES returned non-finite trajectory values.")

    return AndesTrajectory(
        time_s=time_s,
        generator_speed_pu=generator_speed,
        bus_voltage_pu=bus_voltage,
        fault_bus=fault_bus,
        fault_start_s=fault_start_s,
        fault_clear_s=fault_clear_s,
        andes_version=str(andes.__version__),
    )


def resample_trajectory(
    trajectory: AndesTrajectory,
    *,
    time_grid_s: np.ndarray,
) -> AndesTrajectory:
    """Linearly resample a simulator trajectory onto a deterministic time grid."""
    grid = np.asarray(time_grid_s, dtype=float).reshape(-1)
    if grid.size < 2:
        raise ValueError("time_grid_s must contain at least two points.")
    if np.any(np.diff(grid) <= 0):
        raise ValueError("time_grid_s must be strictly increasing.")
    if grid[0] < trajectory.time_s[0] or grid[-1] > trajectory.time_s[-1]:
        raise ValueError("Requested time grid lies outside the source trajectory.")

    speed = np.column_stack(
        [
            np.interp(grid, trajectory.time_s, trajectory.generator_speed_pu[:, i])
            for i in range(trajectory.generator_speed_pu.shape[1])
        ]
    )
    voltage = np.column_stack(
        [
            np.interp(grid, trajectory.time_s, trajectory.bus_voltage_pu[:, i])
            for i in range(trajectory.bus_voltage_pu.shape[1])
        ]
    )

    return AndesTrajectory(
        time_s=grid,
        generator_speed_pu=speed,
        bus_voltage_pu=voltage,
        fault_bus=trajectory.fault_bus,
        fault_start_s=trajectory.fault_start_s,
        fault_clear_s=trajectory.fault_clear_s,
        andes_version=trajectory.andes_version,
    )
