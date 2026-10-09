"""ANDES transient-stability reference adapters.

This module intentionally contains no surrogate logic. It establishes
reproducible simulator-backed reference trajectories that later learned models
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
    case_name: str
    generator_angle_rad: np.ndarray | None = None
    generator_mechanical_torque_pu: np.ndarray | None = None
    generator_electrical_torque_pu: np.ndarray | None = None
    machine_inertia_M: np.ndarray | None = None
    machine_damping_D: np.ndarray | None = None
    machine_frequency_hz: np.ndarray | None = None

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

        n_generators = self.generator_speed_pu.shape[1]
        trajectory_fields = (
            ("generator_angle_rad", self.generator_angle_rad),
            ("generator_mechanical_torque_pu", self.generator_mechanical_torque_pu),
            ("generator_electrical_torque_pu", self.generator_electrical_torque_pu),
        )
        for name, value in trajectory_fields:
            if value is None:
                continue
            if value.ndim != 2 or value.shape != self.generator_speed_pu.shape:
                raise ValueError(
                    f"{name} must have shape {self.generator_speed_pu.shape}."
                )

        parameter_fields = (
            ("machine_inertia_M", self.machine_inertia_M),
            ("machine_damping_D", self.machine_damping_D),
            ("machine_frequency_hz", self.machine_frequency_hz),
        )
        for name, value in parameter_fields:
            if value is None:
                continue
            if np.asarray(value).shape != (n_generators,):
                raise ValueError(f"{name} must have shape ({n_generators},).")

    @property
    def fault_duration_s(self) -> float:
        return self.fault_clear_s - self.fault_start_s

    def metrics(self) -> dict[str, float | int | str | None]:
        speed_deviation = np.abs(self.generator_speed_pu - 1.0)
        if self.generator_angle_rad is None:
            angle_excursion = None
        else:
            angle_excursion = float(
                np.max(
                    np.abs(
                        self.generator_angle_rad - self.generator_angle_rad[0:1, :]
                    )
                )
            )
        return {
            "andes_version": self.andes_version,
            "case_name": self.case_name,
            "fault_bus": self.fault_bus,
            "fault_start_s": self.fault_start_s,
            "fault_clear_s": self.fault_clear_s,
            "fault_duration_s": self.fault_duration_s,
            "n_time_points": int(self.time_s.size),
            "n_generators": int(self.generator_speed_pu.shape[1]),
            "n_buses": int(self.bus_voltage_pu.shape[1]),
            "max_abs_speed_deviation_pu": float(np.max(speed_deviation)),
            "max_abs_angle_excursion_rad": angle_excursion,
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


def _extract_trajectory(
    system,
    *,
    andes_version: str,
    case_name: str,
    fault_bus: int,
    fault_start_s: float,
    fault_clear_s: float,
) -> AndesTrajectory:
    time_s = np.asarray(system.dae.ts.t, dtype=float).copy()
    generator_speed = np.asarray(
        system.dae.ts.x[:, system.GENROU.omega.a], dtype=float
    ).copy()
    generator_angle = np.asarray(
        system.dae.ts.x[:, system.GENROU.delta.a], dtype=float
    ).copy()
    mechanical_torque = np.asarray(
        system.dae.ts.y[:, system.GENROU.tm.a], dtype=float
    ).copy()
    electrical_torque = np.asarray(
        system.dae.ts.y[:, system.GENROU.te.a], dtype=float
    ).copy()
    bus_voltage = np.asarray(
        system.dae.ts.y[:, system.Bus.v.a], dtype=float
    ).copy()

    arrays = (
        time_s,
        generator_speed,
        generator_angle,
        mechanical_torque,
        electrical_torque,
        bus_voltage,
    )
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise RuntimeError("ANDES returned non-finite trajectory values.")

    return AndesTrajectory(
        time_s=time_s,
        generator_speed_pu=generator_speed,
        generator_angle_rad=generator_angle,
        generator_mechanical_torque_pu=mechanical_torque,
        generator_electrical_torque_pu=electrical_torque,
        machine_inertia_M=np.asarray(system.GENROU.M.v, dtype=float).copy(),
        machine_damping_D=np.asarray(system.GENROU.D.v, dtype=float).copy(),
        machine_frequency_hz=np.asarray(system.GENROU.fn.v, dtype=float).copy(),
        bus_voltage_pu=bus_voltage,
        fault_bus=fault_bus,
        fault_start_s=fault_start_s,
        fault_clear_s=fault_clear_s,
        andes_version=andes_version,
        case_name=case_name,
    )


def _run_tds(system, *, simulation_end_s: float) -> None:
    system.PFlow.run()
    if system.exit_code != 0:
        raise RuntimeError(f"ANDES power flow failed with exit_code={system.exit_code}.")
    system.TDS.config.tf = simulation_end_s
    system.TDS.config.no_tqdm = 1
    system.TDS.run()
    if system.exit_code != 0:
        raise RuntimeError(f"ANDES TDS failed with exit_code={system.exit_code}.")


def run_ieee14_packaged_fault(*, simulation_end_s: float = 2.0) -> AndesTrajectory:
    """Run ANDES's maintained IEEE-14 three-phase-fault test case."""
    if simulation_end_s <= 1.1:
        raise ValueError("simulation_end_s must extend beyond fault clearing.")

    andes = _import_andes()
    system = andes.load(andes.get_case("ieee14/ieee14_fault.xlsx"))
    _run_tds(system, simulation_end_s=simulation_end_s)
    return _extract_trajectory(
        system,
        andes_version=str(andes.__version__),
        case_name="ieee14/ieee14_fault.xlsx",
        fault_bus=9,
        fault_start_s=1.0,
        fault_clear_s=1.1,
    )


def run_ieee14_fault(
    *,
    fault_bus: int,
    fault_start_s: float = 1.0,
    fault_clear_s: float = 1.1,
    simulation_end_s: float = 2.0,
    fault_resistance_pu: float = 0.0,
    fault_reactance_pu: float = 1e-4,
) -> AndesTrajectory:
    """Run a programmatically specified three-phase fault on dynamic IEEE-14."""
    if not 0.0 < fault_start_s < fault_clear_s < simulation_end_s:
        raise ValueError(
            "Require 0 < fault_start_s < fault_clear_s < simulation_end_s."
        )

    andes = _import_andes()
    system = andes.load(andes.get_case("ieee14/ieee14.json"), setup=False)
    system.add(
        "Fault",
        bus=fault_bus,
        tf=fault_start_s,
        tc=fault_clear_s,
        rf=fault_resistance_pu,
        xf=fault_reactance_pu,
    )
    system.setup()
    _run_tds(system, simulation_end_s=simulation_end_s)
    return _extract_trajectory(
        system,
        andes_version=str(andes.__version__),
        case_name="ieee14/ieee14.json+programmatic_fault",
        fault_bus=fault_bus,
        fault_start_s=fault_start_s,
        fault_clear_s=fault_clear_s,
    )


def run_kundur_fault(
    *,
    fault_bus: int = 5,
    fault_start_s: float = 1.0,
    fault_clear_s: float = 1.1,
    simulation_end_s: float = 5.0,
    fault_resistance_pu: float = 0.0,
    fault_reactance_pu: float = 1e-6,
) -> AndesTrajectory:
    """Run the packaged Kundur case with one added three-phase bus fault."""
    if not 0.0 < fault_start_s < fault_clear_s < simulation_end_s:
        raise ValueError(
            "Require 0 < fault_start_s < fault_clear_s < simulation_end_s."
        )

    andes = _import_andes()
    system = andes.load(andes.get_case("kundur/kundur_full.xlsx"), setup=False)
    if system.Toggle.n:
        system.Toggle.set("u", system.Toggle.idx.v[0], 0)
    system.add(
        "Fault",
        bus=fault_bus,
        tf=fault_start_s,
        tc=fault_clear_s,
        rf=fault_resistance_pu,
        xf=fault_reactance_pu,
    )
    system.setup()
    _run_tds(system, simulation_end_s=simulation_end_s)
    return _extract_trajectory(
        system,
        andes_version=str(andes.__version__),
        case_name="kundur/kundur_full.xlsx",
        fault_bus=fault_bus,
        fault_start_s=fault_start_s,
        fault_clear_s=fault_clear_s,
    )


def _resample_optional(
    values: np.ndarray | None,
    source_time: np.ndarray,
    target_time: np.ndarray,
) -> np.ndarray | None:
    if values is None:
        return None
    return np.column_stack(
        [
            np.interp(target_time, source_time, values[:, i])
            for i in range(values.shape[1])
        ]
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

    return AndesTrajectory(
        time_s=grid,
        generator_speed_pu=_resample_optional(
            trajectory.generator_speed_pu, trajectory.time_s, grid
        ),
        generator_angle_rad=_resample_optional(
            trajectory.generator_angle_rad, trajectory.time_s, grid
        ),
        generator_mechanical_torque_pu=_resample_optional(
            trajectory.generator_mechanical_torque_pu, trajectory.time_s, grid
        ),
        generator_electrical_torque_pu=_resample_optional(
            trajectory.generator_electrical_torque_pu, trajectory.time_s, grid
        ),
        machine_inertia_M=trajectory.machine_inertia_M,
        machine_damping_D=trajectory.machine_damping_D,
        machine_frequency_hz=trajectory.machine_frequency_hz,
        bus_voltage_pu=_resample_optional(
            trajectory.bus_voltage_pu, trajectory.time_s, grid
        ),
        fault_bus=trajectory.fault_bus,
        fault_start_s=trajectory.fault_start_s,
        fault_clear_s=trajectory.fault_clear_s,
        andes_version=trajectory.andes_version,
        case_name=trajectory.case_name,
    )
