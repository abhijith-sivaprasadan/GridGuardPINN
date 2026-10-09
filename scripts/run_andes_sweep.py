"""Run the frozen ANDES IEEE-14 fault-location/duration sweep."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from gridguardpinn.andes_reference import (
    resample_trajectory,
    run_ieee14_fault,
)


FAULT_BUSES = (2, 4, 5, 9, 12, 14)
FAULT_DURATIONS_S = (0.06, 0.10, 0.14)
FAULT_START_S = 1.0
SIMULATION_END_S = 2.0
GRID = np.linspace(0.0, SIMULATION_END_S, 401)


def main() -> None:
    output = Path("artifacts/andes_ieee14_sweep_v0_1")
    output.mkdir(parents=True, exist_ok=True)

    case_rows: list[dict[str, object]] = []
    trajectory_rows: list[list[object]] = []
    case_id = 0

    for bus in FAULT_BUSES:
        for duration in FAULT_DURATIONS_S:
            case_id += 1
            clear_time = FAULT_START_S + duration
            base = {
                "case_id": case_id,
                "fault_bus": bus,
                "fault_duration_s": duration,
                "fault_start_s": FAULT_START_S,
                "fault_clear_s": clear_time,
            }

            try:
                native = run_ieee14_fault(
                    fault_bus=bus,
                    fault_start_s=FAULT_START_S,
                    fault_clear_s=clear_time,
                    simulation_end_s=SIMULATION_END_S,
                    fault_reactance_pu=1e-4,
                )
                trajectory = resample_trajectory(native, time_grid_s=GRID)
                metrics = native.metrics()
                case_rows.append(
                    {
                        **base,
                        "status": "success",
                        "error_type": "",
                        "error_message": "",
                        "native_time_points": metrics["n_time_points"],
                        "max_abs_speed_deviation_pu": metrics[
                            "max_abs_speed_deviation_pu"
                        ],
                        "min_bus_voltage_pu": metrics["min_bus_voltage_pu"],
                        "max_bus_voltage_pu": metrics["max_bus_voltage_pu"],
                    }
                )

                for i, t in enumerate(trajectory.time_s):
                    trajectory_rows.append(
                        [case_id, bus, duration, float(t)]
                        + trajectory.generator_speed_pu[i].tolist()
                        + trajectory.bus_voltage_pu[i].tolist()
                    )
            except RuntimeError as exc:  # simulator failures are recorded outcomes
                case_rows.append(
                    {
                        **base,
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "native_time_points": "",
                        "max_abs_speed_deviation_pu": "",
                        "min_bus_voltage_pu": "",
                        "max_bus_voltage_pu": "",
                    }
                )

    case_fields = list(case_rows[0].keys())
    with (output / "case_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=case_fields)
        writer.writeheader()
        writer.writerows(case_rows)

    trajectory_header = [
        "case_id",
        "fault_bus",
        "fault_duration_s",
        "time_s",
    ]
    trajectory_header += [f"omega_gen_{i}" for i in range(5)]
    trajectory_header += [f"v_bus_{i}" for i in range(14)]
    with (output / "trajectories.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(trajectory_header)
        writer.writerows(trajectory_rows)

    successful = [row for row in case_rows if row["status"] == "success"]
    failed = [row for row in case_rows if row["status"] == "failed"]
    speed_values = [
        float(row["max_abs_speed_deviation_pu"])
        for row in successful
    ]
    voltage_values = [float(row["min_bus_voltage_pu"]) for row in successful]

    summary = {
        "protocol": "andes-ieee14-sweep-v0.1",
        "planned_cases": len(case_rows),
        "successful_cases": len(successful),
        "failed_cases": len(failed),
        "fault_buses": list(FAULT_BUSES),
        "fault_durations_s": list(FAULT_DURATIONS_S),
        "successful_speed_deviation_range_pu": (
            [min(speed_values), max(speed_values)] if speed_values else None
        ),
        "successful_min_bus_voltage_range_pu": (
            [min(voltage_values), max(voltage_values)] if voltage_values else None
        ),
        "failures": [
            {
                "case_id": row["case_id"],
                "fault_bus": row["fault_bus"],
                "fault_duration_s": row["fault_duration_s"],
                "error_type": row["error_type"],
                "error_message": row["error_message"],
            }
            for row in failed
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    if not successful:
        raise RuntimeError("All ANDES sweep cases failed; no reference dataset produced.")

    print("ANDES_SWEEP_SUMMARY_JSON=" + json.dumps(summary))


if __name__ == "__main__":
    main()
