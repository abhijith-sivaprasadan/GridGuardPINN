"""Run and export ANDES IEEE-14 transient reference cases."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from gridguardpinn.andes_reference import (
    resample_trajectory,
    run_ieee14_fault,
    run_ieee14_packaged_fault,
)


def _export_trajectory(path: Path, trajectory) -> None:
    header = ["time_s"]
    header += [f"omega_gen_{i}" for i in range(trajectory.generator_speed_pu.shape[1])]
    header += [f"v_bus_{i}" for i in range(trajectory.bus_voltage_pu.shape[1])]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for i, t in enumerate(trajectory.time_s):
            writer.writerow(
                [float(t)]
                + trajectory.generator_speed_pu[i].tolist()
                + trajectory.bus_voltage_pu[i].tolist()
            )


def main() -> None:
    output = Path("artifacts/andes_ieee14_smoke")
    output.mkdir(parents=True, exist_ok=True)
    grid = np.linspace(0.0, 2.0, 401)

    packaged_native = run_ieee14_packaged_fault(simulation_end_s=2.0)
    packaged = resample_trajectory(packaged_native, time_grid_s=grid)

    custom_native = run_ieee14_fault(
        fault_bus=9,
        fault_start_s=1.0,
        fault_clear_s=1.1,
        simulation_end_s=2.0,
        fault_reactance_pu=1e-4,
    )
    custom = resample_trajectory(custom_native, time_grid_s=grid)

    if packaged.generator_speed_pu.shape != custom.generator_speed_pu.shape:
        raise RuntimeError("Packaged and custom cases expose different generator shapes.")
    if packaged.bus_voltage_pu.shape != custom.bus_voltage_pu.shape:
        raise RuntimeError("Packaged and custom cases expose different bus-voltage shapes.")

    comparison = {
        "max_abs_generator_speed_difference_pu": float(
            np.max(np.abs(packaged.generator_speed_pu - custom.generator_speed_pu))
        ),
        "max_abs_bus_voltage_difference_pu": float(
            np.max(np.abs(packaged.bus_voltage_pu - custom.bus_voltage_pu))
        ),
    }
    summary = {
        "packaged": {
            **packaged_native.metrics(),
            "resampled_time_points": int(packaged.time_s.size),
        },
        "programmatic": {
            **custom_native.metrics(),
            "resampled_time_points": int(custom.time_s.size),
        },
        "comparison": comparison,
    }

    (output / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    _export_trajectory(output / "packaged_trajectory.csv", packaged)
    _export_trajectory(output / "programmatic_trajectory.csv", custom)

    print("ANDES_REFERENCE_SUMMARY_JSON=" + json.dumps(summary))


if __name__ == "__main__":
    main()
