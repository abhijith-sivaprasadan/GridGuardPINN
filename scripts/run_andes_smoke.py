"""Run and export the first ANDES-backed Kundur transient reference case."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from gridguardpinn.andes_reference import resample_trajectory, run_kundur_fault


def main() -> None:
    output = Path("artifacts/andes_kundur_smoke")
    output.mkdir(parents=True, exist_ok=True)

    trajectory = run_kundur_fault(
        fault_bus=5,
        fault_start_s=1.0,
        fault_clear_s=1.1,
        simulation_end_s=5.0,
    )
    uniform = resample_trajectory(
        trajectory,
        time_grid_s=np.linspace(0.0, 5.0, 501),
    )

    metrics = {
        **trajectory.metrics(),
        "resampled_time_points": int(uniform.time_s.size),
        "resampled_max_abs_speed_deviation_pu": float(
            np.max(np.abs(uniform.generator_speed_pu - 1.0))
        ),
        "resampled_min_bus_voltage_pu": float(np.min(uniform.bus_voltage_pu)),
    }
    (output / "summary.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    header = ["time_s"]
    header += [f"omega_gen_{i}" for i in range(uniform.generator_speed_pu.shape[1])]
    header += [f"v_bus_{i}" for i in range(uniform.bus_voltage_pu.shape[1])]

    with (output / "trajectory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for i, t in enumerate(uniform.time_s):
            writer.writerow(
                [float(t)]
                + uniform.generator_speed_pu[i].tolist()
                + uniform.bus_voltage_pu[i].tolist()
            )

    print("ANDES_REFERENCE_SUMMARY_JSON=" + json.dumps(metrics))


if __name__ == "__main__":
    main()
