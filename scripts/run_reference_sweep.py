"""Run the frozen v0.1 SMIB reference scenario splits."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from gridguardpinn.reference import simulate_reference
from gridguardpinn.scenarios import canonical_splits


def main() -> None:
    output_dir = Path("artifacts")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "reference_sweep.csv"

    rows: list[dict[str, float | str]] = []
    for split_name, scenarios in canonical_splits().items():
        for case_id, scenario in enumerate(scenarios):
            result = simulate_reference(scenario)
            rows.append(
                {
                    "split": split_name,
                    "case_id": case_id,
                    "H": scenario.H,
                    "D": scenario.D,
                    "t_clear": scenario.t_clear,
                    "fault_ratio": scenario.Pmax_fault / scenario.Pmax_pre,
                    "max_abs_omega": float(np.max(np.abs(result.omega))),
                    "max_delta": float(np.max(result.delta)),
                    "final_delta": float(result.delta[-1]),
                    "final_omega": float(result.omega[-1]),
                }
            )

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} cases to {output_path}")


if __name__ == "__main__":
    main()
