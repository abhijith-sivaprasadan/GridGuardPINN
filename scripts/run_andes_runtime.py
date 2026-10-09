"""Research-only inference CLI with pinned checkpoint and ANDES fallback."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from gridguardpinn.andes_runtime import load_checkpoint, run_model_case


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sha256", required=True, help="Independently pinned checkpoint SHA-256")
    parser.add_argument("--protocol", required=True, help="Expected frozen training protocol")
    parser.add_argument("--bus", required=True, type=int)
    parser.add_argument("--duration", required=True, type=float)
    parser.add_argument("--inertia", nargs=5, required=True, type=float)
    parser.add_argument("--damping", nargs=5, required=True, type=float)
    parser.add_argument("--frequency", nargs=5, required=True, type=float)
    parser.add_argument("--output-dir", default="artifacts/andes_runtime_cli")
    args = parser.parse_args()

    model, threshold, summary = load_checkpoint(
        args.checkpoint,
        expected_sha256=args.sha256,
        expected_protocol=args.protocol,
    )
    result = run_model_case(
        model=model,
        fault_bus=args.bus,
        fault_duration_s=args.duration,
        residual_threshold=threshold,
        inertia_M=np.asarray(args.inertia),
        damping_D=np.asarray(args.damping),
        frequency_hz=np.asarray(args.frequency),
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "electromechanical.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["time_s"] +
            [f"{quantity}_machine_{i}" for quantity in ("delta", "omega", "tm", "te")
             for i in range(1, 6)]
        )
        writer.writerows(
            np.column_stack([result.time_s, result.electromechanical])
        )
    info = {
        "source": result.source,
        "reason": result.decision.reason,
        "fault_bus": args.bus,
        "fault_duration_s": args.duration,
        "model_protocol": summary["protocol"],
        "checkpoint_sha256": args.sha256,
        "reference_includes_bus_voltage": result.reference is not None,
        "warning": "Research-only screening; not an operational safety determination",
    }
    (output / "routing.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
