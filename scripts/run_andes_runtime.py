"""Research-only inference CLI with pinned checkpoint and ANDES fallback."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from gridguardpinn.andes_runtime import load_checkpoint, run_model_case, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sha256", required=True, help="Independently pinned checkpoint SHA-256")
    parser.add_argument("--protocol", required=True, help="Expected frozen training protocol")
    parser.add_argument("--bus", required=True, type=int)
    parser.add_argument("--duration", required=True, type=float)
    parser.add_argument("--output-dir", default="artifacts/andes_runtime_cli")
    args = parser.parse_args()

    from gridguardpinn.andes_reference import _import_andes

    andes = _import_andes()
    case_file = andes.get_case("ieee14/ieee14.json")
    system = andes.load(case_file, setup=False)
    # Never permit caller-supplied machine constants or a different machine order.
    generator_buses = tuple(int(bus) for bus in system.GENROU.bus.v)
    if len(generator_buses) != 5:
        raise RuntimeError("Expected exactly five ordered GENROU machines")
    constants = {
        "inertia_M": np.asarray(system.GENROU.M.v, dtype=float),
        "damping_D": np.asarray(system.GENROU.D.v, dtype=float),
        "frequency_hz": np.asarray(system.GENROU.fn.v, dtype=float),
    }
    if any(array.shape != (5,) or not np.all(np.isfinite(array)) for array in constants.values()):
        raise RuntimeError("Invalid machine constants from canonical ANDES case")
    case_sha256 = sha256_file(case_file)
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
        **constants,
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
        "andes_version": str(andes.__version__),
        "case_sha256": case_sha256,
        "case_name": "ieee14/ieee14.json",
        "ordered_generator_buses": generator_buses,
        "residual_threshold": threshold,
        "reference_includes_bus_voltage": result.reference is not None,
        "warning": "Research-only screening; not an operational safety determination",
    }
    (output / "routing.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
