"""Run one frozen ANDES IEEE-14 multi-machine surrogate experiment."""

from __future__ import annotations

import argparse
import json

from gridguardpinn.andes_experiment import (
    run_surrogate_experiment,
    write_experiment_artifacts,
)
from gridguardpinn.andes_multimachine import generate_reference_batch
from gridguardpinn.andes_surrogate_protocol import (
    assert_protocol_integrity,
    surrogate_splits_v01,
)
from gridguardpinn.andes_training import AndesTrainingConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--anchors", type=int, default=101)
    parser.add_argument("--collocation", type=int, default=56)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--physics-weight", type=float, default=0.02)
    parser.add_argument("--run-label", default="physics-informed")
    parser.add_argument(
        "--output-dir",
        default="artifacts/andes_surrogate_v0_1",
    )
    args = parser.parse_args()

    if args.physics_weight < 0:
        parser.error("--physics-weight must be non-negative.")

    assert_protocol_integrity()
    splits = surrogate_splits_v01()
    all_cases = list(
        dict.fromkeys(
            case
            for cases in splits.values()
            for case in cases
        )
    )
    reference_batch = generate_reference_batch(all_cases, samples=401)

    config = AndesTrainingConfig(
        epochs=args.epochs,
        anchors_per_case=args.anchors,
        collocation_per_case=args.collocation,
        physics_weight=args.physics_weight,
        seed=args.seed,
    )
    result = run_surrogate_experiment(
        reference_batch,
        splits,
        config=config,
        protocol="andes-surrogate-v0.1",
        run_label=args.run_label,
    )
    write_experiment_artifacts(
        result,
        splits,
        output_dir=args.output_dir,
    )
    print(
        "ANDES_SURROGATE_SUMMARY_JSON="
        + json.dumps(result.summary)
    )


if __name__ == "__main__":
    main()
