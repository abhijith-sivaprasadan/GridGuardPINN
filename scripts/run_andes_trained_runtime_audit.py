"""Re-evaluate authenticated frozen trained checkpoints through actual routing.

These cases were previously inspected in v0.3. This is a reproducibility and
runtime-integration audit, NOT new blind generalisation evidence.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from gridguardpinn.andes_multimachine import generate_reference_batch, target_matrix
from gridguardpinn.andes_runtime import (
    andes_reference_fault,
    load_checkpoint,
    run_model_case,
)
from gridguardpinn.andes_surrogate_protocol import (
    assert_protocol_integrity,
    surrogate_splits_v01,
)

ARTIFACT_RUN = 37922112172
ARTIFACT_ID = 11612523064
ARTIFACT_ZIP_SHA256 = "02a82e4d53544203376749e7054778b9b93a2a5d6af3c83ab7d2820fd24f6666"
TRAINED_SHA256 = {
    17: "c2e39a1b0618336bb4342037d51e957a866a8312afe887cdb3546de61e0ebfd5",
    29: "4a136596171930052b59dda7adeeb76bc8b998a4d6474e8fec2a910399d38ffa",
    41: "e3f8fe19a44edebb1616b741f797cd7057300627ff292e4c3ae7afae27fa0367",
}
SPLITS = ("test_id", "ood_duration", "ood_location")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--output-dir", default="artifacts/andes_trained_runtime")
    args = parser.parse_args()
    assert_protocol_integrity()
    splits = surrogate_splits_v01()
    cases = list(dict.fromkeys(case for split in SPLITS for case in splits[split]))
    references = generate_reference_batch(cases, samples=401)
    first = next(iter(references.trajectories.values()))
    constants = {
        "inertia_M": first.machine_inertia_M,
        "damping_D": first.machine_damping_D,
        "frequency_hz": first.machine_frequency_hz,
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed, digest in TRAINED_SHA256.items():
        checkpoint = Path(args.artifact_dir) / "one_hot" / f"seed_{seed}" / "model.pt"
        model, threshold, summary = load_checkpoint(
            checkpoint,
            expected_sha256=digest,
            expected_protocol="andes-location-encoding-v0.3",
        )
        assert summary["experiment_metadata"]["location_encoding"] == "one_hot"
        for split in SPLITS:
            for case in splits[split]:
                result = run_model_case(
                    model=model,
                    fault_bus=case.fault_bus,
                    fault_duration_s=case.fault_duration_s,
                    residual_threshold=threshold,
                    reference=andes_reference_fault,
                    **constants,
                )
                truth = target_matrix(references.trajectories[case])
                if result.source == "reference":
                    assert result.reference is not None
                    # Independently rerun ANDES; any reproducibility differences
                    # remain visible, not falsely labelled zero error.
                diff = result.electromechanical - truth
                angle_rmse = float(np.sqrt(np.mean(diff[:, :5] ** 2, axis=0)).max())
                speed_rmse = float(np.sqrt(np.mean(diff[:, 5:10] ** 2, axis=0)).max())
                error = max(angle_rmse / 0.05, speed_rmse / 0.0005)
                if not np.isfinite(error):
                    raise RuntimeError("Non-finite routed error.")
                rows.append({
                    "seed": seed,
                    "split": split,
                    "fault_bus": case.fault_bus,
                    "duration_s": case.fault_duration_s,
                    "source": result.source,
                    "reason": result.decision.reason,
                    "max_angle_rmse_rad": angle_rmse,
                    "max_speed_rmse_pu": speed_rmse,
                    "composite_error_ratio": error,
                    "false_accept": int(result.source == "surrogate" and error > 1),
                    "reference_simulation_seconds": references.case_seconds[case],
                })
    csv_path = output / "case_ledger.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    reports = {}
    for seed in TRAINED_SHA256:
        reports[str(seed)] = {}
        for split in SPLITS:
            sample = [row for row in rows if row["seed"] == seed and row["split"] == split]
            accepted = [row for row in sample if row["source"] == "surrogate"]
            reports[str(seed)][split] = {
                "cases": len(sample),
                "surrogate_accepted": len(accepted),
                "reference_fallbacks": len(sample) - len(accepted),
                "false_accepts": sum(row["false_accept"] for row in sample),
                "mean_routed_composite_error": float(np.mean(
                    [row["composite_error_ratio"] for row in sample]
                )),
            }
    result = {
        "purpose": "trained checkpoint runtime reproducibility audit (previously inspected cases)",
        "source_run": ARTIFACT_RUN,
        "source_artifact": ARTIFACT_ID,
        "artifact_zip_sha256": ARTIFACT_ZIP_SHA256,
        "checkpoint_sha256": TRAINED_SHA256,
        "seeds": list(TRAINED_SHA256),
        "splits": list(SPLITS),
        "screen": {"rotor_angle_rmse_rad": 0.05, "speed_rmse_pu": 0.0005},
        "reports": reports,
        "note": "Not a blind holdout, operational guarantee, or evidence of unseen-bus coverage.",
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("TRAINED_RUNTIME_SUMMARY_JSON=" + json.dumps(result))


if __name__ == "__main__":
    main()
