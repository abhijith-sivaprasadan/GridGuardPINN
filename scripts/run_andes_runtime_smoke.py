"""Exercise the real ANDES reference and the research runtime together.

The model here is deliberately untrained: this validates plumbing, not accuracy.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from gridguardpinn.andes_reference import run_ieee14_fault
from gridguardpinn.andes_runtime import (
    andes_reference_fault,
    compute_model_residual,
    run_model_case,
    run_routed_case,
)
from gridguardpinn.andes_surrogate import AndesMultiMachineSurrogate


def main() -> None:
    native = run_ieee14_fault(
        fault_bus=3,
        fault_start_s=1.0,
        fault_clear_s=1.08,
        simulation_end_s=2.0,
        fault_reactance_pu=1e-4,
    )
    assert native.generator_angle_rad.shape[1] == 5
    model = AndesMultiMachineSurrogate(
        np.zeros(20, dtype=np.float32),
        np.ones(20, dtype=np.float32),
    )
    output = Path("artifacts/andes_runtime_smoke")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "smoke_model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "training": {"config": {"hidden_width": 96, "hidden_layers": 5}},
            "gate": {"residual_threshold": 0.0},
            "summary": {"protocol": "runtime-smoke", "experiment_metadata": {"location_encoding": "one_hot"}},
        },
        checkpoint_path,
    )
    from gridguardpinn.andes_runtime import load_checkpoint, sha256_file

    restored, restored_threshold, _ = load_checkpoint(
        checkpoint_path,
        expected_sha256=sha256_file(checkpoint_path),
        expected_protocol="runtime-smoke",
    )
    assert restored_threshold == 0.0
    assert all(torch.equal(a, b) for a, b in zip(
        model.state_dict().values(), restored.state_dict().values(), strict=True
    ))
    score = compute_model_residual(
        model,
        fault_bus=3,
        fault_duration_s=0.08,
        inertia_M=native.machine_inertia_M,
        damping_D=native.machine_damping_D,
        frequency_hz=native.machine_frequency_hz,
    )
    assert np.isfinite(score)

    # End-to-end call using computed physics signal, with forced fallback
    # from a zero threshold. This is not evidence of learned accuracy.
    routed = run_model_case(
        model=model,
        fault_bus=3,
        fault_duration_s=0.08,
        residual_threshold=0.0,
        inertia_M=native.machine_inertia_M,
        damping_D=native.machine_damping_D,
        frequency_hz=native.machine_frequency_hz,
        reference=andes_reference_fault,
    )
    assert routed.source == "reference"
    assert routed.electromechanical.shape == (401, 20)

    # Unseen location refuses without invoking the surrogate.
    unseen = run_routed_case(
        fault_bus=1,
        fault_duration_s=0.08,
        residual_score=0.0,
        residual_threshold=1.0,
        predict=lambda *_: (_ for _ in ()).throw(AssertionError("unexpected")),
        reference=andes_reference_fault,
    )
    assert unseen.source == "reference"
    assert unseen.decision.reason == "unseen_fault_bus"

    result = {
        "test": "real_andes_runtime_smoke",
        "trained_model": False,
        "physics_score_finite": True,
        "checkpoint_roundtrip_verified": True,
        "trained_location_fallback": routed.source,
        "unseen_location_fallback": unseen.source,
        "unseen_refusal_reason": unseen.decision.reason,
        "andes_version": native.andes_version,
        "result": "passed",
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
