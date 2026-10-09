import numpy as np

from gridguardpinn.calibration import apply_gate, calibrate_gate, gate_report


def _row(residual: float, ood: float, error: float) -> dict[str, float]:
    return {
        "residual_score": residual,
        "ood_score": ood,
        "composite_error_ratio": error,
    }


def test_calibration_maximises_safe_validation_coverage():
    validation = [
        _row(0.1, 0.5, 0.2),
        _row(0.2, 0.6, 0.8),
        _row(0.3, 0.7, 1.5),
        _row(0.4, 0.8, 2.0),
    ]
    gate = calibrate_gate(
        validation,
        in_distribution_ood_scores=np.asarray([0.2, 0.5, 0.6, 0.7, 0.8]),
    )
    accepted = apply_gate(validation, gate)
    assert accepted.tolist() == [True, True, False, False]

    report = gate_report(validation, gate)
    assert report["false_accepts"] == 0
    assert report["coverage"] == 0.5
