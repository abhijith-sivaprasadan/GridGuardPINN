import numpy as np

from gridguardpinn.andes_gate import calibrate_gate, gate_reports


def _row(error, residual, *, location_ood=False, duration_ood=False):
    return {
        "composite_error_ratio": error,
        "residual_score": residual,
        "location_ood": location_ood,
        "duration_ood": duration_ood,
    }


def test_validation_calibration_maximizes_zero_false_accept_coverage():
    validation = [
        _row(0.5, 0.1),
        _row(0.8, 0.2),
        _row(1.4, 0.3),
    ]
    gate = calibrate_gate(validation)
    assert gate.residual_threshold == 0.2


def test_strict_gate_rejects_explicit_ood():
    rows = [
        _row(0.5, 0.1),
        _row(0.5, 0.1, duration_ood=True),
        _row(0.5, 0.1, location_ood=True),
    ]
    gate = calibrate_gate([_row(0.5, 0.1)])
    reports = gate_reports(rows, gate)
    assert reports["combined_strict"]["accepted"] == 1
    assert reports["combined_location_hard"]["accepted"] == 2
    assert np.isclose(reports["combined_strict"]["coverage"], 1 / 3)
