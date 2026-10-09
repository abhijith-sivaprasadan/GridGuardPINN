import numpy as np

from gridguardpinn.trust import TrustGate, TrustSignals, evaluate_gate


def test_gate_reports_rejection_reasons():
    gate = TrustGate(residual_threshold=0.1, ood_threshold=3.0)
    decision = gate.decide(TrustSignals(residual_rms=0.2, ood_score=1.0))
    assert not decision.accepted
    assert decision.reasons == ("physics_residual",)


def test_gate_metrics_count_false_accepts_and_false_escalations():
    accepted = np.asarray([True, True, False, False])
    errors = np.asarray([0.01, 0.20, 0.01, 0.20])
    metrics = evaluate_gate(accepted, errors, error_tolerance=0.05)

    assert metrics["false_accepts"] == 1
    assert metrics["false_escalations"] == 1
    assert metrics["accepted"] == 2
    assert metrics["coverage"] == 0.5
    assert metrics["false_accept_rate_accepted"] == 0.5
