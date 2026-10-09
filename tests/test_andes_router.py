import math

import pytest

from gridguardpinn.andes_router import Route, route_andes_case


def decide(**overrides):
    args = dict(
        fault_bus=3,
        fault_duration_s=0.08,
        residual_score=0.1,
        residual_threshold=0.2,
    )
    args.update(overrides)
    return route_andes_case(**args)


def test_eligible_case_passes():
    decision = decide()
    assert decision.route == Route.ACCEPT_SURROGATE
    assert decision.use_surrogate


@pytest.mark.parametrize("bus", [1, 2, 4, 5, 10, 12, 13, 14.0, None])
def test_unseen_or_invalid_bus_rejected(bus):
    assert not decide(fault_bus=bus).use_surrogate


@pytest.mark.parametrize("duration", [0.039, 0.121, float("nan"), math.inf])
def test_duration_outside_envelope_rejected(duration):
    assert not decide(fault_duration_s=duration).use_surrogate


@pytest.mark.parametrize("score", [0.201, float("nan"), math.inf, -0.01])
def test_residual_failure_rejected(score):
    assert not decide(residual_score=score).use_surrogate


def test_threshold_and_endpoints():
    assert decide(residual_score=0.2).use_surrogate
    assert decide(fault_duration_s=0.04).use_surrogate
    assert decide(fault_duration_s=0.12).use_surrogate
    assert not decide(residual_threshold=float("nan")).use_surrogate


def test_unready_or_missing_reference_never_enables_surrogate():
    assert decide(model_ready=False).route == Route.RUN_REFERENCE
    decision = decide(reference_available=False)
    assert not decision.use_surrogate
    assert decision.reason == "reference_unavailable_stop"
