import numpy as np
import pytest

from gridguardpinn.andes_reference import AndesTrajectory
from gridguardpinn.andes_runtime import run_routed_case


def fake_reference(bus, duration):
    t = np.linspace(0, 2, 401)
    zeros = np.zeros((t.size, 5))
    return AndesTrajectory(
        time_s=t,
        generator_speed_pu=zeros + 1,
        bus_voltage_pu=np.ones((t.size, 14)),
        generator_angle_rad=zeros,
        generator_mechanical_torque_pu=zeros,
        generator_electrical_torque_pu=zeros,
        fault_bus=bus,
        fault_start_s=1,
        fault_clear_s=1 + duration,
        andes_version="test",
        case_name="test",
    )


def options(**extra):
    values = {
        "fault_bus": 3,
        "fault_duration_s": 0.08,
        "residual_score": 0.1,
        "residual_threshold": 0.2,
        "predict": lambda grid, bus, duration: np.zeros((len(grid), 20)),
        "reference": fake_reference,
    }
    values.update(extra)
    return values


def test_accepted_surrogate_without_reference_execution():
    result = run_routed_case(**options(reference=lambda *args: pytest.fail("called")))
    assert result.source == "surrogate"
    assert result.electromechanical.shape == (401, 20)


def test_unseen_bus_executes_reference_not_surrogate():
    result = run_routed_case(
        **options(fault_bus=1, predict=lambda *args: pytest.fail("called"))
    )
    assert result.source == "reference"
    assert result.decision.reason == "unseen_fault_bus"
    assert result.reference.bus_voltage_pu.shape == (401, 14)


def test_outside_duration_executes_reference():
    assert run_routed_case(**options(fault_duration_s=0.14)).source == "reference"


def test_invalid_prediction_falls_back_to_reference():
    result = run_routed_case(
        **options(predict=lambda *args: np.full((401, 20), np.nan))
    )
    assert result.source == "reference"
    assert result.decision.reason == "surrogate_prediction_failed"


def test_reference_failure_propagates():
    def fail(*args):
        raise RuntimeError("ANDES failed")

    with pytest.raises(RuntimeError, match="ANDES failed"):
        run_routed_case(**options(fault_bus=1, reference=fail))


def test_invalid_time_grid_is_rejected():
    with pytest.raises(ValueError, match="time grid"):
        run_routed_case(**options(time_grid_s=np.array([0.0, 0.0, 1.0])))


def test_residual_failure_executes_reference_with_specific_reason(monkeypatch):
    import gridguardpinn.andes_runtime as runtime

    def failed_residual(*args, **kwargs):
        raise ValueError("Physics derivative unavailable")

    monkeypatch.setattr(runtime, "compute_model_residual", failed_residual)
    result = runtime.run_model_case(
        model=object(),
        fault_bus=3,
        fault_duration_s=0.08,
        residual_threshold=0.2,
        inertia_M=np.ones(5),
        damping_D=np.ones(5),
        frequency_hz=np.full(5, 50.0),
        reference=fake_reference,
    )
    assert result.source == "reference"
    assert result.decision.reason == "residual_computation_failed"
    assert result.electromechanical.shape == (401, 20)
