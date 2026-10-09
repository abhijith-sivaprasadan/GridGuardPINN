import numpy as np
import pytest

from gridguardpinn.andes_reference import AndesTrajectory, resample_trajectory


def _trajectory():
    t = np.asarray([0.0, 0.5, 1.0])
    omega = np.column_stack([1.0 + 0.01 * t, 1.0 - 0.02 * t])
    angle = np.column_stack([0.2 + 0.03 * t, -0.1 + 0.04 * t])
    tm = np.column_stack([0.8 + 0.0 * t, 0.7 + 0.0 * t])
    te = np.column_stack([0.8 - 0.01 * t, 0.7 + 0.02 * t])
    voltage = np.column_stack([1.0 - 0.1 * t, 1.0 - 0.05 * t])
    return AndesTrajectory(
        time_s=t,
        generator_speed_pu=omega,
        generator_angle_rad=angle,
        generator_mechanical_torque_pu=tm,
        generator_electrical_torque_pu=te,
        machine_inertia_M=np.asarray([5.0, 6.0]),
        machine_damping_D=np.asarray([1.0, 1.5]),
        machine_frequency_hz=np.asarray([60.0, 60.0]),
        bus_voltage_pu=voltage,
        fault_bus=5,
        fault_start_s=0.2,
        fault_clear_s=0.3,
        andes_version="test",
        case_name="synthetic-test",
    )


def test_andes_metrics_and_resampling():
    trajectory = _trajectory()
    metrics = trajectory.metrics()
    assert metrics["n_generators"] == 2
    assert metrics["n_buses"] == 2
    assert metrics["fault_duration_s"] == pytest.approx(0.1)
    assert metrics["max_abs_angle_excursion_rad"] == pytest.approx(0.04)

    grid = np.linspace(0.0, 1.0, 5)
    resampled = resample_trajectory(trajectory, time_grid_s=grid)
    assert resampled.generator_speed_pu.shape == (5, 2)
    assert resampled.generator_angle_rad is not None
    assert resampled.generator_mechanical_torque_pu is not None
    assert resampled.generator_electrical_torque_pu is not None
    assert resampled.generator_angle_rad.shape == (5, 2)
    assert resampled.bus_voltage_pu.shape == (5, 2)
    assert resampled.generator_speed_pu[-1, 0] == pytest.approx(1.01)
    assert np.array_equal(resampled.machine_inertia_M, np.asarray([5.0, 6.0]))


def test_resampling_rejects_out_of_bounds_grid():
    trajectory = _trajectory()
    with pytest.raises(ValueError):
        resample_trajectory(
            trajectory,
            time_grid_s=np.asarray([-0.1, 0.5, 1.0]),
        )
