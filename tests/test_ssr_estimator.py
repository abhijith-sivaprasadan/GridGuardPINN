"""Independent analytic, source-table, conservation and integration checks."""

import numpy as np
import pytest

from gridguardpinn.ssr_estimator import FBM, shaft_modes, shaft_ringdown
from gridguardpinn.ssr_estimator import parameters as p
from gridguardpinn.ssr_estimator.estimation import compare_modes, matrix_pencil, welch_modes


def test_published_mechanical_frequencies():
    frequencies, _ = shaft_modes()
    assert frequencies[0] < 1e-5
    # Source frequencies are rounded to 0.01 Hz. This is an independent table
    # comparison, not a frozen copy of the implementation's eigenvalues.
    np.testing.assert_allclose(frequencies[1:5], p.PUBLISHED_HZ, atol=0.005, rtol=0)


@pytest.mark.parametrize("decay", [0.4, -0.12, 0.0])
def test_pencil_analytic_frequency_decay_amplitude(decay):
    time = np.arange(1440) / 240
    signal = (0.7 * np.exp(-decay * time) * np.cos(2 * np.pi * 18.3 * time + 0.31)
              + 0.21 * np.exp(-0.17 * time) * np.cos(2 * np.pi * 28.1 * time - 0.7)
              + 1.3)
    modes = matrix_pencil(signal, 240, poles=6)
    assert len(modes) == 2
    np.testing.assert_allclose([m.frequency_hz for m in modes], [18.3, 28.1], atol=1e-6)
    np.testing.assert_allclose([m.decay_per_s for m in modes], [decay, 0.17], atol=1e-6)
    np.testing.assert_allclose([m.amplitude for m in modes], [0.7, 0.21], atol=1e-6)


def test_pencil_with_seeded_measurement_noise():
    time = np.arange(1440) / 240
    signal = np.exp(-0.25 * time) * np.cos(2 * np.pi * 21.4 * time)
    signal += np.random.default_rng(42).normal(0, 0.003, len(time))
    modes = matrix_pencil(signal, 240, poles=4)
    match = min(modes, key=lambda mode: abs(mode.frequency_hz - 21.4))
    assert abs(match.frequency_hz - 21.4) < 0.02
    assert abs(match.decay_per_s - 0.25) < 0.02


def test_welch_stationary_amplitude_and_frequency():
    time = np.arange(2880) / 240
    modes, _ = welch_modes(0.7 * np.cos(2 * np.pi * 18.3 * time), 240)
    assert len(modes) == 1
    assert abs(modes[0].frequency_hz - 18.3) < 0.02
    assert abs(modes[0].amplitude - 0.7) < 0.01
    assert modes[0].decay_per_s is None  # window width is not physical damping


def test_shaft_reference_and_honest_weak_mode_miss():
    _, speed = shaft_ringdown()
    pencil, _ = compare_modes(matrix_pencil(speed[:, 4], 240), p.PUBLISHED_HZ,
                              p.NO_LOAD_DECREMENT)
    assert all(row["status"] == "resolved" for row in pencil)
    assert max(row["absolute_decay_error_per_s"] for row in pencil) < 1e-7
    spectral, _ = compare_modes(welch_modes(speed[:, 4], 240)[0], p.PUBLISHED_HZ)
    assert spectral[1]["status"] == "missed"
    assert spectral[1]["estimate"] is None


@pytest.mark.parametrize("method", [matrix_pencil, welch_modes])
def test_aliasing_and_invalid_inputs_rejected(method):
    with pytest.raises(ValueError, match="Nyquist"):
        method(np.ones(500), 60)
    with pytest.raises(ValueError, match="finite"):
        method(np.full(500, np.nan), 240)
    modes = method(np.ones(500), 240)
    assert (modes[0] if method is welch_modes else modes) == []


def test_circuit_equilibrium_matches_published_operating_point():
    model = FBM()
    assert max(abs(model.rhs(0, model.initial))) < 1e-10
    voltage = model.terminal_voltage(model.initial)
    assert abs(voltage - 1) < 1e-12
    current = complex(*model.initial[12:14]) * np.exp(1j * model.initial[4])
    power = voltage * np.conj(current)
    assert abs(power.real - 0.9) < 1e-12
    assert abs(power.real / abs(power) - 0.9) < 1e-12


def test_electrical_resonance_is_near_source_approximate_40_hz():
    model = FBM()
    # Source gives "approximately 40", not a precision reference. Two hertz
    # is a deliberately coarse sanity bound; this is not a tuned error claim.
    assert abs(model.frozen_electrical_resonance() - 40) < 2
    assert abs(model.frozen_electrical_resonance(fault=True) - 40) < 2


def test_electromechanical_energy_balance_away_from_equilibrium():
    model = FBM()
    y = model.initial + np.random.default_rng(91).normal(0, 0.002, 20)
    xline = p.TRANSFORMER_X + p.LINE_X + p.SYSTEM_X

    def energy(state):
        value = np.sum(p.H * (1 + state[6:12])**2)
        value += state[:6] @ p.stiffness_matrix() @ state[:6] / (2 * p.OMEGA_BASE)
        electrical = xline * np.sum(state[12:14]**2)
        electrical += np.sum(state[18:20]**2) / model.capacitor_x
        for axis, (leakages, mutual) in enumerate([(p.XD_LEAKAGES, p.XD_MUTUAL),
                                                  (p.XQ_LEAKAGES, p.XQ_MUTUAL)]):
            rotor_flux = state[14 + 2 * axis:16 + 2 * axis]
            rotor_i = np.linalg.solve(np.diag(leakages[1:]) + mutual,
                                      rotor_flux + mutual * state[12 + axis])
            currents = np.r_[-state[12 + axis], rotor_i]
            electrical += currents @ (np.diag(leakages) + mutual) @ currents
        return value + electrical / (2 * p.OMEGA_BASE)

    rotor_i = []
    for axis, mutual in enumerate([p.XD_MUTUAL, p.XQ_MUTUAL]):
        rotor_i.append(np.linalg.solve(model.rotor_matrices[axis],
                                       y[14 + 2 * axis:16 + 2 * axis] + mutual * y[12 + axis]))
    rotor_i = np.array(rotor_i)
    infinity = model.infinite_bus * np.exp(-1j * y[4])
    delivered = np.dot([infinity.real, infinity.imag], y[12:14])
    balance = (p.POWER * p.TORQUE_SHARES @ (1 + y[6:12])
               + model.field_voltage * rotor_i[0, 0] - delivered
               - p.LINE_R * np.sum(y[12:14]**2)
               - np.sum(np.array([p.RD, p.RQ]) * rotor_i**2))
    dy = model.rhs(0, y)
    epsilon = 1e-7
    numerical = (energy(y + epsilon * dy) - energy(y - epsilon * dy)) / (2 * epsilon)
    assert abs(numerical - balance) < 1e-7


def test_fault_trajectory_converges_and_produces_requested_signals():
    model = FBM()
    first = model.simulate(duration=0.8, sample_rate=120, fault_start=0.1,
                           rtol=1e-7, max_step=1 / 1200)
    second = model.simulate(duration=0.8, sample_rate=120, fault_start=0.1,
                            rtol=1e-9, max_step=1 / 2400)
    assert first.speed_pu.shape == (97, 6)
    np.testing.assert_allclose(first.speed_pu, second.speed_pu, atol=1e-7, rtol=0)
    np.testing.assert_allclose(first.electrical_torque_pu, second.electrical_torque_pu,
                               atol=1e-6, rtol=0)
    assert np.ptp(first.speed_pu[:, 4]) > 0.001
    assert np.min(first.voltage_pu) < 0.9
    assert np.all(np.isfinite(first.frequency_hz))
