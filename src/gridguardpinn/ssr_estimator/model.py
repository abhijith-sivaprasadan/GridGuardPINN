"""Six-mass FBM circuit ODE, with an explicitly approximate balanced fault.

Rotor-frame generator-output stator currents; positive rotor winding currents.
No ANDES calls. Equations, switching approximation and units: docs/ssr/MODEL.md.
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eigh
from scipy.signal import resample_poly

from . import parameters as p


def shaft_modes():
    """Mass-normalized angular-displacement eigenvectors, including rigid mode."""
    values, vectors = eigh(p.stiffness_matrix(), np.diag(2 * p.H / p.OMEGA_BASE))
    return np.sqrt(np.maximum(values, 0)) / (2 * np.pi), vectors


def shaft_ringdown(duration=12.0, sample_rate=240, no_load_damping=True):
    """Exact linear shaft response to a GEN speed kick, independent of estimator.

    Fig. 4 supplies decrements for modes 1-4. Rigid and unverified fifth mode
    are left undamped. These are configured reference inputs, not predictions.
    """
    frequencies, vectors = shaft_modes()
    decrements = np.r_[0.0, p.NO_LOAD_DECREMENT, 0.0] if no_load_damping else np.zeros(6)
    time = np.arange(round(duration * sample_rate) + 1) / sample_rate
    initial_velocity = np.zeros(6)
    initial_velocity[4] = 0.001 * p.OMEGA_BASE
    modal_velocity = vectors.T @ (2 * p.H / p.OMEGA_BASE * initial_velocity)
    natural = 2 * np.pi * frequencies
    damped = np.sqrt(np.maximum(natural**2 - decrements**2, 0))
    response = np.ones((len(time), 6))
    for k in range(1, 6):
        response[:, k] = np.exp(-decrements[k] * time) * (
            np.cos(damped[k] * time)
            - decrements[k] / damped[k] * np.sin(damped[k] * time)
        )
    speed = (response * modal_velocity) @ vectors.T / p.OMEGA_BASE
    return time, speed


@dataclass
class Simulation:
    time: np.ndarray
    speed_pu: np.ndarray
    electrical_torque_pu: np.ndarray
    voltage_pu: np.ndarray
    frequency_hz: np.ndarray
    sample_rate: int
    equilibrium_residual: float


class FBM:
    """20-state transient model: six angles/speeds, 2 currents, 4 rotor fluxes,
    and two series-capacitor voltages. Constant field voltage and shaft torque.
    """

    def __init__(self, capacitor_x=p.CAPACITOR_X):
        if not np.isfinite(capacitor_x) or not 0 < capacitor_x < p.LINE_X:
            raise ValueError("capacitor_x must lie strictly between 0 and line X=0.5")
        self.capacitor_x = float(capacitor_x)
        self.stiffness = p.stiffness_matrix()
        self.rotor_matrices = []
        self.couplings = []
        self.effective_x = []
        for leakage, mutual in [(p.XD_LEAKAGES, p.XD_MUTUAL),
                                (p.XQ_LEAKAGES, p.XQ_MUTUAL)]:
            rotor = np.diag(leakage[1:]) + mutual * np.ones((2, 2))
            coupling = np.linalg.solve(rotor, np.full(2, mutual))
            self.rotor_matrices.append(rotor)
            self.couplings.append(coupling)
            self.effective_x.append(leakage[0] + mutual - mutual * sum(coupling))
        self.couplings = np.array(self.couplings)
        self.effective_x = np.array(self.effective_x)
        self.rotor_inverses = np.linalg.inv(np.array(self.rotor_matrices))
        self.initial = self._equilibrium()

    def _equilibrium(self):
        current = complex(p.POWER, -p.POWER * np.tan(np.arccos(p.POWER_FACTOR)))
        external_x = p.TRANSFORMER_X + p.LINE_X + p.SYSTEM_X - self.capacitor_x
        self.infinite_bus = 1 - complex(p.LINE_R, external_x) * current
        theta = np.arctan2(-(1 + (p.XQ_MUTUAL + p.XQ_LEAKAGES[0]) * -current.imag),
                           (p.XQ_MUTUAL + p.XQ_LEAKAGES[0]) * current.real)
        voltage = np.exp(-1j * theta)
        rotor_current = current * np.exp(-1j * theta)
        field_current = (voltage.imag + (p.XD_MUTUAL + p.XD_LEAKAGES[0])
                         * rotor_current.real) / p.XD_MUTUAL
        self.field_voltage = p.RD[0] * field_current
        y = np.zeros(20)
        forcing = p.POWER * (p.TORQUE_SHARES - np.eye(6)[4])
        free = [0, 1, 2, 3, 5]
        y[free] = np.linalg.solve(self.stiffness[np.ix_(free, free)], forcing[free])
        y[:6] += theta
        y[12:14] = [rotor_current.real, rotor_current.imag]
        y[14:16] = (self.rotor_matrices[0] @ np.array([field_current, 0])
                    - p.XD_MUTUAL * rotor_current.real)
        y[16:18] = -p.XQ_MUTUAL * rotor_current.imag
        cap = -1j * self.capacitor_x * rotor_current
        y[18:20] = [cap.real, cap.imag]
        return y

    def _circuit(self, y):
        current = y[12:14]
        rotor_flux = y[14:18].reshape(2, 2)
        rotor_current = np.einsum(
            "kij,kj->ki", self.rotor_inverses,
            rotor_flux + np.array([p.XD_MUTUAL, p.XQ_MUTUAL])[:, None] * current[:, None],
        )
        flux = -self.effective_x * current + np.sum(self.couplings * rotor_flux, axis=1)
        rotor_derivative = -p.OMEGA_BASE * np.array([p.RD, p.RQ]) * rotor_current
        rotor_derivative[0, 0] += p.OMEGA_BASE * self.field_voltage
        torque = flux[0] * current[1] - flux[1] * current[0]
        return flux, rotor_derivative, torque

    def rhs(self, time, y, fault=False):
        del time
        omega = 1 + y[10]
        flux, rotor_derivative, torque = self._circuit(y)
        scale = p.FAULT_X / (p.FAULT_X + p.SYSTEM_X) if fault else 1.0
        xline = p.TRANSFORMER_X + p.LINE_X + p.SYSTEM_X * scale
        infinity = scale * self.infinite_bus * np.exp(-1j * y[4])
        voltage = np.array([infinity.real, infinity.imag]) + y[18:20]
        effective_flux = flux - xline * y[12:14]
        rotational = omega * np.array([effective_flux[1], -effective_flux[0]])
        result = np.zeros(20)
        result[:6] = p.OMEGA_BASE * y[6:12]
        result[6:12] = (p.POWER * p.TORQUE_SHARES - self.stiffness @ y[:6]
                        - torque * np.eye(6)[4]) / (2 * p.H)
        result[12:14] = (
            np.sum(self.couplings * rotor_derivative, axis=1)
            - p.OMEGA_BASE * (voltage + (p.LINE_R + p.ARMATURE_R) * y[12:14]
                              + rotational)
        ) / (self.effective_x + xline)
        result[14:18] = rotor_derivative.ravel()
        result[18:20] = p.OMEGA_BASE * (
            self.capacitor_x * y[12:14] + omega * np.array([y[19], -y[18]])
        )
        return result

    def terminal_voltage(self, y, fault=False):
        derivative = self.rhs(0, y, fault)
        flux, rotor_derivative, _ = self._circuit(y)
        flux_derivative = (-self.effective_x * derivative[12:14]
                           + np.sum(self.couplings * rotor_derivative, axis=1))
        rotor_voltage = (-p.ARMATURE_R * y[12:14] + flux_derivative / p.OMEGA_BASE
                         + (1 + y[10]) * np.array([-flux[1], flux[0]]))
        return complex(*rotor_voltage) * np.exp(1j * y[4])

    def frozen_electrical_resonance(self, fault=False):
        """Sub-synchronous circuit pole with shaft held at synchronous speed.

        The circuit is affine at fixed angle/speed. Map the low-frequency
        rotor-frame pole to its 60-f sideband; this is not a coupled shaft mode.
        """
        step = 1e-6
        jacobian = np.column_stack([
            (self.rhs(0, self.initial + step * np.eye(20)[k], fault)
             - self.rhs(0, self.initial - step * np.eye(20)[k], fault))[12:] / (2 * step)
            for k in range(12, 20)
        ])
        poles = np.linalg.eigvals(jacobian)
        candidates = [pole for pole in poles if 0 < pole.imag / (2 * np.pi) < p.BASE_HZ]
        if len(candidates) != 1:
            raise ValueError("Electrical sideband is not uniquely identifiable")
        return float(p.BASE_HZ - candidates[0].imag / (2 * np.pi))

    def simulate(self, duration=3.0, sample_rate=240, fault_start=0.5,
                 fault_duration=p.FAULT_DURATION, rtol=1e-8, max_step=1 / 2400):
        if not isinstance(sample_rate, int) or sample_rate < 120 or 2400 % sample_rate:
            raise ValueError("Use a reporting rate >=120 Hz dividing internal 2400 Hz")
        clear = fault_start + fault_duration
        if not 0 < fault_start < clear < duration:
            raise ValueError("Fault must begin and clear strictly inside the simulation")
        times = np.arange(int(np.floor(duration * 2400)) + 1) / 2400
        states = np.empty((len(times), 20))
        y = self.initial.copy()
        for start, stop, fault in [(0, fault_start, False),
                                   (fault_start, clear, True), (clear, duration, False)]:
            solution = solve_ivp(lambda t, x, active=fault: self.rhs(t, x, active),
                                 (start, stop), y,
                                 method="DOP853", rtol=rtol, atol=rtol * 0.01,
                                 max_step=max_step, dense_output=True)
            if not solution.success:
                raise RuntimeError(solution.message)
            mask = (times >= start) & (times <= stop)
            states[mask] = solution.sol(times[mask]).T
            y = solution.y[:, -1]
        torque = np.array([self._circuit(y)[2] for y in states])
        voltage = np.array([self.terminal_voltage(y, fault_start <= t < clear)
                            for t, y in zip(times, states)])
        factor = 2400 // sample_rate

        def report(x):
            return resample_poly(x, 1, factor, axis=0, padtype="line")

        voltage = report(voltage)
        frequency = p.BASE_HZ + np.gradient(np.unwrap(np.angle(voltage)),
                                            1 / sample_rate) / (2 * np.pi)
        frequency[np.abs(voltage) < 0.05] = np.nan
        return Simulation(times[::factor], 1 + report(states[:, 6:12]), report(torque),
                          np.abs(voltage), frequency, sample_rate,
                          float(np.max(np.abs(self.rhs(0, self.initial)))))
