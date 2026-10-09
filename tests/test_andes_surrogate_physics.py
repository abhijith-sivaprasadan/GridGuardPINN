import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from gridguardpinn.andes_surrogate import (
    N_BUSES,
    N_GENERATORS,
    raw_input,
    swing_physics_loss,
    swing_residuals,
)


class ExactSwingModel(nn.Module):
    def __init__(self, inertia, damping, frequency, torque_bias=0.0):
        super().__init__()
        self.register_buffer("inertia", inertia)
        self.register_buffer("damping", damping)
        self.register_buffer("frequency", frequency)
        self.torque_bias = float(torque_bias)

    def forward(self, x):
        t = x[:, 0:1]
        slopes = torch.linspace(
            2e-4,
            6e-4,
            N_GENERATORS,
            dtype=x.dtype,
            device=x.device,
        ).reshape(1, -1)
        omega = 1.0 + t * slopes
        delta = (
            0.1
            + math.pi
            * self.frequency.reshape(1, -1)
            * slopes
            * t**2
        )
        tm = torch.full_like(omega, 0.8)
        te = (
            tm
            - self.inertia.reshape(1, -1) * slopes
            - self.damping.reshape(1, -1) * (omega - 1.0)
            + self.torque_bias
        )
        return torch.cat([delta, omega, tm, te], dim=1)


def test_raw_input_fault_bus_is_one_hot():
    x = raw_input(np.asarray([0.0, 1.0]), 0.08, 9)
    assert x.shape == (2, 2 + N_BUSES)
    assert np.allclose(x[:, 0], [0.0, 1.0])
    assert np.allclose(x[:, 1], 0.08)
    assert np.allclose(x[:, 2:].sum(axis=1), 1.0)
    assert np.allclose(x[:, 1 + 9], 1.0)


def test_exact_swing_solution_has_near_zero_residual():
    inertia = torch.linspace(4.0, 8.0, N_GENERATORS)
    damping = torch.linspace(0.5, 1.5, N_GENERATORS)
    frequency = torch.full((N_GENERATORS,), 60.0)
    model = ExactSwingModel(inertia, damping, frequency)

    x = torch.tensor(
        raw_input(np.linspace(0.0, 2.0, 41), 0.08, 9),
        dtype=torch.float32,
        requires_grad=True,
    )
    r_delta, r_omega = swing_residuals(
        model,
        x,
        inertia_M=inertia,
        damping_D=damping,
        frequency_hz=frequency,
    )
    assert float(torch.max(torch.abs(r_delta))) < 2e-5
    assert float(torch.max(torch.abs(r_omega))) < 2e-6
    assert float(
        swing_physics_loss(
            model,
            x,
            inertia_M=inertia,
            damping_D=damping,
            frequency_hz=frequency,
        )
    ) < 1e-8


def test_torque_bias_is_detected_by_physics_residual():
    inertia = torch.linspace(4.0, 8.0, N_GENERATORS)
    damping = torch.linspace(0.5, 1.5, N_GENERATORS)
    frequency = torch.full((N_GENERATORS,), 60.0)
    model = ExactSwingModel(
        inertia,
        damping,
        frequency,
        torque_bias=0.01,
    )

    x = torch.tensor(
        raw_input(np.linspace(0.0, 2.0, 41), 0.08, 9),
        dtype=torch.float32,
        requires_grad=True,
    )
    _, r_omega = swing_residuals(
        model,
        x,
        inertia_M=inertia,
        damping_D=damping,
        frequency_hz=frequency,
    )
    assert torch.allclose(
        r_omega,
        torch.full_like(r_omega, 0.01),
        atol=2e-6,
        rtol=0.0,
    )
