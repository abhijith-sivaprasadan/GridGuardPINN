"""Parametric physics-informed neural network for SMIB trajectories."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


class ParametricPINN(nn.Module):
    """MLP conditioned on time and four scenario parameters."""

    def __init__(
        self,
        centre: np.ndarray,
        scale: np.ndarray,
        *,
        hidden_width: int = 64,
        hidden_layers: int = 4,
        delta0: float,
        t_end: float = 5.0,
        delta_output_scale: float = 1.0,
        omega_output_scale: float = 0.01,
    ) -> None:
        super().__init__()
        self.register_buffer("centre", torch.as_tensor(centre, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))
        self.delta0 = float(delta0)
        self.t_end = float(t_end)
        self.delta_output_scale = float(delta_output_scale)
        self.omega_output_scale = float(omega_output_scale)

        layers: list[nn.Module] = [nn.Linear(5, hidden_width), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_width, hidden_width), nn.Tanh()])
        layers.append(nn.Linear(hidden_width, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = (x - self.centre) / self.scale
        raw = self.network(z)
        tau = x[:, 0:1] / self.t_end
        delta = self.delta0 + tau * self.delta_output_scale * raw[:, 0:1]
        omega = tau * self.omega_output_scale * raw[:, 1:2]
        return torch.cat([delta, omega], dim=1)


def physics_residuals(
    model: nn.Module,
    x: torch.Tensor,
    *,
    Pm: float = 0.8,
    Pmax_pre: float = 1.2,
    Pmax_post: float = 1.0,
    t_fault: float = 0.10,
    frequency_hz: float = 50.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return swing-equation residuals for x=[t,H,D,t_clear,fault_ratio]."""
    if x.ndim != 2 or x.shape[1] != 5:
        raise ValueError("x must have shape [N, 5]")
    if not x.requires_grad:
        x = x.clone().detach().requires_grad_(True)

    output = model(x)
    delta = output[:, 0:1]
    omega = output[:, 1:2]

    grad_delta = torch.autograd.grad(
        delta, x, grad_outputs=torch.ones_like(delta), create_graph=True, retain_graph=True
    )[0][:, 0:1]
    grad_omega = torch.autograd.grad(
        omega, x, grad_outputs=torch.ones_like(omega), create_graph=True, retain_graph=True
    )[0][:, 0:1]

    t = x[:, 0:1]
    H = x[:, 1:2]
    D = x[:, 2:3]
    t_clear = x[:, 3:4]
    fault_ratio = x[:, 4:5]

    pmax_fault = Pmax_pre * fault_ratio
    pmax = torch.where(
        t < t_fault,
        torch.full_like(t, Pmax_pre),
        torch.where(t < t_clear, pmax_fault, torch.full_like(t, Pmax_post)),
    )
    pe = pmax * torch.sin(delta)

    omega_base = 2.0 * math.pi * frequency_hz
    residual_delta = grad_delta - omega_base * omega
    residual_omega = grad_omega - (Pm - pe - D * omega) / (2.0 * H)
    return residual_delta, residual_omega


def normalized_physics_loss(
    model: nn.Module,
    x: torch.Tensor,
    *,
    delta_residual_scale: float = 1.0,
    omega_residual_scale: float = 0.1,
) -> torch.Tensor:
    r_delta, r_omega = physics_residuals(model, x)
    return torch.mean((r_delta / delta_residual_scale) ** 2) + torch.mean(
        (r_omega / omega_residual_scale) ** 2
    )
