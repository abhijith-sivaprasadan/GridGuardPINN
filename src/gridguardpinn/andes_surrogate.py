"""Swing-equation-informed multi-machine surrogate for ANDES IEEE-14 cases."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

N_GENERATORS = 5
N_BUSES = 14
OUTPUT_DIM = 4 * N_GENERATORS


def raw_input(t: np.ndarray, fault_duration_s: float, fault_bus: int) -> np.ndarray:
    """Return [time, duration, 14-way one-hot bus] rows."""
    t = np.asarray(t, dtype=float).reshape(-1)
    if not 1 <= fault_bus <= N_BUSES:
        raise ValueError("fault_bus must be in [1, 14].")
    x = np.zeros((t.size, 2 + N_BUSES), dtype=float)
    x[:, 0] = t
    x[:, 1] = fault_duration_s
    x[:, 1 + fault_bus] = 1.0
    return x


class AndesMultiMachineSurrogate(nn.Module):
    """Map time/disturbance inputs to five GENROU electromechanical trajectories."""

    def __init__(
        self,
        output_shift: np.ndarray,
        output_scale: np.ndarray,
        *,
        hidden_width: int = 96,
        hidden_layers: int = 5,
        t_end_s: float = 2.0,
        duration_min_s: float = 0.04,
        duration_max_s: float = 0.12,
    ) -> None:
        super().__init__()
        shift = np.asarray(output_shift, dtype=np.float32).reshape(OUTPUT_DIM)
        scale = np.asarray(output_scale, dtype=np.float32).reshape(OUTPUT_DIM)
        scale = np.maximum(scale, 1e-5)
        self.register_buffer("output_shift", torch.tensor(shift))
        self.register_buffer("output_scale", torch.tensor(scale))
        self.t_end_s = float(t_end_s)
        self.duration_min_s = float(duration_min_s)
        self.duration_max_s = float(duration_max_s)

        encoded_dim = 2 + 2 + 2 + 8 + N_BUSES
        layers: list[nn.Module] = [
            nn.Linear(encoded_dim, hidden_width),
            nn.Tanh(),
        ]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_width, hidden_width), nn.Tanh()])
        layers.append(nn.Linear(hidden_width, OUTPUT_DIM))
        self.net = nn.Sequential(*layers)
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != 2 + N_BUSES:
            raise ValueError(f"x must have shape [N, {2 + N_BUSES}].")
        t = x[:, 0:1]
        duration = x[:, 1:2]
        buses = x[:, 2:]

        t_scaled = 2.0 * t / self.t_end_s - 1.0
        duration_span = max(self.duration_max_s - self.duration_min_s, 1e-6)
        duration_scaled = (
            2.0 * (duration - self.duration_min_s) / duration_span - 1.0
        )

        fault_start = torch.full_like(t, 1.0)
        fault_clear = fault_start + duration
        rel_fault = (t - fault_start) / self.t_end_s
        rel_clear = (t - fault_clear) / self.t_end_s
        fault_on = ((t >= fault_start) & (t < fault_clear)).to(t.dtype)
        post_fault = (t >= fault_clear).to(t.dtype)

        fourier: list[torch.Tensor] = []
        for frequency_hz in (0.5, 1.0, 2.0, 4.0):
            phase = 2.0 * math.pi * frequency_hz * t
            fourier.extend([torch.sin(phase), torch.cos(phase)])

        return torch.cat(
            [
                t_scaled,
                duration_scaled,
                rel_fault,
                rel_clear,
                fault_on,
                post_fault,
                *fourier,
                buses,
            ],
            dim=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.net(self.encode(x))
        return normalized * self.output_scale + self.output_shift


def split_outputs(output: torch.Tensor) -> tuple[torch.Tensor, ...]:
    if output.ndim != 2 or output.shape[1] != OUTPUT_DIM:
        raise ValueError(f"output must have shape [N, {OUTPUT_DIM}].")
    return (
        output[:, 0:5],
        output[:, 5:10],
        output[:, 10:15],
        output[:, 15:20],
    )


def swing_residuals(
    model: nn.Module,
    x: torch.Tensor,
    *,
    inertia_M: torch.Tensor,
    damping_D: torch.Tensor,
    frequency_hz: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the two shared GENBase electromechanical residual families."""
    if not x.requires_grad:
        x = x.clone().detach().requires_grad_(True)
    delta, omega, tm, te = split_outputs(model(x))

    ddelta_columns: list[torch.Tensor] = []
    domega_columns: list[torch.Tensor] = []
    for machine in range(N_GENERATORS):
        ddelta = torch.autograd.grad(
            delta[:, machine].sum(),
            x,
            create_graph=True,
            retain_graph=True,
        )[0][:, 0]
        domega = torch.autograd.grad(
            omega[:, machine].sum(),
            x,
            create_graph=True,
            retain_graph=True,
        )[0][:, 0]
        ddelta_columns.append(ddelta)
        domega_columns.append(domega)

    ddelta_dt = torch.stack(ddelta_columns, dim=1)
    domega_dt = torch.stack(domega_columns, dim=1)

    inertia = inertia_M.reshape(1, -1).to(x)
    damping = damping_D.reshape(1, -1).to(x)
    frequency = frequency_hz.reshape(1, -1).to(x)

    r_delta = ddelta_dt - 2.0 * math.pi * frequency * (omega - 1.0)
    r_omega = inertia * domega_dt - (tm - te - damping * (omega - 1.0))
    return r_delta, r_omega


def swing_physics_loss(
    model: nn.Module,
    x: torch.Tensor,
    *,
    inertia_M: torch.Tensor,
    damping_D: torch.Tensor,
    frequency_hz: torch.Tensor,
) -> torch.Tensor:
    """Scaled residual loss; scaling prevents unit magnitude from dominating."""
    r_delta, r_omega = swing_residuals(
        model,
        x,
        inertia_M=inertia_M,
        damping_D=damping_D,
        frequency_hz=frequency_hz,
    )
    return torch.mean((r_delta / 0.2) ** 2) + torch.mean((r_omega / 0.05) ** 2)
