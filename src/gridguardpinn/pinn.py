"""PyTorch PINN baseline for parametric SMIB trajectories.

Torch is an optional dependency so the reference-solver and trust-gate core can
remain lightweight.
"""

from __future__ import annotations

import math


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            'PyTorch is required for PINN components. Install with: pip install -e ".[ml]"'
        ) from exc
    return torch


def build_mlp(
    input_dim: int = 5,
    output_dim: int = 2,
    hidden_width: int = 64,
    hidden_layers: int = 4,
):
    torch = _torch()
    layers = [torch.nn.Linear(input_dim, hidden_width), torch.nn.Tanh()]
    for _ in range(hidden_layers - 1):
        layers.extend([torch.nn.Linear(hidden_width, hidden_width), torch.nn.Tanh()])
    layers.append(torch.nn.Linear(hidden_width, output_dim))
    return torch.nn.Sequential(*layers)


def physics_residuals(
    model,
    x,
    *,
    Pm: float = 0.8,
    Pmax_pre: float = 1.2,
    Pmax_post: float = 1.0,
    t_fault: float = 0.10,
    frequency_hz: float = 50.0,
):
    """Return swing-equation residuals for x=[t,H,D,t_clear,fault_ratio].

    The network output is [delta, omega]. Derivatives are computed with respect
    to physical time, while scenario parameters are treated as conditioning
    inputs.
    """
    torch = _torch()
    if x.ndim != 2 or x.shape[1] != 5:
        raise ValueError("x must have shape [N, 5]: t,H,D,t_clear,fault_ratio")

    if not x.requires_grad:
        x = x.clone().detach().requires_grad_(True)

    output = model(x)
    delta = output[:, 0:1]
    omega = output[:, 1:2]

    grad_delta = torch.autograd.grad(
        delta,
        x,
        grad_outputs=torch.ones_like(delta),
        create_graph=True,
        retain_graph=True,
    )[0][:, 0:1]
    grad_omega = torch.autograd.grad(
        omega,
        x,
        grad_outputs=torch.ones_like(omega),
        create_graph=True,
        retain_graph=True,
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


def physics_loss(model, x, **kwargs):
    torch = _torch()
    r_delta, r_omega = physics_residuals(model, x, **kwargs)
    return torch.mean(r_delta**2) + torch.mean(r_omega**2)
