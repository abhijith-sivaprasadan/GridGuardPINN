"""Train a first single-scenario hybrid PINN baseline.

This script is intentionally a baseline. It establishes that the PINN and
physics-residual machinery runs end-to-end before the parametric trust-gate
experiment is attempted.
"""

from __future__ import annotations

import argparse

import numpy as np

from gridguardpinn.dynamics import SMIBScenario
from gridguardpinn.reference import simulate_reference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    try:
        import torch
    except ImportError as exc:
        raise SystemExit('Install ML extras first: pip install -e ".[ml]"') from exc

    from gridguardpinn.pinn import build_mlp, physics_loss

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    scenario = SMIBScenario()
    reference = simulate_reference(scenario, samples=401)

    t = torch.tensor(reference.t[:, None], dtype=torch.float32)
    H = torch.full_like(t, scenario.H)
    D = torch.full_like(t, scenario.D)
    t_clear = torch.full_like(t, scenario.t_clear)
    fault_ratio = torch.full_like(t, scenario.Pmax_fault / scenario.Pmax_pre)
    x = torch.cat([t, H, D, t_clear, fault_ratio], dim=1)
    y = torch.tensor(reference.states, dtype=torch.float32)

    model = build_mlp(hidden_width=64, hidden_layers=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        prediction = model(x)
        data_loss = torch.mean((prediction - y) ** 2)
        phys_loss = physics_loss(model, x)
        loss = data_loss + 0.1 * phys_loss
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 200 == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:5d} loss={loss.item():.6e} "
                f"data={data_loss.item():.6e} physics={phys_loss.item():.6e}"
            )


if __name__ == "__main__":
    main()
