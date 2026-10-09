"""Training loop for the frozen parametric PINN experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch

from .dataset import (
    build_collocation_points,
    build_supervised_dataset,
    normalization_from_training,
)
from .dynamics import SMIBScenario
from .pinn import ParametricPINN, normalized_physics_loss


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 500
    anchors_per_case: int = 61
    collocation_per_case: int = 48
    learning_rate: float = 1e-3
    physics_weight: float = 0.1
    delta_data_scale: float = 0.5
    omega_data_scale: float = 0.005
    seed: int = 7
    hidden_width: int = 64
    hidden_layers: int = 4


@dataclass
class TrainingResult:
    model: ParametricPINN
    history: list[dict[str, float]]
    config: TrainingConfig
    centre: np.ndarray
    scale: np.ndarray

    def metadata(self) -> dict[str, object]:
        return {
            "config": asdict(self.config),
            "centre": self.centre.tolist(),
            "scale": self.scale.tolist(),
        }


def train_parametric_pinn(
    scenarios: list[SMIBScenario],
    *,
    config: TrainingConfig,
) -> TrainingResult:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.set_num_threads(1)

    supervised = build_supervised_dataset(
        scenarios, samples_per_case=config.anchors_per_case
    )
    collocation = build_collocation_points(
        scenarios,
        points_per_case=config.collocation_per_case,
        seed=config.seed,
    )
    centre, scale = normalization_from_training(supervised.x)

    x_data = torch.tensor(supervised.x, dtype=torch.float32)
    y_data = torch.tensor(supervised.y, dtype=torch.float32)
    x_phys = torch.tensor(collocation, dtype=torch.float32)

    base = scenarios[0]
    model = ParametricPINN(
        centre,
        scale,
        hidden_width=config.hidden_width,
        hidden_layers=config.hidden_layers,
        delta0=base.initial_delta,
        t_end=base.t_end,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    history: list[dict[str, float]] = []
    for epoch in range(1, config.epochs + 1):
        optimizer.zero_grad(set_to_none=True)

        prediction = model(x_data)
        delta_loss = torch.mean(
            ((prediction[:, 0] - y_data[:, 0]) / config.delta_data_scale) ** 2
        )
        omega_loss = torch.mean(
            ((prediction[:, 1] - y_data[:, 1]) / config.omega_data_scale) ** 2
        )
        data_loss = delta_loss + omega_loss
        physics_loss = normalized_physics_loss(model, x_phys)
        total_loss = data_loss + config.physics_weight * physics_loss

        total_loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 25 == 0 or epoch == config.epochs:
            history.append(
                {
                    "epoch": float(epoch),
                    "loss": float(total_loss.detach()),
                    "data_loss": float(data_loss.detach()),
                    "physics_loss": float(physics_loss.detach()),
                }
            )
    return TrainingResult(model, history, config, centre, scale)
