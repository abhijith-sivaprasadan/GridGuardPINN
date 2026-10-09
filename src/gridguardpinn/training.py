"""Training loop for the parametric PINN experiments."""

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
    epochs: int = 2500
    anchors_per_case: int = 121
    collocation_per_case: int = 96
    learning_rate: float = 1e-3
    final_learning_rate: float = 5e-5
    physics_weight: float = 0.1
    physics_warmup_epochs: int = 250
    physics_ramp_epochs: int = 600
    delta_data_scale: float = 0.5
    omega_data_scale: float = 0.005
    seed: int = 7
    hidden_width: int = 96
    hidden_layers: int = 5
    event_aware_sampling: bool = True
    phase_stratified_collocation: bool = True
    checkpoint_every: int = 25


@dataclass
class TrainingResult:
    model: ParametricPINN
    history: list[dict[str, float]]
    config: TrainingConfig
    centre: np.ndarray
    scale: np.ndarray
    best_epoch: int
    best_validation_loss: float | None

    def metadata(self) -> dict[str, object]:
        return {
            "config": asdict(self.config),
            "centre": self.centre.tolist(),
            "scale": self.scale.tolist(),
            "best_epoch": self.best_epoch,
            "best_validation_loss": self.best_validation_loss,
        }


def _physics_weight(epoch: int, config: TrainingConfig) -> float:
    if epoch <= config.physics_warmup_epochs:
        return 0.0
    progress = (epoch - config.physics_warmup_epochs) / max(
        1, config.physics_ramp_epochs
    )
    return config.physics_weight * min(1.0, max(0.0, progress))


def _scaled_data_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    config: TrainingConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    delta_loss = torch.mean(
        ((prediction[:, 0] - target[:, 0]) / config.delta_data_scale) ** 2
    )
    omega_loss = torch.mean(
        ((prediction[:, 1] - target[:, 1]) / config.omega_data_scale) ** 2
    )
    return delta_loss + omega_loss, delta_loss, omega_loss


def train_parametric_pinn(
    scenarios: list[SMIBScenario],
    *,
    config: TrainingConfig,
    validation_scenarios: list[SMIBScenario] | None = None,
) -> TrainingResult:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.set_num_threads(1)

    supervised = build_supervised_dataset(
        scenarios,
        samples_per_case=config.anchors_per_case,
        event_aware=config.event_aware_sampling,
    )
    collocation = build_collocation_points(
        scenarios,
        points_per_case=config.collocation_per_case,
        seed=config.seed,
        phase_stratified=config.phase_stratified_collocation,
    )
    centre, scale = normalization_from_training(supervised.x)

    x_data = torch.tensor(supervised.x, dtype=torch.float32)
    y_data = torch.tensor(supervised.y, dtype=torch.float32)
    x_phys = torch.tensor(collocation, dtype=torch.float32)

    x_validation: torch.Tensor | None = None
    y_validation: torch.Tensor | None = None
    if validation_scenarios:
        validation = build_supervised_dataset(
            validation_scenarios,
            samples_per_case=config.anchors_per_case,
            event_aware=True,
        )
        x_validation = torch.tensor(validation.x, dtype=torch.float32)
        y_validation = torch.tensor(validation.y, dtype=torch.float32)

    base = scenarios[0]
    model = ParametricPINN(
        centre,
        scale,
        hidden_width=config.hidden_width,
        hidden_layers=config.hidden_layers,
        delta0=base.initial_delta,
        t_fault=base.t_fault,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=config.final_learning_rate,
    )

    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = config.epochs
    best_validation_loss = float("inf") if x_validation is not None else None

    for epoch in range(1, config.epochs + 1):
        optimizer.zero_grad(set_to_none=True)

        prediction = model(x_data)
        data_loss, delta_loss, omega_loss = _scaled_data_loss(
            prediction, y_data, config
        )
        physics_loss = normalized_physics_loss(model, x_phys)
        physics_weight = _physics_weight(epoch, config)
        total_loss = data_loss + physics_weight * physics_loss

        total_loss.backward()
        optimizer.step()
        scheduler.step()

        checkpoint = (
            epoch == 1
            or epoch % config.checkpoint_every == 0
            or epoch == config.epochs
        )
        if checkpoint:
            validation_loss_value = float("nan")
            if x_validation is not None and y_validation is not None:
                model.eval()
                with torch.no_grad():
                    validation_prediction = model(x_validation)
                    validation_loss, _, _ = _scaled_data_loss(
                        validation_prediction, y_validation, config
                    )
                validation_loss_value = float(validation_loss)
                model.train()

                if (
                    best_validation_loss is not None
                    and validation_loss_value < best_validation_loss
                ):
                    best_validation_loss = validation_loss_value
                    best_epoch = epoch
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }

            history.append(
                {
                    "epoch": float(epoch),
                    "loss": float(total_loss.detach()),
                    "data_loss": float(data_loss.detach()),
                    "delta_data_loss": float(delta_loss.detach()),
                    "omega_data_loss": float(omega_loss.detach()),
                    "physics_loss": float(physics_loss.detach()),
                    "physics_weight": float(physics_weight),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "validation_data_loss": validation_loss_value,
                }
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    return TrainingResult(
        model=model,
        history=history,
        config=config,
        centre=centre,
        scale=scale,
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
    )
