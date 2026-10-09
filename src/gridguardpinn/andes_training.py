"""Training loop for the first ANDES multi-machine surrogate."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from .andes_multimachine import (
    collocation_array,
    machine_constants,
    supervised_arrays,
)
from .andes_surrogate import AndesMultiMachineSurrogate, swing_physics_loss


@dataclass(frozen=True)
class AndesTrainingConfig:
    epochs: int = 1000
    anchors_per_case: int = 101
    collocation_per_case: int = 56
    learning_rate: float = 1e-3
    final_learning_rate: float = 5e-5
    physics_weight: float = 0.02
    physics_warmup_epochs: int = 150
    physics_ramp_epochs: int = 300
    hidden_width: int = 96
    hidden_layers: int = 5
    checkpoint_every: int = 25
    seed: int = 17


@dataclass
class AndesTrainingResult:
    model: AndesMultiMachineSurrogate
    history: list[dict[str, float]]
    config: AndesTrainingConfig
    best_epoch: int
    best_validation_loss: float
    training_seconds: float

    def metadata(self) -> dict[str, object]:
        return {
            "config": asdict(self.config),
            "best_epoch": self.best_epoch,
            "best_validation_loss": self.best_validation_loss,
            "training_seconds": self.training_seconds,
        }


def _physics_weight(epoch: int, config: AndesTrainingConfig) -> float:
    if epoch <= config.physics_warmup_epochs:
        return 0.0
    progress = (epoch - config.physics_warmup_epochs) / max(
        1,
        config.physics_ramp_epochs,
    )
    return config.physics_weight * min(1.0, max(0.0, progress))


def _output_scaling(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shift = np.mean(y, axis=0)
    scale = np.std(y, axis=0)
    floors = np.concatenate(
        [
            np.full(5, 0.05),
            np.full(5, 1e-3),
            np.full(5, 0.02),
            np.full(5, 0.02),
        ]
    )
    return shift, np.maximum(scale, floors)


def _scaled_loss(prediction, target, scale):
    return torch.mean(((prediction - target) / scale) ** 2)


def train_andes_surrogate(
    reference_map,
    train_cases,
    validation_cases,
    *,
    config: AndesTrainingConfig,
) -> AndesTrainingResult:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.set_num_threads(2)

    x_train_np, y_train_np = supervised_arrays(
        reference_map,
        train_cases,
        anchors_per_case=config.anchors_per_case,
    )
    x_val_np, y_val_np = supervised_arrays(
        reference_map,
        validation_cases,
        anchors_per_case=config.anchors_per_case,
    )
    x_phys_np = collocation_array(
        train_cases,
        points_per_case=config.collocation_per_case,
        seed=config.seed,
    )
    constants = machine_constants(reference_map)
    shift, scale = _output_scaling(y_train_np)

    model = AndesMultiMachineSurrogate(
        shift,
        scale,
        hidden_width=config.hidden_width,
        hidden_layers=config.hidden_layers,
    )
    x_train = torch.tensor(x_train_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.float32)
    x_val = torch.tensor(x_val_np, dtype=torch.float32)
    y_val = torch.tensor(y_val_np, dtype=torch.float32)
    x_phys = torch.tensor(x_phys_np, dtype=torch.float32)
    output_scale = model.output_scale.reshape(1, -1)
    inertia = torch.tensor(constants.inertia_M, dtype=torch.float32)
    damping = torch.tensor(constants.damping_D, dtype=torch.float32)
    frequency = torch.tensor(constants.frequency_hz, dtype=torch.float32)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=config.final_learning_rate,
    )

    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_val = float("inf")
    started = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        data_loss = _scaled_loss(model(x_train), y_train, output_scale)
        weight = _physics_weight(epoch, config)
        if weight > 0.0:
            physics_loss = swing_physics_loss(
                model,
                x_phys,
                inertia_M=inertia,
                damping_D=damping,
                frequency_hz=frequency,
            )
            physics_loss_value = float(physics_loss.detach())
            loss = data_loss + weight * physics_loss
        else:
            physics_loss_value = float("nan")
            loss = data_loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        checkpoint = (
            epoch == 1
            or epoch % config.checkpoint_every == 0
            or epoch == config.epochs
        )
        if checkpoint:
            model.eval()
            with torch.no_grad():
                val_loss = float(
                    _scaled_loss(model(x_val), y_val, output_scale)
                )
            history.append(
                {
                    "epoch": float(epoch),
                    "total_loss": float(loss.detach()),
                    "data_loss": float(data_loss.detach()),
                    "physics_loss": physics_loss_value,
                    "physics_weight": float(weight),
                    "validation_scaled_mse": val_loss,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
            )
            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }

    if best_state is not None:
        model.load_state_dict(best_state)

    return AndesTrainingResult(
        model=model,
        history=history,
        config=config,
        best_epoch=best_epoch,
        best_validation_loss=best_val,
        training_seconds=time.perf_counter() - started,
    )
