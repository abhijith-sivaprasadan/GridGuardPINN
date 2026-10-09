"""Frozen scenario protocols for the parametric SMIB experiment."""

from __future__ import annotations

from dataclasses import replace
from itertools import product

import numpy as np

from .dynamics import SMIBScenario


def scenario_vector(scenario: SMIBScenario) -> np.ndarray:
    """Feature vector used by the parameter-space OOD detector."""
    fault_ratio = scenario.Pmax_fault / scenario.Pmax_pre
    return np.asarray([scenario.H, scenario.D, scenario.t_clear, fault_ratio], dtype=float)


def make_grid(
    *,
    H_values: tuple[float, ...],
    D_values: tuple[float, ...],
    t_clear_values: tuple[float, ...],
    fault_ratio_values: tuple[float, ...],
    base: SMIBScenario | None = None,
) -> list[SMIBScenario]:
    base = base or SMIBScenario()
    scenarios: list[SMIBScenario] = []
    for H, D, t_clear, fault_ratio in product(
        H_values, D_values, t_clear_values, fault_ratio_values
    ):
        scenarios.append(
            replace(
                base,
                H=H,
                D=D,
                t_clear=t_clear,
                Pmax_fault=base.Pmax_pre * fault_ratio,
            )
        )
    return scenarios


def canonical_splits() -> dict[str, list[SMIBScenario]]:
    """Frozen v0.2 train/calibration/test protocol."""
    train = make_grid(
        H_values=(4.0, 5.0, 6.0),
        D_values=(0.8, 1.2),
        t_clear_values=(0.17, 0.21),
        fault_ratio_values=(0.10, 0.25),
    )
    validation = make_grid(
        H_values=(4.25, 4.75, 5.25, 5.75),
        D_values=(0.9, 1.1),
        t_clear_values=(0.18, 0.22),
        fault_ratio_values=(0.175,),
    )
    test_id = make_grid(
        H_values=(4.5, 5.5),
        D_values=(0.85, 1.15),
        t_clear_values=(0.185, 0.215),
        fault_ratio_values=(0.14, 0.21),
    )
    ood = make_grid(
        H_values=(2.5, 7.5),
        D_values=(0.3, 1.8),
        t_clear_values=(0.14, 0.27),
        fault_ratio_values=(0.05, 0.35),
    )
    return {"train": train, "validation": validation, "test_id": test_id, "ood": ood}
