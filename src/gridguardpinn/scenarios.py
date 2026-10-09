"""Frozen scenario protocols for the parametric SMIB experiments."""

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


def _train_and_validation() -> tuple[list[SMIBScenario], list[SMIBScenario]]:
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
    return train, validation


def splits_v02() -> dict[str, list[SMIBScenario]]:
    """Original v0.2 split retained for baseline reproducibility."""
    train, validation = _train_and_validation()
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


def _random_id_case(rng: np.random.Generator) -> SMIBScenario:
    base = SMIBScenario()
    ratio = float(rng.uniform(0.115, 0.235))
    return replace(
        base,
        H=float(rng.uniform(4.1, 5.9)),
        D=float(rng.uniform(0.83, 1.17)),
        t_clear=float(rng.uniform(0.174, 0.206)),
        Pmax_fault=base.Pmax_pre * ratio,
    )


def splits_v03() -> dict[str, list[SMIBScenario]]:
    """Fresh v0.3 held-out sets frozen after observing the v0.2 baseline.

    The seed and generation ranges are fixed before evaluating the improved
    surrogate. OOD cases shift exactly one scenario factor outside the training
    box so failure attribution is more informative than the all-extreme v0.2 set.
    """
    train, validation = _train_and_validation()
    rng = np.random.default_rng(20261009)

    test_id = [_random_id_case(rng) for _ in range(24)]

    ood: list[SMIBScenario] = []
    factors = ("H", "D", "t_clear", "fault_ratio")
    for factor in factors:
        for index in range(6):
            case = _random_id_case(rng)
            low_side = index % 2 == 0
            if factor == "H":
                value = rng.uniform(2.7, 3.6) if low_side else rng.uniform(6.4, 7.3)
                case = replace(case, H=float(value))
            elif factor == "D":
                value = rng.uniform(0.35, 0.65) if low_side else rng.uniform(1.35, 1.70)
                case = replace(case, D=float(value))
            elif factor == "t_clear":
                value = rng.uniform(0.125, 0.155) if low_side else rng.uniform(0.235, 0.275)
                case = replace(case, t_clear=float(value))
            else:
                ratio = rng.uniform(0.03, 0.08) if low_side else rng.uniform(0.30, 0.38)
                case = replace(case, Pmax_fault=case.Pmax_pre * float(ratio))
            ood.append(case)

    return {"train": train, "validation": validation, "test_id": test_id, "ood": ood}


def classify_ood_factors(scenario: SMIBScenario) -> str:
    factors: list[str] = []
    ratio = scenario.Pmax_fault / scenario.Pmax_pre
    if not 4.0 <= scenario.H <= 6.0:
        factors.append("H")
    if not 0.8 <= scenario.D <= 1.2:
        factors.append("D")
    if not 0.17 <= scenario.t_clear <= 0.21:
        factors.append("t_clear")
    if not 0.10 <= ratio <= 0.25:
        factors.append("fault_ratio")
    return "+".join(factors) if factors else "in_distribution"


def canonical_splits() -> dict[str, list[SMIBScenario]]:
    """Backwards-compatible alias for the published v0.2 baseline."""
    return splits_v02()
