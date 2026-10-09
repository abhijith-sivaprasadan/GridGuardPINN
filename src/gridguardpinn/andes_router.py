"""Deterministic, fail-closed ANDES surrogate routing for the evaluated domain.

This module does not certify physical safety. It implements conservative routing
based on the frozen training envelope and a validation-calibrated residual gate.
The caller must run ANDES if the decision is not ACCEPT_SURROGATE.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

TRAIN_BUSES = frozenset((3, 6, 7, 8, 9, 11, 14))
MIN_FAULT_DURATION_S = 0.04
MAX_FAULT_DURATION_S = 0.12


class Route(str, Enum):
    ACCEPT_SURROGATE = "accept_surrogate"
    RUN_REFERENCE = "run_reference"


@dataclass(frozen=True)
class RoutingDecision:
    route: Route
    reason: str

    @property
    def use_surrogate(self) -> bool:
        return self.route == Route.ACCEPT_SURROGATE


def route_andes_case(
    *,
    fault_bus: int,
    fault_duration_s: float,
    residual_score: float,
    residual_threshold: float,
    model_ready: bool = True,
    reference_available: bool = True,
) -> RoutingDecision:
    """Return a conservative routing decision without using ground-truth error.

    Fail closed on unsupported parameters, non-finite inputs, and unready models.
    A reference fallback is only operational if the caller has an available,
    independently verified reference solver; absence is never permission to
    use the surrogate.
    """
    if not reference_available:
        return RoutingDecision(Route.RUN_REFERENCE, "reference_unavailable_stop")
    if not model_ready:
        return RoutingDecision(Route.RUN_REFERENCE, "model_not_ready")
    if type(fault_bus) is not int or fault_bus not in TRAIN_BUSES:
        return RoutingDecision(Route.RUN_REFERENCE, "unseen_fault_bus")
    values = (fault_duration_s, residual_score, residual_threshold)
    if any(type(value) not in (int, float) for value in values):
        return RoutingDecision(Route.RUN_REFERENCE, "invalid_input")
    if not all(math.isfinite(value) for value in values):
        return RoutingDecision(Route.RUN_REFERENCE, "non_finite_input")
    if not MIN_FAULT_DURATION_S <= fault_duration_s <= MAX_FAULT_DURATION_S:
        return RoutingDecision(Route.RUN_REFERENCE, "duration_outside_envelope")
    if residual_threshold < 0 or residual_score < 0:
        return RoutingDecision(Route.RUN_REFERENCE, "invalid_residual")
    if residual_score > residual_threshold:
        return RoutingDecision(Route.RUN_REFERENCE, "residual_above_threshold")
    return RoutingDecision(Route.ACCEPT_SURROGATE, "within_evaluated_envelope")
