"""GridGuardPINN core package."""

from .dynamics import SMIBScenario, electrical_power, state_derivative
from .reference import SimulationResult, simulate_reference

__all__ = [
    "SMIBScenario",
    "SimulationResult",
    "electrical_power",
    "simulate_reference",
    "state_derivative",
]

__version__ = "0.1.0"
