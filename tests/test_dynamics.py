import numpy as np

from gridguardpinn.dynamics import SMIBScenario, state_derivative


def test_prefault_equilibrium_has_zero_derivative():
    scenario = SMIBScenario()
    derivative = state_derivative(0.0, scenario.initial_state, scenario)
    assert np.allclose(derivative, [0.0, 0.0], atol=1e-12)


def test_fault_application_accelerates_rotor_from_equilibrium():
    scenario = SMIBScenario()
    derivative = state_derivative(
        scenario.t_fault + 1e-6,
        scenario.initial_state,
        scenario,
    )
    assert derivative[1] > 0.0


def test_invalid_event_order_is_rejected():
    try:
        SMIBScenario(t_fault=0.2, t_clear=0.1)
    except ValueError:
        return
    raise AssertionError("Invalid event ordering should raise ValueError")
