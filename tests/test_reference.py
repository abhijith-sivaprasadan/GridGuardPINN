import numpy as np

from gridguardpinn.dynamics import SMIBScenario
from gridguardpinn.reference import simulate_reference


def test_reference_simulation_is_finite_and_preserves_initial_state():
    scenario = SMIBScenario()
    result = simulate_reference(scenario, samples=301)

    assert result.states.shape == (301, 2)
    assert np.all(np.isfinite(result.states))
    assert np.isclose(result.delta[0], scenario.initial_delta)
    assert np.isclose(result.omega[0], scenario.omega0)


def test_fault_produces_nonzero_speed_response():
    result = simulate_reference(SMIBScenario(), samples=501)
    assert np.max(np.abs(result.omega)) > 1e-4
