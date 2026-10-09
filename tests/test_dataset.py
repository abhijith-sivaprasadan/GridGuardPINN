import numpy as np

from gridguardpinn.dataset import build_collocation_points, build_supervised_dataset
from gridguardpinn.scenarios import canonical_splits


def test_dataset_shapes_and_event_exclusion():
    scenarios = canonical_splits()["train"][:2]
    data = build_supervised_dataset(scenarios, samples_per_case=15)
    assert data.x.shape == (30, 5)
    assert data.y.shape == (30, 2)
    assert np.all(np.isfinite(data.y))

    points = build_collocation_points(
        scenarios, points_per_case=14, event_exclusion_s=0.01, seed=3
    )
    assert points.shape == (28, 5)

    for scenario in scenarios:
        selector = (
            (points[:, 1] == scenario.H)
            & (points[:, 2] == scenario.D)
            & (points[:, 3] == scenario.t_clear)
        )
        rows = points[selector]
        assert np.all(np.abs(rows[:, 0] - scenario.t_fault) > 0.01)
        assert np.all(np.abs(rows[:, 0] - scenario.t_clear) > 0.01)
