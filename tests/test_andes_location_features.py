import numpy as np
import pytest

pytest.importorskip("andes")

from gridguardpinn.andes_location_features import (
    FEATURE_NAMES,
    build_ieee14_fault_location_features,
)
from gridguardpinn.andes_surrogate import N_BUSES, N_GENERATORS


@pytest.fixture(scope="module")
def features():
    return build_ieee14_fault_location_features()


def test_ieee14_location_feature_manifest(features):
    assert len(features.descriptors) == N_BUSES
    assert len(features.generator_buses) == N_GENERATORS
    assert len(FEATURE_NAMES) == N_BUSES
    assert len(features.rows) == N_BUSES


def test_ieee14_location_descriptors_are_finite_and_same_size(features):
    for descriptor in features.descriptors.values():
        assert descriptor.shape == (N_BUSES,)
        assert np.all(np.isfinite(descriptor))


def test_generator_buses_have_zero_self_distance(features):
    for generator_index, bus in enumerate(features.generator_buses):
        descriptor = features.descriptors[bus]
        assert descriptor[generator_index] == pytest.approx(0.0, abs=1e-12)
        assert descriptor[N_GENERATORS + generator_index] == pytest.approx(
            0.0,
            abs=1e-12,
        )


def test_location_descriptors_are_not_one_hot_or_identical(features):
    rows = np.vstack(
        [features.descriptors[bus] for bus in sorted(features.descriptors)]
    )
    assert np.unique(rows, axis=0).shape[0] == N_BUSES
    assert not np.all(np.isin(rows, [0.0, 1.0]))
