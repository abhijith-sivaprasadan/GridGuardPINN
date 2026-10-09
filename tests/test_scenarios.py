import numpy as np

from gridguardpinn.scenarios import (
    canonical_splits,
    scenario_vector,
    splits_v03,
)
from gridguardpinn.trust import MahalanobisOOD


def test_canonical_split_sizes_and_ood_distance():
    splits = canonical_splits()
    assert len(splits["train"]) == 24
    assert len(splits["validation"]) == 16
    assert len(splits["test_id"]) == 16
    assert len(splits["ood"]) == 16

    train_x = np.vstack([scenario_vector(case) for case in splits["train"]])
    validation_x = np.vstack([scenario_vector(case) for case in splits["validation"]])
    ood_x = np.vstack([scenario_vector(case) for case in splits["ood"]])

    detector = MahalanobisOOD().fit(train_x)
    assert np.median(detector.score(ood_x)) > np.median(detector.score(validation_x))


def test_v03_fresh_holdout_sizes():
    splits = splits_v03()
    assert len(splits["train"]) == 24
    assert len(splits["validation"]) == 16
    assert len(splits["test_id"]) == 24
    assert len(splits["ood"]) == 24


def test_v04_fresh_holdout_sizes():
    splits = splits_v04()
    assert len(splits["train"]) == 24
    assert len(splits["validation"]) == 16
    assert len(splits["test_id"]) == 32
    assert len(splits["ood"]) == 32
