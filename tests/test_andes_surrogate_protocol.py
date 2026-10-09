from gridguardpinn.andes_surrogate_protocol import (
    ROBUST_BUSES,
    assert_protocol_integrity,
    surrogate_splits_v01,
)


def test_surrogate_protocol_is_disjoint_and_frozen():
    assert_protocol_integrity()
    splits = surrogate_splits_v01()
    assert len(splits["train"]) == 21
    assert len(splits["validation"]) == 7
    assert len(splits["test_id"]) == 7
    assert len(splits["ood_duration"]) == 7
    assert len(splits["ood_location"]) == 21


def test_location_ood_uses_unseen_fault_buses():
    splits = surrogate_splits_v01()
    train_buses = {case.fault_bus for case in splits["train"]}
    location_buses = {case.fault_bus for case in splits["ood_location"]}
    assert train_buses == set(ROBUST_BUSES)
    assert train_buses.isdisjoint(location_buses)
