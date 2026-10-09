"""Frozen case definitions for the first ANDES multi-machine surrogate."""

from __future__ import annotations

from dataclasses import dataclass

ROBUST_BUSES = (3, 6, 7, 8, 9, 11, 14)
TRAIN_DURATIONS_S = (0.04, 0.08, 0.12)
VALIDATION_DURATION_S = 0.06
ID_TEST_DURATION_S = 0.10
DURATION_OOD_S = 0.14

EDGE_SUCCESS_CASES = (
    (1, 0.04), (1, 0.06), (1, 0.08), (1, 0.10),
    (2, 0.04), (2, 0.06), (2, 0.08), (2, 0.10),
    (4, 0.04), (4, 0.06), (4, 0.08), (4, 0.12), (4, 0.14),
    (5, 0.04), (5, 0.06), (5, 0.08),
    (10, 0.06), (10, 0.08), (10, 0.10),
    (13, 0.12), (13, 0.14),
)


@dataclass(frozen=True, order=True)
class AndesFaultCase:
    fault_bus: int
    fault_duration_s: float


def surrogate_splits_v01() -> dict[str, list[AndesFaultCase]]:
    train = [
        AndesFaultCase(bus, duration)
        for bus in ROBUST_BUSES
        for duration in TRAIN_DURATIONS_S
    ]
    validation = [
        AndesFaultCase(bus, VALIDATION_DURATION_S) for bus in ROBUST_BUSES
    ]
    test_id = [AndesFaultCase(bus, ID_TEST_DURATION_S) for bus in ROBUST_BUSES]
    ood_duration = [AndesFaultCase(bus, DURATION_OOD_S) for bus in ROBUST_BUSES]
    ood_location = [AndesFaultCase(bus, duration) for bus, duration in EDGE_SUCCESS_CASES]
    return {
        "train": train,
        "validation": validation,
        "test_id": test_id,
        "ood_duration": ood_duration,
        "ood_location": ood_location,
    }


def assert_protocol_integrity() -> None:
    splits = surrogate_splits_v01()
    expected = {
        "train": 21,
        "validation": 7,
        "test_id": 7,
        "ood_duration": 7,
        "ood_location": 21,
    }
    assert {name: len(cases) for name, cases in splits.items()} == expected

    all_cases = [case for cases in splits.values() for case in cases]
    if len(all_cases) != len(set(all_cases)):
        raise AssertionError("Surrogate protocol contains duplicate cases.")

    seen_buses = set(ROBUST_BUSES)
    location_buses = {case.fault_bus for case in splits["ood_location"]}
    if seen_buses & location_buses:
        raise AssertionError("Location-OOD buses must be unseen during training.")

    if not all(
        case.fault_duration_s > max(TRAIN_DURATIONS_S)
        for case in splits["ood_duration"]
    ):
        raise AssertionError("Duration-OOD cases must extrapolate beyond training.")
