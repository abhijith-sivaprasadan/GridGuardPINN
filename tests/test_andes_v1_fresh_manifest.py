from runpy import run_path

import pytest


def test_fresh_duration_manifest_unchanged():
    ns = run_path("scripts/run_andes_v1_fresh_duration.py")
    cases = ns["SPLITS"]
    assert len(cases["fresh_interpolation"]) == 28
    assert len(cases["new_duration_ood"]) == 7
    assert len(set(cases["fresh_interpolation"] + cases["new_duration_ood"])) == 35
    assert {duration for _, duration in cases["fresh_interpolation"]} == {
        0.05, 0.07, 0.09, 0.11
    }
    assert {duration for _, duration in cases["new_duration_ood"]} == {0.13}
