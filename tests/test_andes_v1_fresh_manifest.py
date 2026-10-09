import ast
from pathlib import Path

from gridguardpinn.andes_surrogate_protocol import ROBUST_BUSES, surrogate_splits_v01


def test_fresh_duration_manifest_unchanged():
    tree = ast.parse(Path("scripts/run_andes_v1_fresh_duration.py").read_text())
    values = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {"INTERPOLATION_DURATIONS", "STRESS_DURATIONS"}
    }
    assert values["INTERPOLATION_DURATIONS"] == (0.05, 0.07, 0.09, 0.11)
    assert values["STRESS_DURATIONS"] == (0.13,)
    old = {(c.fault_bus, c.fault_duration_s)
           for cases in surrogate_splits_v01().values() for c in cases}
    new = {(bus, duration) for bus in ROBUST_BUSES
           for duration in values["INTERPOLATION_DURATIONS"] + values["STRESS_DURATIONS"]}
    assert len(new) == 35
    assert not (new & old)
