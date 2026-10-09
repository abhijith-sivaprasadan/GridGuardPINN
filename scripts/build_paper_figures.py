"""Rebuild source-grounded vector figures for the GridGuardPINN working paper.

Values below are transcribed from the frozen, linked result documents.
Do not use this script to claim a new data analysis or independent validation.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT = Path("docs/figures")
OUTPUT.mkdir(parents=True, exist_ok=True)

seeds = ["17", "29", "41"]
speed = [1.755102, 1.954452, 1.567609]
lo = [1.429986, 1.867136, 1.240832]
hi = [2.083866, 2.034647, 2.021835]

fig, ax = plt.subplots(figsize=(7.8, 3.6))
y = range(3)
ax.errorbar(speed, y, xerr=[[v-a for v,a in zip(speed,lo)],
                             [a-v for v,a in zip(speed,hi)]],
            fmt="o", capsize=4, color="#234f7d")
ax.axvline(1.0, color="#b35035", linestyle="--", label="No speedup")
ax.set_yticks(list(y), [f"Seed {s}" for s in seeds])
ax.invert_yaxis()
ax.set_xlabel("Complete route speedup (ANDES-only / hybrid); 95% bus-cluster CI")
ax.set_title("Randomized paired benchmark · 50/50 eligible and duration-OOD")
ax.set_xlim(0.9, 2.25)
fig.tight_layout()
fig.savefig(OUTPUT / "paired_speedup_v1.svg", format="svg")
plt.close(fig)

data = [0.1719, 0.1768, 0.1767]
physics = [0.1515, 0.1477, 0.1417]
fig, ax = plt.subplots(figsize=(7.8, 3.6))
x = [0,1,2]
ax.bar([v-.18 for v in x], data, width=.36, label="Data-only")
ax.bar([v+.18 for v in x], physics, width=.36, label="Physics-informed")
ax.set_xticks(x, [f"Seed {s}" for s in seeds])
ax.set_ylabel("Fresh-ID mean composite error (lower better)")
ax.set_title("Frozen matched ANDES physics-loss ablation (v0.2)")
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT / "physics_ablation_v1.svg", format="svg")
plt.close(fig)
print("Saved two source-grounded SVG figures.")
