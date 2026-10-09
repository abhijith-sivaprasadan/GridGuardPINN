# Frozen results: ANDES network-aware fault-location encoding v0.3

**Source:** [successful pre-registered GitHub Actions run 37922112172](https://github.com/abhijith-sivaprasadan/GridGuardPINN/actions/runs/37922112172), commit `db6a6ab7c5122fbff38e3c676d250a8b52e3535a`, 9 October 2026.

The archived experiment artifact includes 34 files and the six trained-model summaries. The three seeds (17, 29, 41) used a matched one-hot baseline and a 14-dimensional static network-aware descriptor, under the previously frozen v0.3 protocol.

## Across-seed median results

| Metric | One-hot | Network-aware |
|---|---:|---:|
| Unseen-location composite-error median (median across seeds) | 2.9149 | 2.0539 |
| Unseen-location good cases, median per seed (of 21) | 0 | 1 |
| Fresh-ID good cases, every seed (of 7) | 7 | 7 |
| Duration-OOD good cases, every seed (of 7) | 7 | 7 |
| Unseen-location residual-gate false accepts, median per seed | 0 | 4 |

On the composite-error median-of-medians statistic, the network-aware encoding is approximately 29.5% lower than one-hot. This aggregate ratio does not imply every seed improved: only **2/3** seeds had lower unseen-location median, and only **2/3** produced at least one acceptable unseen-location prediction. The new encoding still failed most of the 21 unseen-location cases in every seed.

All four preregistered criteria passed: the two generalisation criteria (2/3 each) and the fresh-ID and duration-OOD non-regression criteria (7/7 in all three seeds). **This is conditional evidence of improved unseen-location generalisation, not proof of robust generalisation.**

## Safety-critical counter-result

Residual-only routing is **not suitable for unseen fault buses**: the network-aware arm had 3–4 false accepts per seed on the unseen-location set (median 4). These are surrogate trajectories that violated the frozen research accuracy screen but would be routed without reference simulation. Location-hard routing rejected every held-out bus by construction, so it had zero false accepts there but also zero coverage, including on accurately predicted held-out cases.

The accuracy screen remains maximum machine rotor-angle RMSE ≤0.05 rad and speed RMSE ≤0.0005 pu; these are exploratory thresholds, **not validated grid-protection or stability assurance thresholds**.

## Interpretation and next work

- Preserve v0.3 as a frozen outcome; do not retroactively tune this evaluation.
- Use an explicit location-and-duration envelope and fail-closed reference fallback in the demonstration path.
- Treat improving trustworthy routing coverage as a separate future experiment, with freshly frozen evaluations.
- Do not claim production, real-time protection, deployable grid safety, or guaranteed generalisation.
