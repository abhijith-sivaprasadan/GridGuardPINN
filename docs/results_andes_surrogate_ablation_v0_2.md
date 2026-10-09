# ANDES multi-machine physics ablation result v0.2

Run: GitHub Actions `ANDES multi-machine physics ablation v0.2`  
Workflow run: **37912584795**  
Experiment commit: **c9d9d31932a9db369c27ee3785dccaac35230609**  
Pre-registration: `docs/andes_surrogate_ablation_plan_v0_2.md`  
Seeds: **17, 29, 41**  
Reference: ANDES 2.0.0 IEEE-14 dynamic case with programmatic three-phase faults

This result follows the ablation plan committed before the held-out v0.1 result
was inspected. The architecture, frozen splits, optimisation schedule, output
scaling, checkpoint rule, and accuracy screen are identical between arms. The
only planned training difference is the swing-equation physics-loss weight:

- **data-only:** 0.0;
- **physics-informed:** 0.02 after the frozen warm-up/ramp.

## Reference generation

The 63 unique successful ANDES cases were generated once and shared by all six
model fits.

- pure ANDES solve time: **57.49 s** total;
- complete reference pipeline: **57.51 s**.

The frozen research-screening criterion remained:

- max-machine rotor-angle RMSE <= 0.05 rad;
- max-machine speed RMSE <= 5e-4 pu.

## Accuracy outcome

Every seed in both arms passed the screen on all validation, fresh-ID, and
duration-OOD cases:

- validation: **7/7 good for all 6 models**;
- fresh ID (0.10 s): **7/7 good for all 6 models**;
- duration OOD (0.14 s): **7/7 good for all 6 models**;
- unseen-location edge set: **0/21 good for all 6 models**.

The current one-hot fault-bus representation therefore remains the dominant
location-generalisation limitation.

### Paired error comparison

Composite error is the larger of normalized angle RMSE and normalized speed
RMSE relative to the frozen quality screen.

| Seed | Fresh-ID mean: data | Fresh-ID mean: physics | Change | Duration-OOD mean: data | Duration-OOD mean: physics | Change |
|---:|---:|---:|---:|---:|---:|---:|
| 17 | 0.1719 | 0.1515 | **-11.9%** | 0.4166 | 0.4198 | +0.8% |
| 29 | 0.1768 | 0.1477 | **-16.5%** | 0.4223 | 0.4416 | +4.6% |
| 41 | 0.1767 | 0.1417 | **-19.8%** | 0.4766 | 0.4316 | **-9.4%** |

Fresh-ID mean error improves in the physics-informed arm for **all three
seeds**. The paired median improvement in fresh-ID mean error is **16.5%**.

Duration extrapolation is mixed by mean error. Physics-informed training
improves the median case error in all three seeds, but the mean is slightly
worse for seeds 17 and 29 because of tail cases and better for seed 41.
Therefore the experiment does **not** support a robust claim that the current
physics term improves 0.14 s duration extrapolation.

Across seeds, the median-of-seed fresh-ID composite means were:

- data-only: **0.1767**;
- physics-informed: **0.1477**.

For duration OOD they were:

- data-only: **0.4223**;
- physics-informed: **0.4316**.

## Residual diagnostic

Physics-informed training substantially reduces the absolute swing-residual
scale, as expected. Median case-residual scores across the three seeds were
approximately:

| Split | Data-only | Physics-informed |
|---|---:|---:|
| Validation | 1.025 | 0.238 |
| Fresh ID | 1.235 | 0.245 |
| Duration OOD | 1.735 | 0.393 |
| Unseen location | 2.948 | 1.864 |

Lower residual magnitude by itself is not evidence of better trust
discrimination, because each arm calibrates its own threshold on validation.

Residual/error Spearman association was variable rather than uniformly improved
by physics-informed training. For fresh ID, the median across seeds was about
**0.821** for data-only and **0.786** for physics-informed. For duration OOD it
was about **0.857** and **0.643**, respectively. Individual seeds varied
substantially.

The ablation therefore does **not** support the claim that adding the current
physics loss consistently makes residual magnitude a better error-ranking
signal.

## Trust-gate outcome

For duration OOD, the table below uses the **residual-only / location-hard**
routing logic rather than `combined_strict`, because `combined_strict`
hard-rejects duration extrapolation by definition and therefore cannot test
whether the residual itself is useful.

### Fresh-ID accepted cases

| Arm | Seed 17 | Seed 29 | Seed 41 | False accepts |
|---|---:|---:|---:|---:|
| Data-only | 3/7 | 2/7 | 3/7 | 0 |
| Physics-informed | 4/7 | 6/7 | 3/7 | 0 |

All fresh-ID cases are accurate, so this comparison measures unnecessary
escalation rather than dangerous acceptance. Physics-informed training improves
coverage in two seeds and ties in one, but the variability is too large for a
strong general claim from three seeds.

### Duration-OOD accepted cases

| Arm | Seed 17 | Seed 29 | Seed 41 | False accepts |
|---|---:|---:|---:|---:|
| Data-only | 1/7 | 0/7 | 0/7 | 0 |
| Physics-informed | 0/7 | 0/7 | 0/7 | 0 |

All 21 duration-OOD cases per arm are actually within the frozen trajectory
tolerance. The validation-calibrated residual gate therefore remains
**strongly over-conservative under duration shift**. Physics-informed training
does not fix that problem.

### Unseen-location cases

Every unseen-location case is inaccurate in both arms and residual gating
rejects all 21 cases for every seed. Explicit location-OOD rejection remains
the correct deterministic safeguard for the current categorical encoding.

## Compute cost

Median model-training wall time across seeds:

- data-only: **7.21 s**;
- physics-informed: **44.55 s**;
- physics-informed/data-only training-time ratio: **6.18x**.

The architecture is identical at inference. Median 401-point neural forward
times during the ablation were about **0.83-0.85 ms**, while split-level median
ANDES case solves were roughly **0.6-0.9 s**. The measured reference-solve /
forward-pass ratio was generally about **800-1000x** on the GitHub Actions CPU
runner.

These are research timing measurements, not production service benchmarks.

## Pre-registered interpretation

The pre-registration required a useful physics contribution to improve at least
one of fresh-ID accuracy, duration-extrapolation accuracy, residual/error
association, or safe-gate coverage without a material regression elsewhere.

The observed result supports a **modest physics contribution to fresh-ID
trajectory accuracy**:

- improvement is present for all three fixed seeds;
- both arms already clear the accuracy screen, so physics improves precision
  rather than changing pass/fail success;
- duration-OOD mean-error benefit is not consistent;
- residual/error association is not consistently better;
- duration-shift gate coverage is not improved;
- training is materially more expensive, while inference cost is unchanged.

The appropriate conclusion is therefore:

> **Swing-equation regularisation improves trained-location interpolation
> accuracy in this frozen IEEE-14 experiment, but the present residual is not a
> sufficient trust score for duration shift and the categorical fault-location
> representation does not generalise to unseen buses.**

This is a positive but bounded result, not evidence that the physics-informed
arm dominates the data-only baseline.

## Next research question

The highest-value limitation is now architectural rather than optimisation
related: an unseen one-hot bus category has no learned physical relationship to
the buses seen in training.

The next experiment should therefore be frozen before execution and replace
the categorical location representation with physically meaningful,
network-aware descriptors such as generator-relative electrical/topological
distance and pre-fault bus/network features. A separate trust experiment should
test severity-conditioned or normalized residual calibration rather than
relaxing the existing v0.1 threshold after seeing test data.
