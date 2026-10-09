# Result — v0.4 Fourier/event-aware surrogate

Run commit: `4ace5063d7c828fb02d8fe5f74230317ae46f360`  
GitHub Actions artifact: `gridguardpinn-smib-v0.4`  
Seed: 7  
Training budget: 2,500 Adam epochs, 121 supervised anchors/case, 96 physics-collocation points/case

## Why v0.4 exists

v0.3 improved the surrogate substantially but the validation-calibrated gate
accepted only 1/24 fresh ID-test cases. Because the v0.3 holdout had then been
observed, v0.4 froze a new test set before changing the architecture.

The v0.4 holdout contains:

- 32 fresh in-distribution interpolation cases;
- 32 fresh one-factor OOD cases, eight each for inertia, damping,
  fault-clearing time, and fault-on transfer ratio.

The trajectory-quality targets were **not changed**:

- rotor-angle RMSE <= 0.05 rad;
- speed-deviation RMSE <= 5e-4 pu;
- composite error ratio <= 1.

## v0.4 changes

Relative to v0.3, v0.4 adds:

- fixed Fourier time features at 0.5, 1, 2, and 4 Hz;
- a larger 96-wide, 5-hidden-layer MLP;
- 121 event-aware reference anchors per training case;
- 96 phase-stratified physics points per case;
- 2,500-epoch cosine-decay optimisation;
- validation-selected checkpointing.

The best checkpoint occurred at epoch 2,500 with validation scaled data loss
0.00686.

## Accuracy result

| Split | Cases | Good cases | Good-case fraction | Mean composite error | Median | Maximum |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 16 | 12 | 75.0% | 0.795 | 0.715 | 1.379 |
| Fresh ID test | 32 | 23 | 71.9% | 0.904 | 0.891 | 1.524 |
| Fresh OOD test | 32 | 12 | 37.5% | 48.07 | 1.162 | 1458.50 |

The OOD mean is dominated by a small number of extreme failures, especially
clearing-time extrapolation; the median is therefore more informative than the
mean for the OOD set.

### Improvement over earlier versions

Fresh ID mean composite error:

- v0.2: ~8.20;
- v0.3: ~1.81;
- v0.4: **0.904**.

Fresh-ID good cases:

- v0.2: 0/16;
- v0.3: 2/24;
- v0.4: **23/32**.

The v0.4 surrogate is therefore the first version that meets the stated
trajectory tolerance on a majority of unseen in-distribution cases.

## Trust signal

Residual/error Spearman correlation:

- validation: 0.682;
- fresh ID test: 0.725;
- fresh OOD test: **0.939**.

The residual signal remains useful, especially under extrapolation, although
the association is weaker on in-distribution cases than in v0.3.

The validation-calibrated thresholds were:

- residual score <= 0.14326;
- parameter-space OOD score <= 2.07814.

## Fresh ID routing result

Without a gate, accepting every surrogate case gives:

- 32/32 accepted;
- 9 false accepts;
- false-accept rate among accepted cases: **28.1%**.

The combined residual + OOD gate gives:

- **21/32 accepted (65.6% coverage)**;
- **1 false accept**;
- false-accept rate among accepted cases: **4.76%**;
- 3 false escalations;
- mean composite error among accepted cases: 0.799.

This is the first GridGuardPINN result with both meaningful surrogate coverage
and a large reduction in unsafe acceptance. It is not perfect: the single false
accept means the gate cannot yet be described as fail-safe.

## Fresh OOD routing result

Of 32 fresh OOD cases:

- 12 actually meet the provisional trajectory tolerance;
- always-accept produces 20 false accepts;
- OOD-only routing accepts 19 cases and still produces 9 false accepts;
- residual-only routing accepts 8 cases with 1 false accept;
- combined routing accepts 6 cases with 1 false accept.

Combined OOD coverage is therefore 18.75%, with a 16.7% false-accept rate among
accepted OOD cases.

This is not strong enough to claim reliable extrapolative routing.

## OOD mechanism diagnosis

Always-accept mean composite error by isolated OOD factor:

| OOD factor | Cases | Mean composite error | Combined accepted | Combined false accepts |
|---|---:|---:|---:|---:|
| Damping D | 8 | 1.02 | 2 | 1 |
| Inertia H | 8 | 3.18 | 1 | 0 |
| Fault ratio | 8 | 0.84 | 3 | 0 |
| Clearing time | 8 | 187.24 | 0 | 0 |

Clearing-time extrapolation is the dominant hazard: seven of eight cases violate
the accuracy threshold and extreme cases can diverge catastrophically. The
combined gate rejects all eight clearing-time OOD cases.

The remaining false accept occurs in the damping-shift subset. That is the
clearest next trust-layer weakness.

## What v0.4 supports

The frozen v0.4 experiment supports the following claims:

1. Event-aware + Fourier-feature training can produce a useful parametric SMIB
   PINN over a stated interpolation region.
2. 23/32 fresh ID cases met the unchanged error target.
3. A validation-calibrated deterministic gate reduced ID false accepts from
   9/32 under always-accept to 1/32 while retaining 65.6% surrogate coverage.
4. Physics-residual magnitude remains strongly associated with OOD trajectory
   error.
5. Parameter-space OOD detection alone is not sufficient for safe routing.

## What v0.4 does not support

It does **not** establish:

- zero-false-accept or fail-safe behaviour;
- reliable acceptance of extrapolative/OOD cases;
- transfer from SMIB dynamics to multi-machine grids;
- equivalence to a commercial transient-stability package;
- operational, protection-grade, or planning-grade validity.

## Decision after v0.4

The SMIB phase has now produced a substantial positive result and a clear
remaining limitation. Further tuning on the same reduced-order system has
diminishing portfolio value.

The next project milestone is therefore to preserve the SMIB benchmark as a
regression suite and move the reference side to a standard multi-machine
transient-stability simulator. The planned next integration is ANDES (or an
equivalent suitable open-source simulator), with the same principle:

> fast learned surrogate -> explicit trust signals -> deterministic gate ->
> fallback to the reference simulator.
