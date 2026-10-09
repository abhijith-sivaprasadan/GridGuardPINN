# GridGuardPINN experiment specification v0.4

**Freeze point:** the v0.4 holdout is committed before changing the v0.3 network
capacity, time representation, optimisation budget, or checkpoint-selection
logic.

## Motivation

v0.3 materially improved the surrogate and produced a strong relationship
between physics-residual score and true trajectory error, but only 2/24 fresh
ID cases met the unchanged provisional error tolerance. The combined gate
therefore achieved zero false accepts at only 4.17% coverage.

v0.4 targets surrogate accuracy rather than relaxing the acceptance standard.

## Training and validation

The 24-case training grid and 16-case validation set remain unchanged.

Validation may be used for:

- architecture selection;
- learning-rate/epoch decisions;
- model checkpoint selection;
- trust-gate calibration.

No v0.4 ID-test or OOD-test trajectory result may influence these choices.

## Fresh v0.4 holdout

A new fixed seed (20261010) generates:

- 32 unseen in-distribution interpolation cases;
- 32 one-factor OOD cases, eight each for H, D, clearing time and fault ratio.

The OOD shifts are intentionally closer to the training boundary than v0.3,
making this a harder extrapolation-detection test.

## Planned v0.4 model changes

1. Add fixed Fourier time features to help represent post-fault oscillations.
2. Increase MLP width/depth while keeping the governing model unchanged.
3. Increase event-aware supervised anchors and phase-stratified collocation.
4. Extend optimisation with cosine learning-rate decay.
5. Select the final checkpoint using validation trajectory loss rather than
   blindly taking the last epoch.
6. Keep the same deterministic residual + parameter-OOD trust architecture.

## Unchanged trajectory-quality target

A case remains provisionally acceptable only if:

- rotor-angle RMSE <= 0.05 rad;
- speed-deviation RMSE <= 5e-4 pu.

These remain demonstrator screening tolerances, not industry protection or
transient-stability acceptance standards.

## Success criteria

The primary accuracy target is a material increase in the fraction of fresh
v0.4 ID cases meeting the unchanged tolerance.

The trust layer is useful only if it can then achieve:

- low or zero false accepts;
- materially higher coverage than v0.3's 4.17% fresh-ID coverage.

If accuracy improves but trust coverage does not, both results will be reported.
