# ANDES v1.0 runtime: routing and reference fallback

This runtime is deliberately narrow: IEEE-14 three-phase faults, 20 GENROU angle/speed/torque outputs, and no operational protection claims.

`load_checkpoint(path, expected_sha256=..., expected_protocol=...)` requires a SHA-256 obtained from an independently trusted source. It verifies the digest **before** loading PyTorch weights with `weights_only=True`, and verifies the protocol, architecture shapes, and calibrated threshold. The existing v0.3 artifact includes a model checkpoint, but users must download it and pin its digest independently.

`run_routed_case` takes a fault, a **precomputed physics residual score**, validation-calibrated threshold, model prediction callback, and reference callback. It executes the model only when both the domain check and threshold pass. Rejected faults execute the reference callback, and reference failures propagate as errors.

**Important boundary:** This entrypoint is not a complete online physics-residual calculation: the score must be computed separately using model predictions and trusted IEEE-14 machine constants (see `andes_evaluation.evaluate_case` and `andes_surrogate.swing_residuals`). Do not use a manufactured score to claim trustworthiness. The source-labelled execution is implemented and tested, but an end-to-end trustworthy live residual evaluation and real-ANDES routing benchmark are outstanding.

Accepted surrogate outputs omit bus voltage. Reference outputs include bus voltage, and both expose the 20 electromechanical target channels. Neither output constitutes a stability or protection safety verdict.
