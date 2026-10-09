# Pre-registered network-aware fault-location encoding experiment v0.3

**Freeze point:** committed after recording the v0.2 physics ablation and before
training or evaluating any network-aware location model.

## Research question

The v0.1/v0.2 surrogate uses a 14-way one-hot fault-bus vector. When a bus is
held out from training, its input weight is never trained, so the model has no
representation of that bus's physical relationship to buses seen during
training.

Does replacing that categorical vector with a same-dimensional,
physics/network-aware descriptor improve generalisation to unseen fault
locations without materially degrading trained-location accuracy?

## Frozen comparison

Both arms use the same:

- ANDES 2.0.0 IEEE-14 dynamic reference system;
- 20-output surrogate architecture;
- five 96-unit tanh hidden layers;
- event/Fourier time encoding;
- 21 train / 7 validation / 7 fresh-ID / 7 duration-OOD / 21
  unseen-location split from v0.1;
- 101 supervised anchors per training case;
- 56 physics collocation points per training case;
- 1000 epochs and the frozen optimiser schedule;
- physics weight 0.02 with the frozen warm-up/ramp;
- output scaling and checkpoint rule;
- accuracy screen;
- seeds **17, 29, 41**.

The only model-input difference is the 14-dimensional fault-location
representation.

### Arm A — categorical baseline

The existing 14-way one-hot bus vector.

### Arm B — network-aware descriptor

Fourteen static features computed from the **known base-case network only**:

1. five normalized shortest-path distances from the fault bus to each of the
   five dynamic GENROU generator buses, with active-branch impedance magnitude
   `sqrt(r^2 + x^2)` as edge weight;
2. five normalized unweighted hop distances from the fault bus to those same
   generator buses;
3. scaled pre-fault voltage magnitude at the fault bus;
4. sine of pre-fault bus angle;
5. cosine of pre-fault bus angle;
6. normalized active-branch degree of the fault bus.

Distances and degree are normalized with constants derived from the complete
known IEEE-14 topology. No transient trajectory, held-out target, trajectory
error, residual score, or test-set outcome is used to construct or normalize
the descriptor.

The descriptor has exactly the same dimensionality as the one-hot vector, so
the neural architecture and parameter count remain unchanged.

## Primary endpoint

Unseen-location edge-set trajectory accuracy.

Report for each seed and arm:

- good cases out of 21;
- composite-error mean, median, and maximum;
- max-machine rotor-angle RMSE;
- max-machine speed RMSE.

Evidence that the network-aware encoding adds location generalisation requires:

1. a lower unseen-location composite median in at least **2 of 3 seeds**; and
2. at least one unseen-location case passing the frozen trajectory screen in
   at least **2 of 3 seeds**.

A weaker result may still be scientifically useful but will not be described as
successful unseen-location generalisation.

## Non-regression endpoints

For validation, fresh ID, and duration OOD, report the same trajectory metrics.

A material regression is defined in advance as either:

- any seed losing the existing **7/7** fresh-ID good-case result; or
- any seed losing the existing **7/7** duration-OOD good-case result.

## Trust evaluation

The existing validation-only residual calibration is retained unchanged.

Because unseen bus identity is the variable under study, the primary trust
comparison on the unseen-location set is **residual-only**. The existing
location-hard rejection is still reported as a conservative safety baseline but
cannot demonstrate learned location generalisation because it rejects every
held-out bus by construction.

Report:

- residual/error Spearman correlation;
- residual-only coverage / false accepts / false escalations;
- location-hard and strict gate metrics as safety comparators.

No residual threshold may be retuned on held-out location cases.

## Compute and timing

Report:

- training wall time;
- 401-point surrogate forward time;
- corresponding ANDES solve time.

## Interpretation constraints

Do not change feature definitions, normalization, splits, architecture,
training schedule, accuracy screen, or seeds after inspecting v0.3 held-out
results.

If the graph-aware arm fails, record that result. Do not add extra static
features or tune their scaling until the v0.3 result has been frozen in a
separate result document.
