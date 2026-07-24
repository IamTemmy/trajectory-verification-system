# Motion-prediction evaluation

## Supported input

`evaluate-motion-predictions` reads a serialized official WOMD
`MotionChallengeSubmission` containing single-object motion predictions. The
adapter consumes only the documented fields required for evaluation and remains
wire-compatible without installing TensorFlow or the full Waymo SDK.

For each prediction:

- `scenario_id` must match a supplied WOMD scenario;
- `object_id` must be the track object ID, not the track index;
- predicted agents must exactly match `tracks_to_predict`;
- at most the first six modes are evaluated;
- every mode must contain 16 x/y points;
- points align to WOMD scenario steps 15, 20, 25, …, 90.

Invalid future ground-truth states are excluded from displacement metrics and
their coverage is reported explicitly. An agent with no valid aligned future
state fails evaluation. Duplicate scenarios, mismatched agents, and malformed
horizons also fail explicitly.

## Diagnostic metrics

The local evaluator reports:

- minimum average displacement error (minADE) across modes;
- minimum final displacement error (minFDE) across modes;
- a project-defined miss indicator based on minimum final displacement error;
- valid future-ground-truth coverage;
- per-agent, per-scenario, and batch aggregates.

minADE and minFDE are minimized independently. The default miss threshold is
2.0 metres and can be changed with `--miss-threshold-m`.

These diagnostics intentionally do not claim numerical equivalence with Waymo's
official challenge evaluation. Official scoring applies additional
configuration, including object-type and motion-dependent thresholds, and
reports metrics such as mAP and overlap rate.

## Command

Generate a transparent constant-velocity baseline using only each target's
state and velocity at or before `current_time_index`:

```bash
generate-womd-baseline \
  reports/generated/constant_velocity.binproto \
  data/raw/validation-00007-of-00150
```

Use `--limit N` for a quick pipeline check. The generator does not inspect
future target states when forecasting; future timestamps are taken from the
scenario timeline solely to preserve alignment.

Available transparent models are:

- `constant_velocity`;
- `constant_acceleration`, with acceleration magnitude capped at 4 m/s²;
- `constant_turn_rate`, with yaw rate capped at 0.5 rad/s;
- `kinematic_ensemble`, which emits all three as equally weighted modes.

Select one with `--model`. All models use only states at or before
`current_time_index`.

Then evaluate the generated or model-produced predictions:

```bash
evaluate-motion-predictions \
  predictions.binproto \
  data/raw/validation-00007-of-00150 \
  --miss-threshold-m 2.0 \
  --json-report reports/generated/predictions.json \
  --markdown-report reports/generated/predictions.md \
  --html-report reports/generated/predictions.html
```

Multiple matching ground-truth shards may be supplied. Dataset files and
generated reports remain outside version control.

## Candidate regression gate

Compare two evaluation JSON reports produced over exactly the same scenario and
agent population:

```bash
compare-prediction-evaluations \
  reports/generated/prediction-evaluation.json \
  reports/generated/ensemble-evaluation.json \
  --policy examples/prediction_comparison_policy.json \
  --bootstrap-samples 5000 \
  --bootstrap-seed 0 \
  --json-report reports/generated/prediction-comparison.json \
  --markdown-report reports/generated/prediction-comparison.md \
  --html-report reports/generated/prediction-comparison.html
```

The strict example policy permits no increase in mean minADE, mean minFDE, or
miss rate and no decrease in ground-truth coverage. It also requires the upper
bounds of the paired 95% agent-bootstrap intervals for minADE and minFDE deltas
to remain below zero. Tolerances and significance requirements can be configured
explicitly.

Comparison reports preserve the aggregate gate and add paired uncertainty,
improved/unchanged/regressed agent counts, and the ten most improved and
regressed scenarios and agents. Ranking is based on candidate-minus-baseline
minADE, so negative values indicate improvement. Use `--summary-only` to keep
terminal output compact while retaining complete evidence in report files.

## Verified real-data baseline

The end-to-end pipeline was validated on July 24, 2026 against the first three
scenarios in WOMD v1.3.1 validation shard
`uncompressed_scenario_validation_validation.tfrecord-00007-of-00150`.
The constant-velocity baseline generated an official-format serialized
submission and the evaluator successfully decoded, aligned, scored, and
reported:

- 3 scenarios and 12 prediction-target agents;
- mean minADE: 13.859 m;
- mean minFDE: 36.942 m;
- project-defined 2 m miss rate: 100%;
- mean valid future-ground-truth coverage: 90.625%.

Coverage varied from 68.75% to 100% because some target tracks did not contain
valid labels for the full prediction horizon. Those missing states were excluded
and retained as explicit coverage evidence.

The poor displacement results are expected for a single-mode constant-velocity
forecast over an eight-second horizon and establish a deliberately weak,
transparent baseline for future model comparisons. They do not indicate a
pipeline failure and are not official Waymo challenge scores.

## Verified kinematic-ensemble comparison

Milestone 6 was validated on July 24, 2026 using the same three scenarios and
12 target agents as the constant-velocity run. The strict zero-regression policy
passed:

| Metric | Constant velocity | Kinematic ensemble | Change |
|---|---:|---:|---:|
| Mean minADE | 13.859 m | 7.129 m | -6.730 m (-48.6%) |
| Mean minFDE | 36.942 m | 19.764 m | -17.178 m (-46.5%) |
| 2 m diagnostic miss rate | 100% | 91.67% | -8.33 percentage points |
| Valid ground-truth coverage | 90.625% | 90.625% | unchanged |

Best-minADE mode selection used constant velocity for 2 agents, capped constant
acceleration for 6 agents, and capped constant turn rate for 4 agents. This
shows that each transparent motion hypothesis contributed to the evaluated
population. Agent `339` moved below the project-defined miss threshold, with
1.024 m minADE and 1.875 m minFDE.

The comparison preserved identical scenario and agent identities and reported
no policy violations. Results remain local diagnostics rather than official
Waymo challenge scores.

## Full-shard analysis

Batch reports include deterministic agent-level bootstrap estimates using 1,000
resamples and seed 0 by default. Configure these with `--bootstrap-samples` and
`--bootstrap-seed`; use zero samples to disable intervals.

Reports also include:

- metrics grouped by normalized object type;
- best-mode index contribution counts;
- the ten worst agents ranked by minADE;
- scenarios ranked from highest to lowest mean minADE;
- explicit valid-label coverage.

The normalized scenario stores the source timeline separately from track
histories. Prediction horizons therefore remain correct even when no individual
track contains all 91 valid WOMD states.

Use `--summary-only` for full-shard runs to keep terminal output compact. Full
scenario and agent evidence remains available in the requested report files.

## Verified full-shard benchmark

Milestone 7 was validated on July 24, 2026 over every record in validation shard
`uncompressed_scenario_validation_validation.tfrecord-00007-of-00150`: 276
scenarios and 1,203 prediction-target agents. The strict comparison gate passed
without changing the evaluated population or 93.132% valid-label coverage.

| Metric | Constant velocity | Kinematic ensemble | Change |
|---|---:|---:|---:|
| Mean minADE | 9.633 m | 7.728 m | -1.905 m (-19.8%) |
| Mean minFDE | 24.385 m | 19.938 m | -4.447 m (-18.2%) |
| 2 m diagnostic miss rate | 93.43% | 91.52% | -1.91 percentage points |
| Valid ground-truth coverage | 93.132% | 93.132% | unchanged |

The ensemble improved mean displacement metrics for every object type:

| Object type | Agents | Baseline minADE | Ensemble minADE | Baseline minFDE | Ensemble minFDE |
|---|---:|---:|---:|---:|---:|
| Vehicle | 1,055 | 10.687 m | 8.548 m | 27.131 m | 22.142 m |
| Pedestrian | 126 | 1.609 m | 1.403 m | 3.561 m | 3.061 m |
| Cyclist | 22 | 5.079 m | 4.673 m | 11.968 m | 10.887 m |

Best-minADE mode selection used constant velocity for 501 agents (41.6%),
capped constant acceleration for 205 (17.0%), and capped constant turn rate for
497 (41.3%). The broad contribution pattern supports retaining all three
transparent modes.

Deterministic 95% agent-bootstrap intervals were:

- ensemble mean minADE: 7.393–8.055 m;
- ensemble mean minFDE: 19.072–20.761 m;
- ensemble miss rate: 89.94%–93.10%.

These intervals describe sampling uncertainty within this shard. A paired
bootstrap of per-agent candidate-minus-baseline deltas is the appropriate next
step for directly quantifying improvement uncertainty.

## Verified paired full-shard evidence

Milestone 8 was validated on July 24, 2026 using the same 276 scenarios and
1,203 paired prediction-target agents. The significance-aware gate passed with
no policy violations.

| Paired candidate-minus-baseline delta | Estimate | 95% paired interval |
|---|---:|---:|
| Mean minADE | -1.905 m | [-2.120, -1.692] m |
| Mean minFDE | -4.447 m | [-4.966, -3.918] m |
| Diagnostic miss rate | -1.91 percentage points | [-2.66, -1.16] points |

Every paired interval remained below zero, supporting improvement rather than
an unpaired difference caused by population composition. Ground-truth coverage
was unchanged.

The agent-level audit found:

- minADE improved for 702 agents, was unchanged for 501, and regressed for 0;
- minFDE improved for 676 agents, was unchanged for 527, and regressed for 0;
- miss status improved for 23 agents, was unchanged for 1,180, and regressed
  for 0.

The large unchanged population is expected because the ensemble includes the
constant-velocity baseline as one of its modes. Under independently minimized
minADE and minFDE, adding modes cannot worsen those diagnostics for an
identically aligned agent. This is evidence that the comparison and mode
selection are internally consistent; it is not a claim that the kinematic
ensemble is a competitive learned forecasting model.
