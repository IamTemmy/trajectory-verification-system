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
  --json-report reports/generated/prediction-comparison.json \
  --markdown-report reports/generated/prediction-comparison.md
```

The strict example policy permits no increase in mean minADE, mean minFDE, or
miss rate and no decrease in ground-truth coverage. Tolerances can be configured
explicitly for noisy or larger evaluation sets.

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
