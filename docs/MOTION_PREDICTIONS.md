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
