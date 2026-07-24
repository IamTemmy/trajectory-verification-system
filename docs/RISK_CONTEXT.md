# Prediction risk-context review

Aggregate displacement metrics do not distinguish an isolated forecasting
error from an error occurring near other actors or mapped controls.
`analyze-prediction-risk` adds contextual evidence for engineering review.

## Evidence signals

For every prediction target, the analysis records:

- ground-truth motion class: stationary, straight, or turning;
- other actors within 30 m at the current observation time;
- minimum ground-truth separation from any observed actor in the prediction
  horizon;
- minimum ground-truth separation from the self-driving-car track, when
  available;
- maximum absolute error in predicted versus actual target-to-SDC separation,
  using the minADE-selected mode;
- future proximity to crosswalk polygons and traffic-control points;
- object type, minADE, minFDE, diagnostic miss status, and explicit context
  tags.

## Default review thresholds

| Signal | Default |
|---|---:|
| Close interaction | ≤5 m |
| Dense scene | ≥10 other actors within 30 m |
| Large SDC separation error | ≥5 m |
| Crosswalk context | ≤5 m |
| Traffic-control context | ≤10 m |
| Stationary displacement | ≤2 m |
| Turning angle | ≥0.35 rad |

A case is `high` priority when it is a diagnostic miss and has close,
SDC-separation-error, crosswalk, or traffic-control context. It is `medium`
when either a miss or consequential context is present, and otherwise `low`.

## Interpretation boundary

These categories prioritize recorded dataset cases for review. They do not
estimate collision probability, determine fault, identify unsafe production
behavior, or evaluate the Waymo Driver. Distance thresholds are project-defined
screening assumptions and must not be presented as validated safety limits.

## Commands

The reproducible experiment automatically creates risk reports for every
candidate:

```bash
run-prediction-experiment examples/full_shard_experiment.json
```

Analyze an existing submission independently:

```bash
analyze-prediction-risk predictions.binproto data/raw/SHARD \
  --json-report reports/generated/risk.json \
  --markdown-report reports/generated/risk.md \
  --html-report reports/generated/risk.html \
  --summary-only
```
