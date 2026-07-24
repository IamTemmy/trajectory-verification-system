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

## Verified full-shard context analysis

Milestone 10 was validated on July 24, 2026 from clean Git revision
`7602dc657e4dd9f5b2317696cfff15d396250287`, using the same checksummed WOMD
v1.3.1 shard, 276 scenarios, and 1,203 target agents.

The candidate improved every motion stratum:

| Motion class | Agents | Baseline minADE | Candidate minADE | Baseline minFDE | Candidate minFDE | Baseline miss | Candidate miss |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stationary | 6 | 3.508 m | 2.479 m | 7.135 m | 5.379 m | 50.0% | 50.0% |
| Straight | 739 | 8.728 m | 7.545 m | 21.524 m | 18.921 m | 90.80% | 88.90% |
| Turning | 458 | 11.174 m | 8.093 m | 29.227 m | 21.769 m | 98.25% | 96.29% |

Turning actors remained the hardest group, but received the largest absolute
candidate improvement: 3.080 m in mean minADE and 7.457 m in mean minFDE.

The candidate reduced:

- large SDC-separation-error flags from 908 to 872;
- high-priority review cases from 1,076 to 1,048;
- diagnostic misses by 23 agents overall.

Scene-context counts that depend only on recorded ground truth remained
identical between models, as expected: 697 close-interaction targets, 625
crosswalk-context targets, 491 dense-scene targets, 542 traffic-control-context
targets, and 148 vulnerable-road-user targets.

### Calibration limitation

The default screen labels 1,048 of 1,203 candidate agents as high priority.
This high recall and low selectivity makes it suitable for evidence enrichment
and subsequent filtering, but not as a standalone severity classifier. The
count must not be described as 1,048 dangerous events. Learned ranking,
threshold calibration against human-reviewed cases, and calibrated risk
probabilities remain outside the current evidence.
