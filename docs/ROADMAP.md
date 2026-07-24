# Roadmap

## Milestone 0 — Verification kernel

- [x] Normalized scenario model
- [x] Kinematic and pairwise metrics
- [x] Declarative scalar requirements
- [x] Failure-interval localization
- [x] Structured JSON output
- [x] Deterministic tests and CI

## Milestone 1 — WOMD ingestion

- [x] Document official schema and isolate platform-dependent package loading
- [x] Read uncompressed TFRecord framing without TensorFlow
- [x] Normalize agent histories and scenario identifiers
- [x] Add fixture-based adapter tests
- [x] Validate against an authenticated real WOMD scenario shard
- [x] Visualize a normalized decoded scenario as standalone SVG

## Milestone 2 — Engineering evidence

- [x] Explain failure intervals using contributing signals
- [x] Add threshold sensitivity sweeps
- [x] Add confidence and data-quality annotations
- [x] Generate Markdown and HTML validation reports

## Milestone 3 — Map-aware requirements

- [x] Lane association and lateral offset
- [x] Stop-line and traffic-signal compliance
- [x] Crosswalk and vulnerable-road-user proximity
- [x] Explicit applicability conditions

## Milestone 4 — Regression gates

- [x] Compare baseline and candidate trajectory sets
- [x] Detect newly introduced failures
- [x] Define CI-friendly pass/fail policies
- [x] Produce scenario-level regression summaries

## Milestone 5 — Motion-prediction evaluation

- [x] Decode official single-object motion-submission protobufs
- [x] Align scenario IDs, object IDs, timestamps, and prediction horizons
- [x] Compute multimodal minADE, minFDE, and explicit diagnostic miss rate
- [x] Evaluate multiple scenarios and ground-truth shards
- [x] Generate JSON, Markdown, and HTML batch reports
- [x] Generate a transparent no-future-leakage constant-velocity baseline
- [x] Validate generated baseline predictions against real WOMD scenarios

## Milestone 6 — Kinematic candidates and quality gates

- [x] Add capped constant-acceleration prediction
- [x] Add capped constant-turn-rate prediction
- [x] Build a three-mode kinematic ensemble
- [x] Gate aggregate ADE, FDE, miss-rate, and coverage regressions
- [x] Compare the ensemble with the verified real-data baseline

## Milestone 7 — Full-shard benchmark

- [x] Preserve authoritative scenario timelines independently of track validity
- [x] Add deterministic bootstrap confidence intervals
- [x] Add vehicle, pedestrian, and cyclist performance breakdowns
- [x] Add mode-contribution and worst-agent rankings
- [x] Evaluate constant velocity and kinematic ensemble across a complete shard
- [x] Gate the full-shard candidate comparison

## Milestone 8 — Paired evidence and case studies

- [x] Add deterministic paired agent-level bootstrap intervals
- [x] Gate statistically supported minADE and minFDE improvements
- [x] Count improved, unchanged, and regressed agents by metric
- [x] Rank the most improved and regressed scenarios and agents
- [x] Generate standalone Markdown and HTML comparison reports
- [x] Validate paired evidence across the complete real-data shard

## Milestone 9 — Reproducible experiment manifests

- [x] Capture dataset, model, metric, threshold, and seed configuration
- [x] Record source revisions and artifact checksums
- [x] Run generation, evaluation, and comparison from one command
- [x] Produce a self-contained experiment index
- [ ] Validate the one-command manifest against the complete real-data shard

## Milestone 10 — Prediction-risk verification

- [ ] Connect forecast errors to map and interaction context
- [ ] Detect high-consequence misses and near-conflict cases
- [ ] Stratify failures by motion and scene difficulty
- [ ] Generate review-ready scenario evidence

## Milestone 11 — External model integration

- [ ] Define a stable adapter contract for third-party predictions
- [ ] Validate at least one learned-model output
- [ ] Compare learned, kinematic, and ensemble candidates fairly
- [ ] Add release-quality reproducibility and interpretation guidance
