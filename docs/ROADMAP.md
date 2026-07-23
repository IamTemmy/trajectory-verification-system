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

- [ ] Compare baseline and candidate trajectory sets
- [ ] Detect newly introduced failures
- [ ] Define CI-friendly pass/fail policies
- [ ] Produce scenario-level regression summaries
