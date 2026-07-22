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
- [ ] Validate against an authenticated real WOMD scenario shard
- [x] Visualize a normalized decoded scenario as standalone SVG

## Milestone 2 — Engineering evidence

- [ ] Explain failure intervals using contributing signals
- [ ] Add threshold sensitivity sweeps
- [ ] Add confidence and data-quality annotations
- [ ] Generate Markdown and HTML validation reports

## Milestone 3 — Map-aware requirements

- [ ] Lane association and lateral offset
- [ ] Stop-line and traffic-signal compliance
- [ ] Crosswalk and vulnerable-road-user proximity
- [ ] Explicit applicability conditions

## Milestone 4 — Regression gates

- [ ] Compare baseline and candidate trajectory sets
- [ ] Detect newly introduced failures
- [ ] Define CI-friendly pass/fail policies
- [ ] Produce scenario-level regression summaries
