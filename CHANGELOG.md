# Changelog

All notable changes to this project are documented here.

## [1.0.0] - 2026-09-03

First stable portfolio release of the trajectory-verification system.

### Verification system

- Requirements-driven trajectory verification with localized failure evidence.
- TensorFlow-free WOMD scenario-proto reader, checked against Waymo's official
  decoder across five complete validation shards.
- Map-aware requirements, regression gates, standalone reports and SVG cases.
- Official-compatible motion-prediction ingestion, transparent baselines,
  paired bootstrap comparisons and reproducible experiment manifests.
- Risk-context analysis and behavioral verification of predicted futures.

### Learned-model evidence

- Versioned, provenance-enforced external-prediction contract.
- Compact six-mode cross-attention predictor trained on WOMD training shards.
- Full-shard comparison against constant-velocity and kinematic candidates.
- Documented diagnosis and mitigation of a class-dependent regression.

### Release quality

- Core CI across Python 3.10, 3.11 and 3.12.
- Isolated, pinned training dependencies and CPU training-pipeline smoke tests.
- Reconciled model, dataset-population and milestone documentation.

The numerical results are project diagnostics under documented assumptions,
not official Waymo challenge scores or claims about the production Waymo
Driver. Dataset-derived artifacts remain subject to the Waymo Open Dataset
terms.
