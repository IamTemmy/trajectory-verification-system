# Trajectory Verification System

[![CI](https://github.com/IamTemmy/trajectory-verification-system/actions/workflows/ci.yml/badge.svg)](https://github.com/IamTemmy/trajectory-verification-system/actions/workflows/ci.yml)

A requirements-driven engineering toolkit for verifying recorded or generated autonomous-driving trajectories, localizing failures, and producing traceable evidence.

The first dataset adapter targets the Waymo Open Motion Dataset (WOMD). This project evaluates trajectories contained in or supplied alongside public datasets; it does **not** evaluate or make claims about the production Waymo Driver.

## Why this project exists

Motion-prediction benchmarks usually summarize performance with aggregate metrics. Engineering validation also needs to answer:

- Which behavioral requirement failed?
- When did it fail?
- Which agents and signals support that conclusion?
- How sensitive is the result to the chosen threshold?
- Did a new trajectory version introduce a regression?

This repository is being built around those questions.

## Initial vertical slice

The first milestone provides:

- normalized two-dimensional agent trajectories;
- derived speed, acceleration, jerk, separation, closing speed, and time-to-collision signals;
- declarative threshold requirements;
- contiguous failure-interval localization;
- structured evidence suitable for reports and regression gates;
- deterministic unit tests and continuous integration;
- lightweight WOMD scenario-proto TFRecord ingestion without a TensorFlow dependency;
- standalone SVG trajectory visualization without plotting dependencies.
- deterministic baseline/candidate regression gates for CI.

## Quick start

```bash
python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m trajectory_verification.cli examples/following_scenario.json examples/requirements.json
```

Generate machine-readable evidence plus Markdown and standalone HTML reports:

```bash
verify-trajectories examples/following_scenario.json examples/requirements.json \
  --markdown-report reports/generated/example.md \
  --html-report reports/generated/example.html
```

Run reusable role-based requirements directly on a WOMD scenario shard:

```bash
verify-womd data/raw/SHARD examples/womd_requirements.json \
  --markdown-report reports/generated/womd.md \
  --html-report reports/generated/womd.html \
  --svg-output reports/generated/womd.svg
```

WOMD templates may select `@sdc`, `@prediction:N`, or
`@object_of_interest:N`; reports always record the resolved track IDs.

Run the included regression example (it intentionally exits `1` because the
candidate introduces a speed failure):

```bash
compare-trajectories \
  examples/regression/baseline_manifest.json \
  examples/regression/candidate_manifest.json \
  examples/regression/requirements.json \
  --policy examples/regression/policy.json \
  --json-report reports/generated/regression.json \
  --markdown-report reports/generated/regression.md \
  --html-report reports/generated/regression.html
```

Manifest scenario paths are relative to the manifest. The default policy allows
no new failures and blocks missing candidate scenarios or lost applicability.
The command exits `0` when the policy passes and `1` when it fails.

Evaluate an official WOMD motion-prediction submission against matching local
scenario shards:

```bash
generate-womd-baseline reports/generated/constant_velocity.binproto data/raw/SHARD

evaluate-motion-predictions predictions.binproto data/raw/SHARD \
  --json-report reports/generated/predictions.json \
  --markdown-report reports/generated/predictions.md \
  --html-report reports/generated/predictions.html
```

The evaluator preserves Waymo's documented scenario IDs, object IDs, six-mode
limit, and 16-point prediction horizon. Its minADE, minFDE, and configurable
miss-rate outputs are clearly labeled as project diagnostics rather than
official challenge scores.

Generate a stronger three-mode transparent candidate and compare it with a
baseline evaluation:

```bash
generate-womd-baseline reports/generated/ensemble.binproto data/raw/SHARD \
  --model kinematic_ensemble

compare-prediction-evaluations baseline.json candidate.json \
  --policy examples/prediction_comparison_policy.json \
  --html-report reports/generated/prediction-comparison.html
```

The comparison uses paired agent-level bootstrap intervals, can require
statistically supported improvement, and ranks the strongest gains and
regressions for case-study review.

Reproduce generation, evaluation, comparison, and artifact indexing from one
manifest:

```bash
run-prediction-experiment examples/full_shard_experiment.json
```

The resulting experiment index records the source revision, dataset and
manifest SHA-256 checksums, effective configuration, artifact checksums, and
gate outcome.

The verification core uses only the Python standard library. WOMD decoding adds
Google's cross-platform protobuf runtime behind an isolated adapter, so the core
remains testable without downloading WOMD or installing TensorFlow.

## Architecture

```text
scenario source -> dataset adapter -> normalized trajectories
                                      |
                                      v
                              derived signal engine
                                      |
                                      v
                          declarative requirement engine
                                      |
                                      v
                  failure intervals + evidence + report data
```

## Repository status

This is an active engineering project. Milestone 0 (verification kernel) and
Milestone 1 (WOMD ingestion), and Milestone 2 (engineering evidence) are
complete. Milestone 3 adds map-aware requirements with explicit `PASS`, `FAIL`,
and `NOT APPLICABLE` outcomes; the WOMD milestones are validated on a real
WOMD v1.3.1 scenario. Milestone 4 adds deterministic regression gates, and
Milestone 5 adds official-format motion-prediction ingestion and batch metrics,
validated with generated baseline predictions on three real WOMD scenarios.
Milestone 6 adds a transparent kinematic ensemble that reduced mean minADE by
48.6% and mean minFDE by 46.5% against that baseline under a strict regression
policy. Full-shard reports add deterministic confidence intervals, object-type
breakdowns, mode contributions, and worst-case rankings. Across a verified
276-scenario validation shard, the ensemble improved mean minADE by 19.8% and
mean minFDE by 18.2% over constant velocity while preserving the evaluated
population and label coverage.
Milestone 8 adds paired improvement uncertainty and standalone case-study
reports. On the verified 1,203-agent full-shard comparison, paired 95%
intervals excluded zero for minADE, minFDE, and miss-rate improvements; no
agent regressed because the candidate preserves constant velocity as an
available mode.

## Responsible interpretation

- Recorded actors in WOMD are not necessarily controlled by the Waymo Driver.
- A failed project-defined threshold is not proof of unsafe real-world operation.
- Requirement thresholds must cite their engineering rationale before safety claims are made.
- Dataset and SDK use must comply with Waymo's applicable license terms.

## Roadmap

See [docs/ROADMAP.md](docs/ROADMAP.md), [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md),
[docs/WOMD_SETUP.md](docs/WOMD_SETUP.md), and
[docs/MOTION_PREDICTIONS.md](docs/MOTION_PREDICTIONS.md). Reproducible runs are
documented in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

## License

No project license has been selected yet. All rights are reserved until a license is added. Waymo datasets and SDK components retain their own licenses and terms.
