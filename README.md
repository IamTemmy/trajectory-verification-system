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
and `NOT APPLICABLE` outcomes; all three milestones are validated on a real
WOMD v1.3.1 scenario. Regression gates are next.

## Responsible interpretation

- Recorded actors in WOMD are not necessarily controlled by the Waymo Driver.
- A failed project-defined threshold is not proof of unsafe real-world operation.
- Requirement thresholds must cite their engineering rationale before safety claims are made.
- Dataset and SDK use must comply with Waymo's applicable license terms.

## Roadmap

See [docs/ROADMAP.md](docs/ROADMAP.md), [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md),
and [docs/WOMD_SETUP.md](docs/WOMD_SETUP.md).

## License

No project license has been selected yet. All rights are reserved until a license is added. Waymo datasets and SDK components retain their own licenses and terms.
