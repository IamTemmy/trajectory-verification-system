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
- deterministic unit tests and continuous integration.
- lightweight WOMD scenario-proto TFRecord ingestion without a TensorFlow dependency.
- standalone SVG trajectory visualization without plotting dependencies.

## Quick start

```bash
python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m trajectory_verification.cli examples/following_scenario.json examples/requirements.json
```

The core package intentionally uses only the Python standard library. Dataset-specific integrations will be isolated behind adapters so the verification engine remains testable without downloading WOMD.

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

This is an active engineering project. The normalized core and first threshold evaluator form Milestone 0; WOMD protobuf ingestion and map-aware requirements follow after environment compatibility is validated.

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
