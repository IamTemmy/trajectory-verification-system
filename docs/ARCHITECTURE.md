# Architecture

## Boundaries

The system separates dataset decoding from verification logic.

1. **Adapters** translate source-specific records into normalized scenarios.
2. **Signal engine** derives physical quantities with explicit units.
3. **Requirement engine** evaluates declarative predicates frame by frame.
4. **Failure analysis** groups failed frames into intervals and retains evidence.
5. **Reporting** converts results into human-readable and machine-readable artifacts.

## Normalized model

A scenario contains timestamped agent states. Each state currently includes planar position and may include heading. Velocity and higher derivatives are derived rather than trusted implicitly.

All current quantities use SI units:

- time: seconds;
- position and separation: metres;
- speed: metres per second;
- acceleration: metres per second squared;
- jerk: metres per second cubed.

## First requirement family

Milestone 0 supports scalar threshold requirements over derived signals. A requirement names a metric, comparison operator, threshold, subject agent, and optional counterpart agent. Evaluation yields every sample, failed samples, localized intervals, and extrema.

## WOMD adapter

The WOMD adapter:

- decode scenario protocol buffers;
- preserve scenario and track identifiers;
- translate valid state histories into the normalized model;
- records map-feature availability for later map-aware requirements;
- fail explicitly on missing or inconsistent data.

The adapter will not leak TensorFlow or Waymo SDK types into the core engine.

The adapter reads uncompressed TFRecord framing without TensorFlow and decodes a
wire-compatible subset of the official Scenario schema using Google's protobuf
runtime. See [WOMD_SETUP.md](WOMD_SETUP.md).

## Engineering evidence

The evidence layer wraps requirement results without changing their pass/fail
semantics. It adds:

- quantitative threshold deviation and interval narratives;
- deterministic threshold-sensitivity sweeps;
- scenario and requirement-level data-quality annotations;
- evidence-completeness confidence with an explicit rationale;
- structured JSON, Markdown, and standalone HTML representations.

Confidence describes completeness of the available evidence, not confidence in
real-world safety. Role selectors are resolved at the adapter boundary so reusable
WOMD requirement templates do not leak scenario-specific track IDs into policy.
