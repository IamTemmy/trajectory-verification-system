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

## Planned WOMD adapter

The WOMD adapter will:

- decode scenario protocol buffers;
- preserve scenario and track identifiers;
- translate valid state histories into the normalized model;
- expose map features separately for later map-aware requirements;
- fail explicitly on missing or inconsistent data.

The adapter will not leak TensorFlow or Waymo SDK types into the core engine.
