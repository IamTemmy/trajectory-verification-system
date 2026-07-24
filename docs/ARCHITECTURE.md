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

## Map-aware requirements

The normalized map context contains lane-center polylines, stop-sign positions,
crosswalk polygons, and timestamped traffic-signal states. The geometric layer
provides:

- nearest-lane association and absolute lateral offset;
- stop-sign crossing speed;
- stop-state signal crossing indicators;
- agent-to-crosswalk distance;
- vehicle-to-pedestrian/cyclist separation conditioned on mapped crosswalk occupancy.

Map requirements have three outcomes. `PASS` and `FAIL` require sufficient
context and evaluated samples. `NOT APPLICABLE` retains a concrete reason—such
as no stop-state signal, no stop-sign crossing, or no crosswalk-context vulnerable
road user—and is never counted as a passed requirement.

## Regression gates

Scenario-set manifests provide stable baseline and candidate inputs. The
regression layer evaluates the same declarative requirements on matching
scenario IDs and classifies each transition as unchanged, newly failed,
resolved, newly applicable and passing, or lost applicability. Missing and
added scenarios remain explicit rather than disappearing from aggregate counts.

A policy sets the allowed new-failure budget and whether missing candidate
scenarios or lost applicability block the gate. JSON is the machine interface;
Markdown and standalone HTML provide review evidence. The CLI returns exit code
`0` for a passing policy and `1` for a blocking regression.

## Motion-prediction evaluation

The motion-submission adapter decodes the official single-object prediction
wire format into dataset-independent multimodal trajectories. It validates
scenario and object identity, the six-mode limit, and the documented 16-point
WOMD horizon before attaching timestamps from normalized ground truth.

The prediction metric layer aligns by timestamp and computes per-agent minADE,
minFDE, and a configurable final-displacement miss diagnostic. Batch reporting
aggregates scenarios without importing Waymo SDK types into the verification
core. Reports state that these local diagnostics are not substitutes for
Waymo's official challenge evaluation.

Transparent kinematic candidates share the same history boundary and official
serialization path. The multimodal ensemble combines constant velocity,
constant acceleration, and constant turn rate without selecting a future-aware
mode during generation. Evaluation may select the best mode according to the
declared displacement metric, matching the purpose of multimodal forecasting.

Prediction comparison consumes two evaluation artifacts, first requiring
identical scenario and agent populations. A separate policy then gates increases
in mean minADE, mean minFDE, or miss rate and decreases in valid-label coverage.

Full-shard reporting treats the scenario timestamp sequence as source metadata,
not as an accidental property of the longest retained agent track. Uncertainty
is estimated by deterministic nonparametric bootstrap resampling over evaluated
agents. Object-type groups, best-mode counts, and worst-case rankings retain
the identities required for targeted follow-up.

## Reproducible experiments

An experiment manifest binds dataset identity and shard paths to candidate
models, metric thresholds, bootstrap seeds, comparison policy, and output
location. The runner decodes the scenario population once and uses that same
in-memory population for every candidate, preventing accidental population
drift between baseline and candidate runs.

Each run writes official-format prediction submissions, complete evaluation
reports, paired comparison evidence, and `experiment-index.json`. The index
records the Git source revision, dirty-worktree state, manifest and shard
SHA-256 checksums, artifact sizes and checksums, configuration, and gate result.
Large dataset records and generated evidence remain outside version control.

## Prediction risk context

The risk-context layer joins each scored prediction back to its normalized
scenario. It derives motion class, 30 m scene density, minimum future actor
separation, SDC-relative separation error, and proximity to crosswalk and
traffic-control map features. These signals stratify aggregate displacement
errors and rank cases for human review.

Priority is categorical and threshold-driven. A high-priority case requires a
diagnostic miss plus at least one consequential context tag. This is an
engineering-review heuristic, not a calibrated probability of collision,
severity, unsafe behavior, or production-system risk.

## External model boundary

Learned-model runtimes remain outside the verification process. A versioned
JSON interchange contract carries model and checkpoint provenance, coordinate
frame, future-data declaration, scenario/agent identities, mode confidences,
and trajectory coordinates. The adapter validates this contract against the
same normalized WOMD population before writing the official-compatible
submission wire format.

This makes GPU framework and preprocessing choices replaceable while keeping
identity, alignment, scoring, contextual review, and regression policy inside
the tested verification boundary.
