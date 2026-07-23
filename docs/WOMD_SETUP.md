# WOMD ingestion setup

## Supported input

The current adapter reads **uncompressed scenario-proto TFRecord shards**. It does not read the fixed-shape `tf.Example` representation used by the official motion tutorial.

The official scenario schema aligns every track state by index with `timestamps_seconds`. The adapter:

- preserves valid states and their original timestamps;
- preserves object type and box/velocity fields;
- identifies the self-driving-car track by `sdc_track_index`;
- resolves objects of interest and prediction targets to agent IDs;
- records the number of map features without coupling the core model to Waymo types.

## Why TensorFlow is not required

TFRecord is a simple binary framing format. The project reads that framing with
Python's standard library and passes each payload to a wire-compatible subset of
the official `Scenario` schema. Protobuf safely skips fields not consumed by the
normalizer. This keeps ingestion lightweight on Apple Silicon without TensorFlow.

CRC fields are consumed but not yet validated. Dataset shards should be validated by checksum at download time; optional CRC32C validation is planned.

## Runtime dependency

Real shards require only Google's cross-platform protobuf runtime, installed with
the project:

```bash
python3 -m pip install -e .
```

The official Waymo wheel is unnecessary here. Its published artifact targets
Linux x86-64 and is unsuitable for Apple Silicon macOS.

## Inspect a shard

```bash
PYTHONPATH=src python3 -m trajectory_verification.womd_cli \
  /path/to/uncompressed_scenario_validation.tfrecord-00000-of-00150 \
  --limit 3 \
  --svg-output reports/generated/first_scenario.svg
```

The SVG renderer uses only the Python standard library. It plots normalized agent
trajectories, highlights the SDC track, and labels each final position by track ID
and object type.

## Verified environment

The ingestion path was validated on July 22, 2026 against an authenticated WOMD
Motion Dataset v1.3.1 uncompressed validation shard:

- platform: Apple Silicon macOS (`arm64`);
- Python: 3.11.5;
- protobuf runtime: 6.33.6;
- shard: `uncompressed_scenario_validation_validation.tfrecord-00007-of-00150`;
- result: three scenarios decoded and normalized, with 14, 56, and 57 retained
  tracks and 324, 319, and 316 map features respectively;
- visualization: standalone SVG generated from the first decoded scenario.

The raw shard and generated report remain local and are excluded from version
control. This validates schema compatibility and the end-to-end ingestion path;
it is not a claim about production Waymo Driver behavior.

## Verified engineering-evidence run

Milestone 2 was validated on July 22, 2026 using scenario
`4992809c590076fe` from the same authenticated v1.3.1 shard. Reusable selectors
resolved `@sdc` to track `873` and `@prediction:0` to track `842`.

The report evaluated 270 derived samples across three demonstration data-review
requirements:

- SDC speed: 90 samples, observed 6.23–11.13 m/s, passed;
- SDC acceleration: 89 samples, observed -0.10–1.20 m/s², passed;
- selected-pair separation: 91 samples, observed 27.45–47.06 m, passed.

Each result included a three-point threshold-sensitivity sweep and high
requirement-level evidence confidence. Scenario-level quality analysis separately
flagged irregular sampling on tracks `865`, `867`, and `872`; those tracks were
not inputs to the evaluated requirements. Markdown, standalone HTML, structured
JSON, and SVG outputs were generated successfully and remain gitignored.

These thresholds are demonstrative data-review checks, not validated safety
limits, and the results make no claim about production Waymo Driver behavior.

## Verified map-aware run

Milestone 3 was validated on July 23, 2026 using scenario
`4992809c590076fe`. The SDC role resolved to track `873`. Lane association was
applicable for all 91 SDC states and passed the 3.5 m demonstration threshold,
with observed center offset from 0.044 m to 0.446 m.

The remaining requirements were correctly reported as `NOT APPLICABLE`:

- no stop-state traffic signal was present for the signal-crossing check;
- the SDC did not cross a mapped stop-sign position;
- no pedestrian or cyclist occupied a mapped crosswalk context.

Thus the scenario result is one of one applicable requirement passed, with three
requirements excluded for recorded, machine-readable reasons. The scenario-level
irregular-sampling warning remained limited to tracks `865`, `867`, and `872`,
which were not inputs to the lane-offset evaluation. Map-overlay SVG, Markdown,
HTML, and JSON artifacts were generated locally and remain gitignored.

## Dataset access

Waymo requires users to apply for Open Dataset access and authenticate with the account used for that application. Dataset files are not committed to this repository.
