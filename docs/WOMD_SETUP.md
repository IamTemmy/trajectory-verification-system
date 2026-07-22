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

## Dataset access

Waymo requires users to apply for Open Dataset access and authenticate with the account used for that application. Dataset files are not committed to this repository.
