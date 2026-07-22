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

```python
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

## Dataset access

Waymo requires users to apply for Open Dataset access and authenticate with the account used for that application. Dataset files are not committed to this repository.
