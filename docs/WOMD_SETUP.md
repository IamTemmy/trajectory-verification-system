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

TFRecord is a simple binary framing format. The project reads that framing with Python's standard library and passes each payload to the official generated `Scenario` protobuf class. This keeps the ingestion path lightweight and testable on macOS.

CRC fields are consumed but not yet validated. Dataset shards should be validated by checksum at download time; optional CRC32C validation is planned.

## Optional runtime dependency

Real shards require a package that provides:

```python
from waymo_open_dataset.protos import scenario_pb2
```

Waymo's official motion tutorial currently installs `waymo-open-dataset-tf-2-12-0==1.6.7`, which is a substantial TensorFlow-oriented distribution. Do not install it into the project environment until platform compatibility has been confirmed.

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
