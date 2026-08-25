# Verifying the reader against Waymo's official decoder

The project decodes WOMD shards with a hand-built schema subset and its own
TFRecord framing reader, avoiding TensorFlow and the Waymo wheel. That choice is
only defensible if the result is provably identical to the official decoder.

This procedure establishes that. It requires a Linux machine with the official
wheel, which Google Colab provides free; no local hardware purchase is needed.

## Design

Both sides run the **same** normalizer, `scenario_from_proto`. Only the proto
class and the record iterator differ:

| | proto class | record iterator |
|---|---|---|
| Official | `waymo_open_dataset.protos.scenario_pb2.Scenario` | `tf.data.TFRecordDataset` |
| Project | `adapters.womd_proto.Scenario` | `adapters.womd.iter_tfrecord_records` |

`scenario_from_proto` is duck-typed against protobuf-shaped attributes, so it
accepts Waymo's generated class unchanged. Holding normalization constant
isolates the schema subset and the framing reader as the only variables. Two
independent normalization paths would not prove the same thing.

`tools/normalization_digest.py` canonicalizes every normalized field — track
identities, object types, all ten per-state quantities, map geometry, signal
states, prediction targets — and reduces each scenario to a SHA-256 digest.
Matching combined digests mean the decoders agree bit for bit.

## Local reference

```bash
python tools/normalization_digest.py \
  data/raw/uncompressed_scenario_validation_validation.tfrecord-00007-of-00150 \
  --output project-digest.json
```

Recorded for shard `...tfrecord-00007-of-00150`, 276 scenarios:

```
combined digest: 014b7e50eff675d0f5796f7cdd2a01cbd9c557f0af24fd5a76ba794aa18d5ec8
```

## Colab procedure

### 1. Confirm the runtime can host the wheel

The official wheel ships for Linux x86-64 and pins a TensorFlow version, which
in turn pins a Python version. Establish this before anything else — it is the
step most likely to fail.

```python
import sys, platform
print(sys.version)
print(platform.machine(), platform.system())
```

### 2. Install the official decoder

```python
!pip install -q waymo-open-dataset-tf-2-12-0
```

If pip reports no compatible distribution, the runtime's Python is too new.
Colab exposes older runtimes, or `condacolab` can pin an older interpreter.
Resolve this here rather than working around it later.

```python
import tensorflow as tf
from waymo_open_dataset.protos import scenario_pb2
print("tensorflow", tf.__version__)
```

### 3. Provide the shard

Either mount Drive after uploading the shard:

```python
from google.colab import drive
drive.mount('/content/drive')
SHARD = '/content/drive/MyDrive/uncompressed_scenario_validation_validation.tfrecord-00007-of-00150'
```

Or copy it directly from Waymo's bucket, which is far faster than uploading
from home. Authenticate with the Google account that accepted the dataset
terms, and use the exact `gsutil` path shown on the Waymo download page:

```python
from google.colab import auth
auth.authenticate_user()
!gsutil cp gs://<path-from-the-download-page>/uncompressed_scenario_validation_validation.tfrecord-00007-of-00150 /content/
```

The shard must be the same one used for the local reference. A different shard
produces a different digest and proves nothing.

### 4. Install this project

```python
!pip install -q git+https://github.com/IamTemmy/trajectory-verification-system.git
!git clone -q https://github.com/IamTemmy/trajectory-verification-system.git /content/tvs
```

### 5. Produce the official digest

```python
!python /content/tvs/tools/normalization_digest.py "$SHARD" \
  --official --output /content/official-digest.json
```

### 6. Retrieve the result

```python
from google.colab import files
files.download('/content/official-digest.json')
```

## Comparing

```bash
python tools/compare_digests.py project-digest.json official-digest.json
```

Equal combined digests establish equivalence over the evaluated shard. Any
mismatch is reported per scenario so the disagreeing field can be located.

## What this does and does not establish

It establishes that the two decoders agree on every normalized field, for the
scenarios in the shards compared.

It does not establish agreement on fields the subset schema never declares, on
shards with features absent from those tested, or on future schema revisions.
Extending coverage means repeating the procedure on additional shards.
