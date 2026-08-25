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

### 2. Install the official decoder beside the notebook interpreter

The wheel pins TensorFlow 2.12, which supports Python 3.8 to 3.11. Colab's
default interpreter is newer than that, so installing directly fails: pip finds
no matching wheel, falls back to the source distribution, and the build dies
needing Google's build system.

Create a separate interpreter for the wheel rather than changing the notebook's:

```python
!pip install -q uv
!uv python install 3.10
!uv venv --python 3.10 /content/wod-env
!uv pip install --python /content/wod-env/bin/python waymo-open-dataset-tf-2-12-0
```

Confirm before continuing:

```python
!/content/wod-env/bin/python -c "import tensorflow as tf; from waymo_open_dataset.protos import scenario_pb2; print('tensorflow', tf.__version__); print('scenario proto OK')"
```

CUDA and TensorRT warnings are expected and harmless; the comparison is
CPU-only.

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

### 4. Provide this project

Installing the package into the 3.10 environment fails: its declared
`requires-python` floor is 3.11, and the wheel's ceiling is 3.11. Clone the
source and put it on `PYTHONPATH` instead. Nothing outside the standard library
is needed on this path, so there is nothing to install.

```python
!git clone -q https://github.com/IamTemmy/trajectory-verification-system.git /content/tvs
```

### 5. Produce the official digest

```python
!PYTHONPATH=/content/tvs/src /content/wod-env/bin/python \
  /content/tvs/tools/normalization_digest.py "$SHARD" \
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

## Verified result

Executed August 24-25, 2026 across five shards of the WOMD validation split.
The procedure was established on a 20-record slice, then the complete shard
`...tfrecord-00007-of-00150`, then four further shards.

```
MATCH - 306 scenarios decoded identically.   validation.tfrecord-00001-of-00150
MATCH - 287 scenarios decoded identically.   validation.tfrecord-00002-of-00150
MATCH - 294 scenarios decoded identically.   validation.tfrecord-00003-of-00150
MATCH - 282 scenarios decoded identically.   validation.tfrecord-00004-of-00150
MATCH - 276 scenarios decoded identically.   ...validation.tfrecord-00007-of-00150
```

| Quantity | Shard 00007 | All five shards |
|---|---:|---:|
| Scenarios | 276 | **1,445** |
| Agent tracks | 17,525 | **99,330** |
| Agent states | 864,001 | **4,856,314** |
| Compared per-state values | 8,640,010 | **48,563,140** |

Every run matched on every per-scenario digest, not only on the combined digest,
and the two readers resolved identical sets of scenario identifiers throughout.

The combined digest for shard 00007 is
`014b7e50eff675d0f5796f7cdd2a01cbd9c557f0af24fd5a76ba794aa18d5ec8`; its digest
files are retained under `docs/evidence/`. The four later shards were compared
in place on the runtime that produced them, since holding both decoders on one
machine removes the need to move 235 MB shards between machines at all.

The slice was produced with `tools/slice_shard.py` and digests identically to
the first 20 scenarios of the complete shard, confirming it is a byte-faithful
prefix rather than a re-encoding.

### Environment

| | Project reader | Official reader |
|---|---|---|
| Platform | macOS arm64 | Google Colab, x86-64 Linux |
| Python | 3.14.5 | 3.10.21 |
| Decoder | `womd_proto` schema subset | `waymo-open-dataset-tf-2-12-0` 1.6.5 |
| Framing | `iter_tfrecord_records` | TensorFlow 2.12.0 `TFRecordDataset` |

Colab's default interpreter was Python 3.13, which the official wheel does not
support; a separate 3.10 environment was created alongside it. The project was
placed on `PYTHONPATH` rather than installed, because its declared
`requires-python` floor of 3.11 conflicts with the wheel's ceiling. The package
imports nothing beyond the standard library on this path, so no installation is
required.

## What this does and does not establish

It establishes that the two decoders agree on every normalized field, for the
scenarios in the shards compared.

It does not establish agreement on fields the subset schema never declares, on
shards with features absent from those tested, or on future schema revisions.
Extending coverage means repeating the procedure on additional shards.
