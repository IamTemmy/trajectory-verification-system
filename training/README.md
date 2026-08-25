# Training a learned prediction model

The verification core depends on nothing but the standard library. A learned
model is an external prediction source under
[docs/EXTERNAL_MODELS.md](../docs/EXTERNAL_MODELS.md), so everything here lives
outside `src/` and carries its own dependencies.

Only the reader is imported from the package, which means training consumes
exactly the decoder proven equivalent to Waymo's in
[docs/READER_VERIFICATION.md](../docs/READER_VERIFICATION.md).

## Task

Fixed by the official submission format the evaluator already reads:

| | |
|---|---|
| History | scenario steps 0-10 (~1.1 s) |
| Prediction made at | step 10 |
| Future | 16 points at steps 15, 20, ... 90 (8 s) |
| Output | 6 modes with confidences |

## Features

Each example is expressed in a frame centred on the target at step 10, rotated
so the target faces +x. Without that, capacity is wasted learning that a vehicle
at map coordinate (3301, -328) behaves like one at (8456, 1737).

| Tensor | Shape | Contents |
|---|---|---|
| `target_history` | (11, 7) | x, y, cos h, sin h, vx, vy, valid |
| `neighbors` | (16, 11, 10) | the above plus a three-way type indicator |
| `map_points` | (256, 5) | x, y, direction x, direction y, valid |
| `future` | (16, 2) | ground truth in the local frame |
| `future_mask` | (16,) | which future steps exist |

Roughly half of all agents leave the scene before the horizon ends, so futures
are frequently partial. The mask carries that through and the loss must respect
it, or the model learns to predict toward the origin for vanished agents.

## Training population

WOMD designates about 3.5 prediction targets per scenario, and those targets are
essentially always in motion. Restricting training to them wastes most of the
recorded data; admitting every agent instead floods it with parked cars.

`--all-agents` therefore keeps every designated target unfiltered and admits
others only if they are moving at the anchor step or travel a meaningful
distance. Measured over 20 verified scenarios:

| Population | Examples | Moving | Median speed |
|---|---:|---:|---:|
| Designated targets | 71 | 100% | 6.45 m/s |
| All agents, unfiltered | 616 | 25% | — |
| **All agents, filtered** | **271** | **92%** | **6.61 m/s** |

That is 3.8x the supervision at a speed distribution matching the designated
targets. Evaluation still runs on designated targets only, so results stay
comparable to the challenge definition.

## Preprocessing

Decoding protobufs dominates the cost, so conversion runs once and training
reads arrays. One archive per shard keeps the job resumable.

```bash
python training/preprocess.py /content/train/training.tfrecord-* \
  --out-dir /content/features --all-agents --skip-existing
```

About 0.15 s per scenario, roughly 6.5 KB per example.
