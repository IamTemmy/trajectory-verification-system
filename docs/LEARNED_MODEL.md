# Learned model results

The verification stack was built around candidates written by hand: constant
velocity and a three-mode kinematic ensemble. Those exercise the machinery but
cannot test whether it says anything useful about a model nobody designed to be
easy to evaluate.

This records the first learned candidate scored through it.

## Why the model had to be trained rather than downloaded

No public checkpoint exists for WOMD motion prediction. MTR, MotionCNN, SMART
and UniTraj all publish code without weights. Waymo's terms explicitly permit
releasing trained weights under section 2.3, so the absence is convention among
competition entrants rather than a licensing restriction.

Training on WOMD carries obligations onto anything published from it:
non-commercial use only, a prominent notice that the terms apply downstream, a
copy of the agreement to recipients, and citation. Section 4.1 additionally
forbids use in operating a vehicle.

## Data

Trained on 10 shards of the WOMD **training** split, evaluated on the
**validation** shard already proven byte-identical to Waymo's official decoder
in [READER_VERIFICATION.md](READER_VERIFICATION.md). The split boundary is
Waymo's own, so train and test sets are disjoint by construction.

| | |
|---|---:|
| Training shards | 10 of 1,000 |
| Training examples | 67,574 |
| Evaluation scenarios | 276 |
| Evaluation agents | 1,203 designated targets |

Training widened the population beyond the designated targets to every
sufficiently mobile agent, as described in [../training/README.md](../training/README.md).
Evaluation did not.

## Model

Roughly 0.8 M parameters. The target agent forms a single query that
cross-attends to context tokens built from neighbouring agents and pooled lane
geometry, producing six modes of sixteen points with confidences.

Thirty epochs at about 27 s each on a Colab T4 — under fifteen minutes of
training.

## Result

Scored by `evaluate-motion-predictions` on the designated targets:

| Candidate | mean minADE | mean minFDE | miss rate | coverage |
|---|---:|---:|---:|---:|
| Constant velocity | 9.633 m | 24.385 m | 0.934 | 0.9313 |
| Kinematic ensemble | 7.728 m | 19.938 m | 0.915 | 0.9313 |
| **Learned model** | **2.008 m** | **4.638 m** | **0.657** | 0.9313 |

Against constant velocity: 79.2% lower minADE, 81.0% lower minFDE. Against the
kinematic ensemble: 74.0% and 76.7%. Ground-truth coverage is identical across
all three, so the evaluated population did not drift.

### Paired comparison against the ensemble

Agent-level paired bootstrap, 95% intervals on the improvement:

| Metric | Interval | Excludes zero |
|---|---|---|
| minADE | [-6.042, -5.404] | yes |
| minFDE | [-16.187, -14.476] | yes |
| miss | [-0.286, -0.233] | yes |

The regression gate passed.

### Where it regresses

Unlike the kinematic ensemble, the learned candidate makes some agents worse:

| Metric | Improved | Unchanged | Regressed |
|---|---:|---:|---:|
| minADE | 1,105 | 0 | **98** |
| minFDE | 1,081 | 0 | **122** |
| miss | 328 | 858 | 17 |

The ensemble regressed no agent because it retains constant velocity as an
available mode, so it can always fall back. A learned model has no such floor.
Eight percent of agents getting worse is the price of a much better average, and
it is the kind of trade an aggregate metric hides completely.

## Failure structure

This is what the risk-context layer exists to surface.

### By motion class

| Class | Agents | minADE | minFDE | miss rate |
|---|---:|---:|---:|---:|
| Stationary | 6 | 0.450 m | 1.300 m | 0.167 |
| Straight | 739 | 1.616 m | 3.388 m | 0.545 |
| **Turning** | **458** | **2.661 m** | **6.697 m** | **0.843** |

Turning agents are the dominant failure mode. Their final-displacement error is
roughly double that of agents travelling straight, and 84% miss the 2 m
threshold against 55% for straight motion. The aggregate 2.008 m conceals two
populations behaving very differently.

The ranking is not an artefact of the model: the kinematic ensemble found
turning hardest too, at 8.093 m against 7.545 m for straight motion. The learned
model improves both, by 67.1% and 78.6% respectively, but narrows the gap rather
than closing it.

### By object type

| Type | Agents | minADE | minFDE | miss rate |
|---|---:|---:|---:|---:|
| Pedestrian | 126 | 0.870 m | 1.799 m | 0.262 |
| Vehicle | 1,055 | 2.136 m | 4.956 m | 0.700 |
| Cyclist | 22 | 2.398 m | 5.649 m | 0.864 |

Pedestrians are predicted best, which follows from their low speeds. Cyclists
are worst and are also the smallest population, so that figure rests on 22
agents and should not be over-read.

### Mode usage

The six modes were selected as best 217, 204, 137, 161, 265 and 219 times. No
mode collapse: the winner-takes-all objective kept the modes distinct rather
than letting them converge on a single averaged trajectory.

## What the regressions have in common

Ninety-eight agents scored worse than the kinematic ensemble. Characterising
them turns out to explain itself.

| Type | Agents | Ensemble | Learned | Improvement | Regressed |
|---|---:|---:|---:|---:|---:|
| Vehicle | 1,055 | 8.548 m | 2.136 m | 75.0% | 6.2% |
| Pedestrian | 126 | 1.403 m | 0.870 m | 37.9% | **23.0%** |
| Cyclist | 22 | 4.673 m | 2.398 m | 48.7% | **18.2%** |

The regression rate tracks training representation. A sample of the training
population is about 93% vehicles and 7% pedestrians, with cyclists barely
present. Vehicles regress at 6%, pedestrians at 23%, cyclists at 18% - the
classes the model saw least are the classes it degrades most.

Two further properties separate the two groups. Pedestrians make up 30% of the
regressions against 9% of the improvements, and regressed agents travel a median
26.7 m over the horizon against 41.7 m for improved ones. The model is worse
than physics specifically on slow, under-represented agents, which is exactly
where constant velocity is hardest to beat: a pedestrian walking steadily is
almost perfectly described by it.

The trade is heavily favourable in magnitude. The median regression costs 0.39 m
while the median improvement gains 4.88 m, and the worst single regression is
6.20 m.

None of this is visible in the aggregate. A mean of 2.008 m reports a large win
and says nothing about a class-dependent failure that a rebalanced training set
would plausibly fix - which is a concrete, testable next experiment rather than
an observation.

## Limits

These are project diagnostics under documented assumptions, **not** official
challenge scores, and they are not comparable to leaderboard numbers.

Published state-of-the-art on WOMD is roughly 0.6 m minADE. At 2.008 m this
model is some three times worse, which is unsurprising: it has 0.8 M parameters,
saw 10 of 1,000 available training shards, and trained for fifteen minutes. It
exists to demonstrate that the verification stack produces meaningful evidence
about a model nobody hand-designed for it, not to compete.

The miss rate of 0.657 remains high. At a 2 m final-displacement threshold over
an 8 second horizon, most agents are still missed even when the average error is
small.

Results cover one validation shard. Extending them means repeating the procedure
on more shards.
