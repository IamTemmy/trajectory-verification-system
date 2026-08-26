# Reproducing and interpreting these results

Every number this project reports is a project diagnostic computed under
documented assumptions. None of them is an official Waymo challenge score, and
none is comparable to a leaderboard entry.

This page states what the results mean, what they do not mean, and how to
regenerate them.

## What the numbers are

| Quantity | Definition here |
|---|---|
| minADE | Mean displacement error of the mode closest to the recorded future, averaged over the future steps that exist |
| minFDE | Displacement at each agent's last observed future step, for that same mode |
| Miss rate | Fraction of agents whose minFDE exceeds a project-chosen threshold, 2 m by default |
| Coverage | Fraction of the 16 submission steps for which ground truth exists |

Two choices differ from a naive reading. Errors are averaged over **available**
future steps rather than all sixteen, because roughly 43% of agents leave the
scene before the horizon ends. And final displacement uses each agent's **last
observed step**, not step sixteen, so an agent that vanishes at four seconds is
scored on where it actually was rather than on a fabricated endpoint.

Coverage is reported alongside every result precisely so that a candidate cannot
appear to improve by quietly evaluating a smaller population.

## What they do not mean

**A threshold failure is not proof of unsafe operation.** Requirement thresholds
in this project are chosen to exercise the machinery. Before any safety claim,
a threshold needs a cited engineering rationale.

**Risk priorities are screening labels, not probabilities.** The `high`, `medium`
and `low` labels in the risk-context reports rank cases for human review. They
are not collision likelihoods and are not calibrated.

**Nothing here evaluates the Waymo Driver.** WOMD records the behaviour of many
road users; the recorded actors are not necessarily under Waymo control, and
this project scores forecasts of recorded trajectories.

**Aggregates conceal class-dependent behaviour.** This is demonstrated rather
than asserted in [REGRESSION_EXPERIMENT.md](REGRESSION_EXPERIMENT.md), where an
aggregate that moved 1.4% accompanied a pedestrian regression rate that moved
31%. Read the per-class and per-motion-class breakdowns before drawing a
conclusion from a mean.

## Statistical claims

Comparisons use paired agent-level bootstrap intervals with a fixed seed, so a
rerun on the same inputs reproduces the same interval. An improvement is
reported as supported only when the 95% interval excludes zero.

Subgroup claims are held to the same standard, which is why the cyclist result
in the regression experiment is recorded as unproven: 22 agents cannot support
it, however encouraging the point estimate looks.

Where the regression gate refuses to certify a change, that refusal is reported
rather than worked around.

## Reproducing

### Verification, without any dataset

```bash
python3 -m unittest discover -s tests -v
verify-trajectories examples/intersection_scenario.json \
  examples/intersection_requirements.json --markdown-report report.md
```

The committed scenario is synthetic, so this needs no download and no
credentials. It exits `1` because one requirement fails by design.

### The reader equivalence proof

Follow [READER_VERIFICATION.md](READER_VERIFICATION.md). Requires a WOMD shard
and a Linux runtime for the official decoder; Colab suffices.

### The learned model

Follow [../training/README.md](../training/README.md) for preprocessing and
training, then export and score:

```bash
python training/predict.py CHECKPOINT SHARD --output predictions.json
import-external-predictions predictions.json candidate.binproto SHARD
evaluate-motion-predictions candidate.binproto SHARD --json-report evaluation.json
compare-prediction-evaluations baseline.json evaluation.json \
  --policy examples/prediction_comparison_policy.json --json-report comparison.json
```

Training is stochastic. A rerun with the same seed and hardware should land
close to the recorded figures but is not guaranteed to match them exactly;
GPU kernel non-determinism alone is enough to shift the third decimal.

The deterministic parts are the reader, the metrics, the bootstrap intervals and
the report contents. Those reproduce exactly.

### Case studies

```bash
visualize-predictions candidate.binproto SHARD --out-dir cases --worst 10
```

## Provenance

`run-prediction-experiment` records the Git commit, whether tracked files were
modified, SHA-256 and byte size for the manifest and every dataset shard and
artifact, the effective configuration, and the gate outcome. It deliberately
omits a timestamp so that identical inputs yield an identical index.

Checksums establish byte identity, not authenticity. Obtaining WOMD under its
terms and confirming the source remains the operator's responsibility.

## Dataset and model terms

The Waymo Open Dataset licence governs the data and anything trained on it.
Publishing model weights derived from it is permitted under section 2.3, subject
to four conditions: non-commercial use only, a prominent notice that the terms
apply to downstream recipients, supplying those recipients a copy of the
agreement, and citation. Section 4.1 additionally forbids use in operating a
vehicle or in production systems.

No dataset content is distributed with this repository. The committed evidence
files contain scenario identifiers and digests, not trajectories.

## Scope of the published results

| Result | Covers |
|---|---|
| Reader equivalence | Five validation shards, 1,445 scenarios |
| Learned model evaluation | The same shard, 1,203 designated targets |
| Regression experiment | The same shard, three candidates |
| Behavioral verification | The same shard, three candidates, three requirements |

One shard is not the dataset. Extending any of these claims means repeating the
procedure on more shards, and the procedures are written to make that a matter
of repetition rather than reinvention.
