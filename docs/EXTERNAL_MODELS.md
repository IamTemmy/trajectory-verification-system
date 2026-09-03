# External learned-model integration

The verification core does not import PyTorch, TensorFlow, CUDA extensions, or
model-specific preprocessing code. Learned models exchange predictions through
the versioned JSON contract in
`schemas/external_predictions.schema.json`.

## Contract

An external artifact must declare:

- model name and version;
- source repository and exact source revision;
- SHA-256 of the checkpoint used for inference;
- `scenario_global` coordinates in metres;
- that future ground truth was not used.

Each scenario must contain exactly the WOMD `tracks_to_predict` population.
Each agent may provide one to six modes. Every mode contains exactly 16 x/y
points corresponding to WOMD steps 15, 20, 25, …, 90. Confidences must be
finite and non-negative; the importer normalizes them to sum to one.

The importer rejects duplicate or mismatched scenario and agent identities,
unsupported coordinate frames, malformed horizons, non-finite coordinates,
invalid checkpoint hashes, and declared future-label use.

```bash
import-external-predictions \
  external_predictions.json \
  reports/generated/external-model.binproto \
  data/raw/SHARD
```

The output uses the same official-compatible `MotionChallengeSubmission` wire
format as locally generated candidates and can enter the existing evaluator,
risk-context analysis, and regression gates.

## Learned-model target assessment

MTR was the first integration target considered because its official repository
reports strong WOMD validation performance. Its public training and evaluation
commands expect a user-produced checkpoint; the repository does not currently
provide downloadable pretrained weights. Reproducing it would therefore require
both training and inference in an older Linux/CUDA/Waymo-TensorFlow stack, not
just adapting an existing artifact, and it is not a native Apple-silicon
workload.

The first validated learned source is instead the compact cross-attention model
under `training/`. It was trained on a free Colab T4 and exports only provenance
plus scenario-global predictions through the same contract. MTR remains a
compatible future source:

1. train and run it in a pinned Linux/CUDA environment;
2. export model provenance and scenario-global predictions;
3. validate and evaluate the artifact on any platform through this repository.

This boundary keeps model execution replaceable and avoids pretending that a
CUDA model is supported on the user's Mac.

Primary references:

- [MTR official repository](https://github.com/sshaoshuai/MTR)
- [MTR installation](https://github.com/sshaoshuai/MTR/blob/master/docs/INSTALL.md)
- [MTR dataset preparation](https://github.com/sshaoshuai/MTR/blob/master/docs/DATASET_PREPARATION.md)
- [Waymo Open Dataset repository](https://github.com/waymo-research/waymo-open-dataset)
