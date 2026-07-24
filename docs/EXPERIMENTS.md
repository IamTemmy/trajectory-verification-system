# Reproducible experiments

`run-prediction-experiment` replaces separate generation, evaluation, and
comparison commands with one versioned manifest.

## Run the verified full-shard configuration

The included manifest expects the previously validated WOMD v1.3.1 shard at
`data/raw/uncompressed_scenario_validation_validation.tfrecord-00007-of-00150`:

```bash
run-prediction-experiment examples/full_shard_experiment.json
```

Outputs are written to `reports/generated/full-shard-experiment/`:

- official-format submission protobufs for every candidate;
- JSON, Markdown, and HTML evaluation reports;
- JSON, Markdown, and HTML paired comparison reports;
- `experiment-index.json`.

## Manifest contract

The manifest records:

- a stable experiment identifier;
- dataset name, version, and one or more paths relative to the manifest;
- named candidate models;
- diagnostic miss threshold;
- evaluation and paired-bootstrap sample counts and seeds;
- baseline/candidate identities and the regression policy;
- an output directory relative to the manifest.

Candidate and experiment identifiers are restricted to letters, numbers,
underscores, and hyphens so they cannot escape the output directory.

## Provenance and integrity

The experiment index captures:

- the checked-out Git commit and whether tracked files were modified;
- SHA-256, byte size, and path for the manifest and dataset shards;
- SHA-256, byte size, and path for every generated artifact;
- the effective model, metric, bootstrap, and comparison configuration;
- the final gate result and violations.

Checksums establish byte identity, not dataset authenticity. The operator must
still obtain WOMD under its applicable terms and confirm the expected source.
The dataset remains local and is never committed by the runner.

The index intentionally does not contain a timestamp. Given identical source,
manifest, dataset bytes, protobuf runtime, and Python implementation, this keeps
the evidence focused on reproducible inputs rather than wall-clock metadata.
