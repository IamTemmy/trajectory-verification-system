"""Run a reproducible motion-prediction experiment manifest."""

from __future__ import annotations

import argparse
import json

from .experiment import load_experiment_manifest, run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate, evaluate, compare, and index a prediction experiment"
    )
    parser.add_argument("manifest")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    index = run_experiment(load_experiment_manifest(args.manifest))
    print(json.dumps({
        "experiment_id": index["experiment_id"],
        "source_revision": index["source_revision"],
        "dataset": index["dataset"],
        "result": index["result"],
        "artifact_count": len(index["artifacts"]),
    }, indent=2))
    return 0 if index["result"]["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
