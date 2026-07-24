"""Convert model-neutral external predictions to official WOMD wire format."""

from __future__ import annotations

import argparse
import json

from .adapters.external_predictions import load_external_predictions
from .adapters.motion_submission import write_motion_submission
from .adapters.womd import iter_womd_scenarios


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and import external motion predictions"
    )
    parser.add_argument("external_json")
    parser.add_argument("output")
    parser.add_argument("shards", nargs="+")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    scenarios = tuple(iter_womd_scenarios(args.shards))
    scenario_ids = [item.scenario_id for item in scenarios]
    if len(scenario_ids) != len(set(scenario_ids)):
        raise ValueError("ground-truth shards contain duplicate scenario IDs")
    artifact = load_external_predictions(args.external_json, scenarios)
    write_motion_submission(artifact.predictions, args.output)
    print(json.dumps({
        "model": artifact.provenance.to_dict(),
        "scenarios": len(artifact.predictions),
        "agents": sum(len(item.agents) for item in artifact.predictions),
        "output": args.output,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
