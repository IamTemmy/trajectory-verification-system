"""Command-line entry point for the verification kernel."""

from __future__ import annotations

import argparse
import json

from .io import load_requirements, load_scenario
from .requirements import evaluate_requirement


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate trajectory requirements")
    parser.add_argument("scenario", help="normalized scenario JSON")
    parser.add_argument("requirements", help="requirement definitions JSON")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    scenario = load_scenario(args.scenario)
    results = [
        evaluate_requirement(scenario, requirement).to_dict()
        for requirement in load_requirements(args.requirements)
    ]
    print(json.dumps({"scenario_id": scenario.scenario_id, "results": results}, indent=2))
    return 1 if any(not result["passed"] for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
