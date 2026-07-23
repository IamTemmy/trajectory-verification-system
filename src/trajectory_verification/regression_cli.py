"""Command-line trajectory regression gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .io import load_requirements, load_scenario_manifest
from .regression import RegressionPolicy, compare_scenario_sets
from .regression_reporting import write_regression_reports


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare trajectory scenario sets")
    parser.add_argument("baseline_manifest")
    parser.add_argument("candidate_manifest")
    parser.add_argument("requirements")
    parser.add_argument("--policy", help="regression policy JSON")
    parser.add_argument("--json-report")
    parser.add_argument("--markdown-report")
    parser.add_argument("--html-report")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    policy = RegressionPolicy()
    if args.policy:
        policy = RegressionPolicy.from_dict(
            json.loads(Path(args.policy).read_text(encoding="utf-8"))
        )
    summary = compare_scenario_sets(
        load_scenario_manifest(args.baseline_manifest),
        load_scenario_manifest(args.candidate_manifest),
        load_requirements(args.requirements),
        policy,
    )
    payload = summary.to_dict()
    if args.json_report:
        output = Path(args.json_report)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_regression_reports(summary, args.markdown_report, args.html_report)
    print(json.dumps(payload, indent=2))
    return 0 if summary.gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
