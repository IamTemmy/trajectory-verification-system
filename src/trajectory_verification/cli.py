"""Command-line entry point for the verification kernel."""

from __future__ import annotations

import argparse
import json

from .io import load_requirements, load_scenario
from .evidence import default_sensitivity_thresholds, explain_requirement
from .reporting import build_validation_report, write_validation_reports


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate trajectory requirements")
    parser.add_argument("scenario", help="normalized scenario JSON")
    parser.add_argument("requirements", help="requirement definitions JSON")
    parser.add_argument("--markdown-report", help="write a Markdown evidence report")
    parser.add_argument("--html-report", help="write a standalone HTML evidence report")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    scenario = load_scenario(args.scenario)
    requirements = load_requirements(args.requirements)
    evidence = [
        explain_requirement(
            scenario,
            requirement,
            sensitivity_thresholds=default_sensitivity_thresholds(requirement),
        ).to_dict()
        for requirement in requirements
    ]
    report = build_validation_report(scenario, requirements)
    write_validation_reports(
        report,
        markdown_path=args.markdown_report,
        html_path=args.html_report,
    )
    print(json.dumps({"scenario_id": scenario.scenario_id, "evidence": evidence}, indent=2))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
