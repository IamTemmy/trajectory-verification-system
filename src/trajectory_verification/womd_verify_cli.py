"""Verify one WOMD scenario and emit engineering evidence reports."""

from __future__ import annotations

import argparse
import json
from itertools import islice

from .adapters.womd import iter_womd_scenarios
from .io import load_requirements
from .reporting import build_validation_report, write_validation_reports
from .selectors import resolve_requirement_selectors
from .visualization import write_scenario_svg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify a WOMD scenario")
    parser.add_argument("shard", help="uncompressed scenario-proto TFRecord shard")
    parser.add_argument("requirements", help="requirement definitions JSON")
    parser.add_argument("--scenario-index", type=int, default=0)
    parser.add_argument("--markdown-report", help="write a Markdown evidence report")
    parser.add_argument("--html-report", help="write a standalone HTML evidence report")
    parser.add_argument("--svg-output", help="write the selected scenario trajectory SVG")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.scenario_index < 0:
        raise SystemExit("--scenario-index must be non-negative")
    selected = tuple(islice(iter_womd_scenarios(args.shard), args.scenario_index, args.scenario_index + 1))
    if not selected:
        raise SystemExit(f"scenario index {args.scenario_index} is outside the shard")
    scenario = selected[0]
    requirements = tuple(
        resolve_requirement_selectors(scenario, requirement)
        for requirement in load_requirements(args.requirements)
    )
    report = build_validation_report(scenario, requirements)
    write_validation_reports(
        report,
        markdown_path=args.markdown_report,
        html_path=args.html_report,
    )
    if args.svg_output:
        write_scenario_svg(scenario, args.svg_output)
    print(json.dumps({
        "scenario_id": report.scenario_id,
        "passed": report.passed,
        "requirements_passed": report.passed_count,
        "requirements_total": len(report.requirement_evidence),
        "quality_annotations": [item.to_dict() for item in report.quality_annotations],
        "results": [item.to_dict() for item in report.requirement_evidence],
    }, indent=2))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
