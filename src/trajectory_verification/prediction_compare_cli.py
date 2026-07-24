"""Compare prediction evaluation reports and enforce quality policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .prediction_comparison import (
    PredictionComparisonPolicy,
    compare_prediction_evaluations,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gate prediction metric regressions")
    parser.add_argument("baseline_report")
    parser.add_argument("candidate_report")
    parser.add_argument("--policy")
    parser.add_argument("--json-report")
    parser.add_argument("--markdown-report")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    load = lambda path: json.loads(Path(path).read_text(encoding="utf-8"))
    policy = (
        PredictionComparisonPolicy.from_dict(load(args.policy))
        if args.policy else PredictionComparisonPolicy()
    )
    comparison = compare_prediction_evaluations(
        load(args.baseline_report), load(args.candidate_report), policy
    )
    payload = comparison.to_dict()
    if args.json_report:
        output = Path(args.json_report)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.markdown_report:
        output = Path(args.markdown_report)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(_to_markdown(comparison), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if comparison.gate_passed else 1


def _to_markdown(comparison) -> str:
    lines = [
        "# Prediction Regression Report",
        "",
        f"**Gate result:** {'PASS' if comparison.gate_passed else 'FAIL'}",
        "",
        f"- Scenarios: {comparison.scenario_count}",
        f"- Agents: {comparison.agent_count}",
        "",
        "| Metric | Baseline | Candidate | Delta |",
        "|---|---:|---:|---:|",
    ]
    for metric in comparison.baseline:
        lines.append(
            f"| `{metric}` | {comparison.baseline[metric]:.6g} | "
            f"{comparison.candidate[metric]:.6g} | "
            f"{comparison.deltas[metric]:+.6g} |"
        )
    if comparison.violations:
        lines.extend(["", "## Policy violations", ""])
        lines.extend(f"- {item}" for item in comparison.violations)
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
