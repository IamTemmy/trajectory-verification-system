"""Analyze prediction errors in motion, interaction, and map context."""

from __future__ import annotations

import argparse
import json

from .adapters.motion_submission import load_motion_submission
from .adapters.womd import iter_womd_scenarios
from .prediction_metrics import score_scenario_predictions
from .risk_analysis import RiskThresholds, analyze_prediction_risk
from .risk_reporting import write_risk_reports


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prioritize contextual prediction failures for review"
    )
    parser.add_argument("submission")
    parser.add_argument("shards", nargs="+")
    parser.add_argument("--miss-threshold-m", type=float, default=2.0)
    parser.add_argument("--json-report")
    parser.add_argument("--markdown-report")
    parser.add_argument("--html-report")
    parser.add_argument("--summary-only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    scenarios = tuple(iter_womd_scenarios(args.shards))
    predictions = load_motion_submission(args.submission, scenarios)
    scenario_by_id = {item.scenario_id: item for item in scenarios}
    scores = tuple(
        score_scenario_predictions(
            scenario_by_id[item.scenario_id],
            item,
            miss_threshold_m=args.miss_threshold_m,
        )
        for item in predictions
    )
    analysis = analyze_prediction_risk(
        scenarios, predictions, scores, RiskThresholds()
    )
    write_risk_reports(
        analysis, args.json_report, args.markdown_report, args.html_report
    )
    payload = analysis.to_dict()
    if args.summary_only:
        payload = {
            "interpretation": payload["interpretation"],
            "thresholds": payload["thresholds"],
            "summary": payload["summary"],
            "by_motion_class": payload["by_motion_class"],
            "highest_priority_cases": payload["highest_priority_cases"][:5],
        }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
