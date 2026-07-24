"""Evaluate an official WOMD motion-prediction submission on local ground truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adapters.motion_submission import load_motion_submission
from .adapters.womd import iter_womd_scenarios
from .prediction_metrics import score_scenario_predictions
from .prediction_reporting import PredictionEvaluation, write_prediction_reports


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate WOMD motion predictions")
    parser.add_argument("submission", help="serialized MotionChallengeSubmission protobuf")
    parser.add_argument("shards", nargs="+", help="matching uncompressed WOMD scenario shards")
    parser.add_argument("--miss-threshold-m", type=float, default=2.0)
    parser.add_argument("--json-report")
    parser.add_argument("--markdown-report")
    parser.add_argument("--html-report")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    ground_truth = tuple(iter_womd_scenarios(args.shards))
    identifiers = [item.scenario_id for item in ground_truth]
    if len(identifiers) != len(set(identifiers)):
        raise SystemExit("ground-truth shards contain duplicate scenario IDs")
    truth_by_id = {item.scenario_id: item for item in ground_truth}
    predictions = load_motion_submission(args.submission, ground_truth)
    scores = tuple(
        score_scenario_predictions(
            truth_by_id[item.scenario_id],
            item,
            miss_threshold_m=args.miss_threshold_m,
        )
        for item in predictions
    )
    if not scores:
        raise SystemExit("submission contains no scenario predictions")
    evaluation = PredictionEvaluation(scores, args.miss_threshold_m)
    payload = evaluation.to_dict()
    if args.json_report:
        output = Path(args.json_report)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_prediction_reports(evaluation, args.markdown_report, args.html_report)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
