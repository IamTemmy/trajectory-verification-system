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
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--json-report")
    parser.add_argument("--markdown-report")
    parser.add_argument("--html-report")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="print aggregate sections while retaining complete report files",
    )
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
    if args.bootstrap_samples < 0:
        raise SystemExit("--bootstrap-samples must be non-negative")
    evaluation = PredictionEvaluation(
        scores,
        args.miss_threshold_m,
        args.bootstrap_samples,
        args.bootstrap_seed,
    )
    payload = evaluation.to_dict()
    if args.json_report:
        output = Path(args.json_report)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_prediction_reports(evaluation, args.markdown_report, args.html_report)
    terminal_payload = payload
    if args.summary_only:
        terminal_payload = {
            name: payload[name]
            for name in (
                "assumptions",
                "summary",
                "confidence_intervals",
                "by_object_type",
                "best_mode_counts",
            )
        }
    print(json.dumps(terminal_payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
