"""Generate a transparent constant-velocity WOMD prediction baseline."""

from __future__ import annotations

import argparse
from itertools import islice

from .adapters.motion_submission import write_motion_submission
from .adapters.womd import iter_womd_scenarios
from .baselines import constant_velocity_predictions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate WOMD constant-velocity predictions")
    parser.add_argument("output", help="output MotionChallengeSubmission protobuf")
    parser.add_argument("shards", nargs="+", help="uncompressed WOMD scenario shards")
    parser.add_argument("--limit", type=int, help="maximum scenarios to generate")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least one")
    scenarios = iter_womd_scenarios(args.shards)
    if args.limit is not None:
        scenarios = islice(scenarios, args.limit)
    predictions = tuple(constant_velocity_predictions(item) for item in scenarios)
    if not predictions:
        raise SystemExit("no scenarios were decoded")
    write_motion_submission(predictions, args.output)
    print(f"wrote {len(predictions)} scenario prediction(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
