"""Render predicted futures against the recorded one for review.

Aggregate error reports that a model was some number of metres wrong. This
renders where it went instead, which is what a case study needs. Selecting by
``--worst`` draws the agents the evaluator scored least accurately, since those
are the cases review exists to examine.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .adapters.motion_submission import load_motion_submission
from .adapters.womd import iter_womd_scenarios
from .prediction_metrics import score_scenario_predictions
from .visualization import write_prediction_svg


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", help="official-format submission protobuf")
    parser.add_argument("shards", nargs="+", help="matching WOMD scenario shards")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--scenario-id", help="render this scenario only")
    parser.add_argument("--agent-id", help="render this agent only")
    parser.add_argument("--worst", type=int, default=0,
                        help="render the N agents with the highest minADE")
    parser.add_argument("--miss-threshold-m", type=float, default=2.0)
    args = parser.parse_args()

    scenarios = {}
    for shard in args.shards:
        for scenario in iter_womd_scenarios(shard):
            scenarios[scenario.scenario_id] = scenario
    submissions = {
        item.scenario_id: item
        for item in load_motion_submission(args.submission, scenarios.values())
    }

    ranked: list[tuple[float, str, str]] = []
    for scenario_id, predictions in submissions.items():
        if args.scenario_id and scenario_id != args.scenario_id:
            continue
        score = score_scenario_predictions(
            scenarios[scenario_id], predictions,
            miss_threshold_m=args.miss_threshold_m,
        )
        for agent in score.agents:
            if args.agent_id and agent.agent_id != args.agent_id:
                continue
            ranked.append((agent.min_ade_m, scenario_id, agent.agent_id))

    if not ranked:
        raise SystemExit("no agents matched the selection")
    ranked.sort(reverse=True)
    selected = ranked[: args.worst] if args.worst else ranked

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for min_ade, scenario_id, agent_id in selected:
        prediction = next(
            item for item in submissions[scenario_id].agents
            if item.agent_id == agent_id
        )
        path = args.out_dir / f"{scenario_id}-{agent_id}.svg"
        write_prediction_svg(scenarios[scenario_id], prediction, path)
        print(f"{path}  minADE {min_ade:.3f} m")
    print(f"rendered {len(selected)} agents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
