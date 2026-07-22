"""Inspect and normalize WOMD scenario-proto TFRecord shards."""

from __future__ import annotations

import argparse
import json

from .adapters.womd import iter_womd_scenarios
from .visualization import write_scenario_svg


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect WOMD scenario-proto TFRecords")
    parser.add_argument("shard", nargs="+", help="uncompressed scenario TFRecord shard(s)")
    parser.add_argument("--limit", type=int, default=1, help="maximum scenarios to inspect")
    parser.add_argument(
        "--svg-output",
        help="optional SVG path for the first decoded scenario",
    )
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be at least one")

    summaries = []
    for index, scenario in enumerate(iter_womd_scenarios(args.shard)):
        if index == 0 and args.svg_output:
            write_scenario_svg(scenario, args.svg_output)
        summaries.append(
            {
                "scenario_id": scenario.scenario_id,
                "track_count": len(scenario.tracks),
                "sdc_agent_id": scenario.sdc_agent_id,
                "current_time_index": scenario.current_time_index,
                "objects_of_interest": scenario.objects_of_interest,
                "tracks_to_predict": scenario.tracks_to_predict,
                "map_feature_count": scenario.map_feature_count,
            }
        )
        if index + 1 >= args.limit:
            break
    print(json.dumps({"scenarios": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
