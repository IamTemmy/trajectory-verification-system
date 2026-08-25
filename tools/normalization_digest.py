"""Emit a canonical digest of normalized WOMD scenarios.

This script exists to prove that the project's hand-built schema subset and
TFRecord framing decode a shard identically to Waymo's official reader.

Both sides run the *same* normalizer, ``scenario_from_proto``. Only the proto
class and record iterator differ:

    official   waymo_open_dataset scenario_pb2.Scenario + tf.data.TFRecordDataset
    project    trajectory_verification.adapters.womd_proto.Scenario
               + trajectory_verification.adapters.womd.iter_tfrecord_records

Matching digests therefore isolate the schema subset and the framing reader as
equivalent, rather than comparing two independent normalization paths.

Usage (project reader, on any platform):

    python tools/normalization_digest.py SHARD --output project-digest.json

Usage (official reader, on Linux with the Waymo wheel installed):

    python tools/normalization_digest.py SHARD --official --output official-digest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from typing import Any, Iterator

from trajectory_verification.adapters.womd import (
    iter_tfrecord_records,
    scenario_from_proto,
)
from trajectory_verification.models import Scenario

# Coordinates are stored as double and dimensions as float, so both readers
# should agree exactly. Rounding guards against formatting drift only.
PLACES = 6


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), PLACES)


def canonical_scenario(scenario: Scenario) -> dict[str, Any]:
    """Represent every normalized field a decode could plausibly get wrong."""

    context = scenario.map_context
    return {
        "scenario_id": scenario.scenario_id,
        "current_time_index": scenario.current_time_index,
        "sdc_agent_id": scenario.sdc_agent_id,
        "objects_of_interest": list(scenario.objects_of_interest),
        "tracks_to_predict": list(scenario.tracks_to_predict),
        "map_feature_count": scenario.map_feature_count,
        "timestamps_s": [_round(value) for value in scenario.timestamps_s],
        "map_context": {
            "lanes": [
                {
                    "feature_id": lane.feature_id,
                    "lane_type": lane.lane_type,
                    "speed_limit_mph": _round(lane.speed_limit_mph),
                    "polyline": [
                        [_round(p.x_m), _round(p.y_m), _round(p.z_m)]
                        for p in lane.polyline
                    ],
                }
                for lane in context.lanes
            ],
            "stop_signs": [
                {
                    "feature_id": sign.feature_id,
                    "lane_ids": list(sign.lane_ids),
                    "position": [
                        _round(sign.position.x_m),
                        _round(sign.position.y_m),
                        _round(sign.position.z_m),
                    ],
                }
                for sign in context.stop_signs
            ],
            "crosswalks": [
                {
                    "feature_id": walk.feature_id,
                    "polygon": [
                        [_round(p.x_m), _round(p.y_m), _round(p.z_m)]
                        for p in walk.polygon
                    ],
                }
                for walk in context.crosswalks
            ],
            "traffic_signals": [
                {
                    "time_s": _round(signal.time_s),
                    "lane_id": signal.lane_id,
                    "state": signal.state,
                    "stop_point": [
                        _round(signal.stop_point.x_m),
                        _round(signal.stop_point.y_m),
                        _round(signal.stop_point.z_m),
                    ],
                }
                for signal in context.traffic_signals
            ],
        },
        "tracks": [
            {
                "agent_id": track.agent_id,
                "object_type": track.object_type,
                "states": [
                    [
                        _round(s.time_s), _round(s.x_m), _round(s.y_m),
                        _round(s.heading_rad), _round(s.z_m),
                        _round(s.velocity_x_mps), _round(s.velocity_y_mps),
                        _round(s.length_m), _round(s.width_m), _round(s.height_m),
                    ]
                    for s in track.states
                ],
            }
            for track in scenario.tracks
        ],
    }


def digest_of(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def iter_project_scenarios(shard: str) -> Iterator[Scenario]:
    from trajectory_verification.adapters.womd_proto import Scenario as ProtoScenario

    for payload in iter_tfrecord_records(shard):
        message = ProtoScenario()
        message.ParseFromString(payload)
        yield scenario_from_proto(message)


def iter_official_scenarios(shard: str) -> Iterator[Scenario]:
    """Decode with Waymo's own generated proto and TensorFlow's record reader."""

    import tensorflow as tf
    from waymo_open_dataset.protos import scenario_pb2

    for record in tf.data.TFRecordDataset([shard], compression_type=""):
        message = scenario_pb2.Scenario()
        message.ParseFromString(record.numpy())
        yield scenario_from_proto(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard", help="WOMD scenario-proto TFRecord shard")
    parser.add_argument("--official", action="store_true",
                        help="decode with the official Waymo wheel instead")
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after this many scenarios")
    parser.add_argument("--output", required=True, help="digest JSON destination")
    args = parser.parse_args()

    source = iter_official_scenarios if args.official else iter_project_scenarios
    entries = []
    for index, scenario in enumerate(source(args.shard)):
        if args.limit is not None and index >= args.limit:
            break
        canonical = canonical_scenario(scenario)
        entries.append({
            "index": index,
            "scenario_id": scenario.scenario_id,
            "track_count": len(scenario.tracks),
            "state_count": sum(len(t.states) for t in scenario.tracks),
            "digest": digest_of(canonical),
        })

    report = {
        "reader": "official" if args.official else "project",
        "shard_name": args.shard.rsplit("/", 1)[-1],
        "scenario_count": len(entries),
        "combined_digest": digest_of({"entries": entries}),
        "scenarios": entries,
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"{report['reader']}: {report['scenario_count']} scenarios")
    print(f"combined digest: {report['combined_digest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
