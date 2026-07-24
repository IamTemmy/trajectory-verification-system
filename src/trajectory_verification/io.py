"""JSON serialization for normalized scenarios and requirements."""

from __future__ import annotations

import json
from pathlib import Path

from .models import AgentTrack, Scenario, State
from .requirements import Requirement


def load_scenario(path: str | Path) -> Scenario:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    tracks = tuple(
        AgentTrack(
            agent_id=str(track["agent_id"]),
            object_type=str(track.get("object_type", "unknown")),
            states=tuple(
                State(
                    time_s=float(state["time_s"]),
                    x_m=float(state["x_m"]),
                    y_m=float(state["y_m"]),
                    heading_rad=(float(state["heading_rad"]) if "heading_rad" in state else None),
                    z_m=(float(state["z_m"]) if "z_m" in state else None),
                    velocity_x_mps=(
                        float(state["velocity_x_mps"]) if "velocity_x_mps" in state else None
                    ),
                    velocity_y_mps=(
                        float(state["velocity_y_mps"]) if "velocity_y_mps" in state else None
                    ),
                    length_m=(float(state["length_m"]) if "length_m" in state else None),
                    width_m=(float(state["width_m"]) if "width_m" in state else None),
                    height_m=(float(state["height_m"]) if "height_m" in state else None),
                )
                for state in track["states"]
            ),
        )
        for track in data["tracks"]
    )
    return Scenario(
        scenario_id=str(data["scenario_id"]),
        tracks=tracks,
        current_time_index=(
            int(data["current_time_index"]) if data.get("current_time_index") is not None else None
        ),
        sdc_agent_id=(str(data["sdc_agent_id"]) if data.get("sdc_agent_id") else None),
        objects_of_interest=tuple(str(item) for item in data.get("objects_of_interest", [])),
        tracks_to_predict=tuple(str(item) for item in data.get("tracks_to_predict", [])),
        map_feature_count=int(data.get("map_feature_count", 0)),
        timestamps_s=tuple(float(value) for value in data.get("timestamps_s", [])),
    )


def load_requirements(path: str | Path) -> tuple[Requirement, ...]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return tuple(Requirement.from_dict(item) for item in data["requirements"])


def load_scenario_manifest(path: str | Path) -> tuple[Scenario, ...]:
    """Load scenarios listed by paths relative to a manifest."""
    manifest_path = Path(path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = data.get("scenarios")
    if not isinstance(entries, list) or not entries:
        raise ValueError("scenario manifest must contain a non-empty 'scenarios' list")
    scenarios = tuple(load_scenario(manifest_path.parent / str(item)) for item in entries)
    identifiers = [scenario.scenario_id for scenario in scenarios]
    duplicates = sorted({item for item in identifiers if identifiers.count(item) > 1})
    if duplicates:
        raise ValueError(f"duplicate scenario IDs in manifest: {', '.join(duplicates)}")
    return scenarios
