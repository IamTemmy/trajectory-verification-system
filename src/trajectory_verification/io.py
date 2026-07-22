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
                )
                for state in track["states"]
            ),
        )
        for track in data["tracks"]
    )
    return Scenario(scenario_id=str(data["scenario_id"]), tracks=tracks)


def load_requirements(path: str | Path) -> tuple[Requirement, ...]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return tuple(Requirement.from_dict(item) for item in data["requirements"])
