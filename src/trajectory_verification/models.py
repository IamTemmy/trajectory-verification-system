"""Dataset-independent trajectory data model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class State:
    """One valid planar agent state in SI units."""

    time_s: float
    x_m: float
    y_m: float
    heading_rad: float | None = None
    z_m: float | None = None
    velocity_x_mps: float | None = None
    velocity_y_mps: float | None = None
    length_m: float | None = None
    width_m: float | None = None
    height_m: float | None = None


@dataclass(frozen=True, slots=True)
class AgentTrack:
    """Time-ordered states belonging to one tracked agent."""

    agent_id: str
    states: tuple[State, ...]
    object_type: str = "unknown"

    def __post_init__(self) -> None:
        if not self.agent_id:
            raise ValueError("agent_id must not be empty")
        if not self.states:
            raise ValueError("an agent track must contain at least one state")
        times = [state.time_s for state in self.states]
        if any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError("state timestamps must be strictly increasing")


@dataclass(frozen=True, slots=True)
class Scenario:
    """A normalized collection of agent trajectories."""

    scenario_id: str
    tracks: tuple[AgentTrack, ...]
    current_time_index: int | None = None
    sdc_agent_id: str | None = None
    objects_of_interest: tuple[str, ...] = ()
    tracks_to_predict: tuple[str, ...] = ()
    map_feature_count: int = 0

    def __post_init__(self) -> None:
        if not self.scenario_id:
            raise ValueError("scenario_id must not be empty")
        ids = [track.agent_id for track in self.tracks]
        if len(ids) != len(set(ids)):
            raise ValueError("agent IDs must be unique within a scenario")
        if self.current_time_index is not None and self.current_time_index < 0:
            raise ValueError("current_time_index must be non-negative")

    def track(self, agent_id: str) -> AgentTrack:
        for track in self.tracks:
            if track.agent_id == agent_id:
                return track
        raise KeyError(f"unknown agent: {agent_id}")
