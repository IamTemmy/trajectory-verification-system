"""Dataset-independent multimodal trajectory prediction model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PredictionPoint:
    time_s: float
    x_m: float
    y_m: float


@dataclass(frozen=True, slots=True)
class PredictedTrajectory:
    confidence: float
    points: tuple[PredictionPoint, ...]

    def __post_init__(self) -> None:
        if not self.points:
            raise ValueError("predicted trajectory must contain at least one point")
        if any(b.time_s <= a.time_s for a, b in zip(self.points, self.points[1:])):
            raise ValueError("prediction timestamps must be strictly increasing")


@dataclass(frozen=True, slots=True)
class AgentPrediction:
    agent_id: str
    trajectories: tuple[PredictedTrajectory, ...]

    def __post_init__(self) -> None:
        if not self.agent_id:
            raise ValueError("prediction agent_id must not be empty")
        if not self.trajectories:
            raise ValueError("agent prediction must contain at least one trajectory")


@dataclass(frozen=True, slots=True)
class ScenarioPredictions:
    scenario_id: str
    agents: tuple[AgentPrediction, ...]

    def __post_init__(self) -> None:
        identifiers = [item.agent_id for item in self.agents]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("prediction agent IDs must be unique within a scenario")

