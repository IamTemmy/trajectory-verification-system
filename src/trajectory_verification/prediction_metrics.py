"""Multimodal trajectory-prediction metrics with explicit assumptions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import hypot
from statistics import fmean

from .models import Scenario
from .predictions import AgentPrediction, PredictedTrajectory, ScenarioPredictions


@dataclass(frozen=True, slots=True)
class AgentPredictionScore:
    agent_id: str
    modes: int
    evaluated_points: int
    expected_points: int
    min_ade_m: float
    min_fde_m: float
    miss: bool
    best_mode_index: int

    @property
    def ground_truth_coverage(self) -> float:
        return self.evaluated_points / self.expected_points

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["ground_truth_coverage"] = self.ground_truth_coverage
        return payload


@dataclass(frozen=True, slots=True)
class ScenarioPredictionScore:
    scenario_id: str
    agents: tuple[AgentPredictionScore, ...]

    @property
    def mean_min_ade_m(self) -> float:
        return fmean(item.min_ade_m for item in self.agents)

    @property
    def mean_min_fde_m(self) -> float:
        return fmean(item.min_fde_m for item in self.agents)

    @property
    def miss_rate(self) -> float:
        return fmean(float(item.miss) for item in self.agents)

    def to_dict(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "mean_min_ade_m": self.mean_min_ade_m,
            "mean_min_fde_m": self.mean_min_fde_m,
            "miss_rate": self.miss_rate,
            "agents": [item.to_dict() for item in self.agents],
        }


def score_scenario_predictions(
    ground_truth: Scenario,
    predictions: ScenarioPredictions,
    *,
    miss_threshold_m: float = 2.0,
) -> ScenarioPredictionScore:
    """Score each agent using the mode with minimum average displacement error.

    ``miss_threshold_m`` is a project-defined final-displacement threshold, not
    Waymo's official object-type and speed-scaled miss-rate configuration.
    """
    if miss_threshold_m < 0:
        raise ValueError("miss_threshold_m must be non-negative")
    if predictions.scenario_id != ground_truth.scenario_id:
        raise ValueError("prediction and ground-truth scenario IDs differ")
    scores = tuple(
        _score_agent(ground_truth, agent, miss_threshold_m)
        for agent in predictions.agents
    )
    if not scores:
        raise ValueError("scenario contains no agent predictions")
    return ScenarioPredictionScore(ground_truth.scenario_id, scores)


def _score_agent(
    ground_truth: Scenario,
    prediction: AgentPrediction,
    miss_threshold_m: float,
) -> AgentPredictionScore:
    truth = ground_truth.track(prediction.agent_id)
    truth_by_time = {round(state.time_s, 6): state for state in truth.states}
    mode_errors: list[tuple[float, float, int]] = []
    for mode in prediction.trajectories:
        errors = _mode_errors(mode, truth_by_time)
        mode_errors.append((fmean(errors), errors[-1], len(errors)))
    best_index = min(range(len(mode_errors)), key=lambda index: mode_errors[index][0])
    ade, _, points = mode_errors[best_index]
    fde = min(item[1] for item in mode_errors)
    return AgentPredictionScore(
        prediction.agent_id,
        len(prediction.trajectories),
        points,
        len(prediction.trajectories[0].points),
        ade,
        fde,
        fde > miss_threshold_m,
        best_index,
    )


def _mode_errors(
    mode: PredictedTrajectory, truth_by_time: dict[float, object]
) -> tuple[float, ...]:
    errors: list[float] = []
    for point in mode.points:
        truth = truth_by_time.get(round(point.time_s, 6))
        if truth is not None:
            errors.append(hypot(point.x_m - truth.x_m, point.y_m - truth.y_m))
    if not errors:
        raise ValueError("prediction has no valid aligned future ground truth")
    return tuple(errors)
