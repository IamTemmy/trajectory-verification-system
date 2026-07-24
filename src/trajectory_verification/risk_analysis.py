"""Contextual review evidence for motion-prediction errors.

This module prioritizes dataset cases for engineering review. It does not
estimate collision probability or real-world safety risk.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from math import atan2, hypot, pi
from statistics import fmean

from .map_metrics import _distance_to_polygon
from .models import Scenario, State
from .prediction_metrics import AgentPredictionScore, ScenarioPredictionScore
from .predictions import AgentPrediction, ScenarioPredictions


@dataclass(frozen=True, slots=True)
class RiskThresholds:
    close_interaction_m: float = 5.0
    dense_scene_agents: int = 10
    large_separation_error_m: float = 5.0
    crosswalk_context_m: float = 5.0
    control_context_m: float = 10.0
    stationary_displacement_m: float = 2.0
    turning_angle_rad: float = 0.35

    def __post_init__(self) -> None:
        if (
            self.close_interaction_m <= 0
            or self.dense_scene_agents < 1
            or self.large_separation_error_m <= 0
            or self.crosswalk_context_m <= 0
            or self.control_context_m <= 0
            or self.stationary_displacement_m < 0
            or not 0 < self.turning_angle_rad <= pi
        ):
            raise ValueError("risk-analysis thresholds are outside valid ranges")


@dataclass(frozen=True, slots=True)
class AgentRiskEvidence:
    scenario_id: str
    agent_id: str
    object_type: str
    motion_class: str
    review_priority: str
    risk_tags: tuple[str, ...]
    min_ade_m: float
    min_fde_m: float
    miss: bool
    future_displacement_m: float
    scene_density_30m: int
    min_actor_separation_m: float | None
    min_sdc_separation_m: float | None
    max_sdc_separation_error_m: float | None
    near_crosswalk: bool
    near_traffic_control: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PredictionRiskAnalysis:
    evidence: tuple[AgentRiskEvidence, ...]
    thresholds: RiskThresholds

    def to_dict(self) -> dict[str, object]:
        priority_order = {"high": 0, "medium": 1, "low": 2}
        ranked = sorted(
            self.evidence,
            key=lambda item: (
                priority_order[item.review_priority],
                -(item.max_sdc_separation_error_m or 0.0),
                -item.min_ade_m,
                item.scenario_id,
                item.agent_id,
            ),
        )
        return {
            "interpretation": (
                "contextual engineering-review priority; not collision probability "
                "or a production safety assessment"
            ),
            "thresholds": asdict(self.thresholds),
            "summary": {
                "agents": len(self.evidence),
                "review_priority": dict(sorted(Counter(
                    item.review_priority for item in self.evidence
                ).items())),
                "motion_class": dict(sorted(Counter(
                    item.motion_class for item in self.evidence
                ).items())),
                "risk_tags": dict(sorted(Counter(
                    tag for item in self.evidence for tag in item.risk_tags
                ).items())),
            },
            "by_motion_class": _stratify(self.evidence, "motion_class"),
            "by_review_priority": _stratify(self.evidence, "review_priority"),
            "highest_priority_cases": [item.to_dict() for item in ranked[:20]],
            "agents": [item.to_dict() for item in self.evidence],
        }


def analyze_prediction_risk(
    scenarios: tuple[Scenario, ...],
    predictions: tuple[ScenarioPredictions, ...],
    scores: tuple[ScenarioPredictionScore, ...],
    thresholds: RiskThresholds = RiskThresholds(),
) -> PredictionRiskAnalysis:
    scenario_by_id = {item.scenario_id: item for item in scenarios}
    prediction_by_id = {item.scenario_id: item for item in predictions}
    score_by_id = {item.scenario_id: item for item in scores}
    identities = set(scenario_by_id)
    if set(prediction_by_id) != identities or set(score_by_id) != identities:
        raise ValueError("scenario, prediction, and score identities must match")
    evidence = []
    for scenario_id in sorted(identities):
        scenario = scenario_by_id[scenario_id]
        predictions_by_agent = {
            item.agent_id: item for item in prediction_by_id[scenario_id].agents
        }
        scores_by_agent = {
            item.agent_id: item for item in score_by_id[scenario_id].agents
        }
        if set(predictions_by_agent) != set(scores_by_agent):
            raise ValueError(
                f"prediction and score agent identities differ in {scenario_id}"
            )
        evidence.extend(
            _agent_evidence(
                scenario,
                predictions_by_agent[agent_id],
                scores_by_agent[agent_id],
                thresholds,
            )
            for agent_id in sorted(predictions_by_agent)
        )
    return PredictionRiskAnalysis(tuple(evidence), thresholds)


def _agent_evidence(
    scenario: Scenario,
    prediction: AgentPrediction,
    score: AgentPredictionScore,
    thresholds: RiskThresholds,
) -> AgentRiskEvidence:
    truth = scenario.track(prediction.agent_id)
    best_mode = prediction.trajectories[score.best_mode_index]
    prediction_times = {round(point.time_s, 6) for point in best_mode.points}
    future = tuple(
        state for state in truth.states if round(state.time_s, 6) in prediction_times
    )
    if not future:
        raise ValueError(f"agent {truth.agent_id} has no aligned future states")
    current_time = (
        scenario.timestamps_s[scenario.current_time_index]
        if scenario.timestamps_s and scenario.current_time_index is not None
        else min(point.time_s for point in best_mode.points)
    )
    current = _latest_at_or_before(truth.states, current_time)
    density = _scene_density(
        scenario, truth.agent_id, current, current_time, 30.0
    )
    min_actor_separation = _minimum_actor_separation(
        scenario, truth.agent_id, future
    )
    min_sdc_separation, max_sdc_error = _sdc_interaction(
        scenario, truth.agent_id, future, best_mode
    )
    displacement = hypot(
        future[-1].x_m - current.x_m, future[-1].y_m - current.y_m
    )
    motion_class = _motion_class(
        current, future, displacement, thresholds
    )
    near_crosswalk = _near_crosswalk(scenario, future, thresholds.crosswalk_context_m)
    near_control = _near_control(scenario, future, thresholds.control_context_m)
    tags = []
    if min_actor_separation is not None and min_actor_separation <= thresholds.close_interaction_m:
        tags.append("close_interaction")
    if density >= thresholds.dense_scene_agents:
        tags.append("dense_scene")
    if (
        max_sdc_error is not None
        and max_sdc_error >= thresholds.large_separation_error_m
    ):
        tags.append("large_sdc_separation_error")
    if near_crosswalk:
        tags.append("crosswalk_context")
    if near_control:
        tags.append("traffic_control_context")
    if truth.object_type in {"pedestrian", "cyclist"}:
        tags.append("vulnerable_road_user")
    consequential_context = any(
        tag in tags
        for tag in (
            "close_interaction",
            "large_sdc_separation_error",
            "crosswalk_context",
            "traffic_control_context",
        )
    )
    priority = (
        "high"
        if score.miss and consequential_context
        else "medium"
        if score.miss or consequential_context
        else "low"
    )
    return AgentRiskEvidence(
        scenario.scenario_id,
        truth.agent_id,
        truth.object_type,
        motion_class,
        priority,
        tuple(tags),
        score.min_ade_m,
        score.min_fde_m,
        score.miss,
        displacement,
        density,
        min_actor_separation,
        min_sdc_separation,
        max_sdc_error,
        near_crosswalk,
        near_control,
    )


def _latest_at_or_before(states: tuple[State, ...], time_s: float) -> State:
    eligible = tuple(state for state in states if state.time_s <= time_s + 1e-6)
    if not eligible:
        raise ValueError("track has no state at or before the current time")
    return eligible[-1]


def _state_at(track, time_s: float) -> State | None:
    target = round(time_s, 6)
    return next(
        (state for state in track.states if round(state.time_s, 6) == target),
        None,
    )


def _scene_density(
    scenario: Scenario,
    subject_id: str,
    current: State,
    current_time: float,
    radius_m: float,
) -> int:
    return sum(
        hypot(state.x_m - current.x_m, state.y_m - current.y_m) <= radius_m
        for track in scenario.tracks
        for state in (_state_at(track, current_time),)
        if state is not None and track.agent_id != subject_id
    )


def _minimum_actor_separation(
    scenario: Scenario, subject_id: str, future: tuple[State, ...]
) -> float | None:
    separations = [
        hypot(subject.x_m - other.x_m, subject.y_m - other.y_m)
        for subject in future
        for track in scenario.tracks
        if track.agent_id != subject_id
        for other in (_state_at(track, subject.time_s),)
        if other is not None
    ]
    return min(separations) if separations else None


def _sdc_interaction(
    scenario: Scenario,
    subject_id: str,
    future: tuple[State, ...],
    mode,
) -> tuple[float | None, float | None]:
    if scenario.sdc_agent_id is None or scenario.sdc_agent_id == subject_id:
        return None, None
    sdc = scenario.track(scenario.sdc_agent_id)
    truth_by_time = {round(item.time_s, 6): item for item in future}
    separations = []
    errors = []
    for point in mode.points:
        time_key = round(point.time_s, 6)
        target = truth_by_time.get(time_key)
        sdc_state = _state_at(sdc, point.time_s)
        if target is None or sdc_state is None:
            continue
        actual = hypot(target.x_m - sdc_state.x_m, target.y_m - sdc_state.y_m)
        predicted = hypot(point.x_m - sdc_state.x_m, point.y_m - sdc_state.y_m)
        separations.append(actual)
        errors.append(abs(predicted - actual))
    return (
        min(separations) if separations else None,
        max(errors) if errors else None,
    )


def _motion_class(
    current: State,
    future: tuple[State, ...],
    displacement: float,
    thresholds: RiskThresholds,
) -> str:
    if displacement <= thresholds.stationary_displacement_m:
        return "stationary"
    points = (current,) + future
    bearings = [
        atan2(second.y_m - first.y_m, second.x_m - first.x_m)
        for first, second in zip(points, points[1:])
        if hypot(second.x_m - first.x_m, second.y_m - first.y_m) > 0.1
    ]
    if len(bearings) >= 2:
        change = abs(_wrapped_angle(bearings[-1] - bearings[0]))
        if change >= thresholds.turning_angle_rad:
            return "turning"
    return "straight"


def _wrapped_angle(value: float) -> float:
    while value > pi:
        value -= 2 * pi
    while value < -pi:
        value += 2 * pi
    return value


def _near_crosswalk(
    scenario: Scenario, future: tuple[State, ...], threshold_m: float
) -> bool:
    return any(
        _distance_to_polygon(state.x_m, state.y_m, crosswalk.polygon) <= threshold_m
        for state in future
        for crosswalk in scenario.map_context.crosswalks
    )


def _near_control(
    scenario: Scenario, future: tuple[State, ...], threshold_m: float
) -> bool:
    points = [item.position for item in scenario.map_context.stop_signs]
    points.extend(item.stop_point for item in scenario.map_context.traffic_signals)
    return any(
        hypot(state.x_m - point.x_m, state.y_m - point.y_m) <= threshold_m
        for state in future
        for point in points
    )


def _stratify(evidence: tuple[AgentRiskEvidence, ...], attribute: str):
    output = {}
    for value in sorted({getattr(item, attribute) for item in evidence}):
        selected = tuple(
            item for item in evidence if getattr(item, attribute) == value
        )
        output[value] = {
            "agents": len(selected),
            "mean_min_ade_m": fmean(item.min_ade_m for item in selected),
            "mean_min_fde_m": fmean(item.min_fde_m for item in selected),
            "miss_rate": fmean(float(item.miss) for item in selected),
        }
    return output
