"""Normalize official WOMD motion-prediction submission protobufs."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..models import Scenario
from ..predictions import (
    AgentPrediction,
    PredictedTrajectory,
    PredictionPoint,
    ScenarioPredictions,
)


OFFICIAL_PREDICTION_STEPS = tuple(range(15, 91, 5))
MAX_MODES = 6


def scenario_predictions_from_proto(
    message: Any, ground_truth: Scenario
) -> ScenarioPredictions:
    """Normalize single-object predictions and attach WOMD timestamps."""
    if str(message.scenario_id) != ground_truth.scenario_id:
        raise ValueError(
            f"prediction scenario {message.scenario_id!s} does not match "
            f"ground truth {ground_truth.scenario_id}"
        )
    agents: list[AgentPrediction] = []
    predictions = message.single_predictions.predictions
    for item in predictions:
        agent_id = str(item.object_id)
        truth = ground_truth.track(agent_id)
        timestamp_by_index = _source_timestamps(ground_truth, truth)
        modes = tuple(
            _trajectory_from_proto(mode, timestamp_by_index)
            for mode in tuple(item.trajectories)[:MAX_MODES]
        )
        agents.append(AgentPrediction(agent_id, modes))
    expected = set(ground_truth.tracks_to_predict)
    actual = {item.agent_id for item in agents}
    if expected and actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"prediction agents do not match tracks_to_predict; "
            f"missing={missing}, extra={extra}"
        )
    return ScenarioPredictions(ground_truth.scenario_id, tuple(agents))


def load_motion_submission(
    path: str | Path, ground_truth_scenarios: Iterable[Scenario]
) -> tuple[ScenarioPredictions, ...]:
    """Decode a serialized official MotionChallengeSubmission."""
    from .motion_submission_proto import MotionChallengeSubmission

    message = MotionChallengeSubmission()
    message.ParseFromString(Path(path).read_bytes())
    truth_by_id = {item.scenario_id: item for item in ground_truth_scenarios}
    output: list[ScenarioPredictions] = []
    seen: set[str] = set()
    for item in message.scenario_predictions:
        scenario_id = str(item.scenario_id)
        if scenario_id in seen:
            raise ValueError(f"duplicate prediction scenario: {scenario_id}")
        if scenario_id not in truth_by_id:
            raise ValueError(f"prediction has no matching ground truth: {scenario_id}")
        seen.add(scenario_id)
        output.append(scenario_predictions_from_proto(item, truth_by_id[scenario_id]))
    return tuple(output)


def _source_timestamps(ground_truth: Scenario, truth_track) -> dict[int, float]:
    reference = max(ground_truth.tracks, key=lambda track: len(track.states))
    if len(reference.states) <= OFFICIAL_PREDICTION_STEPS[-1]:
        raise ValueError("ground-truth scenario does not contain official prediction horizon")
    reference_times = tuple(state.time_s for state in reference.states)
    truth_times = {state.time_s for state in truth_track.states}
    selected = {step: reference_times[step] for step in OFFICIAL_PREDICTION_STEPS}
    missing = [step for step, time_s in selected.items() if time_s not in truth_times]
    if missing:
        raise ValueError(
            f"ground-truth agent {truth_track.agent_id} is invalid at prediction steps {missing}"
        )
    return selected


def _trajectory_from_proto(
    mode: Any, timestamp_by_index: dict[int, float]
) -> PredictedTrajectory:
    xs = tuple(float(value) for value in mode.trajectory.center_x)
    ys = tuple(float(value) for value in mode.trajectory.center_y)
    if len(xs) != len(OFFICIAL_PREDICTION_STEPS) or len(ys) != len(xs):
        raise ValueError(
            "official WOMD trajectory must contain exactly 16 center_x and center_y values"
        )
    points = tuple(
        PredictionPoint(timestamp_by_index[step], x_m, y_m)
        for step, x_m, y_m in zip(OFFICIAL_PREDICTION_STEPS, xs, ys)
    )
    return PredictedTrajectory(float(mode.confidence), points)

