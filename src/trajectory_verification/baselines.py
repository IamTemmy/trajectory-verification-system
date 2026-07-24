"""Transparent motion-prediction baselines for pipeline validation."""

from __future__ import annotations

from .adapters.motion_submission import OFFICIAL_PREDICTION_STEPS
from .models import AgentTrack, Scenario, State
from .predictions import (
    AgentPrediction,
    PredictedTrajectory,
    PredictionPoint,
    ScenarioPredictions,
)


def constant_velocity_predictions(scenario: Scenario) -> ScenarioPredictions:
    """Forecast tracks_to_predict using only state at or before current time."""
    if scenario.current_time_index is None:
        raise ValueError("scenario current_time_index is required")
    reference = max(scenario.tracks, key=lambda track: len(track.states))
    if len(reference.states) <= OFFICIAL_PREDICTION_STEPS[-1]:
        raise ValueError("scenario does not contain the official prediction horizon")
    current_time = reference.states[scenario.current_time_index].time_s
    future_times = tuple(reference.states[index].time_s for index in OFFICIAL_PREDICTION_STEPS)
    agents = tuple(
        AgentPrediction(agent_id, (
            _forecast_track(scenario.track(agent_id), current_time, future_times),
        ))
        for agent_id in scenario.tracks_to_predict
    )
    if not agents:
        raise ValueError("scenario contains no tracks_to_predict")
    return ScenarioPredictions(scenario.scenario_id, agents)


def _forecast_track(
    track: AgentTrack, current_time: float, future_times: tuple[float, ...]
) -> PredictedTrajectory:
    history = tuple(state for state in track.states if state.time_s <= current_time)
    if not history:
        raise ValueError(f"agent {track.agent_id} has no valid state at or before current time")
    current = history[-1]
    velocity_x, velocity_y = _velocity(history)
    points = tuple(
        PredictionPoint(
            time_s,
            current.x_m + velocity_x * (time_s - current.time_s),
            current.y_m + velocity_y * (time_s - current.time_s),
        )
        for time_s in future_times
    )
    return PredictedTrajectory(1.0, points)


def _velocity(history: tuple[State, ...]) -> tuple[float, float]:
    current = history[-1]
    if current.velocity_x_mps is not None and current.velocity_y_mps is not None:
        return current.velocity_x_mps, current.velocity_y_mps
    if len(history) < 2:
        raise ValueError("constant-velocity baseline needs velocity or two historical states")
    previous = history[-2]
    elapsed = current.time_s - previous.time_s
    return (
        (current.x_m - previous.x_m) / elapsed,
        (current.y_m - previous.y_m) / elapsed,
    )

