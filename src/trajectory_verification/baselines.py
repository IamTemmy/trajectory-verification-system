"""Transparent motion-prediction baselines for pipeline validation."""

from __future__ import annotations

from math import atan2, cos, hypot, sin

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
    return baseline_predictions(scenario, "constant_velocity")


def baseline_predictions(
    scenario: Scenario, model: str = "constant_velocity"
) -> ScenarioPredictions:
    """Generate a transparent kinematic baseline or multimodal ensemble."""
    supported = {
        "constant_velocity",
        "constant_acceleration",
        "constant_turn_rate",
        "kinematic_ensemble",
    }
    if model not in supported:
        raise ValueError(f"unsupported baseline model: {model}")
    if scenario.current_time_index is None:
        raise ValueError("scenario current_time_index is required")
    reference = max(scenario.tracks, key=lambda track: len(track.states))
    if len(reference.states) <= OFFICIAL_PREDICTION_STEPS[-1]:
        raise ValueError("scenario does not contain the official prediction horizon")
    current_time = reference.states[scenario.current_time_index].time_s
    future_times = tuple(reference.states[index].time_s for index in OFFICIAL_PREDICTION_STEPS)
    agents = tuple(
        AgentPrediction(
            agent_id,
            _forecast_modes(
                scenario.track(agent_id), current_time, future_times, model
            ),
        )
        for agent_id in scenario.tracks_to_predict
    )
    if not agents:
        raise ValueError("scenario contains no tracks_to_predict")
    return ScenarioPredictions(scenario.scenario_id, agents)


def _forecast_modes(
    track: AgentTrack,
    current_time: float,
    future_times: tuple[float, ...],
    model: str,
) -> tuple[PredictedTrajectory, ...]:
    history = tuple(state for state in track.states if state.time_s <= current_time)
    if not history:
        raise ValueError(f"agent {track.agent_id} has no valid state at or before current time")
    models = (
        ("constant_velocity", "constant_acceleration", "constant_turn_rate")
        if model == "kinematic_ensemble" else (model,)
    )
    confidence = 1.0 / len(models)
    return tuple(
        PredictedTrajectory(
            confidence,
            _forecast_points(history, future_times, selected),
        )
        for selected in models
    )


def _forecast_points(
    history: tuple[State, ...],
    future_times: tuple[float, ...],
    model: str,
) -> tuple[PredictionPoint, ...]:
    current = history[-1]
    velocity_x, velocity_y = _velocity(history)
    if model == "constant_velocity":
        return tuple(
            PredictionPoint(
                time_s,
                current.x_m + velocity_x * (time_s - current.time_s),
                current.y_m + velocity_y * (time_s - current.time_s),
            )
            for time_s in future_times
        )
    if model == "constant_acceleration":
        acceleration_x, acceleration_y = _acceleration(history)
        return tuple(
            PredictionPoint(
                time_s,
                current.x_m + velocity_x * elapsed
                + 0.5 * acceleration_x * elapsed * elapsed,
                current.y_m + velocity_y * elapsed
                + 0.5 * acceleration_y * elapsed * elapsed,
            )
            for time_s in future_times
            for elapsed in (time_s - current.time_s,)
        )
    return _turn_rate_points(history, future_times, velocity_x, velocity_y)


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


def _acceleration(history: tuple[State, ...]) -> tuple[float, float]:
    if len(history) < 2:
        return 0.0, 0.0
    current_velocity = _velocity(history)
    previous_velocity = _velocity(history[:-1])
    elapsed = history[-1].time_s - history[-2].time_s
    raw_x = (current_velocity[0] - previous_velocity[0]) / elapsed
    raw_y = (current_velocity[1] - previous_velocity[1]) / elapsed
    magnitude = hypot(raw_x, raw_y)
    max_acceleration = 4.0
    scale = min(1.0, max_acceleration / magnitude) if magnitude else 1.0
    return raw_x * scale, raw_y * scale


def _turn_rate_points(
    history: tuple[State, ...],
    future_times: tuple[float, ...],
    velocity_x: float,
    velocity_y: float,
) -> tuple[PredictionPoint, ...]:
    current = history[-1]
    heading = (
        current.heading_rad
        if current.heading_rad is not None
        else atan2(velocity_y, velocity_x)
    )
    yaw_rate = 0.0
    if len(history) >= 2 and history[-2].heading_rad is not None:
        delta = _wrapped_angle(heading - history[-2].heading_rad)
        yaw_rate = delta / (current.time_s - history[-2].time_s)
        yaw_rate = max(-0.5, min(0.5, yaw_rate))
    speed = hypot(velocity_x, velocity_y)
    if abs(yaw_rate) < 1e-6:
        return _forecast_points(history, future_times, "constant_velocity")
    return tuple(
        PredictionPoint(
            time_s,
            current.x_m
            + speed / yaw_rate * (sin(heading + yaw_rate * elapsed) - sin(heading)),
            current.y_m
            + speed / yaw_rate * (-cos(heading + yaw_rate * elapsed) + cos(heading)),
        )
        for time_s in future_times
        for elapsed in (time_s - current.time_s,)
    )


def _wrapped_angle(value: float) -> float:
    while value > 3.141592653589793:
        value -= 6.283185307179586
    while value < -3.141592653589793:
        value += 6.283185307179586
    return value
