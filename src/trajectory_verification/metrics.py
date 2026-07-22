"""Derived trajectory signals with explicit SI units."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot

from .models import AgentTrack, State


@dataclass(frozen=True, slots=True)
class Sample:
    time_s: float
    value: float


def _derivative(samples: tuple[Sample, ...]) -> tuple[Sample, ...]:
    output: list[Sample] = []
    for previous, current in zip(samples, samples[1:]):
        dt = current.time_s - previous.time_s
        if dt <= 0:
            raise ValueError("sample timestamps must be strictly increasing")
        output.append(Sample(current.time_s, (current.value - previous.value) / dt))
    return tuple(output)


def speed(track: AgentTrack) -> tuple[Sample, ...]:
    samples: list[Sample] = []
    for previous, current in zip(track.states, track.states[1:]):
        dt = current.time_s - previous.time_s
        distance = hypot(current.x_m - previous.x_m, current.y_m - previous.y_m)
        samples.append(Sample(current.time_s, distance / dt))
    return tuple(samples)


def acceleration(track: AgentTrack) -> tuple[Sample, ...]:
    return _derivative(speed(track))


def jerk(track: AgentTrack) -> tuple[Sample, ...]:
    return _derivative(acceleration(track))


def separation(subject: AgentTrack, other: AgentTrack) -> tuple[Sample, ...]:
    pairs = _aligned_states(subject, other)
    return tuple(
        Sample(a.time_s, hypot(b.x_m - a.x_m, b.y_m - a.y_m)) for a, b in pairs
    )


def closing_speed(subject: AgentTrack, other: AgentTrack) -> tuple[Sample, ...]:
    """Rate at which Euclidean separation decreases; positive means closing."""

    distance = separation(subject, other)
    return tuple(Sample(item.time_s, -item.value) for item in _derivative(distance))


def time_to_collision(subject: AgentTrack, other: AgentTrack) -> tuple[Sample, ...]:
    """Constant-closing-rate TTC; infinity when agents are not closing."""

    distances = {sample.time_s: sample.value for sample in separation(subject, other)}
    samples: list[Sample] = []
    for closing in closing_speed(subject, other):
        value = distances[closing.time_s] / closing.value if closing.value > 0 else float("inf")
        samples.append(Sample(closing.time_s, value))
    return tuple(samples)


def _aligned_states(
    subject: AgentTrack, other: AgentTrack
) -> tuple[tuple[State, State], ...]:
    other_by_time = {state.time_s: state for state in other.states}
    pairs = tuple(
        (state, other_by_time[state.time_s])
        for state in subject.states
        if state.time_s in other_by_time
    )
    if len(pairs) < 2:
        raise ValueError("tracks need at least two exactly aligned timestamps")
    return pairs
