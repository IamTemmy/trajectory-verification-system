"""Declarative scalar requirement evaluation and failure localization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from operator import ge, gt, le, lt
from typing import Callable

from .metrics import Sample, acceleration, jerk, separation, speed, time_to_collision
from .map_metrics import (
    NotApplicableError,
    crosswalk_proximity,
    lane_lateral_offset,
    red_stop_line_violation,
    stop_sign_crossing_speed,
    vru_crosswalk_proximity,
)
from .models import Scenario


COMPARATORS: dict[str, Callable[[float, float], bool]] = {
    "greater_than": gt,
    "greater_than_or_equal": ge,
    "less_than": lt,
    "less_than_or_equal": le,
}


@dataclass(frozen=True, slots=True)
class Requirement:
    requirement_id: str
    description: str
    metric: str
    operator: str
    threshold: float
    units: str
    subject_agent_id: str
    other_agent_id: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "Requirement":
        return cls(
            requirement_id=str(data["id"]),
            description=str(data["description"]),
            metric=str(data["metric"]),
            operator=str(data["operator"]),
            threshold=float(data["threshold"]),
            units=str(data["units"]),
            subject_agent_id=str(data["subject_agent_id"]),
            other_agent_id=(str(data["other_agent_id"]) if data.get("other_agent_id") else None),
        )


@dataclass(frozen=True, slots=True)
class FailureInterval:
    start_time_s: float
    end_time_s: float
    sample_count: int
    worst_value: float


@dataclass(frozen=True, slots=True)
class RequirementResult:
    requirement_id: str
    passed: bool | None
    evaluated_samples: int
    failed_samples: int
    failure_intervals: tuple[FailureInterval, ...]
    observed_min: float | None
    observed_max: float | None
    units: str
    applicable: bool = True
    not_applicable_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_requirement(scenario: Scenario, requirement: Requirement) -> RequirementResult:
    if requirement.operator not in COMPARATORS:
        raise ValueError(f"unsupported operator: {requirement.operator}")
    try:
        samples = metric_samples(scenario, requirement)
    except NotApplicableError as exc:
        return RequirementResult(
            requirement_id=requirement.requirement_id,
            passed=None,
            evaluated_samples=0,
            failed_samples=0,
            failure_intervals=(),
            observed_min=None,
            observed_max=None,
            units=requirement.units,
            applicable=False,
            not_applicable_reason=str(exc),
        )
    compare = COMPARATORS[requirement.operator]
    failures = tuple(sample for sample in samples if not compare(sample.value, requirement.threshold))
    intervals = _localize_failures(failures, requirement.operator)
    values = tuple(sample.value for sample in samples)
    return RequirementResult(
        requirement_id=requirement.requirement_id,
        passed=not failures,
        evaluated_samples=len(samples),
        failed_samples=len(failures),
        failure_intervals=intervals,
        observed_min=min(values) if values else None,
        observed_max=max(values) if values else None,
        units=requirement.units,
    )


def metric_samples(scenario: Scenario, requirement: Requirement) -> tuple[Sample, ...]:
    """Return the derived samples evaluated by a requirement."""
    subject = scenario.track(requirement.subject_agent_id)
    if requirement.metric == "speed":
        return speed(subject)
    if requirement.metric == "acceleration":
        return acceleration(subject)
    if requirement.metric == "jerk":
        return jerk(subject)
    if requirement.metric == "lane_lateral_offset":
        return lane_lateral_offset(scenario, subject)
    if requirement.metric == "crosswalk_proximity":
        return crosswalk_proximity(scenario, subject)
    if requirement.metric == "vru_crosswalk_proximity":
        return vru_crosswalk_proximity(scenario, subject)
    if requirement.metric == "red_stop_line_violation":
        return red_stop_line_violation(scenario, subject)
    if requirement.metric == "stop_sign_crossing_speed":
        return stop_sign_crossing_speed(scenario, subject)
    if requirement.other_agent_id is None:
        raise ValueError(f"metric {requirement.metric} requires other_agent_id")
    other = scenario.track(requirement.other_agent_id)
    if requirement.metric == "separation":
        return separation(subject, other)
    if requirement.metric == "time_to_collision":
        return time_to_collision(subject, other)
    raise ValueError(f"unsupported metric: {requirement.metric}")


def _localize_failures(
    failures: tuple[Sample, ...], operator_name: str
) -> tuple[FailureInterval, ...]:
    if not failures:
        return ()
    groups: list[list[Sample]] = [[failures[0]]]
    nominal_step: float | None = None
    if len(failures) > 1:
        positive_steps = [
            b.time_s - a.time_s
            for a, b in zip(failures, failures[1:])
            if b.time_s > a.time_s
        ]
        nominal_step = min(positive_steps) if positive_steps else None
    for sample in failures[1:]:
        previous = groups[-1][-1]
        contiguous = nominal_step is None or sample.time_s - previous.time_s <= nominal_step * 1.5
        if contiguous:
            groups[-1].append(sample)
        else:
            groups.append([sample])
    lower_bound = operator_name.startswith("greater")
    return tuple(
        FailureInterval(
            start_time_s=group[0].time_s,
            end_time_s=group[-1].time_s,
            sample_count=len(group),
            worst_value=min(item.value for item in group) if lower_bound else max(item.value for item in group),
        )
        for group in groups
    )
