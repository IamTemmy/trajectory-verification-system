"""Failure explanations, data-quality annotations, and sensitivity analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from statistics import median
from typing import Iterable

from .models import Scenario
from .requirements import (
    FailureInterval,
    Requirement,
    RequirementResult,
    evaluate_requirement,
)


@dataclass(frozen=True, slots=True)
class DataQualityAnnotation:
    code: str
    severity: str
    message: str
    agent_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FailureExplanation:
    interval: FailureInterval
    threshold: float
    worst_value: float
    threshold_deviation: float
    narrative: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SensitivityPoint:
    threshold: float
    passed: bool
    failed_samples: int
    failed_fraction: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RequirementEvidence:
    requirement: Requirement
    result: RequirementResult
    explanations: tuple[FailureExplanation, ...]
    sensitivity: tuple[SensitivityPoint, ...]
    quality_annotations: tuple[DataQualityAnnotation, ...]
    evidence_confidence: str
    confidence_rationale: str

    def to_dict(self) -> dict[str, object]:
        return {
            "requirement": asdict(self.requirement),
            "result": self.result.to_dict(),
            "explanations": [item.to_dict() for item in self.explanations],
            "sensitivity": [item.to_dict() for item in self.sensitivity],
            "quality_annotations": [item.to_dict() for item in self.quality_annotations],
            "evidence_confidence": self.evidence_confidence,
            "confidence_rationale": self.confidence_rationale,
        }


def assess_scenario_quality(scenario: Scenario) -> tuple[DataQualityAnnotation, ...]:
    """Identify conditions that affect interpretation without changing results."""

    annotations: list[DataQualityAnnotation] = []
    short_tracks = tuple(track.agent_id for track in scenario.tracks if len(track.states) < 2)
    if short_tracks:
        annotations.append(DataQualityAnnotation(
            "INSUFFICIENT_TRACK_SAMPLES",
            "error",
            "Some tracks contain fewer than two valid states and cannot support derived metrics.",
            short_tracks,
        ))

    irregular: list[str] = []
    for track in scenario.tracks:
        steps = tuple(
            current.time_s - previous.time_s
            for previous, current in zip(track.states, track.states[1:])
        )
        if len(steps) >= 2 and max(steps) > median(steps) * 1.5:
            irregular.append(track.agent_id)
    if irregular:
        annotations.append(DataQualityAnnotation(
            "IRREGULAR_SAMPLING",
            "warning",
            "One or more tracks contain time gaps greater than 1.5 times their median step.",
            tuple(irregular),
        ))

    state_count = sum(len(track.states) for track in scenario.tracks)
    velocity_count = sum(
        state.velocity_x_mps is not None and state.velocity_y_mps is not None
        for track in scenario.tracks for state in track.states
    )
    if state_count and velocity_count < state_count:
        annotations.append(DataQualityAnnotation(
            "PARTIAL_REPORTED_VELOCITY",
            "info",
            "Reported velocity is incomplete; position-derived kinematics remain available.",
        ))
    if scenario.map_feature_count == 0:
        annotations.append(DataQualityAnnotation(
            "NO_MAP_FEATURES",
            "info",
            "No map features are attached; map-aware requirements are not applicable.",
        ))
    return tuple(annotations)


def explain_requirement(
    scenario: Scenario,
    requirement: Requirement,
    *,
    sensitivity_thresholds: Iterable[float] = (),
) -> RequirementEvidence:
    result = evaluate_requirement(scenario, requirement)
    lower_bound = requirement.operator.startswith("greater")
    relation = "below" if lower_bound else "above"
    explanations = tuple(
        _explain_interval(interval, requirement, relation)
        for interval in result.failure_intervals
    )
    sensitivity = tuple(
        _sensitivity_point(scenario, requirement, float(threshold))
        for threshold in sensitivity_thresholds
    )
    relevant_ids = {requirement.subject_agent_id, requirement.other_agent_id}
    quality = tuple(
        item for item in assess_scenario_quality(scenario)
        if not item.agent_ids or relevant_ids.intersection(item.agent_ids)
    )
    confidence, rationale = _evidence_confidence(result, quality)
    return RequirementEvidence(
        requirement, result, explanations, sensitivity, quality, confidence, rationale
    )


def default_sensitivity_thresholds(requirement: Requirement) -> tuple[float, ...]:
    """Return deterministic -10%, nominal, and +10% threshold points."""

    value = requirement.threshold
    delta = abs(value) * 0.1 or 0.1
    return (value - delta, value, value + delta)


def _explain_interval(
    interval: FailureInterval,
    requirement: Requirement,
    relation: str,
) -> FailureExplanation:
    deviation = abs(interval.worst_value - requirement.threshold)
    narrative = (
        f"{requirement.metric} was {relation} the {requirement.threshold:g} "
        f"{requirement.units} threshold from {interval.start_time_s:g} s to "
        f"{interval.end_time_s:g} s; the worst value was "
        f"{interval.worst_value:g} {requirement.units} "
        f"({deviation:g} {requirement.units} beyond the limit)."
    )
    return FailureExplanation(
        interval=interval,
        threshold=requirement.threshold,
        worst_value=interval.worst_value,
        threshold_deviation=deviation,
        narrative=narrative,
    )


def _sensitivity_point(
    scenario: Scenario, requirement: Requirement, threshold: float
) -> SensitivityPoint:
    result = evaluate_requirement(scenario, replace(requirement, threshold=threshold))
    fraction = result.failed_samples / result.evaluated_samples if result.evaluated_samples else 0.0
    return SensitivityPoint(threshold, result.passed, result.failed_samples, fraction)


def _evidence_confidence(
    result: RequirementResult,
    annotations: tuple[DataQualityAnnotation, ...],
) -> tuple[str, str]:
    """Grade evidence completeness, not real-world safety confidence."""

    severities = {item.severity for item in annotations}
    if result.evaluated_samples < 2 or "error" in severities:
        return "low", "Too few evaluated samples or a blocking data-quality error."
    if result.evaluated_samples < 5 or "warning" in severities:
        return "medium", "Evidence is usable but limited by sample count or a quality warning."
    return "high", "At least five samples were evaluated with no blocking quality warning."
