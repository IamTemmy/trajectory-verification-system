"""Evaluate behavioral requirements against predicted futures.

Displacement error cannot say whether being wrong mattered. A forecast can be
five metres out harmlessly, as a parallel offset along an empty road, and two
metres out catastrophically, by predicting that a vehicle stops at a line it
actually crosses. The metric is the same in both cases.

This module asks a different question. It rebuilds the scenario as the model
believes it will unfold, runs the same declarative requirements used on recorded
trajectories, and compares the verdicts. What comes back is not a distance but a
behavioural claim: the model predicted a violation that did not occur, or missed
one that did.

Both views are sampled at the prediction timestamps so that derived signals are
computed over identical spacing. Comparing a future sampled every 0.5 s against
history sampled every 0.1 s would attribute the difference in sampling to the
model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .models import AgentTrack, Scenario, State
from .predictions import AgentPrediction, PredictedTrajectory
from .requirements import Requirement, RequirementResult, evaluate_requirement

AGREEMENT_CORRECT = "correct"
AGREEMENT_FALSE_ALARM = "false_alarm"
AGREEMENT_MISSED_VIOLATION = "missed_violation"
AGREEMENT_NOT_COMPARABLE = "not_comparable"


@dataclass(frozen=True, slots=True)
class PredictedRequirementOutcome:
    """One requirement judged on the recorded future and on a predicted one."""

    requirement_id: str
    agent_id: str
    mode_index: int
    confidence: float
    recorded: RequirementResult
    predicted: RequirementResult
    agreement: str

    @property
    def consequential(self) -> bool:
        """Whether the disagreement is one a reviewer must look at."""

        return self.agreement in (AGREEMENT_FALSE_ALARM, AGREEMENT_MISSED_VIOLATION)

    def to_dict(self) -> dict[str, object]:
        return {
            "requirement_id": self.requirement_id,
            "agent_id": self.agent_id,
            "mode_index": self.mode_index,
            "confidence": self.confidence,
            "agreement": self.agreement,
            "recorded": self.recorded.to_dict(),
            "predicted": self.predicted.to_dict(),
        }


def _states_at(track: AgentTrack, times: Sequence[float]) -> tuple[State, ...]:
    lookup = {round(state.time_s, 4): state for state in track.states}
    return tuple(
        state
        for state in (lookup.get(round(time_s, 4)) for time_s in times)
        if state is not None
    )


def _resample(scenario: Scenario, times: Sequence[float]) -> dict[str, tuple[State, ...]]:
    return {track.agent_id: _states_at(track, times) for track in scenario.tracks}


def _scenario_with(
    scenario: Scenario,
    sampled: dict[str, tuple[State, ...]],
    replacement: tuple[str, tuple[State, ...]] | None = None,
) -> Scenario:
    by_id = {track.agent_id: track for track in scenario.tracks}
    tracks: list[AgentTrack] = []
    for agent_id, states in sampled.items():
        if replacement is not None and agent_id == replacement[0]:
            states = replacement[1]
        if len(states) < 2:
            continue  # derived signals need at least one interval
        tracks.append(
            AgentTrack(agent_id, states, by_id[agent_id].object_type)
        )
    if not tracks:
        raise ValueError("no agent retained enough samples to evaluate")
    return Scenario(
        scenario.scenario_id,
        tuple(tracks),
        sdc_agent_id=scenario.sdc_agent_id,
        objects_of_interest=scenario.objects_of_interest,
        tracks_to_predict=scenario.tracks_to_predict,
        map_feature_count=scenario.map_feature_count,
        map_context=scenario.map_context,
    )


def _classify(recorded: RequirementResult, predicted: RequirementResult) -> str:
    if recorded.passed is None or predicted.passed is None:
        return AGREEMENT_NOT_COMPARABLE
    if recorded.passed == predicted.passed:
        return AGREEMENT_CORRECT
    # The model claims a violation the record does not contain.
    if predicted.passed is False:
        return AGREEMENT_FALSE_ALARM
    return AGREEMENT_MISSED_VIOLATION


def select_mode(
    prediction: AgentPrediction, mode_index: int | None
) -> tuple[int, PredictedTrajectory]:
    """Choose which predicted future to judge.

    The default is the most confident mode, because that is the one a consumer
    would act on. Scoring the closest mode instead would flatter the model by
    picking the trajectory with hindsight.
    """

    if mode_index is not None:
        return mode_index, prediction.trajectories[mode_index]
    best = max(
        range(len(prediction.trajectories)),
        key=lambda index: prediction.trajectories[index].confidence,
    )
    return best, prediction.trajectories[best]


def evaluate_predicted_requirements(
    ground_truth: Scenario,
    prediction: AgentPrediction,
    requirements: Sequence[Requirement],
    *,
    mode_index: int | None = None,
) -> tuple[PredictedRequirementOutcome, ...]:
    """Judge each requirement on the recorded future and on a predicted one."""

    index, mode = select_mode(prediction, mode_index)
    times = [point.time_s for point in mode.points]
    sampled = _resample(ground_truth, times)

    recorded_scenario = _scenario_with(ground_truth, sampled)
    target = ground_truth.track(prediction.agent_id)
    predicted_states = tuple(
        State(point.time_s, point.x_m, point.y_m) for point in mode.points
    )
    predicted_scenario = _scenario_with(
        ground_truth, sampled, replacement=(target.agent_id, predicted_states)
    )

    outcomes: list[PredictedRequirementOutcome] = []
    for requirement in requirements:
        try:
            recorded = evaluate_requirement(recorded_scenario, requirement)
            predicted = evaluate_requirement(predicted_scenario, requirement)
        except KeyError:
            continue  # a referenced agent did not survive resampling
        outcomes.append(
            PredictedRequirementOutcome(
                requirement_id=requirement.requirement_id,
                agent_id=prediction.agent_id,
                mode_index=index,
                confidence=mode.confidence,
                recorded=recorded,
                predicted=predicted,
                agreement=_classify(recorded, predicted),
            )
        )
    return tuple(outcomes)


def summarize(
    outcomes: Sequence[PredictedRequirementOutcome],
) -> dict[str, object]:
    """Confusion counts over behavioural verdicts, per requirement and overall."""

    per_requirement: dict[str, dict[str, int]] = {}
    totals = {
        AGREEMENT_CORRECT: 0,
        AGREEMENT_FALSE_ALARM: 0,
        AGREEMENT_MISSED_VIOLATION: 0,
        AGREEMENT_NOT_COMPARABLE: 0,
    }
    for outcome in outcomes:
        bucket = per_requirement.setdefault(
            outcome.requirement_id, dict.fromkeys(totals, 0)
        )
        bucket[outcome.agreement] += 1
        totals[outcome.agreement] += 1

    comparable = sum(
        totals[key] for key in totals if key != AGREEMENT_NOT_COMPARABLE
    )
    return {
        "evaluated": len(outcomes),
        "comparable": comparable,
        "totals": totals,
        "behavioural_accuracy": (
            totals[AGREEMENT_CORRECT] / comparable if comparable else None
        ),
        "per_requirement": per_requirement,
    }
