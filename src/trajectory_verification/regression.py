"""Deterministic baseline/candidate requirement regression comparison."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

from .models import Scenario
from .requirements import Requirement, RequirementResult, evaluate_requirement
from .selectors import resolve_requirement_selectors


def result_status(result: RequirementResult) -> str:
    if not result.applicable:
        return "NOT_APPLICABLE"
    return "PASS" if result.passed else "FAIL"


@dataclass(frozen=True, slots=True)
class RegressionPolicy:
    max_new_failures: int = 0
    fail_on_missing_candidate_scenarios: bool = True
    fail_on_lost_applicability: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "RegressionPolicy":
        policy = cls(
            max_new_failures=int(data.get("max_new_failures", 0)),
            fail_on_missing_candidate_scenarios=bool(
                data.get("fail_on_missing_candidate_scenarios", True)
            ),
            fail_on_lost_applicability=bool(data.get("fail_on_lost_applicability", True)),
        )
        if policy.max_new_failures < 0:
            raise ValueError("max_new_failures must be non-negative")
        return policy


@dataclass(frozen=True, slots=True)
class RequirementTransition:
    requirement_id: str
    baseline_status: str
    candidate_status: str
    classification: str
    baseline_failed_samples: int
    candidate_failed_samples: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ScenarioRegression:
    scenario_id: str
    status: str
    transitions: tuple[RequirementTransition, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "status": self.status,
            "transitions": [item.to_dict() for item in self.transitions],
        }


@dataclass(frozen=True, slots=True)
class RegressionSummary:
    gate_passed: bool
    new_failures: int
    resolved_failures: int
    lost_applicability: int
    missing_candidate_scenarios: tuple[str, ...]
    added_candidate_scenarios: tuple[str, ...]
    violations: tuple[str, ...]
    scenarios: tuple[ScenarioRegression, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "gate_passed": self.gate_passed,
            "counts": {
                "new_failures": self.new_failures,
                "resolved_failures": self.resolved_failures,
                "lost_applicability": self.lost_applicability,
                "missing_candidate_scenarios": len(self.missing_candidate_scenarios),
                "added_candidate_scenarios": len(self.added_candidate_scenarios),
            },
            "missing_candidate_scenarios": list(self.missing_candidate_scenarios),
            "added_candidate_scenarios": list(self.added_candidate_scenarios),
            "violations": list(self.violations),
            "scenarios": [item.to_dict() for item in self.scenarios],
        }


def compare_scenario_sets(
    baseline: Sequence[Scenario],
    candidate: Sequence[Scenario],
    requirements: Sequence[Requirement],
    policy: RegressionPolicy = RegressionPolicy(),
) -> RegressionSummary:
    baseline_by_id = {item.scenario_id: item for item in baseline}
    candidate_by_id = {item.scenario_id: item for item in candidate}
    missing = tuple(sorted(baseline_by_id.keys() - candidate_by_id.keys()))
    added = tuple(sorted(candidate_by_id.keys() - baseline_by_id.keys()))
    scenario_results: list[ScenarioRegression] = []

    for scenario_id in sorted(baseline_by_id.keys() & candidate_by_id.keys()):
        base_scenario = baseline_by_id[scenario_id]
        candidate_scenario = candidate_by_id[scenario_id]
        transitions = tuple(
            _compare_requirement(base_scenario, candidate_scenario, requirement)
            for requirement in requirements
        )
        status = "REGRESSED" if any(
            item.classification in {"NEW_FAILURE", "LOST_APPLICABILITY"}
            for item in transitions
        ) else "IMPROVED" if any(
            item.classification == "RESOLVED" for item in transitions
        ) else "UNCHANGED"
        scenario_results.append(ScenarioRegression(scenario_id, status, transitions))

    all_transitions = tuple(
        transition for scenario in scenario_results for transition in scenario.transitions
    )
    new_failures = sum(item.classification == "NEW_FAILURE" for item in all_transitions)
    resolved = sum(item.classification == "RESOLVED" for item in all_transitions)
    lost = sum(item.classification == "LOST_APPLICABILITY" for item in all_transitions)
    violations: list[str] = []
    if new_failures > policy.max_new_failures:
        violations.append(
            f"new failures {new_failures} exceed allowed maximum {policy.max_new_failures}"
        )
    if missing and policy.fail_on_missing_candidate_scenarios:
        violations.append(f"{len(missing)} baseline scenario(s) are missing from candidate")
    if lost and policy.fail_on_lost_applicability:
        violations.append(f"{lost} requirement(s) lost applicability")
    return RegressionSummary(
        gate_passed=not violations,
        new_failures=new_failures,
        resolved_failures=resolved,
        lost_applicability=lost,
        missing_candidate_scenarios=missing,
        added_candidate_scenarios=added,
        violations=tuple(violations),
        scenarios=tuple(scenario_results),
    )


def _compare_requirement(
    baseline: Scenario, candidate: Scenario, requirement: Requirement
) -> RequirementTransition:
    baseline_result = evaluate_requirement(
        baseline, resolve_requirement_selectors(baseline, requirement)
    )
    candidate_result = evaluate_requirement(
        candidate, resolve_requirement_selectors(candidate, requirement)
    )
    baseline_status = result_status(baseline_result)
    candidate_status = result_status(candidate_result)
    classification = _classify_transition(baseline_status, candidate_status)
    return RequirementTransition(
        requirement_id=requirement.requirement_id,
        baseline_status=baseline_status,
        candidate_status=candidate_status,
        classification=classification,
        baseline_failed_samples=baseline_result.failed_samples,
        candidate_failed_samples=candidate_result.failed_samples,
    )


def _classify_transition(baseline: str, candidate: str) -> str:
    if baseline == candidate:
        return f"UNCHANGED_{candidate}"
    if candidate == "FAIL":
        return "NEW_FAILURE"
    if baseline == "FAIL" and candidate == "PASS":
        return "RESOLVED"
    if candidate == "NOT_APPLICABLE":
        return "LOST_APPLICABILITY"
    if baseline == "NOT_APPLICABLE" and candidate == "PASS":
        return "NEWLY_APPLICABLE_PASS"
    raise AssertionError(f"unclassified transition: {baseline} -> {candidate}")
