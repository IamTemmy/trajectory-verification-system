"""Regression gates for aggregate motion-prediction evaluation reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class PredictionComparisonPolicy:
    max_ade_increase_m: float = 0.0
    max_fde_increase_m: float = 0.0
    max_miss_rate_increase: float = 0.0
    max_coverage_decrease: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "PredictionComparisonPolicy":
        policy = cls(
            float(data.get("max_ade_increase_m", 0.0)),
            float(data.get("max_fde_increase_m", 0.0)),
            float(data.get("max_miss_rate_increase", 0.0)),
            float(data.get("max_coverage_decrease", 0.0)),
        )
        if any(value < 0 for value in asdict(policy).values()):
            raise ValueError("prediction comparison tolerances must be non-negative")
        return policy


@dataclass(frozen=True, slots=True)
class PredictionComparison:
    gate_passed: bool
    baseline: dict[str, float]
    candidate: dict[str, float]
    deltas: dict[str, float]
    violations: tuple[str, ...]
    scenario_count: int
    agent_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "gate_passed": self.gate_passed,
            "scenario_count": self.scenario_count,
            "agent_count": self.agent_count,
            "baseline": self.baseline,
            "candidate": self.candidate,
            "deltas": self.deltas,
            "violations": list(self.violations),
        }


def compare_prediction_evaluations(
    baseline_report: dict[str, object],
    candidate_report: dict[str, object],
    policy: PredictionComparisonPolicy = PredictionComparisonPolicy(),
) -> PredictionComparison:
    scenario_count, agent_count = _validate_identity(baseline_report, candidate_report)
    baseline = _summary(baseline_report)
    candidate = _summary(candidate_report)
    deltas = {name: candidate[name] - baseline[name] for name in baseline}
    violations: list[str] = []
    limits = {
        "mean_min_ade_m": policy.max_ade_increase_m,
        "mean_min_fde_m": policy.max_fde_increase_m,
        "miss_rate": policy.max_miss_rate_increase,
    }
    for metric, limit in limits.items():
        if deltas[metric] > limit + 1e-12:
            violations.append(
                f"{metric} increased by {deltas[metric]:g}, above allowed {limit:g}"
            )
    coverage_drop = -deltas["mean_ground_truth_coverage"]
    if coverage_drop > policy.max_coverage_decrease + 1e-12:
        violations.append(
            f"mean_ground_truth_coverage decreased by {coverage_drop:g}, "
            f"above allowed {policy.max_coverage_decrease:g}"
        )
    return PredictionComparison(
        not violations,
        baseline,
        candidate,
        deltas,
        tuple(violations),
        scenario_count,
        agent_count,
    )


def _summary(report: dict[str, object]) -> dict[str, float]:
    source = report.get("summary")
    if not isinstance(source, dict):
        raise ValueError("evaluation report is missing summary")
    return {
        name: float(source[name])
        for name in (
            "mean_min_ade_m",
            "mean_min_fde_m",
            "miss_rate",
            "mean_ground_truth_coverage",
        )
    }


def _validate_identity(
    baseline: dict[str, object], candidate: dict[str, object]
) -> tuple[int, int]:
    def identities(report):
        scenarios = report.get("scenarios")
        if not isinstance(scenarios, list):
            raise ValueError("evaluation report is missing scenarios")
        return {
            (str(scenario["scenario_id"]), str(agent["agent_id"]))
            for scenario in scenarios
            for agent in scenario["agents"]
        }, {str(scenario["scenario_id"]) for scenario in scenarios}

    baseline_agents, baseline_scenarios = identities(baseline)
    candidate_agents, candidate_scenarios = identities(candidate)
    if baseline_agents != candidate_agents or baseline_scenarios != candidate_scenarios:
        raise ValueError("baseline and candidate evaluation identities do not match")
    return len(baseline_scenarios), len(baseline_agents)

