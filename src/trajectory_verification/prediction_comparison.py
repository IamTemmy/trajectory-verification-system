"""Regression gates for aggregate motion-prediction evaluation reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil, floor
from random import Random
from statistics import fmean


@dataclass(frozen=True, slots=True)
class PredictionComparisonPolicy:
    max_ade_increase_m: float = 0.0
    max_fde_increase_m: float = 0.0
    max_miss_rate_increase: float = 0.0
    max_coverage_decrease: float = 0.0
    require_significant_ade_improvement: bool = False
    require_significant_fde_improvement: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "PredictionComparisonPolicy":
        policy = cls(
            float(data.get("max_ade_increase_m", 0.0)),
            float(data.get("max_fde_increase_m", 0.0)),
            float(data.get("max_miss_rate_increase", 0.0)),
            float(data.get("max_coverage_decrease", 0.0)),
            bool(data.get("require_significant_ade_improvement", False)),
            bool(data.get("require_significant_fde_improvement", False)),
        )
        if any(
            value < 0
            for name, value in asdict(policy).items()
            if name.startswith("max_")
        ):
            raise ValueError("prediction comparison tolerances must be non-negative")
        return policy


@dataclass(frozen=True, slots=True)
class PairedInterval:
    lower: float
    upper: float
    confidence: float = 0.95

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PredictionComparison:
    gate_passed: bool
    baseline: dict[str, float]
    candidate: dict[str, float]
    deltas: dict[str, float]
    violations: tuple[str, ...]
    scenario_count: int
    agent_count: int
    paired_confidence_intervals: dict[str, PairedInterval]
    agent_change_counts: dict[str, dict[str, int]]
    most_improved_scenarios: tuple[dict[str, object], ...]
    most_regressed_scenarios: tuple[dict[str, object], ...]
    most_improved_agents: tuple[dict[str, object], ...]
    most_regressed_agents: tuple[dict[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "gate_passed": self.gate_passed,
            "scenario_count": self.scenario_count,
            "agent_count": self.agent_count,
            "baseline": self.baseline,
            "candidate": self.candidate,
            "deltas": self.deltas,
            "violations": list(self.violations),
            "paired_confidence_intervals": {
                name: interval.to_dict()
                for name, interval in self.paired_confidence_intervals.items()
            },
            "agent_change_counts": self.agent_change_counts,
            "most_improved_scenarios": list(self.most_improved_scenarios),
            "most_regressed_scenarios": list(self.most_regressed_scenarios),
            "most_improved_agents": list(self.most_improved_agents),
            "most_regressed_agents": list(self.most_regressed_agents),
        }


def compare_prediction_evaluations(
    baseline_report: dict[str, object],
    candidate_report: dict[str, object],
    policy: PredictionComparisonPolicy = PredictionComparisonPolicy(),
    *,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 0,
) -> PredictionComparison:
    if bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be non-negative")
    if (
        bootstrap_samples == 0
        and (
            policy.require_significant_ade_improvement
            or policy.require_significant_fde_improvement
        )
    ):
        raise ValueError(
            "bootstrap_samples must be positive when significance is required"
        )
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
    agent_deltas = _agent_deltas(baseline_report, candidate_report)
    intervals = (
        {
            metric: _paired_bootstrap_interval(
                tuple(float(item[metric]) for item in agent_deltas),
                bootstrap_samples,
                bootstrap_seed + index,
            )
            for index, metric in enumerate(("min_ade_m", "min_fde_m", "miss"))
        }
        if bootstrap_samples
        else {}
    )
    if (
        policy.require_significant_ade_improvement
        and intervals["min_ade_m"].upper >= 0
    ):
        violations.append("paired minADE improvement is not significant at 95% confidence")
    if (
        policy.require_significant_fde_improvement
        and intervals["min_fde_m"].upper >= 0
    ):
        violations.append("paired minFDE improvement is not significant at 95% confidence")
    counts = {
        metric: {
            "improved": sum(float(item[metric]) < -1e-12 for item in agent_deltas),
            "unchanged": sum(abs(float(item[metric])) <= 1e-12 for item in agent_deltas),
            "regressed": sum(float(item[metric]) > 1e-12 for item in agent_deltas),
        }
        for metric in ("min_ade_m", "min_fde_m", "miss")
    }
    scenario_deltas = _scenario_deltas(agent_deltas)
    return PredictionComparison(
        not violations,
        baseline,
        candidate,
        deltas,
        tuple(violations),
        scenario_count,
        agent_count,
        intervals,
        counts,
        tuple(
            sorted(
                scenario_deltas, key=lambda item: item["mean_min_ade_delta_m"]
            )[:10]
        ),
        tuple(
            sorted(
                scenario_deltas,
                key=lambda item: item["mean_min_ade_delta_m"],
                reverse=True,
            )[:10]
        ),
        tuple(sorted(agent_deltas, key=lambda item: item["min_ade_m"])[:10]),
        tuple(sorted(agent_deltas, key=lambda item: item["min_ade_m"], reverse=True)[:10]),
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


def _agent_deltas(
    baseline: dict[str, object], candidate: dict[str, object]
) -> tuple[dict[str, object], ...]:
    def rows(report):
        return {
            (str(scenario["scenario_id"]), str(agent["agent_id"])): agent
            for scenario in report["scenarios"]
            for agent in scenario["agents"]
        }

    baseline_rows, candidate_rows = rows(baseline), rows(candidate)
    output = []
    for identity in sorted(baseline_rows):
        base, candidate = baseline_rows[identity], candidate_rows[identity]
        output.append(
            {
                "scenario_id": identity[0],
                "agent_id": identity[1],
                "object_type": str(candidate.get("object_type", "unknown")),
                "baseline_min_ade_m": float(base["min_ade_m"]),
                "candidate_min_ade_m": float(candidate["min_ade_m"]),
                "min_ade_m": float(candidate["min_ade_m"])
                - float(base["min_ade_m"]),
                "baseline_min_fde_m": float(base["min_fde_m"]),
                "candidate_min_fde_m": float(candidate["min_fde_m"]),
                "min_fde_m": float(candidate["min_fde_m"])
                - float(base["min_fde_m"]),
                "miss": float(bool(candidate["miss"])) - float(bool(base["miss"])),
                "candidate_best_mode_index": int(candidate["best_mode_index"]),
            }
        )
    return tuple(output)


def _scenario_deltas(agent_deltas) -> tuple[dict[str, object], ...]:
    scenario_ids = sorted({str(item["scenario_id"]) for item in agent_deltas})
    return tuple(
        {
            "scenario_id": scenario_id,
            "agents": len(items),
            "mean_min_ade_delta_m": fmean(
                float(item["min_ade_m"]) for item in items
            ),
            "mean_min_fde_delta_m": fmean(
                float(item["min_fde_m"]) for item in items
            ),
            "miss_rate_delta": fmean(float(item["miss"]) for item in items),
        }
        for scenario_id in scenario_ids
        for items in (
            tuple(
                item
                for item in agent_deltas
                if item["scenario_id"] == scenario_id
            ),
        )
    )


def _paired_bootstrap_interval(
    deltas: tuple[float, ...], samples: int, seed: int
) -> PairedInterval:
    if samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    random = Random(seed)
    means = sorted(
        fmean(deltas[random.randrange(len(deltas))] for _ in deltas)
        for _ in range(samples)
    )
    return PairedInterval(
        means[floor(0.025 * (samples - 1))],
        means[ceil(0.975 * (samples - 1))],
    )
