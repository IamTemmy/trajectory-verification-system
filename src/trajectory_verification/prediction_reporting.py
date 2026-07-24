"""Batch summaries for motion-prediction evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from html import escape
from math import ceil, floor
from pathlib import Path
from random import Random
from statistics import fmean

from .prediction_metrics import ScenarioPredictionScore


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    lower: float
    upper: float
    confidence: float = 0.95

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PredictionEvaluation:
    scenarios: tuple[ScenarioPredictionScore, ...]
    miss_threshold_m: float
    bootstrap_samples: int = 1000
    bootstrap_seed: int = 0

    @property
    def agent_count(self) -> int:
        return sum(len(item.agents) for item in self.scenarios)

    @property
    def mean_min_ade_m(self) -> float:
        return fmean(agent.min_ade_m for item in self.scenarios for agent in item.agents)

    @property
    def mean_min_fde_m(self) -> float:
        return fmean(agent.min_fde_m for item in self.scenarios for agent in item.agents)

    @property
    def miss_rate(self) -> float:
        return fmean(float(agent.miss) for item in self.scenarios for agent in item.agents)

    @property
    def mean_ground_truth_coverage(self) -> float:
        return fmean(
            agent.ground_truth_coverage for item in self.scenarios for agent in item.agents
        )

    @property
    def agents(self):
        return tuple(agent for scenario in self.scenarios for agent in scenario.agents)

    def confidence_intervals(self) -> dict[str, ConfidenceInterval]:
        if self.bootstrap_samples < 1:
            return {}
        extractors = {
            "mean_min_ade_m": lambda item: item.min_ade_m,
            "mean_min_fde_m": lambda item: item.min_fde_m,
            "miss_rate": lambda item: float(item.miss),
        }
        return {
            name: _bootstrap_mean_interval(
                self.agents, extractor, self.bootstrap_samples, self.bootstrap_seed + index
            )
            for index, (name, extractor) in enumerate(extractors.items())
        }

    def by_object_type(self) -> dict[str, dict[str, float | int]]:
        output: dict[str, dict[str, float | int]] = {}
        for object_type in sorted({item.object_type for item in self.agents}):
            agents = tuple(item for item in self.agents if item.object_type == object_type)
            output[object_type] = {
                "agents": len(agents),
                "mean_min_ade_m": fmean(item.min_ade_m for item in agents),
                "mean_min_fde_m": fmean(item.min_fde_m for item in agents),
                "miss_rate": fmean(float(item.miss) for item in agents),
            }
        return output

    def best_mode_counts(self) -> dict[str, int]:
        indices = sorted({item.best_mode_index for item in self.agents})
        return {
            str(index): sum(item.best_mode_index == index for item in self.agents)
            for index in indices
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "assumptions": {
                "miss_threshold_m": self.miss_threshold_m,
                "metric_scope": "project-defined diagnostic; not official Waymo challenge scoring",
                "mode_selection": "minADE and minFDE minimized independently across modes",
            },
            "summary": {
                "scenarios": len(self.scenarios),
                "agents": self.agent_count,
                "mean_min_ade_m": self.mean_min_ade_m,
                "mean_min_fde_m": self.mean_min_fde_m,
                "miss_rate": self.miss_rate,
                "mean_ground_truth_coverage": self.mean_ground_truth_coverage,
            },
            "confidence_intervals": {
                name: interval.to_dict()
                for name, interval in self.confidence_intervals().items()
            },
            "by_object_type": self.by_object_type(),
            "best_mode_counts": self.best_mode_counts(),
            "worst_agents_by_min_ade": [
                {
                    "scenario_id": scenario.scenario_id,
                    **agent.to_dict(),
                }
                for scenario, agent in sorted(
                    (
                        (scenario, agent)
                        for scenario in self.scenarios
                        for agent in scenario.agents
                    ),
                    key=lambda pair: pair[1].min_ade_m,
                    reverse=True,
                )[:10]
            ],
            "scenarios": [item.to_dict() for item in self.scenarios],
        }


def evaluation_to_markdown(evaluation: PredictionEvaluation) -> str:
    lines = [
        "# Motion Prediction Evaluation",
        "",
        "This is a project-defined diagnostic, not official Waymo challenge scoring.",
        "",
        f"- Scenarios: {len(evaluation.scenarios)}",
        f"- Predicted agents: {evaluation.agent_count}",
        f"- Mean minADE: {evaluation.mean_min_ade_m:.3f} m",
        f"- Mean minFDE: {evaluation.mean_min_fde_m:.3f} m",
        f"- Miss rate at {evaluation.miss_threshold_m:g} m: {evaluation.miss_rate:.1%}",
        f"- Mean valid ground-truth coverage: {evaluation.mean_ground_truth_coverage:.1%}",
        "",
        "## Bootstrap uncertainty",
        "",
    ]
    intervals = evaluation.confidence_intervals()
    if intervals:
        lines.extend([
            "| Metric | Estimate | 95% interval |",
            "|---|---:|---:|",
            f"| Mean minADE | {evaluation.mean_min_ade_m:.3f} m | "
            f"{intervals['mean_min_ade_m'].lower:.3f}–{intervals['mean_min_ade_m'].upper:.3f} m |",
            f"| Mean minFDE | {evaluation.mean_min_fde_m:.3f} m | "
            f"{intervals['mean_min_fde_m'].lower:.3f}–{intervals['mean_min_fde_m'].upper:.3f} m |",
            f"| Miss rate | {evaluation.miss_rate:.1%} | "
            f"{intervals['miss_rate'].lower:.1%}–{intervals['miss_rate'].upper:.1%} |",
            "",
        ])
    lines.extend([
        "## Object-type breakdown",
        "",
        "| Object type | Agents | Mean minADE | Mean minFDE | Miss rate |",
        "|---|---:|---:|---:|---:|",
    ])
    for object_type, values in evaluation.by_object_type().items():
        lines.append(
            f"| {object_type} | {values['agents']} | "
            f"{values['mean_min_ade_m']:.3f} m | {values['mean_min_fde_m']:.3f} m | "
            f"{values['miss_rate']:.1%} |"
        )
    lines.extend([
        "",
        f"Best-mode index counts: `{evaluation.best_mode_counts()}`.",
        "",
        "## Scenario ranking",
        "",
        "| Scenario | Agents | Mean minADE | Mean minFDE | Miss rate | Coverage |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for item in sorted(evaluation.scenarios, key=lambda score: score.mean_min_ade_m, reverse=True):
        lines.append(
            f"| `{item.scenario_id}` | {len(item.agents)} | "
            f"{item.mean_min_ade_m:.3f} m | {item.mean_min_fde_m:.3f} m | "
            f"{item.miss_rate:.1%} | "
            f"{fmean(agent.ground_truth_coverage for agent in item.agents):.1%} |"
        )
    lines.append("")
    return "\n".join(lines)


def _bootstrap_mean_interval(
    items: tuple,
    extractor,
    samples: int,
    seed: int,
) -> ConfidenceInterval:
    if not items:
        raise ValueError("cannot bootstrap an empty evaluation")
    random = Random(seed)
    means = sorted(
        fmean(extractor(items[random.randrange(len(items))]) for _ in items)
        for _ in range(samples)
    )
    lower_index = floor(0.025 * (samples - 1))
    upper_index = ceil(0.975 * (samples - 1))
    return ConfidenceInterval(means[lower_index], means[upper_index])


def evaluation_to_html(evaluation: PredictionEvaluation) -> str:
    rows = "".join(
        f"<tr><td><code>{escape(item.scenario_id)}</code></td>"
        f"<td>{len(item.agents)}</td><td>{item.mean_min_ade_m:.3f} m</td>"
        f"<td>{item.mean_min_fde_m:.3f} m</td><td>{item.miss_rate:.1%}</td>"
        f"<td>{fmean(agent.ground_truth_coverage for agent in item.agents):.1%}</td></tr>"
        for item in sorted(
            evaluation.scenarios, key=lambda score: score.mean_min_ade_m, reverse=True
        )
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Motion prediction evaluation</title><style>
body{{font:15px/1.5 system-ui,sans-serif;max-width:1000px;margin:40px auto;padding:0 20px;color:#172033}}
.card{{border:1px solid #ddd;border-radius:12px;padding:22px;margin:18px 0}}table{{width:100%;border-collapse:collapse}}
th,td{{padding:10px;text-align:left;border-bottom:1px solid #ddd}}code{{font-family:monospace}}.note{{border-left:4px solid #f79009}}
</style></head><body><section class="card"><h1>Motion Prediction Evaluation</h1>
<p>{len(evaluation.scenarios)} scenarios · {evaluation.agent_count} agents</p>
<p>Mean minADE {evaluation.mean_min_ade_m:.3f} m · Mean minFDE {evaluation.mean_min_fde_m:.3f} m · Miss rate {evaluation.miss_rate:.1%} · Valid ground-truth coverage {evaluation.mean_ground_truth_coverage:.1%}</p></section>
<section class="card"><table><thead><tr><th>Scenario</th><th>Agents</th><th>Mean minADE</th><th>Mean minFDE</th><th>Miss rate</th><th>Coverage</th></tr></thead><tbody>{rows}</tbody></table></section>
<section class="card note"><strong>Interpretation boundary:</strong> project-defined diagnostic using a {evaluation.miss_threshold_m:g} m final-displacement threshold; not official Waymo challenge scoring.</section>
</body></html>"""


def write_prediction_reports(
    evaluation: PredictionEvaluation,
    markdown_path: str | Path | None = None,
    html_path: str | Path | None = None,
) -> tuple[Path, ...]:
    written: list[Path] = []
    for path, content in (
        (markdown_path, evaluation_to_markdown(evaluation)),
        (html_path, evaluation_to_html(evaluation)),
    ):
        if path:
            output = Path(path)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(content, encoding="utf-8")
            written.append(output)
    return tuple(written)
