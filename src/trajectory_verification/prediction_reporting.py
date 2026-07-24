"""Batch summaries for motion-prediction evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
from statistics import fmean

from .prediction_metrics import ScenarioPredictionScore


@dataclass(frozen=True, slots=True)
class PredictionEvaluation:
    scenarios: tuple[ScenarioPredictionScore, ...]
    miss_threshold_m: float

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
        "| Scenario | Agents | Mean minADE | Mean minFDE | Miss rate | Coverage |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in sorted(evaluation.scenarios, key=lambda score: score.mean_min_ade_m, reverse=True):
        lines.append(
            f"| `{item.scenario_id}` | {len(item.agents)} | "
            f"{item.mean_min_ade_m:.3f} m | {item.mean_min_fde_m:.3f} m | "
            f"{item.miss_rate:.1%} | "
            f"{fmean(agent.ground_truth_coverage for agent in item.agents):.1%} |"
        )
    lines.append("")
    return "\n".join(lines)


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
