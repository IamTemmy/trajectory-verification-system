"""Reports for contextual prediction-error review evidence."""

from __future__ import annotations

from html import escape
import json
from pathlib import Path

from .risk_analysis import PredictionRiskAnalysis


def risk_to_markdown(analysis: PredictionRiskAnalysis) -> str:
    payload = analysis.to_dict()
    summary = payload["summary"]
    lines = [
        "# Prediction Risk-Context Review",
        "",
        "This report prioritizes dataset cases for engineering review. It is not "
        "a collision-probability estimate or production safety assessment.",
        "",
        f"- Agents: {summary['agents']}",
        f"- Review priorities: `{summary['review_priority']}`",
        f"- Motion classes: `{summary['motion_class']}`",
        "",
        "## Stratification by motion class",
        "",
        "| Motion | Agents | Mean minADE | Mean minFDE | Miss rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, values in payload["by_motion_class"].items():
        lines.append(
            f"| {name} | {values['agents']} | "
            f"{values['mean_min_ade_m']:.3f} m | "
            f"{values['mean_min_fde_m']:.3f} m | "
            f"{values['miss_rate']:.1%} |"
        )
    lines.extend([
        "",
        "## Highest-priority cases",
        "",
        "| Priority | Scenario | Agent | Type | Motion | minADE | minFDE | "
        "Min actor separation | SDC separation error | Context tags |",
        "|---|---|---|---|---|---:|---:|---:|---:|---|",
    ])
    for item in payload["highest_priority_cases"]:
        separation = _format_optional(item["min_actor_separation_m"])
        sdc_error = _format_optional(item["max_sdc_separation_error_m"])
        lines.append(
            f"| {item['review_priority']} | `{item['scenario_id']}` | "
            f"`{item['agent_id']}` | {item['object_type']} | "
            f"{item['motion_class']} | {item['min_ade_m']:.3f} m | "
            f"{item['min_fde_m']:.3f} m | {separation} | {sdc_error} | "
            f"{', '.join(item['risk_tags']) or 'none'} |"
        )
    lines.extend([
        "",
        "## Declared thresholds",
        "",
        "```json",
        json.dumps(payload["thresholds"], indent=2),
        "```",
        "",
    ])
    return "\n".join(lines)


def risk_to_html(analysis: PredictionRiskAnalysis) -> str:
    payload = analysis.to_dict()
    summary = payload["summary"]
    rows = "".join(
        "<tr>"
        f"<td>{escape(item['review_priority'])}</td>"
        f"<td><code>{escape(item['scenario_id'])}</code></td>"
        f"<td><code>{escape(item['agent_id'])}</code></td>"
        f"<td>{escape(item['object_type'])}</td>"
        f"<td>{escape(item['motion_class'])}</td>"
        f"<td>{item['min_ade_m']:.3f} m</td>"
        f"<td>{item['min_fde_m']:.3f} m</td>"
        f"<td>{escape(', '.join(item['risk_tags']) or 'none')}</td>"
        "</tr>"
        for item in payload["highest_priority_cases"]
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Prediction risk-context review</title><style>
body{{font:15px/1.5 system-ui;max-width:1200px;margin:40px auto;padding:0 20px;color:#172033}}
.card{{border:1px solid #ddd;border-radius:12px;padding:22px;margin:18px 0}}
table{{width:100%;border-collapse:collapse}}th,td{{padding:9px;text-align:left;border-bottom:1px solid #ddd}}
.note{{border-left:4px solid #f79009}}code{{font-family:monospace}}
</style></head><body><section class="card"><h1>Prediction Risk-Context Review</h1>
<p>{summary['agents']} agents · priorities {escape(str(summary['review_priority']))}</p></section>
<section class="card"><h2>Highest-priority cases</h2><table><thead><tr>
<th>Priority</th><th>Scenario</th><th>Agent</th><th>Type</th><th>Motion</th>
<th>minADE</th><th>minFDE</th><th>Context tags</th></tr></thead>
<tbody>{rows}</tbody></table></section>
<section class="card note"><strong>Interpretation boundary:</strong>
contextual engineering-review priority only; not collision probability or a
production safety assessment.</section></body></html>"""


def write_risk_reports(
    analysis: PredictionRiskAnalysis,
    json_path: str | Path | None = None,
    markdown_path: str | Path | None = None,
    html_path: str | Path | None = None,
) -> tuple[Path, ...]:
    written = []
    for path, content in (
        (
            json_path,
            json.dumps(analysis.to_dict(), indent=2) + "\n",
        ),
        (markdown_path, risk_to_markdown(analysis)),
        (html_path, risk_to_html(analysis)),
    ):
        if path:
            output = Path(path)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(content, encoding="utf-8")
            written.append(output)
    return tuple(written)


def _format_optional(value) -> str:
    return "n/a" if value is None else f"{value:.3f} m"
