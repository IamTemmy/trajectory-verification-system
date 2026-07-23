"""Reports for baseline/candidate regression gates."""

from __future__ import annotations

from html import escape
from pathlib import Path

from .regression import RegressionSummary


def regression_to_markdown(summary: RegressionSummary) -> str:
    status = "PASS" if summary.gate_passed else "FAIL"
    lines = [
        "# Trajectory Regression Report",
        "",
        f"**Gate result:** {status}",
        "",
        f"- New failures: {summary.new_failures}",
        f"- Resolved failures: {summary.resolved_failures}",
        f"- Lost applicability: {summary.lost_applicability}",
        f"- Missing candidate scenarios: {len(summary.missing_candidate_scenarios)}",
        f"- Added candidate scenarios: {len(summary.added_candidate_scenarios)}",
        "",
    ]
    if summary.violations:
        lines.extend(["## Policy violations", ""])
        lines.extend(f"- {item}" for item in summary.violations)
        lines.append("")
    lines.extend([
        "## Requirement transitions",
        "",
        "| Scenario | Requirement | Baseline | Candidate | Classification |",
        "|---|---|---:|---:|---|",
    ])
    for scenario in summary.scenarios:
        for item in scenario.transitions:
            lines.append(
                f"| `{scenario.scenario_id}` | `{item.requirement_id}` | "
                f"{item.baseline_status} | {item.candidate_status} | "
                f"{item.classification} |"
            )
    if summary.missing_candidate_scenarios:
        lines.extend(["", "## Missing candidate scenarios", ""])
        lines.extend(f"- `{item}`" for item in summary.missing_candidate_scenarios)
    lines.append("")
    return "\n".join(lines)


def regression_to_html(summary: RegressionSummary) -> str:
    status = "PASS" if summary.gate_passed else "FAIL"
    rows = "".join(
        "<tr>"
        f"<td><code>{escape(scenario.scenario_id)}</code></td>"
        f"<td><code>{escape(item.requirement_id)}</code></td>"
        f"<td>{escape(item.baseline_status)}</td><td>{escape(item.candidate_status)}</td>"
        f"<td>{escape(item.classification)}</td></tr>"
        for scenario in summary.scenarios for item in scenario.transitions
    )
    violations = "".join(f"<li>{escape(item)}</li>" for item in summary.violations) or "<li>None</li>"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Trajectory regression report</title><style>
body{{font:15px/1.5 system-ui,sans-serif;max-width:1050px;margin:40px auto;padding:0 20px;color:#172033}}
.card{{border:1px solid #ddd;border-radius:12px;padding:22px;margin:18px 0}}.pass{{color:#067647}}.fail{{color:#b42318}}
table{{width:100%;border-collapse:collapse}}th,td{{padding:10px;text-align:left;border-bottom:1px solid #ddd}}code{{font-family:monospace}}
</style></head><body><section class="card"><h1>Trajectory Regression Report</h1>
<h2 class="{status.lower()}">Gate result: {status}</h2>
<p>New failures: {summary.new_failures} · Resolved: {summary.resolved_failures} · Lost applicability: {summary.lost_applicability}</p>
</section><section class="card"><h2>Policy violations</h2><ul>{violations}</ul></section>
<section class="card"><h2>Requirement transitions</h2><table><thead><tr><th>Scenario</th><th>Requirement</th><th>Baseline</th><th>Candidate</th><th>Classification</th></tr></thead><tbody>{rows}</tbody></table></section>
</body></html>"""


def write_regression_reports(
    summary: RegressionSummary,
    markdown_path: str | Path | None = None,
    html_path: str | Path | None = None,
) -> tuple[Path, ...]:
    written: list[Path] = []
    for path, content in (
        (markdown_path, regression_to_markdown(summary)),
        (html_path, regression_to_html(summary)),
    ):
        if path:
            output = Path(path)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(content, encoding="utf-8")
            written.append(output)
    return tuple(written)
