"""Deterministic Markdown and standalone HTML engineering reports."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Sequence

from .evidence import (
    DataQualityAnnotation,
    RequirementEvidence,
    assess_scenario_quality,
    default_sensitivity_thresholds,
    explain_requirement,
)
from .models import Scenario
from .requirements import Requirement


@dataclass(frozen=True, slots=True)
class ValidationReport:
    scenario_id: str
    track_count: int
    requirement_evidence: tuple[RequirementEvidence, ...]
    quality_annotations: tuple[DataQualityAnnotation, ...]

    @property
    def passed(self) -> bool:
        return all(item.result.passed for item in self.requirement_evidence)

    @property
    def passed_count(self) -> int:
        return sum(item.result.passed for item in self.requirement_evidence)


def build_validation_report(
    scenario: Scenario,
    requirements: Sequence[Requirement],
    *,
    include_sensitivity: bool = True,
) -> ValidationReport:
    evidence = tuple(
        explain_requirement(
            scenario,
            requirement,
            sensitivity_thresholds=(
                default_sensitivity_thresholds(requirement)
                if include_sensitivity else ()
            ),
        )
        for requirement in requirements
    )
    return ValidationReport(
        scenario_id=scenario.scenario_id,
        track_count=len(scenario.tracks),
        requirement_evidence=evidence,
        quality_annotations=assess_scenario_quality(scenario),
    )


def report_to_markdown(report: ValidationReport) -> str:
    status = "PASS" if report.passed else "FAIL"
    total = len(report.requirement_evidence)
    lines = [
        f"# Trajectory Validation Report — `{report.scenario_id}`",
        "",
        f"**Overall result:** {status}",
        "",
        f"- Tracks: {report.track_count}",
        f"- Requirements passed: {report.passed_count}/{total}",
        "",
        "## Requirement summary",
        "",
        "| Requirement | Result | Confidence | Evaluated | Failed | Observed range |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in report.requirement_evidence:
        result = item.result
        observed = _range_text(result.observed_min, result.observed_max, result.units)
        lines.append(
            f"| `{item.requirement.requirement_id}` | "
            f"{'PASS' if result.passed else 'FAIL'} | {item.evidence_confidence.upper()} | "
            f"{result.evaluated_samples} | "
            f"{result.failed_samples} | {observed} |"
        )

    lines.extend(["", "## Data-quality and applicability annotations", ""])
    if report.quality_annotations:
        for item in report.quality_annotations:
            agents = f" Agents: {', '.join(item.agent_ids)}." if item.agent_ids else ""
            lines.append(
                f"- **{item.severity.upper()} — `{item.code}`:** {item.message}{agents}"
            )
    else:
        lines.append("- No quality annotations were raised.")

    for item in report.requirement_evidence:
        requirement, result = item.requirement, item.result
        lines.extend([
            "",
            f"## `{requirement.requirement_id}` — {'PASS' if result.passed else 'FAIL'}",
            "",
            requirement.description,
            "",
            f"Predicate: `{requirement.metric} {requirement.operator} "
            f"{requirement.threshold:g} {requirement.units}`.",
            "",
            f"Resolved agents: subject `{requirement.subject_agent_id}`"
            + (f", counterpart `{requirement.other_agent_id}`." if requirement.other_agent_id else "."),
            "",
            f"Evidence confidence: **{item.evidence_confidence.upper()}** — "
            f"{item.confidence_rationale}",
        ])
        if item.explanations:
            lines.extend(["", "### Failure evidence", ""])
            lines.extend(f"- {explanation.narrative}" for explanation in item.explanations)
        else:
            lines.extend(["", "No failed samples were observed."])
        if item.sensitivity:
            lines.extend([
                "",
                "### Threshold sensitivity",
                "",
                "| Threshold | Result | Failed samples | Failed fraction |",
                "|---:|---:|---:|---:|",
            ])
            for point in item.sensitivity:
                lines.append(
                    f"| {point.threshold:g} {requirement.units} | "
                    f"{'PASS' if point.passed else 'FAIL'} | {point.failed_samples} | "
                    f"{point.failed_fraction:.1%} |"
                )

    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "A project-defined threshold failure is evidence about this requirement and "
        "trajectory record. It is not, by itself, proof of unsafe real-world operation "
        "or a claim about the production Waymo Driver.",
        "",
    ])
    return "\n".join(lines)


def report_to_html(report: ValidationReport) -> str:
    status = "PASS" if report.passed else "FAIL"
    status_class = "pass" if report.passed else "fail"
    summary_rows = "".join(
        "<tr>"
        f"<td><code>{escape(item.requirement.requirement_id)}</code></td>"
        f"<td><span class='pill {'pass' if item.result.passed else 'fail'}'>"
        f"{'PASS' if item.result.passed else 'FAIL'}</span></td>"
        f"<td>{escape(item.evidence_confidence.upper())}</td>"
        f"<td>{item.result.evaluated_samples}</td><td>{item.result.failed_samples}</td>"
        f"<td>{escape(_range_text(item.result.observed_min, item.result.observed_max, item.result.units))}</td>"
        "</tr>"
        for item in report.requirement_evidence
    )
    quality = "".join(
        f"<li><strong>{escape(item.severity.upper())} — <code>{escape(item.code)}</code></strong>: "
        f"{escape(item.message)}</li>" for item in report.quality_annotations
    ) or "<li>No quality annotations were raised.</li>"
    details = "".join(_evidence_html(item) for item in report.requirement_evidence)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Trajectory validation — {escape(report.scenario_id)}</title>
<style>
:root{{--ink:#172033;--muted:#667085;--line:#e4e7ec;--bg:#f5f7fb;--card:#fff;--pass:#067647;--pass-bg:#ecfdf3;--fail:#b42318;--fail-bg:#fef3f2}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,sans-serif}}
main{{max-width:1050px;margin:0 auto;padding:48px 24px 80px}}h1{{font-size:32px;margin:0 0 8px}}h2{{margin-top:34px}}h3{{margin-top:24px}}
.muted{{color:var(--muted)}}.card{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:24px;margin:18px 0;box-shadow:0 1px 3px #10182810}}
.hero{{display:flex;justify-content:space-between;gap:24px;align-items:center}}.score{{font-size:14px;color:var(--muted)}}.score b{{font-size:28px;color:var(--ink)}}
.pill{{display:inline-block;border-radius:999px;padding:5px 10px;font-weight:700;font-size:12px}}.pill.pass{{color:var(--pass);background:var(--pass-bg)}}.pill.fail{{color:var(--fail);background:var(--fail-bg)}}
table{{width:100%;border-collapse:collapse}}th,td{{text-align:left;padding:11px;border-bottom:1px solid var(--line)}}th{{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.04em}}
code{{font-family:ui-monospace,SFMono-Regular,monospace}}.boundary{{border-left:4px solid #f79009}}ul{{padding-left:22px}}
</style></head><body><main>
<section class="card hero"><div><div class="muted">Trajectory validation report</div><h1>{escape(report.scenario_id)}</h1><span class="pill {status_class}">{status}</span></div><div class="score"><b>{report.passed_count}/{len(report.requirement_evidence)}</b><br>requirements passed<br>{report.track_count} tracks</div></section>
<section class="card"><h2>Requirement summary</h2><table><thead><tr><th>Requirement</th><th>Result</th><th>Confidence</th><th>Evaluated</th><th>Failed</th><th>Observed range</th></tr></thead><tbody>{summary_rows}</tbody></table></section>
<section class="card"><h2>Data quality and applicability</h2><ul>{quality}</ul></section>
{details}
<section class="card boundary"><h2>Interpretation boundary</h2><p>A project-defined threshold failure is evidence about this requirement and trajectory record. It is not, by itself, proof of unsafe real-world operation or a claim about the production Waymo Driver.</p></section>
</main></body></html>"""


def write_validation_reports(
    report: ValidationReport,
    *,
    markdown_path: str | Path | None = None,
    html_path: str | Path | None = None,
) -> tuple[Path, ...]:
    written: list[Path] = []
    for path, content in (
        (markdown_path, report_to_markdown(report)),
        (html_path, report_to_html(report)),
    ):
        if path is not None:
            output = Path(path)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(content, encoding="utf-8")
            written.append(output)
    return tuple(written)


def _range_text(minimum: float | None, maximum: float | None, units: str) -> str:
    if minimum is None or maximum is None:
        return "not evaluated"
    return f"{minimum:g}–{maximum:g} {units}"


def _evidence_html(item: RequirementEvidence) -> str:
    requirement, result = item.requirement, item.result
    failures = "".join(
        f"<li>{escape(explanation.narrative)}</li>" for explanation in item.explanations
    ) or "<li>No failed samples were observed.</li>"
    sensitivity = "".join(
        "<tr>"
        f"<td>{point.threshold:g} {escape(requirement.units)}</td>"
        f"<td>{'PASS' if point.passed else 'FAIL'}</td>"
        f"<td>{point.failed_samples}</td><td>{point.failed_fraction:.1%}</td>"
        "</tr>" for point in item.sensitivity
    )
    return f"""<section class="card"><h2><code>{escape(requirement.requirement_id)}</code> — {'PASS' if result.passed else 'FAIL'}</h2>
<p>{escape(requirement.description)}</p><p class="muted">{escape(requirement.metric)} {escape(requirement.operator)} {requirement.threshold:g} {escape(requirement.units)} · subject {escape(requirement.subject_agent_id)}{(' · counterpart ' + escape(requirement.other_agent_id)) if requirement.other_agent_id else ''}</p>
<p><strong>Evidence confidence: {escape(item.evidence_confidence.upper())}</strong> — {escape(item.confidence_rationale)}</p>
<h3>Failure evidence</h3><ul>{failures}</ul>
<h3>Threshold sensitivity</h3><table><thead><tr><th>Threshold</th><th>Result</th><th>Failed samples</th><th>Failed fraction</th></tr></thead><tbody>{sensitivity}</tbody></table></section>"""
