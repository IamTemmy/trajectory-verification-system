"""Compare prediction evaluation reports and enforce quality policy."""

from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path

from .prediction_comparison import (
    PredictionComparisonPolicy,
    compare_prediction_evaluations,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gate prediction metric regressions")
    parser.add_argument("baseline_report")
    parser.add_argument("candidate_report")
    parser.add_argument("--policy")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--json-report")
    parser.add_argument("--markdown-report")
    parser.add_argument("--html-report")
    parser.add_argument("--summary-only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    load = lambda path: json.loads(Path(path).read_text(encoding="utf-8"))
    policy = (
        PredictionComparisonPolicy.from_dict(load(args.policy))
        if args.policy else PredictionComparisonPolicy()
    )
    comparison = compare_prediction_evaluations(
        load(args.baseline_report),
        load(args.candidate_report),
        policy,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    payload = comparison.to_dict()
    if args.json_report:
        output = Path(args.json_report)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.markdown_report:
        output = Path(args.markdown_report)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(comparison_to_markdown(comparison), encoding="utf-8")
    if args.html_report:
        output = Path(args.html_report)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(comparison_to_html(comparison), encoding="utf-8")
    terminal_payload = _summary_payload(comparison) if args.summary_only else payload
    print(json.dumps(terminal_payload, indent=2))
    return 0 if comparison.gate_passed else 1


def comparison_to_markdown(comparison) -> str:
    lines = [
        "# Prediction Regression Report",
        "",
        f"**Gate result:** {'PASS' if comparison.gate_passed else 'FAIL'}",
        "",
        f"- Scenarios: {comparison.scenario_count}",
        f"- Agents: {comparison.agent_count}",
        "",
        "| Metric | Baseline | Candidate | Delta |",
        "|---|---:|---:|---:|",
    ]
    for metric in comparison.baseline:
        lines.append(
            f"| `{metric}` | {comparison.baseline[metric]:.6g} | "
            f"{comparison.candidate[metric]:.6g} | "
            f"{comparison.deltas[metric]:+.6g} |"
        )
    if comparison.paired_confidence_intervals:
        lines.extend(
            [
                "",
                "## Paired agent-level uncertainty",
                "",
                "| Metric delta | 95% paired bootstrap interval | Improved | "
                "Unchanged | Regressed |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for metric, interval in comparison.paired_confidence_intervals.items():
            counts = comparison.agent_change_counts[metric]
            lines.append(
                f"| `{metric}` | [{interval.lower:+.6g}, "
                f"{interval.upper:+.6g}] | {counts['improved']} | "
                f"{counts['unchanged']} | {counts['regressed']} |"
            )
    lines.extend(_ranking_markdown("Most improved scenarios", comparison.most_improved_scenarios))
    lines.extend(_ranking_markdown("Most regressed scenarios", comparison.most_regressed_scenarios))
    lines.extend(_agent_markdown("Most improved agents", comparison.most_improved_agents))
    lines.extend(_agent_markdown("Most regressed agents", comparison.most_regressed_agents))
    if comparison.violations:
        lines.extend(["", "## Policy violations", ""])
        lines.extend(f"- {item}" for item in comparison.violations)
    lines.append("")
    return "\n".join(lines)


def _ranking_markdown(title, rows) -> list[str]:
    lines = [
        "",
        f"## {title}",
        "",
        "| Scenario | Agents | Mean minADE Δ (m) | Mean minFDE Δ (m) | Miss-rate Δ |",
        "|---|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| `{row['scenario_id']}` | {row['agents']} | "
        f"{row['mean_min_ade_delta_m']:+.6g} | "
        f"{row['mean_min_fde_delta_m']:+.6g} | "
        f"{row['miss_rate_delta']:+.6g} |"
        for row in rows
    )
    return lines


def _agent_markdown(title, rows) -> list[str]:
    lines = [
        "",
        f"## {title}",
        "",
        "| Scenario | Agent | Type | minADE Δ (m) | minFDE Δ (m) | Miss Δ | Mode |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| `{row['scenario_id']}` | `{row['agent_id']}` | "
        f"{row['object_type']} | {row['min_ade_m']:+.6g} | "
        f"{row['min_fde_m']:+.6g} | {row['miss']:+.0f} | "
        f"{row['candidate_best_mode_index']} |"
        for row in rows
    )
    return lines


def comparison_to_html(comparison) -> str:
    status = "PASS" if comparison.gate_passed else "FAIL"
    status_class = "pass" if comparison.gate_passed else "fail"
    metric_rows = "".join(
        "<tr>"
        f"<td><code>{escape(metric)}</code></td>"
        f"<td>{comparison.baseline[metric]:.6g}</td>"
        f"<td>{comparison.candidate[metric]:.6g}</td>"
        f"<td>{comparison.deltas[metric]:+.6g}</td>"
        "</tr>"
        for metric in comparison.baseline
    )
    interval_rows = "".join(
        "<tr>"
        f"<td><code>{escape(metric)}</code></td>"
        f"<td>[{interval.lower:+.6g}, {interval.upper:+.6g}]</td>"
        f"<td>{comparison.agent_change_counts[metric]['improved']}</td>"
        f"<td>{comparison.agent_change_counts[metric]['unchanged']}</td>"
        f"<td>{comparison.agent_change_counts[metric]['regressed']}</td>"
        "</tr>"
        for metric, interval in comparison.paired_confidence_intervals.items()
    )
    sections = "".join(
        _html_ranking(title, rows, agent)
        for title, rows, agent in (
            ("Most improved scenarios", comparison.most_improved_scenarios, False),
            ("Most regressed scenarios", comparison.most_regressed_scenarios, False),
            ("Most improved agents", comparison.most_improved_agents, True),
            ("Most regressed agents", comparison.most_regressed_agents, True),
        )
    )
    violations = "".join(f"<li>{escape(item)}</li>" for item in comparison.violations)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Prediction comparison</title>
<style>
body{{font:16px system-ui;line-height:1.45;margin:0;background:#f5f7fa;color:#172033}}
main{{max-width:1100px;margin:auto;padding:36px 24px}} .card{{background:white;
border:1px solid #dce2ea;border-radius:12px;padding:22px;margin:18px 0}}
table{{border-collapse:collapse;width:100%}} th,td{{padding:9px;text-align:left;
border-bottom:1px solid #e5e9ef}} th{{background:#f2f5f8}} .pass{{color:#08783e}}
.fail{{color:#b42318}} code{{font-size:.9em}} h1,h2{{line-height:1.2}}
</style></head><body><main>
<h1>Prediction comparison</h1>
<p class="{status_class}"><strong>Gate result: {status}</strong></p>
<p>{comparison.scenario_count} scenarios · {comparison.agent_count} paired agents</p>
<section class="card"><h2>Aggregate metrics</h2><table>
<thead><tr><th>Metric</th><th>Baseline</th><th>Candidate</th><th>Delta</th></tr></thead>
<tbody>{metric_rows}</tbody></table></section>
<section class="card"><h2>Paired agent-level uncertainty</h2><table>
<thead><tr><th>Metric delta</th><th>95% interval</th><th>Improved</th>
<th>Unchanged</th><th>Regressed</th></tr></thead><tbody>{interval_rows}</tbody></table></section>
{sections}
<section class="card"><h2>Policy violations</h2><ul>{violations or "<li>None</li>"}</ul></section>
</main></body></html>
"""


def _html_ranking(title, rows, agent: bool) -> str:
    if agent:
        headers = "<th>Scenario</th><th>Agent</th><th>Type</th><th>minADE Δ</th><th>minFDE Δ</th><th>Miss Δ</th><th>Mode</th>"
        body = "".join(
            "<tr>"
            f"<td><code>{escape(str(row['scenario_id']))}</code></td>"
            f"<td><code>{escape(str(row['agent_id']))}</code></td>"
            f"<td>{escape(str(row['object_type']))}</td>"
            f"<td>{row['min_ade_m']:+.6g}</td><td>{row['min_fde_m']:+.6g}</td>"
            f"<td>{row['miss']:+.0f}</td><td>{row['candidate_best_mode_index']}</td>"
            "</tr>"
            for row in rows
        )
    else:
        headers = "<th>Scenario</th><th>Agents</th><th>Mean minADE Δ</th><th>Mean minFDE Δ</th><th>Miss-rate Δ</th>"
        body = "".join(
            "<tr>"
            f"<td><code>{escape(str(row['scenario_id']))}</code></td>"
            f"<td>{row['agents']}</td><td>{row['mean_min_ade_delta_m']:+.6g}</td>"
            f"<td>{row['mean_min_fde_delta_m']:+.6g}</td>"
            f"<td>{row['miss_rate_delta']:+.6g}</td></tr>"
            for row in rows
        )
    return f'<section class="card"><h2>{escape(title)}</h2><table><thead><tr>{headers}</tr></thead><tbody>{body}</tbody></table></section>'


def _summary_payload(comparison) -> dict[str, object]:
    return {
        "gate_passed": comparison.gate_passed,
        "scenario_count": comparison.scenario_count,
        "agent_count": comparison.agent_count,
        "deltas": comparison.deltas,
        "paired_confidence_intervals": {
            name: interval.to_dict()
            for name, interval in comparison.paired_confidence_intervals.items()
        },
        "agent_change_counts": comparison.agent_change_counts,
        "violations": list(comparison.violations),
    }


if __name__ == "__main__":
    raise SystemExit(main())
