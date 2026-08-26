"""Judge predicted futures against behavioral requirements.

`evaluate-motion-predictions` reports how far a forecast was from the record.
This asks whether the forecast would have broken the rules: the same declarative
requirements, evaluated on the scenario as the model believes it will unfold,
and compared with the verdict on what actually happened.

Requirements are templates. A ``subject_agent_id`` of ``@target`` binds to each
predicted agent in turn, so one file applies across a whole shard.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .adapters.motion_submission import load_motion_submission
from .adapters.womd import iter_womd_scenarios
from .io import load_requirements
from .prediction_requirements import (
    AGREEMENT_FALSE_ALARM,
    AGREEMENT_MISSED_VIOLATION,
    evaluate_predicted_requirements,
    summarize,
)

TARGET_TOKEN = "@target"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", help="official-format submission protobuf")
    parser.add_argument("shards", nargs="+", help="matching WOMD scenario shards")
    parser.add_argument("requirements", help="requirement templates JSON")
    parser.add_argument("--json-report", type=Path)
    parser.add_argument("--mode-index", type=int, default=None,
                        help="judge this mode instead of the most confident one")
    parser.add_argument("--max-cases", type=int, default=25,
                        help="how many disagreeing cases to retain in the report")
    args = parser.parse_args()

    scenarios = {}
    for shard in args.shards:
        for scenario in iter_womd_scenarios(shard):
            scenarios[scenario.scenario_id] = scenario
    templates = load_requirements(args.requirements)

    outcomes = []
    for item in load_motion_submission(args.submission, scenarios.values()):
        scenario = scenarios[item.scenario_id]
        for prediction in item.agents:
            bound = tuple(
                replace(template, subject_agent_id=prediction.agent_id)
                if template.subject_agent_id == TARGET_TOKEN else template
                for template in templates
            )
            outcomes.extend(
                evaluate_predicted_requirements(
                    scenario, prediction, bound, mode_index=args.mode_index
                )
            )

    summary = summarize(outcomes)
    cases = [item for item in outcomes if item.consequential]
    cases.sort(key=lambda item: item.agreement != AGREEMENT_MISSED_VIOLATION)

    report = {
        "interpretation": (
            "A disagreement is a behavioural claim, not a displacement. A false "
            "alarm means the forecast implies a violation the record does not "
            "contain; a missed violation means the record contains one the "
            "forecast does not. Requirement thresholds are project-defined and "
            "carry no safety determination on their own."
        ),
        "scenarios": len(scenarios),
        "summary": summary,
        "cases": [item.to_dict() for item in cases[: args.max_cases]],
    }
    if args.json_report:
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    totals = summary["totals"]
    print(f"scenarios {len(scenarios)}  judgements {summary['evaluated']}")
    print(f"behavioural accuracy {summary['behavioural_accuracy']:.1%}")
    print(f"  false alarms      {totals[AGREEMENT_FALSE_ALARM]}")
    print(f"  missed violations {totals[AGREEMENT_MISSED_VIOLATION]}")
    for requirement_id, counts in summary["per_requirement"].items():
        comparable = sum(v for k, v in counts.items() if k != "not_comparable")
        if comparable:
            print(f"  {requirement_id}: {counts['correct'] / comparable:.1%} correct "
                  f"({counts['false_alarm']} false alarm, "
                  f"{counts['missed_violation']} missed)")
    # A missed violation is the failure a reviewer cannot afford to ignore.
    return 1 if totals[AGREEMENT_MISSED_VIOLATION] else 0


if __name__ == "__main__":
    raise SystemExit(main())
