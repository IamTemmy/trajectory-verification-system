"""Resolve scenario-relative agent selectors in reusable requirements."""

from __future__ import annotations

from dataclasses import replace

from .models import Scenario
from .requirements import Requirement


def resolve_requirement_selectors(
    scenario: Scenario, requirement: Requirement
) -> Requirement:
    return replace(
        requirement,
        subject_agent_id=resolve_agent_selector(scenario, requirement.subject_agent_id),
        other_agent_id=(
            resolve_agent_selector(scenario, requirement.other_agent_id)
            if requirement.other_agent_id is not None else None
        ),
    )


def resolve_agent_selector(scenario: Scenario, value: str) -> str:
    if not value.startswith("@"):
        scenario.track(value)
        return value
    if value == "@sdc":
        if scenario.sdc_agent_id is None:
            raise ValueError("scenario has no SDC agent")
        return scenario.sdc_agent_id
    for prefix, candidates in (
        ("@prediction:", scenario.tracks_to_predict),
        ("@object_of_interest:", scenario.objects_of_interest),
    ):
        if value.startswith(prefix):
            suffix = value.removeprefix(prefix)
            try:
                index = int(suffix)
                agent_id = candidates[index]
            except (ValueError, IndexError) as exc:
                raise ValueError(f"selector cannot be resolved: {value}") from exc
            scenario.track(agent_id)
            return agent_id
    raise ValueError(f"unsupported agent selector: {value}")
