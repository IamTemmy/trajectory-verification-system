import unittest

from trajectory_verification.models import AgentTrack, Scenario, State
from trajectory_verification.requirements import Requirement
from trajectory_verification.selectors import (
    resolve_agent_selector,
    resolve_requirement_selectors,
)


class SelectorTests(unittest.TestCase):
    def setUp(self):
        tracks = tuple(
            AgentTrack(agent_id, (State(0.0, 0.0, 0.0), State(1.0, 1.0, 0.0)))
            for agent_id in ("ego", "predicted", "interactive")
        )
        self.scenario = Scenario(
            "selectors", tracks, sdc_agent_id="ego",
            tracks_to_predict=("predicted",),
            objects_of_interest=("interactive",),
        )

    def test_resolves_supported_roles(self):
        self.assertEqual("ego", resolve_agent_selector(self.scenario, "@sdc"))
        self.assertEqual("predicted", resolve_agent_selector(self.scenario, "@prediction:0"))
        self.assertEqual("interactive", resolve_agent_selector(self.scenario, "@object_of_interest:0"))

    def test_resolves_requirement_without_mutating_template(self):
        template = Requirement("R", "pair", "separation", "greater_than", 1.0, "m", "@sdc", "@prediction:0")
        resolved = resolve_requirement_selectors(self.scenario, template)
        self.assertEqual("ego", resolved.subject_agent_id)
        self.assertEqual("predicted", resolved.other_agent_id)
        self.assertEqual("@sdc", template.subject_agent_id)

    def test_rejects_unavailable_selector(self):
        with self.assertRaisesRegex(ValueError, "cannot be resolved"):
            resolve_agent_selector(self.scenario, "@prediction:2")


if __name__ == "__main__":
    unittest.main()
