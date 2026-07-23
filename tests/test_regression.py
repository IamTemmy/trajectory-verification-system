import unittest

from trajectory_verification.models import AgentTrack, Scenario, State
from trajectory_verification.regression import RegressionPolicy, compare_scenario_sets
from trajectory_verification.requirements import Requirement


def scenario(identifier: str, distance: float) -> Scenario:
    return Scenario(identifier, (
        AgentTrack("ego", (
            State(0.0, 0.0, 0.0),
            State(1.0, distance, 0.0),
            State(2.0, distance * 2, 0.0),
        ), "vehicle"),
    ))


REQ = Requirement("SPEED", "speed", "speed", "less_than_or_equal", 12, "m/s", "ego")


class RegressionTests(unittest.TestCase):
    def test_new_failure_blocks_gate(self):
        result = compare_scenario_sets([scenario("a", 10)], [scenario("a", 16)], [REQ])
        self.assertFalse(result.gate_passed)
        self.assertEqual(result.new_failures, 1)
        self.assertEqual(result.scenarios[0].transitions[0].classification, "NEW_FAILURE")

    def test_resolved_failure_passes_gate(self):
        result = compare_scenario_sets([scenario("a", 16)], [scenario("a", 10)], [REQ])
        self.assertTrue(result.gate_passed)
        self.assertEqual(result.resolved_failures, 1)

    def test_missing_candidate_is_policy_controlled(self):
        blocked = compare_scenario_sets([scenario("a", 10)], [], [REQ])
        allowed = compare_scenario_sets(
            [scenario("a", 10)], [], [REQ],
            RegressionPolicy(fail_on_missing_candidate_scenarios=False),
        )
        self.assertFalse(blocked.gate_passed)
        self.assertTrue(allowed.gate_passed)
        self.assertEqual(blocked.missing_candidate_scenarios, ("a",))

    def test_failure_budget_is_respected(self):
        result = compare_scenario_sets(
            [scenario("a", 10)], [scenario("a", 16)], [REQ],
            RegressionPolicy(max_new_failures=1),
        )
        self.assertTrue(result.gate_passed)


if __name__ == "__main__":
    unittest.main()
