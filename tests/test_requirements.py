import unittest

from trajectory_verification.models import AgentTrack, Scenario, State
from trajectory_verification.requirements import Requirement, evaluate_requirement


class RequirementTests(unittest.TestCase):
    def setUp(self):
        ego = AgentTrack(
            "ego",
            tuple(State(float(t), float(x), 0.0) for t, x in [(0, 0), (1, 10), (2, 20), (3, 30)]),
        )
        lead = AgentTrack(
            "lead",
            tuple(State(float(t), float(x), 0.0) for t, x in [(0, 30), (1, 38), (2, 46), (3, 54)]),
        )
        self.scenario = Scenario("scene", (ego, lead))

    def test_localizes_ttc_failure(self):
        requirement = Requirement(
            "SAFE_FOLLOWING_001",
            "Maintain TTC",
            "time_to_collision",
            "greater_than_or_equal",
            13.5,
            "s",
            "ego",
            "lead",
        )
        result = evaluate_requirement(self.scenario, requirement)
        self.assertFalse(result.passed)
        self.assertEqual(2, result.failed_samples)
        self.assertEqual(1, len(result.failure_intervals))
        self.assertEqual(2.0, result.failure_intervals[0].start_time_s)
        self.assertEqual(3.0, result.failure_intervals[0].end_time_s)
        self.assertEqual(12.0, result.failure_intervals[0].worst_value)

    def test_passing_speed_requirement(self):
        requirement = Requirement(
            "SPEED_001", "Limit speed", "speed", "less_than_or_equal", 10.0, "m/s", "ego"
        )
        result = evaluate_requirement(self.scenario, requirement)
        self.assertTrue(result.passed)
        self.assertEqual(0, result.failed_samples)


if __name__ == "__main__":
    unittest.main()
