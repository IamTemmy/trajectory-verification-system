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

    def test_one_short_gap_does_not_split_a_contiguous_failure(self):
        """Irregular sampling must not fragment a single failure interval.

        Speed samples land at t = 1.0, 1.2, 2.2 and 3.2, so the gaps are
        0.2, 1.0 and 1.0 seconds. Inferring the nominal step from the minimum
        gap would treat the two one-second gaps as discontinuities and report
        three intervals; the median keeps the failure whole.
        """
        track = AgentTrack(
            "jittery",
            tuple(
                State(time_s, x_m, 0.0)
                for time_s, x_m in [(0.0, 0.0), (1.0, 100.0), (1.2, 120.0), (2.2, 220.0), (3.2, 320.0)]
            ),
        )
        requirement = Requirement(
            "SPEED_002", "Limit speed", "speed", "less_than_or_equal", 10.0, "m/s", "jittery"
        )
        result = evaluate_requirement(Scenario("jitter", (track,)), requirement)
        self.assertFalse(result.passed)
        self.assertEqual(4, result.failed_samples)
        self.assertEqual(1, len(result.failure_intervals))
        self.assertEqual(1.0, result.failure_intervals[0].start_time_s)
        self.assertEqual(3.2, result.failure_intervals[0].end_time_s)
        self.assertEqual(4, result.failure_intervals[0].sample_count)

    def test_passing_speed_requirement(self):
        requirement = Requirement(
            "SPEED_001", "Limit speed", "speed", "less_than_or_equal", 10.0, "m/s", "ego"
        )
        result = evaluate_requirement(self.scenario, requirement)
        self.assertTrue(result.passed)
        self.assertEqual(0, result.failed_samples)

    def test_map_requirement_is_explicitly_not_applicable_without_map(self):
        requirement = Requirement(
            "LANE_001", "Stay near lane center", "lane_lateral_offset",
            "less_than_or_equal", 2.0, "m", "ego",
        )
        result = evaluate_requirement(self.scenario, requirement)
        self.assertFalse(result.applicable)
        self.assertIsNone(result.passed)
        self.assertIn("no lane-center geometry", result.not_applicable_reason)


if __name__ == "__main__":
    unittest.main()
