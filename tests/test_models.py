import unittest

from trajectory_verification.models import AgentTrack, Scenario, State


class ModelTests(unittest.TestCase):
    def test_rejects_non_increasing_timestamps(self):
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            AgentTrack("car", (State(1.0, 0.0, 0.0), State(1.0, 1.0, 0.0)))

    def test_rejects_duplicate_agent_ids(self):
        track = AgentTrack("car", (State(0.0, 0.0, 0.0),))
        with self.assertRaisesRegex(ValueError, "unique"):
            Scenario("scene", (track, track))


if __name__ == "__main__":
    unittest.main()
