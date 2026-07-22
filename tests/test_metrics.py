import math
import unittest

from trajectory_verification.metrics import acceleration, closing_speed, speed, time_to_collision
from trajectory_verification.models import AgentTrack, State


def track(agent_id, positions):
    return AgentTrack(agent_id, tuple(State(float(t), float(x), 0.0) for t, x in positions))


class MetricTests(unittest.TestCase):
    def test_speed_and_acceleration(self):
        subject = track("car", [(0, 0), (1, 2), (2, 6)])
        self.assertEqual([2.0, 4.0], [sample.value for sample in speed(subject)])
        self.assertEqual([2.0], [sample.value for sample in acceleration(subject)])

    def test_closing_speed_and_ttc(self):
        subject = track("ego", [(0, 0), (1, 10), (2, 20)])
        lead = track("lead", [(0, 30), (1, 38), (2, 46)])
        self.assertEqual([2.0, 2.0], [sample.value for sample in closing_speed(subject, lead)])
        self.assertEqual([14.0, 13.0], [sample.value for sample in time_to_collision(subject, lead)])

    def test_ttc_is_infinite_when_not_closing(self):
        subject = track("ego", [(0, 0), (1, 5)])
        lead = track("lead", [(0, 10), (1, 16)])
        self.assertTrue(math.isinf(time_to_collision(subject, lead)[0].value))


if __name__ == "__main__":
    unittest.main()
