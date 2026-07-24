import tempfile
import unittest
from pathlib import Path

from trajectory_verification.adapters.motion_submission import (
    OFFICIAL_PREDICTION_STEPS,
    load_motion_submission,
    scenario_predictions_from_proto,
)
from trajectory_verification.adapters.motion_submission_proto import (
    ChallengeScenarioPredictions,
    MotionChallengeSubmission,
)
from trajectory_verification.models import AgentTrack, Scenario, State


def ground_truth() -> Scenario:
    states = tuple(State(index * 0.1, float(index), 0.0) for index in range(91))
    return Scenario("scenario-a", (AgentTrack("42", states, "vehicle"),),
                    tracks_to_predict=("42",))


def prediction_message(offset: float = 0.0):
    message = ChallengeScenarioPredictions()
    message.scenario_id = "scenario-a"
    agent = message.single_predictions.predictions.add()
    agent.object_id = 42
    mode = agent.trajectories.add()
    mode.confidence = 0.8
    mode.trajectory.center_x.extend(step + offset for step in OFFICIAL_PREDICTION_STEPS)
    mode.trajectory.center_y.extend(0.0 for _ in OFFICIAL_PREDICTION_STEPS)
    return message


class MotionSubmissionTests(unittest.TestCase):
    def test_normalizes_official_timestamps_and_object_id(self):
        result = scenario_predictions_from_proto(prediction_message(), ground_truth())
        self.assertEqual(result.agents[0].agent_id, "42")
        self.assertEqual(len(result.agents[0].trajectories[0].points), 16)
        self.assertAlmostEqual(result.agents[0].trajectories[0].points[0].time_s, 1.5)
        self.assertAlmostEqual(result.agents[0].trajectories[0].points[-1].time_s, 9.0)

    def test_rejects_wrong_trajectory_length(self):
        message = prediction_message()
        del message.single_predictions.predictions[0].trajectories[0].trajectory.center_x[-1]
        with self.assertRaisesRegex(ValueError, "exactly 16"):
            scenario_predictions_from_proto(message, ground_truth())

    def test_decodes_serialized_submission(self):
        submission = MotionChallengeSubmission()
        submission.scenario_predictions.add().CopyFrom(prediction_message())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "submission.binproto"
            path.write_bytes(submission.SerializeToString())
            result = load_motion_submission(path, [ground_truth()])
        self.assertEqual(result[0].scenario_id, "scenario-a")


if __name__ == "__main__":
    unittest.main()

