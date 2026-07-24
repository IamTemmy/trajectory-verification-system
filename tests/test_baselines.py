import tempfile
import unittest
from pathlib import Path

from trajectory_verification.adapters.motion_submission import (
    load_motion_submission,
    write_motion_submission,
)
from trajectory_verification.baselines import constant_velocity_predictions
from trajectory_verification.models import AgentTrack, Scenario, State
from trajectory_verification.prediction_metrics import score_scenario_predictions


def linear_scenario() -> Scenario:
    states = tuple(
        State(index * 0.1, index * 0.5, 0.0, velocity_x_mps=5.0, velocity_y_mps=0.0)
        for index in range(91)
    )
    return Scenario(
        "linear", (AgentTrack("42", states, "vehicle"),),
        current_time_index=10, tracks_to_predict=("42",),
    )


class BaselineTests(unittest.TestCase):
    def test_constant_velocity_uses_current_state_and_velocity(self):
        truth = linear_scenario()
        predictions = constant_velocity_predictions(truth)
        result = score_scenario_predictions(truth, predictions)
        self.assertAlmostEqual(result.mean_min_ade_m, 0.0)
        self.assertAlmostEqual(result.mean_min_fde_m, 0.0)

    def test_official_serialization_round_trip(self):
        truth = linear_scenario()
        predictions = constant_velocity_predictions(truth)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.binproto"
            write_motion_submission([predictions], path)
            decoded = load_motion_submission(path, [truth])
        self.assertEqual(decoded, (predictions,))

    def test_requires_prediction_targets(self):
        truth = linear_scenario()
        without_targets = Scenario(
            truth.scenario_id, truth.tracks, current_time_index=truth.current_time_index
        )
        with self.assertRaisesRegex(ValueError, "tracks_to_predict"):
            constant_velocity_predictions(without_targets)


if __name__ == "__main__":
    unittest.main()
