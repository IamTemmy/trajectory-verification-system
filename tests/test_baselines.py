import tempfile
import unittest
from pathlib import Path

from trajectory_verification.adapters.motion_submission import (
    load_motion_submission,
    write_motion_submission,
)
from trajectory_verification.baselines import (
    baseline_predictions,
    constant_velocity_predictions,
)
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

    def test_kinematic_ensemble_contains_three_distinct_models(self):
        prediction = baseline_predictions(linear_scenario(), "kinematic_ensemble")
        self.assertEqual(len(prediction.agents[0].trajectories), 3)
        self.assertAlmostEqual(
            sum(item.confidence for item in prediction.agents[0].trajectories), 1.0
        )

    def test_rejects_unknown_model(self):
        with self.assertRaisesRegex(ValueError, "unsupported baseline"):
            baseline_predictions(linear_scenario(), "unknown")

    def test_scenario_timeline_does_not_require_a_complete_track(self):
        states = linear_scenario().tracks[0].states[:20]
        scenario = Scenario(
            "sparse",
            (AgentTrack("42", states, "vehicle"),),
            current_time_index=10,
            tracks_to_predict=("42",),
            timestamps_s=tuple(index * 0.1 for index in range(91)),
        )
        prediction = baseline_predictions(scenario)
        self.assertEqual(len(prediction.agents[0].trajectories[0].points), 16)


if __name__ == "__main__":
    unittest.main()
