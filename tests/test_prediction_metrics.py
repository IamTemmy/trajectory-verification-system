import unittest

from trajectory_verification.models import AgentTrack, Scenario, State
from trajectory_verification.prediction_metrics import score_scenario_predictions
from trajectory_verification.predictions import (
    AgentPrediction,
    PredictedTrajectory,
    PredictionPoint,
    ScenarioPredictions,
)


def scenario() -> Scenario:
    return Scenario("a", (AgentTrack("ego", (
        State(1.0, 1.0, 0.0),
        State(2.0, 2.0, 0.0),
        State(3.0, 3.0, 0.0),
    )),))


def mode(offset: float, confidence: float = 0.5) -> PredictedTrajectory:
    return PredictedTrajectory(confidence, tuple(
        PredictionPoint(float(index), float(index) + offset, 0.0)
        for index in range(1, 4)
    ))


class PredictionMetricTests(unittest.TestCase):
    def test_selects_mode_with_minimum_ade(self):
        predictions = ScenarioPredictions("a", (
            AgentPrediction("ego", (mode(4.0), mode(1.0))),
        ))
        result = score_scenario_predictions(scenario(), predictions)
        self.assertAlmostEqual(result.mean_min_ade_m, 1.0)
        self.assertAlmostEqual(result.mean_min_fde_m, 1.0)
        self.assertEqual(result.agents[0].best_mode_index, 1)
        self.assertFalse(result.agents[0].miss)

    def test_miss_threshold_is_explicit(self):
        predictions = ScenarioPredictions("a", (AgentPrediction("ego", (mode(2.1),)),))
        result = score_scenario_predictions(scenario(), predictions, miss_threshold_m=2.0)
        self.assertEqual(result.miss_rate, 1.0)

    def test_rejects_missing_aligned_ground_truth(self):
        predictions = ScenarioPredictions("a", (
            AgentPrediction("ego", (
                PredictedTrajectory(1.0, (PredictionPoint(4.0, 4.0, 0.0),)),
            )),
        ))
        with self.assertRaisesRegex(ValueError, "no valid ground truth"):
            score_scenario_predictions(scenario(), predictions)


if __name__ == "__main__":
    unittest.main()

