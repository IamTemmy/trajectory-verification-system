import unittest

from trajectory_verification.prediction_metrics import (
    AgentPredictionScore,
    ScenarioPredictionScore,
)
from trajectory_verification.prediction_reporting import (
    PredictionEvaluation,
    evaluation_to_html,
    evaluation_to_markdown,
)


class PredictionReportingTests(unittest.TestCase):
    def test_aggregates_and_states_interpretation_boundary(self):
        evaluation = PredictionEvaluation((
            ScenarioPredictionScore("a", (
                AgentPredictionScore("ego", 2, 12, 16, 1.0, 2.5, True, 0),
            )),
        ), 2.0)
        payload = evaluation.to_dict()
        self.assertEqual(payload["summary"]["agents"], 1)
        self.assertEqual(payload["summary"]["miss_rate"], 1.0)
        self.assertEqual(payload["summary"]["mean_ground_truth_coverage"], 0.75)
        self.assertIn("not official Waymo", evaluation_to_markdown(evaluation))
        self.assertIn("not official Waymo", evaluation_to_html(evaluation))

    def test_bootstrap_and_breakdowns_are_deterministic(self):
        evaluation = PredictionEvaluation((
            ScenarioPredictionScore("a", (
                AgentPredictionScore("one", 3, 16, 16, 1.0, 2.0, False, 1, "vehicle"),
                AgentPredictionScore("two", 3, 16, 16, 3.0, 6.0, True, 2, "pedestrian"),
            )),
        ), 2.0, bootstrap_samples=100, bootstrap_seed=7)
        first = evaluation.to_dict()
        second = evaluation.to_dict()
        self.assertEqual(first["confidence_intervals"], second["confidence_intervals"])
        self.assertEqual(first["best_mode_counts"], {"1": 1, "2": 1})
        self.assertEqual(first["by_object_type"]["vehicle"]["agents"], 1)
        self.assertEqual(first["worst_agents_by_min_ade"][0]["agent_id"], "two")


if __name__ == "__main__":
    unittest.main()
