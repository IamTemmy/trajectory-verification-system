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
                AgentPredictionScore("ego", 2, 16, 1.0, 2.5, True, 0),
            )),
        ), 2.0)
        payload = evaluation.to_dict()
        self.assertEqual(payload["summary"]["agents"], 1)
        self.assertEqual(payload["summary"]["miss_rate"], 1.0)
        self.assertIn("not official Waymo", evaluation_to_markdown(evaluation))
        self.assertIn("not official Waymo", evaluation_to_html(evaluation))


if __name__ == "__main__":
    unittest.main()
