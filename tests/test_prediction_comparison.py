import unittest

from trajectory_verification.prediction_comparison import (
    PredictionComparisonPolicy,
    compare_prediction_evaluations,
)


def report(ade, fde, miss, coverage=1.0):
    return {
        "summary": {
            "mean_min_ade_m": ade,
            "mean_min_fde_m": fde,
            "miss_rate": miss,
            "mean_ground_truth_coverage": coverage,
        },
        "scenarios": [{"scenario_id": "a", "agents": [{"agent_id": "ego"}]}],
    }


class PredictionComparisonTests(unittest.TestCase):
    def test_improvement_passes_strict_policy(self):
        result = compare_prediction_evaluations(
            report(4.0, 8.0, 1.0), report(3.0, 6.0, 0.5)
        )
        self.assertTrue(result.gate_passed)
        self.assertEqual(result.deltas["mean_min_ade_m"], -1.0)

    def test_regression_fails_strict_policy(self):
        result = compare_prediction_evaluations(
            report(4.0, 8.0, 0.5), report(4.1, 8.0, 0.5)
        )
        self.assertFalse(result.gate_passed)
        self.assertIn("mean_min_ade_m increased", result.violations[0])

    def test_tolerance_can_allow_small_regression(self):
        result = compare_prediction_evaluations(
            report(4.0, 8.0, 0.5),
            report(4.1, 8.0, 0.5),
            PredictionComparisonPolicy(max_ade_increase_m=0.2),
        )
        self.assertTrue(result.gate_passed)

    def test_identity_mismatch_is_rejected(self):
        candidate = report(3.0, 6.0, 0.5)
        candidate["scenarios"][0]["scenario_id"] = "other"
        with self.assertRaisesRegex(ValueError, "identities do not match"):
            compare_prediction_evaluations(report(4.0, 8.0, 1.0), candidate)


if __name__ == "__main__":
    unittest.main()
