import unittest

from trajectory_verification.prediction_comparison import (
    PredictionComparisonPolicy,
    compare_prediction_evaluations,
)


def report(ade, fde, miss, coverage=1.0, scenario_id="a", agent_id="ego"):
    return {
        "summary": {
            "mean_min_ade_m": ade,
            "mean_min_fde_m": fde,
            "miss_rate": miss,
            "mean_ground_truth_coverage": coverage,
        },
        "scenarios": [
            {
                "scenario_id": scenario_id,
                "agents": [
                    {
                        "agent_id": agent_id,
                        "object_type": "vehicle",
                        "min_ade_m": ade,
                        "min_fde_m": fde,
                        "miss": bool(miss),
                        "best_mode_index": 0,
                    }
                ],
            }
        ],
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

    def test_paired_intervals_counts_and_rankings_are_deterministic(self):
        baseline = report(4.0, 8.0, 1.0)
        candidate = report(3.0, 6.0, 0.0)
        first = compare_prediction_evaluations(
            baseline, candidate, bootstrap_samples=100, bootstrap_seed=7
        )
        second = compare_prediction_evaluations(
            baseline, candidate, bootstrap_samples=100, bootstrap_seed=7
        )
        self.assertEqual(
            first.paired_confidence_intervals,
            second.paired_confidence_intervals,
        )
        self.assertEqual(
            first.paired_confidence_intervals["min_ade_m"].to_dict(),
            {"lower": -1.0, "upper": -1.0, "confidence": 0.95},
        )
        self.assertEqual(
            first.agent_change_counts["min_ade_m"],
            {"improved": 1, "unchanged": 0, "regressed": 0},
        )
        self.assertEqual(first.most_improved_scenarios[0]["scenario_id"], "a")
        self.assertEqual(first.most_improved_agents[0]["agent_id"], "ego")

    def test_significance_policy_passes_clear_paired_improvement(self):
        result = compare_prediction_evaluations(
            report(4.0, 8.0, 1.0),
            report(3.0, 6.0, 0.0),
            PredictionComparisonPolicy(
                require_significant_ade_improvement=True,
                require_significant_fde_improvement=True,
            ),
            bootstrap_samples=100,
        )
        self.assertTrue(result.gate_passed)

    def test_significance_requires_bootstrap(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            compare_prediction_evaluations(
                report(4.0, 8.0, 1.0),
                report(3.0, 6.0, 0.0),
                PredictionComparisonPolicy(
                    require_significant_ade_improvement=True
                ),
                bootstrap_samples=0,
            )

    def test_zero_bootstrap_disables_intervals(self):
        result = compare_prediction_evaluations(
            report(4.0, 8.0, 1.0),
            report(3.0, 6.0, 0.0),
            bootstrap_samples=0,
        )
        self.assertEqual(result.paired_confidence_intervals, {})


if __name__ == "__main__":
    unittest.main()
