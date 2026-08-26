import unittest

from trajectory_verification.models import AgentTrack, Scenario, State
from trajectory_verification.predictions import (
    AgentPrediction, PredictedTrajectory, PredictionPoint,
)
from trajectory_verification.prediction_requirements import (
    AGREEMENT_CORRECT, AGREEMENT_FALSE_ALARM, AGREEMENT_MISSED_VIOLATION,
    evaluate_predicted_requirements, select_mode, summarize,
)
from trajectory_verification.requirements import Requirement

TIMES = (1.0, 2.0, 3.0, 4.0)


def scenario_with_speed(step_m: float) -> Scenario:
    """A scenario whose target moves ``step_m`` each second along +x."""
    states = tuple(State(t, step_m * t, 0.0) for t in (0.0,) + TIMES)
    return Scenario("s", (AgentTrack("target", states, "vehicle"),))


def prediction_with_speed(step_m: float, confidence: float = 1.0) -> AgentPrediction:
    return AgentPrediction(
        "target",
        (PredictedTrajectory(
            confidence,
            tuple(PredictionPoint(t, step_m * t, 0.0) for t in TIMES),
        ),),
    )


SPEED_LIMIT = (
    Requirement("LIMIT", "Stay at or below 10 m/s", "speed",
                "less_than_or_equal", 10.0, "m/s", "target"),
)


class PredictedRequirementTests(unittest.TestCase):
    def test_agreement_when_both_stay_within_the_limit(self):
        outcome, = evaluate_predicted_requirements(
            scenario_with_speed(5.0), prediction_with_speed(5.0), SPEED_LIMIT
        )
        self.assertEqual(AGREEMENT_CORRECT, outcome.agreement)
        self.assertFalse(outcome.consequential)

    def test_forecast_claiming_a_violation_that_did_not_happen(self):
        """The record stays legal; the forecast speeds. That is a false alarm."""
        outcome, = evaluate_predicted_requirements(
            scenario_with_speed(5.0), prediction_with_speed(20.0), SPEED_LIMIT
        )
        self.assertEqual(AGREEMENT_FALSE_ALARM, outcome.agreement)
        self.assertTrue(outcome.recorded.passed)
        self.assertFalse(outcome.predicted.passed)
        self.assertTrue(outcome.consequential)

    def test_forecast_missing_a_violation_that_did_happen(self):
        """The record speeds; the forecast does not. That is the costly failure."""
        outcome, = evaluate_predicted_requirements(
            scenario_with_speed(20.0), prediction_with_speed(5.0), SPEED_LIMIT
        )
        self.assertEqual(AGREEMENT_MISSED_VIOLATION, outcome.agreement)
        self.assertFalse(outcome.recorded.passed)
        self.assertTrue(outcome.predicted.passed)

    def test_both_views_are_sampled_at_the_prediction_timestamps(self):
        """Comparing differently spaced samples would blame the model for sampling.

        The recorded track carries states every 0.5 s while the forecast is
        yearly-spaced; if the recorded view kept its native spacing, its derived
        speed would differ purely because of the step size.
        """
        dense = tuple(State(t / 2, 5.0 * (t / 2), 0.0) for t in range(0, 9))
        scenario = Scenario("dense", (AgentTrack("target", dense, "vehicle"),))
        outcome, = evaluate_predicted_requirements(
            scenario, prediction_with_speed(5.0), SPEED_LIMIT
        )
        self.assertEqual(len(TIMES) - 1, outcome.recorded.evaluated_samples)
        self.assertEqual(
            outcome.recorded.evaluated_samples, outcome.predicted.evaluated_samples
        )

    def test_defaults_to_the_most_confident_mode_not_the_closest(self):
        """Scoring the closest mode would flatter the model with hindsight."""
        prediction = AgentPrediction(
            "target",
            (
                PredictedTrajectory(0.2, tuple(
                    PredictionPoint(t, 5.0 * t, 0.0) for t in TIMES)),
                PredictedTrajectory(0.8, tuple(
                    PredictionPoint(t, 20.0 * t, 0.0) for t in TIMES)),
            ),
        )
        index, mode = select_mode(prediction, None)
        self.assertEqual(1, index)
        self.assertEqual(0.8, mode.confidence)

        outcome, = evaluate_predicted_requirements(
            scenario_with_speed(5.0), prediction, SPEED_LIMIT
        )
        self.assertEqual(AGREEMENT_FALSE_ALARM, outcome.agreement)

    def test_summary_counts_and_accuracy(self):
        outcomes = (
            *evaluate_predicted_requirements(
                scenario_with_speed(5.0), prediction_with_speed(5.0), SPEED_LIMIT),
            *evaluate_predicted_requirements(
                scenario_with_speed(5.0), prediction_with_speed(20.0), SPEED_LIMIT),
            *evaluate_predicted_requirements(
                scenario_with_speed(20.0), prediction_with_speed(5.0), SPEED_LIMIT),
        )
        summary = summarize(outcomes)
        self.assertEqual(3, summary["evaluated"])
        self.assertEqual(1, summary["totals"][AGREEMENT_CORRECT])
        self.assertEqual(1, summary["totals"][AGREEMENT_FALSE_ALARM])
        self.assertEqual(1, summary["totals"][AGREEMENT_MISSED_VIOLATION])
        self.assertAlmostEqual(1 / 3, summary["behavioural_accuracy"])


if __name__ == "__main__":
    unittest.main()
