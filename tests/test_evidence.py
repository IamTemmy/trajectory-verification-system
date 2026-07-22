import unittest

from trajectory_verification.evidence import (
    assess_scenario_quality,
    default_sensitivity_thresholds,
    explain_requirement,
)
from trajectory_verification.models import AgentTrack, Scenario, State
from trajectory_verification.requirements import Requirement


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        ego = AgentTrack("ego", tuple(State(float(t), float(x), 0.0) for t, x in ((0, 0), (1, 10), (2, 20), (3, 30))))
        lead = AgentTrack("lead", tuple(State(float(t), float(x), 0.0) for t, x in ((0, 30), (1, 38), (2, 46), (3, 54))))
        self.scenario = Scenario("evidence-scene", (ego, lead))
        self.requirement = Requirement(
            "TTC_001", "Maintain TTC", "time_to_collision",
            "greater_than_or_equal", 13.5, "s", "ego", "lead",
        )

    def test_explains_interval_and_margin(self):
        evidence = explain_requirement(self.scenario, self.requirement)
        self.assertFalse(evidence.result.passed)
        self.assertEqual(1, len(evidence.explanations))
        explanation = evidence.explanations[0]
        self.assertEqual(1.5, explanation.threshold_deviation)
        self.assertIn("2 s to 3 s", explanation.narrative)
        self.assertIn("worst value was 12 s", explanation.narrative)
        self.assertEqual("medium", evidence.evidence_confidence)
        self.assertIn("sample count", evidence.confidence_rationale)

    def test_sensitivity_sweep(self):
        thresholds = default_sensitivity_thresholds(self.requirement)
        evidence = explain_requirement(
            self.scenario, self.requirement, sensitivity_thresholds=thresholds
        )
        self.assertEqual(3, len(evidence.sensitivity))
        self.assertEqual(thresholds, tuple(point.threshold for point in evidence.sensitivity))
        self.assertFalse(evidence.sensitivity[-1].passed)
        self.assertLess(
            evidence.sensitivity[0].failed_samples,
            evidence.sensitivity[-1].failed_samples,
        )

    def test_quality_annotations_are_explicit(self):
        annotations = assess_scenario_quality(self.scenario)
        codes = {item.code for item in annotations}
        self.assertIn("PARTIAL_REPORTED_VELOCITY", codes)
        self.assertIn("NO_MAP_FEATURES", codes)


if __name__ == "__main__":
    unittest.main()
