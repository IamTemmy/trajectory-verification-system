import tempfile
import unittest
from pathlib import Path

from trajectory_verification.models import AgentTrack, Scenario, State
from trajectory_verification.reporting import (
    build_validation_report,
    report_to_html,
    report_to_markdown,
    write_validation_reports,
)
from trajectory_verification.requirements import Requirement


class ReportingTests(unittest.TestCase):
    def setUp(self):
        ego = AgentTrack("ego", tuple(State(float(t), float(x), 0.0) for t, x in ((0, 0), (1, 10), (2, 20), (3, 30))))
        lead = AgentTrack("lead", tuple(State(float(t), float(x), 0.0) for t, x in ((0, 30), (1, 38), (2, 46), (3, 54))))
        scenario = Scenario("report-scene", (ego, lead))
        requirements = (
            Requirement("TTC_001", "Maintain TTC", "time_to_collision", "greater_than_or_equal", 13.5, "s", "ego", "lead"),
            Requirement("SPEED_001", "Limit speed", "speed", "less_than_or_equal", 10.0, "m/s", "ego"),
        )
        self.report = build_validation_report(scenario, requirements)

    def test_report_summary(self):
        self.assertFalse(self.report.passed)
        self.assertEqual(1, self.report.passed_count)
        markdown = report_to_markdown(self.report)
        self.assertIn("Overall result:** FAIL", markdown)
        self.assertIn("Threshold sensitivity", markdown)
        self.assertIn("Evidence confidence", markdown)
        self.assertIn("Interpretation boundary", markdown)

    def test_html_is_standalone_and_escaped(self):
        html = report_to_html(self.report)
        self.assertTrue(html.startswith("<!doctype html>"))
        self.assertIn("Trajectory validation report", html)
        self.assertIn("1/2", html)
        self.assertNotIn("<script", html)

    def test_writes_both_report_formats(self):
        with tempfile.TemporaryDirectory() as directory:
            outputs = write_validation_reports(
                self.report,
                markdown_path=Path(directory) / "report.md",
                html_path=Path(directory) / "report.html",
            )
            self.assertEqual(2, len(outputs))
            self.assertTrue(all(path.exists() for path in outputs))


if __name__ == "__main__":
    unittest.main()
