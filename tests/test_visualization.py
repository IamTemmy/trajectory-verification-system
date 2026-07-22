import tempfile
import unittest
from pathlib import Path

from trajectory_verification.models import AgentTrack, Scenario, State
from trajectory_verification.visualization import scenario_to_svg, write_scenario_svg


class VisualizationTests(unittest.TestCase):
    def setUp(self):
        self.scenario = Scenario(
            "visual-fixture",
            (
                AgentTrack(
                    "ego",
                    (State(0.0, 0.0, 0.0), State(1.0, 5.0, 2.0)),
                    "vehicle",
                ),
                AgentTrack(
                    "walker",
                    (State(0.0, 3.0, 4.0), State(1.0, 3.5, 4.5)),
                    "pedestrian",
                ),
            ),
            sdc_agent_id="ego",
        )

    def test_renders_tracks_and_labels(self):
        svg = scenario_to_svg(self.scenario)
        self.assertIn("<svg", svg)
        self.assertEqual(2, svg.count("<polyline"))
        self.assertIn("ego · vehicle", svg)
        self.assertIn("walker · pedestrian", svg)
        self.assertIn('stroke-width="4"', svg)

    def test_writes_standalone_svg(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_scenario_svg(self.scenario, Path(directory) / "plot.svg")
            self.assertTrue(path.exists())
            self.assertTrue(path.read_text(encoding="utf-8").endswith("</svg>\n"))

    def test_rejects_invalid_canvas(self):
        with self.assertRaisesRegex(ValueError, "canvas"):
            scenario_to_svg(self.scenario, width_px=50, padding_px=30)


if __name__ == "__main__":
    unittest.main()
