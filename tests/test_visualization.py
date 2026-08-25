import re
import tempfile
import unittest
from pathlib import Path

from trajectory_verification.models import (
    AgentTrack, CrosswalkFeature, LaneFeature, MapContext, MapPoint,
    Scenario, State, StopSignFeature,
)
from trajectory_verification.predictions import (
    AgentPrediction, PredictedTrajectory, PredictionPoint,
)
from trajectory_verification.visualization import (
    prediction_to_svg, scenario_to_svg, write_scenario_svg,
)


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

    def test_same_type_agents_get_distinct_colours(self):
        """Two vehicles must be told apart; type colouring alone hid conflicts."""
        scenario = Scenario(
            "two-vehicles",
            (
                AgentTrack("first", (State(0.0, 0.0, 0.0), State(1.0, 5.0, 0.0)), "vehicle"),
                AgentTrack("second", (State(0.0, 0.0, 6.0), State(1.0, 5.0, 6.0)), "vehicle"),
            ),
        )
        strokes = re.findall(r'<polyline[^>]*stroke="(#[0-9a-f]{6})"', scenario_to_svg(scenario))
        self.assertEqual(2, len(strokes))
        self.assertNotEqual(strokes[0], strokes[1])

    def test_endpoint_labels_stay_inside_the_canvas(self):
        """A track ending at the right edge must not render its label off-canvas."""
        scenario = Scenario(
            "edge-label",
            (AgentTrack("a_very_long_agent_identifier",
                        (State(0.0, 0.0, 0.0), State(1.0, 100.0, 0.0)), "vehicle"),),
        )
        svg = scenario_to_svg(scenario, width_px=900, height_px=700)
        match = re.search(r'<text x="([\d.]+)"[^>]*text-anchor="(\w+)"[^>]*>([^<]+)</text>', svg)
        x, anchor, text = float(match.group(1)), match.group(2), match.group(3)
        self.assertEqual("end", anchor)
        self.assertGreaterEqual(x - 6.6 * len(text), 0.0)
        self.assertLessEqual(x, 900.0)

    def test_label_omits_redundant_type(self):
        scenario = Scenario(
            "stutter",
            (AgentTrack("cyclist", (State(0.0, 0.0, 0.0), State(1.0, 5.0, 0.0)), "cyclist"),),
        )
        svg = scenario_to_svg(scenario)
        self.assertNotIn("cyclist \u00b7 cyclist", svg)
        self.assertIn(">cyclist<", svg)

    def test_rejects_invalid_canvas(self):
        with self.assertRaisesRegex(ValueError, "canvas"):
            scenario_to_svg(self.scenario, width_px=50, padding_px=30)

    def test_renders_map_context_behind_trajectories(self):
        context = MapContext(
            lanes=(LaneFeature("lane", (MapPoint(0, 0), MapPoint(6, 0))),),
            stop_signs=(StopSignFeature("stop", ("lane",), MapPoint(4, 0)),),
            crosswalks=(CrosswalkFeature("walk", (
                MapPoint(2, -1), MapPoint(3, -1), MapPoint(3, 1), MapPoint(2, 1),
            )),),
        )
        scenario = Scenario(
            self.scenario.scenario_id, self.scenario.tracks,
            sdc_agent_id="ego", map_context=context,
        )
        svg = scenario_to_svg(scenario)
        self.assertIn("stroke-dasharray", svg)
        self.assertIn("<polygon", svg)
        self.assertIn("rotate(45", svg)


class PredictionVisualizationTests(unittest.TestCase):
    def setUp(self):
        history = tuple(State(float(t), float(t) * 2.0, 0.0) for t in range(5))
        future = tuple(State(float(t), float(t) * 2.0, 0.0) for t in range(5, 9))
        self.scenario = Scenario(
            "pred-fixture",
            (AgentTrack("target", history + future, "vehicle"),
             AgentTrack("other", history + future, "pedestrian")),
        )
        self.prediction = AgentPrediction(
            "target",
            (
                PredictedTrajectory(0.7, tuple(
                    PredictionPoint(float(t), float(t) * 2.0, 0.0) for t in range(5, 9))),
                PredictedTrajectory(0.3, tuple(
                    PredictionPoint(float(t), float(t) * 2.0, 9.0) for t in range(5, 9))),
            ),
        )

    def test_draws_history_truth_and_every_mode(self):
        svg = prediction_to_svg(self.scenario, self.prediction)
        self.assertIn("<svg", svg)
        self.assertIn("pred-fixture", svg)
        self.assertIn("agent target", svg)
        for label in ("observed history", "recorded future",
                      "closest predicted mode", "other modes"):
            self.assertIn(label, svg)

    def test_selects_the_mode_closest_to_the_recorded_future(self):
        """The highlighted mode must be the accurate one, not the confident one.

        The second mode is 9 m off but the first is exact, so the drawing must
        highlight the first even though both are present.
        """
        svg = prediction_to_svg(self.scenario, self.prediction)
        highlighted = re.findall(
            r'<polyline points="([^"]+)"[^>]*stroke="#dc2626"', svg
        )
        self.assertEqual(1, len(highlighted), "exactly one mode may be highlighted")
        # The accurate mode lies on y = 0 in world space; the 9 m offset mode
        # would render at a different height.
        ys = {round(float(pair.split(",")[1]), 2)
              for pair in highlighted[0].split()}
        self.assertEqual(1, len(ys), "highlighted mode should be the flat, exact one")

    def test_rejects_an_agent_with_no_observed_history(self):
        prediction = AgentPrediction(
            "other",
            (PredictedTrajectory(1.0, tuple(
                PredictionPoint(float(t), 0.0, 0.0) for t in range(0, 4))),),
        )
        with self.assertRaisesRegex(ValueError, "no observed history"):
            prediction_to_svg(self.scenario, prediction)


if __name__ == "__main__":
    unittest.main()
