import unittest

from trajectory_verification.map_metrics import (
    NotApplicableError,
    crosswalk_proximity,
    lane_lateral_offset,
    red_stop_line_violation,
    stop_sign_crossing_speed,
    vru_crosswalk_proximity,
)
from trajectory_verification.models import (
    AgentTrack, CrosswalkFeature, LaneFeature, MapContext, MapPoint,
    Scenario, State, StopSignFeature, TrafficSignalState,
)


def mapped_scenario(signal_state="stop"):
    lane = LaneFeature("lane-1", (MapPoint(0, 0), MapPoint(20, 0)), 35.0, "surface_street")
    crosswalk = CrosswalkFeature("crosswalk-1", (
        MapPoint(9, -2), MapPoint(11, -2), MapPoint(11, 2), MapPoint(9, 2),
    ))
    sign = StopSignFeature("sign-1", ("lane-1",), MapPoint(10, 0))
    signals = tuple(
        TrafficSignalState(float(t), "lane-1", signal_state, MapPoint(10, 0))
        for t in (0, 1, 2)
    )
    track = AgentTrack("ego", (
        State(0.0, 8.0, 1.0), State(1.0, 9.5, 1.0), State(2.0, 10.5, 1.0),
    ))
    return Scenario("mapped", (track,), sdc_agent_id="ego", map_feature_count=3,
                    map_context=MapContext((lane,), (sign,), (crosswalk,), signals))


class MapMetricTests(unittest.TestCase):
    def test_lane_offset_and_crosswalk_distance(self):
        scenario = mapped_scenario()
        track = scenario.track("ego")
        self.assertEqual([1.0, 1.0, 1.0], [sample.value for sample in lane_lateral_offset(scenario, track)])
        self.assertEqual(0.0, crosswalk_proximity(scenario, track)[-1].value)

    def test_detects_red_stop_line_crossing(self):
        samples = red_stop_line_violation(mapped_scenario(), mapped_scenario().track("ego"))
        self.assertEqual(1.0, samples[-1].value)

    def test_measures_stop_sign_crossing_speed(self):
        scenario = mapped_scenario()
        samples = stop_sign_crossing_speed(scenario, scenario.track("ego"))
        self.assertEqual(1.0, len(samples))
        self.assertAlmostEqual(1.0, samples[0].value)

    def test_missing_map_context_is_not_applicable(self):
        track = AgentTrack("ego", (State(0, 0, 0), State(1, 1, 0)))
        scenario = Scenario("no-map", (track,))
        with self.assertRaises(NotApplicableError):
            lane_lateral_offset(scenario, track)

    def test_vru_proximity_is_conditioned_on_crosswalk_occupancy(self):
        base = mapped_scenario()
        pedestrian = AgentTrack("walker", (
            State(0.0, 10.0, 0.0), State(1.0, 10.0, 0.0), State(2.0, 10.0, 0.0),
        ), object_type="pedestrian")
        scenario = Scenario(
            base.scenario_id, base.tracks + (pedestrian,), sdc_agent_id="ego",
            map_feature_count=base.map_feature_count, map_context=base.map_context,
        )
        samples = vru_crosswalk_proximity(scenario, scenario.track("ego"))
        self.assertEqual(3, len(samples))
        self.assertAlmostEqual(1.11803398875, samples[-1].value)


if __name__ == "__main__":
    unittest.main()
