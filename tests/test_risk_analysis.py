import unittest

from trajectory_verification.models import (
    AgentTrack,
    CrosswalkFeature,
    MapContext,
    MapPoint,
    Scenario,
    State,
    TrafficSignalState,
)
from trajectory_verification.prediction_metrics import score_scenario_predictions
from trajectory_verification.predictions import (
    AgentPrediction,
    PredictedTrajectory,
    PredictionPoint,
    ScenarioPredictions,
)
from trajectory_verification.risk_analysis import (
    RiskThresholds,
    analyze_prediction_risk,
)
from trajectory_verification.risk_reporting import risk_to_html, risk_to_markdown


def fixture():
    target = AgentTrack("42", (
        State(0.0, 0.0, 0.0),
        State(1.0, 1.0, 0.0),
        State(2.0, 1.0, 2.0),
    ), "vehicle")
    sdc = AgentTrack("1", (
        State(0.0, 0.0, 1.0),
        State(1.0, 0.0, 1.0),
        State(2.0, 0.0, 1.0),
    ), "vehicle")
    pedestrian = AgentTrack("3", (
        State(0.0, 1.0, 1.0),
        State(1.0, 1.0, 1.0),
        State(2.0, 1.0, 1.0),
    ), "pedestrian")
    crosswalk = CrosswalkFeature("cw", (
        MapPoint(0.0, -1.0),
        MapPoint(2.0, -1.0),
        MapPoint(2.0, 3.0),
        MapPoint(0.0, 3.0),
    ))
    signal = TrafficSignalState(1.0, "lane", "stop", MapPoint(1.0, 1.0))
    scenario = Scenario(
        "risk-scenario",
        (target, sdc, pedestrian),
        current_time_index=0,
        sdc_agent_id="1",
        tracks_to_predict=("42",),
        map_context=MapContext(
            crosswalks=(crosswalk,),
            traffic_signals=(signal,),
        ),
        timestamps_s=(0.0, 1.0, 2.0),
    )
    predictions = ScenarioPredictions("risk-scenario", (
        AgentPrediction("42", (
            PredictedTrajectory(1.0, (
                PredictionPoint(1.0, 10.0, 0.0),
                PredictionPoint(2.0, 10.0, 2.0),
            )),
        )),
    ))
    score = score_scenario_predictions(
        scenario, predictions, miss_threshold_m=2.0
    )
    return scenario, predictions, score


class RiskAnalysisTests(unittest.TestCase):
    def test_prioritizes_missed_close_map_context(self):
        scenario, predictions, score = fixture()
        analysis = analyze_prediction_risk(
            (scenario,),
            (predictions,),
            (score,),
            RiskThresholds(dense_scene_agents=2),
        )
        item = analysis.evidence[0]
        self.assertEqual(item.review_priority, "high")
        self.assertEqual(item.motion_class, "turning")
        self.assertIn("close_interaction", item.risk_tags)
        self.assertIn("dense_scene", item.risk_tags)
        self.assertIn("crosswalk_context", item.risk_tags)
        self.assertIn("traffic_control_context", item.risk_tags)
        self.assertGreater(item.max_sdc_separation_error_m, 5.0)

    def test_reports_state_interpretation_boundary(self):
        scenario, predictions, score = fixture()
        analysis = analyze_prediction_risk(
            (scenario,), (predictions,), (score,)
        )
        self.assertIn("not collision probability", analysis.to_dict()["interpretation"])
        self.assertIn("not a collision-probability", risk_to_markdown(analysis))
        self.assertIn("not collision probability", risk_to_html(analysis))

    def test_rejects_identity_mismatch(self):
        scenario, predictions, score = fixture()
        wrong = ScenarioPredictions("other", predictions.agents)
        with self.assertRaisesRegex(ValueError, "identities must match"):
            analyze_prediction_risk((scenario,), (wrong,), (score,))


if __name__ == "__main__":
    unittest.main()
