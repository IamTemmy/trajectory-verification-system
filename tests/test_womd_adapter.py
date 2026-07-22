import struct
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path

from trajectory_verification.adapters.womd import (
    TFRecordFormatError,
    iter_tfrecord_records,
    iter_womd_scenarios,
    scenario_from_proto,
)


@dataclass
class FakeState:
    center_x: float
    center_y: float
    valid: bool = True
    center_z: float = 0.0
    length: float = 4.5
    width: float = 2.0
    height: float = 1.5
    heading: float = 0.0
    velocity_x: float = 0.0
    velocity_y: float = 0.0


@dataclass
class FakeTrack:
    id: int
    object_type: int
    states: list[FakeState]


@dataclass
class FakePrediction:
    track_index: int


@dataclass
class FakeScenario:
    scenario_id: str = "womd-fixture-001"
    timestamps_seconds: list[float] = field(default_factory=lambda: [0.0, 0.1, 0.2])
    tracks: list[FakeTrack] = field(default_factory=list)
    current_time_index: int = 1
    sdc_track_index: int = 0
    objects_of_interest: list[int] = field(default_factory=lambda: [22])
    tracks_to_predict: list[FakePrediction] = field(
        default_factory=lambda: [FakePrediction(track_index=1)]
    )
    map_features: list[object] = field(default_factory=lambda: [object(), object()])

    def ParseFromString(self, payload):
        if payload != b"fixture":
            raise ValueError("unexpected fixture payload")
        populated = make_scenario()
        self.__dict__.update(populated.__dict__)


def make_scenario():
    return FakeScenario(
        tracks=[
            FakeTrack(11, 1, [FakeState(0, 0), FakeState(1, 0), FakeState(2, 0)]),
            FakeTrack(
                22,
                2,
                [FakeState(5, 1), FakeState(5, 1, valid=False), FakeState(5, 2)],
            ),
            FakeTrack(33, 3, [FakeState(9, 9, valid=False)] * 3),
        ]
    )


def write_tfrecord(path, payloads):
    with path.open("wb") as stream:
        for payload in payloads:
            stream.write(struct.pack("<Q", len(payload)))
            stream.write(b"LCRC")
            stream.write(payload)
            stream.write(b"DCRC")


class WOMDAdapterTests(unittest.TestCase):
    def test_bundled_proto_decodes_consumed_womd_fields(self):
        from trajectory_verification.adapters.womd_proto import Scenario

        message = Scenario(
            scenario_id="real-schema-shape",
            timestamps_seconds=[0.0, 0.1],
            current_time_index=1,
            sdc_track_index=0,
            objects_of_interest=[42],
        )
        track = message.tracks.add(id=42, object_type=1)
        track.states.add(center_x=1.0, center_y=2.0, velocity_x=3.0, valid=True)
        track.states.add(center_x=1.3, center_y=2.0, velocity_x=3.0, valid=True)
        message.tracks_to_predict.add(track_index=0, difficulty=1)
        message.map_features.add()

        decoded = Scenario.FromString(message.SerializeToString())
        normalized = scenario_from_proto(decoded)

        self.assertEqual("real-schema-shape", normalized.scenario_id)
        self.assertEqual("42", normalized.sdc_agent_id)
        self.assertEqual(("42",), normalized.objects_of_interest)
        self.assertEqual(("42",), normalized.tracks_to_predict)
        self.assertEqual(1, normalized.map_feature_count)

    def test_normalizes_valid_states_and_metadata(self):
        scenario = scenario_from_proto(make_scenario())
        self.assertEqual("womd-fixture-001", scenario.scenario_id)
        self.assertEqual(["11", "22"], [track.agent_id for track in scenario.tracks])
        self.assertEqual("vehicle", scenario.tracks[0].object_type)
        self.assertEqual("pedestrian", scenario.tracks[1].object_type)
        self.assertEqual([0.0, 0.2], [state.time_s for state in scenario.tracks[1].states])
        self.assertEqual("11", scenario.sdc_agent_id)
        self.assertEqual(("22",), scenario.objects_of_interest)
        self.assertEqual(("22",), scenario.tracks_to_predict)
        self.assertEqual(2, scenario.map_feature_count)

    def test_rejects_state_timestamp_mismatch(self):
        message = make_scenario()
        message.tracks[0].states.pop()
        with self.assertRaisesRegex(ValueError, "states for"):
            scenario_from_proto(message)

    def test_reads_tfrecord_framing_without_tensorflow(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.tfrecord"
            write_tfrecord(path, [b"one", b"two"])
            self.assertEqual([b"one", b"two"], list(iter_tfrecord_records(path)))

    def test_detects_truncated_tfrecord(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.tfrecord"
            path.write_bytes(struct.pack("<Q", 10) + b"LCRC" + b"short")
            with self.assertRaisesRegex(TFRecordFormatError, "truncated payload"):
                list(iter_tfrecord_records(path))

    def test_decodes_with_injected_message_factory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.tfrecord"
            write_tfrecord(path, [b"fixture"])
            scenarios = list(iter_womd_scenarios(path, scenario_factory=FakeScenario))
            self.assertEqual("womd-fixture-001", scenarios[0].scenario_id)


if __name__ == "__main__":
    unittest.main()
