"""Waymo Open Motion Dataset scenario-proto ingestion.

The adapter deliberately depends on protobuf-shaped attributes rather than concrete
generated classes. Unit tests can therefore exercise normalization without installing
TensorFlow or the Waymo SDK. Real decoding uses a wire-compatible schema subset.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
import struct
from typing import Any

from ..models import (
    AgentTrack,
    CrosswalkFeature,
    LaneFeature,
    MapContext,
    MapPoint,
    Scenario,
    State,
    StopSignFeature,
    TrafficSignalState,
)


OBJECT_TYPES = {
    0: "unset",
    1: "vehicle",
    2: "pedestrian",
    3: "cyclist",
    4: "other",
}

LANE_TYPES = {0: "undefined", 1: "freeway", 2: "surface_street", 3: "bike_lane"}
SIGNAL_STATES = {
    0: "unknown", 1: "arrow_stop", 2: "arrow_caution", 3: "arrow_go",
    4: "stop", 5: "caution", 6: "go", 7: "flashing_stop",
    8: "flashing_caution",
}


class WOMDDependencyError(RuntimeError):
    """Raised when real WOMD protobuf decoding is requested without its package."""


class TFRecordFormatError(ValueError):
    """Raised when an uncompressed TFRecord is truncated or malformed."""


def scenario_from_proto(message: Any, *, min_valid_states: int = 1) -> Scenario:
    """Normalize one WOMD ``Scenario`` protobuf-like object.

    Invalid object states are omitted while timestamp alignment is preserved. Tracks
    with fewer than ``min_valid_states`` valid samples are excluded.
    """

    if min_valid_states < 1:
        raise ValueError("min_valid_states must be at least one")
    timestamps = tuple(float(value) for value in message.timestamps_seconds)
    if not timestamps:
        raise ValueError("WOMD scenario contains no timestamps")

    normalized_tracks: list[AgentTrack] = []
    retained_ids: set[str] = set()
    source_index_to_id: dict[int, str] = {}
    for source_index, track in enumerate(message.tracks):
        if len(track.states) != len(timestamps):
            raise ValueError(
                f"track {track.id} has {len(track.states)} states for {len(timestamps)} timestamps"
            )
        agent_id = str(track.id)
        source_index_to_id[source_index] = agent_id
        states = tuple(
            _state_from_proto(timestamp, state)
            for timestamp, state in zip(timestamps, track.states)
            if bool(state.valid)
        )
        if len(states) < min_valid_states:
            continue
        retained_ids.add(agent_id)
        normalized_tracks.append(
            AgentTrack(
                agent_id=agent_id,
                object_type=OBJECT_TYPES.get(int(track.object_type), f"unknown_{track.object_type}"),
                states=states,
            )
        )

    if not normalized_tracks:
        raise ValueError("WOMD scenario contains no retained valid tracks")

    sdc_index = int(message.sdc_track_index)
    sdc_agent_id = source_index_to_id.get(sdc_index)
    if sdc_agent_id not in retained_ids:
        sdc_agent_id = None

    prediction_ids = tuple(
        source_index_to_id[int(item.track_index)]
        for item in message.tracks_to_predict
        if int(item.track_index) in source_index_to_id
        and source_index_to_id[int(item.track_index)] in retained_ids
    )
    interest_ids = tuple(
        str(agent_id) for agent_id in message.objects_of_interest if str(agent_id) in retained_ids
    )
    map_context = _map_context_from_proto(message, timestamps)

    return Scenario(
        scenario_id=str(message.scenario_id),
        tracks=tuple(normalized_tracks),
        current_time_index=int(message.current_time_index),
        sdc_agent_id=sdc_agent_id,
        objects_of_interest=interest_ids,
        tracks_to_predict=prediction_ids,
        map_feature_count=len(message.map_features),
        map_context=map_context,
    )


def _map_context_from_proto(message: Any, timestamps: tuple[float, ...]) -> MapContext:
    lanes: list[LaneFeature] = []
    stop_signs: list[StopSignFeature] = []
    crosswalks: list[CrosswalkFeature] = []
    for feature in message.map_features:
        if not hasattr(feature, "id"):
            continue
        feature_id = str(feature.id)
        if _has_field(feature, "lane") and feature.lane.polyline:
            lanes.append(LaneFeature(
                feature_id,
                tuple(_map_point(point) for point in feature.lane.polyline),
                float(feature.lane.speed_limit_mph),
                LANE_TYPES.get(int(feature.lane.type), f"unknown_{feature.lane.type}"),
            ))
        elif _has_field(feature, "stop_sign") and _has_field(feature.stop_sign, "position"):
            stop_signs.append(StopSignFeature(
                feature_id,
                tuple(str(lane_id) for lane_id in feature.stop_sign.lane),
                _map_point(feature.stop_sign.position),
            ))
        elif _has_field(feature, "crosswalk") and feature.crosswalk.polygon:
            crosswalks.append(CrosswalkFeature(
                feature_id,
                tuple(_map_point(point) for point in feature.crosswalk.polygon),
            ))

    signals: list[TrafficSignalState] = []
    for time_s, dynamic_state in zip(timestamps, getattr(message, "dynamic_map_states", ())):
        for lane_state in dynamic_state.lane_states:
            if _has_field(lane_state, "stop_point"):
                signals.append(TrafficSignalState(
                    time_s,
                    str(lane_state.lane),
                    SIGNAL_STATES.get(int(lane_state.state), f"unknown_{lane_state.state}"),
                    _map_point(lane_state.stop_point),
                ))
    return MapContext(tuple(lanes), tuple(stop_signs), tuple(crosswalks), tuple(signals))


def _map_point(point: Any) -> MapPoint:
    return MapPoint(float(point.x), float(point.y), float(point.z))


def _has_field(message: Any, field_name: str) -> bool:
    has_field = getattr(message, "HasField", None)
    if has_field is not None:
        try:
            return bool(has_field(field_name))
        except (ValueError, TypeError):
            pass
    return getattr(message, field_name, None) is not None


def _state_from_proto(timestamp: float, state: Any) -> State:
    return State(
        time_s=timestamp,
        x_m=float(state.center_x),
        y_m=float(state.center_y),
        heading_rad=float(state.heading),
        z_m=float(state.center_z),
        velocity_x_mps=float(state.velocity_x),
        velocity_y_mps=float(state.velocity_y),
        length_m=float(state.length),
        width_m=float(state.width),
        height_m=float(state.height),
    )


def iter_tfrecord_records(path: str | Path) -> Iterator[bytes]:
    """Yield payloads from an uncompressed TFRecord without TensorFlow.

    TFRecord CRC fields are consumed but not validated because Python's standard
    library does not provide CRC32C. Protobuf parsing still validates payload shape.
    """

    record_path = Path(path)
    with record_path.open("rb") as stream:
        record_index = 0
        while True:
            length_bytes = stream.read(8)
            if not length_bytes:
                return
            if len(length_bytes) != 8:
                raise TFRecordFormatError(f"truncated length at record {record_index}")
            length_crc = stream.read(4)
            if len(length_crc) != 4:
                raise TFRecordFormatError(f"missing length CRC at record {record_index}")
            (length,) = struct.unpack("<Q", length_bytes)
            payload = stream.read(length)
            if len(payload) != length:
                raise TFRecordFormatError(f"truncated payload at record {record_index}")
            data_crc = stream.read(4)
            if len(data_crc) != 4:
                raise TFRecordFormatError(f"missing data CRC at record {record_index}")
            yield payload
            record_index += 1


def iter_womd_scenarios(
    paths: str | Path | Sequence[str | Path],
    *,
    scenario_factory: Callable[[], Any] | None = None,
    min_valid_states: int = 1,
) -> Iterator[Scenario]:
    """Decode one or more scenario-proto WOMD TFRecord shards lazily."""

    if isinstance(paths, (str, Path)):
        normalized_paths: Sequence[str | Path] = (paths,)
    else:
        normalized_paths = paths
    factory = scenario_factory or _official_scenario_factory()
    for path in normalized_paths:
        for payload in iter_tfrecord_records(path):
            message = factory()
            message.ParseFromString(payload)
            yield scenario_from_proto(message, min_valid_states=min_valid_states)


def _official_scenario_factory() -> Callable[[], Any]:
    try:
        from .womd_proto import Scenario as ScenarioProto
    except ImportError as exc:
        raise WOMDDependencyError(
            "WOMD protobuf decoding requires the 'protobuf' package. Install "
            "project dependencies, or pass scenario_factory for testing."
        ) from exc
    return ScenarioProto
