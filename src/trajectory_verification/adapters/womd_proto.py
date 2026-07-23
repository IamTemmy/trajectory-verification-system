"""Minimal wire-compatible protobuf class for WOMD motion scenarios."""

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory


def _field(message, name, number, field_type, *, label=1, type_name=None):
    field = message.field.add()
    field.name, field.number, field.label, field.type = name, number, label, field_type
    if type_name:
        field.type_name = type_name


def _build_scenario_class():
    file = descriptor_pb2.FileDescriptorProto(
        name="trajectory_verification/womd_scenario_subset.proto",
        package="waymo.open_dataset",
        syntax="proto2",
    )
    state = file.message_type.add(); state.name = "ObjectState"
    for name, number, kind in (
        ("center_x", 2, 1), ("center_y", 3, 1), ("center_z", 4, 1),
        ("length", 5, 2), ("width", 6, 2), ("height", 7, 2),
        ("heading", 8, 2), ("velocity_x", 9, 2), ("velocity_y", 10, 2),
        ("valid", 11, 8),
    ):
        _field(state, name, number, kind)

    track = file.message_type.add(); track.name = "Track"
    enum = track.enum_type.add(); enum.name = "ObjectType"
    for name, number in (("TYPE_UNSET", 0), ("TYPE_VEHICLE", 1),
                         ("TYPE_PEDESTRIAN", 2), ("TYPE_CYCLIST", 3),
                         ("TYPE_OTHER", 4)):
        value = enum.value.add(); value.name, value.number = name, number
    _field(track, "id", 1, 5)
    _field(track, "object_type", 2, 14,
           type_name=".waymo.open_dataset.Track.ObjectType")
    _field(track, "states", 3, 11, label=3,
           type_name=".waymo.open_dataset.ObjectState")

    prediction = file.message_type.add(); prediction.name = "RequiredPrediction"
    difficulty = prediction.enum_type.add(); difficulty.name = "DifficultyLevel"
    for name, number in (("NONE", 0), ("LEVEL_1", 1), ("LEVEL_2", 2)):
        value = difficulty.value.add(); value.name, value.number = name, number
    _field(prediction, "track_index", 1, 5)
    _field(prediction, "difficulty", 2, 14,
           type_name=".waymo.open_dataset.RequiredPrediction.DifficultyLevel")

    point = file.message_type.add(); point.name = "MapPoint"
    _field(point, "x", 1, 1); _field(point, "y", 2, 1); _field(point, "z", 3, 1)

    lane = file.message_type.add(); lane.name = "LaneCenter"
    _field(lane, "speed_limit_mph", 1, 1)
    _field(lane, "type", 2, 5)
    _field(lane, "polyline", 8, 11, label=3,
           type_name=".waymo.open_dataset.MapPoint")

    stop_sign = file.message_type.add(); stop_sign.name = "StopSign"
    _field(stop_sign, "lane", 1, 3, label=3)
    _field(stop_sign, "position", 2, 11,
           type_name=".waymo.open_dataset.MapPoint")

    crosswalk = file.message_type.add(); crosswalk.name = "Crosswalk"
    _field(crosswalk, "polygon", 1, 11, label=3,
           type_name=".waymo.open_dataset.MapPoint")

    map_feature = file.message_type.add(); map_feature.name = "MapFeature"
    _field(map_feature, "id", 1, 3)
    _field(map_feature, "lane", 3, 11,
           type_name=".waymo.open_dataset.LaneCenter")
    _field(map_feature, "stop_sign", 7, 11,
           type_name=".waymo.open_dataset.StopSign")
    _field(map_feature, "crosswalk", 8, 11,
           type_name=".waymo.open_dataset.Crosswalk")

    signal = file.message_type.add(); signal.name = "TrafficSignalLaneState"
    _field(signal, "lane", 1, 3)
    _field(signal, "state", 2, 5)
    _field(signal, "stop_point", 3, 11,
           type_name=".waymo.open_dataset.MapPoint")

    dynamic = file.message_type.add(); dynamic.name = "DynamicMapState"
    _field(dynamic, "lane_states", 1, 11, label=3,
           type_name=".waymo.open_dataset.TrafficSignalLaneState")

    scenario = file.message_type.add(); scenario.name = "Scenario"
    _field(scenario, "timestamps_seconds", 1, 1, label=3)
    _field(scenario, "tracks", 2, 11, label=3,
           type_name=".waymo.open_dataset.Track")
    _field(scenario, "objects_of_interest", 4, 5, label=3)
    _field(scenario, "scenario_id", 5, 9)
    _field(scenario, "sdc_track_index", 6, 5)
    _field(scenario, "dynamic_map_states", 7, 11, label=3,
           type_name=".waymo.open_dataset.DynamicMapState")
    _field(scenario, "map_features", 8, 11, label=3,
           type_name=".waymo.open_dataset.MapFeature")
    _field(scenario, "current_time_index", 10, 5)
    _field(scenario, "tracks_to_predict", 11, 11, label=3,
           type_name=".waymo.open_dataset.RequiredPrediction")

    descriptor = descriptor_pool.DescriptorPool().Add(file)
    return message_factory.GetMessageClass(
        descriptor.message_types_by_name["Scenario"]
    )


Scenario = _build_scenario_class()
