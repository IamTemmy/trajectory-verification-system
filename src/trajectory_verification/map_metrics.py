"""Dependency-free geometric signals for normalized map-aware verification."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot

from .metrics import Sample
from .models import AgentTrack, LaneFeature, MapPoint, Scenario, State


class NotApplicableError(ValueError):
    """Raised when a requirement lacks the map context needed for evaluation."""


@dataclass(frozen=True, slots=True)
class LaneAssociation:
    time_s: float
    lane_id: str
    lateral_offset_m: float
    distance_m: float


def lane_associations(
    track: AgentTrack,
    lanes: tuple[LaneFeature, ...],
    *,
    max_distance_m: float = 10.0,
) -> tuple[LaneAssociation, ...]:
    if not lanes:
        raise NotApplicableError("scenario contains no lane-center geometry")
    associations: list[LaneAssociation] = []
    for state in track.states:
        candidates = []
        for lane in lanes:
            projection = _nearest_polyline_projection(state.x_m, state.y_m, lane)
            if projection is not None:
                distance, signed_offset, _ = projection
                candidates.append((distance, signed_offset, lane.feature_id))
        if candidates:
            distance, signed_offset, lane_id = min(candidates, key=lambda item: item[0])
            if distance <= max_distance_m:
                associations.append(
                    LaneAssociation(state.time_s, lane_id, signed_offset, distance)
                )
    if not associations:
        raise NotApplicableError(
            f"subject is never within {max_distance_m:g} m of a lane center"
        )
    return tuple(associations)


def lane_lateral_offset(scenario: Scenario, track: AgentTrack) -> tuple[Sample, ...]:
    return tuple(
        Sample(item.time_s, abs(item.lateral_offset_m))
        for item in lane_associations(track, scenario.map_context.lanes)
    )


def crosswalk_proximity(scenario: Scenario, track: AgentTrack) -> tuple[Sample, ...]:
    crosswalks = scenario.map_context.crosswalks
    if not crosswalks:
        raise NotApplicableError("scenario contains no crosswalk polygons")
    return tuple(
        Sample(
            state.time_s,
            min(_distance_to_polygon(state.x_m, state.y_m, item.polygon) for item in crosswalks),
        )
        for state in track.states
    )


def vru_crosswalk_proximity(scenario: Scenario, track: AgentTrack) -> tuple[Sample, ...]:
    """Distance to the nearest pedestrian/cyclist occupying a crosswalk context."""

    crosswalks = scenario.map_context.crosswalks
    if not crosswalks:
        raise NotApplicableError("scenario contains no crosswalk polygons")
    grouped: dict[float, list[State]] = {}
    for candidate in scenario.tracks:
        if candidate.object_type not in {"pedestrian", "cyclist"}:
            continue
        for state in candidate.states:
            distance = min(
                _distance_to_polygon(state.x_m, state.y_m, item.polygon)
                for item in crosswalks
            )
            if distance <= 1.0:
                grouped.setdefault(state.time_s, []).append(state)
    samples = tuple(
        Sample(
            state.time_s,
            min(hypot(state.x_m - vru.x_m, state.y_m - vru.y_m) for vru in grouped[state.time_s]),
        )
        for state in track.states
        if state.time_s in grouped
    )
    if not samples:
        raise NotApplicableError(
            "no pedestrian or cyclist is observed in a mapped crosswalk context"
        )
    return samples


def red_stop_line_violation(scenario: Scenario, track: AgentTrack) -> tuple[Sample, ...]:
    lanes = {lane.feature_id: lane for lane in scenario.map_context.lanes}
    if not lanes:
        raise NotApplicableError("scenario contains no lane-center geometry")
    stop_states = tuple(
        signal for signal in scenario.map_context.traffic_signals
        if signal.state in {"stop", "arrow_stop", "flashing_stop"}
        and signal.lane_id in lanes
    )
    if not stop_states:
        raise NotApplicableError("scenario contains no stop-state traffic signals")
    states_by_time = {state.time_s: state for state in track.states}
    previous_by_time = {
        current.time_s: previous for previous, current in zip(track.states, track.states[1:])
    }
    samples: list[Sample] = []
    for signal in stop_states:
        current = states_by_time.get(signal.time_s)
        previous = previous_by_time.get(signal.time_s)
        if current is None or previous is None:
            continue
        lane = lanes[signal.lane_id]
        if _point_to_lane_distance(current, lane) > 7.5:
            continue
        crossed = _crossed_stop_point(previous, current, signal.stop_point, lane)
        samples.append(Sample(signal.time_s, 1.0 if crossed else 0.0))
    if not samples:
        raise NotApplicableError("subject is not aligned with an active stop-state signal")
    return tuple(samples)


def stop_sign_crossing_speed(scenario: Scenario, track: AgentTrack) -> tuple[Sample, ...]:
    lanes = {lane.feature_id: lane for lane in scenario.map_context.lanes}
    if not lanes or not scenario.map_context.stop_signs:
        raise NotApplicableError("scenario contains no usable stop-sign lane geometry")
    samples: list[Sample] = []
    for previous, current in zip(track.states, track.states[1:]):
        for sign in scenario.map_context.stop_signs:
            for lane_id in sign.lane_ids:
                lane = lanes.get(lane_id)
                if lane is None or _point_to_lane_distance(current, lane) > 7.5:
                    continue
                if _crossed_stop_point(previous, current, sign.position, lane):
                    dt = current.time_s - previous.time_s
                    velocity = hypot(current.x_m - previous.x_m, current.y_m - previous.y_m) / dt
                    samples.append(Sample(current.time_s, velocity))
    if not samples:
        raise NotApplicableError("subject does not cross a mapped stop-sign position")
    return tuple(samples)


def _nearest_polyline_projection(
    x: float, y: float, lane: LaneFeature
) -> tuple[float, float, tuple[float, float]] | None:
    best = None
    for start, end in zip(lane.polyline, lane.polyline[1:]):
        dx, dy = end.x_m - start.x_m, end.y_m - start.y_m
        length_sq = dx * dx + dy * dy
        if length_sq == 0:
            continue
        fraction = max(0.0, min(1.0, ((x - start.x_m) * dx + (y - start.y_m) * dy) / length_sq))
        projected_x, projected_y = start.x_m + fraction * dx, start.y_m + fraction * dy
        offset_x, offset_y = x - projected_x, y - projected_y
        distance = hypot(offset_x, offset_y)
        signed = (dx * offset_y - dy * offset_x) / hypot(dx, dy)
        candidate = (distance, signed, (dx, dy))
        if best is None or candidate[0] < best[0]:
            best = candidate
    return best


def _distance_to_polygon(x: float, y: float, polygon: tuple[MapPoint, ...]) -> float:
    if len(polygon) < 3:
        return min((hypot(x - point.x_m, y - point.y_m) for point in polygon), default=float("inf"))
    if _point_in_polygon(x, y, polygon):
        return 0.0
    closed = polygon + (polygon[0],)
    return min(_distance_to_segment(x, y, start, end) for start, end in zip(closed, closed[1:]))


def _point_in_polygon(x: float, y: float, polygon: tuple[MapPoint, ...]) -> bool:
    inside = False
    previous = polygon[-1]
    for current in polygon:
        if (current.y_m > y) != (previous.y_m > y):
            x_intersection = (previous.x_m - current.x_m) * (y - current.y_m) / (previous.y_m - current.y_m) + current.x_m
            if x < x_intersection:
                inside = not inside
        previous = current
    return inside


def _distance_to_segment(x: float, y: float, start: MapPoint, end: MapPoint) -> float:
    dx, dy = end.x_m - start.x_m, end.y_m - start.y_m
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return hypot(x - start.x_m, y - start.y_m)
    fraction = max(0.0, min(1.0, ((x - start.x_m) * dx + (y - start.y_m) * dy) / length_sq))
    return hypot(x - (start.x_m + fraction * dx), y - (start.y_m + fraction * dy))


def _point_to_lane_distance(state: State, lane: LaneFeature) -> float:
    projection = _nearest_polyline_projection(state.x_m, state.y_m, lane)
    return projection[0] if projection is not None else float("inf")


def _crossed_stop_point(
    previous: State, current: State, stop_point: MapPoint, lane: LaneFeature
) -> bool:
    projection = _nearest_polyline_projection(stop_point.x_m, stop_point.y_m, lane)
    if projection is None:
        return False
    tangent_x, tangent_y = projection[2]
    before = (previous.x_m - stop_point.x_m) * tangent_x + (previous.y_m - stop_point.y_m) * tangent_y
    after = (current.x_m - stop_point.x_m) * tangent_x + (current.y_m - stop_point.y_m) * tangent_y
    return before <= 0.0 < after
