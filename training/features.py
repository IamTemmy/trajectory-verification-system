"""Turn normalized WOMD scenarios into agent-centric model features.

This lives outside ``src/trajectory_verification`` on purpose. The verification
core depends on nothing but the standard library, and a learned model is an
external prediction source under the contract in ``docs/EXTERNAL_MODELS.md``.
Only the reader is imported from the package, so training consumes exactly the
decoder that was proven equivalent to Waymo's.

Timing follows the official challenge definition. History is scenario steps 0
through 10, the prediction is made at step 10, and the future is the 16 steps
15, 20, ... 90 that a submission must contain.

Every quantity is expressed in a frame centred on the target agent at step 10
and rotated so the agent faces +x. Without that normalization the model would
spend capacity learning that a vehicle at map coordinate (3301, -328) behaves
like one at (8456, 1737).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, hypot, sin
from typing import Iterator, Sequence

import numpy as np

from trajectory_verification.models import AgentTrack, Scenario, State

CURRENT_INDEX = 10
HISTORY_STEPS = 11                       # steps 0..10 inclusive
FUTURE_INDICES = tuple(range(15, 91, 5))  # 16 points, matching the submission
MAX_NEIGHBORS = 16
MAX_MAP_POINTS = 256
MAP_RADIUS_M = 75.0
NEIGHBOR_RADIUS_M = 100.0

# Motion filter for agents added beyond the designated targets. Chosen so the
# admitted population matches the designated targets' speed distribution.
MIN_ANCHOR_SPEED_MPS = 0.5
MIN_DISPLACEMENT_M = 2.0

TARGET_FEATURES = 7    # x, y, cos h, sin h, vx, vy, valid
NEIGHBOR_FEATURES = 10  # the above plus a three-way object-type indicator
MAP_FEATURES = 5       # x, y, direction x, direction y, valid

OBJECT_TYPE_INDEX = {"vehicle": 0, "pedestrian": 1, "cyclist": 2}


@dataclass(frozen=True, slots=True)
class Example:
    """One prediction target, ready for the model."""

    scenario_id: str
    agent_id: str
    object_type: str
    target_history: np.ndarray   # (HISTORY_STEPS, TARGET_FEATURES)
    neighbors: np.ndarray        # (MAX_NEIGHBORS, HISTORY_STEPS, NEIGHBOR_FEATURES)
    map_points: np.ndarray       # (MAX_MAP_POINTS, MAP_FEATURES)
    future: np.ndarray           # (16, 2)
    future_mask: np.ndarray      # (16,)
    origin: np.ndarray           # (2,) global position of the local frame
    heading: float               # global heading the local frame was rotated by


def _states_by_time(track: AgentTrack) -> dict[float, State]:
    """Index a track by timestamp.

    The normalizer drops invalid states, so ``track.states`` is variable length
    and its positions do not correspond to scenario steps. Each retained state
    carries its true ``time_s``, so alignment must go through the timeline.
    """

    return {round(state.time_s, 4): state for state in track.states}


def _rotate(dx: float, dy: float, cos_t: float, sin_t: float) -> tuple[float, float]:
    return dx * cos_t + dy * sin_t, -dx * sin_t + dy * cos_t


def _agent_row(
    state: State | None,
    origin_x: float,
    origin_y: float,
    cos_t: float,
    sin_t: float,
) -> list[float]:
    if state is None:
        return [0.0] * TARGET_FEATURES
    x, y = _rotate(state.x_m - origin_x, state.y_m - origin_y, cos_t, sin_t)
    heading = (state.heading_rad or 0.0)
    # The frame is rotated by the target's heading, so local heading is relative.
    local_heading_x, local_heading_y = _rotate(cos(heading), sin(heading), cos_t, sin_t)
    vx, vy = _rotate(
        state.velocity_x_mps or 0.0, state.velocity_y_mps or 0.0, cos_t, sin_t
    )
    return [x, y, local_heading_x, local_heading_y, vx, vy, 1.0]


def _map_rows(
    scenario: Scenario,
    origin_x: float,
    origin_y: float,
    cos_t: float,
    sin_t: float,
) -> np.ndarray:
    """Nearest lane-centre points, each carrying its local direction of travel."""

    collected: list[tuple[float, list[float]]] = []
    for lane in scenario.map_context.lanes:
        polyline = lane.polyline
        for index, point in enumerate(polyline):
            distance = hypot(point.x_m - origin_x, point.y_m - origin_y)
            if distance > MAP_RADIUS_M:
                continue
            following = polyline[min(index + 1, len(polyline) - 1)]
            step_x, step_y = following.x_m - point.x_m, following.y_m - point.y_m
            norm = hypot(step_x, step_y) or 1.0
            local_x, local_y = _rotate(
                point.x_m - origin_x, point.y_m - origin_y, cos_t, sin_t
            )
            dir_x, dir_y = _rotate(step_x / norm, step_y / norm, cos_t, sin_t)
            collected.append((distance, [local_x, local_y, dir_x, dir_y, 1.0]))

    collected.sort(key=lambda item: item[0])
    rows = np.zeros((MAX_MAP_POINTS, MAP_FEATURES), dtype=np.float32)
    for slot, (_, row) in enumerate(collected[:MAX_MAP_POINTS]):
        rows[slot] = row
    return rows


def scenario_examples(
    scenario: Scenario,
    *,
    include_all_agents: bool = False,
    min_future_points: int = 1,
) -> Iterator[Example]:
    """Yield examples for a scenario's prediction targets.

    ``include_all_agents`` widens the set from the official ``tracks_to_predict``
    to every agent with an anchor state, which yields several times more
    supervision per scenario. Use it for training only: evaluation must stay on
    the designated targets, or the numbers stop being comparable to the
    challenge definition.

    ``min_future_points`` drops agents whose future is too sparse to supervise.
    Roughly half of all agents leave the scene before the horizon ends, so their
    futures are partial rather than absent, and the mask carries that through.

    Agents added by ``include_all_agents`` must also pass a motion filter.
    WOMD designates prediction targets that are essentially always in motion,
    whereas most other agents in a scene are parked. Admitting them unfiltered
    shifts the training distribution towards standing still and teaches exactly
    the wrong behaviour; the designated targets themselves are never filtered.
    """

    timestamps = scenario.timestamps_s
    if len(timestamps) <= FUTURE_INDICES[-1]:
        return
    history_times = [round(timestamps[i], 4) for i in range(HISTORY_STEPS)]
    future_times = [round(timestamps[i], 4) for i in FUTURE_INDICES]
    current_time = history_times[CURRENT_INDEX]

    indexed = {track.agent_id: _states_by_time(track) for track in scenario.tracks}
    by_id = {track.agent_id: track for track in scenario.tracks}

    designated = set(scenario.tracks_to_predict)
    selected = tuple(by_id) if include_all_agents else scenario.tracks_to_predict
    for agent_id in selected:
        states = indexed.get(agent_id)
        if states is None:
            continue
        current = states.get(current_time)
        if current is None:
            continue  # no anchor state, so no frame to predict from

        origin_x, origin_y = current.x_m, current.y_m
        heading = current.heading_rad or 0.0
        cos_t, sin_t = cos(heading), sin(heading)

        target_history = np.array(
            [
                _agent_row(states.get(t), origin_x, origin_y, cos_t, sin_t)
                for t in history_times
            ],
            dtype=np.float32,
        )

        ranked: list[tuple[float, str]] = []
        for other_id, other_states in indexed.items():
            if other_id == agent_id:
                continue
            other_current = other_states.get(current_time)
            if other_current is None:
                continue
            distance = hypot(other_current.x_m - origin_x, other_current.y_m - origin_y)
            if distance <= NEIGHBOR_RADIUS_M:
                ranked.append((distance, other_id))
        ranked.sort(key=lambda item: item[0])

        neighbors = np.zeros(
            (MAX_NEIGHBORS, HISTORY_STEPS, NEIGHBOR_FEATURES), dtype=np.float32
        )
        for slot, (_, other_id) in enumerate(ranked[:MAX_NEIGHBORS]):
            other_states = indexed[other_id]
            type_index = OBJECT_TYPE_INDEX.get(by_id[other_id].object_type)
            for step, t in enumerate(history_times):
                row = _agent_row(other_states.get(t), origin_x, origin_y, cos_t, sin_t)
                neighbors[slot, step, :TARGET_FEATURES] = row
                if type_index is not None and row[-1] == 1.0:
                    neighbors[slot, step, TARGET_FEATURES + type_index] = 1.0

        future = np.zeros((len(FUTURE_INDICES), 2), dtype=np.float32)
        future_mask = np.zeros(len(FUTURE_INDICES), dtype=np.float32)
        for step, t in enumerate(future_times):
            state = states.get(t)
            if state is None:
                continue
            future[step] = _rotate(
                state.x_m - origin_x, state.y_m - origin_y, cos_t, sin_t
            )
            future_mask[step] = 1.0

        if future_mask.sum() < min_future_points:
            continue

        if agent_id not in designated:
            anchor_speed = hypot(
                current.velocity_x_mps or 0.0, current.velocity_y_mps or 0.0
            )
            valid = np.nonzero(future_mask)[0]
            displacement = float(np.linalg.norm(future[valid[-1]])) if len(valid) else 0.0
            if anchor_speed <= MIN_ANCHOR_SPEED_MPS and displacement <= MIN_DISPLACEMENT_M:
                continue

        yield Example(
            scenario_id=scenario.scenario_id,
            agent_id=agent_id,
            object_type=by_id[agent_id].object_type,
            target_history=target_history,
            neighbors=neighbors,
            map_points=_map_rows(scenario, origin_x, origin_y, cos_t, sin_t),
            future=future,
            future_mask=future_mask,
            origin=np.array([origin_x, origin_y], dtype=np.float32),
            heading=float(heading),
        )
