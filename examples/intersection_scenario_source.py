"""Author the synthetic intersection committed as examples/intersection_scenario.json.

Five agents meet at an unsignalised four-way junction. The turning vehicle cuts
across the through vehicle's path with 3.97 m of minimum separation, which the
committed requirements flag as a failure; every other pairing stays clear, so the
demonstration reports exactly one violation.

Geometry is chosen so the scene's bounding box matches the 900x700 render canvas,
keeping the drawing free of dead space. Regenerate with:

    python examples/intersection_scenario_source.py
"""
import json
import math

DT, N = 0.1, 91              # 9.0 s at 10 Hz, matching the WOMD sampling cadence
V, V_THROUGH = 5.0, 6.4      # m/s
R, LANE = 8.0, 3.5           # left-turn radius and lane offset, m
X0, Y0, SOUTH_Y0 = -38.0, -22.0, 12.0

def through_vehicle(t):
    return (X0 + V_THROUGH * t, 0.0, 0.0)

def turning_vehicle(t):
    """Northbound vehicle turning left across the through lane."""
    s, s1, arc = V * t, -LANE - Y0, math.pi / 2 * R
    if s <= s1:
        return (LANE, Y0 + s, math.pi / 2)
    if s <= s1 + arc:
        a = (s - s1) / R
        return (LANE - R + R * math.cos(a), -LANE + R * math.sin(a), math.pi / 2 + a)
    return (LANE - R - (s - s1 - arc), -LANE + R, math.pi)

def southbound_vehicle(t):
    return (-LANE, SOUTH_Y0 - V * t, -math.pi / 2)

def cyclist(t):
    return (X0 + 12 + 0.5 * V_THROUGH * t, 9.0, 0.0)

def pedestrian(t):
    return (-12.0, -12.0 + 1.4 * t, math.pi / 2)

AGENTS = [
    ("through_vehicle", "vehicle", through_vehicle),
    ("turning_vehicle", "vehicle", turning_vehicle),
    ("southbound_vehicle", "vehicle", southbound_vehicle),
    ("cyclist", "cyclist", cyclist),
    ("pedestrian", "pedestrian", pedestrian),
]

ts = [round(i * DT, 1) for i in range(N)]
tracks = [
    {
        "agent_id": agent_id,
        "object_type": object_type,
        "states": [
            dict(zip(("time_s", "x_m", "y_m", "heading_rad"),
                     (t, *(round(v, 3) for v in fn(t)))))
            for t in ts
        ],
    }
    for agent_id, object_type, fn in AGENTS
]

json.dump(
    {
        "scenario_id": "synthetic-intersection-001",
        "tracks": tracks,
        "current_time_index": 10,
        "sdc_agent_id": "through_vehicle",
        "tracks_to_predict": ["turning_vehicle", "pedestrian"],
        "timestamps_s": ts,
    },
    open("examples/intersection_scenario.json", "w"),
    indent=2,
)

sep, at = min(
    (math.dist(through_vehicle(t)[:2], turning_vehicle(t)[:2]), t) for t in ts
)
print(f"minimum through/turning separation {sep:.2f} m at t={at}s")
