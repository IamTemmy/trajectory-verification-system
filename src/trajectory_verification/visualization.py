"""Dependency-free SVG rendering for normalized trajectory scenarios."""

from __future__ import annotations

from html import escape
from math import hypot, isfinite
from pathlib import Path

from .models import AgentTrack, Scenario, State
from .predictions import AgentPrediction, PredictedTrajectory


COLORS = {
    "vehicle": "#2563eb",
    "pedestrian": "#dc2626",
    "cyclist": "#16a34a",
    "other": "#7c3aed",
    "unset": "#64748b",
}

AGENT_PALETTE = (
    "#2563eb", "#ea580c", "#16a34a", "#7c3aed",
    "#db2777", "#0891b2", "#ca8a04", "#dc2626",
)

DEFAULT_COLOR = "#64748b"


def _assign_colors(scenario: Scenario) -> dict[str, str]:
    """Choose a stroke colour for every track.

    Scenes small enough to read agent by agent get one palette entry each, so
    that two agents sharing an object type stay distinguishable — the common
    case for a hand-authored conflict between two vehicles. Crowded scenes fall
    back to colouring by object type, because a real WOMD scenario carries
    dozens of tracks and per-agent hues would read as noise.
    """

    tracks = scenario.tracks
    if len(tracks) <= len(AGENT_PALETTE):
        return {
            track.agent_id: AGENT_PALETTE[index]
            for index, track in enumerate(tracks)
        }
    return {
        track.agent_id: COLORS.get(track.object_type, DEFAULT_COLOR)
        for track in tracks
    }


CHAR_WIDTH_PX = 6.6
TITLE_BASELINE_PX = 30


def _place_label(
    label: str, end_x: float, end_y: float, *, width_px: int, height_px: int
) -> tuple[float, float, str]:
    """Position an endpoint label so it stays inside the canvas.

    Labels sit to the right of the end marker by default. A track finishing near
    the right edge would otherwise render its label off-canvas, so those flip to
    the left of the marker instead. The vertical position is clamped clear of the
    title and the bottom edge.
    """

    estimated_width = CHAR_WIDTH_PX * len(label)
    if end_x + 10 + estimated_width > width_px - 6:
        label_x, anchor = end_x - 10, "end"
    else:
        label_x, anchor = end_x + 10, "start"
    label_y = min(max(end_y - 10, TITLE_BASELINE_PX + 18), height_px - 8)
    return label_x, label_y, anchor


def scenario_to_svg(
    scenario: Scenario,
    *,
    width_px: int = 900,
    height_px: int = 700,
    padding_px: int = 50,
) -> str:
    """Render all valid trajectories into a standalone SVG document."""

    if width_px <= padding_px * 2 or height_px <= padding_px * 2:
        raise ValueError("canvas must be larger than twice the padding")
    points = [
        (state.x_m, state.y_m)
        for track in scenario.tracks
        for state in track.states
        if isfinite(state.x_m) and isfinite(state.y_m)
    ]
    map_points = [
        (point.x_m, point.y_m)
        for lane in scenario.map_context.lanes
        for point in lane.polyline
    ] + [
        (point.x_m, point.y_m)
        for crosswalk in scenario.map_context.crosswalks
        for point in crosswalk.polygon
    ] + [
        (sign.position.x_m, sign.position.y_m)
        for sign in scenario.map_context.stop_signs
    ]
    points.extend(map_points)
    if not points:
        raise ValueError("scenario contains no finite trajectory points")
    xs, ys = zip(*points)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    x_span = max(max_x - min_x, 1.0)
    y_span = max(max_y - min_y, 1.0)
    scale = min(
        (width_px - 2 * padding_px) / x_span,
        (height_px - 2 * padding_px) / y_span,
    )

    def project(x_m: float, y_m: float) -> tuple[float, float]:
        x = padding_px + (x_m - min_x) * scale
        # SVG y increases downward; world y increases upward.
        y = height_px - padding_px - (y_m - min_y) * scale
        return x, y

    elements = [
        f'<rect width="{width_px}" height="{height_px}" fill="#f8fafc"/>',
        (
            f'<text x="{padding_px}" y="30" font-family="system-ui" '
            f'font-size="18" font-weight="600" fill="#0f172a">'
            f'{escape(scenario.scenario_id)}</text>'
        ),
    ]
    for lane in scenario.map_context.lanes:
        projected = [project(point.x_m, point.y_m) for point in lane.polyline]
        point_text = " ".join(f"{x:.2f},{y:.2f}" for x, y in projected)
        elements.append(
            f'<polyline points="{point_text}" fill="none" stroke="#cbd5e1" '
            'stroke-width="2" stroke-dasharray="7 5" opacity="0.9"/>'
        )
    for crosswalk in scenario.map_context.crosswalks:
        projected = [project(point.x_m, point.y_m) for point in crosswalk.polygon]
        point_text = " ".join(f"{x:.2f},{y:.2f}" for x, y in projected)
        elements.append(
            f'<polygon points="{point_text}" fill="#fef3c7" stroke="#f59e0b" '
            'stroke-width="1.5" opacity="0.65"/>'
        )
    for sign in scenario.map_context.stop_signs:
        x, y = project(sign.position.x_m, sign.position.y_m)
        elements.append(
            f'<rect x="{x - 4:.2f}" y="{y - 4:.2f}" width="8" height="8" '
            'fill="#dc2626" transform="rotate(45 ' + f'{x:.2f} {y:.2f}' + ')"/>'
        )
    colors = _assign_colors(scenario)
    for track in scenario.tracks:
        projected = [project(state.x_m, state.y_m) for state in track.states]
        if not projected:
            continue
        color = colors.get(track.agent_id, DEFAULT_COLOR)
        stroke_width = 4 if track.agent_id == scenario.sdc_agent_id else 2
        point_text = " ".join(f"{x:.2f},{y:.2f}" for x, y in projected)
        elements.append(
            f'<polyline points="{point_text}" fill="none" stroke="{color}" '
            f'stroke-width="{stroke_width}" stroke-linecap="round" '
            'stroke-linejoin="round" opacity="0.85"/>'
        )
        # Hollow marker where the track starts, filled where it ends, so the
        # direction of travel is readable from a still image.
        start_x, start_y = projected[0]
        elements.append(
            f'<circle cx="{start_x:.2f}" cy="{start_y:.2f}" r="4" fill="#f8fafc" '
            f'stroke="{color}" stroke-width="2"/>'
        )
        end_x, end_y = projected[-1]
        elements.append(f'<circle cx="{end_x:.2f}" cy="{end_y:.2f}" r="5" fill="{color}"/>')
        if track.agent_id == track.object_type:
            label = escape(track.agent_id)
        else:
            label = escape(f"{track.agent_id} · {track.object_type}")
        label_x, label_y, anchor = _place_label(
            label, end_x, end_y, width_px=width_px, height_px=height_px
        )
        elements.append(
            f'<text x="{label_x:.2f}" y="{label_y:.2f}" text-anchor="{anchor}" '
            f'font-family="system-ui" font-size="12" fill="#334155">{label}</text>'
        )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width_px}" '
        f'height="{height_px}" viewBox="0 0 {width_px} {height_px}">\n'
        + "\n".join(elements)
        + "\n</svg>\n"
    )


def write_scenario_svg(scenario: Scenario, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(scenario_to_svg(scenario), encoding="utf-8")
    return output


PREDICTION_STYLE = {
    "history": "#0f172a",
    "truth": "#059669",
    "best": "#dc2626",
    "mode": "#94a3b8",
    "context": "#cbd5e1",
}


def _agent_state_at(track: AgentTrack, time_s: float) -> State | None:
    for state in track.states:
        if abs(state.time_s - time_s) < 1e-4:
            return state
    return None


def prediction_to_svg(
    scenario: Scenario,
    prediction: AgentPrediction,
    *,
    width_px: int = 900,
    height_px: int = 620,
    padding_px: int = 56,
    context_radius_m: float = 60.0,
) -> str:
    """Draw one agent's predicted futures against what actually happened.

    Aggregate error tells you a model was some number of metres wrong. This
    shows where it went instead: observed history, the recorded future, every
    predicted mode, and the mode the evaluator selected as closest.
    """

    track = scenario.track(prediction.agent_id)
    modes = tuple(
        sorted(prediction.trajectories, key=lambda item: item.confidence, reverse=True)
    )
    horizon = {round(point.time_s, 4) for mode in modes for point in mode.points}
    split = min(horizon) if horizon else float("inf")

    history = [s for s in track.states if s.time_s < split]
    truth = [s for s in track.states if round(s.time_s, 4) in horizon]
    if not history:
        raise ValueError(f"agent {prediction.agent_id} has no observed history")

    anchor = history[-1]
    points: list[tuple[float, float]] = [(s.x_m, s.y_m) for s in history + truth]
    for mode in modes:
        points.extend((p.x_m, p.y_m) for p in mode.points)

    xs, ys = zip(*points)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)
    scale = min(
        (width_px - 2 * padding_px) / span_x, (height_px - 2 * padding_px) / span_y
    )
    offset_x = padding_px + ((width_px - 2 * padding_px) - span_x * scale) / 2
    offset_y = padding_px + ((height_px - 2 * padding_px) - span_y * scale) / 2

    def project(x_m: float, y_m: float) -> tuple[float, float]:
        return (
            offset_x + (x_m - min_x) * scale,
            height_px - offset_y - (y_m - min_y) * scale,
        )

    def polyline(coords, color, width, opacity, dash="") -> str:
        text = " ".join(f"{x:.2f},{y:.2f}" for x, y in coords)
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        return (
            f'<polyline points="{text}" fill="none" stroke="{color}" '
            f'stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round" '
            f'opacity="{opacity}"{extra}/>'
        )

    elements = [
        f'<rect width="{width_px}" height="{height_px}" fill="#f8fafc"/>',
    ]

    for lane in scenario.map_context.lanes:
        near = [
            project(p.x_m, p.y_m)
            for p in lane.polyline
            if hypot(p.x_m - anchor.x_m, p.y_m - anchor.y_m) <= context_radius_m
        ]
        if len(near) > 1:
            elements.append(polyline(near, PREDICTION_STYLE["context"], 2, 0.7, "6 5"))

    for other in scenario.tracks:
        if other.agent_id == prediction.agent_id:
            continue
        state = _agent_state_at(other, anchor.time_s)
        if state is None:
            continue
        if hypot(state.x_m - anchor.x_m, state.y_m - anchor.y_m) > context_radius_m:
            continue
        x, y = project(state.x_m, state.y_m)
        elements.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.5" fill="#cbd5e1" '
            'stroke="#94a3b8" stroke-width="1"/>'
        )

    # Every mode faintly, so the spread of the forecast is visible.
    anchor_point = project(anchor.x_m, anchor.y_m)
    best_index = 0
    if truth:
        def average_error(mode: PredictedTrajectory) -> float:
            paired = [
                (p, _agent_state_at(track, round(p.time_s, 4))) for p in mode.points
            ]
            usable = [(p, s) for p, s in paired if s is not None]
            if not usable:
                return float("inf")
            return sum(hypot(p.x_m - s.x_m, p.y_m - s.y_m) for p, s in usable) / len(usable)

        best_index = min(range(len(modes)), key=lambda i: average_error(modes[i]))

    for index, mode in enumerate(modes):
        if index == best_index:
            continue
        coords = [anchor_point] + [project(p.x_m, p.y_m) for p in mode.points]
        elements.append(
            polyline(coords, PREDICTION_STYLE["mode"], 2, 0.35 + 0.5 * mode.confidence)
        )

    if truth:
        elements.append(
            polyline(
                [anchor_point] + [project(s.x_m, s.y_m) for s in truth],
                PREDICTION_STYLE["truth"], 3.5, 0.95,
            )
        )

    chosen = modes[best_index]
    elements.append(
        polyline(
            [anchor_point] + [project(p.x_m, p.y_m) for p in chosen.points],
            PREDICTION_STYLE["best"], 3, 0.95, "8 4",
        )
    )

    elements.append(
        polyline(
            [project(s.x_m, s.y_m) for s in history],
            PREDICTION_STYLE["history"], 3.5, 0.9,
        )
    )
    elements.append(
        f'<circle cx="{anchor_point[0]:.2f}" cy="{anchor_point[1]:.2f}" r="6" '
        f'fill="{PREDICTION_STYLE["history"]}"/>'
    )

    legend = (
        ("observed history", PREDICTION_STYLE["history"], ""),
        ("recorded future", PREDICTION_STYLE["truth"], ""),
        ("closest predicted mode", PREDICTION_STYLE["best"], "8 4"),
        ("other modes", PREDICTION_STYLE["mode"], ""),
    )
    elements.append(
        f'<text x="{padding_px}" y="30" font-family="system-ui" font-size="17" '
        f'font-weight="600" fill="#0f172a">'
        f'{escape(scenario.scenario_id)} &#183; agent {escape(prediction.agent_id)}'
        "</text>"
    )
    for row, (label, color, dash) in enumerate(legend):
        y = height_px - 22 - row * 17
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        elements.append(
            f'<line x1="{padding_px}" y1="{y}" x2="{padding_px + 26}" y2="{y}" '
            f'stroke="{color}" stroke-width="3" stroke-linecap="round"{extra}/>'
        )
        elements.append(
            f'<text x="{padding_px + 34}" y="{y + 4}" font-family="system-ui" '
            f'font-size="12" fill="#334155">{escape(label)}</text>'
        )

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width_px}" '
        f'height="{height_px}" viewBox="0 0 {width_px} {height_px}">\n'
        + "\n".join(elements)
        + "\n</svg>\n"
    )


def write_prediction_svg(
    scenario: Scenario, prediction: AgentPrediction, path: str | Path
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(prediction_to_svg(scenario, prediction), encoding="utf-8")
    return output
