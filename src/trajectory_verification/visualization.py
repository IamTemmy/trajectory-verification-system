"""Dependency-free SVG rendering for normalized trajectory scenarios."""

from __future__ import annotations

from html import escape
from math import isfinite
from pathlib import Path

from .models import Scenario


COLORS = {
    "vehicle": "#2563eb",
    "pedestrian": "#dc2626",
    "cyclist": "#16a34a",
    "other": "#7c3aed",
    "unset": "#64748b",
}


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
    for track in scenario.tracks:
        projected = [project(state.x_m, state.y_m) for state in track.states]
        if not projected:
            continue
        color = COLORS.get(track.object_type, "#64748b")
        stroke_width = 4 if track.agent_id == scenario.sdc_agent_id else 2
        point_text = " ".join(f"{x:.2f},{y:.2f}" for x, y in projected)
        elements.append(
            f'<polyline points="{point_text}" fill="none" stroke="{color}" '
            f'stroke-width="{stroke_width}" stroke-linecap="round" '
            'stroke-linejoin="round" opacity="0.85"/>'
        )
        end_x, end_y = projected[-1]
        elements.append(f'<circle cx="{end_x:.2f}" cy="{end_y:.2f}" r="5" fill="{color}"/>')
        label = escape(f"{track.agent_id} · {track.object_type}")
        elements.append(
            f'<text x="{end_x + 8:.2f}" y="{end_y - 8:.2f}" '
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
