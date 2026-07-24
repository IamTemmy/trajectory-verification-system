"""Model-neutral JSON interchange for externally generated predictions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from math import isfinite
from pathlib import Path
import re

from .motion_submission import MAX_MODES, OFFICIAL_PREDICTION_STEPS
from ..models import Scenario
from ..predictions import (
    AgentPrediction,
    PredictedTrajectory,
    PredictionPoint,
    ScenarioPredictions,
)


SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class ExternalModelProvenance:
    model_name: str
    model_version: str
    source_repository: str
    source_revision: str
    checkpoint_sha256: str
    coordinate_frame: str
    future_data_used: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ExternalPredictionArtifact:
    schema_version: int
    provenance: ExternalModelProvenance
    predictions: tuple[ScenarioPredictions, ...]


def load_external_predictions(
    path: str | Path,
    scenarios: tuple[Scenario, ...],
) -> ExternalPredictionArtifact:
    """Load and validate model output without importing its ML framework."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("external prediction schema_version must be 1")
    provenance_data = data.get("provenance")
    if not isinstance(provenance_data, dict):
        raise ValueError("external predictions require provenance")
    provenance = _load_provenance(provenance_data)
    scenario_by_id = {item.scenario_id: item for item in scenarios}
    entries = data.get("predictions")
    if not isinstance(entries, list) or not entries:
        raise ValueError("external predictions must be a non-empty list")
    predictions = tuple(
        _scenario_predictions(item, scenario_by_id)
        for item in entries
        if isinstance(item, dict)
    )
    if len(predictions) != len(entries):
        raise ValueError("every external scenario prediction must be an object")
    identifiers = [item.scenario_id for item in predictions]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("external predictions contain duplicate scenario IDs")
    if set(identifiers) != set(scenario_by_id):
        missing = sorted(set(scenario_by_id) - set(identifiers))
        extra = sorted(set(identifiers) - set(scenario_by_id))
        raise ValueError(
            f"external scenario identities do not match; missing={missing}, extra={extra}"
        )
    return ExternalPredictionArtifact(1, provenance, predictions)


def write_external_predictions(
    artifact: ExternalPredictionArtifact, path: str | Path
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": artifact.schema_version,
        "provenance": artifact.provenance.to_dict(),
        "predictions": [
            {
                "scenario_id": scenario.scenario_id,
                "agents": [
                    {
                        "agent_id": agent.agent_id,
                        "modes": [
                            {
                                "confidence": mode.confidence,
                                "xy_m": [
                                    [point.x_m, point.y_m]
                                    for point in mode.points
                                ],
                            }
                            for mode in agent.trajectories
                        ],
                    }
                    for agent in scenario.agents
                ],
            }
            for scenario in artifact.predictions
        ],
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output


def _load_provenance(data: dict[str, object]) -> ExternalModelProvenance:
    required = (
        "model_name",
        "model_version",
        "source_repository",
        "source_revision",
        "checkpoint_sha256",
        "coordinate_frame",
        "future_data_used",
    )
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError(f"external provenance is missing: {', '.join(missing)}")
    text_names = tuple(name for name in required if name != "future_data_used")
    if any(not isinstance(data[name], str) for name in text_names):
        raise ValueError("external provenance text fields must be strings")
    text_values = {name: data[name].strip() for name in text_names}
    if any(not value for value in text_values.values()):
        raise ValueError("external provenance text fields must be non-empty")
    if text_values["coordinate_frame"] != "scenario_global":
        raise ValueError("coordinate_frame must be scenario_global")
    checksum = text_values["checkpoint_sha256"]
    if SHA256_PATTERN.fullmatch(checksum) is None:
        raise ValueError("checkpoint_sha256 must contain 64 lowercase hex characters")
    future_data_used = data["future_data_used"]
    if not isinstance(future_data_used, bool):
        raise ValueError("future_data_used must be boolean")
    if future_data_used:
        raise ValueError("external predictions using future ground truth are not admissible")
    return ExternalModelProvenance(
        text_values["model_name"],
        text_values["model_version"],
        text_values["source_repository"],
        text_values["source_revision"],
        checksum,
        text_values["coordinate_frame"],
        future_data_used,
    )


def _scenario_predictions(
    data: dict[str, object],
    scenario_by_id: dict[str, Scenario],
) -> ScenarioPredictions:
    scenario_id = str(data.get("scenario_id", ""))
    if scenario_id not in scenario_by_id:
        raise ValueError(f"external prediction has unknown scenario: {scenario_id}")
    scenario = scenario_by_id[scenario_id]
    agents_data = data.get("agents")
    if not isinstance(agents_data, list) or not agents_data:
        raise ValueError(f"external scenario {scenario_id} contains no agents")
    agents = tuple(
        _agent_prediction(item, scenario)
        for item in agents_data
        if isinstance(item, dict)
    )
    if len(agents) != len(agents_data):
        raise ValueError("every external agent prediction must be an object")
    actual = {item.agent_id for item in agents}
    expected = set(scenario.tracks_to_predict)
    if actual != expected:
        raise ValueError(
            f"external agents do not match tracks_to_predict for {scenario_id}; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    return ScenarioPredictions(scenario_id, agents)


def _agent_prediction(
    data: dict[str, object], scenario: Scenario
) -> AgentPrediction:
    agent_id = str(data.get("agent_id", ""))
    modes_data = data.get("modes")
    if not isinstance(modes_data, list) or not 1 <= len(modes_data) <= MAX_MODES:
        raise ValueError(f"agent {agent_id} must contain 1 to {MAX_MODES} modes")
    timeline = scenario.timestamps_s
    if len(timeline) <= OFFICIAL_PREDICTION_STEPS[-1]:
        raise ValueError("scenario does not contain the official prediction horizon")
    times = tuple(timeline[index] for index in OFFICIAL_PREDICTION_STEPS)
    modes = tuple(
        _mode(item, times)
        for item in modes_data
        if isinstance(item, dict)
    )
    if len(modes) != len(modes_data):
        raise ValueError("every external mode must be an object")
    total_confidence = sum(item.confidence for item in modes)
    if total_confidence <= 0:
        raise ValueError(f"agent {agent_id} mode confidences must sum above zero")
    normalized = tuple(
        PredictedTrajectory(
            item.confidence / total_confidence,
            item.points,
        )
        for item in modes
    )
    return AgentPrediction(agent_id, normalized)


def _mode(data: dict[str, object], times: tuple[float, ...]) -> PredictedTrajectory:
    confidence = float(data.get("confidence", 0.0))
    if not isfinite(confidence) or confidence < 0:
        raise ValueError("external mode confidence must be finite and non-negative")
    coordinates = data.get("xy_m")
    if not isinstance(coordinates, list) or len(coordinates) != len(times):
        raise ValueError("external mode must contain exactly 16 xy_m points")
    points = []
    for time_s, coordinate in zip(times, coordinates):
        if not isinstance(coordinate, list) or len(coordinate) != 2:
            raise ValueError("each external xy_m point must contain [x, y]")
        x_m, y_m = float(coordinate[0]), float(coordinate[1])
        if not isfinite(x_m) or not isfinite(y_m):
            raise ValueError("external coordinates must be finite")
        points.append(PredictionPoint(time_s, x_m, y_m))
    return PredictedTrajectory(confidence, tuple(points))
