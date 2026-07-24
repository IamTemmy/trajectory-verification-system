"""Reproducible motion-prediction experiments driven by one manifest."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any

from .adapters.motion_submission import load_motion_submission, write_motion_submission
from .adapters.womd import iter_womd_scenarios
from .baselines import baseline_predictions
from .prediction_compare_cli import comparison_to_html, comparison_to_markdown
from .prediction_comparison import (
    PredictionComparisonPolicy,
    compare_prediction_evaluations,
)
from .prediction_metrics import score_scenario_predictions
from .prediction_reporting import PredictionEvaluation, write_prediction_reports


SUPPORTED_MODELS = {
    "constant_velocity",
    "constant_acceleration",
    "constant_turn_rate",
    "kinematic_ensemble",
}


@dataclass(frozen=True, slots=True)
class CandidateConfig:
    name: str
    model: str


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    experiment_id: str
    manifest_path: Path
    shards: tuple[Path, ...]
    dataset_name: str
    dataset_version: str
    candidates: tuple[CandidateConfig, ...]
    baseline_name: str
    candidate_name: str
    miss_threshold_m: float
    evaluation_bootstrap_samples: int
    evaluation_bootstrap_seed: int
    comparison_bootstrap_samples: int
    comparison_bootstrap_seed: int
    policy: PredictionComparisonPolicy
    output_directory: Path
    limit: int | None = None


def load_experiment_manifest(path: str | Path) -> ExperimentConfig:
    manifest_path = Path(path).resolve()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    experiment_id = _required_text(data, "experiment_id")
    _validate_identifier(experiment_id, "experiment_id")
    dataset = _required_mapping(data, "dataset")
    shard_entries = dataset.get("shards")
    if not isinstance(shard_entries, list) or not shard_entries:
        raise ValueError("dataset.shards must be a non-empty list")
    candidates_data = data.get("candidates")
    if not isinstance(candidates_data, list) or len(candidates_data) < 2:
        raise ValueError("candidates must contain at least two entries")
    candidates = tuple(
        CandidateConfig(
            _required_text(item, "name"),
            _required_text(item, "model"),
        )
        for item in candidates_data
        if isinstance(item, dict)
    )
    if len(candidates) != len(candidates_data):
        raise ValueError("every candidate must be an object")
    names = [item.name for item in candidates]
    for name in names:
        _validate_identifier(name, "candidate name")
    if len(names) != len(set(names)):
        raise ValueError("candidate names must be unique")
    unsupported = sorted({item.model for item in candidates} - SUPPORTED_MODELS)
    if unsupported:
        raise ValueError(f"unsupported candidate models: {', '.join(unsupported)}")
    comparison = _required_mapping(data, "comparison")
    baseline_name = _required_text(comparison, "baseline")
    candidate_name = _required_text(comparison, "candidate")
    if baseline_name not in names or candidate_name not in names:
        raise ValueError("comparison names must reference configured candidates")
    if baseline_name == candidate_name:
        raise ValueError("comparison baseline and candidate must differ")
    evaluation = data.get("evaluation", {})
    if not isinstance(evaluation, dict):
        raise ValueError("evaluation must be an object")
    outputs = _required_mapping(data, "outputs")
    policy_data = comparison.get("policy", {})
    if not isinstance(policy_data, dict):
        raise ValueError("comparison.policy must be an object")
    limit = data.get("limit")
    if limit is not None and (not isinstance(limit, int) or limit < 1):
        raise ValueError("limit must be a positive integer")
    evaluation_samples = int(evaluation.get("bootstrap_samples", 1000))
    comparison_samples = int(comparison.get("bootstrap_samples", 2000))
    if evaluation_samples < 0 or comparison_samples < 0:
        raise ValueError("bootstrap sample counts must be non-negative")
    miss_threshold_m = float(evaluation.get("miss_threshold_m", 2.0))
    if miss_threshold_m <= 0:
        raise ValueError("evaluation.miss_threshold_m must be positive")
    return ExperimentConfig(
        experiment_id=experiment_id,
        manifest_path=manifest_path,
        shards=tuple((root / str(item)).resolve() for item in shard_entries),
        dataset_name=_required_text(dataset, "name"),
        dataset_version=_required_text(dataset, "version"),
        candidates=candidates,
        baseline_name=baseline_name,
        candidate_name=candidate_name,
        miss_threshold_m=miss_threshold_m,
        evaluation_bootstrap_samples=evaluation_samples,
        evaluation_bootstrap_seed=int(evaluation.get("bootstrap_seed", 0)),
        comparison_bootstrap_samples=comparison_samples,
        comparison_bootstrap_seed=int(comparison.get("bootstrap_seed", 0)),
        policy=PredictionComparisonPolicy.from_dict(policy_data),
        output_directory=(root / _required_text(outputs, "directory")).resolve(),
        limit=limit,
    )


def run_experiment(config: ExperimentConfig) -> dict[str, object]:
    missing = [str(path) for path in config.shards if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"dataset shards not found: {', '.join(missing)}")
    scenarios = tuple(iter_womd_scenarios(config.shards))
    if config.limit is not None:
        scenarios = scenarios[: config.limit]
    if not scenarios:
        raise ValueError("experiment decoded no scenarios")
    identifiers = [scenario.scenario_id for scenario in scenarios]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("dataset shards contain duplicate scenario IDs")

    output = config.output_directory
    output.mkdir(parents=True, exist_ok=True)
    evaluations: dict[str, dict[str, object]] = {}
    artifact_paths: list[Path] = []
    for candidate in config.candidates:
        submission_path = output / f"{candidate.name}.binproto"
        predictions = tuple(
            baseline_predictions(scenario, candidate.model) for scenario in scenarios
        )
        write_motion_submission(predictions, submission_path)
        evaluation = _evaluate_submission(config, scenarios, submission_path)
        payload = evaluation.to_dict()
        evaluations[candidate.name] = payload
        json_path = output / f"{candidate.name}-evaluation.json"
        markdown_path = output / f"{candidate.name}-evaluation.md"
        html_path = output / f"{candidate.name}-evaluation.html"
        json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        write_prediction_reports(evaluation, markdown_path, html_path)
        artifact_paths.extend((submission_path, json_path, markdown_path, html_path))

    comparison = compare_prediction_evaluations(
        evaluations[config.baseline_name],
        evaluations[config.candidate_name],
        config.policy,
        bootstrap_samples=config.comparison_bootstrap_samples,
        bootstrap_seed=config.comparison_bootstrap_seed,
    )
    comparison_payload = comparison.to_dict()
    comparison_json = output / "comparison.json"
    comparison_markdown = output / "comparison.md"
    comparison_html = output / "comparison.html"
    comparison_json.write_text(
        json.dumps(comparison_payload, indent=2) + "\n", encoding="utf-8"
    )
    comparison_markdown.write_text(
        comparison_to_markdown(comparison), encoding="utf-8"
    )
    comparison_html.write_text(comparison_to_html(comparison), encoding="utf-8")
    artifact_paths.extend((comparison_json, comparison_markdown, comparison_html))

    index = {
        "schema_version": 1,
        "experiment_id": config.experiment_id,
        "source_revision": _source_revision(config.manifest_path.parent),
        "dataset": {
            "name": config.dataset_name,
            "version": config.dataset_version,
            "scenario_count": len(scenarios),
            "shards": [_artifact_record(path, config.manifest_path.parent) for path in config.shards],
        },
        "configuration": {
            "manifest": _artifact_record(
                config.manifest_path, config.manifest_path.parent
            ),
            "limit": config.limit,
            "miss_threshold_m": config.miss_threshold_m,
            "evaluation_bootstrap": {
                "samples": config.evaluation_bootstrap_samples,
                "seed": config.evaluation_bootstrap_seed,
            },
            "comparison_bootstrap": {
                "samples": config.comparison_bootstrap_samples,
                "seed": config.comparison_bootstrap_seed,
            },
            "candidates": [
                {"name": item.name, "model": item.model}
                for item in config.candidates
            ],
            "comparison": {
                "baseline": config.baseline_name,
                "candidate": config.candidate_name,
            },
        },
        "result": {
            "gate_passed": comparison.gate_passed,
            "deltas": comparison.deltas,
            "violations": list(comparison.violations),
        },
        "artifacts": [
            _artifact_record(path, output) for path in sorted(artifact_paths)
        ],
    }
    index_path = output / "experiment-index.json"
    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return index


def _evaluate_submission(
    config: ExperimentConfig, scenarios, submission_path: Path
) -> PredictionEvaluation:
    truth_by_id = {item.scenario_id: item for item in scenarios}
    predictions = load_motion_submission(submission_path, scenarios)
    scores = tuple(
        score_scenario_predictions(
            truth_by_id[item.scenario_id],
            item,
            miss_threshold_m=config.miss_threshold_m,
        )
        for item in predictions
    )
    return PredictionEvaluation(
        scores,
        config.miss_threshold_m,
        config.evaluation_bootstrap_samples,
        config.evaluation_bootstrap_seed,
    )


def _artifact_record(path: Path, relative_to: Path) -> dict[str, object]:
    return {
        "path": os.path.relpath(path, relative_to),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_revision(root: Path) -> dict[str, object]:
    configured = os.environ.get("TVS_SOURCE_REVISION")
    if configured:
        return {"commit": configured, "dirty": None, "source": "environment"}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty, "source": "git"}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "source": "unavailable"}


def _required_mapping(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _required_text(data: dict[str, Any], name: str) -> str:
    value = data.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _validate_identifier(value: str, label: str) -> None:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value) is None:
        raise ValueError(
            f"{label} must use only letters, numbers, underscores, and hyphens"
        )
