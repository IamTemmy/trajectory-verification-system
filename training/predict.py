"""Score a trained checkpoint into the external-prediction contract.

The model works in a frame centred on each target, so predictions are rotated
and translated back to scenario-global coordinates before export. Output
satisfies schemas/external_predictions.schema.json, which the project's importer
converts into the official submission wire format for evaluation.

Only the designated ``tracks_to_predict`` are scored. The wider agent population
used for training would not be comparable to the challenge definition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from math import cos, sin
from pathlib import Path

import numpy as np
import torch

from trajectory_verification.adapters.womd import iter_womd_scenarios

from features import scenario_examples
from model import NUM_MODES, TrajectoryPredictor


def checkpoint_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            check=True, cwd=Path(__file__).resolve().parent,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def to_global(local: np.ndarray, origin: np.ndarray, heading: float) -> np.ndarray:
    """Undo the agent-centric transform applied during feature extraction."""

    c, s = cos(heading), sin(heading)
    rotation = np.array([[c, -s], [s, c]], dtype=np.float64)
    return local.astype(np.float64) @ rotation.T + origin.astype(np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("shards", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", default="tvs-crossattn")
    parser.add_argument("--model-version", default="0.1.0")
    parser.add_argument(
        "--source-repository",
        default="https://github.com/IamTemmy/trajectory-verification-system",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = state.get("config", {})
    model = TrajectoryPredictor(
        dim=config.get("dim", 128), blocks=config.get("blocks", 3)
    ).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    predictions: list[dict] = []
    scored = 0
    for shard in args.shards:
        for scenario in iter_womd_scenarios(shard):
            examples = list(scenario_examples(scenario))  # designated targets only
            if not examples:
                continue
            agents: list[dict] = []
            for start in range(0, len(examples), args.batch_size):
                chunk = examples[start:start + args.batch_size]
                batch = {
                    "target_history": torch.from_numpy(
                        np.stack([e.target_history for e in chunk])
                    ).to(device),
                    "neighbors": torch.from_numpy(
                        np.stack([e.neighbors for e in chunk])
                    ).to(device),
                    "map_points": torch.from_numpy(
                        np.stack([e.map_points for e in chunk])
                    ).to(device),
                }
                with torch.no_grad():
                    trajectories, logits = model(**batch)
                    confidences = torch.softmax(logits.float(), dim=-1).cpu().numpy()
                trajectories = trajectories.float().cpu().numpy()

                for row, example in enumerate(chunk):
                    modes = [
                        {
                            "confidence": float(confidences[row, mode]),
                            "xy_m": [
                                [float(x), float(y)]
                                for x, y in to_global(
                                    trajectories[row, mode], example.origin, example.heading
                                )
                            ],
                        }
                        for mode in range(NUM_MODES)
                    ]
                    agents.append({"agent_id": example.agent_id, "modes": modes})
                    scored += 1
            predictions.append({"scenario_id": scenario.scenario_id, "agents": agents})

    document = {
        "schema_version": 1,
        "provenance": {
            "model_name": args.model_name,
            "model_version": args.model_version,
            "source_repository": args.source_repository,
            "source_revision": source_revision(),
            "checkpoint_sha256": checkpoint_digest(args.checkpoint),
            "coordinate_frame": "scenario_global",
            "future_data_used": False,
        },
        "predictions": predictions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document), encoding="utf-8")
    print(f"scored {scored} agents across {len(predictions)} scenarios -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
