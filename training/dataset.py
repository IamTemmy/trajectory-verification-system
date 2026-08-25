"""Memory-mapped access to preprocessed features.

Per-shard archives are compressed, which keeps the preprocessing job resumable
but makes them slow to read repeatedly. Consolidating once into flat ``.npy``
files lets training memory-map them, so the working set is bounded by what the
sampler touches rather than by the size of the corpus. That is what allows the
shard count to grow without the training loop running out of RAM.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

ARRAYS = ("target_history", "neighbors", "map_points", "future", "future_mask")
LABELS = ("scenario_id", "agent_id", "object_type")

# Index 0 is reserved for an unknown or absent type.
OBJECT_TYPE_INDEX = {"vehicle": 1, "pedestrian": 2, "cyclist": 3}


def consolidate(archive_dir: Path, out_dir: Path) -> int:
    """Merge per-shard archives into flat memory-mappable arrays."""

    archives = sorted(archive_dir.glob("*.npz"))
    if not archives:
        raise FileNotFoundError(f"no .npz archives under {archive_dir}")

    counts, shapes = [], {}
    for path in archives:
        with np.load(path, allow_pickle=True) as data:
            counts.append(int(data["target_history"].shape[0]))
            for name in ARRAYS:
                shapes[name] = data[name].shape[1:]
    total = sum(counts)

    out_dir.mkdir(parents=True, exist_ok=True)
    writers = {
        name: np.lib.format.open_memmap(
            out_dir / f"{name}.npy", mode="w+",
            dtype=np.float32, shape=(total, *shapes[name]),
        )
        for name in ARRAYS
    }
    labels: dict[str, list[str]] = {name: [] for name in LABELS}

    offset = 0
    for path in archives:
        with np.load(path, allow_pickle=True) as data:
            size = int(data["target_history"].shape[0])
            for name in ARRAYS:
                writers[name][offset:offset + size] = data[name]
            for name in LABELS:
                labels[name].extend(str(value) for value in data[name])
            offset += size
    for writer in writers.values():
        writer.flush()

    (out_dir / "labels.json").write_text(json.dumps(labels), encoding="utf-8")
    (out_dir / "meta.json").write_text(
        json.dumps({"count": total, "shards": [p.name for p in archives]}),
        encoding="utf-8",
    )
    return total


def object_type_codes(directory: Path) -> np.ndarray:
    """Per-example object type as an integer code, in consolidation order."""

    labels = json.loads((Path(directory) / "labels.json").read_text(encoding="utf-8"))
    return np.array(
        [OBJECT_TYPE_INDEX.get(name, 0) for name in labels["object_type"]],
        dtype=np.int64,
    )


class FeatureDataset(Dataset):
    def __init__(self, directory: Path, indices: np.ndarray | None = None) -> None:
        self.directory = Path(directory)
        self.arrays = {
            name: np.load(self.directory / f"{name}.npy", mmap_mode="r")
            for name in ARRAYS
        }
        self.object_type = object_type_codes(self.directory)
        count = self.arrays["target_history"].shape[0]
        self.indices = np.arange(count) if indices is None else indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> dict[str, torch.Tensor]:
        row = int(self.indices[position])
        item = {
            # Copy out of the memory map: torch cannot own a read-only buffer.
            name: torch.from_numpy(
                np.array(self.arrays[name][row], dtype=np.float32, copy=True)
            )
            for name in ARRAYS
        }
        item["object_type"] = torch.tensor(self.object_type[row], dtype=torch.long)
        return item


def sampling_weights(codes: np.ndarray, damping: float = 0.5) -> np.ndarray:
    """Per-example weights that lift under-represented object types.

    The training population is roughly 93% vehicles, and the classes the model
    saw least are the ones it degrades most. Full inverse frequency would
    over-correct and starve the majority class, so the weight is raised to a
    damping power: 0 leaves the distribution alone, 1 equalises it outright.
    """

    values, counts = np.unique(codes, return_counts=True)
    frequency = {int(v): c / counts.sum() for v, c in zip(values, counts)}
    return np.array(
        [(1.0 / frequency[int(code)]) ** damping for code in codes], dtype=np.float64
    )


def split_indices(count: int, holdout: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic train/monitor split.

    The monitor set only tracks progress during training. Reported results come
    from the WOMD validation shard, scored through the project's own evaluator
    on the designated prediction targets.
    """

    generator = np.random.default_rng(seed)
    order = generator.permutation(count)
    cut = int(count * (1.0 - holdout))
    return order[:cut], order[cut:]
