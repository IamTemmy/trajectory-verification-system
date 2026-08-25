"""Convert WOMD scenario shards into compact per-shard feature archives.

Decoding protobufs is far slower than reading arrays, so the conversion runs
once and training reads the result repeatedly. One archive per shard keeps the
job resumable: a dropped Colab session loses at most the shard in flight.

Reading goes through the project's own decoder, the one proven equivalent to
Waymo's official reader in docs/READER_VERIFICATION.md, so the training data
inherits that guarantee.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from trajectory_verification.adapters.womd import iter_womd_scenarios

from features import scenario_examples


def convert_shard(shard: Path, destination: Path, limit: int | None = None,
                  include_all_agents: bool = False) -> int:
    histories, neighbors, maps, futures, masks = [], [], [], [], []
    scenario_ids, agent_ids, types, origins, headings = [], [], [], [], []

    count = 0
    for scenario in iter_womd_scenarios(shard):
        for example in scenario_examples(
            scenario, include_all_agents=include_all_agents
        ):
            histories.append(example.target_history)
            neighbors.append(example.neighbors)
            maps.append(example.map_points)
            futures.append(example.future)
            masks.append(example.future_mask)
            scenario_ids.append(example.scenario_id)
            agent_ids.append(example.agent_id)
            types.append(example.object_type)
            origins.append(example.origin)
            headings.append(example.heading)
            count += 1
        if limit is not None and count >= limit:
            break

    if not histories:
        return 0

    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        target_history=np.stack(histories),
        neighbors=np.stack(neighbors),
        map_points=np.stack(maps),
        future=np.stack(futures),
        future_mask=np.stack(masks),
        origin=np.stack(origins),
        heading=np.array(headings, dtype=np.float32),
        scenario_id=np.array(scenario_ids),
        agent_id=np.array(agent_ids),
        object_type=np.array(types),
    )
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after this many examples per shard")
    parser.add_argument("--all-agents", action="store_true",
                        help="train on every agent, not only official targets")
    parser.add_argument("--skip-existing", action="store_true",
                        help="leave already-converted shards alone, for resuming")
    args = parser.parse_args()

    total = 0
    for shard in args.shards:
        destination = args.out_dir / f"{shard.name}.npz"
        if args.skip_existing and destination.exists():
            print(f"skip   {shard.name} (exists)", flush=True)
            continue
        started = time.time()
        written = convert_shard(shard, destination, limit=args.limit,
                                include_all_agents=args.all_agents)
        elapsed = time.time() - started
        size_mb = destination.stat().st_size / 1_048_576 if written else 0.0
        print(f"{shard.name}: {written} examples, {elapsed:.0f}s, {size_mb:.0f} MB",
              flush=True)
        total += written
    print(f"total examples: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
