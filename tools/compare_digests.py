"""Compare two normalization digests and report where the decoders disagree.

Reads the JSON produced by ``tools/normalization_digest.py`` from the project
reader and from Waymo's official reader, and establishes whether they decoded
the shard identically. Exits 0 when they agree and 1 when they do not.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", help="a digest JSON")
    parser.add_argument("second", help="the digest JSON to compare against")
    args = parser.parse_args()

    a, b = load(args.first), load(args.second)
    label_a, label_b = a.get("reader", args.first), b.get("reader", args.second)

    if a.get("shard_name") != b.get("shard_name"):
        print("REFUSED: digests describe different shards, so a comparison "
              "would prove nothing.")
        print(f"  {label_a}: {a.get('shard_name')}")
        print(f"  {label_b}: {b.get('shard_name')}")
        return 1

    print(f"shard: {a.get('shard_name')}")
    print(f"  {label_a}: {a['scenario_count']} scenarios")
    print(f"  {label_b}: {b['scenario_count']} scenarios")

    if a["combined_digest"] == b["combined_digest"]:
        print(f"\nMATCH — {a['scenario_count']} scenarios decoded identically.")
        print(f"combined digest: {a['combined_digest']}")
        return 0

    print("\nMISMATCH")
    by_id_a = {entry["scenario_id"]: entry for entry in a["scenarios"]}
    by_id_b = {entry["scenario_id"]: entry for entry in b["scenarios"]}

    only_a = sorted(set(by_id_a) - set(by_id_b))
    only_b = sorted(set(by_id_b) - set(by_id_a))
    for scenario_id in only_a:
        print(f"  only in {label_a}: {scenario_id}")
    for scenario_id in only_b:
        print(f"  only in {label_b}: {scenario_id}")

    differing = [
        scenario_id
        for scenario_id in sorted(set(by_id_a) & set(by_id_b))
        if by_id_a[scenario_id]["digest"] != by_id_b[scenario_id]["digest"]
    ]
    print(f"  {len(differing)} shared scenarios differ")
    for scenario_id in differing[:20]:
        left, right = by_id_a[scenario_id], by_id_b[scenario_id]
        detail = []
        if left["track_count"] != right["track_count"]:
            detail.append(f"tracks {left['track_count']} vs {right['track_count']}")
        if left["state_count"] != right["state_count"]:
            detail.append(f"states {left['state_count']} vs {right['state_count']}")
        suffix = f" ({', '.join(detail)})" if detail else " (same shape, differing values)"
        print(f"    {scenario_id}{suffix}")
    if len(differing) > 20:
        print(f"    ... and {len(differing) - 20} more")
    return 1


if __name__ == "__main__":
    sys.exit(main())
