"""Copy the first N records of a TFRecord shard, byte for byte.

Uploading a full 235 MB shard to a remote runtime is slow, and a verification
run that fails after the upload wastes the whole transfer. Slicing a small
prefix makes the first end-to-end pass cheap.

Records are copied verbatim, including their CRC fields. Re-framing the records
would require CRC32C, which the standard library does not provide, and readers
that validate checksums — TensorFlow's among them — would reject the result.
Copying the original bytes sidesteps that entirely.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

HEADER = 12  # uint64 length + uint32 masked CRC of the length
FOOTER = 4   # uint32 masked CRC of the payload


def copy_records(source: Path, destination: Path, count: int) -> int:
    copied = 0
    with source.open("rb") as reader, destination.open("wb") as writer:
        while copied < count:
            header = reader.read(HEADER)
            if not header:
                break
            if len(header) != HEADER:
                raise ValueError(f"truncated header at record {copied}")
            (length,) = struct.unpack("<Q", header[:8])
            body = reader.read(length + FOOTER)
            if len(body) != length + FOOTER:
                raise ValueError(f"truncated payload at record {copied}")
            writer.write(header)
            writer.write(body)
            copied += 1
    return copied


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()

    copied = copy_records(args.source, args.destination, args.count)
    size_mb = args.destination.stat().st_size / 1_048_576
    print(f"copied {copied} records -> {args.destination} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
