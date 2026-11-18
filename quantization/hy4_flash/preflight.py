"""Read-only Hy4-flash source audit and scale-inclusive package projection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from kimi_k3.uniform_k2.storage import atomic_json
from .source import HeaderSource


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    source = HeaderSource(args.source)
    result = dict(status="passed", verification="complete source index and safetensors headers",
                  source_snapshot=str(source.snapshot), source_shards=len(source.shard_offsets),
                  source_identities=source.source_fingerprints, **source.audit)
    if args.receipt:
        atomic_json(args.receipt, result)
    print(json.dumps({key: value for key, value in result.items() if key != "source_identities"}, indent=2))


if __name__ == "__main__":
    main()
