"""Header-only inspection of the complete pinned Hy4-flash source."""
from __future__ import annotations

import json
from pathlib import Path

from kimi_k3.uniform_k2.storage import file_identity, read_header, sha256_file
from . import recipe


class HeaderSource:
    """Read safetensors headers and bind the aliases to their current stat identities."""

    def __init__(self, snapshot):
        self.snapshot = Path(snapshot).expanduser().resolve(strict=True)
        if self.snapshot.name != recipe.BASE_REVISION:
            raise ValueError("Hy4-flash source does not name the pinned revision")
        config_path = self.snapshot / "config.json"
        index_path = self.snapshot / "model.safetensors.index.json"
        if (sha256_file(config_path) != recipe.SOURCE_CONFIG_SHA256
                or sha256_file(index_path) != recipe.SOURCE_INDEX_SHA256):
            raise ValueError("Hy4-flash pinned source metadata changed")
        self.document = json.loads(config_path.read_bytes())
        self.config = self.document["text_config"]
        self.index = json.loads(index_path.read_bytes())
        self.weight_map = self.index["weight_map"]
        by_shard = {}
        for name, shard in self.weight_map.items():
            by_shard.setdefault(shard, set()).add(name)
        expected = {f"model-{n:05d}-of-{recipe.SHARD_COUNT:05d}.safetensors"
                    for n in range(1, recipe.SHARD_COUNT + 1)}
        actual = {p.name for p in self.snapshot.glob("*.safetensors") if p.is_file()}
        if expected != by_shard.keys() or expected != actual:
            raise ValueError(f"Hy4-flash source shard inventory differs or download incomplete "
                             f"({len(expected & actual)}/{recipe.SHARD_COUNT})")
        self.locations, self.specs, self.sizes = {}, {}, {}
        self.raw_headers, self.shard_offsets, self.source_fingerprints = {}, {}, {}
        for shard, names in sorted(by_shard.items()):
            path = self.snapshot / shard
            identity = file_identity(path)
            header, data_start = read_header(path)
            if header.keys() - {"__metadata__"} != names or file_identity(path) != identity:
                raise ValueError(f"Hy4-flash source shard header or identity differs: {shard}")
            self.shard_offsets[shard] = data_start
            self.source_fingerprints[shard] = identity
            for name in sorted(names):
                spec = header[name]
                self.locations[name] = dict(shard=shard, source_name=name)
                self.specs[name] = (spec["dtype"], spec["shape"])
                self.sizes[name] = spec["data_offsets"][1] - spec["data_offsets"][0]
                self.raw_headers[name] = spec
        self.payload_bytes = sum(self.sizes.values())
        self.tensor_count = len(self.specs)
        declared = self.index["metadata"]["total_size"]
        if declared != self.payload_bytes:
            raise ValueError("Hy4-flash index payload total differs from headers")
        self.payload_totals = dict(declared=declared, measured=self.payload_bytes,
                                   reviewed=recipe.BASE_PAYLOAD_BYTES)
        self.audit = recipe.audit_source(self)

    def as_inventory(self):
        return dict(mapping=self.weight_map, locations=self.locations, specs=self.specs,
                    sizes=self.sizes, payload_bytes=self.payload_bytes,
                    tensor_count=self.tensor_count, payload_totals=self.payload_totals)

    def source_entries(self, names):
        return {name: (self.snapshot / self.locations[name]["shard"], name) for name in names}
