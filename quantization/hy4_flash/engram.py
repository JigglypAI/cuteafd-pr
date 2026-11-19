"""Stream Hy4 tables and verify the FP8 alternate against the stock BF16 lookup."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import importlib
import json
import os
from pathlib import Path
import struct
import sys

from kimi_k3.uniform_k2.storage import (
    atomic_json, canonical, file_identity, fsync_directory, read_header, sha256_file, verify_receipt,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/lib"))
import mc_manifest

TABLE_ROWS = {1: 164_994_380, 11: 164_999_008, 21: 165_004_086, 31: 165_008_942}
TENSOR_TEMPLATE = "model.layers.{}.engram_layer.multi_head_embedding.embedding.weight"
WIDTH = 96
ROW_BYTES = mc_manifest.ENGRAM_ROW_BYTES
MAX_PART_BYTES = mc_manifest.ENGRAM_MAX_PART_BYTES
MAX_CHUNK_ROWS = 131_072
REFERENCE_REVISION = "c2ec850510a37978ea83df226589623d5f2be776"
REFERENCE_SHA256 = "43341929e4bec215c224fc837b24b534b4de41954992673a0655a65060e38b9b"
SOURCE_REVISION = "d88c9dfd96ba4f2e6b92adba0b8e8f1c8696f959"


@lru_cache(maxsize=1)
def vendor_round_trip():
    module = importlib.import_module("transformers.models.hy_v4_flash.modeling_hy_v4_flash")
    if sha256_file(module.__file__) != REFERENCE_SHA256:
        raise ValueError("Engram proof requires the pinned Transformers model source")
    return module._fp8_round_trip


def _codes_and_scale(x):
    import torch

    maximum = torch.finfo(torch.float8_e4m3fn).max
    # Transformers floors the maximum before the divide.
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / maximum
    codes = (x / scale).clamp(-maximum, maximum).to(torch.float8_e4m3fn)
    return codes, scale


def pack_rows(values):
    """Select E4M3 codes in BF16 and store each rounded scale as F32."""
    import torch

    if sys.byteorder != "little":
        raise ValueError("FP8 Engram export requires little-endian scale storage")
    if values.dtype != torch.bfloat16 or values.ndim != 2 or values.shape[1] != WIDTH or not values.shape[0]:
        raise ValueError("FP8 Engram codec requires nonempty BF16 [rows, 96]")
    codes, scale = _codes_and_scale(values)
    return torch.cat((codes.view(torch.uint8), scale.float().contiguous().view(torch.uint8)), dim=1)


def decode_rows(packed):
    """Decode stored codes and scale with the specified final BF16 cast."""
    import torch

    if packed.dtype != torch.uint8 or packed.ndim != 2 or packed.shape[1] != ROW_BYTES:
        raise ValueError("invalid FP8 Engram row buffer")
    codes = packed[:, :WIDTH].contiguous().view(torch.float8_e4m3fn).float()
    # A one-row slice can be contiguous at a byte offset not aligned for F32.
    scale = packed[:, WIDTH:].clone(memory_format=torch.contiguous_format).view(torch.float32)
    return (codes * scale).to(torch.bfloat16)


def _checked_pack(values, tensor, row_start):
    import torch

    finite = torch.isfinite(values)
    if not finite.all().item():
        first = int((~finite).any(dim=1).nonzero()[0].item())
        raise ValueError(f"nonfinite Engram source for {tensor} at row {row_start + first}")
    reference = vendor_round_trip()(values)
    packed = pack_rows(values)
    # The complete expression can change the sign of a zero code.
    codes = packed[:, :WIDTH]
    zero_codes = codes.bitwise_and(0x7F) == 0
    signed_zero = torch.signbit(reference).to(torch.uint8) * 0x80
    codes.copy_(torch.where(zero_codes, signed_zero, codes))
    same = decode_rows(packed).view(torch.int16) == reference.view(torch.int16)
    different_rows = (~same).any(dim=1)
    if different_rows.any().item():
        mismatches = different_rows.nonzero().flatten()
        first_rows = [row_start + index for index in mismatches[:16].tolist()]
        raise ValueError(
            f"FP8 Engram BF16 bit mismatch for {tensor}: {len(mismatches)} rows in chunk, "
            f"first {first_rows[0]}, examples {first_rows}"
        )
    return packed


def _shard_prefix(tensor, dtype, shape, payload_bytes, metadata=None):
    header = {tensor: dict(dtype=dtype, shape=shape, data_offsets=[0, payload_bytes])}
    if metadata:
        header["__metadata__"] = metadata
    encoded = canonical(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    return struct.pack("<Q", len(encoded)) + encoded


def _part_specs(tensor, layer, rows, limit):
    specs = []
    start = 0
    while start < rows:
        number = len(specs)
        if number >= 100_000:
            raise ValueError("too many Engram parts")
        name, filename = mc_manifest.engram_part_names(tensor, layer, number)
        meta = dict(engram_layout=mc_manifest.ENGRAM_LAYOUT, row_start=str(start))
        count = min(rows - start, (limit - 1) // ROW_BYTES)
        while count > 0:
            prefix = _shard_prefix(name, "U8", [count, ROW_BYTES], count * ROW_BYTES, meta)
            excess = len(prefix) + count * ROW_BYTES - limit + 1
            if excess <= 0:
                break
            count -= (excess + ROW_BYTES - 1) // ROW_BYTES
        if count <= 0:
            raise ValueError("Engram part size cannot fit one row and its header")
        specs.append((name, filename, start, count, prefix))
        start += count
    return specs


class _Shard:
    def __init__(self, root, filename, prefix, payload_bytes):
        self.path = root / filename
        self.temporary = self.path.with_suffix(".safetensors.partial")
        self.prefix = prefix
        self.payload_bytes = payload_bytes
        self.digest = hashlib.sha256()
        self.written = 0

    def write(self, stream, data):
        if stream.write(data) != len(data):
            raise OSError("short safetensors write")
        self.digest.update(data)
        self.written += len(data)

    def open(self, stack):
        stream = stack.enter_context(self.temporary.open("xb"))
        self.write(stream, self.prefix)
        return stream

    def flush(self, stream):
        if self.written != len(self.prefix) + self.payload_bytes:
            raise ValueError("incomplete Engram shard extent")
        stream.flush()
        os.fsync(stream.fileno())

    def commit(self):
        os.replace(self.temporary, self.path)
        return dict(sha256=self.digest.hexdigest(), file_identity=file_identity(self.path),
                    payload_bytes=self.payload_bytes)


def _table_plan(output, source, tensor, layer, reference_revision, device, limit):
    source = Path(source).resolve(strict=True)
    before = file_identity(source)
    header, base = read_header(source)
    value = header.get(tensor, {})
    rows = TABLE_ROWS[layer]
    if value.get("dtype") != "BF16" or value.get("shape") != [rows, WIDTH]:
        raise ValueError(f"unexpected Hy4 Engram source header for {tensor}")
    if file_identity(source) != before:
        raise ValueError("Engram source changed while reading header")
    if source == output or output in source.parents:
        raise ValueError("Engram source must not live under output")
    offset = base + value["data_offsets"][0]
    parts = _part_specs(tensor, layer, rows, limit)
    bf16 = f"engram-bf16-layer-{layer:02d}.safetensors"
    identity = dict(tensor=tensor, source_path=str(source), source_file_identity=before, device=str(device),
                    source_offset=offset, rows=rows, layout=mc_manifest.ENGRAM_LAYOUT,
                    recipe=mc_manifest.ENGRAM_RECIPE, reference_revision=reference_revision,
                    reference_sha256=REFERENCE_SHA256,
                    files=[bf16, *(part[1] for part in parts)], limit=limit)
    return identity, parts


def _regular_owned(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_uid != os.getuid():
        raise ValueError(f"Engram output is not an owned regular file: {path}")


def _write_table(output, identity, parts, device, chunk_rows, progress):
    import torch

    tensor = identity["tensor"]
    layer = mc_manifest.engram_layer(tensor)
    known = [output / n for n in identity["files"]]
    receipt_path = output / f"engram-layer-{layer:02d}.receipt.json"
    plan_path = output / f"engram-layer-{layer:02d}.plan.json"
    paths = known + [p.with_suffix(".safetensors.partial") for p in known]
    if plan_path.exists() or plan_path.is_symlink():
        _regular_owned(plan_path)
        if json.loads(plan_path.read_text()) != identity:
            raise ValueError("Engram recovery identity changed")
    else:
        if any(p.exists() or p.is_symlink() for p in [receipt_path, *paths]):
            raise ValueError("unowned Engram output exists")
        atomic_json(plan_path, identity)
    if receipt_path.exists() or receipt_path.is_symlink():
        _regular_owned(receipt_path)
        saved = json.loads(receipt_path.read_text())
        if saved["identity"] != identity or set(saved["files"]) != set(identity["files"]):
            raise ValueError("Engram receipt inventory differs from validated plan")
        if saved.get("proof") != _table_proof(identity["rows"]):
            raise ValueError("Engram receipt lacks a complete BF16 proof")
        for filename, recorded in saved["files"].items():
            _regular_owned(output / filename)
            verify_receipt(output / filename, recorded)
        if progress:
            progress(tensor, identity["rows"], identity["rows"])
        return saved
    # A matching plan owns its incomplete table. Replaying that table never
    # alters other committed tables or the source.
    for path in paths:
        if path.exists() or path.is_symlink():
            _regular_owned(path)
            path.unlink()
    source = Path(identity["source_path"])
    rows = identity["rows"]
    bf16_bytes = rows * WIDTH * 2
    bf16_prefix = _shard_prefix(tensor, "BF16", [rows, WIDTH], bf16_bytes, {"format": "pt"})
    bf16 = _Shard(output, known[0].name, bf16_prefix, bf16_bytes)
    tensor_digest = hashlib.sha256()
    files, manifest_parts = {}, []
    next_progress = 1
    with ExitStack() as outer, torch.inference_mode():
        original = outer.enter_context(source.open("rb"))
        original.seek(identity["source_offset"])
        native_stream = bf16.open(outer)
        row_bytes = WIDTH * 2
        for name, filename, start, count, prefix in parts:
            alternate = _Shard(output, filename, prefix, count * ROW_BYTES)
            with ExitStack() as inner:
                fp8_stream = alternate.open(inner)
                for position in range(start, start + count, chunk_rows):
                    n = min(chunk_rows, start + count - position)
                    raw = bytearray(original.read(n * row_bytes))
                    if len(raw) != n * row_bytes:
                        raise ValueError("short Engram source span")
                    values = torch.frombuffer(raw, dtype=torch.bfloat16).reshape(n, WIDTH)
                    values = values.to(device)
                    packed = _checked_pack(values, tensor, position)
                    bf16.write(native_stream, raw)
                    tensor_digest.update(raw)
                    alternate.write(fp8_stream, memoryview(packed.cpu().numpy()).cast("B"))
                    processed = position + n
                    if progress and (processed >= next_progress or processed == rows):
                        progress(tensor, processed, rows)
                        next_progress = processed + 4_194_304
                alternate.flush(fp8_stream)
            files[filename] = alternate.commit()
            manifest_parts.append(dict(tensor=name, file=filename, rows=count, sha256=files[filename]["sha256"]))
        bf16.flush(native_stream)
    if file_identity(source) != identity["source_file_identity"]:
        raise ValueError("native Engram source changed during streaming export")
    files[known[0].name] = bf16.commit()
    fsync_directory(output)
    digest = tensor_digest.hexdigest()
    files[known[0].name]["original_tensor_proofs"] = {
        tensor: dict(sha256=digest, source_name=tensor, source_path=str(source),
                     source_file_identity=identity["source_file_identity"])}
    report = dict(identity=identity, files=files, table=known[0].name,
                  manifest=dict(tensor=tensor, sha256=digest, parts=manifest_parts),
                  proof=_table_proof(rows))
    atomic_json(receipt_path, report)
    return report


def _table_proof(rows):
    return dict(rows=rows, mismatch_count=0, reference_dtype="BF16",
                expression="x + (decoded - x)")


def export_tables(output, sources, reference_revision, *, device="cpu", chunk_rows=65_536,
                  max_part_bytes=MAX_PART_BYTES, progress=None):
    """Write both Engram forms directly, with SHA256 and bit proofs in one pass.

    `sources` maps each source table tensor name to its immutable shard path.
    `device` controls bounded conversion chunks only. No full table is loaded.
    """
    output = Path(output).resolve()
    if reference_revision != REFERENCE_REVISION:
        raise ValueError("Engram export requires the pinned Transformers reference commit")
    vendor_round_trip()
    if type(chunk_rows) is not int or not 1 <= chunk_rows <= MAX_CHUNK_ROWS:
        raise ValueError("invalid Engram conversion chunk size")
    if type(max_part_bytes) is not int or not 1 <= max_part_bytes <= MAX_PART_BYTES:
        raise ValueError("invalid Engram part byte cap")
    names = {TENSOR_TEMPLATE.format(layer) for layer in TABLE_ROWS}
    if set(sources) != names:
        raise ValueError("Hy4 requires its four exact Engram tables")
    plans = []
    for layer in sorted(TABLE_ROWS):
        tensor = TENSOR_TEMPLATE.format(layer)
        plans.append(_table_plan(output, sources[tensor], tensor, layer,
                                 reference_revision, device, max_part_bytes))
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(layout=mc_manifest.ENGRAM_LAYOUT, row_bytes=ROW_BYTES, code_dtype="F8_E4M3",
                    scale_dtype="F32", recipe=f"{mc_manifest.ENGRAM_RECIPE}; reference={reference_revision}",
                    layers=[])
    files, tables, proof = {}, {}, {}
    for identity, parts in plans:
        report = _write_table(output, identity, parts, device, chunk_rows, progress)
        tensor = identity["tensor"]
        files.update(report["files"])
        tables[tensor] = report["table"]
        proof[tensor] = report["proof"]
        manifest["layers"].append(report["manifest"])
    mc_manifest.validate_engram_fp8(manifest, strict=True)
    return dict(manifest=manifest, files=files, tables=tables, proof=proof)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    if source.name != SOURCE_REVISION:
        raise ValueError("require the pinned Hy4 source snapshot")
    mapping = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    sources = {}
    for layer in TABLE_ROWS:
        tensor = TENSOR_TEMPLATE.format(layer)
        filename = mapping[tensor]
        if Path(filename).name != filename:
            raise ValueError("invalid Engram source shard basename")
        sources[tensor] = source / filename

    def progress(tensor, processed, total):
        print(f"{datetime.now(timezone.utc).isoformat()} {tensor}: verified {processed:,}/{total:,} rows", flush=True)

    result = export_tables(args.output, sources, REFERENCE_REVISION, progress=progress)
    atomic_json(args.receipt, result)


if __name__ == "__main__":
    main()
