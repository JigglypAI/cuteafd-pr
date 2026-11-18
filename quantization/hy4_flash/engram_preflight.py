"""Audit the brief's Engram alternate against pinned Transformers arithmetic."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import os
from pathlib import Path

from kimi_k3.uniform_k2.storage import atomic_json
from .source import HeaderSource

MODEL = "transformers.models.hy_v4_flash.modeling_hy_v4_flash"
REFERENCE_SHA256 = "43341929e4bec215c224fc837b24b534b4de41954992673a0655a65060e38b9b"
REFERENCE_REVISION = "c2ec850510a37978ea83df226589623d5f2be776"


def evaluate(rows, round_trip):
    import torch

    def stored_decode(x):
        scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / 448
        codes = (x / scale).clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
        codes = torch.where(codes == 0x80, 0, codes).view(torch.float8_e4m3fn)
        return codes.float() * scale.float()

    def compare(actual, expected):
        bit_dtype = torch.int16 if expected.dtype == torch.bfloat16 else torch.int32
        different = actual.contiguous().view(bit_dtype) != expected.contiguous().view(bit_dtype)
        return dict(mismatch_values=int(different.sum()), mismatch_rows=int(different.any(-1).sum()),
                    max_abs_difference=float((actual.float() - expected.float()).abs().max()))

    if rows.dtype != torch.bfloat16 or rows.ndim != 2 or rows.shape[1] != 96:
        raise ValueError("expected source BF16 Engram rows of width 96")
    native = round_trip(rows)
    exported = stored_decode(rows.float())
    bf16_decode = stored_decode(rows)
    return dict(rows=len(rows), values=rows.numel(),
                brief_vs_official_fp32=compare(exported, round_trip(rows.float())),
                brief_vs_official_bf16=compare(exported.bfloat16(), native),
                bf16_alternate_with_cast=compare(bf16_decode.bfloat16(), native),
                bf16_alternate_plain_fp32=compare(bf16_decode, native.float()))


def audit(snapshot, rows_per_range=512):
    import torch

    module = importlib.import_module(MODEL)
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != REFERENCE_SHA256:
        raise ValueError("Engram arithmetic must come from the pinned Transformers helper")
    source = HeaderSource(snapshot)
    if source.specs["model.embed_tokens.weight"][0] != "BF16":
        raise ValueError("review the source embedding and hidden dtype")
    results, read_bytes = {}, 0
    for name, (dtype, shape) in source.specs.items():
        if not name.endswith(".multi_head_embedding.embedding.weight"):
            continue
        rows, width = shape
        if dtype != "BF16" or width != 96 or not 1 <= rows_per_range <= rows // 4:
            raise ValueError("invalid Engram table or sample geometry")
        shard = source.weight_map[name]
        start = source.shard_offsets[shard] + source.raw_headers[name]["data_offsets"][0]
        positions = (0, rows // 4, rows // 2, rows - rows_per_range)
        size = rows_per_range * width * 2
        with (source.snapshot / shard).open("rb", buffering=0) as stream:
            chunks = [os.pread(stream.fileno(), size, start + row * width * 2) for row in positions]
        if any(len(chunk) != size for chunk in chunks):
            raise ValueError("truncated Engram sample")
        raw = bytearray(b"".join(chunks))
        read_bytes += len(raw)
        values = torch.frombuffer(raw, dtype=torch.bfloat16).reshape(-1, width)
        results[name] = dict(audit_ranges=[dict(start=row, rows=rows_per_range) for row in positions],
                             **evaluate(values, module._fp8_round_trip))
    if len(results) != 4:
        raise ValueError("incomplete Engram witness")
    passed = all(not row["brief_vs_official_bf16"]["mismatch_values"]
                 and not row["brief_vs_official_fp32"]["mismatch_values"] for row in results.values())
    return dict(status="passed-on-sample" if passed else "blocked", time=datetime.now(timezone.utc).isoformat(),
                source_revision=source.snapshot.name, reference_revision=REFERENCE_REVISION,
                source_payload_read_bytes=read_bytes, torch_version=torch.__version__, tables=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    report = audit(args.source)
    atomic_json(args.receipt, report)
    if report["status"] == "blocked":
        raise SystemExit("Engram FP32 alternate failed official BF16 parity. See the preflight receipt.")


if __name__ == "__main__":
    main()
