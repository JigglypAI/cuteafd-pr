"""CPU oracles for the Hy4 Engram alternate. No real table payloads."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hy4_flash import engram
from kimi_k3.uniform_k2.storage import read_header


def bf16_reference(x):
    # Stock vendor expression, independent of the alternate codec.
    info = torch.finfo(torch.float8_e4m3fn)
    scale = x.detach().abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / info.max
    decoded = (x / scale).clamp(-info.max, info.max).to(torch.float8_e4m3fn).to(x.dtype) * scale
    return x + (decoded - x).detach()


@pytest.fixture(autouse=True)
def synthetic_reference(monkeypatch):
    monkeypatch.setattr(engram, "vendor_round_trip", lambda: bf16_reference)


def sample_rows(rows):
    x = torch.randn(rows, 96, generator=torch.Generator().manual_seed(14)).to(torch.bfloat16)
    # Include signed zeros at both all-zero and nonzero scales, and a row
    # whose absolute maximum is below the pre-divide scale floor.
    x[0, :] = -0.0
    x[1, 0:3] = torch.tensor([-0.0, -1e-20, 1e-20])
    x[2, :] *= 1e-17
    return x


def test_witness_zero_normalization_and_exact_bf16_expression():
    rows = sample_rows(8)
    packed = engram._checked_pack(rows, "test", 0)
    result = engram.decode_rows(packed)
    assert torch.equal(result.view(torch.int16), bf16_reference(rows).view(torch.int16))
    assert torch.count_nonzero(packed[:, :96] == 0x80) == 0
    assert torch.count_nonzero(packed[0, :96]) == 0
    assert not torch.signbit(result[0]).any()
    codes, scale = engram._codes_and_scale(rows)
    stored_scale = packed[:, 96:].contiguous().view(torch.float32)
    assert torch.equal(scale.float(), stored_scale)
    assert torch.equal(packed[:, :96] & 0x7F, codes.view(torch.uint8) & 0x7F)


def test_streamed_shards_preserve_source_and_resume(tmp_path, monkeypatch):
    # Small tables with different row counts, nonzero source spans, and
    # multiple under-cap parts. Normal production uses pinned full geometry.
    counts = {1: 13, 11: 9}
    monkeypatch.setattr(engram, "TABLE_ROWS", counts)
    sources, payloads = {}, {}
    for layer, rows in counts.items():
        name = engram.TENSOR_TEMPLATE.format(layer)
        source = tmp_path / f"source-{layer}.safetensors"
        native = sample_rows(rows).view(torch.uint8).numpy().tobytes()
        # Make the table appear after an unrelated source tensor.
        offset = len(b"abc")
        header = {"extra": dict(dtype="U8", shape=[offset], data_offsets=[0, offset]),
                  name: dict(dtype="BF16", shape=[rows, 96], data_offsets=[offset, offset + len(native)])}
        encoded = json.dumps(header).encode()
        source.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"abc" + native)
        sources[name] = source
        payloads[name] = native
    output = tmp_path / "output"
    revision = engram.REFERENCE_REVISION
    record = engram.export_tables(output, sources, revision, chunk_rows=2, max_part_bytes=900)
    for layer in record["manifest"]["layers"]:
        name = layer["tensor"]
        rows = counts[engram.mc_manifest.engram_layer(name)]
        assert layer["sha256"] == hashlib.sha256(payloads[name]).hexdigest()
        bf16 = output / record["tables"][name]
        header, offset = read_header(bf16)
        assert set(header) == {name, "__metadata__"}
        assert header[name]["dtype"] == "BF16"
        assert bf16.read_bytes()[offset:] == payloads[name]
        start, parts = 0, []
        assert len(layer["parts"]) > 1
        for part in layer["parts"]:
            path = output / part["file"]
            header, offset = read_header(path)
            assert header["__metadata__"]["engram_layout"] == engram.mc_manifest.ENGRAM_LAYOUT
            assert header["__metadata__"]["row_start"] == str(start)
            assert path.stat().st_size < 900
            assert hashlib.sha256(path.read_bytes()).hexdigest() == part["sha256"]
            assert header[part["tensor"]]["shape"] == [part["rows"], 100]
            raw = torch.frombuffer(bytearray(path.read_bytes()[offset:]), dtype=torch.uint8).reshape(-1, 100)
            parts.append(engram.decode_rows(raw))
            start += part["rows"]
        assert start == rows
        decoded = torch.cat(parts).view(torch.int16)
        assert torch.equal(decoded, bf16_reference(sample_rows(rows)).view(torch.int16))
        assert record["proof"][name]["reference_dtype"] == "BF16"
        assert record["proof"][name]["mismatch_count"] == 0

    def forbid_pack(*args):
        pytest.fail("a committed table was converted again")

    monkeypatch.setattr(engram, "_checked_pack", forbid_pack)
    again = engram.export_tables(output, sources, revision, chunk_rows=2, max_part_bytes=900)
    assert again == record
    with pytest.raises(ValueError, match="recovery identity changed"):
        engram.export_tables(output, sources, revision, max_part_bytes=850)


def test_export_bit_check_refuses_nonzero_code_corruption(monkeypatch):
    real_pack = engram.pack_rows

    def corrupt(values):
        packed = real_pack(values)
        packed[1, 0] = 0x7E
        return packed

    monkeypatch.setattr(engram, "pack_rows", corrupt)
    with pytest.raises(ValueError, match=r"1 rows in chunk, first 51"):
        engram._checked_pack(sample_rows(5), "test.table", 50)
