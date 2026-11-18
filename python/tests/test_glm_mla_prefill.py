"""GLM/GLMF native prefill against the fork's MG BF16 and exact-record FP32.

Run in the CUDA dev image with CUTEAFD_NATIVE_LIB pointing at a candidate build.
As a script, pass --output to record warmed SM row curves in addition to pytest's oracle.
"""

import argparse
import ctypes
import json
import math
import os
from pathlib import Path
import statistics
import sys

import pytest

FORK = Path(__file__).resolve().parents[2] / "third_party" / "sparkinfer"
ROWS = (64, 1536, 2048, 2688, 2752, 4096)  # 2688 is the tail of 8192 at a 2752 unit.
NATIVE_ARMS = {"native-fp8": 1, "native-bf16": 2, "exact-f16": 3}
ARMS = (*NATIVE_ARMS, "mg-bf16")


def run_case(fmt, library, timing=False):
    import torch

    sys.path.insert(0, str(FORK))
    from benchmarks.bench_mg_inline_precision import INLINE, arm_traits
    from b12x.attention._shared.mla.prefill import _mg_head_partitions
    from b12x.attention._shared.mla.prefill_mg import run_unified_prefill_mg
    from b12x.attention._shared.mla.reference import (
        pack_mla_kv_cache_reference, unpack_mla_kv_cache_reference,
    )
    from b12x.attention._shared.mla.smem_mg import max_mg_head_groups

    heads, qdim, topk, scale = INLINE[fmt]
    traits = arm_traits(fmt, "bf16")
    partitions = _mg_head_partitions(heads, 16, max_mg_head_groups(traits))
    native = ctypes.CDLL(str(library)).cuteafd_glm_mla_prefill_lse
    p = ctypes.c_void_p
    native.argtypes = [p] * 5 + [ctypes.c_int] * 4 + [ctypes.c_float, ctypes.c_int, p, p]
    native.restype = ctypes.c_int
    dev = torch.device("cuda")
    rng = torch.Generator(device=dev).manual_seed(41512)
    pool = 8192
    width = max(ROWS) if timing else 65
    raw = (torch.randn((pool, 576), device=dev, generator=rng)
           * torch.linspace(.3, 1.2, 576, device=dev)).bfloat16()
    packed = pack_mla_kv_cache_reference(raw[:, :512], raw[:, 512:] if qdim == 576 else None)
    keys = unpack_mla_kv_cache_reference(packed).squeeze(1)
    rec = packed.shape[-1]
    cache = packed.reshape(pool // 64, -1)
    row_ids = torch.arange(width, device=dev)[:, None]
    idx = (1 + (row_ids * 37 + torch.arange(topk, device=dev)[None] * 17) % (pool - 1)).int()
    full = torch.full((width,), topk, device=dev, dtype=torch.int32)

    def alloc(rows):
        return (torch.full((rows, heads, 512), float("nan"), device=dev, dtype=torch.bfloat16),
                torch.full((rows, heads), float("nan"), device=dev))

    def launch(arm, q, indices, lengths, out, lse):
        if arm == "mg-bf16":
            for groups, active, offset in partitions:
                run_unified_prefill_mg(
                    q=q, kv_cache=cache, topk_indices=indices, topk_length=lengths,
                    sm_scale=scale, page_block_size=64, output=out, lse_out=lse,
                    mg_n_hg=groups, active_heads=active, head_offset=offset, traits_override=traits,
                )
        else:
            ptrs = [t.data_ptr() for t in (q, packed, indices, lengths, out)]
            rc = native(*ptrs, q.shape[0], heads, topk, rec, scale * math.log2(math.e),
                        NATIVE_ARMS[arm], torch.cuda.current_stream().cuda_stream,
                        lse.data_ptr() if lse is not None else None)
            assert rc == 0, (fmt, arm, rc)

    def query(rows, pattern):
        q = torch.randn((rows, heads, qdim), device=dev, generator=rng)
        q *= {"spread": .03, "medium": 1.6, "peaked": .2}[pattern]
        if pattern == "peaked":
            target = keys[idx[:rows, 0].long()]
            q += (target * (14 / scale / target.square().sum(-1, keepdim=True)))[:, None]
        return q.bfloat16()

    def reference(q, indices, lengths):
        # Only oracle rows are retained. At 4096 rows avoid materializing a giant K/V gather.
        out, lse = [], []
        for qb, ib, lb in zip(q.split(4), indices.split(4), lengths.split(4)):
            k = keys[ib.clamp_min(0).long()]
            valid = (torch.arange(topk, device=dev)[None] < lb[:, None]) & (ib >= 0)
            scores = (torch.bmm(qb.float(), k.transpose(1, 2)) * scale).masked_fill(~valid[:, None], -math.inf)
            probs = scores.softmax(-1).masked_fill(~valid[:, None], 0)
            lse.append(scores.logsumexp(-1))
            out.append(torch.bmm(probs, k[..., :512]))
        return torch.cat(out), torch.cat(lse)

    def compare(pattern, q, indices, lengths, arms=ARMS):
        expected, expected_lse = reference(q, indices, lengths)
        report = {}
        live = expected_lse.isfinite()
        for arm in arms:
            out, lse = alloc(len(q))
            launch(arm, q, indices, lengths, out, lse)
            assert torch.isfinite(out).all().item(), (fmt, pattern, arm, "nonfinite output")
            assert ((out[~live] == 0).all() & torch.isneginf(lse[~live]).all()).item(), (fmt, pattern, arm, "empty")
            got_lse = lse * math.log(2) if arm == "mg-bf16" else lse
            diff = out.float() - expected
            rms = (diff.norm() / expected.norm()).item()
            lse_max = (got_lse[live] - expected_lse[live]).abs().max().item()
            # E4M3 math approximates scores; only BF16 exact QK has a strict LSE bound.
            assert rms < (0.075 if arm == "native-fp8" else 0.006), (fmt, pattern, arm, rms)
            assert math.isfinite(lse_max) and lse_max < (2 if arm == "native-fp8" else 3e-4), (fmt, pattern, arm, lse_max)
            report[arm] = {"relative_rms": rms, "max_abs": diff.abs().max().item(), "lse_nats_max_abs": lse_max}
            if pattern == "masked":
                # Also check LSE bytes; numeric equalities can miss changes in float bit patterns.
                saved = out.clone(), lse.clone()
                clean = packed[0].clone()
                packed[0].fill_(255)
                try:
                    launch(arm, q, indices, lengths, out, lse)
                    assert torch.equal(saved[0].view(torch.int16), out.view(torch.int16)), (fmt, arm, "slot poison output")
                    assert torch.equal(saved[1].view(torch.int32), lse.view(torch.int32)), (fmt, arm, "slot poison LSE")
                finally:
                    packed[0].copy_(clean)
                one_out, one_lse = alloc(1)
                launch(arm, q[-1:], indices[-1:], lengths[-1:], one_out, one_lse)
                assert torch.equal(saved[0][-1:].view(torch.int16), one_out.view(torch.int16)), (fmt, arm, "row output")
                assert torch.equal(saved[1][-1:].view(torch.int32), one_lse.view(torch.int32)), (fmt, arm, "row LSE")
        return report

    scores = {pattern: compare(pattern, query(24, pattern), idx[:24], full[:24])
              for pattern in ("spread", "medium", "peaked")}
    edges = [0, 1, 2, 31, 32, 33, 63, 64, 65, 127, 128, 129, 777, topk - 65, topk - 1, topk]
    mask = idx[:len(edges)].clone()
    mask[:, 5::11] = -1
    scores["masked"] = compare("masked", query(len(edges), "spread"), mask,
                               torch.tensor(edges, device=dev, dtype=torch.int32))
    # MG's empty-row contract is a zero length. Exercise the native kernels'
    # nonzero-length/all-negative row independently.
    mask[1].fill_(-1)
    scores["all_negative_native"] = compare("all_negative", query(3, "spread"), mask[:3],
                                            torch.tensor(edges[:3], device=dev, dtype=torch.int32), NATIVE_ARMS)
    result = {"device": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
              "heads": heads, "topk": topk, "oracle": scores, "curve": {}}
    if timing:
        q = query(width, "medium")
        for rows in ROWS:
            outputs = {arm: alloc(rows) for arm in ARMS}
            graphs = {}
            for arm in ARMS:
                out, lse = outputs[arm]
                args = arm, q[:rows], idx[:rows], full[:rows], out, lse if arm == "mg-bf16" else None
                for _ in range(3):
                    launch(*args)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    launch(*args)
                graphs[arm] = graph
            samples = {arm: [] for arm in ARMS}
            for sample in range(5):
                for arm in ARMS[::1 if sample % 2 == 0 else -1]:
                    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(10):
                        graphs[arm].replay()
                    stop.record()
                    stop.synchronize()
                    samples[arm].append(start.elapsed_time(stop) * 1000 / 10 / rows)
            result["curve"][rows] = {arm: {"median_us_per_row": statistics.median(values), "samples": values}
                                     for arm, values in samples.items()}
    return result


def run_oracle(fmt, library, timing=False):
    import torch

    # IEEE FP32, including when a caller has enabled high/medium matmul precision.
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        return run_case(fmt, library, timing)
    finally:
        torch.set_float32_matmul_precision(previous)


@pytest.mark.parametrize("fmt", ("glm", "glmf"))
def test_glm_mla_prefill(fmt):
    torch = pytest.importorskip("torch")
    library = os.environ.get("CUTEAFD_NATIVE_LIB")
    if not library or not torch.cuda.is_available():
        pytest.skip("needs a CUDA device and CUTEAFD_NATIVE_LIB")
    print(json.dumps(run_oracle(fmt, library)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--oracle-only", action="store_true")
    args = parser.parse_args()
    receipt = {}
    for model in ("glm", "glmf"):
        receipt[model] = run_oracle(model, args.library, not args.oracle_only)
        args.output.write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps({model: receipt[model]}), flush=True)
