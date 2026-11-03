"""Qualify generic drafter BF16 projection row invariance and warm timing.

Run inside the coordinator dev container with a task-owned native library:
  python native/tests/draft_bf16_projection_selftest.py LIB [--shape K,N ...]
Each family contributes its context FC, QKV, context KV, O, gate/up and down.
"""
import argparse
import ctypes as ct
import json

import torch


def shapes():
    # hidden, query heads, KV heads, head width, MLP width, tap count
    families = {
        "dspark-qwen": (5120, 40, 8, 128, 10240, 5),
        "dspark-kimi": (7168, 64, 16, 64, 14336, 5),
        "dflash2-glm": (6144, 64, 8, 128, 12288, 6),
        "dflash2-glmf": (4096, 32, 8, 128, 12288, 5),
        "dflash-qwen27": (5120, 32, 8, 128, 17408, 5),
        "dflash-qwen35": (2048, 32, 8, 128, 6144, 8),
        "dflash-mimo-pro": (6144, 128, 8, 128, 16384, 5),
        "dflash-mimo-flash": (4096, 64, 8, 128, 16384, 5),
    }
    result = {}
    for family, (h, heads, kv, d, inter, taps) in families.items():
        for k, n in ((h * taps, h), (h, (heads + 2 * kv) * d),
                     (h, 2 * kv * d), (heads * d, h), (h, 2 * inter), (inter, h)):
            result.setdefault((k, n), []).append(family)
    for h in (2048, 4096, 5120, 6144):
        result.setdefault((2 * h, h), []).append("mimo/qwen-mtp")
    result.setdefault((30720, 6144), []).append("ml4-context-fc")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library")
    parser.add_argument("--shape", action="append", help="restrict to K,N; repeatable")
    args = parser.parse_args()
    library = ct.CDLL(args.library)
    launch = library.cuteafd_cuda_linear_bf16_cublas_async
    launch.argtypes = [ct.c_void_p] * 4 + [ct.c_size_t] * 3 + [ct.c_void_p]
    launch.restype = ct.c_int
    inventory = shapes()
    if args.shape:
        requested = {tuple(map(int, value.split(","))) for value in args.shape}
        inventory = {shape: inventory.get(shape, ["explicit"]) for shape in sorted(requested)}
    stream = torch.cuda.current_stream().cuda_stream
    torch.manual_seed(1045)
    for (k, n), families in sorted(inventory.items()):
        x = (torch.randn(2048, k, device="cuda") * .1).bfloat16()
        # Repeat one row at tile-boundary and mid-tile offsets; the oracle is n=1.
        offsets = (0, 1, 17, 63, 64, 65, 127, 139, 511, 1023, 2047)
        row = x[0].clone()
        for offset in offsets:
            x[offset].copy_(row)
        w = (torch.randn(n, k, device="cuda") * .02).bfloat16()
        outputs = {}

        def run(input_, output):
            status = launch(input_.data_ptr(), w.data_ptr(), None, output.data_ptr(),
                            input_.shape[0], k, n, stream)
            assert status == 0, (k, n, input_.shape[0], status)

        for rows in (1, 16, 140, 2048):
            out = torch.empty((rows, n), dtype=torch.bfloat16, device="cuda")
            run(x[:rows], out)
            torch.cuda.synchronize()
            outputs[rows] = out
        assert torch.equal(outputs[140], outputs[2048][:140]), (k, n, "n140/n2048")
        for rows, out in outputs.items():
            for offset in offsets:
                if offset < rows:
                    assert torch.equal(outputs[1][0], out[offset]), (k, n, rows, offset)
        # First shape use occurs eagerly; capture/replay may not allocate or reselect.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run(x[:140], outputs[140])
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(outputs[140], outputs[2048][:140]), (k, n, "graph")
        # Compare against the default algorithm through torch's BF16 GEMM.
        timing = {}
        for rows in (1, 16, 140, 2048):
            out = outputs[rows]
            arms = {"default": lambda: torch.mm(x[:rows], w.t(), out=out),
                    "candidate": lambda: run(x[:rows], out)}
            for arm, operation in arms.items():
                operation()
                measured = torch.cuda.CUDAGraph()
                with torch.cuda.graph(measured):
                    operation()
                for _ in range(5):
                    measured.replay()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(30):
                    measured.replay()
                end.record()
                end.synchronize()
                timing[f"{arm}_n{rows}_us"] = start.elapsed_time(end) * 1000 / 30
        print(json.dumps(dict(k=k, n=n, families=families, row_invariant=True,
                              offsets=offsets, timing=timing)), flush=True)
    print(f"PASS: {len(inventory)} projection shapes, n1/n16/n140/n2048, offsets and graphs", flush=True)


if __name__ == "__main__":
    main()
