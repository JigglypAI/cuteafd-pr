"""Opt-in GPU oracle for native compact+wire fusion (run in a sealed slot)."""
from pathlib import Path
import ctypes as C
import os
import sys

import pytest


@pytest.fixture(scope="module")
def reply_programs():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("native reply oracle requires CUDA")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/lib"))
    import _pinned_sparkinfer  # noqa: F401
    from native_reply import compile_native_reply_aot
    from b12x.moe._shared.kernels.w4a16.expert_reply import compile_expert_reply_aot
    from b12x._lib.runtime_control import kernel_resolution_guard
    native = C.CDLL(os.environ["CUTEAFD_NATIVE_LIB"])
    native.cuteafd_set_expert_hidden.argtypes = [C.c_uint32]
    native.cuteafd_set_expert_hidden.restype = C.c_int32
    functions = []
    for name in ("cuteafd_compact_tokens_bf16_async", "cuteafd_compact_routes_bf16_async",
                 "cuteafd_compact_bf16_routes_async"):
        function = getattr(native, name)
        function.argtypes = [C.c_void_p, C.c_void_p, C.c_uint32, C.c_void_p]
        function.restype = C.c_int32
        functions.append(function)
    return torch, compile_native_reply_aot, compile_expert_reply_aot, functions, native, kernel_resolution_guard


@pytest.mark.parametrize("width", [2048, 3072, 4096, 4608, 5120, 6144, 7168])
@pytest.mark.parametrize("format", [0, 1, 2])
@pytest.mark.parametrize("kind", [0, 1, 2])
def test_native_fused_reply_matches_legacy_bytes_and_graph(reply_programs, width, format, kind):
    torch, compile_native, compile_codec, functions, native, guard = reply_programs
    torch.manual_seed(width + kind + format)
    assert native.cuteafd_set_expert_hidden(width) == 0
    fused = compile_native(width, format)
    quantize = compile_codec(quantize=True)
    decode = compile_codec()
    topk = 6
    row_bytes = width * 2 if format == 0 else width + width // 32 if format == 1 else width // 2 + width // 16
    for rows in (1, 17, 65):
        shape = (rows, width) if kind == 0 else (rows, topk, width)
        source = torch.randn(shape, dtype=torch.float32, device="cuda")
        if kind == 2:
            source = source.bfloat16()
        source[..., :32] = -0.0
        source[..., 32:64] = 0.0
        source[..., 64:96] *= 1e-5
        # Large cancellation checks fixed route order and the sole BF16 boundary.
        if kind == 1:
            source[:, 0, 128:160] = 2.0 ** 24
            source[:, 1, 128:160] = 1.0
            source[:, 2, 128:160] = -(2.0 ** 24)
        legacy = torch.empty((rows, width), dtype=torch.bfloat16, device="cuda")
        expected = torch.empty((rows, row_bytes), dtype=torch.uint8, device="cuda")
        guarded = torch.full((rows * row_bytes + 512,), 0xA5, dtype=torch.uint8, device="cuda")
        output = guarded[256:-256].view(rows, row_bytes)
        restored = torch.empty_like(legacy)
        reference_restored = torch.empty_like(legacy)

        def reference():
            assert functions[kind](source.data_ptr(), legacy.data_ptr(), rows,
                                   torch.cuda.current_stream().cuda_stream) == 0
            if format == 0:
                expected.copy_(legacy.view(torch.uint8))
            else:
                quantize.launch(legacy, expected, None, scalars=(rows, width, format))

        with guard("native_reply_oracle"):
            reference()
            fused.launch(source, output, scalars=(rows, topk, kind))
            torch.cuda.synchronize()
            assert torch.equal(output, expected), (width, rows, kind, format, "eager")
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fused.launch(source, output, scalars=(rows, topk, kind))
            for factor in (0.5, -2.0, 0.0):
                source.mul_(factor)
                reference()
                guarded.fill_(0xA5)
                graph.replay()
                torch.cuda.synchronize()
                assert torch.equal(output, expected), (width, rows, kind, format, "replay")
                assert (guarded[:256] == 0xA5).all() and (guarded[-256:] == 0xA5).all()
                if format:
                    decode.launch(output, restored, None, scalars=(rows, width, format))
                    decode.launch(expected, reference_restored, None, scalars=(rows, width, format))
                    torch.cuda.synchronize()
                    assert torch.equal(restored.view(torch.int16), reference_restored.view(torch.int16))


def test_inkling_hidden_dispatch_reduce_compact_and_wire():
    torch = pytest.importorskip("torch")
    assert torch.cuda.is_available(), "run this oracle in a GPU slot"
    native = C.CDLL(os.environ["CUTEAFD_NATIVE_LIB"])
    native.cuteafd_set_expert_hidden.argtypes = [C.c_uint32]
    native.cuteafd_set_expert_hidden.restype = C.c_int32
    assert native.cuteafd_set_expert_hidden(4608) == 0
    ptr, u32 = C.c_void_p, C.c_uint32
    compact = native.cuteafd_compact_routes_bf16_async
    compact.argtypes = [ptr, ptr, u32, ptr]
    compact.restype = C.c_int32
    reduce = native.cuteafd_reduce_routes_async
    reduce.argtypes = [C.POINTER(ptr), ptr, ptr, u32, u32, u32, ptr]
    reduce.restype = C.c_int32
    initialize = native.cuteafd_exl3_wire_initialize
    initialize.argtypes = [C.POINTER(ptr)]
    initialize.restype = C.c_int32
    decode = native.cuteafd_exl3_wire_decode
    decode.argtypes = [ptr, ptr, C.c_uint64, ptr, C.c_uint64, u32, ptr]
    decode.restype = C.c_int32
    native.cuteafd_exl3_wire_destroy.argtypes = [ptr]
    handle = ptr()
    assert initialize(C.byref(handle)) == 0
    try:
        for rows in (1, 17, 65):
            routes = torch.arange(6, device="cuda", dtype=torch.float32).view(1, 6, 1).expand(rows, 6, 4608).contiguous()
            reduced = torch.empty((rows, 4608), device="cuda", dtype=torch.bfloat16)
            compacted = torch.empty_like(reduced)
            wire = torch.empty((rows, 4752), device="cuda", dtype=torch.uint8)
            wire[:, :4608] = 0x38  # E4M3 1.0
            wire[:, 4608:] = 127  # UE8M0 1.0
            restored = torch.empty_like(reduced)
            planes = (ptr * 4)(routes.data_ptr(), routes.data_ptr(), routes.data_ptr(), routes.data_ptr())
            def launch():
                stream = torch.cuda.current_stream().cuda_stream
                assert compact(routes.data_ptr(), compacted.data_ptr(), rows, stream) == 0
                assert reduce(planes, None, reduced.data_ptr(), rows, 4, 6, stream) == 0
                assert decode(handle, wire.data_ptr(), wire.numel(), restored.data_ptr(), restored.numel()*2, rows, stream) == 0

            launch()
            torch.cuda.synchronize()
            assert torch.equal(compacted, torch.full_like(compacted, 15))
            assert torch.equal(reduced, torch.full_like(reduced, 60))
            assert torch.equal(restored, torch.ones_like(restored))
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                launch()
            routes.fill_(2)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(compacted, torch.full_like(compacted, 12))
            assert torch.equal(reduced, torch.full_like(reduced, 48))
            assert torch.equal(restored, torch.ones_like(restored))
    finally:
        native.cuteafd_exl3_wire_destroy(handle)
