"""CUDA witness for the full Hy4 held-out expert workspace."""
from pathlib import Path
import sys

import pytest
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hy4.forward import native_projection
from hy4_flash import gates
from hy4_flash.forward import reference
from hy4_flash.storage import Routes


def original_add(witness, expert, matrices, *, names=None, native=None, metrics=None):
    """Original equations and temporary lifetimes at a51d6bb62."""
    rows, gamma = witness.routes.expert(expert)
    rows = rows.to(witness.combined.device)
    inputs = witness.routes.latent.index_select(0, rows)
    check = witness.layer in witness.target_gates and metrics is not None
    selection = "heldout-natural-routes"
    if not len(rows):
        if not check:
            return
        inputs = witness.routes.latent[-128:]
        selection = "heldout-latent-tail-no-natural-routes"
    gate, up, down = matrices
    gated = witness.projection("", inputs, gate).clamp_max(witness.limit)
    lifted = witness.projection("", inputs, up).clamp(-witness.limit, witness.limit)
    active = F.silu(gated) * lifted
    if len(rows):
        output = witness.projection("", active, down)
        witness.combined.index_copy_(0, rows, witness.combined.index_select(0, rows)
                                    + output * gamma.to(output.device)[:, None])
    if check:
        down_inputs = active * gamma.to(active.device)[:, None] if len(rows) else active
        for row, name, xs, weight, restored in zip(metrics[-3:], names, (inputs, inputs, down_inputs),
                                                   native, matrices, strict=True):
            assert row["module"] == name
            gates.evaluated(row, xs, weight, restored, selection, projection=witness.projection)
            row["heldout_natural_rows"] = len(rows)
            assert row["heldout_passed"]


def witness_inputs(rows, width, hidden, device):
    rng = torch.Generator().manual_seed(701)
    latent = torch.randn((rows, width), generator=rng)
    # Two experts contribute to every row, one expert has zero routes.
    ids = torch.arange(2).repeat(rows, 1)
    ids[1::2] = ids[1::2].flip(1)
    gamma = torch.rand((rows, 2), generator=rng).softmax(dim=1)
    routes = Routes([(0, dict(latent=latent, ids=ids, gamma=gamma))], 3, rows, device=device)
    native = tuple((torch.randn((out, inp), generator=rng) / inp**0.5).to(device)
                   for out, inp in ((hidden, width), (hidden, width), (width, hidden)))
    restored = tuple(weight.bfloat16().float() for weight in native)
    return routes, native, restored


def test_heldout_workspace_matches_original():
    if not torch.cuda.is_available():
        pytest.skip("the held-out workspace witness requires CUDA")
    reference.configure_f32()
    device = torch.device("cuda:0")
    names = ("gate", "up", "down")

    with torch.inference_mode():
        routes, native, restored = witness_inputs(257, 256, 128, device)
        for gated in (False, True):
            results = []
            for add in (original_add, gates.ExpertWitness.add):
                gemms = 0

                def project(name, inputs, weight):
                    nonlocal gemms
                    gemms += 1
                    return native_projection(name, inputs, weight)

                witness = gates.ExpertWitness(routes, 9, 1.75, target_gates={9} if gated else set(),
                                              projection=project)
                provenance = []
                for expert in range(3):
                    metrics = [dict(module=name, bits=4) for name in names]
                    add(witness, expert, restored, names=names, native=native, metrics=metrics)
                    if gated:
                        assert all(m["heldout_passed"] and m["heldout_nmse"] > 0. for m in metrics)
                        assert all(m["heldout_natural_rows"] == (0 if expert == 2 else 257) for m in metrics)
                    provenance.append(metrics)
                results.append((witness.combined.cpu().view(torch.int32), provenance, gemms))
            assert torch.equal(results[0][0], results[1][0])
            assert results[0][1:] == results[1][1:]

        # Time one warmed expert at full Hy4 matrix widths, without gate scoring.
        # The allocator peak records extra live bytes beyond the persistent buffers.
        rows, width, hidden = 2048, 6144, 2048
        routes, _, matrices = witness_inputs(rows, width, hidden, device)
        witness = gates.ExpertWitness(routes, 9, 1.75, target_gates=set(), projection=native_projection)
        for label, add in (("original", original_add), ("bounded", gates.ExpertWitness.add)):
            add(witness, 0, matrices)
            witness.combined.zero_()
            torch.cuda.synchronize(device)
            baseline = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            add(witness, 0, matrices)
            end.record()
            end.synchronize()
            extra = torch.cuda.max_memory_allocated(device) - baseline
            print(f"heldout {label} rows={rows} width={width} intermediate={hidden} "
                  f"elapsed_ms={start.elapsed_time(end):.3f} extra_allocated_bytes={extra}")
            witness.combined.zero_()
