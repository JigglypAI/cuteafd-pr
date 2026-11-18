"""CPU checks for Hy4 causal encodes and bounded sequential recovery."""
from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hy4_flash import run
from hy4_flash.forward import Layer
from hy4_flash.storage import Routes, Stages
from kimi_k3.uniform_k2.storage import assemble_shard, read_header


def test_sealed_segment_supports_selective_replay_and_direct_assembly(tmp_path):
    store = Stages(tmp_path / "phases", {"test": True})
    packed = torch.arange(24, dtype=torch.int16).reshape(3, 8)
    first = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    second = first + 100
    manifest = store.write_batches("moe", [
        (0, dict(packed=packed, weighted=first, empty=torch.empty((2, 0), dtype=torch.int32))),
        (1, dict(packed=packed + 10, weighted=second)),
    ])
    assert manifest["entries"][0]["filename"] == manifest["entries"][1]["filename"]
    assert manifest["entries"][0]["hash_algorithm"] == "xxh3-128"

    reloaded = Stages(store.root, {"test": True})
    assert torch.equal(reloaded.fields("moe", 1, ("weighted",))["weighted"], second)
    assert reloaded.fields("moe", 0, ("empty",))["empty"].shape == (2, 0)
    refs = {f"p{n}": reloaded.reference("moe", n, "packed") for n in range(2)}
    target = tmp_path / "final.safetensors"
    assemble_shard(refs, target)
    from safetensors.torch import load_file
    observed = load_file(str(target))
    assert torch.equal(observed["p0"], packed)
    assert torch.equal(observed["p1"], packed + 10)
    with pytest.raises(ValueError, match="invalid safetensors record extent"):
        read_header(refs["p1"][0], offset=refs["p1"][2], length=2)


def test_prepared_but_uncommitted_segment_is_replayed(tmp_path):
    store = Stages(tmp_path, {"test": True})
    def interrupted():
        yield 0, {"value": torch.ones(2)}
        raise RuntimeError("preempted")

    with pytest.raises(RuntimeError, match="preempted"):
        store.write_batches("partial", interrupted(), segment_batches=1)
    assert store.load("partial") is None
    landed = store.write_batches("partial", [(0, {"value": torch.zeros(2)})], segment_batches=1)
    assert len(landed["entries"]) == 1
    assert torch.equal(store.read_batch("partial", 0)["value"], torch.zeros(2))


def test_stable_expert_dispatch_preserves_reference_rounding_order():
    batch = dict(latent=torch.tensor([[10.0], [20.0], [30.0]]),
                 ids=torch.tensor([[2, 1, 0], [0, 1, 2], [1, 2, 0]]),
                 gamma=torch.tensor([[.5, .3, .2], [.3, .2, .5], [.2, .5, .3]]))
    routes = Routes([(0, batch)], experts=3, tokens=3)
    for expert in range(3):
        rows, weights = routes.expert(expert)
        assert rows.tolist() == [0, 1, 2]
        assert torch.equal(weights, batch["gamma"][batch["ids"] == expert])
    outputs = [torch.full((3, 1), value) for value in (1e20, -1e20, 3.)]
    assert torch.equal(routes.combined(enumerate(outputs)), outputs[-1])
    with pytest.raises(ValueError, match="out of order"):
        routes.combined([(1, outputs[1])])


def test_both_encodes_receive_same_hessian_and_only_serialized_k8_replays(tmp_path, monkeypatch):
    name = "layer.gate_proj"
    inputs = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    source = torch.zeros(2, 2)
    runner = run.Runner.__new__(run.Runner)
    runner.device = torch.device("cpu")
    runner.core = {name: dict(in_features=2, out_features=2)}
    runner.stages = Stages(tmp_path / "stages", {"test": True})
    runner.progress = SimpleNamespace(event=lambda *args, **kwargs: None)
    attempts, reloads = [], []

    def quantize(weight, raw, *, bits, device):
        assert weight is source
        attempts.append((bits, raw))
        return {suffix: torch.full((2, 2), bits) for suffix in run.recipe.SUFFIXES}, {}, {"bits": bits}

    def reconstruct(packed, shape, *, device, dtype):
        reloads.append(dtype)
        return packed["trellis"].to(dtype)

    numerical = SimpleNamespace(quantize_projection=quantize, reconstruct_projection=reconstruct,
                                covariance_accumulator=run.numerical.covariance_accumulator)
    monkeypatch.setattr(run, "numerical", numerical)
    layer = Layer(0, "layer.", {"gate_proj.weight": source})
    overrides = {}
    fit = runner.core_fit("phase", layer, [(name,)], lambda: [(inputs,)], overrides)
    assert [bits for bits, _ in attempts] == [8, 6]
    assert attempts[0][1] is attempts[1][1]
    assert torch.equal(attempts[0][1]["H"], inputs.T @ inputs)
    assert torch.equal(overrides[name], torch.full((2, 2), 8, dtype=torch.float32))
    assert reloads == [torch.bfloat16]

    def should_not_run():
        raise AssertionError("already committed Hessian was replayed")

    runner.core_fit(fit, layer, [(name,)], should_not_run, overrides)
    assert len(attempts) == 2
