"""Full Hy4 host-input recovery witness using real K4 encode, serialization and replay."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hy4.run import Runner
from hy4_flash import run
from hy4_flash.storage import Routes, Stages


def test_host_inputs_match_resident_gpu_encode_and_replay(tmp_path, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("the K4 relocation witness requires CUDA")
    device = torch.device("cuda:0")
    width, tokens = 128, 2105
    generator = torch.Generator().manual_seed(1847)
    latent = torch.randn((tokens, width), generator=generator)
    # Overlap every token across two experts. Expert 0 spans two chunks,
    # expert 1 fits in one, and expert 2 needs near-route top-up.
    other = torch.where(torch.arange(tokens) % 5 < 3, 1, 2)
    ids = torch.stack((torch.zeros_like(other), other), dim=1)
    gamma = torch.rand((tokens, 2), generator=generator).softmax(dim=1)
    batch = dict(latent=latent, ids=ids, gamma=gamma, candidate_ids=(3 - other)[:, None].int(),
                 candidate_gamma=gamma[:, 1:].contiguous())
    weights = [tuple((torch.randn((width, width), generator=generator) / width**0.5).bfloat16()
                     for _ in range(3)) for _ in range(3)]
    layer = SimpleNamespace(index=9, prefix="model.layers.9.")
    progress = SimpleNamespace(base={}, event=lambda *args, **kwargs: None)
    runner = Runner.__new__(Runner)
    runner.device = device
    runner.model = SimpleNamespace(config=SimpleNamespace(swiglu_limit=8., moe_intermediate_size=width))
    runner.reader = SimpleNamespace(expert_weights=lambda index, expert: weights[expert])
    runner.expert_witness = None
    runner.progress = progress
    monkeypatch.setattr(runner.recipe, "EXPERTS_PER_LAYER", 3)
    monkeypatch.setattr(run, "EXPERT_GROUP", 1)
    # One production-sized 2048-row chunk fits the device staging budget.
    monkeypatch.setattr(run, "EXPERT_INPUT_STAGE_BYTES", run.EXPERT_ROWS * width * 4)
    run.reference.configure_f32()

    original_encode = run.encode
    hessians = {}

    def capture(name, weight, raw, bits, device, metrics):
        hessians[name] = (raw["count"], raw["identity_rows"], raw["H"].cpu().view(torch.int32).clone())
        return original_encode(name, weight, raw, bits, device, metrics)

    monkeypatch.setattr(run, "encode", capture)
    with torch.inference_mode():
        results = []
        for host_inputs in (False, True):
            runner.host_expert_inputs = host_inputs
            bank_device = torch.device("cpu") if host_inputs else device
            spool = Routes([(0, batch)], experts=3, tokens=tokens, device=bank_device, recover=True)
            assert spool.latent.device == bank_device
            stage_root = tmp_path / ("host" if host_inputs else "device")
            runner.stages = Stages(stage_root, {"oracle": True})
            _, metrics, stages, combined = runner.experts("layer", layer, spool)
            assert len(metrics) == 9 and len(stages) == 3 and combined.device == device
            assert metrics[-1]["augmented_rows"] > 0
            packed = [(runner.stages.directory(s) / "segment-00000.bin").read_bytes() for s in stages]
            results.append((hessians.copy(), metrics, packed, combined.cpu().view(torch.int32)))
            hessians.clear()

        old, new = results
        assert len(old[0]) == 9 and old[0].keys() == new[0].keys()
        for name, (count, identity_rows, hessian) in old[0].items():
            assert new[0][name][:2] == (count, identity_rows), name
            assert torch.equal(hessian, new[0][name][2]), name
        assert old[1] == new[1]
        assert old[2] == new[2]
        assert torch.equal(old[3], new[3])

        # Resume a stage written by the resident path using the host bank.
        # A fresh store removes the producer's cached manifests from the test.
        runner.stages = Stages(tmp_path / "device", {"oracle": True})
        _, replay_metrics, _, replayed = runner.experts("layer", layer, spool)
        assert not hessians
        assert replay_metrics == old[1]
        assert replayed.device == device
        assert torch.equal(replayed.cpu().view(torch.int32), old[3])
