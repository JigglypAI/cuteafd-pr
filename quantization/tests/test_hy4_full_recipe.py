"""Full Hy4 reviewed geometry and rolling BF16 frontier checks."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hy4 import recipe
from hy4_flash.storage import Stages


def test_full_hy4_core_and_fused_expert_geometry():
    core = recipe.reviewed_core_modules()
    assert len(core) == 748
    assert sum(name.endswith(".kv_b_proj") for name in core) == 78 + 5
    assert sum(g["group"] == "mtp" for g in core.values()) == 45
    assert all(g["codebook"] == "mul1" for g in core.values())
    assert recipe.ENCODE_COUNT == 60_632
    assert set(g["bits"] for g in recipe.reviewed_expert_modules(77).values()) == {4}
    assert not recipe.reviewed_expert_modules(0)
    prefix = "model.layers.1.mlp.experts."
    assert recipe.source_tensor(prefix + "255.up_proj") == (prefix + "gate_up_proj", 255, (2048, 4096))
    assert recipe.source_tensor(prefix + "255.down_proj") == (prefix + "down_proj", 255, None)
    with pytest.raises(ValueError):
        recipe.source_tensor(prefix + "256.gate_proj")
    assert core["model.layers.77.self_attn.kv_b_proj"]["out_features"] == 28672
    assert core["model.dfly_layers.4.mlp.up_proj"]["out_features"] == 18432


def test_selective_bf16_residual_read_is_bitwise_exact(tmp_path):
    store = Stages(tmp_path / "stages", {"test": "hy4"})
    residual = torch.arange(48).reshape(4, 4, 3).to(torch.bfloat16)
    store.write_activations("layer-boundary", lambda start: [(0, {"state": residual, "extra": torch.ones(1)})], 1)
    restored = store.fields("layer-boundary", 0, ("state",))["state"]
    assert restored.dtype == torch.bfloat16
    assert torch.equal(restored.view(torch.int16), residual.view(torch.int16))
