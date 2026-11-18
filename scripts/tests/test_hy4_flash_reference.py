"""Focused checks for independent Hy4 text arithmetic and native shard access."""
import json
from pathlib import Path
import sys

import pytest

pytestmark = pytest.mark.container_torch
torch = pytest.importorskip("torch")
safetensors = pytest.importorskip("safetensors.torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python/reference"))
from families.hy4_flash import model


def test_recall_uses_only_complete_groups_and_keeps_current():
    queries = torch.tensor([[[1.]], [[1.]], [[1.]]])
    keys = torch.tensor([[1.], [3.], [2.], [9.]])
    weights = torch.ones(3, 1)
    ids = model.recall_groups(queries, keys, weights, torch.tensor([2, 3, 4]), 3)
    assert ids.tolist() == [[1, 0, 2], [1, 2, 3], [3, 1, 4]]


def test_attention_sink_and_sliding_window_are_prefix_stable():
    from types import SimpleNamespace
    c = SimpleNamespace(attention=lambda _: (1, 1, 2, 1), layer_types=[model.SLIDING],
                        rope_parameters={model.SLIDING: {"rope_theta": 10000}}, rms_norm_eps=.01,
                        kv_lora_rank=1, sliding_window=2)
    weights = {f"self_attn.{k}.weight": torch.tensor(v) for k, v in {
        "q_a_proj": [[0.]], "q_a_layernorm": [1.], "q_b_proj": [[0.], [0.], [0.]],
        "kv_a_proj_with_mqa": [[1.], [0.], [0.]], "kv_a_layernorm": [1.],
        "kv_b_proj": [[0.], [1.]], "linear_gate": [[0.]], "o_proj": [[1.]], "post_proj_norm": [1.],
    }.items()}
    weights["self_attn.learnable_sink_param"] = torch.zeros(1)
    x = torch.tensor([[1.], [2.], [-3.], [-4.]])
    values = x / (x.square() + c.rms_norm_eps).sqrt()
    mean = torch.cat((values[:1] / 2, (values[1:] + values[:-1]) / 3)) / 2
    expected = mean / (mean.square() + c.rms_norm_eps).sqrt()
    result = model.attention(x, weights, c, 0)
    torch.testing.assert_close(result, expected)
    assert torch.equal(model.attention(x[:3], weights, c, 0), result[:3])


def test_fp8_read_roundtrips_rows_without_upcasting_the_table(tmp_path):
    name = "model.layers.1." + model.TABLE
    path = tmp_path / "model.safetensors"
    table = torch.arange(24, dtype=torch.bfloat16).reshape(6, 4)
    safetensors.save_file({name: table}, path)
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {name: path.name}}))
    ckpt = model.Checkpoint(tmp_path)
    ckpt.check({name: ([6, 4], "BF16")})
    raw = ckpt.raw(name)
    chosen = raw.index_select(0, torch.tensor([1, 5])).float()
    assert raw.dtype == torch.bfloat16 and torch.equal(raw, table)
    assert model.fp8_roundtrip(chosen).dtype == torch.float32
    assert not torch.equal(model.fp8_roundtrip(chosen), chosen)
    assert torch.equal(model.fp8_roundtrip(chosen[:1]), model.fp8_roundtrip(chosen)[:1])
    with pytest.raises(ValueError, match="shape"):
        ckpt.check({name: ([4, 6], "BF16")})
