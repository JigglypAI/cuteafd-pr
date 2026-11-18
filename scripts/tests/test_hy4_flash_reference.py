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


def test_fp8_codes_and_scale_match_full_roundtrip_at_bf16_boundaries():
    # Choose BF16 values on both sides of every positive E4M3FN midpoint,
    # spanning both sides of the row scale floor and the BF16 exponent range.
    fp8 = torch.float8_e4m3fn
    levels = torch.arange(127, dtype=torch.uint8).view(fp8).float()
    midpoints = (levels[:-1] + levels[1:]) / 2
    exponents = torch.arange(-133, 128, dtype=torch.float32)
    peaks = (torch.exp2(exponents[:, None]) * torch.tensor([1., 1.0078125, 1.5, 1.9921875])).bfloat16().float().flatten()
    scale = peaks[:, None].clamp_min(1e-12) / 448.
    boundary = (scale * midpoints).bfloat16().view(torch.int16).int()
    around = (boundary[..., None] + torch.arange(-2, 3)).clamp(0, 0x7f7f)
    values = around.to(torch.int16).view(torch.bfloat16).float().flatten(1)
    values = torch.minimum(values, peaks[:, None])
    values = torch.cat((values, -values), dim=1)
    values = torch.nn.functional.pad(values, (0, -values.shape[1] % 95))
    rows = torch.cat((peaks[:, None, None].expand(-1, values.shape[1] // 95, -1),
                      values.reshape(len(peaks), -1, 95)), dim=-1).reshape(-1, 96)
    scale = rows.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 448.
    codes = (rows / scale).clamp(-448., 448.).to(fp8).view(torch.uint8)
    codes = torch.where(codes == 0x80, 0, codes)
    decoded = codes.view(fp8).float() * scale
    assert torch.equal(decoded.view(torch.int32), model.fp8_roundtrip(rows).view(torch.int32))
