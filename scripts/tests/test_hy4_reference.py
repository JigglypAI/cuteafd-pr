"""Focused checks for independent Hy4 text arithmetic, IndexCache, and RoPE."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.container_torch
torch = pytest.importorskip("torch")
safetensors = pytest.importorskip("safetensors.torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python/reference"))
from families.hy4 import model, sources


def _tiny_config(index_topk=2, n_layers=2, mlp_layer_types=("dense", "sparse")):
    return SimpleNamespace(
        hidden_size=4,
        hc_mult=2,
        hc_eps=1e-6,
        hc_magnitude=2.0,
        rms_norm_eps=1e-5,
        num_attention_heads=1,
        qk_nope_head_dim=2,
        qk_rope_head_dim=2,
        v_head_dim=2,
        q_lora_rank=2,
        kv_lora_rank=2,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=index_topk,
        indexer_types=["full"] + ["shared"] * (n_layers - 1),
        layer_types=["deepseek_sparse_attention"] * n_layers,
        mlp_layer_types=list(mlp_layer_types),
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default", "mrope_interleaved": True},
        intermediate_size=4,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        moe_intermediate_size=2,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=2.827,
        swiglu_limit=10.0,
        vocab_size=8,
        num_hidden_layers=n_layers,
    )


def _tiny_attn_weights(c, *, has_indexer=True):
    torch.manual_seed(0)
    h, q, kv = c.hidden_size, c.q_lora_rank, c.kv_lora_rank
    nh, nope, rope, v = c.num_attention_heads, c.qk_nope_head_dim, c.qk_rope_head_dim, c.v_head_dim
    w = {
        "self_attn.q_a_proj.weight": torch.randn(q, h) * 0.2,
        "self_attn.q_a_layernorm.weight": torch.ones(q),
        "self_attn.q_b_proj.weight": torch.randn(nh * (nope + rope), q) * 0.2,
        "self_attn.kv_a_proj_with_mqa.weight": torch.randn(kv + rope, h) * 0.2,
        "self_attn.kv_a_layernorm.weight": torch.ones(kv),
        "self_attn.kv_b_proj.weight": torch.randn(nh * (nope + v), kv) * 0.2,
        "self_attn.linear_gate.weight": torch.randn(nh * v, h) * 0.2,
        "self_attn.o_proj.weight": torch.randn(h, nh * v) * 0.2,
        "self_attn.learnable_sink_param": torch.zeros(nh),
    }
    if has_indexer:
        d, n = c.index_head_dim, c.index_n_heads
        w.update({
            "self_attn.indexer.wq_b.weight": torch.randn(n * d, q) * 0.2,
            "self_attn.indexer.wk.weight": torch.randn(d, h) * 0.2,
            "self_attn.indexer.k_norm.weight": torch.ones(d),
            "self_attn.indexer.k_norm.bias": torch.zeros(d),
            "self_attn.indexer.weights_proj.weight": torch.randn(n, h) * 0.2,
        })
    return w


def test_dsa_indexer_and_indexcache_are_prefix_identical():
    c = _tiny_config(index_topk=2, n_layers=2)
    w0 = _tiny_attn_weights(c, has_indexer=True)
    w1 = _tiny_attn_weights(c, has_indexer=False)
    torch.manual_seed(7)
    x = torch.randn(6, c.hidden_size)
    out0_full, topk_full = model.attention(x, w0, c, 0)
    out1_full, topk1_full = model.attention(out0_full, w1, c, 1, prev_topk_indices=topk_full)
    assert topk_full is not None and topk_full.shape == (6, 2)
    assert torch.equal(topk1_full, topk_full)

    out0_pref, topk_pref = model.attention(x[:4], w0, c, 0)
    out1_pref, _ = model.attention(out0_pref, w1, c, 1, prev_topk_indices=topk_pref)
    assert torch.equal(topk_pref, topk_full[:4])
    assert torch.equal(out0_pref, out0_full[:4])
    assert torch.equal(out1_pref, out1_full[:4])


def test_interleaved_and_split_half_rope_differ_on_4d_rope():
    pos = torch.arange(5)
    torch.manual_seed(11)
    x = torch.randn(5, 2, 4)
    c1, s1 = model.phase(pos, 4, 10000.0)
    c2, s2 = model.phase_split_half(pos, 4, 10000.0)
    r1 = model.rotate(x, c1, s1)
    r2 = model.rotate_split_half(x, c2, s2)
    assert not torch.allclose(r1, r2)
    # Position 0 is identity for both conventions.
    torch.testing.assert_close(r1[0], x[0])
    torch.testing.assert_close(r2[0], x[0])


def test_routed_experts_swiglu_clamp_applies_only_to_routed_experts():
    c = _tiny_config(n_layers=2)
    h, ne, i = c.hidden_size, c.n_routed_experts, c.moe_intermediate_size
    w = {
        "mlp.gate.weight": torch.zeros(ne, h),
        "mlp.gate.e_score_correction_bias": torch.tensor([10.0, 9.0, 0.0, 0.0]),
        "mlp.experts.gate_up_proj": torch.ones(ne, 2 * i, h, dtype=torch.bfloat16) * 20.0,
        "mlp.experts.down_proj": torch.ones(ne, h, i, dtype=torch.bfloat16),
        "mlp.shared_experts.gate_proj.weight": torch.zeros(i, h),
        "mlp.shared_experts.up_proj.weight": torch.zeros(i, h),
        "mlp.shared_experts.down_proj.weight": torch.zeros(h, i),
    }
    x = torch.ones(1, h)
    out = model.feed_forward(x, w, c, 1)
    # gate and up are 80.0 before clamp, both clamped to swiglu_limit=10.0;
    # sum over 2 intermediate dims * 2.827 routed scaling factor.
    expected_val = float(torch.nn.functional.silu(torch.tensor(10.0)) * 10.0) * i * c.routed_scaling_factor
    torch.testing.assert_close(out, torch.full((1, h), expected_val), rtol=1e-4, atol=1e-4)


def test_tensor_contract_matches_full_and_shared_indexer_layers(tmp_path):
    c = _tiny_config(n_layers=2)
    full_keys = model.tensor_contract(c, 0)
    shared_keys = model.tensor_contract(c, 1)
    assert "model.layers.0.self_attn.indexer.k_norm.bias" in full_keys
    assert "model.layers.0.mlp.gate_proj.weight" in full_keys
    assert not any("indexer" in k for k in shared_keys)
    assert "model.layers.1.mlp.experts.gate_up_proj" in shared_keys
    assert sources.CHECKPOINT == "tencent/Hy4"

    name = "model.layers.1.mlp.experts.down_proj"
    path = tmp_path / "shard.safetensors"
    tensor = torch.zeros((c.n_routed_experts, c.hidden_size, c.moe_intermediate_size), dtype=torch.bfloat16)
    safetensors.save_file({name: tensor}, path)
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {name: path.name}}))
    ckpt = model.Hy4Checkpoint(tmp_path)
    ckpt.check({name: shared_keys[name]})
    assert torch.equal(ckpt.load(name, native=True), tensor)
    assert torch.equal(ckpt.load(name), tensor.float())
    with pytest.raises(ValueError, match="shape"):
        ckpt.check({name: ([1, 1, 1], "BF16")})
