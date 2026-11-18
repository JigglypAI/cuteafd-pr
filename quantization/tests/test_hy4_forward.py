"""Bounded CPU witnesses for the Hy4 stage boundaries."""
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from hy4_flash.forward import AttentionTap, Layer, Model, native_projection, recalled_context, reference


def test_staged_attention_matches_independent_f32_reference():
    generator = torch.Generator().manual_seed(13)

    def rand(*shape):
        return torch.randn(*shape, generator=generator)

    config = SimpleNamespace(
        attention=lambda _: (2, 4, 4, 3),
        layer_types=[reference.SPARSE], qk_rope_head_dim=4, kv_lora_rank=6,
        rms_norm_eps=1e-5, sliding_window=4,
        rope_parameters={reference.SPARSE: {"rope_theta": 10000000}},
        compressed_indexer_compress_ratio=2, compressed_indexer_group_topk=3,
        compressed_indexer_candidate_topk=6, compressed_indexer_num_attention_heads=2,
        compressed_indexer_head_dim=6,
    )
    weights = {f"self_attn.{name}.weight": rand(*shape) for name, shape in {
        "q_a_proj": (6, 8), "q_b_proj": (16, 6), "q_a_layernorm": (6,),
        "kv_a_proj_with_mqa": (10, 8), "kv_b_proj": (14, 6), "kv_a_layernorm": (6,),
        "linear_gate": (6, 8), "o_proj": (8, 6), "post_proj_norm": (8,),
        "compressed_indexer.token_key_projection": (6, 8),
        "compressed_indexer.token_key_norm": (6,),
        "compressed_indexer.query_projection": (12, 6),
        "compressed_indexer.key_value_projection": (6, 6),
        "compressed_indexer.key_gate_projection": (6, 6),
        "compressed_indexer.key_norm": (6,),
        "compressed_indexer.head_weight_projection": (2, 8),
    }.items()}
    weights["self_attn.learnable_sink_param"] = rand(2)
    weights["self_attn.compressed_indexer.intra_group_ape"] = rand(2, 6)
    adapter = Model(SimpleNamespace(config=config))
    layer = Layer(0, "model.layers.0.", weights)
    inputs = rand(23, 8)
    captured = []

    def capture(name, rows, matrix):
        captured.append((name.removeprefix(layer.prefix), rows.clone()))
        return native_projection(name, rows, matrix)

    actual, tap = adapter.attention(inputs, layer, capture, capture_tap=True)
    expected = reference.attention(inputs, weights, config, 0)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    assert [name for name, _ in captured] == ["self_attn.q_a_proj", "self_attn.kv_a_proj_with_mqa",
        "self_attn.linear_gate", "self_attn.q_b_proj", "self_attn.kv_b_proj", "self_attn.o_proj"]
    assert tap.groups.shape == (23, 3)
    assert torch.equal(captured[4][1], tap.kv)
    assert torch.equal(captured[0][1], inputs)
    # A reconstructed q_a projection must change the inputs captured from q_b.
    def reconstruction(name, rows, matrix):
        output = capture(name, rows, matrix)
        return output + 0.2 if name.endswith(".q_a_proj") else output

    adapter.attention(inputs, layer, reconstruction)
    assert not torch.equal(captured[3][1], captured[9][1])


def test_dfly_context_uses_last_committed_recall_without_duplication():
    kv = torch.zeros((7, 6))
    rope_key = torch.zeros((7, 4))
    groups = torch.tensor([[0, -1, -1], [0, -1, -1], [0, 1, -1], [0, 1, -1],
                           [1, 2, -1], [1, 2, -1], [2, 1, 3]], dtype=torch.int32)
    assert recalled_context(AttentionTap(kv, rope_key, groups), 7, 2).tolist() == [4, 5, 2, 3, 6]
    assert recalled_context(AttentionTap(kv, rope_key, groups), 3, 2).tolist() == [0, 1, 2]
    dense = groups[:, :0]
    assert recalled_context(AttentionTap(kv, rope_key, dense), 5, 2).tolist() == [0, 1, 2, 3, 4]
