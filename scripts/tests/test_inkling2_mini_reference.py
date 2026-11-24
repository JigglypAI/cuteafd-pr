"""Numerical witnesses for Mini's unique loop and NVFP4 arithmetic."""
from pathlib import Path
import sys

import pytest

pytestmark = pytest.mark.container_torch
torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python/reference"))
from families.inkling2_mini import model
from test_inkling2_reference import hand_attention, tiny_config


def test_loop_cache_slots_and_message_query_order():
    root = tiny_config()
    root["model_type"] = "inkling2_loop"
    root["text_config"].update(model_type="inkling2_loop_text", num_hidden_layers=24,
                               n_layers_per_attn_res_block=6,
                               loop_config=dict(core_start=6, core_end=18, num_passes=3,
                                                gate_rank=64, is_varloop=False))
    cfg = model.Config.from_dict(root)
    assert [cfg.cache_slot(i, 0) for i in range(24)] == list(range(24))
    assert [cfg.cache_slot(i, 1) for i in range(6, 18)] == list(range(24, 36))
    assert [cfg.cache_slot(i, 2) for i in range(6, 18)] == list(range(36, 48))
    with pytest.raises(ValueError, match="invalid.*visit"):
        cfg.cache_slot(18, 1)
    a = torch.arange(4)[:, None]
    assert model.message_queries(a, a + 10).squeeze(-1).tolist() == [[0, 10, 11, 1], [2, 12, 13, 3]]


def test_controller_uses_normed_inputs_but_blends_raw_values():
    x = torch.tensor([[3., 4.]])
    message = torch.tensor([[-2., 0.]])
    layer = {"controllers.attn.down_weight": torch.tensor([[0., 1., 1., 0.]]),
             "controllers.attn.forget_weight": torch.tensor([[.5], [-.25]]),
             "controllers.attn.gain_weight": torch.tensor([[.3], [.7]])}
    eps = 1e-6
    rank = 4 / (12.5 + eps) ** .5 - 2 / (2 + eps) ** .5
    expected = ((1 - (rank * layer["controllers.attn.forget_weight"].T).tanh()) * x
                + (rank * layer["controllers.attn.gain_weight"].T).tanh() * message)
    torch.testing.assert_close(model.controller(x, message, layer, "attn", eps), expected)


def test_controller_narrows_rank_projection_before_f32_forget_and_gain():
    x, message = torch.tensor([[1.11, -.16]]), torch.tensor([[-1.79, .42]])
    down = torch.tensor([[.79, -.28, .16, .43]], dtype=torch.bfloat16)
    forget = torch.tensor([[.71], [-.39]], dtype=torch.bfloat16)
    gain = torch.tensor([[-.23], [.84]], dtype=torch.bfloat16)
    layer = {f"controllers.attn.{name}_weight": w for name, w in (
        ("down", down), ("forget", forget), ("gain", gain))}
    rms = lambda y: y * (y.square().mean(-1, keepdim=True) + 1e-6).rsqrt()
    paired = torch.cat((rms(x), rms(message)), -1)
    rank = (paired.bfloat16() @ down.T).float()
    expected = (1 - (rank @ forget.float().T).tanh()) * x + (rank @ gain.float().T).tanh() * message
    assert torch.equal(model.controller(x, message, layer, "attn", 1e-6), expected)


def test_bf16_attention_joint_f32_sink_softmax_and_f32_gate():
    q = torch.tensor([[[.75, 1.25]]], dtype=torch.bfloat16)
    k = torch.tensor([[[.58, -.62]]], dtype=torch.bfloat16)
    v = torch.tensor([[[1.42, -2.31]]], dtype=torch.bfloat16)
    rel = torch.tensor([[[.22]]], dtype=torch.bfloat16)
    gate = torch.tensor([[[.43, -.73]]], dtype=torch.bfloat16)
    sink = torch.tensor([[.81]], dtype=torch.bfloat16)
    score = (q.squeeze(1) @ k.squeeze(1).T) * (2 ** -.5) + rel.reshape(1, 1)
    attended = torch.cat((score, sink), -1).float().softmax(-1)[:, :1].bfloat16() @ v.squeeze(1)
    expected = (attended.float() * gate.squeeze(1).float().sigmoid()).bfloat16().unsqueeze(1)
    assert torch.equal(model.relative_attention(q, k, v, rel, gate, sink, None), expected)


def test_nvfp4_decodes_low_nibble_first_and_scales_per_expert():
    packed = torch.tensor([[0x91, 0xF7, 0xB5, 0xA6, 0, 0, 0, 0]], dtype=torch.uint8)
    scale = torch.tensor([[2]], dtype=torch.float8_e4m3fn)
    vals = model.dequant_nvfp4(packed, scale, torch.tensor(3.), torch.tensor([4.]))
    assert vals.tolist() == [[.75, -.75, 9, -9, 4.5, -2.25, 6, -1.5] + [0.] * 8]


def test_nvfp4_fouroversix_chooses_smallest_block_error_and_per_token_scale():
    # amax 6 makes the two scales 256 and 384. The first row is exact on
    # the max/4 grid; the second row is exact on the max/6 grid.
    six = torch.tensor([6., 3., -4., 1., .5, -1.5, 2., 0.] * 2)
    four = torch.tensor([6., 4.5, -3., 2.25, 1.5, -.75, 0., 3.] * 2)
    rows = torch.stack((four, six * 2, torch.zeros_like(six)))
    assert torch.equal(model.quantize_nvfp4(rows), rows)



def test_nvfp4_scale_division_does_not_round_via_reciprocal_multiply():
    block = torch.tensor([float.fromhex(v) for v in (
        "0x1.d74f54p+0", "0x1.3696c0p+2", "-0x1.6b615ap+1", "0x1.3fb5d8p+1",
        "0x1.1796acp+0", "0x1.3e88bap+0", "0x1.9a6f64p-1", "-0x1.fd129ap+0",
        "0x1.5b7c24p+2", "-0x1.a0fb5ep+3", "-0x1.131224p+2", "0x1.2f7a24p+2",
        "0x1.13015ep+1", "0x1.150cfcp+0", "-0x1.77c082p+2", "0x1.99cc62p+1",
    )])
    actual = model.quantize_nvfp4(block.unsqueeze(0))
    # Reciprocal-multiply instead yields 6.5153427 for this lane.
    assert actual[0, 8].item() == 4.34356164932251


@pytest.mark.parametrize("window", [None, 4])
def test_mini_attention_matches_scalar_and_preserves_prefix_rows(window):
    rng = torch.Generator().manual_seed(3)
    q, k, v = (torch.randn(9, heads, 2, generator=rng) for heads in (4, 2, 2))
    rel = torch.randn(9, 4, 3, generator=rng)
    gate = torch.randn(9, 4, 2, generator=rng)
    sink = torch.randn(9, 4, generator=rng)
    args = (q, k, v, rel, gate, sink)
    full = model.relative_attention(*args, window)
    torch.testing.assert_close(full, hand_attention(*args, window), atol=2e-7, rtol=2e-6)
    assert torch.equal(model.relative_attention(*(x[:7] for x in args), window), full[:7])
