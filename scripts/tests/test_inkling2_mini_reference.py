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
