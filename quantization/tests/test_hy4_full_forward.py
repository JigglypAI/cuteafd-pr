"""The full family's token index cache, RoPE and BF16 residual boundary."""

from pathlib import Path
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hy4 import forward
from hy4.parity import adjacent_rope


def test_adjacent_tail_matches_pr_for_heads_and_shared_keys():
    pos = torch.tensor([0, 1, 13, 35])
    phase = forward.reference.phase(pos, 8, 1e7)
    pairs = torch.randn(4, 3, 14)
    hf_phase = [torch.cat((x[:, 0, ::2], x[:, 0, ::2]), -1)[None] for x in phase]
    expected, _ = adjacent_rope(pairs[None, :, :, -8:], pairs[None, :, :, -8:], *hf_phase, unsqueeze_dim=2)
    result = forward.Model.rotate_tail(pairs, *phase)
    torch.testing.assert_close(result[..., -8:], expected[0], atol=0, rtol=0)
    assert torch.equal(result[..., :-8], pairs[..., :-8])


def test_shared_selection_is_inherited_and_required_at_long_lengths():
    c = SimpleNamespace(index_topk=3, indexer_types=["full", "shared"])
    m = forward.Model(SimpleNamespace(config=c))
    x = torch.randn(8, 4)
    data = torch.randn(8, 2, 4)
    layer = forward.Layer(1, "", {})
    m.attention_expand = lambda *args: (data, data, data, data[:,0], torch.zeros(2))
    first = SimpleNamespace(kv=x, gate=torch.ones(8, 8))
    selected = torch.arange(3).expand(8, 3).int()
    import pytest
    with pytest.raises(ValueError, match="preceding"):
        m.attention_output_inputs(x, first, layer)
    a, t = m.attention_output_inputs(x, first, layer, previous_selection=selected, capture_tap=True)
    b, _ = m.attention_output_inputs(x, first, layer, previous_selection=selected + 1)
    assert t.groups is selected
    assert not torch.equal(a[-1], b[-1])
    state = torch.randn(8, 4, 8).bfloat16()
    post = torch.randn(8, 4, 1)
    assert torch.equal(m.apply_residual(state, post, a), (post * a[:,None] + state.float()).bfloat16())
