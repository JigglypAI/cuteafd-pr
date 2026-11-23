"""Independent IEEE-F32 Hy4 text forward over original checkpoint tensors."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from shape_invariant import ROWS, linear
from families.hy4_common import (
    Checkpoint,
    EXPERTS,
    OUTER_NAMES,
    SPARSE,
    configure_f32,
    dense,
    head_features,
    hyper,
    inverse_rms,
    mla_projections,
    native_routes,
    norm,
    phase,
    phase_split_half,
    project,
    rotate,
    rotate_split_half,
    rotate_tail,
    routed_experts,
    routing,
    routing_scores,
    tiled_sink_attention,
)


@dataclass(frozen=True)
class Config:
    text: dict

    @classmethod
    def parse(cls, raw):
        c = raw["text_config"]
        if (raw["model_type"] != "hy_v4_vl" or c["model_type"] != "hy_v4"
                or raw.get("quantization_config") or c.get("quantization_config")
                or c["hidden_act"] != "silu" or c["attention_bias"]
                or not all(c[k] for k in ("enable_ihc", "gated_mla", "enable_lm_head_fp32", "learnable_sink"))
                or c.get("gating_type") != "elementwise"
                or raw.get("tie_word_embeddings") or c.get("tie_word_embeddings")):
            raise ValueError("unsupported Hy4 text config")
        for key, allowed in (
            ("layer_types", {SPARSE}),
            ("mlp_layer_types", {"dense", "sparse"}),
            ("indexer_types", {"full", "shared"}),
        ):
            if len(c[key]) != c["num_hidden_layers"] or set(c[key]) - allowed:
                raise ValueError(f"invalid Hy4 {key}")
        if c["indexer_types"][0] != "full":
            raise ValueError("Hy4 layer 0 must have indexer_types 'full'")
        rope = c["rope_parameters"]
        if rope.get("rope_type") != "default" or not rope.get("mrope_interleaved"):
            raise ValueError(f"unsupported Hy4 rope_parameters: {rope}")
        return cls(dict(c))

    def __getattr__(self, name):
        try:
            return self.text[name]
        except KeyError as error:
            raise AttributeError(name) from error


def tensor_contract(c, index=None):
    """Return the exact text tensor names, shapes and storage dtypes."""
    h, hc, q, kv = c.hidden_size, c.hc_mult, c.q_lora_rank, c.kv_lora_rank
    items = {}

    def add(name, *shape, dtype="BF16"):
        items[name] = (list(shape), dtype)

    if index is None:
        add(OUTER_NAMES[0], c.vocab_size, h)
        add(OUTER_NAMES[1], hc, hc * h, dtype="F32")
        add(OUTER_NAMES[2], hc, dtype="F32")
        add(OUTER_NAMES[3], 1, dtype="F32")
        add(OUTER_NAMES[4], h)
        add(OUTER_NAMES[5], c.vocab_size, h)
        return items

    nh, nope, rope, v = c.num_attention_heads, c.qk_nope_head_dim, c.qk_rope_head_dim, c.v_head_dim
    add("input_layernorm.weight", h)
    add("post_attention_layernorm.weight", h)
    for name in ("hc_attn_layer", "hc_mlp_layer"):
        add(f"{name}.hc_pre.hc_fn", 2 * hc, hc * h, dtype="F32")
        add(f"{name}.hc_pre.hc_base", 2 * hc, dtype="F32")
        add(f"{name}.hc_pre.hc_scale", 2, dtype="F32")
    for tag, dims in {
        "q_a_proj": (q, h), "q_b_proj": (nh * (nope + rope), q), "q_a_layernorm": (q,),
        "kv_a_proj_with_mqa": (kv + rope, h), "kv_b_proj": (nh * (nope + v), kv), "kv_a_layernorm": (kv,),
        "linear_gate": (nh * v, h), "o_proj": (h, nh * v),
    }.items():
        add(f"self_attn.{tag}.weight", *dims)
    add("self_attn.learnable_sink_param", nh, dtype="F32")

    if c.indexer_types[index] == "full":
        d, n = c.index_head_dim, c.index_n_heads
        p = "self_attn.indexer."
        add(p + "wq_b.weight", n * d, q)
        add(p + "wk.weight", d, h)
        add(p + "k_norm.weight", d)
        add(p + "k_norm.bias", d)
        add(p + "weights_proj.weight", n, h)

    def add_dense(prefix, width):
        for name in ("gate", "up"):
            add(f"{prefix}{name}_proj.weight", width, h)
        add(f"{prefix}down_proj.weight", h, width)

    if c.mlp_layer_types[index] == "dense":
        add_dense("mlp.", c.intermediate_size)
    else:
        ne, i = c.n_routed_experts, c.moe_intermediate_size
        add_dense("mlp.shared_experts.", i * c.n_shared_experts)
        add("mlp.gate.weight", ne, h)
        add("mlp.gate.e_score_correction_bias", ne, dtype="F32")
        add(EXPERTS[0], ne, 2 * i, h)
        add(EXPERTS[1], ne, h, i)

    return {f"model.layers.{index}.{name}": value for name, value in items.items()}


class Hy4Checkpoint(Checkpoint):
    def layer(self, config, index, *, stream_experts=False):
        p = f"model.layers.{index}."
        result = {}
        for name in tensor_contract(config, index):
            key = name.removeprefix(p)
            if stream_experts and key in EXPERTS:
                # Host mmap for short preflight/probe passes: fault only active experts.
                result[key] = self.raw(name)
            else:
                result[key] = self.load(name, native=key in EXPERTS)
        return result


def rope_helpers(rope_style):
    if rope_style == "interleaved":
        return phase, rotate
    if rope_style == "split_half":
        return phase_split_half, rotate_split_half
    raise ValueError(f"unsupported Hy4 rope_style: {rope_style}")


def dsa_indexer(x, q_resid, w, c, cos, sin, *, rotate_fn=rotate):
    """Fixed-geometry causal DSA top-k indexer matching HYV4Indexer in F32."""
    p = "self_attn.indexer."
    seq_len = len(x)
    heads, dim = c.index_n_heads, c.index_head_dim
    q = project(q_resid, w, p + "wq_b").reshape(seq_len, heads, dim)
    q = rotate_tail(q, cos, sin, rotate_fn=rotate_fn)
    k = F.layer_norm(
        project(x, w, p + "wk"), (dim,), w[p + "k_norm.weight"], w[p + "k_norm.bias"], c.rms_norm_eps
    )
    k = rotate_tail(k.unsqueeze(1), cos, sin, rotate_fn=rotate_fn).squeeze(1)
    weights = project(x, w, p + "weights_proj") * (heads ** -0.5) * (dim ** -0.5)
    topk = min(c.index_topk, seq_len)
    out = []
    for start in range(0, seq_len, ROWS):
        stop = min(start + ROWS, seq_len)
        width = start + ROWS
        q_tile = F.pad(q[start:stop], (0, 0, 0, 0, 0, ROWS - (stop - start)))
        w_tile = F.pad(weights[start:stop], (0, 0, 0, ROWS - (stop - start)))
        k_tile = F.pad(k[:stop], (0, 0, 0, width - stop))
        scores = (q_tile @ k_tile.T).relu()
        index_scores = (w_tile.unsqueeze(-2) @ scores).squeeze(-2)
        key_id = torch.arange(width, device=x.device)
        query_id = torch.arange(start, start + ROWS, device=x.device)
        allowed = (query_id[:, None] >= key_id) & (key_id < stop)
        index_scores = index_scores.masked_fill(~allowed, float("-inf"))
        tile_topk = min(topk, width)
        indices = index_scores[:stop - start].topk(tile_topk, dim=-1).indices.to(torch.int32)
        if tile_topk < topk:
            # Rows with start + ROWS < topk < seq_len can attend to every causal key
            # in [0, start + ROWS); pad unused slots with 0 (already causal).
            indices = F.pad(indices, (0, topk - tile_topk), value=0)
        out.append(indices)
    return torch.cat(out, dim=0)


def attention(x, w, c, index, prev_topk_indices=None, *, rope_style="interleaved", indexer_rope_style=None, qa_kva_eps=1e-6):
    p = "self_attn."
    h, nope, rope, vdim = c.num_attention_heads, c.qk_nope_head_dim, c.qk_rope_head_dim, c.v_head_dim
    base = c.rope_parameters["rope_theta"]
    pos = torch.arange(len(x), device=x.device)
    phase_fn, rotate_fn = rope_helpers(rope_style)
    cos, sin = phase_fn(pos, rope, base)
    q0, q, k, v, gate, sink = mla_projections(
        x, w, c, h, nope, rope, vdim, cos, sin, qa_kva_eps=qa_kva_eps, rotate_fn=rotate_fn
    )
    if c.indexer_types[index] == "full":
        if len(x) > c.index_topk:
            idx_style = rope_style if indexer_rope_style is None else indexer_rope_style
            idx_phase_fn, idx_rotate_fn = rope_helpers(idx_style)
            idx_cos, idx_sin = (cos, sin) if idx_style == rope_style else idx_phase_fn(pos, rope, base)
            topk_indices = dsa_indexer(x, q0, w, c, idx_cos, idx_sin, rotate_fn=idx_rotate_fn)
        else:
            topk_indices = None
    else:
        topk_indices = prev_topk_indices if len(x) > c.index_topk else None

    tile_mask = None
    if topk_indices is not None:
        def tile_mask(start, stop, first, width, query_id):
            slots = topk_indices[start:stop].long().clamp_max(width)
            selected = torch.zeros((ROWS, width + 1), dtype=torch.bool, device=x.device)
            selected[:stop - start].scatter_(1, slots, True)
            return selected[:, :width] | (query_id[:, None] < c.index_topk)

    out = tiled_sink_attention(q, k, v, gate, sink, (nope + rope) ** -0.5, tile_mask=tile_mask)
    return project(out, w, p + "o_proj"), topk_indices


def feed_forward(x, w, c, index):
    if c.mlp_layer_types[index] == "dense":
        return dense(x, w, "mlp.")
    ids, probs = routing(x, w, c)
    return routed_experts(x, ids, probs, w, c) + dense(x, w, "mlp.shared_experts.")


def layer_forward(x, w, c, index, prev_topk_indices=None, *, rope_style="interleaved", indexer_rope_style=None, qa_kva_eps=1e-6):
    collapsed, post = hyper(x, w, "hc_attn_layer", c)
    attn_out, topk_indices = attention(
        norm(collapsed, w["input_layernorm.weight"], c.rms_norm_eps),
        w, c, index, prev_topk_indices=prev_topk_indices,
        rope_style=rope_style, indexer_rope_style=indexer_rope_style, qa_kva_eps=qa_kva_eps,
    )
    x = post * attn_out.unsqueeze(1) + x
    collapsed, post = hyper(x, w, "hc_mlp_layer", c)
    ffn = feed_forward(norm(collapsed, w["post_attention_layernorm.weight"], c.rms_norm_eps), w, c, index)
    return post * ffn.unsqueeze(1) + x, topk_indices
