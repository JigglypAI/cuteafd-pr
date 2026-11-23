"""Independent IEEE-F32 Hy4-flash text forward over original checkpoint tensors."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from shape_invariant import ROWS, linear
from families.hy4_common import (
    Checkpoint as _BaseCheckpoint,
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
    project,
    rotate,
    rotate_tail,
    routed_experts,
    routing,
    routing_scores,
    tiled_sink_attention,
)

TABLE = "engram_layer.multi_head_embedding.embedding.weight"
SLIDING = "sliding_attention"


@dataclass(frozen=True)
class Config:
    text: dict
    masked_tokens: tuple

    @classmethod
    def parse(cls, raw):
        c = raw["text_config"]
        if (raw["model_type"] != "hy_v4_flash" or raw.get("quantization_config")
                or c.get("quantization_config") or c["hidden_act"] != "silu" or c["attention_bias"]
                or not all(c[k] for k in ("enable_ihc", "use_mla", "gated_mla", "enable_lm_head_fp32",
                                          "moe_router_enable_expert_bias", "moe_router_use_sigmoid"))
                or c["compressed_indexer_key_pooling"] != "gated" or not c["compressed_indexer_skip_fine_select"]
                or c["dsa_skip_attn_post_norm"] or raw.get("tie_word_embeddings")):
            raise ValueError("unsupported Hy4-flash text config")
        for key, allowed in (("layer_types", {SPARSE, SLIDING}), ("mlp_layer_types", {"dense", "sparse"})):
            if len(c[key]) != c["num_hidden_layers"] or set(c[key]) - allowed:
                raise ValueError(f"invalid Hy4-flash {key}")
        for key, rope in c["rope_parameters"].items():
            if key not in (SPARSE, SLIDING) or rope["rope_type"] != "default" or not rope["mrope_interleaved"]:
                raise ValueError(f"unsupported Hy4-flash rope: {key}")
        c = dict(c, engram_lookup_table_size=c.get("engram_lookup_table_size", 120818))
        return cls(c, (raw["image_token_id"], raw["video_token_id"], c["engram_pad_id"]))

    def __getattr__(self, name):
        try:
            return self.text[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def attention(self, index):
        local = self.layer_types[index] == SLIDING
        return ((self.swa_num_attention_heads, self.swa_qk_nope_head_dim, self.swa_qk_rope_head_dim, self.swa_v_head_dim)
                if local else (self.num_attention_heads, self.qk_nope_head_dim, self.qk_rope_head_dim, self.v_head_dim))

    def vocab_sizes(self, index):
        return tuple(size for order in self.engram_layer_head_vocab_sizes[str(index + 1)] for size in order)


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

    nh, nope, rope, v = c.attention(index)
    add("input_layernorm.weight", h)
    add("post_attention_layernorm.weight", h)
    for name in ("hc_attn_layer", "hc_mlp_layer"):
        add(f"{name}.hc_pre.hc_fn", 2 * hc, hc * h, dtype="F32")
        add(f"{name}.hc_pre.hc_base", 2 * hc, dtype="F32")
        add(f"{name}.hc_pre.hc_scale", 2, dtype="F32")
    for tag, dims in {
        "q_a_proj": (q, h), "q_b_proj": (nh * (nope + rope), q), "q_a_layernorm": (q,),
        "kv_a_proj_with_mqa": (kv + rope, h), "kv_b_proj": (nh * (nope + v), kv), "kv_a_layernorm": (kv,),
        "linear_gate": (nh * v, h), "o_proj": (h, nh * v), "post_proj_norm": (h,),
    }.items():
        add(f"self_attn.{tag}.weight", *dims)
    add("self_attn.learnable_sink_param", nh, dtype="F32")
    if c.layer_types[index] == SPARSE:
        d, n = c.compressed_indexer_head_dim, c.compressed_indexer_num_attention_heads
        prefix = "self_attn.compressed_indexer."
        for tag, dims in {"query_projection": (n * d, q), "head_weight_projection": (n, h),
                          "token_key_projection": (d, h), "token_key_norm": (d,)}.items():
            add(f"{prefix}{tag}.weight", *dims)
        add(prefix + "key_norm.weight", d, dtype="F32")
        for tag in ("key_value_projection", "key_gate_projection"):
            add(f"{prefix}{tag}.weight", d, d, dtype="F32")
        add(f"{prefix}intra_group_ape", c.compressed_indexer_compress_ratio, d, dtype="F32")

    def add_dense(prefix, width):
        for name in ("gate", "up"):
            add(f"{prefix}{name}_proj.weight", width, h)
        add(f"{prefix}down_proj.weight", h, width)

    if c.mlp_layer_types[index] == "dense":
        add_dense("mlp.", c.intermediate_size)
        add("mlp.post_fc2_norm.weight", h)
    else:
        ne, i, latent = c.n_routed_experts, c.moe_intermediate_size, c.routed_expert_hidden_size
        add_dense("mlp.shared_experts.", i * c.n_shared_experts)
        add("mlp.gate.weight", ne, h)
        add("mlp.gate.e_score_correction_bias", ne, dtype="F32")
        add("mlp.routed_expert_down_proj.weight", latent, h)
        add("mlp.routed_expert_up_proj.weight", h, latent)
        add(EXPERTS[0], ne, 2 * i, latent)
        add(EXPERTS[1], ne, latent, i)
        add("mlp.combine_output_norm.weight", h)
    if index + 1 in c.engram_layer_ids:
        sizes = c.vocab_sizes(index)
        width = c.engram_n_embed_per_ngram // c.engram_n_head_per_ngram
        p = "engram_layer."
        add(TABLE, sum(sizes), width)
        add(p + "lookup_table", c.engram_lookup_table_size, dtype="I32")
        add(p + "unified_pre_gamma", h)
        add(p + "value_proj.weight", h, width * len(sizes))
        add(p + "value_proj.bias", h)
        for j in range(len(sizes)):
            add(f"{p}lookup_per_head_norms.{j}.weight", width)
        for j in range(hc):
            add(f"{p}key_projs.{j}.weight", h, width * len(sizes))
            add(f"{p}key_projs.{j}.bias", h)
            for name in ("norm_key", "norm_query"):
                add(f"{p}{name}.{j}.weight", h)
            add(f"{p}unified_per_hc_scales.{j}", h)
    return {f"model.layers.{index}.{name}": value for name, value in items.items()}


class Checkpoint(_BaseCheckpoint):
    def layer(self, config, index):
        p = f"model.layers.{index}."
        result = {}
        for name in tensor_contract(config, index):
            key = name.removeprefix(p)
            if key == TABLE:
                # Safetensors maps the table on the host; index_select touches just the requested rows.
                result[key] = self.raw(name)
            else:
                result[key] = self.load(name, native=key in EXPERTS or key == "engram_layer.lookup_table")
        return result


def index_projections(x, q_residual, w, c, pos, base):
    p = "self_attn.compressed_indexer."
    ratio = c.compressed_indexer_compress_ratio
    heads, dim = c.compressed_indexer_num_attention_heads, c.compressed_indexer_head_dim
    token_keys = norm(project(x, w, p + "token_key_projection"), w[p + "token_key_norm.weight"], c.rms_norm_eps)
    group_count = len(x) // ratio
    tokens = token_keys[:group_count * ratio].reshape(group_count, ratio, dim)
    values = project(tokens, w, p + "key_value_projection")
    gates = project(tokens, w, p + "key_gate_projection") + w[p + "intra_group_ape"]
    pooled = norm((values * gates.softmax(dim=1)).sum(dim=1), w[p + "key_norm.weight"], c.rms_norm_eps)
    keys = rotate_tail(pooled.unsqueeze(1), *phase(pos[:group_count * ratio:ratio], c.qk_rope_head_dim, base)).squeeze(1)
    queries = project(q_residual, w, p + "query_projection").reshape(len(x), heads, dim)
    queries = rotate_tail(queries, *phase(pos, c.qk_rope_head_dim, base))
    weights = project(x, w, p + "head_weight_projection") * (heads ** -.5 * dim ** -.5)
    return queries, keys, weights


def recall_groups(queries, keys, weights, groups, group_topk):
    scores = (queries @ keys.T).relu()
    scores = (weights.unsqueeze(-2) @ scores).squeeze(-2)
    ids = torch.arange(len(keys), device=keys.device)
    recalled = scores.masked_fill(ids >= groups[:, None], -torch.inf).topk(min(len(keys), group_topk - 1), dim=-1).indices
    return torch.cat((recalled, groups[:, None]), dim=-1)


def attention(x, w, c, index):
    p = "self_attn."
    h, nope, rope, vdim = c.attention(index)
    kind = c.layer_types[index]
    base = c.rope_parameters[kind]["rope_theta"]
    pos = torch.arange(len(x), device=x.device)
    cos, sin = phase(pos, rope, base)
    q0, q, k, v, gate, sink = mla_projections(x, w, c, h, nope, rope, vdim, cos, sin)
    indexed = kind == SPARSE and len(x) > c.compressed_indexer_candidate_topk
    tile_mask = None
    if indexed:
        index_q, index_k, index_w = index_projections(x, q0, w, c, pos, base)
        ratio = c.compressed_indexer_compress_ratio

        def tile_mask(start, stop, first, width, query_id):
            if stop <= c.compressed_indexer_candidate_topk:
                return torch.ones((ROWS, width), dtype=torch.bool, device=x.device)
            recalled = recall_groups(index_q[start:stop], index_k, index_w[start:stop],
                                     pos[start:stop] // ratio, c.compressed_indexer_group_topk)
            slots = (recalled[..., None] * ratio + torch.arange(ratio, device=x.device)).flatten(1)
            # The official path clips the incomplete final group to the last
            # physical token; future groups outside this tile go to a sentinel.
            slots = slots.clamp_max(len(x) - 1).clamp_max(width)
            selected = torch.zeros((ROWS, width + 1), dtype=torch.bool, device=x.device)
            selected[:stop - start].scatter_(1, slots, True)
            selected = selected[:, :width]
            # Short official calls skip the indexer; before the recall budget is full every prior group fits.
            return selected | (query_id[:, None] < c.compressed_indexer_candidate_topk)

    out = tiled_sink_attention(
        q, k, v, gate, sink, (nope + rope) ** -.5,
        sliding_window=c.sliding_window if kind == SLIDING else None,
        tile_mask=tile_mask,
    )
    out = project(out, w, p + "o_proj")
    return norm(out, w[p + "post_proj_norm.weight"], c.rms_norm_eps)


def feed_forward(x, w, c, index):
    if c.mlp_layer_types[index] == "dense":
        return norm(dense(x, w, "mlp."), w["mlp.post_fc2_norm.weight"], c.rms_norm_eps)
    ids, probs = routing(x, w, c)
    latent = project(x, w, "mlp.routed_expert_down_proj")
    routed = routed_experts(latent, ids, probs, w, c)
    result = project(routed, w, "mlp.routed_expert_up_proj") + dense(x, w, "mlp.shared_experts.")
    return norm(result, w["mlp.combine_output_norm.weight"], c.rms_norm_eps)


def engram_rows(tokens, lookup, c, index):
    pad = c.engram_pad_id
    window = torch.stack([tokens, *(F.pad(tokens, (j, 0), value=pad)[:len(tokens)]
                                    for j in range(1, c.engram_max_ngram_size))], dim=-1)
    compressed = lookup[torch.where(window < len(lookup), window, pad)].long()
    bound = max(1, np.iinfo(np.int64).max // (int(lookup.max()) + 1) // 2)
    rng = np.random.default_rng(c.engram_seed + 10007 * (index + 1))
    factors = torch.as_tensor(rng.integers(0, bound, size=c.engram_max_ngram_size, dtype=np.int64) * 2 + 1,
                              device=tokens.device)
    sizes = torch.tensor(c.vocab_sizes(index), device=tokens.device)
    offsets = sizes.cumsum(0) - sizes
    mixed = compressed * factors
    acc = mixed[:, 0]
    parts = []
    for order in range(1, c.engram_max_ngram_size):
        acc = acc ^ mixed[:, order]
        j = (order - 1) * c.engram_n_head_per_ngram
        parts.append(acc[:, None] % sizes[j:j + c.engram_n_head_per_ngram])
    return torch.cat(parts, dim=-1) + offsets


def fp8_roundtrip(x):
    cap = torch.finfo(torch.float8_e4m3fn).max
    scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / cap
    decoded = (x / scale).clamp(-cap, cap).to(torch.float8_e4m3fn).float() * scale
    return x + (decoded - x)


def engram(x, tokens, w, c, index, *, round_trip=fp8_roundtrip):
    p = "engram_layer."
    rows = engram_rows(tokens, w[p + "lookup_table"], c, index)
    table = w[TABLE]
    embeddings = table.index_select(0, rows.flatten().to(table.device)).to(x.device, dtype=torch.float32)
    embeddings = embeddings.reshape(*rows.shape, -1)
    if c.engram_fp8_embedding:
        embeddings = round_trip(embeddings)
    eps = torch.finfo(torch.float32).eps  # nn.RMSNorm defaults, not text_config.rms_norm_eps.
    embedding = torch.cat([norm(embeddings[:, j], w[f"{p}lookup_per_head_norms.{j}.weight"], eps)
                           for j in range(rows.shape[-1])], dim=-1)
    gates = []
    for j in range(c.hc_mult):
        key = norm(project(embedding, w, f"{p}key_projs.{j}", bias=True), w[f"{p}norm_key.{j}.weight"], eps)
        query = norm(x[:, j], w[f"{p}norm_query.{j}.weight"], eps)
        gate = (key * query).sum(dim=-1) * (c.hidden_size ** -.5)
        gates.append((gate.abs().clamp_min(1e-6).sqrt() * gate.sign()).sigmoid())
    value = project(embedding, w, p + "value_proj", bias=True)
    value = w[p + "unified_pre_gamma"] * (value * inverse_rms(value, eps))
    branches = torch.stack([w[f"{p}unified_per_hc_scales.{j}"] * value for j in range(c.hc_mult)], dim=1)
    output = torch.stack(gates, dim=1).unsqueeze(-1) * value.unsqueeze(1) + branches
    masked = torch.zeros_like(tokens, dtype=torch.bool)
    for token_id in c.masked_tokens:
        masked |= tokens == token_id
    return output.masked_fill(masked[:, None, None], 0.)


def layer_forward(x, tokens, w, c, index):
    if index + 1 in c.engram_layer_ids:
        x = x + engram(x, tokens, w, c, index)
    collapsed, post = hyper(x, w, "hc_attn_layer", c)
    x = post * attention(norm(collapsed, w["input_layernorm.weight"], c.rms_norm_eps), w, c, index).unsqueeze(1) + x
    collapsed, post = hyper(x, w, "hc_mlp_layer", c)
    ffn = feed_forward(norm(collapsed, w["post_attention_layernorm.weight"], c.rms_norm_eps), w, c, index)
    return post * ffn.unsqueeze(1) + x
