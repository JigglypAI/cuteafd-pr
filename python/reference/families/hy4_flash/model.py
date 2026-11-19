"""Independent IEEE-F32 Hy4-flash text forward over original checkpoint tensors."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import struct

import numpy as np
import torch
import torch.nn.functional as F

from shape_invariant import ROWS, linear

OUTER_NAMES = ("model.embed_tokens.weight", "model.hc_head.hc_head_fn", "model.hc_head.hc_head_base",
               "model.hc_head.hc_head_scale", "model.norm.weight", "lm_head.weight")
EXPERTS = ("mlp.experts.gate_up_proj", "mlp.experts.down_proj")
TABLE = "engram_layer.multi_head_embedding.embedding.weight"
SPARSE = "deepseek_sparse_attention"
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

    def dense(prefix, width):
        for name in ("gate", "up"):
            add(f"{prefix}{name}_proj.weight", width, h)
        add(f"{prefix}down_proj.weight", h, width)

    if c.mlp_layer_types[index] == "dense":
        dense("mlp.", c.intermediate_size)
        add("mlp.post_fc2_norm.weight", h)
    else:
        ne, i, latent = c.n_routed_experts, c.moe_intermediate_size, c.routed_expert_hidden_size
        dense("mlp.shared_experts.", i * c.n_shared_experts)
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


class Checkpoint:
    def __init__(self, snapshot, device="cpu"):
        self.snapshot = Path(snapshot)
        self.device = torch.device(device)
        self.index = json.loads((self.snapshot / "model.safetensors.index.json").read_bytes())["weight_map"]
        self.headers, self.maps = {}, {}
        self.read_bytes, self.read_seconds = 0, 0.

    def descriptor(self, name):
        shard = self.index.get(name)
        if not isinstance(shard, str) or Path(shard).name != shard or not shard.endswith(".safetensors"):
            raise ValueError(f"missing tensor or invalid Hy4 shard path: {name}")
        if shard not in self.headers:
            path = self.snapshot / shard
            with path.open("rb") as stream:
                prefix = stream.read(8)
                size, = struct.unpack("<Q", prefix)
                if not 2 <= size <= 64 * 2**20:
                    raise ValueError(f"invalid Hy4 safetensors header size: {shard}")
                header = json.loads(stream.read(size))
            if max(entry["data_offsets"][1] for key, entry in header.items() if key != "__metadata__") + 8 + size != path.stat().st_size:
                raise ValueError(f"incomplete Hy4 shard: {shard}")
            self.headers[shard] = header
        return self.headers[shard][name]

    def check(self, contract):
        for name, (shape, dtype) in contract.items():
            d = self.descriptor(name)
            first, last = d["data_offsets"]
            if d["shape"] != shape or d["dtype"] != dtype or first < 0 or last - first != math.prod(shape) * (2 if dtype == "BF16" else 4):
                raise ValueError(f"wrong Hy4 tensor shape, dtype or extent: {name}: {d}")

    def raw(self, name):
        from safetensors import safe_open
        self.descriptor(name)
        shard = self.index[name]
        if shard not in self.maps:
            self.maps[shard] = safe_open(str(self.snapshot / shard), framework="pt", device="cpu")
        return self.maps[shard].get_tensor(name)

    def load(self, name, *, native=False):
        import time
        tick = time.monotonic()
        value = self.raw(name)
        self.read_bytes += value.numel() * value.element_size()
        value = value.to(self.device, dtype=value.dtype if native else torch.float32, copy=True)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.read_seconds += time.monotonic() - tick
        return value

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

    def close(self):
        self.maps.clear()


def inverse_rms(x, eps):
    return torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)


def norm(x, w, eps):
    return (x * inverse_rms(x, eps)) * w


def project(x, w, name, bias=False):
    return linear(x, w[name + ".weight"], w[name + ".bias"] if bias else None)


def hyper(x, weights, name, c):
    p = name + ".hc_pre.hc_"
    flat = x.flatten(-2)
    pre, post = (linear(flat, weights[p + "fn"]) * inverse_rms(flat, c.rms_norm_eps)).chunk(2, dim=-1)
    bp, bo = weights[p + "base"].chunk(2, dim=-1)
    sp, so = weights[p + "scale"].unbind(0)
    pre = (pre * sp + bp).sigmoid() + c.hc_eps
    post = c.hc_magnitude * (post * so + bo).sigmoid() + c.hc_eps
    return (x * pre.unsqueeze(-1)).sum(-2), post.unsqueeze(-1)


def head_features(hidden, w, c):
    flat = hidden.flatten(-2)
    mix = linear(flat, w[OUTER_NAMES[1]]) * inverse_rms(flat, c.rms_norm_eps)
    pre = (mix * w[OUTER_NAMES[3]] + w[OUTER_NAMES[2]]).sigmoid() + c.hc_eps
    return norm((hidden * pre.unsqueeze(-1)).sum(-2), w[OUTER_NAMES[4]], c.rms_norm_eps)


def phase(positions, width, base):
    inv = 1. / (base ** (torch.arange(0, width, 2, dtype=torch.float32, device=positions.device) / width))
    freq = positions.float().unsqueeze(-1) * inv
    return tuple(f(freq).repeat_interleave(2, dim=-1).unsqueeze(1) for f in (torch.cos, torch.sin))


def rotate(x, cos, sin):
    alternating = torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).flatten(-2)
    return x * cos + alternating * sin


def rotate_tail(x, cos, sin):
    prefix, tail = x.split([x.shape[-1] - cos.shape[-1], cos.shape[-1]], dim=-1)
    return torch.cat((prefix, rotate(tail, cos, sin)), dim=-1)


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
    q0 = norm(project(x, w, p + "q_a_proj"), w[p + "q_a_layernorm.weight"], c.rms_norm_eps)
    q = rotate_tail(project(q0, w, p + "q_b_proj").reshape(len(x), h, nope + rope), cos, sin)
    k0, kr = project(x, w, p + "kv_a_proj_with_mqa").split([c.kv_lora_rank, rope], dim=-1)
    k0 = norm(k0, w[p + "kv_a_layernorm.weight"], c.rms_norm_eps)
    k, v = project(k0, w, p + "kv_b_proj").reshape(len(x), h, nope + vdim).split([nope, vdim], dim=-1)
    k = torch.cat((k, rotate(kr.unsqueeze(1), cos, sin).expand(-1, h, -1)), dim=-1)
    gate = project(x, w, p + "linear_gate").reshape(len(x), h, vdim).sigmoid()
    sink = w[p + "learnable_sink_param"][:, None, None]
    indexed = kind == SPARSE and len(x) > c.compressed_indexer_candidate_topk
    if indexed:
        index_q, index_k, index_w = index_projections(x, q0, w, c, pos, base)
    result = []
    for start in range(0, len(x), ROWS):
        stop = min(start + ROWS, len(x))
        first = max(0, start + 1 - c.sliding_window) if kind == SLIDING else 0
        width = start + ROWS - first
        num_valid = stop - first
        # Pad the tail to preserve causal row and key geometry across prefix extensions.
        query = F.pad(q[start:stop].transpose(0, 1), (0, 0, 0, ROWS - (stop - start)))
        keys = F.pad(k[first:stop].transpose(0, 1), (0, 0, 0, width - num_valid))
        vals = F.pad(v[first:stop].transpose(0, 1), (0, 0, 0, width - num_valid))
        key_id = torch.arange(first, start + ROWS, device=x.device)
        query_id = torch.arange(start, start + ROWS, device=x.device)
        distance = query_id[:, None] - key_id
        allowed = (distance >= 0) & (key_id < stop)
        if kind == SLIDING:
            allowed &= distance < c.sliding_window
        elif indexed and stop > c.compressed_indexer_candidate_topk:
            ratio = c.compressed_indexer_compress_ratio
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
            allowed &= selected | (query_id[:, None] < c.compressed_indexer_candidate_topk)
        score = (query @ keys.transpose(-1, -2)) * ((nope + rope) ** -.5)
        score = score.masked_fill(~allowed, torch.finfo(score.dtype).min)
        score = torch.cat((score, sink.expand(-1, ROWS, -1)), dim=-1)
        probabilities = (score - score.max(dim=-1, keepdim=True).values).softmax(-1)[..., :-1]
        result.append((probabilities @ vals).transpose(0, 1)[:stop - start] * gate[start:stop])
    out = project(torch.cat(result).reshape(len(x), -1), w, p + "o_proj")
    return norm(out, w[p + "post_proj_norm.weight"], c.rms_norm_eps)


def dense(x, w, prefix):
    return project(F.silu(project(x, w, prefix + "gate_proj")) * project(x, w, prefix + "up_proj"),
                   w, prefix + "down_proj")


def routing_scores(x, w, c):
    scores = project(x, w, "mlp.gate").sigmoid()
    choice = scores + w["mlp.gate.e_score_correction_bias"]
    group_scores = choice.reshape(len(x), c.n_group, -1).topk(2, dim=-1).values.sum(dim=-1)
    groups = group_scores.topk(c.topk_group, dim=-1, sorted=False).indices
    valid = torch.zeros_like(group_scores, dtype=torch.bool).scatter_(1, groups, True)
    return scores, choice.masked_fill(~valid.repeat_interleave(scores.shape[-1] // c.n_group, dim=1), -torch.inf)


def native_routes(scores, choice, c):
    ids = choice.topk(c.num_experts_per_tok, dim=-1, sorted=False).indices
    probs = scores.gather(1, ids)
    if c.norm_topk_prob:
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-20)
    return ids, probs * c.routed_scaling_factor


def routing(x, w, c):
    return native_routes(*routing_scores(x, w, c), c)


def feed_forward(x, w, c, index):
    if c.mlp_layer_types[index] == "dense":
        return norm(dense(x, w, "mlp."), w["mlp.post_fc2_norm.weight"], c.rms_norm_eps)
    ids, probs = routing(x, w, c)
    latent = project(x, w, "mlp.routed_expert_down_proj")
    routed = torch.zeros_like(latent)
    # Expert order follows the official eager accumulation. Within each expert,
    # sort by token, so extending a prefix never changes its GEMM tile membership.
    flat = ids.flatten()
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=c.n_routed_experts).tolist()
    start = 0
    for expert, count in enumerate(counts):
        if not count:
            continue
        slots = order[start:start + count]
        tokens = slots // c.num_experts_per_tok
        gate, up = linear(latent[tokens], w[EXPERTS[0]][expert].float()).chunk(2, dim=-1)
        active = F.silu(gate.clamp_max(c.swiglu_limit)) * up.clamp(-c.swiglu_limit, c.swiglu_limit)
        down = linear(active, w[EXPERTS[1]][expert].float())
        routed[tokens] = routed[tokens] + down * probs.flatten()[slots, None]
        start += count
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


def configure_f32():
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {"dtype": "float32", "matmul_precision": torch.get_float32_matmul_precision(),
            "torch_matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32}
