"""Independent text oracle for the original fixed-loop Inkling 2 Mini.

One weight bank per physical layer. A logical visit owns its own KV and four
convolution histories. Layer-major full-sequence evaluation reconstructs each
visit's history causally without ever reusing another pass's cache.
"""
from __future__ import annotations

from dataclasses import dataclass
import json

import torch
import torch.nn.functional as F

from families.inkling2 import model as small
from shape_invariant import ROWS, linear

CHECKPOINT = "thinkingmachines/Inkling-2-Mini"
REVISION = "7a67ae2314b44ef05c87fd1d4d311c58aaeb18f6"
TEXT = small.TEXT
NVFP4 = {"block_size": 16, "format": "nvfp4_e2m1", "rounding": "nearest",
         "scale_dtype": "float8_e4m3fn", "amax_type": "per_token",
         "scale_selection_rule": "fouroversix_mae"}
MXFP8 = {"block_size": 32, "format": "mxfp8_e4m3", "rounding": "nearest", "scale_dtype": "ue8m0"}


@dataclass(frozen=True)
class Config(small.Config):
    core_start: int
    core_end: int
    num_passes: int
    gate_rank: int

    @classmethod
    def from_dict(cls, root):
        if root.get("model_type") != "inkling2_loop" or root["text_config"].get("model_type") != "inkling2_loop_text":
            raise ValueError("expected a fixed-loop Inkling 2 text model")
        c = small.Config.from_dict({**root, "model_type": "inkling2"})
        loop = c.loop_config
        start, end, passes, rank = (loop[k] for k in ("core_start", "core_end", "num_passes", "gate_rank"))
        block = c.n_layers_per_attn_res_block
        if (loop.get("is_varloop") is not False or any(type(v) is not int for v in (start, end, passes, rank))
                or not 0 < start < end < c.num_hidden_layers or start % block or end % block
                or passes < 2 or rank < 1):
            raise ValueError("unsupported Inkling 2 Mini loop geometry")
        return cls(c.text, start, end, passes, rank)

    def cache_slot(self, layer, pass_index):
        if not pass_index:
            return layer
        if not (0 < pass_index < self.num_passes and self.core_start <= layer < self.core_end):
            raise ValueError("invalid Inkling 2 Mini visit")
        return self.num_hidden_layers + (pass_index - 1) * (self.core_end - self.core_start) + layer - self.core_start


def tensor_contract(cfg, formats, index=None):
    result = small.tensor_contract(cfg, index)
    h, e = cfg.hidden_size, cfg.n_routed_experts
    if index is None:
        for stem in ("middle_cycle_cross_pass_queries", "middle_cycle_cross_pass_kv_queries"):
            result[TEXT + stem] = ([2 * (cfg.core_end - cfg.core_start), h], "F32")
        return result
    stem = TEXT + f"layers.{index}."
    if cfg.core_start <= index < cfg.core_end:
        for kind in ("attn", "k", "v", "mlp"):
            for op, shape in (("down", [cfg.gate_rank, 2*h]), ("forget", [h, cfg.gate_rank]),
                              ("gain", [h, cfg.gate_rank])):
                result[stem + f"controllers.{kind}.{op}_weight"] = (shape, "BF16")
    for name, rows, cols in (("w13", 2 * cfg.intermediate_size, h), ("w2", h, cfg.intermediate_size)):
        key = stem + f"mlp.experts.{name}_weight"
        if formats[key] == "nvfp4":
            result[key] = ([e, rows, cols // 2], "U8")
            result[key + "_scale"] = ([e, rows, cols // 16], "F8_E4M3")
            result[key + "_amax"] = ([e], "F32")
            result[key + "_outer_scale_denominator"] = ([1], "F32")
    return result


class Weights(small.Weights):
    def __init__(self, snapshot, cfg, device):
        super().__init__(snapshot, device)
        quant = json.loads((snapshot / "hf_quant_config.json").read_bytes())
        if quant.get("quant_method") != "inkling2":
            raise ValueError("unexpected Inkling 2 Mini quantization method")
        groups = {}
        for group in quant["model"]["parameter_groups"]:
            for name in group["tensors"]:
                if name in groups:
                    raise ValueError(f"duplicate quantization group for {name}")
                groups[name] = group
        self.formats = {name: groups[name]["compute_dtype"] for name in self.index
                        if name.startswith(TEXT) and name.endswith((".w13_weight", ".w2_weight"))}
        self.contracts = [tensor_contract(cfg, self.formats, i) for i in range(cfg.num_hidden_layers)]
        expected = tensor_contract(cfg, self.formats)
        for contract in self.contracts:
            expected.update(contract)
        if {name for name in self.index if name.startswith(TEXT)} != expected.keys():
            raise ValueError("unexpected Inkling 2 Mini text tensors")
        for name, (shape, dtype) in expected.items():
            desc = self.descriptor(name)
            if desc["shape"] != shape or desc["dtype"] != dtype:
                raise ValueError(f"unexpected Mini tensor layout: {name}")
            group = groups.get(name)
            if name in self.formats:
                fmt = self.formats[name]
                act, shape, storage = {"nvfp4": (NVFP4, [1, 16], "uint8"),
                                       "mxfp8": (MXFP8, [32, 32], "float8_e4m3fn")}[fmt]
                if (group.get("required_activation_quantization") != act or group.get("weight_block_shape") != shape
                        or group.get("weight_transform") != fmt + "_nearest" or group.get("output_dtype") != storage
                        or (fmt == "nvfp4" and group.get("weight_scale_format") != "amax_denominator")):
                    raise ValueError(f"unexpected Mini expert policy: {name}")
            elif ".experts." not in name and (group is None or group.get("weight_transform") != "cast"
                    or group.get("compute_dtype") != {"F32": "float32", "BF16": "bfloat16"}[dtype]):
                raise ValueError(f"unexpected Mini dense precision: {name}")

    def load_layer(self, index):
        prefix = TEXT + f"layers.{index}."
        return {(n.removeprefix(prefix) if n.startswith(prefix) else n.removeprefix(TEXT)):
                (self.tensor(n) if ".experts." in n else self.tensor(n).float()) for n in self.contracts[index]}


def e2m1_values(device):
    return torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32, device=device)


def dequant_nvfp4(weight, scale, amax, denominator):
    magnitudes = e2m1_values(weight.device)
    lookup = torch.cat((magnitudes, -magnitudes))
    codes = torch.arange(256, device=weight.device)
    pair = torch.stack((lookup[codes & 15], lookup[codes >> 4]), -1)
    values = pair[weight.long()].reshape(*scale.shape, 16) * scale.float().unsqueeze(-1)
    return (values * (amax.float() / denominator.float())).flatten(-2)


def quantize_nvfp4(x):
    """Per-token amax, E4M3 block scale, nearest-even E2M1; lowest MAE wins.

    Error uses explicit two products and the vendor's adjacent-pair reduction.
    Ties between scales choose the max/6 candidate; ties between E2M1 codes
    choose the even code. The scale encoder deliberately uses 1536, not 2688.
    """
    x = x.float()
    blocks = x.reshape(*x.shape[:-1], -1, 16)
    amax = x.abs().amax(-1, keepdim=True).unsqueeze(-1)
    largest = torch.finfo(torch.float32).max
    # Scalar / tensor may lower to reciprocal times scalar, not div.rn.
    encoder = torch.full_like(amax, 1536.).div(amax).clamp_max(largest)
    decoder = torch.ones_like(encoder).div(encoder)
    outer = amax * (1. / 1536.)
    bmax = blocks.abs().amax(-1, keepdim=True)
    levels = e2m1_values(x.device)
    cuts = (levels[:-1] + levels[1:]) * .5

    def candidate(inverse):
        scale = (bmax * (encoder * inverse)).clamp_max(448.).to(torch.float8_e4m3fn).float()
        inverse_scale = torch.ones_like(scale).div(scale * decoder).clamp_max(largest)
        normalized = blocks * inverse_scale
        magnitude = normalized.abs()
        # Alternate strict/non-strict cuts to give each exact midpoint to the even nibble.
        code = torch.bucketize(magnitude.contiguous(), cuts)
        upper = (code < 7) & ((code & 1) != 0) & (magnitude == cuts[code.clamp_max(6)])
        value = torch.copysign(levels[code + upper], normalized)
        reconstructed = (value * scale) * outer
        error = (reconstructed - blocks).abs()
        for _ in range(4):
            error = error[..., ::2] + error[..., 1::2]
        return reconstructed, error

    six, err6 = candidate(1. / 6.)
    four, err4 = candidate(.25)
    return torch.where(err4 < err6, four, six).reshape(x.shape)


def message_queries(query_mlp, key_value):
    query_mlp = query_mlp.unflatten(0, (-1, 2))
    key_value = key_value.unflatten(0, (-1, 2))
    return torch.cat((query_mlp[:, :1], key_value, query_mlp[:, 1:]), dim=1)


def controller(x, message, layer, kind, eps):
    stem = "controllers." + kind + "."
    paired = torch.cat((small.rms_norm(x, None, eps), small.rms_norm(message, None, eps)), -1)
    rank = linear(paired, layer[stem + "down_weight"])
    forget = linear(rank, layer[stem + "forget_weight"]).tanh()
    gain = linear(rank, layer[stem + "gain_weight"]).tanh()
    return (1. - forget) * x + gain * message


def relative_attention(q, k, v, relative, gate, sinks, window):
    """Fixed-width causal tiles with batched GQA and separate sink normalization."""
    length, heads, dim = q.shape
    k = k.repeat_interleave(heads // k.shape[1], dim=1).transpose(0, 1)
    v = v.repeat_interleave(heads // v.shape[1], dim=1).transpose(0, 1)
    extent = relative.shape[-1]
    output = torch.empty_like(q)
    for start in range(0, length, ROWS):
        end, next_tile = min(length, start + ROWS), start + ROWS
        first = 0 if window is None else max(0, start - window + 1)
        qpad, kpad = next_tile - end, max(0, next_tile - length)
        distance = (torch.arange(start, next_tile, device=q.device)[:, None]
                    - torch.arange(first, next_tile, device=q.device)[None, :])
        valid = distance >= 0
        if window is not None:
            valid &= distance < window
        keys = F.pad(k[:, first:min(length, next_tile)], (0, 0, 0, kpad))
        values = F.pad(v[:, first:min(length, next_tile)], (0, 0, 0, kpad))
        query = F.pad(q[start:end].transpose(0, 1), (0, 0, 0, qpad))
        rel = F.pad(relative[start:end].transpose(0, 1), (0, 0, 0, qpad))
        bias = rel.gather(-1, distance.clamp(0, extent - 1).expand(heads, -1, -1))
        bias.masked_fill_((distance < 0) | (distance >= extent), 0.)
        scores = (query @ keys.transpose(1, 2)) * dim ** -.5 + bias
        scores.masked_fill_(~valid, -float("inf"))
        sink = F.pad(sinks[start:end].T, (0, qpad)).unsqueeze(-1)
        p = torch.cat((scores, sink), -1).softmax(-1)[..., :-1]
        output[start:end] = (p @ values)[:, :end-start].transpose(0, 1) * gate[start:end].sigmoid()
    return output


def attention(inputs, layer, cfg, index):
    local = index in cfg.local_layer_ids
    heads, kv_heads, dim = cfg.geometry(index)
    names = ("attn_norm",) * 3 if local else ("q_input_norm", "k_input_norm", "v_input_norm")
    xq, xk, xv = (small.rms_norm(x, layer[name + ".weight"], cfg.rms_norm_eps)
                  for x, name in zip(inputs, names, strict=True))

    def project(x, name):
        return linear(x, layer["attn." + name + ".weight"])

    q = project(xq, "wq_du").reshape(-1, heads, dim)
    r = project(xq, "wr_du").reshape(-1, heads, cfg.d_rel)
    gate = project(xq, "gate_attn").reshape(-1, heads, dim)
    sinks = project(xq, "attn_sink_proj")
    kv = []
    for x, name, proj in ((xk, "k", "wk_dv"), (xv, "v", "wv_dv")):
        stem = "attn." + name + "_sconv."
        dynamic = None if local else linear(x, layer[stem + "weight_proj"]).reshape(-1, kv_heads, cfg.sconv_kernel_size)
        value = small.short_conv(project(x, proj), layer[stem + ("weight" if local else "static_w")], dynamic)
        kv.append(value.reshape(-1, kv_heads, dim))
    k, v = kv
    clip = small.l2_clip
    q = clip(q, cfg.qk_l2_clip_threshold, 1e-6, layer["attn.q_l2_clip.scale"])
    k = clip(k, cfg.qk_l2_clip_threshold, 1e-6)
    v = small.rms_norm(v, None, cfg.rms_norm_eps)
    r = clip(r, cfg.rel_l2_clip_threshold, 1e-6, layer["attn.rel_l2_clip.scale"])
    if not local and cfg.log_scaling_n_floor is not None:
        positions = torch.arange(1, len(q) + 1, device=q.device)
        tau = 1 + cfg.log_scaling_alpha * (positions.float() / cfg.log_scaling_n_floor).clamp_min(1.).log()
        q, r = q * tau[:, None, None], r * tau[:, None, None]
    relative = linear(r, layer["attn.rel_logits_proj.proj"].T.contiguous())
    out = relative_attention(q, k, v, relative, gate, sinks, cfg.sliding_window_size if local else None)
    return linear(out.reshape(len(q), -1), layer["attn.wo_ud.weight"])


class Experts:
    """Native checkpoint weight decode, with an explicit unquantized diagnostic mode."""
    def __init__(self, layer, cfg, *, native=True):
        self.layer, self.cfg, self.native = layer, cfg, native
        prefix = "mlp.experts."
        self.keys = [prefix + "w13_weight", prefix + "w2_weight"]
        self.qdq = [quantize_nvfp4 if layer[key].dtype == torch.uint8 else small.quantize_mxfp8 for key in self.keys]
        self.decoded = {}

    def weight(self, i, expert):
        key = self.keys[i]
        w, s = self.layer[key][expert], self.layer[key + "_scale"][expert]
        if w.dtype == torch.uint8:
            return dequant_nvfp4(w, s, self.layer[key + "_amax"][expert], self.layer[key + "_outer_scale_denominator"])
        return small.dequant_mxfp8(w, s)

    def __call__(self, x):
        layer, cfg = self.layer, self.cfg
        scores = linear(x, layer["mlp.gate.weight"]).sigmoid()
        ids = (scores + layer["mlp.gate.bias"]).argsort(dim=-1, descending=True, stable=True)[:, :cfg.num_experts_per_tok]
        weights = scores.gather(-1, ids)
        gammas = (weights / (weights.sum(-1, keepdim=True) + 1e-20)) * cfg.route_scale
        inputs = self.qdq[0](x) if self.native else x
        slots = x.new_empty((len(x), cfg.num_experts_per_tok, cfg.hidden_size))
        rows, choices = torch.where(torch.ones_like(ids, dtype=torch.bool))
        ordered = ids.flatten().argsort(stable=True)
        counts = torch.bincount(ids.flatten(), minlength=cfg.n_routed_experts).tolist()
        offset = 0
        for expert, count in enumerate(counts):
            if not count:
                continue
            order = ordered[offset:offset + count]
            offset += count
            r, c = rows[order], choices[order]
            if expert not in self.decoded:
                self.decoded[expert] = (self.weight(0, expert), self.weight(1, expert))
            up, down = self.decoded[expert]
            gate_up = linear(inputs[r], up)
            gamma = gammas[r, c]
            values = small.capped_swiglu(gate_up, gamma if self.native else torch.ones_like(gamma),
                                          cfg.mlp_gate_softcap, cfg.mlp_up_softcap)
            if self.native:
                slots[r, c] = linear(self.qdq[1](values), down)
            else:
                slots[r, c] = linear(values, down) * gamma[:, None]
        # Sum in ascending expert id, like HF eager index_add, not router order.
        return slots.gather(1, ids.argsort(-1)[..., None].expand_as(slots)).sum(1)


def layer_forward(partial, blocks, layer, cfg, index, experts, messages=None):
    eps = cfg.rms_norm_eps
    inputs = (small.attn_res(blocks, partial, layer[f"attn_res.attn_queries.{index - 1}"], eps)
              if index else (partial,))
    if len(inputs) == 1:
        inputs *= 3
    if messages is not None:
        inputs = tuple(controller(x, m, layer, kind, eps)
                       for x, m, kind in zip(inputs, messages[:3], ("attn", "k", "v"), strict=True))
    out = small.short_conv(attention(inputs, layer, cfg, index), layer["attn_sconv.weight"])
    if index % cfg.n_layers_per_attn_res_block == 0:
        blocks = (*blocks, partial)
    else:
        out = partial + out
    mlp_in = small.attn_res(blocks, out, layer[f"attn_res.mlp_queries.{index}"], eps)[0]
    if messages is not None:
        mlp_in = controller(mlp_in, messages[3], layer, "mlp", eps)
    mlp_in = small.rms_norm(mlp_in, layer["mlp_norm.weight"], eps)
    mlp = small.short_conv(experts(mlp_in), layer["mlp_sconv.weight"])
    return out + mlp, blocks


def forward(weights, cfg, token_batches, *, native=True):
    """Layer-major text forward. Each state is one unpadded token sequence."""
    embed, norm = (weights.tensor(TEXT + key) for key in ("embed.weight", "embed_norm.weight"))
    states = [(small.rms_norm(F.embedding(ids, embed), norm, cfg.rms_norm_eps), ()) for ids in token_batches]
    del embed, norm
    queries = message_queries(*(weights.tensor(TEXT + "middle_cycle_cross_pass_" + k)
                                for k in ("queries", "kv_queries")))

    def visit(states, layers, previous=None):
        for i in layers:
            layer = weights.load_layer(i)
            experts = Experts(layer, cfg, native=native)
            for row, (partial, blocks) in enumerate(states):
                messages = None
                if previous is not None:
                    prev_partial, prev_blocks = previous[row]
                    messages = small.attn_res(prev_blocks, prev_partial, queries[i-cfg.core_start], cfg.rms_norm_eps)
                states[row] = layer_forward(partial, blocks, layer, cfg, i, experts, messages)
            del layer, experts
        return states

    initial = visit(states, range(cfg.core_start))
    end_of_pass = visit(initial.copy(), range(cfg.core_start, cfg.core_end))
    for _ in range(1, cfg.num_passes):
        # Skip the pre-loop block appended at the first core layer: it is
        # already present verbatim as the next pass's starting partial.
        previous = [(partial, blocks[len(initial[row][1])+1:])
                    for row, (partial, blocks) in enumerate(end_of_pass)]
        end_of_pass = visit(initial.copy(), range(cfg.core_start, cfg.core_end), previous)
    final = visit(end_of_pass, range(cfg.core_end, cfg.num_hidden_layers))
    query, norm = (weights.tensor(TEXT + key) for key in ("attn_res.output_query", "norm.weight"))
    return [small.output_norm(p, b, query, norm, cfg) for p, b in final]
