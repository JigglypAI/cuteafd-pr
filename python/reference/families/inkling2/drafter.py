"""Independent eager oracle for the bundled recurrent-hidden block drafter.

Arithmetic sources are pinned in sources.py. This is not a serving fallback.
Target features are the raw AttnRes partial at layer entry (except final norm).
All tensor operands remain native BF16, including per-operation RNN rounding.
"""
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Config:
    hidden: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    intermediate: int
    local: tuple
    window: int
    block: int
    rank: int
    taps: tuple
    eps: float
    theta: float
    padded_vocab: int
    valid_vocab: int
    softcap: float

    @classmethod
    def from_dict(cls, root):
        c, t = root["mtp_config"], root["text_config"]
        if c["markov_head_type"] != "rnn_hidden":
            raise ValueError("oracle requires rnn_hidden")
        result = cls(c["hidden_size"], c["num_hidden_layers"], c["num_attention_heads"],
                     c["num_key_value_heads"], c["head_dim"], c["intermediate_size"],
                     tuple(c["local_layer_ids"]), c["sliding_window_size"], c["block_size"],
                     c["markov_rank"], tuple(c["target_layer_ids"]), c["rms_norm_eps"],
                     c["rope_theta"], t["vocab_size"], t["unpadded_vocab_size"],
                     t["final_logit_softcapping"])
        if (result.hidden != t["hidden_size"] or result.heads % result.kv_heads
                or result.block < 2 or result.head_dim % 2
                or not 0 < result.valid_vocab <= result.padded_vocab):
            raise ValueError("invalid recurrent draft geometry")
        if tuple(sorted(set(result.taps))) != result.taps or result.taps[-1] != t["num_hidden_layers"]:
            raise ValueError("draft taps must ascend through final norm")
        return result


def norm(x, weight, eps):
    value = x.float()
    value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)
    return (value * weight.float()).bfloat16()


def add_norm(delta, residual, weight, eps):
    # FlashInfer's fused add+RMSNorm computes norm from the unrounded sum.
    summed = delta.float() + residual.float()
    return norm(summed, weight, eps), summed.bfloat16()


def mix(features, alpha, weight, eps):
    if features.ndim != 3 or features.shape[1] != len(alpha):
        raise ValueError("expected unprojected [rows,taps,hidden] target features")
    # Accumulate the taps in their reference order, never round before norm.
    mixed = torch.zeros_like(features[:, 0], dtype=torch.float32)
    for index in range(len(alpha)):
        mixed = mixed + features[:, index].float() * alpha[index].float()
    return norm(mixed, weight, eps)


def init_block(anchors, mask, block):
    result = mask.expand(len(anchors), block, -1).clone()
    result[:, 0] = anchors
    return result


def rope(x, positions, theta):
    width = x.shape[-1]
    inverse = 1.0 / (theta ** (torch.arange(0, width, 2, device=x.device).float()/width))
    angle = positions.float()[:, None] * inverse[None, :]
    cos, sin = angle.cos()[:, None, :], angle.sin()[:, None, :]
    first, second = x.float().chunk(2, -1)
    return torch.cat((first*cos - second*sin, second*cos + first*sin), -1).bfloat16()


def attention(q, context_k, context_v, draft_k, draft_v, *, local_window=None):
    # The window is anchored at the block's first position, not at each query.
    if local_window is not None:
        context_k, context_v = context_k[-local_window:], context_v[-local_window:]
    key, value = torch.cat((context_k, draft_k)), torch.cat((context_v, draft_v))
    group = q.shape[1] // key.shape[1]
    key, value = key.repeat_interleave(group, 1), value.repeat_interleave(group, 1)
    logits = torch.einsum("qhd,khd->hqk", q.float(), key.float()) / math.sqrt(q.shape[-1])
    probability = logits.softmax(-1)
    return torch.einsum("hqk,khd->qhd", probability, value.float()).bfloat16()


def recurrent_gate(raw, state):
    gate_raw, candidate, output = raw.bfloat16().chunk(3, -1)
    gate = gate_raw.sigmoid()
    return gate*state + (1.0-gate)*candidate.tanh(), output.tanh()


def fold_head(head, hidden_out, bias):
    # Match vendor startup folding; this is NOT head(hidden_out(latent)) with
    # an extra hidden-sized BF16 rounding between the two projections.
    projection, offset = hidden_out.double(), bias.double()
    folded = head.new_empty((head.shape[0], hidden_out.shape[1]))
    folded_bias = head.new_empty((head.shape[0],))
    for first in range(0, head.shape[0], 4096):
        rows = slice(first, first+4096)
        source = head[rows].double()
        folded[rows] = source @ projection
        folded_bias[rows] = source @ offset
    return folded, folded_bias


class Drafter:
    def __init__(self, config, weights, embedding, embedding_norm, lm_head):
        self.config, self.weights = config, weights
        self.embedding, self.embedding_norm, self.lm_head = embedding, embedding_norm, lm_head
        if any(value.dtype != torch.bfloat16 for value in weights.values()):
            raise ValueError("recurrent oracle operands must be native BF16")
        self.alpha = weights["fuse_weight"].float().softmax(-1).bfloat16()
        self.fold_weight, self.fold_bias = fold_head(lm_head, weights["markov_head.hidden_out.weight"],
                                                    weights["markov_head.hidden_out.bias"])

    def context(self, features, first=0):
        c, w = self.config, self.weights
        if features.shape[1:] != (len(c.taps), c.hidden):
            raise ValueError("target feature count or hidden width differs")
        positions = torch.arange(first, first+len(features), device=features.device)
        result = []
        for layer in range(c.layers):
            p = f"layers.{layer}"
            mixed = mix(features, self.alpha[layer], w[f"{p}.dflare_norm.weight"], c.eps)
            key = F.linear(mixed, w[f"{p}.self_attn.k_proj_ctx.weight"])
            key = norm(key.view(-1, c.kv_heads, c.head_dim), w[f"{p}.self_attn.k_norm_ctx.weight"], c.eps)
            key = rope(key, positions, c.theta)
            value = F.linear(mixed, w[f"{p}.self_attn.v_proj_ctx.weight"]).view(-1, c.kv_heads, c.head_dim)
            result.append((key, value))
        return result

    def forward(self, anchor, features, *, ranks=1):
        """One real block, with optional simulated head/FFN TP2 reductions.

        `features` contains all committed rows preceding the block's anchor.
        The oracle materializes complete context deliberately; serving must page
        the global layer and use independent local rings. No cache approximation.
        """
        c, w = self.config, self.weights
        if ranks not in (1, 2) or c.heads % ranks or c.kv_heads % ranks or c.intermediate % ranks:
            raise ValueError("TP split cuts heads or FFN columns")
        if len(features) == 0:
            raise ValueError("recurrent feature requires a last committed target token")
        positions = torch.arange(len(features), len(features)+c.block, device=features.device)
        token = torch.tensor([anchor], dtype=torch.long, device=features.device)
        anchors = norm(F.embedding(token, self.embedding), self.embedding_norm, c.eps)
        hidden = init_block(anchors, w["mask_embed"], c.block)[0]
        contexts = self.context(features)
        residual = hidden
        normalized = norm(hidden, w["layers.0.input_layernorm.weight"], c.eps)
        for layer in range(c.layers):
            p, a = f"layers.{layer}", f"layers.{layer}.self_attn"
            q = F.linear(normalized, w[f"{a}.q_proj.weight"]).view(c.block, c.heads, c.head_dim)
            k = F.linear(normalized, w[f"{a}.k_proj.weight"]).view(c.block, c.kv_heads, c.head_dim)
            v = F.linear(normalized, w[f"{a}.v_proj.weight"]).view(c.block, c.kv_heads, c.head_dim)
            q = rope(norm(q, w[f"{a}.q_norm.weight"], c.eps), positions, c.theta)
            k = rope(norm(k, w[f"{a}.k_norm.weight"], c.eps), positions, c.theta)
            context_k, context_v = contexts[layer]
            deltas = []
            for rank in range(ranks):
                qspan = slice(rank*c.heads//ranks, (rank+1)*c.heads//ranks)
                kvspan = slice(rank*c.kv_heads//ranks, (rank+1)*c.kv_heads//ranks)
                attended = attention(q[:, qspan], context_k[:, kvspan], context_v[:, kvspan], k[:, kvspan], v[:, kvspan],
                                     local_window=c.window if layer in c.local else None).flatten(1)
                columns = slice(rank*c.heads*c.head_dim//ranks, (rank+1)*c.heads*c.head_dim//ranks)
                deltas.append(F.linear(attended.float(), w[f"{a}.o_proj.weight"][:, columns].float()))
            delta = sum(deltas).bfloat16()
            normalized, residual = add_norm(delta, residual, w[f"{p}.post_attention_layernorm.weight"], c.eps)
            deltas = []
            for rank in range(ranks):
                span = slice(rank*c.intermediate//ranks, (rank+1)*c.intermediate//ranks)
                gate = F.linear(normalized, w[f"{p}.mlp.gate_proj.weight"][span])
                up = F.linear(normalized, w[f"{p}.mlp.up_proj.weight"][span])
                activation = (F.silu(gate.float()) * up.float()).bfloat16()
                deltas.append(F.linear(activation.float(), w[f"{p}.mlp.down_proj.weight"][:, span].float()))
            delta = sum(deltas).bfloat16()
            gamma = w["norm.weight"] if layer+1 == c.layers else w[f"layers.{layer+1}.input_layernorm.weight"]
            normalized, residual = add_norm(delta, residual, gamma, c.eps)
        # Fill-in layout: anchor row is not a prediction. Seven mask rows are.
        predicted_hidden = normalized[1:]
        f_rnn = mix(features[-1:], self.alpha[-1], w["markov_head.dflare_norm.weight"], c.eps)
        state = predicted_hidden.new_zeros(1, c.rank)
        previous = token
        logits, states, tokens = [], [], []
        for hidden in predicted_hidden:
            inputs = torch.cat((state, F.embedding(previous, w["markov_head.markov_w1.weight"]),
                                hidden[None], f_rnn), -1)
            raw = F.linear(inputs.float(), w["markov_head.rnn_proj.weight"].float(),
                           w["markov_head.rnn_proj.bias"].float())
            state, latent = recurrent_gate(raw, state)
            base = F.linear(hidden[None], self.lm_head)
            bias = F.linear(latent, self.fold_weight, self.fold_bias)
            value = (c.softcap * (((base+bias).float())/c.softcap).tanh()).bfloat16()[:, :c.valid_vocab]
            previous = value.float().argmax(-1)
            logits.append(value)
            states.append(state)
            tokens.append(previous)
        return {"hidden": normalized, "logits": torch.cat(logits), "states": torch.cat(states),
                "tokens": torch.cat(tokens), "contexts": contexts}
