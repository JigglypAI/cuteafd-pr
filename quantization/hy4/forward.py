"""Hy4 text calibration. BF16 iHC frontiers, FP32 sublayers and ordered expert sums."""
from __future__ import annotations

import json

import torch
import torch.nn.functional as F

from hy4_flash import forward as flash
from hy4_flash.forward import AttentionFirst, AttentionTap, reference

DRAFT_BLOCK_SIZE = 7


def native_projection(name, inputs, weight):
    return F.linear(inputs, weight)


class Layer(flash.Layer):
    def project(self, name, inputs, project):
        if project is flash.native_projection:
            project = native_projection
        return super().project(name, inputs, project)


class Config(reference.Config):
    @classmethod
    def parse(cls, raw):
        c = raw["text_config"]
        if (raw["model_type"] != "hy_v4_vl" or c["model_type"] != "hy_v4"
                or set(c["layer_types"]) != {reference.SPARSE}
                or set(c["indexer_types"]) != {"full", "shared"}
                or c["indexer_types"][0] != "full"
                or len(c["indexer_types"]) != c["num_hidden_layers"]
                or not c["rope_parameters"]["mrope_interleaved"]
                or c["rope_parameters"]["rope_type"] != "default"):
            raise ValueError("unsupported Hy4 MLA/IndexCache configuration")
        return cls(c | {k: v for k, v in raw.items() if k.startswith("dfly_")}, ())

    def attention(self, index):
        return self.num_attention_heads, self.qk_nope_head_dim, self.qk_rope_head_dim, self.v_head_dim


def draft_targets(c):
    kv, indices = tuple(c.dfly_target_kv_layers), tuple(c.dfly_target_index_layer)
    if (indices != kv or len(kv) != c.dfly_num_hidden_layers or len(set(kv)) != len(kv)
            or any(c.indexer_types[i] != "full" for i in indices)
            or c.dfly_block_size != DRAFT_BLOCK_SIZE or c.dfly_enable_ihc
            or c.dfly_mask_token_id >= c.vocab_size or c.dfly_mtp_fusion != "slot0"
            or not c.dfly_enable_hidden_correction or not c.dfly_attn_causal
            or c.dfly_context_local_window or c.dfly_target_index_topk != c.index_topk):
        raise ValueError("unsupported Hy4 DFly2 configuration")
    return kv, indices


class Reader(flash.Reader):
    draft_targets = staticmethod(draft_targets)

    def __init__(self, snapshot, device="cpu"):
        reference.Checkpoint.__init__(self, snapshot, device)
        self.document = json.loads((self.snapshot / "config.json").read_bytes())
        self.config = Config.parse(self.document)

    def load_draft_layer(self, index):
        layer = super().load_draft_layer(index)
        return Layer(layer.index, layer.prefix, layer.weights)

    def load_layer(self, index):
        prefix = f"model.layers.{index}."
        weights = {}
        for name in self.index:
            if name.startswith(prefix):
                leaf = name.removeprefix(prefix)
                weights[leaf] = self.raw(name) if leaf in reference.EXPERTS else self.load(name)
        if not weights:
            raise ValueError(f"missing Hy4 layer {index}")
        return Layer(index, prefix, weights)


class Model(flash.Model):
    @staticmethod
    def rotate(x, cos, sin):
        # #50589 rotates adjacent pairs and stores their real then imaginary components.
        pairs = reference.rotate(x, cos, sin)
        return torch.cat((pairs[..., ::2], pairs[..., 1::2]), -1)

    @classmethod
    def rotate_tail(cls, x, cos, sin):
        prefix, rotary = x.split((x.shape[-1] - cos.shape[-1], cos.shape[-1]), -1)
        return torch.cat((prefix, cls.rotate(rotary, cos, sin)), -1)

    def rope_base(self, index):
        return self.config.rope_parameters["rope_theta"]

    def initial(self, tokens, embedding=None):
        return super().initial(tokens, embedding).to(torch.bfloat16)

    def attention_start(self, state, tokens, layer):
        hidden, post = reference.hyper(state.float(), layer.weights, "hc_attn_layer", self.config, F.linear)
        return state, post, reference.norm(hidden, layer.weights["input_layernorm.weight"], self.config.rms_norm_eps)

    def index_selection(self, inputs, q_resid, layer, positions):
        """Exact top-k over causal tokens. Bound FP32 index-score memory by query rows."""
        c, w = self.config, layer.weights
        p = "self_attn.indexer."
        q = native_projection("", q_resid, w[p + "wq_b.weight"]).view(len(inputs), c.index_n_heads, c.index_head_dim)
        k = native_projection("", inputs, w[p + "wk.weight"])
        k = F.layer_norm(k, (c.index_head_dim,), w[p + "k_norm.weight"], w[p + "k_norm.bias"], c.rms_norm_eps)
        phase = reference.phase(positions, c.qk_rope_head_dim, c.rope_parameters["rope_theta"])
        q = self.rotate_tail(q, *phase)
        k = self.rotate_tail(k[:, None], *phase).squeeze(1)
        weights = native_projection("", inputs, w[p + "weights_proj.weight"])
        weights = weights * (c.index_n_heads ** -.5) * (c.index_head_dim ** -.5)
        selections = []
        for start in range(0, len(inputs), reference.ROWS):
            stop = min(start + reference.ROWS, len(inputs))
            scores = (q[start:stop] @ k.T).relu()
            scores = (weights[start:stop, None, :] @ scores).squeeze(1)
            scores.masked_fill_(positions[None, :] > positions[start:stop, None], -torch.inf)
            ids = scores.topk(c.index_topk, dim=-1).indices
            selections.append(ids.masked_fill(ids > positions[start:stop, None], -1).to(torch.int32))
        return torch.cat(selections)

    def attention_output_inputs(self, inputs, first, layer, project=native_projection, *,
                                capture_tap=False, previous_selection=None):
        c = self.config
        pos = torch.arange(len(inputs), device=inputs.device)
        q, k, v, kr, sink = self.attention_expand(first, layer, pos, project)
        selection = None
        if len(inputs) > c.index_topk:
            if c.indexer_types[layer.index] == "full":
                selection = self.index_selection(inputs, first.normalized_q, layer, pos)
            else:
                selection = previous_selection
                if selection is None or selection.shape != (len(inputs), c.index_topk):
                    raise ValueError("shared Hy4 indexer needs preceding full-layer token selections")
        attended = []
        for start in range(0, len(inputs), reference.ROWS):
            stop = min(len(inputs), start + reference.ROWS)
            allowed = pos[:stop][None, :] <= pos[start:stop, None]
            if selection is not None:
                ids = selection[start:stop].long()
                mask = allowed.new_zeros(len(ids), stop + 1)
                mask.scatter_(1, torch.where((ids >= 0) & (ids < stop), ids, stop), True)
                allowed &= mask[:, :stop]
            attended.append(flash._with_sink(q[start:stop], k[:stop], v[:stop], allowed, sink))
        tap = None
        if capture_tap:
            groups = pos.new_empty((len(pos), 0), dtype=torch.int32) if selection is None else selection
            tap = AttentionTap(first.kv, kr, groups)
        return torch.cat(attended).flatten(1) * first.gate, tap

    def attention_finish(self, inputs, layer, project=native_projection):
        return layer.project("self_attn.o_proj", inputs, project)

    @staticmethod
    def apply_residual(state, post, result):
        return (post * result.unsqueeze(1) + state.float()).to(torch.bfloat16)

    def mlp_start(self, residual, attention_post, output, layer):
        state = self.apply_residual(residual, attention_post, output)
        hidden, post = reference.hyper(state.float(), layer.weights, "hc_mlp_layer", self.config, F.linear)
        return state, post, reference.norm(hidden, layer.weights["post_attention_layernorm.weight"], self.config.rms_norm_eps)

    def routing_scores(self, inputs, layer):
        return reference.routing_scores(inputs, layer.weights, self.config, F.linear)

    def moe_route(self, inputs, layer, project=native_projection):
        return inputs, *reference.native_routes(*self.routing_scores(inputs, layer), self.config)

    def expert(self, index, latent, layer, project=native_projection):
        if project is flash.native_projection:
            project = native_projection
        return super().expert(index, latent, layer, project)

    def mlp_finish(self, state, post, inputs, layer, routed=None, project=native_projection):
        sparse = self.config.mlp_layer_types[layer.index] == "sparse"
        if sparse != (routed is not None):
            raise ValueError("routed sum must be supplied exactly on MoE layers")
        output = self.ffn_parts(inputs, layer, "mlp.shared_experts." if sparse else "mlp.", project)
        return self.apply_residual(state, post, output if routed is None else routed + output)

    def head_features(self, states, weights=None):
        if weights is None:
            weights = {n: self.reader.load(n) for n in reference.OUTER_NAMES[1:5]}
        return reference.head_features(states.float(), weights, self.config, F.linear)

    def head(self, features, project=native_projection, weight=None):
        if project is flash.native_projection:
            project = native_projection
        return super().head(features, project, weight)



class Drafter(flash.Drafter):
    linear = staticmethod(F.linear)
    block_size = DRAFT_BLOCK_SIZE
    target_layers = staticmethod(draft_targets)

    def selection_ratio(self):
        return 1
