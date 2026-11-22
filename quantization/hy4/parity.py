"""Gate the Hy4 calibration trajectory against pinned HF eager operators with adjacent RoPE."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import time
from unittest.mock import patch

import torch
import torch.nn.functional as F

from . import forward
from hy4_flash.parity import TargetCache
from families.hy4_flash.golden import official_key, rel_l2, REL_LIMIT
from kimi_k3.uniform_k2.storage import atomic_json

SOURCE_REVISION = "c10c0de5b6f850703fb0a90e65d1c32b558b9167"
HF_REVISION = "2f1bc5e321b9a736200a590d6e7743ccdeb60772"
ROPE_REVISION = "bd5b6b6d979cd22a7d237d9e84b88d212b08a545"
HF_FILES = {
    "modeling_hy_v4_vl.py": "08847c95e23e75bfb99e2c2ba6cfb57c8060f6d1cf80fdf0e149e9a066e2544b",
    "configuration_hy_v4_vl.py": "f40f93684b722d68acf37604e5905210fcf404d15f8b98435a8e7feac228689c",
}
SCHEMA = "cuteafd-hy4-full-reference-forward-v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_hashes():
    files = (Path(__file__), Path(forward.__file__), Path(forward.flash.__file__),
             Path(forward.reference.__file__), Path(inspect.getfile(forward.native_projection.__globals__["linear"])))
    return {str(path.relative_to(Path(__file__).resolve().parents[2])): sha(path) for path in files}


def adjacent_rope(q, k, cos, sin, unsqueeze_dim=1):
    """The selected transformers #50589 formula, on VL text's MRoPE angles."""
    cos = cos[..., :cos.shape[-1] // 2].unsqueeze(unsqueeze_dim)
    sin = sin[..., :sin.shape[-1] // 2].unsqueeze(unsqueeze_dim)

    def rotate(x):
        real, imag = x[..., ::2], x[..., 1::2]
        return torch.cat((real * cos - imag * sin, imag * cos + real * sin), dim=-1)
    return rotate(q), rotate(k)


class Oracle:
    def __init__(self, reader):
        from transformers.models.hy_v4_vl import modeling_hy_v4_vl as hf
        self.hf = hf
        directory = Path(hf.__file__).parent
        if {name: sha(directory / name) for name in HF_FILES} != HF_FILES:
            raise ValueError("installed HF Hy4 VL differs from the pinned reference")
        self.config = hf.HYV4VLTextConfig.from_dict(reader.document["text_config"])
        self.config._attn_implementation = "eager"
        self.config._experts_implementation = "eager"
        with torch.device(reader.device):
            self.rotary = hf.HYV4VLRotaryEmbedding(self.config)

    def layer(self, source):
        with torch.device("meta"):
            layer = self.hf.HYV4VLDecoderLayer(self.config, source.index)

        class ExpertMatrices(torch.nn.Parameter):
            # Promote only the expert that the pinned eager MoE currently evaluates.
            def __getitem__(self, key):
                return super().__getitem__(key).float()

        weights = {}
        for name, tensor in source.weights.items():
            value = ExpertMatrices(tensor, requires_grad=False) if name in forward.reference.EXPERTS else tensor
            weights[official_key(name)] = value
        layer.load_state_dict(weights, assign=True)
        return layer.eval()

    def head(self, states, outer):
        c = self.config
        with torch.device("meta"):
            hc = self.hf.HYV4VLHyperHead(c)
            norm = self.hf.HYV4VLRMSNorm(c.hidden_size, c.rms_norm_eps)
        names = forward.reference.OUTER_NAMES
        hc.load_state_dict(dict(zip(("hc_fn", "hc_base", "hc_scale"), (outer[n] for n in names[1:4]))),
                           assign=True)
        norm.load_state_dict({"weight": outer[names[4]]}, assign=True)
        return F.linear(norm(hc(states.float()[None]))[0], outer[names[5]])


def check(name, actual, expected):
    error = rel_l2(actual.float(), expected.float())
    if not math.isfinite(error) or error > REL_LIMIT:
        raise ValueError(f"Hy4 {name} relative L2 {error:.6g} exceeds {REL_LIMIT}")
    return error


def oracle_attention(oracle, module, x, positions, mask, previous=None, cache=None):
    phase = oracle.rotary(x, positions[None, None, :])
    out, _, ids = module(x[None], phase, mask[None, None], cache=cache,
                         prev_topk_indices=previous)
    return out[0], ids


def check_layer(oracle, model, layer, hf_layer, state, tokens, *, capture=False, previous=None):
    residual, post, inputs = model.attention_start(state, tokens, layer)
    first = model.attention_first(inputs, layer)
    out, tap = model.attention_output_inputs(inputs, first, layer, capture_tap=capture, previous_selection=previous)
    attended = model.attention_finish(out, layer)
    pos = torch.arange(len(tokens), device=tokens.device)
    mask = torch.zeros(len(pos), len(pos), device=pos.device).masked_fill_(pos[None] > pos[:, None], -torch.inf)
    e_post, e_in = hf_layer.attn_hc(state.float()[None])
    previous_hf = previous.clamp_min(0)[None] if previous is not None else None
    if previous_hf is None and model.config.indexer_types[layer.index] == "shared":
        # Every token fits in short-row IndexCache. The independent causal mask discards its future.
        if len(tokens) > model.config.index_topk:
            raise ValueError("missing previous full-layer selection for long prompt")
        previous_hf = pos.int().view(1, 1, -1).expand(1, len(pos), -1)
    expected, topk = oracle_attention(oracle, hf_layer.self_attn, hf_layer.input_layernorm(e_in)[0],
                                      pos, mask, previous=previous_hf)
    errors = {"attention": check(f"{layer.index} attention", attended, expected)}
    if capture and len(pos) > model.config.index_topk:
        valid = topk[0].masked_fill(topk[0] > pos[:, None], -1)
        # Ignore filler IDs for causal prefixes with fewer than k legal keys.
        complete = pos >= model.config.index_topk - 1
        if not torch.equal(valid[complete].sort(-1).values, tap.groups[complete].sort(-1).values):
            raise ValueError(f"layer {layer.index} Hy4 DSA index selection differs")
    states, mlp_post, mlp_inputs = model.mlp_start(residual, post, attended, layer)
    expected_states = (state.float() + e_post[0, :, :, None] * expected[:, None, :]).bfloat16()
    # Per-sublayer parity uses the same BF16 boundary input for the next operation.
    errors["attention_residual"] = check(f"{layer.index} attention residual", states, expected_states)
    e_post, e_in = hf_layer.ffn_hc(states.float()[None])
    expected_mlp = hf_layer.mlp(hf_layer.post_attention_layernorm(e_in))[0]
    routed = None
    if model.config.mlp_layer_types[layer.index] == "sparse":
        latent, ids, gamma = model.moe_route(mlp_inputs, layer)
        routed = torch.zeros_like(latent)
        for expert in ids.unique().tolist():
            rows, coeff = model.expert_rows(ids, gamma, expert)
            routed[rows] += model.expert(expert, latent[rows], layer) * coeff[:, None]
    actual = model.mlp_finish(states, mlp_post, mlp_inputs, layer, routed)
    expected = (states.float() + e_post[0, :, :, None] * expected_mlp[:, None, :]).bfloat16()
    errors["mlp_residual"] = check(f"{layer.index} MLP residual", actual, expected)
    return actual, tap, errors


def check_drafter(oracle, model, draft, taps, features, tokens, embedding, head):
    """DFly2 slot0 fusion and sequential correction; pinned HF eager attention and FFN."""
    w, c = draft.aux, model.config
    end, width = len(tokens), forward.DRAFT_BLOCK_SIZE
    pos = torch.arange(end, end + width, device=tokens.device)

    def norm(x, weight):
        return F.rms_norm(x, (x.shape[-1],), weight, c.rms_norm_eps)

    bonus = F.linear(features[-1], head).argmax()
    ids = torch.full_like(pos, c.dfly_mask_token_id)
    ids[0] = bonus
    expected = embedding.index_select(0, ids.cpu()).to(device=tokens.device, dtype=torch.float32)
    p = "mtp_fusion."
    joined = torch.cat((norm(expected[0], w[p+"enorm.weight"]), norm(features[-1], w[p+"hnorm.weight"])))
    expected[0] += F.linear(joined, w[p+"eh_proj.weight"]) * w[p+"gate"]
    actual = draft.start(bonus, features[-1], embedding)
    checks = {"fusion": check("draft fusion", actual, expected), "layers": []}
    for i, layer in enumerate(draft.layers):
        weights = layer.weights
        with torch.device("meta"):
            attn = oracle.hf.HYV4VLAttention(oracle.config, layer.index)
            mlp = oracle.hf.HYV4VLMLP(oracle.config, weights["mlp.gate_proj.weight"].shape[0])
        attn.indexer = None
        attn.load_state_dict({official_key(k).removeprefix("self_attn."): v for k,v in weights.items()
                              if k.startswith("self_attn.")}, assign=True)
        mlp.load_state_dict({k.removeprefix("mlp."): v for k,v in weights.items() if k.startswith("mlp.")},
                            assign=True)
        cache = TargetCache(taps[layer.index], end, 1)
        nc = cache.keys.shape[2]
        mask = expected.new_zeros(width, nc + width)
        mask[:, nc:] = torch.triu(torch.full_like(mask[:, nc:], -torch.inf), diagonal=1)
        selected = torch.arange(nc + width, device=pos.device).view(1, 1, -1).expand(1, width, -1)
        attended, _ = oracle_attention(oracle, attn, norm(expected, weights["input_layernorm.weight"]),
                                       pos, mask, previous=selected, cache=cache)
        expected = expected + attended
        expected = expected + mlp(norm(expected, weights["post_attention_layernorm.weight"]))
        actual = draft.layer(actual, i, taps, end)
        checks["layers"].append(check(f"draft {i}", actual, expected))
    expected = norm(expected, w["final_norm.weight"])
    prev, logits = bonus, []
    p = "hidden_correction."
    for row in expected[1:]:
        prev_embed = embedding[prev.cpu()].to(device=row.device, dtype=torch.float32)
        cat = torch.cat((norm(row, w[p+"hidden_norm.weight"]), norm(prev_embed, w[p+"embed_norm.weight"])))
        active = F.silu(F.linear(cat, w[p+"gate_proj.weight"])) * F.linear(cat, w[p+"up_proj.weight"])
        logit = F.linear(row + F.linear(active, w[p+"down_proj.weight"]), head)
        logits.append(logit)
        prev = logit.argmax()
    expected = torch.stack(logits)
    actual = draft.proposals(draft.finish(actual), bonus, embedding, head_weight=head)
    checks["logits"] = check("draft proposals", actual, expected)
    if not torch.equal(actual.argmax(-1), expected.argmax(-1)):
        raise ValueError("Hy4 DFly2 proposal differs")
    return checks


def verify_preflight(report, corpus_digest):
    if (report["schema"] != SCHEMA or report["status"] != "passed"
            or report["source_revision"] != SOURCE_REVISION or report["hf_revision"] != HF_REVISION
            or report["rope_revision"] != ROPE_REVISION or report["hf_files"] != HF_FILES
            or report["code_identity"] != code_hashes() or report["corpus_manifest_sha256"] != corpus_digest
            or [row["index"] for row in report["layers"]] != list(range(78))
            or [row["index"] for row in report["long_context"]] != [1, 2]
            or report["long_tokens"] <= 2048 or len(report["head"]) != 3 or len(report["dfly2"]) != 3):
        raise ValueError("incomplete or stale Hy4 parity receipt")
    values = list(report["head"])
    for row in report["layers"]:
        if len(row["errors"]) != 3:
            raise ValueError("missing Hy4 parity prompt")
        for result in row["errors"]:
            values.extend(result[k] for k in ("attention", "attention_residual", "mlp_residual"))
    for row in report["long_context"]:
        if row["selector_passed"] is not True:
            raise ValueError("missing long-context selector verification")
        values.extend(row["errors"].values())
    for draft in report["dfly2"]:
        if len(draft["layers"]) != 5:
            raise ValueError("missing drafter layer parity")
        values.extend((draft["fusion"], *draft["layers"], draft["logits"]))
    if not all(math.isfinite(v) and 0 <= v <= REL_LIMIT for v in values):
        raise ValueError("numerical failure in Hy4 parity receipt")


@torch.inference_mode()
def run(source, corpus, output, device, report):
    from .prepare_corpus import load_token_batches
    if source.name != SOURCE_REVISION:
        raise ValueError("Hy4 input is not the pinned snapshot")
    forward.reference.configure_f32()
    reader = forward.Reader(source, device)
    model, oracle = forward.Model(reader), Oracle(reader)
    prompts = iter(load_token_batches(corpus, "screening"))
    ids = [next(prompts)["input_ids"][0,-n:].to(device) for n in (12,16,24)]
    witness = json.loads((corpus/"long-context-smoke.json").read_text())
    long = torch.tensor(witness["input_ids"], device=device)
    if witness["hessian_eligible"] is not False or len(long) <= model.config.index_topk:
        raise ValueError("invalid Hy4 long-context witness")
    report["long_tokens"] = len(long)
    taps = [{} for _ in ids]
    states = [model.initial(tokens) for tokens in ids]
    long_state = model.initial(long)
    selection = None
    with patch.object(oracle.hf, "apply_rotary_pos_emb", adjacent_rope):
        for index in range(model.config.num_hidden_layers):
            layer = reader.load_layer(index)
            # A single layer of fused BF16 experts fits. Both arms use this same transfer.
            for name in forward.reference.EXPERTS:
                if name in layer.weights:
                    layer.weights[name] = layer.weights[name].to(device)
            official = oracle.layer(layer)
            errors = []
            for n, tokens in enumerate(ids):
                states[n], tap, error = check_layer(oracle, model, layer, official, states[n], tokens,
                                                   capture=index in model.config.dfly_target_kv_layers)
                if tap is not None:
                    taps[n][index] = tap
                errors.append(error)
            if index <= 2:
                long_state, tap, error = check_layer(oracle, model, layer, official, long_state, long,
                                                   capture=True, previous=selection)
                selection = tap.groups
                if index in (1,2):
                    report["long_context"].append(dict(index=index, selector_passed=True, errors=error))
                if index == 2:
                    del long_state, selection, tap
            report["layers"].append(dict(index=index, errors=errors))
            atomic_json(output, report)
            print(f"Hy4 reference parity layer {index+1}/{model.config.num_hidden_layers}", flush=True)
            del layer, official
            reader.close()
        outer = {n: reader.load(n) for n in forward.reference.OUTER_NAMES[1:]}
        features = [model.head_features(s, outer) for s in states]
        head = outer["lm_head.weight"]
        report["head"] = [check("FP32 head", model.head(f, weight=head), oracle.head(s, outer))
                          for f,s in zip(features,states)]
        draft = forward.Drafter(model)
        embedding = reader.raw("model.embed_tokens.weight")
        report["dfly2"] = [check_drafter(oracle, model, draft, t, f, tokens, embedding, head)
                           for t,f,tokens in zip(taps,features,ids)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source","corpus","output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Hy4 preflight requires a fresh receipt path")
    digest = sha(args.corpus/"manifest.json")
    report = dict(schema=SCHEMA, status="running", source_revision=SOURCE_REVISION, hf_revision=HF_REVISION,
                  rope_revision=ROPE_REVISION, hf_files=HF_FILES, corpus_manifest_sha256=digest,
                  code_identity=code_hashes(), layers=[], long_context=[])
    start = time.monotonic()
    try:
        run(args.source, args.corpus, args.output, args.device, report)
        report["status"] = "passed"
        verify_preflight(report, digest)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["seconds"] = time.monotonic() - start
        atomic_json(args.output, report)


if __name__ == "__main__":
    main()
