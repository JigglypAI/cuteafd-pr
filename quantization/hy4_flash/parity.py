"""Compare Hy4 calibration against pinned eager operators, with no Hessian capture or encoding."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

import torch
import torch.nn.functional as F

from families.hy4_flash import golden, model as reference, sources
from kimi_k3.uniform_k2.storage import atomic_json
from . import forward, recipe
from .prepare_corpus import load_token_batches


def assert_close(name, actual, expected):
    error = golden.rel_l2(actual, expected)
    if error > golden.REL_LIMIT:
        raise ValueError(f"{name} relative L2 {error} exceeds {golden.REL_LIMIT}")
    return error


def check_layer(model, oracle, layer, state, ids, official_layer, *, capture=False):
    actual, tap = model.forward_layer(state, ids, layer, capture_tap=capture)
    # The F32 layer oracle promotes source rows to the surrounding hidden dtype.
    # Invoke exactly its BF16 variant at the single lookup-QDQ boundary specified
    # by the checkpoint, then resume F32 accumulation on both sides.
    original = oracle.official._fp8_round_trip
    qdq = (patch.object(oracle.official, "_fp8_round_trip", lambda rows: original(rows.bfloat16()).float())
           if reference.TABLE in layer.weights else nullcontext())
    with qdq:
        expected = oracle.official_forward(official_layer, layer.index, state.unsqueeze(0), ids)[0]
    return actual, tap, assert_close(f"target {layer.index}", actual, expected)


class TargetCache:
    """A read-only prefix of compressed target keys for the official eager MLA."""
    def __init__(self, tap, end, ratio):
        selected = tap.groups[end - 1].tolist()
        rows = ([j * ratio + k for j in selected if j >= 0 for k in range(ratio) if j * ratio + k < end]
                if selected else list(range(end)))
        self.keys = tap.kv[rows][None, None]
        self.positions = tap.rope_key[rows][None, None]

    def update(self, keys, positions, *args, **kwargs):
        return torch.cat((self.keys, keys), 2), torch.cat((self.positions, positions), 2)


def check_drafter(oracle, draft, taps, features, tokens, embedding, head):
    """SGLang PR 51159 slot/fusion/FFN/correction equations, with HF eager MLA and target KV."""
    cfg, hf = oracle.official_config, oracle.official
    eps = cfg.rms_norm_eps
    aux = draft.aux

    def norm(x, w):
        return F.rms_norm(x, (x.shape[-1],), w, eps)

    end = len(tokens)
    bonus = F.linear(features[-1], head).argmax()
    width = forward.DRAFT_BLOCK_SIZE
    block = torch.full((width,), draft.config.dfly_mask_token_id, device=bonus.device)
    block[0] = bonus
    expected = embedding[block.cpu()].to(device=bonus.device, dtype=torch.float32)
    p = "mtp_fusion."
    joined = torch.cat((norm(expected[0], aux[p + "enorm.weight"]), norm(features[-1], aux[p + "hnorm.weight"])))
    expected[0] += F.linear(joined, aux[p + "eh_proj.weight"]) * aux[p + "gate"]
    actual = draft.start(bonus, features[-1], embedding)
    checks = {"fusion": assert_close("DFly2 fusion", actual, expected), "layers": []}
    pos = torch.arange(end, end + width, device=bonus.device)[None, None]
    for i, layer in enumerate(draft.layers):
        w = layer.weights
        with torch.device("meta"):
            attention = hf.HYV4FlashAttention(cfg, draft.kv_layers[i])
            mlp = hf.HYV4FlashMLP(cfg, w["mlp.gate_proj.weight"].shape[0], is_residual_sublayer=False)
        attention.compressed_indexer = None  # DFly uses external, last-target-token recall.
        attention.load_state_dict({golden.official_key(k).removeprefix("self_attn."): v
            for k, v in w.items() if k.startswith("self_attn.")}, assign=True)
        mlp.load_state_dict({k.removeprefix("mlp."): v for k, v in w.items() if k.startswith("mlp.")}, assign=True)
        # Each Hy4 DFly stage reuses the same target layer for both KV and recall.
        if draft.kv_layers[i] != draft.index_layers[i]:
            raise ValueError("preflight requires the reviewed paired KV/recall layers")
        cache = TargetCache(taps[draft.kv_layers[i]], end, draft.config.compressed_indexer_compress_ratio)
        mask = expected.new_zeros(width, cache.keys.shape[2] + width)
        mask[:, cache.keys.shape[2]:] = torch.triu(torch.full_like(mask[:, -width:], -torch.inf), diagonal=1)
        phase = oracle.rotary(expected, pos, attention.attention_type)
        values = attention(norm(expected, w["input_layernorm.weight"])[None], phase, mask[None, None], cache=cache)[0][0]
        expected = expected + values
        expected = expected + mlp(norm(expected, w["post_attention_layernorm.weight"]))
        actual = draft.layer(actual, i, taps, end)
        checks["layers"].append(assert_close(f"DFly2 layer {i}", actual, expected))
    expected = norm(expected, aux["final_norm.weight"])
    previous = bonus
    logits = []
    p = "hidden_correction."
    for hidden in expected[1:]:
        embedding_row = embedding[int(previous)].to(hidden.device, dtype=hidden.dtype)
        joined = torch.cat((norm(hidden, aux[p + "hidden_norm.weight"]), norm(embedding_row, aux[p + "embed_norm.weight"])))
        hidden = hidden + F.linear(F.silu(F.linear(joined, aux[p + "gate_proj.weight"])) *
                                   F.linear(joined, aux[p + "up_proj.weight"]), aux[p + "down_proj.weight"])
        row = F.linear(hidden, head)
        logits.append(row)
        previous = row.argmax()
    official_logits = torch.stack(logits)
    actual_logits = draft.proposals(draft.finish(actual), bonus, embedding, head_weight=head)
    checks["logits"] = assert_close("DFly2 greedy corrected logits", actual_logits, official_logits)
    if not torch.equal(actual_logits.argmax(-1), official_logits.argmax(-1)):
        raise ValueError("DFly2 greedy corrected proposals differ")
    return checks


@torch.inference_mode()
def run(args, report):
    reference.configure_f32()
    if not torch.cuda.is_available():
        raise ValueError("Hy4 reference parity requires maxq CUDA")
    reader = forward.Reader(args.source, "cuda:0")
    model = forward.Model(reader)
    oracle = golden.Family(args.source, "cuda:0")
    oracle.initialize_official()
    batches = iter(load_token_batches(args.corpus, "screening"))
    tokens = [next(batches)["input_ids"][0,-size:].to(reader.device) for size in (12, 16, 24)]
    witness = json.loads((args.corpus / "long-context-smoke.json").read_text())
    if witness["hessian_eligible"] is not False or witness["tokens"] != 2709:
        raise ValueError("missing reviewed parity-only context witness")
    long_ids = torch.tensor(witness["input_ids"], device=reader.device)
    states = [model.initial(ids) for ids in tokens]
    taps = [{} for _ in tokens]
    for index in range(recipe.LAYER_COUNT):
        layer = model.load_layer(index)
        weights = {k: v.to(reader.device) if k in reference.EXPERTS else v for k, v in layer.weights.items()}
        # Share the mapped-source transfer between both paths.
        for k in reference.EXPERTS:
            if k in weights:
                layer.weights[k] = weights[k]
        upstream = {"weights": weights, "official": None}
        errors = []
        for n, ids in enumerate(tokens):
            states[n], tap, error = check_layer(model, oracle, layer, states[n], ids, upstream,
                                                capture=index in recipe.DFLY_TARGET_LAYERS)
            if tap is not None:
                taps[n][index] = tap
            errors.append(error)
        if index in (0, recipe.SWA_LAYERS[0]):
            length = len(long_ids) if index == 0 else model.config.sliding_window + 1
            ids = long_ids[:length]
            _, tap, error = check_layer(model, oracle, layer, model.initial(ids), ids, upstream, capture=index == 0)
            if index == 0 and not tap.groups.numel():
                raise ValueError("DSA parity witness did not activate recall")
            report["context"].append(dict(layer=index, tokens=length, relative_l2=error, hessian_eligible=False))
        report["layers"].append(dict(layer=index, relative_l2=errors))
        atomic_json(args.output, report)
        print(f"preflight target {index + 1}/42 maximum relative L2 {max(errors):.3e}", flush=True)
        del layer, upstream, weights
        reader.close()
    outer = {name: reader.load(name) for name in reference.OUTER_NAMES[1:]}
    oracle.outer = outer
    features = [model.head_features(state, outer) for state in states]
    head = outer["lm_head.weight"]
    report["head"] = [assert_close("FP32 head", model.head(f, weight=head), oracle.official_final(s))
                      for f, s in zip(features, states)]
    draft = forward.Drafter(model)
    embedding = reader.raw("model.embed_tokens.weight")
    report["dfly2"] = [check_drafter(oracle, draft, t, f, ids, embedding, head)
                       for t, f, ids in zip(taps, features, tokens)]
    report.update(status="passed", prompts=[len(x) for x in tokens])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "corpus", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("use a fresh parity receipt path")
    paths = [Path(__file__), Path(forward.__file__), Path(reference.__file__), Path(golden.__file__)]
    report = dict(schema="cuteafd-hy4-forward-parity-v2", status="running", layers=[], context=[],
                  source_revision=recipe.BASE_REVISION, reference_revision=sources.TRANSFORMERS_REVISION,
                  dfly_reference_revision=sources.SGLANG_REVISION, relative_l2_limit=golden.REL_LIMIT,
                  corpus_manifest_sha256=hashlib.sha256((args.corpus / "manifest.json").read_bytes()).hexdigest(),
                  code_identity={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    start = time.monotonic()
    try:
        run(args, report)
    except Exception as e:
        report.update(status="failed", error=f"{type(e).__name__}: {e}")
        raise
    finally:
        report["seconds"] = time.monotonic() - start
        atomic_json(args.output, report)


if __name__ == "__main__":
    main()
