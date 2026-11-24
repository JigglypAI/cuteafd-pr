#!/usr/bin/env python3
"""Compare against pinned HF stock and an expert-activation-matched HF control."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import MethodType

import torch
import torch.nn.functional as F

from golden import HF, open_snapshot, start_f32
from families.inkling2.model import capped_swiglu, head_forward, quantize_mxfp8
from shape_invariant import linear as fixed_linear
from families.inkling2_mini.model import TEXT, Experts, forward, quantize_nvfp4

PROMPTS = ["Explain why the sky looks blue at midday.",
           "Write a Python function that performs binary search on a sorted list.",
           "Prove that the sum of the first n positive odd integers is n squared."]


def unpack_hf_expert(layer, expert):
    from transformers.core_model_loading import Inkling2LoopDequantizeExperts, Interleave

    class F32Target:
        def get_parameter(self, _name):
            return torch.empty((), device="meta", dtype=torch.float32)

    unpack = Inkling2LoopDequantizeExperts()
    interleave = Interleave(dim=1)
    matrices = []
    for prefix in ("w13", "w2"):
        key = f"mlp.experts.{prefix}_weight"
        terms = {"packed": layer[key][expert:expert + 1],
                 "scale": layer[key + "_scale"][expert:expert + 1]}
        if terms["packed"].dtype == torch.uint8:
            terms.update(amax=layer[key + "_amax"][expert:expert + 1],
                         denominator=layer[key + "_outer_scale_denominator"])
        projection = unpack.convert(terms, ("packed", "scale", "amax", "denominator"), ("out",),
                                    model=F32Target(), full_layer_name="out")
        if prefix == "w13":
            projection = interleave.convert(projection, ("out",), ("out",))
        matrices.append(projection["out"][0])
    return matrices


def native_expert_control(qdq, layer, observe=None):
    def run(self, hidden, ids, weights):
        result = torch.zeros_like(hidden, dtype=torch.float32)
        source = qdq(hidden)
        for i in ids.unique().tolist():
            row, slot = torch.where(ids == i)
            up_weight, down_weight = unpack_hf_expert(layer, i)
            gate_up = F.linear(source[row], up_weight).to(hidden.dtype).float()
            y = (self._apply_gate(gate_up) * weights[row, slot, None]).to(hidden.dtype)
            down_input = qdq(y)
            down = F.linear(down_input, down_weight).to(hidden.dtype)
            if observe is not None:
                observe(i, row, source[row], gate_up, y, down_input, down)
            result.index_add_(0, row, down.float())
        return result.to(hidden.dtype)
    return run


def compare(a, b):
    delta = (a - b).abs()
    agreed = (a.argmax(-1) == b.argmax(-1)).sum().item()
    return {"max_abs": delta.max().item(), "mean_abs": delta.mean().item(),
            "top1_equal": agreed, "rows": a.shape[0]}


def trace_comparison(path, model, weights, cfg, ids, oracle):
    hf, handles = {}, []
    for i, layer in enumerate(model.model.language_model.layers):
        for name, sub in (("attention", layer.self_attn), ("post_attention_conv", layer.post_attention_convolution),
                          ("expert_input", layer.mlp), ("expert_output", layer.mlp),
                          ("post_mlp_conv", layer.post_mlp_convolution), ("output", layer)):
            def save(_module, args, output, i=i, name=name):
                value = args[0] if name == "expert_input" else (output[0] if isinstance(output, tuple) else output)
                hf.setdefault((i, name), []).append(value[0].detach().cpu())
            handles.append(sub.register_forward_hook(save))
    try:
        model(input_ids=ids[None], use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    visits = [(i, 0) for i in range(cfg.core_start)] + [
        (i, p) for p in range(cfg.num_passes) for i in range(cfg.core_start, cfg.core_end)] + [
        (i, 0) for i in range(cfg.core_end, cfg.num_hidden_layers)]
    results = []
    for i, p in visits:
        observed = oracle[cfg.cache_slot(i, p)]
        results.append({"layer": i, "pass": p, "tensors": {
            name: compare(value.float(), hf[i, name][p].float())
            for name, value in observed.items() if name != "input"}})

    # Give the oracle the exact activation entering HF's first expert block to
    # separate a real QDQ/weight discrepancy from upstream hidden drift.
    first = weights.load_layer(0)
    input_hf = hf[0, "expert_input"][0].to(weights.device)
    native = Experts(first, cfg)
    matched = native(input_hf).cpu()
    hf_layer = model.model.language_model.layers[0].mlp
    _, gamma, indices = hf_layer.gate(input_hf)
    stages = {}

    def compare_expert(e, rows, fc1_input, fc1_output, activation, fc2_input, fc2_output):
        w13, w2 = native.decoded[e]
        gamma_e = gamma[rows, (indices[rows] == e).long().argmax(-1)]
        xq = native.qdq[0](input_hf)[rows]
        z = fixed_linear(xq, w13).to(input_hf.dtype).float()
        y = capped_swiglu(z, gamma_e, cfg.mlp_gate_softcap, cfg.mlp_up_softcap).to(input_hf.dtype)
        yq = native.qdq[1](y)
        down = fixed_linear(yq, w2).to(input_hf.dtype)
        z = torch.cat((z[:, ::2], z[:, 1::2]), -1)
        for name, left, right in zip(("fc1_input", "fc1_output", "activation", "fc2_input", "fc2_output"),
                                    (xq, z, y, yq, down),
                                    (fc1_input, fc1_output, activation, fc2_input, fc2_output), strict=True):
            stat = stages.setdefault(name, {"max_abs": 0., "different_elements": 0})
            stat["max_abs"] = max(stat["max_abs"], (left.float() - right.float()).abs().max().item())
            stat["different_elements"] += int((left != right).count_nonzero())

    replay = native_expert_control(quantize_nvfp4, first, compare_expert)(
        hf_layer.experts, input_hf, indices, gamma).cpu()
    weight_witness = {}
    for part, exact in enumerate(unpack_hf_expert(first, 0)):
        source = native.weight(part, 0)
        if part == 0:
            source = torch.cat((source[::2], source[1::2]), dim=0)
        weight_witness[str(part)] = {"native_converter_max_abs": (source - exact).abs().max().item()}
    out = {"visits": results, "first_expert_same_input": compare(matched.float(), replay.float()),
           "first_expert_replay_vs_trace": compare(replay.float(), hf[0, "expert_output"][0].float()),
           "first_expert_same_input_stages": stages, "first_expert_weight": weight_witness}
    path.write_text(json.dumps(out, indent=2) + "\n")
    torch.save({"oracle": oracle, "hf": hf}, path.with_suffix(".pt"))


@torch.inference_mode()
def main():
    import transformers
    from transformers import AutoConfig, Inkling2LoopForConditionalGeneration, PreTrainedTokenizerFast

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--trace", type=Path, help="save one prompt's 48 visit and sublayer comparison")
    a = p.parse_args()
    if transformers.__version__ != HF["version"]:
        raise ValueError(f"expected Transformers {HF['version']}")
    start_f32()
    cfg, weights = open_snapshot(a.snapshot, a.device)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(a.snapshot, local_files_only=True)
    batches = [torch.tensor(tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], add_generation_prompt=True, return_dict=False), device=a.device)
        for text in (PROMPTS[:1] if a.trace else PROMPTS)]
    head = weights.tensor(TEXT + "unembed.weight").float()
    oracle_trace = {}
    def trace_oracle(_row, slot, name, value):
        oracle_trace.setdefault(slot, {})[name] = value.detach().cpu()
    native = [head_forward(h, head, cfg).cpu() for h in forward(weights, cfg, batches,
              trace=trace_oracle if a.trace else None)]
    unquantized = [cfg.final_logit_softcapping *
                   (F.linear(h, head[:cfg.unpadded_vocab_size].to(h.dtype)).float().cpu()
                    / cfg.final_logit_softcapping).tanh()
                   for h in forward(weights, cfg, batches, native=False)]
    del head
    weights.close()
    if torch.device(a.device).type == "cuda":
        torch.cuda.empty_cache()
    config = AutoConfig.from_pretrained(a.snapshot, local_files_only=True)
    # Use HF's built-in mixed-expert converters, not the unrelated generic
    # ModelOpt NVFP4 loader indicated by the legacy compatibility marker.
    del config.quantization_config
    # Async copying eagerly materializes every large expert before conversion.
    os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
    # Expanded expert weights alone exceed 90 GiB. Accelerate's hooks
    # page in one expert bank at a time, without changing its F32 math.
    placement = {"": a.device}
    offload = None
    if torch.device(a.device).type == "cuda":
        offload = a.out.parent / "hf-offload"
        # A root device entry recursively moves the still-meta offloaded
        # descendants before their hooks run. Map resident leaves explicitly.
        placement = {"lm_head": a.device, "model.vision_tower": "disk"}
        text = "model.language_model"
        for name in ("embed_tokens", "norm", "final_input", "message_input"):
            placement[f"{text}.{name}"] = a.device
        for i in range(cfg.num_hidden_layers):
            layer = f"{text}.layers.{i}"
            if i % 2:
                placement[layer] = a.device
                continue
            for name in ("attention_input", "mlp_input", "self_attn", "input_layernorm",
                         "q_input_layernorm", "k_input_layernorm", "v_input_layernorm",
                         "post_attention_layernorm", "post_attention_convolution", "post_mlp_convolution",
                         "q_message_gate", "k_message_gate", "v_message_gate", "mlp_message_gate", "mlp.gate"):
                placement[f"{layer}.{name}"] = a.device
            placement[f"{layer}.mlp.experts"] = "disk"
    model, info = Inkling2LoopForConditionalGeneration.from_pretrained(
        a.snapshot, config=config, dtype=torch.bfloat16, device_map=placement,
        offload_folder=offload, local_files_only=True, attn_implementation="eager",
        experts_implementation="eager", output_loading_info=True)
    if any(info.get(k) for k in ("missing_keys", "mismatched_keys", "error_msgs", "conversion_errors")):
        raise ValueError(f"HF didn't load the checkpoint: {info}")
    model.eval()
    stock = [model(input_ids=ids[None], use_cache=False).logits[0].cpu() for ids in batches]
    cache_evidence = []
    for ids, baseline in zip(batches, stock, strict=True):
        prefix = model(input_ids=ids[None, :-1], use_cache=True)
        continuation = model(input_ids=ids[None, -1:], cache=prefix.cache, use_cache=True)
        slots = continuation.cache.layers
        expected = cfg.num_hidden_layers + (cfg.num_passes - 1) * (cfg.core_end - cfg.core_start)
        if len(slots) != expected or len({id(slot) for slot in slots}) != expected:
            raise ValueError("unexpected HF loop cache slot identity")
        cache_evidence.append({"slots": len(slots), **compare(continuation.logits[0].cpu(), baseline[-1:])})
    for i, layer in enumerate(model.model.language_model.layers):
        fmt = weights.formats[TEXT + f"layers.{i}.mlp.experts.w13_weight"]
        qdq = quantize_nvfp4 if fmt == "nvfp4" else quantize_mxfp8
        expert = layer.mlp.experts
        packed = {k: v for k, v in weights.load_layer(i).items() if k.startswith("mlp.experts.")}
        implementation = MethodType(native_expert_control(qdq, packed), expert)
        if hasattr(expert, "_hf_hook"):
            expert._old_forward = implementation
        else:
            expert.forward = implementation
    if a.trace:
        trace_comparison(a.trace, model, weights, cfg, batches[0], oracle_trace)
    model.lm_head.float()
    head_cast = model.lm_head.register_forward_pre_hook(lambda _module, args: (args[0].float(),))
    try:
        matched = [model(input_ids=ids[None], use_cache=False).logits[0].cpu() for ids in batches]
    finally:
        head_cast.remove()
    checks = [{"prompt": text, "native_vs_hf_qdq_control": compare(ours, adapted),
               "native_vs_hf_stock": compare(ours, raw), "without_qdq_vs_hf_stock": compare(no_qdq, raw),
               "hf_stock_cached_last": cached}
              for text, ours, adapted, raw, no_qdq, cached in zip(PROMPTS[:len(batches)], native, matched, stock, unquantized,
                                                                cache_evidence, strict=True)]
    result = {"comparison": "Stock HF dequantizes the experts and does not QDQ their activations; matched control decodes original packed experts to F32 with HF's converter, applies native activation QDQ, and upcasts the head. Dense compute is BF16 with HF F32 islands.",
              "upstream": HF, "revision": a.snapshot.name, "prompts": checks,
              "passed": all(c["native_vs_hf_qdq_control"]["max_abs"] <= .25
                            and c["native_vs_hf_qdq_control"]["top1_equal"] == c["native_vs_hf_qdq_control"]["rows"]
                            and c["without_qdq_vs_hf_stock"]["max_abs"] <= .02
                            and c["hf_stock_cached_last"]["top1_equal"] == 1 for c in checks)}
    a.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    if not result["passed"]:
        raise SystemExit("HF F32 comparison failed")


if __name__ == "__main__":
    main()
