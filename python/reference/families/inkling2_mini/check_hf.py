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
from families.inkling2.model import head_forward, quantize_mxfp8
from families.inkling2_mini.model import TEXT, forward, quantize_nvfp4

PROMPTS = ["Explain why the sky looks blue at midday.",
           "Write a Python function that performs binary search on a sorted list.",
           "Prove that the sum of the first n positive odd integers is n squared."]


def native_expert_control(qdq):
    def run(self, hidden, ids, weights):
        result = torch.zeros_like(hidden, dtype=torch.float32)
        source = qdq(hidden)
        for i in ids.unique().tolist():
            row, slot = torch.where(ids == i)
            gate_up = F.linear(source[row], self.gate_up_proj[i])
            y = self._apply_gate(gate_up) * weights[row, slot, None]
            result.index_add_(0, row, F.linear(qdq(y), self.down_proj[i]))
        return result
    return run


def compare(a, b):
    delta = (a - b).abs()
    agreed = (a.argmax(-1) == b.argmax(-1)).sum().item()
    return {"max_abs": delta.max().item(), "mean_abs": delta.mean().item(),
            "top1_equal": agreed, "rows": a.shape[0]}


@torch.inference_mode()
def main():
    import transformers
    from transformers import AutoConfig, Inkling2LoopForConditionalGeneration, PreTrainedTokenizerFast

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    if transformers.__version__ != HF["version"]:
        raise ValueError(f"expected Transformers {HF['version']}")
    start_f32()
    cfg, weights = open_snapshot(a.snapshot, a.device)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(a.snapshot, local_files_only=True)
    batches = [torch.tensor(tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], add_generation_prompt=True, return_dict=False), device=a.device)
        for text in PROMPTS]
    head = weights.tensor(TEXT + "unembed.weight").float()
    native = [head_forward(h, head, cfg).cpu() for h in forward(weights, cfg, batches)]
    unquantized = [head_forward(h, head, cfg).cpu() for h in forward(weights, cfg, batches, native=False)]
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
        placement["model.vision_tower"] = "disk"
        placement.update({f"model.language_model.layers.{i}.mlp.experts": "disk"
                          for i in range(0, cfg.num_hidden_layers, 2)})
    model, info = Inkling2LoopForConditionalGeneration.from_pretrained(
        a.snapshot, config=config, dtype=torch.float32, device_map=placement,
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
        implementation = MethodType(native_expert_control(qdq), expert)
        if hasattr(expert, "_hf_hook"):
            expert._old_forward = implementation
        else:
            expert.forward = implementation
    matched = [model(input_ids=ids[None], use_cache=False).logits[0].cpu() for ids in batches]
    checks = [{"prompt": text, "native_vs_hf_qdq_control": compare(ours, adapted),
               "native_vs_hf_stock": compare(ours, raw), "without_qdq_vs_hf_stock": compare(no_qdq, raw),
               "hf_stock_cached_last": cached}
              for text, ours, adapted, raw, no_qdq, cached in zip(PROMPTS, native, matched, stock, unquantized,
                                                                cache_evidence, strict=True)]
    result = {"comparison": "Stock HF dequantizes the experts and does not QDQ their activations; matched control changes only HF expert forward. Both use F32.",
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
