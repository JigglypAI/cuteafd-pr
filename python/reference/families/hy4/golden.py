#!/usr/bin/env python3
"""Layer-major Hy4 F32 golden; qualify against the pinned HF text forward first."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "python/reference"))
import layer_major
from families import hy4_common
from families.hy4 import model, sources

CONFIG_SHA256 = "dab00b0f4c9265f650eabccc1033bd6aa29080cf1459856a9f2a06ae074d7714"
INDEX_SHA256 = "8433cb5cfa21cf7dda97139c4f3aba58682063ef95df7b4105674284872041d6"
REL_LIMIT = 1e-4
ROPE_STYLE = "interleaved"
QA_KVA_EPS = 1e-6


def expected_source_identity(_runtime=None):
    return {"checkpoint": sources.CHECKPOINT, "revision": sources.REVISION,
            "transformers_revision": sources.TRANSFORMERS_REVISION,
            "transformers_files": sources.TRANSFORMERS_FILES,
            "config_sha256": CONFIG_SHA256, "index_sha256": INDEX_SHA256,
            "common_sha256": layer_major.digest(hy4_common.__file__),
            "model_sha256": layer_major.digest(model.__file__),
            "golden_sha256": layer_major.digest(__file__),
            "sources_sha256": layer_major.digest(sources.__file__),
            "layer_major_sha256": layer_major.digest(layer_major.__file__)}


def load_pinned_transformers():
    import transformers
    root = Path(transformers.__file__).resolve().parent
    for name, digest in sources.TRANSFORMERS_FILES.items():
        if layer_major.digest(root / name) != digest:
            raise ValueError(f"not the pinned Transformers source: {name}")
    from transformers.models.hy_v4 import modeling_hy_v4 as hf
    return hf


def apply_interleaved_rope_to_hf(hf):
    """Patch pinned Transformers main split-half RoPE to adjacent-pair rotation (PR #50589)."""
    import torch

    def apply_rotary_pos_emb_interleave(q, k, cos, sin, unsqueeze_dim=1):
        cos = cos[..., :cos.shape[-1] // 2].unsqueeze(unsqueeze_dim)
        sin = sin[..., :sin.shape[-1] // 2].unsqueeze(unsqueeze_dim)
        q1, q2 = q[..., 0::2], q[..., 1::2]
        k1, k2 = k[..., 0::2], k[..., 1::2]
        q_embed = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
        k_embed = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)
        return q_embed, k_embed

    hf.apply_rotary_pos_emb = apply_rotary_pos_emb_interleave


def official_key(name):
    for old, new in (("self_attn.learnable_sink_param", "self_attn.sinks"),
                     ("self_attn.linear_gate.", "self_attn.gate_proj."),
                     ("hc_attn_layer.hc_pre.hc_", "attn_hc."), ("hc_mlp_layer.hc_pre.hc_", "ffn_hc.")):
        if name.startswith(old):
            return new + name[len(old):]
    return name


class Family:
    name = "hy4"
    compare_official_head = True

    def __init__(self, snapshot, device):
        snapshot = Path(snapshot)
        if (snapshot.name != sources.REVISION or layer_major.digest(snapshot / "config.json") != CONFIG_SHA256
                or layer_major.digest(snapshot / "model.safetensors.index.json") != INDEX_SHA256):
            raise ValueError("Hy4 oracle requires the pinned original BF16 snapshot")
        self.raw_config = json.loads((snapshot / "config.json").read_bytes())
        self.config = model.Config.parse(self.raw_config)
        self.layers = self.config.num_hidden_layers
        self.checkpoint = model.Hy4Checkpoint(snapshot, device)
        self.device = self.checkpoint.device
        self.outer = self.official = self.official_config = self.rotary = None
        self.official_mode = False
        self.upstream_preflight = None

    def initialize_official(self, *, rope_style=ROPE_STYLE):
        import torch
        hf = self.official = load_pinned_transformers()
        if rope_style == "interleaved":
            apply_interleaved_rope_to_hf(hf)
        text_dict = dict(self.raw_config["text_config"])
        # Filter keys accepted by strict HYV4Config; layer_types in HF is indexed_attention.
        text_dict["layer_types"] = ["indexed_attention"] * self.layers
        allowed = set(hf.HYV4Config.__dataclass_fields__) if hasattr(hf.HYV4Config, "__dataclass_fields__") else None
        if allowed is not None:
            text_dict = {k: v for k, v in text_dict.items() if k in allowed}
        c = self.official_config = hf.HYV4Config.from_dict(text_dict)
        c._attn_implementation = "eager"
        c._experts_implementation = "eager"
        with torch.device(self.device):
            self.rotary = hf.HYV4RotaryEmbedding(c)

    configure = staticmethod(model.configure_f32)
    provenance = staticmethod(expected_source_identity)
    observe = staticmethod(layer_major.no_observation)

    def tensors(self, index):
        return model.tensor_contract(self.config, index)

    def check_layer(self, index):
        self.checkpoint.check(self.tensors(index))

    def check_all(self):
        outer = model.tensor_contract(self.config)
        known = set(outer)
        self.checkpoint.check(outer)
        for index in range(self.layers):
            self.check_layer(index)
            known.update(self.tensors(index))
        indexed = {n for n in self.checkpoint.index
                   if n.startswith(("model.layers.", "model.hc_head.")) or n in model.OUTER_NAMES}
        if known != indexed:
            raise ValueError(f"unexpected Hy4 text inventory: {sorted(indexed ^ known)}")
        return len(known)

    def layer_bytes(self, index):
        import math
        return sum(math.prod(shape) * (2 if name.endswith(model.EXPERTS) else 4)
                   for name, (shape, _) in self.tensors(index).items())

    def state_bytes(self, tokens):
        # Hidden 4-stream F32 state plus optional IndexCache top-k int32 carry.
        topk_bytes = tokens * min(tokens, self.config.index_topk) * 4 if tokens > self.config.index_topk else 0
        return tokens * self.config.hidden_size * self.config.hc_mult * 4 + topk_bytes

    def begin_layer_memory(self):
        if self.device.type == "cuda":
            import torch
            torch.cuda.reset_peak_memory_stats(self.device)

    def end_layer_memory(self, stage, index):
        if self.device.type != "cuda":
            return
        import torch
        torch.cuda.empty_cache()
        torch.cuda.synchronize(self.device)
        available = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()
                         if line.startswith("MemAvailable:"))
        allocated = torch.cuda.max_memory_allocated(self.device)
        reserved = torch.cuda.max_memory_reserved(self.device)
        gib = 2 ** 30
        print(f"Hy4 {stage} layer {index}: CUDA peak allocated={allocated/gib:.3f} GiB "
              f"reserved={reserved/gib:.3f} GiB MemAvailable={available/gib:.3f} GiB", flush=True)
        if available < 12 * gib:
            raise RuntimeError(f"Hy4 MemAvailable guard stopped at {stage} layer {index}; no next layer launched")

    def work_bytes(self, tokens):
        # Q/G/K/V, residuals, tiled score/probability/value output and indexer scratch.
        c = self.config
        heads = c.num_attention_heads
        width = max(c.qk_nope_head_dim + c.qk_rope_head_dim, c.v_head_dim)
        return 4 * (tokens * (heads * width * 8 + c.hc_mult * c.hidden_size * 3)
                    + heads * model.ROWS * tokens * 4 + c.index_n_heads * model.ROWS * tokens)

    def outer_tensors(self, stage):
        return model.OUTER_NAMES[:1] if stage == "embed" else model.OUTER_NAMES[1:]

    def embed(self, tokens):
        import torch
        import torch.nn.functional as F
        if self.outer is None:
            self.outer = self.checkpoint.load(model.OUTER_NAMES[0])
            self.checkpoint.close()
        ids = torch.as_tensor(tokens, dtype=torch.long, device=self.device)
        embedding = F.embedding(ids, self.outer)
        return embedding[:, None].expand(-1, self.config.hc_mult, -1).unsqueeze(0).contiguous()

    def release_outer(self):
        self.outer = None

    def load_layer(self, index, *, stream_experts=False):
        return {"weights": self.checkpoint.layer(self.config, index, stream_experts=stream_experts), "official": None}

    def release_layer(self, module):
        self.checkpoint.close()

    def geometry(self):
        return nullcontext()

    def official_geometry(self):
        @contextmanager
        def select():
            self.official_mode = True
            try:
                yield
            finally:
                self.official_mode = False
        return select()

    def official_layer(self, module, index):
        import torch
        hf, c, w = self.official, self.official_config, module["weights"]
        if module["official"] is not None:
            return module["official"]
        with torch.device("meta"):
            layer = hf.HYV4DecoderLayer(c, index)
        if hasattr(layer.mlp, "experts"):
            from torch.nn import Parameter

            class ExpertMatrices(Parameter):
                def __getitem__(self, key):
                    return super().__getitem__(key).to(self.target_device, dtype=torch.float32)

            for key in model.EXPERTS:
                expert = key.rsplit(".", 1)[1]
                param = ExpertMatrices(w[key], requires_grad=False)
                param.target_device = self.device
                setattr(layer.mlp.experts, expert, param)
        state = {official_key(key): value for key, value in w.items() if key not in model.EXPERTS}
        result = layer.load_state_dict(state, strict=False, assign=True)
        expected_missing = set(model.EXPERTS) & w.keys()
        if set(result.missing_keys) != expected_missing or result.unexpected_keys:
            raise ValueError(f"Hy4 Transformers state mapping differs: {result}")
        module["official"] = layer.eval()
        return module["official"]

    def official_forward(self, module, index, hidden, prev_topk_indices=None):
        import torch
        from transformers.masking_utils import create_causal_mask
        layer = self.official_layer(module, index)
        seq_len = hidden.shape[1]
        positions = torch.arange(seq_len, device=self.device).reshape(1, -1)
        mask = create_causal_mask(config=self.official_config, inputs_embeds=hidden[:, :, 0], attention_mask=None,
                                  cache=None, position_ids=positions, allow_is_causal_skip=False)
        phase_emb = self.rotary(hidden, positions)
        prev = prev_topk_indices.unsqueeze(0) if prev_topk_indices is not None else None
        out, topk = layer(hidden, attention_mask=mask, position_ids=positions,
                          position_embeddings=phase_emb, prev_topk_indices=prev)
        return out, topk[0] if topk is not None else None

    def forward(self, module, index, hidden, carry):
        prev_topk = carry.get("topk_indices")
        next_shared = (index + 1 < self.layers) and (self.config.indexer_types[index + 1] == "shared")
        is_shared = self.config.indexer_types[index] == "shared"
        if self.official_mode:
            result, topk = self.official_forward(module, index, hidden, prev_topk)
        else:
            out, topk = model.layer_forward(
                hidden[0], module["weights"], self.config, index,
                prev_topk_indices=prev_topk, rope_style=ROPE_STYLE, qa_kva_eps=QA_KVA_EPS,
            )
            result = out.unsqueeze(0)
        if next_shared and topk is not None:
            return result, {"topk_indices": topk}, (("topk_indices",) if is_shared else ())
        return result, {}, ()

    def load_final(self):
        self.outer = {name: self.checkpoint.load(name) for name in self.outer_tensors("final")}
        self.checkpoint.close()

    def release_final(self):
        self.release_outer()

    def final(self, hidden):
        import torch
        features = model.head_features(hidden, self.outer, self.config)
        # Only a fixed tile of logits lives on device; a long context can score >1 GiB.
        return torch.cat([model.linear(rows, self.outer[model.OUTER_NAMES[5]]).cpu()
                          for rows in features.split(model.ROWS)])

    def official_final(self, hidden):
        import torch
        import torch.nn.functional as F
        w = self.outer
        with torch.device("meta"):
            hc = self.official.HYV4HyperHead(self.official_config)
            norm = self.official.HYV4RMSNorm(self.config.hidden_size, self.config.rms_norm_eps)
        hc.load_state_dict(dict(zip(("hc_fn", "hc_base", "hc_scale"), (w[n] for n in model.OUTER_NAMES[1:4]))), assign=True)
        norm.load_state_dict({"weight": w[model.OUTER_NAMES[4]]}, assign=True)
        return F.linear(norm(hc(hidden.unsqueeze(0)))[0], w[model.OUTER_NAMES[5]])

    def golden_meta(self):
        gates = {"gates": {"upstream_preflight": self.upstream_preflight}} if self is not None and self.upstream_preflight else {}
        return {**gates, "reference": "Independent Hy4 layer-major IEEE F32 text reference",
                "source_identity_provider": "families.hy4.golden",
                "root_arithmetic": sources.ROOT_ARITHMETIC}


def rel_l2(actual, expected):
    if not actual.isfinite().all() or not expected.isfinite().all():
        return float("inf")
    return float((actual - expected).norm() / expected.norm().clamp_min(1e-30))


def rope_ab(family, snapshot, out):
    """Settle split-half vs adjacent-pair RoPE (MLA + indexer) and q_a/kv_a norm eps with real-weights NLL."""
    import torch
    import torch.nn.functional as F
    from checkpoint_tokenizer import load_tokenizer
    tokenizer = load_tokenizer(snapshot / "tokenizer.json", chat=True)
    texts = (
        "In mathematics, the sum of the first n odd positive integers equals n squared. "
        "We prove this by mathematical induction. Base case: for n = 1, the sum is 1 = 1^2. "
        "Inductive step: assume 1 + 3 + ... + (2k - 1) = k^2; adding (2k + 1) yields (k + 1)^2.",
        "def binary_search(nums, target):\n"
        "    lo, hi = 0, len(nums) - 1\n"
        "    while lo <= hi:\n"
        "        mid = lo + (hi - lo) // 2\n"
        "        if nums[mid] == target:\n"
        "            return mid\n"
        "        elif nums[mid] < target:\n"
        "            lo = mid + 1\n"
        "        else:\n"
        "            hi = mid - 1\n"
        "    return -1\n",
        "二分查找的循环不变式是：如果目标值存在于已排序数组中，那么它一定位于当前搜索区间 [lo, hi] 之内。",
    )
    token_lists = [tokenizer.encode(t, add_special_tokens=False) for t in texts]
    # Also build a >2048-token prompt from the checkpoint's own README.md so the DSA indexer
    # is actively pruning keys (positions >= 2048) and IndexCache reuse is exercised.
    readme_ids = tokenizer.encode((snapshot / "README.md").read_text(), add_special_tokens=False)
    long_ids = readme_ids[:family.config.index_topk + 256]
    assert len(long_ids) == family.config.index_topk + 256

    short_arms = {
        "interleaved_eps1e6": ("interleaved", "interleaved", 1e-6),
        "split_half_eps1e6": ("split_half", "split_half", 1e-6),
        "interleaved_eps1e5": ("interleaved", "interleaved", 1e-5),
    }
    long_arms = {
        "mla_interleaved_idx_interleaved": ("interleaved", "interleaved", 1e-6),
        "mla_interleaved_idx_split_half": ("interleaved", "split_half", 1e-6),
    }
    states = {arm: [family.embed(ids)[0] for ids in token_lists] for arm in short_arms}
    long_states = {arm: family.embed(long_ids)[0] for arm in long_arms}
    family.release_outer()
    topks = {arm: [None] * len(token_lists) for arm in short_arms}
    long_topks = {arm: None for arm in long_arms}
    for index in range(family.layers):
        family.begin_layer_memory()
        module = family.load_layer(index, stream_experts=False)
        w = module["weights"]
        for arm, (rope_style, idx_style, qa_eps) in short_arms.items():
            for j in range(len(token_lists)):
                states[arm][j], topks[arm][j] = model.layer_forward(
                    states[arm][j], w, family.config, index,
                    prev_topk_indices=topks[arm][j],
                    rope_style=rope_style, indexer_rope_style=idx_style, qa_kva_eps=qa_eps,
                )
        for arm, (rope_style, idx_style, qa_eps) in long_arms.items():
            long_states[arm], long_topks[arm] = model.layer_forward(
                long_states[arm], w, family.config, index,
                prev_topk_indices=long_topks[arm],
                rope_style=rope_style, indexer_rope_style=idx_style, qa_kva_eps=qa_eps,
            )
        family.release_layer(module)
        del module, w
        if (index + 1) % 10 == 0 or index + 1 == family.layers:
            print(f"Hy4 RoPE A/B layer {index + 1}/{family.layers} done", flush=True)
        family.end_layer_memory("rope-ab", index)
    family.load_final()
    summary = {"prompts": [len(ids) for ids in token_lists], "long_prompt_tokens": len(long_ids),
               "arms": {}, "indexer_long_arms": {}}
    logits_cache = {}
    for arm, (rope_style, idx_style, qa_eps) in short_arms.items():
        per_prompt = []
        tot_nll, tot_tok, tot_top1 = 0.0, 0, 0
        logits_cache[arm] = []
        for j, ids in enumerate(token_lists):
            logits = family.final(states[arm][j])
            logits_cache[arm].append(logits)
            targets = torch.tensor(ids[1:], dtype=torch.long)
            pred = logits[:-1]
            nll = float(F.cross_entropy(pred, targets, reduction="mean"))
            top1 = float((pred.argmax(dim=-1) == targets).float().mean())
            per_prompt.append({"tokens": len(ids), "scored": len(targets), "nll": nll, "top1": top1})
            tot_nll += nll * len(targets)
            tot_top1 += top1 * len(targets)
            tot_tok += len(targets)
        summary["arms"][arm] = {
            "rope_style": rope_style, "indexer_rope_style": idx_style, "qa_kva_eps": qa_eps,
            "mean_nll": tot_nll / tot_tok, "mean_top1": tot_top1 / tot_tok,
            "scored_tokens": tot_tok, "per_prompt": per_prompt,
        }
    tail_start = family.config.index_topk
    long_targets = torch.tensor(long_ids[tail_start:], dtype=torch.long)
    for arm, (rope_style, idx_style, qa_eps) in long_arms.items():
        # Score the 256 tail positions [2048..2303] where the DSA indexer actively prunes keys.
        logits = family.final(long_states[arm][tail_start - 1:-1])
        nll = float(F.cross_entropy(logits, long_targets, reduction="mean"))
        top1 = float((logits.argmax(dim=-1) == long_targets).float().mean())
        summary["indexer_long_arms"][arm] = {
            "rope_style": rope_style, "indexer_rope_style": idx_style, "qa_kva_eps": qa_eps,
            "tail_start": tail_start, "scored_tokens": len(long_targets), "tail_nll": nll, "tail_top1": top1,
        }
    eps_delta = [rel_l2(a, b) for a, b in zip(logits_cache["interleaved_eps1e6"], logits_cache["interleaved_eps1e5"])]
    summary["eps_1e6_vs_1e5_interleaved_rel_l2"] = eps_delta
    family.release_final()
    layer_major.atomic_json(out, summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def preflight(family, snapshot, out):
    """Compare independent streams end-to-end, and each layer on captured oracle inputs."""
    import torch
    from checkpoint_tokenizer import load_tokenizer
    tokenizer = load_tokenizer(snapshot / "tokenizer.json", chat=True)
    prompts = ("Prove the sum of the first n odd numbers is n squared.",
               "Summarize a safe transaction rollback strategy for a ledger.",
               "Explain the loop invariant of binary search in Chinese.")
    token_lists = [tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                                 add_generation_prompt=True) for p in prompts]
    tokens = [torch.tensor(tokenizer.encode(t, add_special_tokens=False), device=family.device) for t in token_lists]
    # Force an active DSA indexer + IndexCache reuse comparison (>2048 tokens) on full layer 1 and shared layer 2.
    long_len = family.config.index_topk + 17
    long_tokens = tokens[0].repeat(1 + long_len // len(tokens[0]))[:long_len]
    adapted = [family.embed(t.tolist()) for t in tokens]
    native = [a.clone() for a in adapted]
    long_hidden = family.embed(long_tokens.tolist())
    family.release_outer()
    adapted_topk = [None] * len(tokens)
    native_topk = [None] * len(tokens)
    long_native_topk = long_adapted_topk = None
    checks = []
    truncated = None
    for index in range(family.layers):
        family.begin_layer_memory()
        module = family.load_layer(index, stream_experts=index not in (1, 2))
        local = []
        for j in range(len(tokens)):
            golden_input = native[j]
            golden_prev_topk = native_topk[j]
            native[j], native_topk[j] = family.official_forward(module, index, golden_input, golden_prev_topk)
            same, _ = model.layer_forward(
                golden_input[0], module["weights"], family.config, index,
                prev_topk_indices=golden_prev_topk, rope_style=ROPE_STYLE, qa_kva_eps=QA_KVA_EPS,
            )
            local.append(rel_l2(same.unsqueeze(0), native[j]))
            adapted_out, adapted_topk[j] = model.layer_forward(
                adapted[j][0], module["weights"], family.config, index,
                prev_topk_indices=adapted_topk[j], rope_style=ROPE_STYLE, qa_kva_eps=QA_KVA_EPS,
            )
            adapted[j] = adapted_out.unsqueeze(0)
        evidence = {"layer": index, "indexer_type": family.config.indexer_types[index],
                    "same_input_relative_l2": local,
                    "trajectory_relative_l2": [rel_l2(a, n) for a, n in zip(adapted, native)]}
        if index in (1, 2):
            # Layer 1 is full MoE+indexer; layer 2 is shared IndexCache reuse.
            g, long_native_topk = family.official_forward(module, index, long_hidden, long_native_topk)
            h, long_adapted_topk = model.layer_forward(
                long_hidden[0], module["weights"], family.config, index,
                prev_topk_indices=long_adapted_topk, rope_style=ROPE_STYLE, qa_kva_eps=QA_KVA_EPS,
            )
            evidence["active_indexer_relative_l2"] = rel_l2(h.unsqueeze(0), g)
            long_hidden = g
            del h
            if index == 2:
                del long_hidden, long_native_topk, long_adapted_topk
                long_native_topk = long_adapted_topk = None
        checks.append(evidence)
        if max(evidence["same_input_relative_l2"] + [evidence.get("active_indexer_relative_l2", 0.)]) > REL_LIMIT:
            layer_major.atomic_json(out, {"passed": False, "relative_l2_max": REL_LIMIT, "layers": checks})
            raise ValueError(f"Hy4 upstream layer {index} fails F32 parity: {evidence}")
        family.release_layer(module)
        del module
        family.end_layer_memory("upstream preflight", index)
        if index == 2:
            family.load_final()
            truncated = [rel_l2(family.final(a[0]).to(family.device), family.official_final(n[0]))
                         for a, n in zip(adapted, native)]
            family.release_final()
            if max(truncated) > REL_LIMIT:
                layer_major.atomic_json(out, {"passed": False, "relative_l2_max": REL_LIMIT, "layers": checks,
                                              "truncated_logit_relative_l2": truncated})
                raise ValueError(f"Hy4 upstream truncated forward fails F32 parity: {truncated}")
        print(f"Hy4 preflight layer {index + 1}/{family.layers}: {max(local):.3e}", flush=True)
    family.load_final()
    full = [rel_l2(family.final(a[0]).to(family.device), family.official_final(n[0])) for a, n in zip(adapted, native)]
    family.release_final()
    passed = max(full) <= REL_LIMIT
    result = {"passed": passed, "relative_l2_max": REL_LIMIT, "rope_style": ROPE_STYLE, "qa_kva_eps": QA_KVA_EPS,
              "tokens": [len(x) for x in tokens], "long_indexer_tokens": long_len,
              "truncated_logit_relative_l2": truncated, "full_logit_relative_l2": full,
              "layers": checks, "source_pin": family.provenance()}
    layer_major.atomic_json(out, result)
    if not passed:
        raise ValueError(f"Hy4 upstream full model fails F32 parity: {full}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--windows", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--rope-ab", action="store_true")
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--pilot-layers", type=int, choices=(2, 3),
                      help="profile the dense, first MoE/full-indexer and shared layers; no logits")
    a = parser.parse_args()
    family = Family(a.snapshot, "cpu" if a.check_only else a.device)
    checked = family.check_all()
    print(f"Native Hy4 headers: {checked} text tensors; official source {sources.TRANSFORMERS_REVISION}", flush=True)
    if a.check_only:
        load_pinned_transformers()
        return
    if a.out is None or (not (a.preflight_only or a.rope_ab) and a.windows is None):
        parser.error("--out and (for golden scoring) --windows are required")
    import torch
    a.out.mkdir(parents=True, exist_ok=True)
    family.configure()
    layer_major.DEVICE = a.device
    if a.rope_ab:
        with torch.inference_mode():
            rope_ab(family, a.snapshot, a.out / "rope-ab.json")
        return
    if a.pilot_layers is not None:
        import resource
        if a.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        timing = layer_major.run(a, family)
        peak = Path("/sys/fs/cgroup/memory.peak")
        record = {"layers": timing, "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  "cgroup_peak_bytes": int(peak.read_text()) if peak.is_file() else None,
                  "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved() if a.device == "cuda" else 0}
        layer_major.atomic_json(a.out / "pilot.json", record)
        print(json.dumps(record), flush=True)
        return
    family.initialize_official()
    report = a.out / "upstream-preflight.json"
    if report.exists():
        evidence = json.loads(report.read_bytes())
        if not evidence.get("passed") or evidence.get("source_pin") != family.provenance():
            raise ValueError("stale or failed Hy4 preflight receipt; use a fresh output directory")
    else:
        with torch.inference_mode():
            evidence = preflight(family, a.snapshot, report)
    print(f"Hy4 pinned upstream preflight passed at relative L2 <= {REL_LIMIT:g}: "
          f"full logits max {max(evidence['full_logit_relative_l2']):.3e}", flush=True)
    family.upstream_preflight = evidence
    if not a.preflight_only:
        layer_major.run(a, family)


if __name__ == "__main__":
    main()
