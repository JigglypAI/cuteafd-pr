#!/usr/bin/env python3
"""Layer-major Hy4-flash F32 golden; qualify against the pinned HF text forward first."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "python/reference"))
import layer_major
from families.hy4_flash import model, sources

CONFIG_SHA256 = "a59ea10f98683e83c6515e340c85e24eb2e48c91d75af491ed3d6e88c3bc1b3c"
INDEX_SHA256 = "0cd31d2ffceb989e4d465d0c80be5731f6941dbf434ad84506e30930fa02e192"
REL_LIMIT = 1e-4


def expected_source_identity(_runtime=None):
    return {"checkpoint": sources.CHECKPOINT, "revision": sources.REVISION,
            "transformers_revision": sources.TRANSFORMERS_REVISION,
            "transformers_files": sources.TRANSFORMERS_FILES,
            "config_sha256": CONFIG_SHA256, "index_sha256": INDEX_SHA256,
            "model_sha256": layer_major.digest(model.__file__), "golden_sha256": layer_major.digest(__file__),
            "sources_sha256": layer_major.digest(sources.__file__),
            "layer_major_sha256": layer_major.digest(layer_major.__file__)}


def load_pinned_transformers():
    import transformers
    root = Path(transformers.__file__).resolve().parent
    for name, digest in sources.TRANSFORMERS_FILES.items():
        if layer_major.digest(root / name) != digest:
            raise ValueError(f"not the pinned Transformers source: {name}")
    from transformers.models.hy_v4_flash import modeling_hy_v4_flash as hf
    return hf


def official_key(name):
    for old, new in (("self_attn.learnable_sink_param", "self_attn.sinks"),
                     ("self_attn.linear_gate.", "self_attn.gate_proj."),
                     ("hc_attn_layer.hc_pre.hc_", "attn_hc."), ("hc_mlp_layer.hc_pre.hc_", "ffn_hc.")):
        if name.startswith(old):
            return new + name[len(old):]
    return name


class Family:
    name = "hy4_flash"
    compare_official_head = True

    def __init__(self, snapshot, device):
        snapshot = Path(snapshot)
        if (snapshot.name != sources.REVISION or layer_major.digest(snapshot / "config.json") != CONFIG_SHA256
                or layer_major.digest(snapshot / "model.safetensors.index.json") != INDEX_SHA256):
            raise ValueError("Hy4 oracle requires the pinned original BF16 snapshot")
        self.raw_config = json.loads((snapshot / "config.json").read_bytes())
        self.config = model.Config.parse(self.raw_config)
        self.layers = self.config.num_hidden_layers
        self.checkpoint = model.Checkpoint(snapshot, device)
        self.device = self.checkpoint.device
        self.outer = self.official = self.official_config = self.rotary = None
        self.official_mode = False
        self.upstream_preflight = None

    def initialize_official(self):
        import torch
        hf = self.official = load_pinned_transformers()
        official_root = hf.HYV4FlashConfig.from_dict(self.raw_config)
        c = self.official_config = official_root.get_text_config()
        c._attn_implementation = "eager"
        c._experts_implementation = "eager"
        with torch.device(self.device):
            self.rotary = hf.HYV4FlashRotaryEmbedding(c)

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
        indexed = {n for n in self.checkpoint.index if n.startswith(("model.layers.", "model.hc_head.")) or n in model.OUTER_NAMES}
        if known != indexed:
            raise ValueError(f"unexpected Hy4 text inventory: {sorted(indexed ^ known)}")
        return len(known)

    def layer_bytes(self, index):
        import math
        return sum(math.prod(shape) * (2 if name.endswith(model.EXPERTS) else 4)
                   for name, (shape, _) in self.tensors(index).items() if not name.endswith(model.TABLE))

    def state_bytes(self, tokens):
        return tokens * self.config.hidden_size * self.config.hc_mult * 4 + tokens * 8

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

    def initial_carry(self, window):
        import torch
        return {"tokens": torch.tensor(window["tokens"], dtype=torch.long)}

    def work_bytes(self, tokens):
        # Q/G/K/V, residuals, tiled score/probability/value output and indexer scratch.
        c = self.config
        heads = max(c.num_attention_heads, c.swa_num_attention_heads)
        width = max(c.qk_nope_head_dim + c.qk_rope_head_dim, c.v_head_dim)
        return 4 * (tokens * (heads * width * 8 + c.hc_mult * c.hidden_size * 3)
                    + heads * model.ROWS * tokens * 4 + c.compressed_indexer_num_attention_heads * model.ROWS * tokens)

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

    def load_layer(self, index):
        return {"weights": self.checkpoint.layer(self.config, index), "official": None}

    def release_layer(self, module):
        self.checkpoint.close()

    def geometry(self):
        return nullcontext()

    def official_geometry(self):
        from contextlib import contextmanager

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
            layer = hf.HYV4FlashDecoderLayer(c, index)
        # Keep the upstream MoE routing/accumulation, but decode BF16 expert matrices
        # on demand instead of creating a second fully expanded copy of the layer.
        if hasattr(layer.mlp, "experts"):
            from torch.nn import Parameter

            class ExpertMatrices(Parameter):
                def __getitem__(self, key):
                    return super().__getitem__(key).float()

            for key in model.EXPERTS:
                expert = key.rsplit(".", 1)[1]
                setattr(layer.mlp.experts, expert, ExpertMatrices(w[key], requires_grad=False))
        state = {official_key(key): value for key, value in w.items() if key not in (*model.EXPERTS, model.TABLE)}
        result = layer.load_state_dict(state, strict=False, assign=True)
        expected_missing = set(model.EXPERTS) & w.keys()
        if model.TABLE in w:
            expected_missing.add(model.TABLE)
        if set(result.missing_keys) != expected_missing or result.unexpected_keys:
            raise ValueError(f"Hy4 Transformers state mapping differs: {result}")
        indexer = layer.self_attn.compressed_indexer
        if indexer is not None:
            indexer.rotary_emb = self.rotary
        e = layer.engram_layer
        if e is not None:
            # Preserve the official lookup/hash/FP8/norm implementation; replace only
            # nn.Embedding storage access, whose whole table cannot be upcast in RAM.
            table = w[model.TABLE]

            class GatherRows(torch.nn.Module):
                def forward(self, rows):
                    return table.index_select(0, rows.flatten().to(table.device)).reshape(*rows.shape, -1).to(
                        rows.device, dtype=torch.float32)

            e.multi_head_embedding.embedding = GatherRows()
            e.multi_head_embedding.offsets = e.multi_head_embedding.head_offsets().to(self.device)
            e.head_vocab_sizes = torch.tensor(e.config_head_vocab_sizes, device=self.device)
            e.hash_multipliers = e.draw_hash_multipliers(int(e.lookup_table.max()) + 1).to(self.device)
        module["official"] = layer.eval()
        return module["official"]

    def official_forward(self, module, index, hidden, tokens):
        import torch
        from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
        layer = self.official_layer(module, index)
        positions = torch.arange(len(tokens), device=self.device).reshape(1, -1)
        # Build masks/position encodings through the exact functions the upstream text model calls.
        creator = (create_sliding_window_causal_mask if layer.attention_type == "sliding_attention"
                   else create_causal_mask)
        mask = creator(config=self.official_config, inputs_embeds=hidden[:, :, 0], attention_mask=None,
                       cache=None, position_ids=positions, allow_is_causal_skip=False)
        phase = self.rotary(hidden, positions.unsqueeze(0), layer.attention_type)
        ng = self.official.HYV4FlashTextModel.build_ngram_tokens(
            type("TextConfigProxy", (), {"config": self.official_config})(), tokens[None], positions)
        ngmask = torch.ones_like(tokens, dtype=torch.bool)
        for token_id in self.config.masked_tokens:
            ngmask &= tokens != token_id
        return layer(hidden, attention_mask=mask, position_ids=positions.unsqueeze(0),
                     position_embeddings=phase, ngram_tokens=ng, engram_mask=ngmask[None])

    def forward(self, module, index, hidden, carry):
        tokens = carry["tokens"]
        result = (self.official_forward(module, index, hidden, tokens) if self.official_mode else
                  model.layer_forward(hidden[0], tokens, module["weights"], self.config, index).unsqueeze(0))
        return result, carry, ("tokens",)

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
            hc = self.official.HYV4FlashHyperHead(self.official_config)
            norm = self.official.HYV4FlashRMSNorm(self.config.hidden_size, self.config.rms_norm_eps)
        hc.load_state_dict(dict(zip(("hc_fn", "hc_base", "hc_scale"), (w[n] for n in model.OUTER_NAMES[1:4]))), assign=True)
        norm.load_state_dict({"weight": w[model.OUTER_NAMES[4]]}, assign=True)
        return F.linear(norm(hc(hidden.unsqueeze(0)))[0], w[model.OUTER_NAMES[5]])

    def golden_meta(self):
        gates = {"gates": {"upstream_preflight": self.upstream_preflight}} if self is not None and self.upstream_preflight else {}
        return {**gates,"reference": "Independent Hy4-flash layer-major IEEE F32 text reference",
                "source_identity_provider": "families.hy4_flash.golden",
                "root_arithmetic": sources.ROOT_ARITHMETIC}


def rel_l2(actual, expected):
    if not actual.isfinite().all() or not expected.isfinite().all():
        return float("inf")
    return float((actual - expected).norm() / expected.norm().clamp_min(1e-30))


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
    # Force a real compressed-selector comparison separately from the short full-model walks.
    long_tokens = tokens[0].repeat(1 + (family.config.compressed_indexer_candidate_topk + 1) // len(tokens[0]))[
        :family.config.compressed_indexer_candidate_topk + 1]
    adapted = [family.embed(t.tolist()) for t in tokens]
    native = [a.clone() for a in adapted]
    long_hidden = family.embed(long_tokens.tolist())
    family.release_outer()
    checks = []
    for index in range(family.layers):
        family.begin_layer_memory()
        module = family.load_layer(index)
        local = []
        for j, ids in enumerate(tokens):
            golden_input = native[j]
            native[j] = family.official_forward(module, index, golden_input, ids)
            same = model.layer_forward(golden_input[0], ids, module["weights"], family.config, index).unsqueeze(0)
            local.append(rel_l2(same, native[j]))
            adapted[j] = model.layer_forward(adapted[j][0], ids, module["weights"], family.config, index).unsqueeze(0)
        evidence = {"layer": index, "same_input_relative_l2": local,
                    "trajectory_relative_l2": [rel_l2(a, n) for a, n in zip(adapted, native)]}
        if index == 0:
            g = family.official_forward(module, index, long_hidden, long_tokens)
            h = model.layer_forward(long_hidden[0], long_tokens, module["weights"], family.config, index).unsqueeze(0)
            evidence["active_indexer_relative_l2"] = rel_l2(h, g)
            del g, h, long_hidden
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
    result = {"passed": passed, "relative_l2_max": REL_LIMIT, "tokens": [len(x) for x in tokens],
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
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--pilot-layers", type=int, choices=(2,),
                      help="profile the dense and first MoE/Engram layers; no logits")
    a = parser.parse_args()
    family = Family(a.snapshot, "cpu" if a.check_only else a.device)
    checked = family.check_all()
    print(f"Native Hy4 headers: {checked} text tensors; official source {sources.TRANSFORMERS_REVISION}", flush=True)
    if a.check_only:
        load_pinned_transformers()
        return
    if a.out is None or (not a.preflight_only and a.windows is None):
        parser.error("--out and (for golden scoring) --windows are required")
    import torch
    a.out.mkdir(parents=True, exist_ok=True)
    family.configure()
    layer_major.DEVICE = a.device
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
