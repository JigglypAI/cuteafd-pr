"""Full Hy4 direct MC on the single K8-causal production stream."""
import sys

import torch
from __future__ import annotations

from hy4_flash import run as shared
from . import forward, host_input_recovery, recipe
from .export import DirectMCWriter, verify_original_discovery
from .gates import check_recovery
from .parity import verify_preflight
from .prepare_corpus import load_token_batches
from .source import HeaderSource


def identity(source, corpus, preflight):
    code = (*shared.CODE, "quantization/hy4_family_source.py",
            *(f"quantization/hy4/{name}.py" for name in
              ("run", "recipe", "forward", "parity", "source", "prepare_corpus", "export", "gates",
               "host_input_recovery")))
    return shared.identity(source, corpus, preflight, model_recipe=recipe, code=code,
                           check=verify_preflight, extra=dict(rope="adjacent-pair-mla-and-indexer",
                           residual_storage="bf16", expert_input="full-post-attention-norm"))


class Runner(shared.Runner):
    forward = forward
    recipe = recipe
    target_gates = frozenset({0, 1, 2, 5, 77})
    header_type = HeaderSource
    writer_type = DirectMCWriter
    load_batches = staticmethod(load_token_batches)
    frozen_identity = staticmethod(identity)
    checkpoint_identity = staticmethod(host_input_recovery.checkpoint_identity)
    verify_discovery = staticmethod(verify_original_discovery)
    use_expert_latent = False
    host_expert_inputs = True
    persist_projected_routes = False
    store_attention_residual = False

    def expert_group_memory(self):
        if torch.device(self.device).type != "cuda":
            return {}
        stats = torch.cuda.memory_stats(self.device)
        report = dict(cuda_allocated_bytes=stats["allocated_bytes.all.current"],
                      cuda_reserved_bytes=stats["reserved_bytes.all.current"],
                      cuda_peak_allocated_bytes=stats["allocated_bytes.all.peak"],
                      cuda_inactive_split_bytes=stats["inactive_split_bytes.all.current"])
        encoder = sys.modules.get("gptqmodel.exllamav3.modules.quant.exl3_lib.quantize")
        if encoder is not None:
            report["trellis_workspace_count"] = len(encoder._workspace_buffers)
        return report

    def save_gate(self, label, metrics, tick, **fields):
        extra = fields.setdefault("extra", {})
        if any(r.get("heldout_passed") is False for r in metrics) or extra.get("published_k6_drafter_bits") == 8:
            extra.setdefault("status", "failed-review-recorded-fallback")
        super().save_gate(label, metrics, tick, **fields)
        check_recovery(self.root, label)

    def __init__(self, args):
        super().__init__(args)
        # All production records fit the dense prefix of DSA. The >2048
        # full/shared IndexCache witness runs in reference parity only.
        if any(len(tokens) > self.model.config.index_topk for tokens in self.records):
            raise ValueError("full Hy4 calibration or heldout includes a long-context parity-only record")


if __name__ == "__main__":
    shared.main(Runner)
