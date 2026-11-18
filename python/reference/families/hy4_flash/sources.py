"""Pinned Hy4 source and known differences from the released metadata."""

CHECKPOINT = "tencent/Hy4-flash"
REVISION = "d88c9dfd96ba4f2e6b92adba0b8e8f1c8696f959"
TRANSFORMERS_REVISION = "c2ec850510a37978ea83df226589623d5f2be776"  # PR 50903
TRANSFORMERS_FILES = {
    "models/hy_v4_flash/modeling_hy_v4_flash.py": "43341929e4bec215c224fc837b24b534b4de41954992673a0655a65060e38b9b",
    "models/hy_v4_flash/configuration_hy_v4_flash.py": "dc62bd7c6bc8f57375d3f71cf4eb861e3a36887d57cd75f6b11752fcf04256cf",
    "conversion_mapping.py": "dcb88ad0abf0859b27e7494c1030271740da863981bffebf19d62e470545fc90",
}
TENCENT_REVISION = "7ab20f35602b1d3fc863f85aee20cb87eb8e53c5"
VLLM_RECIPE_REVISION = "51d9b9ab07f18797c5db0092671b20171a621728"
SGLANG_REVISION = "ccb8d9ccaa309f71dda42e37517a7b0ee6afae8d"  # PR 51159

SEMANTICS = {
    "namespace": "Checkpoint HC, linear_gate and learnable_sink_param keys are renamed by HF conversion_mapping; "
                 "this oracle reads the original keys directly. deepseek_sparse_attention is indexed_attention in HF.",
    "attention": "The raw qk_norm flag does not introduce additional per-head norms in HF; q_a and kv_a are normalized. "
                 "HF expands MLA KV-B before attention; serving recipes may absorb it. Text uses the same position on all M-RoPE axes.",
    "moe": "Expert down projects into latent space and expert up returns to hidden width (HF F.linear convention). "
           "Fused gate/up are contiguous halves. Only the routed expert gates/up channels are clipped.",
    "engram": "The README names zero-based decoder indices; config names one-based layer IDs. The BF16 table stays "
              "BF16 on disk. HF casts the fetched row to the hidden-state dtype before per-vector E4M3 QDQ. "
              "Stock BF16 inference therefore performs QDQ in BF16; the golden deliberately upcasts the rows "
              "and performs QDQ in F32 before head RMS normalization. Those paths are not bitwise equivalent. "
              "The hash multipliers use engram_seed and the loaded map's max + 1, not the table's row count.",
    "serving_differences": "SGLang caches compressed indexer keys in FP8; HF pools and scores them in F32 here. "
                           "SGLang's Engram consumes stored BF16 rows or stored FP8 plus scales, without simulating "
                           "the HF E4M3 round-trip on BF16 rows. It also narrows Engram work to serving BF16 and "
                           "uses a BF16 epsilon for key/query norms. Do not inherit those serving roundings.",
    "scope": "The Tencent serving recipe points to Transformers; its finetuning patches can disable Engram, "
             "which is not an inference setting. vLLM recipe is deployment guidance, not an eager oracle. "
             "Vision, DFly and runtime KV cache are excluded.",
}

ROOT_ARITHMETIC = {
    "source": "Transformers PR 50903 pinned eager F32 text semantics, ported independently",
    "arithmetic": "BF16 checkpoint, IEEE F32 forward; per-vector Engram E4M3 QDQ; "
                  "fixed-M128 projections and query tiles; vision, cache and DFly off.",
}
