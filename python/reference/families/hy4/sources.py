"""Pinned Hy4 source and known differences from the released metadata."""

CHECKPOINT = "tencent/Hy4"
REVISION = "c10c0de5b6f850703fb0a90e65d1c32b558b9167"
TRANSFORMERS_REVISION = "2f1bc5e321b9a736200a590d6e7743ccdeb60772"
TRANSFORMERS_PR50589_REVISION = "bd5b6b6d979cd22a7d237d9e84b88d212b08a545"
TRANSFORMERS_FILES = {
    "models/hy_v4/modeling_hy_v4.py": "29673975dbea3d41e64904019926dd3a71d0acfa1e17232423096515247a1fd3",
    "models/hy_v4/configuration_hy_v4.py": "d55ed3d35a4faf0c1f7a3f8334a4a97729463f48b0365d2751f8aba767e3cad0",
    "conversion_mapping.py": "6e4085f1c941c55b4d12cdddcbb6134a683fe3639af853988bcb1c026ac2d88c",
}
TENCENT_REVISION = "6eb6613298923f528b6f991f1d81c319e3603829"
VLLM_REVISION = "1bdc437402d6eee29d3ca543c91574f8acc72c8d"
SGLANG_REVISION = "6f3f73aef5eacea34bff20e3ac04ab5c8490f826"
ANGELSLIM_GGUF_REVISION = "3ab6319e385d482075b5e9eb6e4bc1c3ffede8ec"
ANGELSLIM_PREVIEW_GGUF_REVISION = "82863242ee88d3bad06d1e42f4e31d6674de692a"

SEMANTICS = {
    "namespace": "Checkpoint HC, linear_gate and learnable_sink_param keys are renamed by HF conversion_mapping "
                 "(hy_v4_vl_text -> hy_v4); this oracle reads the original keys directly. "
                 "deepseek_sparse_attention is indexed_attention in HF.",
    "rope": "Settled by full 78-layer F32 real-weights A/B on cassowary (commit 5565df18f9, golden.py:rope_ab): "
            "the checkpoint uses adjacent-pair (interleaved) RoPE, matching transformers#50589 (bd5b6b6d), vLLM "
            "(1bdc4374), SGLang (6f3f73ae), AngelSlim GGUF, and Hy4-flash. Transformers main (2f1bc5e3) applies "
            "split-half RoPE and degrades short-prompt mean NLL from 0.72924 to 1.64792 (+0.91868 nat, top-1 "
            "80.90% -> 59.80% over 199 scored tokens) and 2,304-token README NLL from 1.10800 to 3.82016 "
            "(+2.71216 nat, top-1 74.69% -> 37.00%). In the 2x2 MLA x indexer grid at 2,304 tokens (256 tail "
            "positions pruned to top-2048), interleaved MLA + interleaved indexer gives 70.31% tail top-1 "
            "(1.09435 NLL) vs 69.14% (1.08950 NLL) with split-half indexer (only ~6% keys pruned at 2,304 tokens).",
    "attention": "Gated DSA MLA in all 78 layers with F32 learnable sinks and elementwise sigmoid output gate. "
                 "No post_proj_norm, no SWA. IndexCache: 21 full indexer layers (0, 1, 5, 9, ..., 77) compute top-2048 "
                 "causal token indices; the 57 shared layers reuse the previous full layer's top-2048 indices. "
                 "Indexer applies RoPE to the trailing qk_rope_head_dim=64 channels of q and LayerNorm'd k.",
    "norm_eps": "config.rms_norm_eps is 1e-5. Transformers main HYV4Attention constructs q_a_layernorm and "
                "kv_a_layernorm without passing eps (defaulting to 1e-6), while using config.rms_norm_eps (1e-5) for "
                "input_layernorm, post_attention_layernorm, model.norm, iHC unweighted RMSNorm, and indexer k_norm. "
                "vLLM uses 1e-5 for q_a_layernorm/kv_a_layernorm and 1e-6 for indexer k_norm; AngelSlim GGUF uses "
                "1e-5 everywhere. On the 3 short real prompts, 1e-6 vs 1e-5 on q_a/kv_a gives 0.72924 vs 0.73023 "
                "mean NLL (identical 80.90% top-1, rel L2 [2.64e-2, 3.80e-2, 5.23e-5] from isolated MoE routing flips). "
                "This oracle matches Transformers main (1e-6 on q_a/kv_a, 1e-5 elsewhere).",
    "moe": "Layer 0 is dense SwiGLU (18432); layers 1..77 are standard MoE (256 routed experts top-8, intermediate 2048, "
           "sigmoid + e_score_correction_bias, norm_topk_prob, routed_scaling_factor 2.827, n_group=1, topk_group=1) "
           "plus 1 shared expert (2048). No LatentMoE, no post_fc2_norm, no combine_output_norm, no Engram. "
           "Fused gate_up_proj is contiguous [gate, up] halves; swiglu_limit=10.0 clamps only routed experts.",
    "head": "lm_head.weight is stored as BF16 on disk and computed in F32 (enable_lm_head_fp32=true, untied).",
    "serving_differences": "vLLM and SGLang quantize DSA indexer Q/K to FP8 (ue8m0 block 128) and permute indexer "
                           "weights; SGLang skips Hadamard because it is orthogonal. Transformers main and this oracle "
                           "score the indexer directly in F32 without FP8 quantization.",
    "scope": "Vision tower (vit.*, 40 layers), bundled 5-layer DFly2 drafter (model.dfly_layers.*), and runtime KV "
             "cache are excluded from the text oracle.",
}

ROOT_ARITHMETIC = {
    "source": "Transformers main 2f1bc5e321b9 pinned eager F32 text semantics with adjacent-pair RoPE (PR 50589, settled by 78-layer A/B), ported independently",
    "arithmetic": "BF16 checkpoint, IEEE F32 forward; adjacent-pair RoPE; F32 DSA indexer with IndexCache top-2048 reuse; "
                  "fixed-M128 projections and query tiles; vision, cache and DFly off.",
}
