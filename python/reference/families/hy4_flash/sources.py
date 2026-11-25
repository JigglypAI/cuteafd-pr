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
             "DFly and runtime KV cache are excluded; vision lives in vision.py (VISION below).",
}

ROOT_ARITHMETIC = {
    "source": "Transformers PR 50903 pinned eager F32 text semantics, ported independently",
    "arithmetic": "BF16 checkpoint, IEEE F32 forward; per-vector Engram E4M3 QDQ; "
                  "fixed-M128 projections and query tiles; vision, cache and DFly off.",
}

SGLANG_VISION_FILES = {
    "python/sglang/srt/models/hunyuan_v4_vl.py": "cdc0beb134fdda71af1b86d53d3fc8e21212660ed30af728a9deec80a53f9e0e",
    "python/sglang/srt/multimodal/processors/hy_v4_vl.py": "62e60fff3e019e5de7a4658cc23049d13bbd70cf0b809548ed68d85d244f7b44",
    "python/sglang/srt/configs/hy_v4_vl_processing.py": "380c38f4fe8bccb095bf193a1c7d7332219364353a084c72ee74f75afdd06a03",
}
TRANSFORMERS_VISION_FILES = {
    "models/hy_v4_flash/image_processing_pil_hy_v4_flash.py": "3742587a5240d3362d85cb195f0818fe1c5f579b4d849f50cd7cc84f0b264bb5",
}

VISION = {
    "authority": "Tencent serving stack (SGLang PR 51159 at SGLANG_REVISION); Transformers PR 50903 cross-checks.",
    "tower": "27 pre-LN SigLIP2 blocks (LayerNorm eps 1e-5 with bias, 16x72 heads with biased q/k/v/o, "
             "exact GELU MLP 1152-4304-1152), no 2D RoPE (use_2d_rope defaults off). "
             "Patch conv 16x16 with bias plus the learned 128x128 table (row 0 unused cls) "
             "resized in F32 with scale_factor (n+0.1)/128, bilinear, align_corners False.",
    "merger": "RMSNorm(1152, 1e-5) -> 2x2/2 conv (1152->4608, bias) -> exact GELU -> 1x1 conv (4608->3072, bias); "
              "the checkpoint's image_newline closes every merged row: h/2*(w/2+1) LM rows.",
    "pixel_order": "SGLang patchify is raster over the (h, w) patch grid with (c, py, px) per row, and its tower "
                   "reads rows that way. The pinned HF processor emits group-major rows (gh, mh, gw, mw, c, ph, pw) "
                   "while HYV4FlashVisionTransformer also reshapes rows as raster: the HF pair is inconsistent. "
                   "This reference, the cross-check and the engine use SGLang's raster order.",
    "processor": "SGLang smart_resize (factor 32, 4096..4194304 px, Python round); Pillow BICUBIC on uint8 RGB "
                 "(pinned HF PIL processor and engine; SGLang's torchvision BICUBIC differs on resized images); "
                 "(v/255-0.5)/0.5 in F32.",
    "positions": "SGLang get_hy_v4_vl_mrope_positions: image rows [s, s+r, s+c] over the newline-inclusive "
                 "columns; text resumes at s+max(rows, cols). Interleaved M-RoPE section [12, 10, 10] over 32 "
                 "pairs: pair p reads H if p%3==1 and p<30, W if p%3==2 and p<30, else T. KV slots, SWA, causal "
                 "masks and indexer group ids stay token-slot based; group keys rotate at their first slot's position.",
    "embedding": "Image rows replace inputs_embeds before the iHC 4-stream expansion; Engram masks image ids.",
    "rms_rounding": "HF HYV4FlashRMSNorm rounds x*rsqrt to BF16 before the weight multiply; SGLang's fused RMSNorm "
                    "multiplies in F32 and rounds once. The serving reference follows SGLang.",
}
