# Inkling 2 Mini

Inkling 2 Mini (`thinkingmachines/Inkling-2-Mini`, `Inkling2LoopForConditionalGeneration`,
`model_type: inkling2_loop`) is CuteAFD's first looped transformer family (`inkling2_mini`),
serving TP1 locally on **1× RTX PRO 6000 (0 Sparks)** ([#2701](https://github.com/tpurtell/cuteafd/issues/2701)).

Automatic KV sizing uses the shared 2M-token target on every card. Placement is
fixed at that target (all 24 physical routed-expert layers resident on GPU 0) and
the global paged KV pool grows into remaining admitted memory, within native
encoding limits. See [memory policy](../options.md#memory-and-pools).

## Architecture

- **Looped backbone:** 24 physical layers executed across **48 visits**:
  - Prelude: physical layers `0..6` (visits `0..6`).
  - Looped core: physical layers `6..18` executed across **3 passes** (visits `6..18`, `18..30`, `30..42`).
    At visit 5 (end of prelude), the pre-core residual state and block history are saved;
    before core passes 1 and 2 (visits 18 and 30), the residual stream restarts from that
    pre-core state while retaining the previous pass's 2-block summary (`h_after_L11`, `h_after_L17`)
    and injecting gated cross-pass messages through per-layer rank-64 controllers (`attn`, `k`, `v`, `mlp`).
  - Coda: physical layers `18..24` (visits `42..48`).
- **48 distinct cache slots:** every visit owns its own KV and short-convolution state:
  - **8 global paged KV slots** (`[5, 11, 17, 23, 29, 35, 41, 47]`), FP8 (default) or BF16.
  - **40 local sliding-window-1024 slots**, BF16 rings.
  - **48 contiguous FP32 short-convolution lag slots** (width `6656 = 2304 + 2304 + 1024 + 1024`, 3 lags per slot).
- **Attention:** GQA 24/8 × 128 (`H=2304`, group size 3), relative-position bias (`d_rel=16`,
  local window 1024, global extent 1024), per-token per-head attention sinks, sigmoid output gate,
  Q/K L2 clip (`22.627 = 2*sqrt(128)`), R L2 clip (`8.0`), global log-scaling (`floor=8192`, `alpha=0.1`),
  and short causal K/V convolutions (static on local layers; static + dynamic per-token taps on global layers).
- **MoE:** 128 routed experts, top-8, intermediate 1152, no shared expert; sigmoid + F32 bias
  selection, log-sigmoid softmax weights × `8.0`, interleaved `w13` gate/up with SiTU softcaps
  (`gate=4`, `up=25`), route weight applied before `w2` input quantization.
  - Physical layers `0..20`: native **NVFP4 W4A4** (packed E2M1, E4M3 per-16 scales, F32 per-expert amax
    and outer scale denominator; per-token amax activations with `fouroversix_mae` using exact `div.rn`).
  - Physical layers `20..24`: native **MXFP8 W8A8** (E4M3 weights, UE8M0 scales per 32-wide K group).
- **Head:** untied BF16 `embed`/`unembed` (`[201024, 2304]`), logits trimmed to `200058`, F32 unembed
  projection with softcap `30.0`.
- **Bundled MTP drafter (`mtp.safetensors`):** 3-layer recurrent dSpark/Markov drafter (`block_size=8`,
  `markov_rank=256`, target visit taps `[5, 23, 41, 48]`), own packed arena, history draft policy.
- **Bundled vision tower:** 23-layer PE-Lang ViT (`1024` hidden, 16 heads, patch 20) + 2×2 merge
  projector (`proj_out=[2304, 4096]`), own packed arena, batched images and prefix-hash spans
  through the shared media path. `VISION=auto` admits the tower when the KV pool still meets
  `KV_MIN_TOKENS`; explicit `VISION=off` leaves it unloaded.

## Authoritative Precision Contract (`f1940f40eb`)

The engine matches the checkpoint's `hf_quant_config.json` parameter groups and official
`transformers@d8f10a82` F32 gate/blend islands (authoritative reference oracle commit **`f1940f40eb`**):

- **BF16 compute:** embedding and layer state; dense projections (`wq_du`, `wr_du`, `wk_dv`, `wv_dv`,
  `gate_attn`, `wo_ud`); QK and PV GEMMs; loop controller `down_proj` GEMM and its rank-64 result.
- **F32 compute:** declared `attn_res` queries (`attn_queries`, `mlp_queries`, `output_query`),
  `middle_cycle_cross_pass_*queries`, and `mlp.gate.bias`; RMS, L2-clip, and log-scale internals;
  residual block and cross-pass message retrieval/mixing; loop controller `forget`/`gain` GEMMs plus
  `tanh` and blend; static and dynamic convolution weights, state, and reductions; router logits,
  scores, and gamma; `attn_sink_proj` sinks; combined main + sink attention softmax (sink dropped,
  probabilities cast to BF16 for PV); output-gate sigmoid and multiply (cast back to BF16).
- **Experts:** native NVFP4 W4A4 (`fouroversix_mae` with exact `div.rn`, bit-exact against oracle
  commit `6bcb2a8da0`/`ece18287ff`) and native MXFP8 W8A8 with IEEE F32 math, BF16 FC1 and SiTU
  epilogues, F32 expert combine, BF16 output.
- **Head:** BF16 stored `unembed.weight` × F32 projection and logits, F32 softcap `30.0`.

## Checkpoints

| Checkpoint | Experts | Core |
|---|---|---|
| `thinkingmachines/Inkling-2-Mini` @ `7a67ae2314b44ef05c87fd1d4d311c58aaeb18f6` | layers `0..20` native NVFP4 W4A4 (`fouroversix_mae`), layers `20..24` native MXFP8 W8A8 | BF16 + official F32 islands |

## Serving and Qualification

- Public serve command: `cuteafd serve --family inkling2_mini` (`serve-inkling2-mini`) or
  `./run.sh --config examples/configs/inkling2-mini-1x-rtx.config --wip SLOT`.
- Public golden command: `cuteafd golden --family inkling2_mini` (`inkling2-mini-golden`) via
  `scripts/bench/golden-run.py --family inkling2_mini`.
- **Fidelity gate (3–6 bpw tier):** Mini's effective routed-expert bpw is `4.67–5.13` (20 NVFP4 + 4 MXFP8),
  measured against published reference set `wrldsuksgo2mars/cuteafd-fidelity` @
  `eabfaec64c7e96206ac1dffbbe51ad6deb73f635` (`inkling2-mini-v1_20261124_own-text-native`, manifest
  SHA-256 `a7276a3e4f10653614bfe8b591acc3b32878d8f88ff1bdf5d9cf22f189ade449`, 64 windows, pooled NLL `0.79864`):
  pooled generated-row **KL ≤ 0.05 nats** and **top-1 ≥ 0.93** (standard tier = 64 decode windows;
  quick tier = 8 windows). Byte-exact gates apply to `N=1` vs `N` width invariance, graph vs eager parity,
  exact 48-slot prefix restore, and same-input QDQ kernel oracles (`div.rn` tie vector + exact-decode negative control).
- Exact-row CUDA decode graphs per visit (`(visit, rows, section, mode)`) bounded by the shared
  `#2697` soft-cap working-set reserve (`CUTEAFD_INKLING2_MINI_GRAPHS=1|0`).
- Exact 48-slot prefix cache (prompt-end and turn-end marks, host offload).
