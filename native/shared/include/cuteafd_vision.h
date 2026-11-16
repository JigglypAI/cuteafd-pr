#pragma once
#include <stdint.h>
#include "cuteafd_media_backings.h"
#ifdef __cplusplus
extern "C" {
#endif

/* Version 1: donated MiMo numerics (FP32 residual, BF16 dense inputs,
 * FP16 rotary Q/K/V). All offsets address an admitted resident weight arena.
 * matrices: BF16; vectors (norm/bias/sinks/inv_freq): FP32. */
#define CUTEAFD_VISION_NUMERICS 1
#define CUTEAFD_VISION_NO_OFFSET UINT64_MAX

typedef struct {
  uint64_t qkv, qkv_bias, proj, proj_bias, gate_up, gate_up_bias;
  uint64_t down, down_bias, norm1, norm2, key0_bias;
  int32_t window, column_order;
} cuteafd_vision_block;

/* Optional PE arithmetic/output extension. Offsets address the same admitted
 * BF16 arena as the base spec; scale arrays contain exactly depth entries.
 * Absolute positions and rotary are independent (both towers use both). */
typedef struct {
  uint32_t activation; /* 0: exact GELU, 1: quick GELU. */
  uint32_t position_grid, class_token, output_kind; /* 0: merge2, 1: conv3s2 twice. */
  uint64_t ln_pre, ln_pre_bias, class_embedding, class_position;
  uint64_t downsample1, downsample1_bias, downsample2, downsample2_bias;
  const uint64_t *scale1, *scale2;
} cuteafd_vision_perception;

/* Independent still-image extension. All offsets address the one admitted
 * weight arena (UINT64_MAX = absent); rotary tables are source-matched FP32.
 * Version/size frame an append-only prefix. Never combine with perception. */
typedef struct {
  uint32_t kind; /* 0 none, 1 LayerNorm, 2 RMSNorm. */
  float epsilon;
  uint64_t weight, bias;
} cuteafd_vision_tower_norm;
typedef struct {
  uint32_t kind, activation; /* kind: 0 none, 1 MLP; activation: 0 none, 1 GELU, 2 tanh GELU. */
  uint32_t input_width, hidden_width, output_width, reserved;
  uint64_t fc1, fc1_bias, fc2, fc2_bias;
} cuteafd_vision_tower_projector;
typedef struct {
  uint32_t version, size; /* 1, sizeof(cuteafd_vision_tower). */
  uint32_t block_norm_kind, activation;
  float block_norm_epsilon;
  uint32_t attention_width, temporal_patch;
  uint32_t pixel_layout; /* 0 none, 1 raster, 2 spatial-group-major. */
  uint32_t temporal_input; /* 0 none, 1 duplicate still; no video in v1. */
  uint32_t rotary_kind; /* 0 none, 1 split-half THW, 2 adjacent-pair WH. */
  uint32_t rotary_axis_dims[3], rotated_dims, passthrough_dims;
  uint64_t inv_freq[3]; /* T,H,W; axis_dims/2 FP32 elements per active axis. */
  uint32_t absolute_kind; /* 0 none, 1 divided-fixed spatial; stills skip time. */
  uint32_t position_height, position_width;
  uint32_t interpolation; /* 0 none, 1 compiled FP32 half-pixel bilinear. */
  uint64_t absolute_position;
  cuteafd_vision_tower_norm pre_norm, final_norm;
  uint32_t spatial_merge;
  uint32_t projection_order; /* 0 none, 1 group-project, 2 project-group-project. */
  cuteafd_vision_tower_projector projectors[2];
  cuteafd_vision_tower_norm post_norm;
} cuteafd_vision_tower;

typedef struct {
  uint32_t storage_abi; /* 5: independent optional tower extension, PE prefix unchanged. */
  uint32_t abi_version, max_tokens, output_width, reserved;
  uint64_t weight_bytes, patch, merger_norm, merger_fc1, merger_fc2, inv_freq;
  const cuteafd_vision_block* blocks;
  /* Numerics 1 = MiMo; numerics 2: reserved Qwen=2 or GLM Flash=3.
   * Every array below contains exactly depth host entries. */
  uint32_t hidden, depth, heads, kv_heads, head_dim, intermediate, patch_size, merger_width;
  float norm_eps; /* Muse: public feature RMS epsilon, internal LayerNorm stays 1e-5. */
  uint32_t geometry_reserved;
  uint64_t patch_bias, pos_embed, merger_norm_bias, merger_fc1_bias, merger_fc2_bias;
  uint64_t merger_extra[8];
  const uint64_t *norm1_bias, *norm2_bias, *q_norm, *k_norm;
  const cuteafd_vision_perception* perception; /* null for other tower contracts. */
  const cuteafd_vision_tower* tower; /* null preserves existing arithmetic. */
} cuteafd_vision_spec;

typedef struct {
  uint64_t weights, scratch, blas_workspace, device_allocations, encodes;
} cuteafd_vision_ledger;

/* Probe-only observer: stage 0 patch, 1..28 blocks, 29 merger norm, 30 output.
 * data is device FP32 (stage 30 is BF16); column_order applies to block rows.
 * The observer must not retain data past its return, and must propagate errors.
 * Owner remains alive until all queued CUDA work has drained, including errors. */
typedef int32_t (*cuteafd_vision_observer)(void*, int32_t stage, const void* data,
                                         int32_t rows, int32_t width, int32_t column_order);

/* Admission is checked before any CUDA allocation. Native create owns a
 * lowest-priority nonblocking stream; no CUDA allocation/capture in encode. */
int32_t cuteafd_vision_required(const cuteafd_vision_spec*, cuteafd_vision_ledger*);
int32_t cuteafd_vision_create(const cuteafd_vision_spec*, int32_t device,
                             uint64_t admitted_bytes, const cuteafd_media_tags* tags, void** owner);
int32_t cuteafd_vision_upload(void* owner, uint64_t offset, const void*, uint64_t bytes);
/* Deferred push exposes the existing admitted arena, zeroing its padding.
 * The caller drains external writes before commit; encode refuses until then. */
int32_t cuteafd_vision_weight_arena(void* owner, void** arena);
int32_t cuteafd_vision_commit_weights(void* owner);
/* RGB8 is resized, row-major HWC, width=grid_w*16 and height=grid_h*16.
 * lut is FP32 [3,256], host-provided exact reference normalization. */
int32_t cuteafd_vision_encode(void* owner, const uint8_t* rgb, uint64_t rgb_bytes,
                             const float* lut, int32_t grid_h, int32_t grid_w,
                             uint16_t* output, uint64_t output_bytes,
                             cuteafd_vision_observer observer, void* observer_context);
int32_t cuteafd_vision_get_ledger(void* owner, cuteafd_vision_ledger*);
int32_t cuteafd_vision_destroy(void* owner);
// Passive: the owner's two device backings; no device change or synchronization.
int32_t cuteafd_vision_get_backings(void* owner, cuteafd_media_backings* out);
// Destroy, reporting which backings were freed (CUTEAFD_MEDIA_RELEASED_*). A
// backing whose free failed stays live and its bit stays clear.
int32_t cuteafd_vision_destroy_backings(void* owner, uint32_t* released);

/* Shared toolkit. Inputs FP32, output BF16; bias may be null. Serialize calls
 * on a current-device stream and retain every extent through completion. */
int32_t cuteafd_vision_rmsnorm_bf16(const float*, const float*, uint16_t*, int32_t, int32_t, float, void*);
int32_t cuteafd_vision_layernorm_bf16(const float*, const float*, const float*, uint16_t*, int32_t, int32_t, float, void*);
int32_t cuteafd_vision_swiglu_bf16(const float*, const float*, uint16_t*, int64_t, int32_t, float, void*);
int32_t cuteafd_vision_gelu_bf16(const float*, uint16_t*, int64_t, int32_t, void*);

/* Scatter feature rows into [rows,copies,width] gathered token embeddings.
 * Indices must be unique (enforced by host admission); out-of-range rows are
 * ignored defensively. No allocation, no synchronization. Returns CUDA status. */
int32_t cuteafd_embed_inject(const uint16_t* features, const uint32_t* indices,
                            uint16_t* out, int32_t feature_rows, int32_t rows,
                            int32_t width, int32_t copies, void* stream);
#ifdef __cplusplus
}
#endif
