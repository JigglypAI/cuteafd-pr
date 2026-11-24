//! Packed target and routed-expert arena layouts for Inkling 2 Mini (TP1).
//!
//! Static target weights (globals, cross-pass queries, 24 physical layers, and 144
//! cross-pass controller matrices on core layers 6..18) pack into one preplanned
//! device arena. Combined execution views (`cross_pass_queries`, `forget_gain_weight`,
//! `qkv_input_norm.weight`) land directly from checkpoint safetensors slices via
//! multi-land GPU streaming with zero extra bytes and zero CPU transforms.
//! Routed experts pack into separate per-format arenas (NVFP4 layers 0..20 and
//! MXFP8 layers 20..24) matching the runtime's ownership boundaries.
use anyhow::{ensure, Context, Result};
use cuteafd_core::memory_layout::MemoryKind;
use cuteafd_core::DType;

use crate::families::inkling2_mini::Inkling2MiniConfig;
use crate::placement::planner::{weight_arena_components, Component, Lifetime, Location};
use crate::plan::checkpoint::Checkpoint;
use crate::plan::families::inkling2::{expert_stem, expert_storage, ExpertStorage, TEXT};
use crate::serving_capacity::{PackedWeightRegions, WeightArenaLayout};
use crate::shared::dense_read::{MountLoad, ReadBatch, STEP_BYTES};

pub const CONTROLLER_STREAMS: [&str; 4] = ["attn", "k", "v", "mlp"];
pub const CROSS_PASS_QUERIES: &str = "model.llm.middle_cycle_cross_pass_queries";
pub const CROSS_PASS_KV_QUERIES: &str = "model.llm.middle_cycle_cross_pass_kv_queries";

/// One contiguous row slice of a checkpoint tensor landing at `dst_offset` bytes
/// inside an admitted target arena view.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MiniSourceSlice {
    pub tensor: String,
    pub rows: std::ops::Range<usize>,
    pub columns: std::ops::Range<usize>,
    pub row_bytes: usize,
    pub dst_offset: usize,
}

impl MiniSourceSlice {
    pub fn bytes(&self) -> usize {
        self.rows.len() * self.columns.len()
    }
}

/// Header-verified TP1 target arena for Inkling 2 Mini.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MiniTargetWeights {
    pub layers: Vec<usize>,
    pub arena: WeightArenaLayout,
    pub core_start: usize,
    pub core_end: usize,
}

impl MiniTargetWeights {
    pub fn plan(checkpoint: &Checkpoint, cfg: &Inkling2MiniConfig) -> Result<Self> {
        let t = &cfg.text;
        let lp = &cfg.loop_cfg;
        let (h, v, d, k) = (t.hidden, t.vocab_size, t.head_dim, t.sconv_kernel);
        let q = t.heads * d;
        let kv = t.kv_heads * d;
        let r = t.heads * t.d_rel;
        let mut tensors: Vec<(String, usize)> = Vec::new();

        let check = |name: &str, dtype: DType, shape: &[usize]| -> Result<usize> {
            let header = checkpoint
                .header(name)
                .with_context(|| format!("Inkling 2 Mini arena requires checkpoint header {name}"))?;
            ensure!(
                header.dtype == dtype && header.shape == shape,
                "Inkling 2 Mini arena {name}: expected {dtype:?} {shape:?}, found {:?} {:?}",
                header.dtype,
                header.shape
            );
            let elem = match dtype {
                DType::Bf16 => 2,
                DType::F32 => 4,
                _ => anyhow::bail!("unsupported Inkling 2 Mini arena dtype {dtype:?}"),
            };
            let bytes = shape
                .iter()
                .try_fold(elem, |acc: usize, dim| acc.checked_mul(*dim))
                .context("Inkling 2 Mini arena byte overflow")?;
            ensure!(
                header.byte_length == bytes as u64,
                "Inkling 2 Mini arena {name}: header byte extent {} differs from {bytes}",
                header.byte_length
            );
            Ok(bytes)
        };

        let plain = |tensors: &mut Vec<(String, usize)>, name: String, dtype: DType, shape: &[usize]| -> Result<()> {
            let bytes = check(&name, dtype, shape)?;
            tensors.push((name, bytes));
            Ok(())
        };

        plain(&mut tensors, format!("{TEXT}embed.weight"), DType::Bf16, &[v, h])?;
        plain(&mut tensors, format!("{TEXT}embed_norm.weight"), DType::Bf16, &[h])?;
        plain(&mut tensors, format!("{TEXT}unembed.weight"), DType::Bf16, &[v, h])?;
        plain(&mut tensors, format!("{TEXT}norm.weight"), DType::Bf16, &[h])?;
        plain(&mut tensors, format!("{TEXT}attn_res.output_query"), DType::F32, &[h])?;
        // Validate both raw cross-pass query tables; their rows land per core layer in
        // `[attn, k, v, mlp]` order matching reference/model.py:178-181.
        check(CROSS_PASS_QUERIES, DType::F32, &[2 * lp.core_layers(), h])?;
        check(CROSS_PASS_KV_QUERIES, DType::F32, &[2 * lp.core_layers(), h])?;

        for layer in 0..t.layers {
            let root = format!("{TEXT}layers.{layer}");
            let local = t.local[layer];
            let a = format!("{root}.attn");
            for (proj, n, cols) in [
                ("wq_du", q, h),
                ("wr_du", r, h),
                ("wk_dv", kv, h),
                ("wv_dv", kv, h),
                ("gate_attn", q, h),
                ("attn_sink_proj", t.heads, h),
                ("wo_ud", h, q),
            ] {
                plain(&mut tensors, format!("{a}.{proj}.weight"), DType::Bf16, &[n, cols])?;
            }
            plain(&mut tensors, format!("{a}.q_l2_clip.scale"), DType::Bf16, &[t.heads, d])?;
            plain(&mut tensors, format!("{a}.rel_l2_clip.scale"), DType::Bf16, &[t.heads, t.d_rel])?;
            plain(
                &mut tensors,
                format!("{a}.rel_logits_proj.proj"),
                DType::Bf16,
                &[t.d_rel, if local { t.window } else { t.rel_extent }],
            )?;
            for name in ["k", "v"] {
                if local {
                    plain(&mut tensors, format!("{a}.{name}_sconv.weight"), DType::Bf16, &[kv, 1, k])?;
                } else {
                    plain(&mut tensors, format!("{a}.{name}_sconv.static_w"), DType::Bf16, &[k, kv])?;
                    plain(&mut tensors, format!("{a}.{name}_sconv.weight_proj"), DType::Bf16, &[t.kv_heads * k, h])?;
                }
            }
            for name in ["attn_sconv", "mlp_sconv"] {
                plain(&mut tensors, format!("{root}.{name}.weight"), DType::Bf16, &[h, 1, k])?;
            }
            if local {
                plain(&mut tensors, format!("{root}.attn_norm.weight"), DType::Bf16, &[h])?;
            } else {
                // Land [q_input_norm, k_input_norm, v_input_norm] back-to-back as one
                // `[3, H]` BF16 view so both individual norms (offsets 0, H*2, 2*H*2)
                // and LoopBlendNorm(slots=3, broadcast_weight=false) share one view.
                let mut total = 0usize;
                for norm in ["q_input_norm", "k_input_norm", "v_input_norm"] {
                    total += check(&format!("{root}.{norm}.weight"), DType::Bf16, &[h])?;
                }
                tensors.push((format!("{root}.qkv_input_norm.weight"), total));
            }
            plain(&mut tensors, format!("{root}.mlp_norm.weight"), DType::Bf16, &[h])?;
            plain(&mut tensors, format!("{root}.mlp.gate.weight"), DType::Bf16, &[t.experts, h])?;
            plain(&mut tensors, format!("{root}.mlp.gate.bias"), DType::F32, &[t.experts])?;
            plain(&mut tensors, format!("{TEXT}attn_res.mlp_queries.{layer}"), DType::F32, &[h])?;
            if layer > 0 {
                let shape: Vec<usize> = if local { vec![h] } else { vec![3, h] };
                plain(&mut tensors, format!("{TEXT}attn_res.attn_queries.{}", layer - 1), DType::F32, &shape)?;
            }
            if lp.is_core_layer(layer) {
                tensors.push((format!("{root}.cross_pass_queries"), 4 * h * 4));
                for stream in CONTROLLER_STREAMS {
                    let c = format!("{root}.controllers.{stream}");
                    plain(&mut tensors, format!("{c}.down_weight"), DType::Bf16, &[lp.gate_rank, 2 * h])?;
                    let f_bytes = check(&format!("{c}.forget_weight"), DType::Bf16, &[h, lp.gate_rank])?;
                    let g_bytes = check(&format!("{c}.gain_weight"), DType::Bf16, &[h, lp.gate_rank])?;
                    // Stacked `[2*H, gate_rank]` BF16 view for one fused `[forget; gain]` GEMM.
                    tensors.push((format!("{c}.forget_gain_weight"), f_bytes + g_bytes));
                }
            }
        }

        Ok(Self {
            layers: (0..t.layers).collect(),
            arena: WeightArenaLayout::new("inkling2_mini/target/0", tensors)?,
            core_start: lp.core_start,
            core_end: lp.core_end,
        })
    }

    /// Execution arena adds the 24 GPU-generated padded `rel_logits_proj` transposes (`1024 x 64 x BF16`).
    pub fn execution_arena(&self) -> Result<WeightArenaLayout> {
        let mut tensors: Vec<(String, usize)> = self
            .arena
            .views()
            .iter()
            .map(|(name, view)| (name.clone(), view.bytes()))
            .collect();
        for &layer in &self.layers {
            tensors.push((format!("{TEXT}layers.{layer}.attn.rel_logits_proj.padded"), 1024 * 64 * 2));
        }
        WeightArenaLayout::new(self.arena.id(), tensors)
    }

    /// Resolves every checkpoint tensor slice that lands into arena view `view_name`.
    pub fn sources(&self, checkpoint: &Checkpoint, view_name: &str) -> Result<Vec<MiniSourceSlice>> {
        let whole = |name: &str, dst_offset: usize| -> Result<MiniSourceSlice> {
            let header = checkpoint
                .header(name)
                .with_context(|| format!("Inkling 2 Mini source lacks {name}"))?;
            let count = if header.shape.len() >= 2 { header.shape[0] } else { 1 };
            let bytes = usize::try_from(header.byte_length)?;
            ensure!(count > 0 && bytes.is_multiple_of(count), "invalid source rows for {name}");
            let row_bytes = bytes / count;
            Ok(MiniSourceSlice {
                tensor: name.to_owned(),
                rows: 0..count,
                columns: 0..row_bytes,
                row_bytes,
                dst_offset,
            })
        };

        if let Some(rest) = view_name.strip_prefix(&format!("{TEXT}layers.")) {
            if let Some((layer_str, suffix)) = rest.split_once('.') {
                let layer: usize = layer_str.parse().context("invalid Mini layer view index")?;
                let root = format!("{TEXT}layers.{layer}");
                if suffix == "qkv_input_norm.weight" {
                    let mut out = Vec::with_capacity(3);
                    let mut offset = 0usize;
                    for norm in ["q_input_norm", "k_input_norm", "v_input_norm"] {
                        let s = whole(&format!("{root}.{norm}.weight"), offset)?;
                        offset += s.bytes();
                        out.push(s);
                    }
                    return Ok(out);
                }
                if suffix == "cross_pass_queries" {
                    ensure!(
                        (self.core_start..self.core_end).contains(&layer),
                        "cross_pass_queries outside core layers: {layer}"
                    );
                    let core_idx = layer - self.core_start;
                    let q_hdr = checkpoint.header(CROSS_PASS_QUERIES).context("cross_pass_queries header")?;
                    let kv_hdr = checkpoint.header(CROSS_PASS_KV_QUERIES).context("cross_pass_kv_queries header")?;
                    let row_bytes = usize::try_from(q_hdr.byte_length)? / q_hdr.shape[0];
                    ensure!(usize::try_from(kv_hdr.byte_length)? / kv_hdr.shape[0] == row_bytes, "cross-pass width differs");
                    // [attn, k, v, mlp] matching reference/model.py:178-181.
                    return Ok(vec![
                        MiniSourceSlice {
                            tensor: CROSS_PASS_QUERIES.into(),
                            rows: 2 * core_idx..2 * core_idx + 1,
                            columns: 0..row_bytes,
                            row_bytes,
                            dst_offset: 0,
                        },
                        MiniSourceSlice {
                            tensor: CROSS_PASS_KV_QUERIES.into(),
                            rows: 2 * core_idx..2 * core_idx + 2,
                            columns: 0..row_bytes,
                            row_bytes,
                            dst_offset: row_bytes,
                        },
                        MiniSourceSlice {
                            tensor: CROSS_PASS_QUERIES.into(),
                            rows: 2 * core_idx + 1..2 * core_idx + 2,
                            columns: 0..row_bytes,
                            row_bytes,
                            dst_offset: 3 * row_bytes,
                        },
                    ]);
                }
                if let Some(stem) = suffix.strip_suffix(".forget_gain_weight") {
                    let forget = whole(&format!("{root}.{stem}.forget_weight"), 0)?;
                    let gain = whole(&format!("{root}.{stem}.gain_weight"), forget.bytes())?;
                    return Ok(vec![forget, gain]);
                }
            }
        }
        Ok(vec![whole(view_name, 0)?])
    }

    /// Bounded mount staging demand across every checkpoint slice landed into the target arena.
    pub fn mount_load(&self, checkpoint: &Checkpoint) -> Result<MountLoad> {
        let mut load = MountLoad::default();
        for (name, view) in self.arena.views() {
            let slices = self.sources(checkpoint, name)?;
            let total: usize = slices.iter().map(MiniSourceSlice::bytes).sum();
            ensure!(
                total == view.bytes(),
                "Inkling 2 Mini mount {name}: source slices ({total}) differ from admitted view ({})",
                view.bytes()
            );
            for s in slices {
                load.merge(MountLoad::read(ReadBatch::new(
                    s.rows.len(),
                    s.row_bytes,
                    s.columns.len(),
                    STEP_BYTES,
                )?));
            }
        }
        Ok(load)
    }

    pub fn weight_components(&self, maximum: u64) -> Result<Vec<Component>, crate::placement::planner::Refusal> {
        let arena = self
            .execution_arena()
            .map_err(|error| crate::placement::planner::Refusal::InvalidFact(format!("{error:#}")))?;
        weight_arena_components(&arena, Location::coordinator(0), Lifetime::RESIDENT, maximum, MemoryKind::Weights)
    }
}

/// Header-verified TP1 routed-expert residency for one Inkling 2 Mini layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MiniLayerExperts {
    pub storage: ExpertStorage,
    pub resident_bytes: u64,
    pub source_bytes: u64,
}

/// Validates and computes TP1 resident bytes for all 24 physical routed layers.
pub fn layer_experts(checkpoint: &Checkpoint, cfg: &Inkling2MiniConfig) -> Result<Vec<MiniLayerExperts>> {
    let t = &cfg.text;
    let (e, h, i) = (t.experts as u64, t.hidden as u64, t.intermediate as u64);
    (0..t.layers)
        .map(|layer| {
            let storage = expert_storage(checkpoint, layer)?;
            let (pack, group, dtype, scale_dtype) = match storage {
                ExpertStorage::Mxfp8 => (1u64, 32u64, DType::F8E4M3, DType::U8),
                ExpertStorage::Nvfp4 => (2u64, 16u64, DType::U8, DType::F8E4M3),
                ExpertStorage::Exl3 { bits } => {
                    anyhow::bail!("Inkling 2 Mini layer {layer}: EXL3 K{bits} is not supported; use native NVFP4/MXFP8");
                }
            };
            let w13 = expert_stem(layer, "w13");
            let w2 = expert_stem(layer, "w2");
            let get = |name: &str| {
                checkpoint
                    .header(name)
                    .with_context(|| format!("Inkling 2 Mini expert inventory lacks {name}"))
            };
            let weight13 = get(&format!("{w13}_weight"))?;
            let weight2 = get(&format!("{w2}_weight"))?;
            let scale13 = get(&format!("{w13}_weight_scale"))?;
            let scale2 = get(&format!("{w2}_weight_scale"))?;
            let check = |meta: &crate::SafetensorsTensorMetadata, expected_dtype: &DType, shape: [u64; 3]| -> Result<()> {
                ensure!(
                    meta.dtype == *expected_dtype
                        && meta.shape.iter().map(|&d| d as u64).collect::<Vec<_>>() == shape,
                    "{}: expected {expected_dtype:?} {shape:?}, found {:?} {:?}",
                    meta.name,
                    meta.dtype,
                    meta.shape
                );
                let bytes = shape.iter().product::<u64>();
                ensure!(
                    meta.byte_length == bytes,
                    "{}: header byte extent {} differs from {bytes}",
                    meta.name,
                    meta.byte_length
                );
                Ok(())
            };
            check(weight13, &dtype, [e, 2 * i, h / pack])?;
            check(weight2, &dtype, [e, h, i / pack])?;
            check(scale13, &scale_dtype, [e, 2 * i, h / group])?;
            check(scale2, &scale_dtype, [e, h, i / group])?;
            let mut extras = 0u64;
            if storage == ExpertStorage::Nvfp4 {
                for stem in [&w13, &w2] {
                    let amax = get(&format!("{stem}_weight_amax"))?;
                    let denom = get(&format!("{stem}_weight_outer_scale_denominator"))?;
                    ensure!(
                        amax.dtype == DType::F32
                            && (amax.shape == [t.experts] || amax.shape == [t.experts, 1, 1])
                            && amax.byte_length == 4 * e,
                        "{}: expected F32 [{e}], found {:?} {:?}",
                        amax.name,
                        amax.dtype,
                        amax.shape
                    );
                    ensure!(
                        denom.dtype == DType::F32
                            && denom.shape.iter().product::<usize>() == 1
                            && denom.byte_length == 4,
                        "{}: expected scalar F32, found {:?} {:?}",
                        denom.name,
                        denom.dtype,
                        denom.shape
                    );
                    extras += amax.byte_length + denom.byte_length;
                }
            }
            let source_bytes =
                weight13.byte_length + weight2.byte_length + scale13.byte_length + scale2.byte_length + extras;
            let resident_bytes = if storage == ExpertStorage::Mxfp8 {
                let sizes = [
                    e * i * h,
                    e * i * (h / group),
                    e * i * h,
                    e * i * (h / group),
                    e * h * i,
                    e * h * (i / group),
                ];
                PackedWeightRegions::new(
                    sizes.into_iter().map(usize::try_from).collect::<std::result::Result<Vec<_>, _>>()?,
                )?
                .bytes as u64
            } else {
                let globals = (e + 1) * 4;
                let sizes = [
                    e * i * h / pack,
                    e * i * (h / group) + globals,
                    e * i * h / pack,
                    e * i * (h / group) + globals,
                    e * h * i / pack,
                    e * h * (i / group) + globals,
                ];
                PackedWeightRegions::new(
                    sizes.into_iter().map(usize::try_from).collect::<std::result::Result<Vec<_>, _>>()?,
                )?
                .bytes as u64
            };
            Ok(MiniLayerExperts {
                storage,
                resident_bytes,
                source_bytes,
            })
        })
        .collect()
}
