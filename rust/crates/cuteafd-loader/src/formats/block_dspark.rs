//! Checkpoint-native geometry for block dSpark drafters.
//! This is admission metadata, not a claim that a matching GPU export exists.
use serde::Deserialize;
use serde_json::Value;
use std::path::Path;

pub mod recurrent;

/// The native Markov kernel tiles independent sequences over fixed 32-sequence arrays.
pub const MAX_SEQUENCES: usize = cuteafd_core::concurrency::MAX_CONCURRENCY as usize;
/// Tiled reductions and context spans stream through one drafter-private slot.
pub const BLOCK_TP2_EXCHANGE_BYTES: usize = 65_536;
pub const fn supports_sequences(sequences: usize) -> bool {
    sequences > 0 && sequences <= MAX_SEQUENCES
}

/// Physical native runtime owners, shared by admission and construction.
#[derive(Debug, Clone)]
pub struct NativeDsparkLayout {
    pub snapshot: std::path::PathBuf,
    pub weights: crate::serving_capacity::WeightArenaLayout,
    pub buffers: Vec<(String, usize)>,
    pub host_landing: usize,
    pub verify_rows: usize,
    pub mark_bytes: usize,
    pub marks: usize,
}

impl NativeDsparkLayout {
    pub fn new(snapshot:&Path, slots:usize, prefill_rows:usize, waves:usize, sms:usize,
        prefix_entries:usize, mark_budget:u64, host_prefix:bool) -> anyhow::Result<Self> {
        use anyhow::{ensure,Context};
        use crate::families::glm5::draft_representation::GlmDraftRepresentation;
        let cfg = BlockDsparkConfig::read(snapshot)?;
        ensure!(cfg.dialect == DsparkDialect::MiniMax && slots > 0 && supports_sequences(slots)
            && prefill_rows > 0 && prefill_rows <= 4096 && waves > 0 && waves <= 4,
            "native block drafter capacity differs from checkpoint");
        ensure!((cfg.hidden,cfg.heads,cfg.kv_heads,cfg.head_dim,cfg.intermediate,cfg.layers,cfg.block,cfg.vocab)
            == (6144,64,4,128,12288,5,7,200064) && cfg.taps == [3,17,31,45,59]
            && cfg.window == Some(4096), "native drafter layout requires installed MiniMax release geometry");
        let weights = crate::serving_capacity::WeightArenaLayout::new("minimax/draft",
            block_dspark_arena_requests(snapshot,GlmDraftRepresentation::Bf16Only)?)?;
        let (h,q,kv,inter,taps) = (cfg.hidden,cfg.heads * cfg.head_dim,cfg.kv_heads * cfg.head_dim,
            cfg.intermediate,cfg.taps.len());
        let rows = slots.checked_mul(cfg.block).context("native draft rows overflow")?;
        let context_rows = 2048usize;
        let mut buffers = Vec::new();
        let mut add = |name:String,bytes:usize| { buffers.push((name,crate::serving_capacity::WeightArenaLayout::allocation_bytes(bytes))); };
        let window = cfg.window.context("native block drafter has no window")?;
        for layer in 0..cfg.layers {
            for plane in ["k_ring","v_ring"] { add(format!("layer.{layer}.{plane}"),slots * window * kv * 2); }
        }
        for (name,bytes) in [("taps",context_rows * taps * h * 2),("fused",context_rows * h * 2),
            ("fused_norm",context_rows * h * 2),("context_kv",context_rows * kv * 4),
            ("context_positions",context_rows * 8),("context_slots",context_rows * 4),
            ("native.norm-input",context_rows * h * 4),("native.norm-rope",context_rows * q * 2),
            ("native.gate",context_rows * inter * 2),("native.up",context_rows * inter * 2),
            ("native.projections.activation",context_rows * inter.max(q).max(h) * 2),
            ("native.projections.inputs",8),("native.projections.records",context_rows.div_ceil(16) * (inter.max(q).max(h) / 32) * 528),
            ("native.projections.output",context_rows * inter.max(q).max(h) * 4)] { add(name.into(),bytes); }
        for (name,width,bytes) in [("h",h,2),("n",h,2),("qkv",q + 2 * kv,2),("q",q,2),
            ("k",kv,2),("v",kv,2),("attn",q,2),("delta",h,2),("gate_up",2 * inter,2),
            ("act",inter,2),("logits",cfg.vocab,4),("ids",1,4),("tokens",1,4),("confidence",1,4),
            ("positions",1,8)] { add(format!("workspace.{name}"),rows * width * bytes); }
        add("workspace.anchors".into(),slots * 4);
        add("workspace.tables".into(),slots * 12);
        add("workspace.head".into(),cuteafd_ffi::programs::VOCABULARY_HEAD_WORKSPACE);
        add("workspace.markov".into(),slots * 296 * 8 + rows * 4);
        add("workspace.attention".into(),usize::try_from(crate::serving_capacity::draft_attention_workspace_bytes(
            slots as u64,cfg.kv_heads as u64,(cfg.heads / cfg.kv_heads * cfg.block).div_ceil(32) as u64,
            (window + cfg.block) as u64,cfg.head_dim as u64,sms as u64)?)?);
        let verify_rows = slots.checked_mul(cfg.block + 1).context("native verify rows overflow")?.min(64);
        add("verify.logits".into(),verify_rows * cfg.vocab * 4);
        add("verify.taps".into(),verify_rows * taps * h * 2);
        for slot in 0..waves {
            add(format!("unit.{slot}.target_taps.narrow"),prefill_rows * h * 2);
            add(format!("unit.{slot}.target_taps.output"),prefill_rows * taps * h * 2);
        }
        add("decode.target_taps.narrow".into(),verify_rows * h * 2);
        add("decode.target_taps.output".into(),verify_rows * taps * h * 2);
        let mark_bytes = (16 + cfg.layers * 2 * window * kv * 2).div_ceil(256) * 256;
        let marks = cuteafd_core::prefix::mark_slots_with_host(slots as u64,prefix_entries as u64,
            mark_bytes as u64,mark_budget,host_prefix) as usize;
        if marks > 0 { add("prefix.marks".into(),mark_bytes.checked_mul(marks).context("native draft marks overflow")?); }
        Ok(Self { snapshot:snapshot.to_owned(),weights,buffers,host_landing:rows * 8,verify_rows,mark_bytes,marks })
    }
}

/// Physical owners of the shared Qwen3 block executor, excluding target taps/head.
/// Window rings are fixed by checkpoint geometry, independent of the target pool.
#[derive(Debug, Clone, PartialEq)]
pub struct BlockDsparkRuntimeLayout {
    pub representation: crate::families::glm5::draft_representation::GlmDraftRepresentation,
    pub config: BlockDsparkConfig,
    pub kv_cache: crate::families::qwen4::Qwen4KvCache,
    pub weight_requests: Vec<(String, usize)>,
    pub capacity: crate::families::glm5::draft_representation::GlmDraftCapacity,
    pub weights: crate::serving_capacity::WeightArenaLayout,
    pub buffers: Vec<(String, usize)>,
    pub host_landing: usize,
    /// Sole shared GqaHeadSplit allocation, not allocated from `buffers`.
    pub reduction_bytes: usize,
    pub reduction_peer: bool,
    /// Allocated only by the retained PeerExchange, never the body workspace.
    pub exchange_buffers: Vec<(String,usize)>,
}

impl BlockDsparkRuntimeLayout {
    pub fn new(snapshot: &Path, slots: usize, sequences: usize, sms: usize, arena: &str)
        -> anyhow::Result<Self> {
        Self::with_kv(snapshot,slots,sequences,sms,arena,crate::families::qwen4::Qwen4KvCache::Bf16)
    }
    pub fn with_kv(snapshot: &Path, slots: usize, sequences: usize, sms: usize, arena: &str,
        kv_cache:crate::families::qwen4::Qwen4KvCache) -> anyhow::Result<Self> {
        Self::with_representation(snapshot, slots, sequences, sms, arena, kv_cache,
            crate::families::glm5::draft_representation::GlmDraftRepresentation::Bf16Only)
    }
    pub fn with_representation(snapshot: &Path, slots: usize, sequences: usize, sms: usize, arena: &str,
        kv_cache: crate::families::qwen4::Qwen4KvCache,
        representation: crate::families::glm5::draft_representation::GlmDraftRepresentation) -> anyhow::Result<Self> {
        use anyhow::{ensure, Context};
        use crate::families::glm5::draft_representation::GlmDraftCapacity;
        let config = BlockDsparkConfig::read(snapshot)?;
        ensure!(config.dialect == DsparkDialect::Qwen3 && config.backbone.is_qwen3(config.head_dim)
            && matches!(config.head_dim, 64 | 128) && matches!(config.rank, 256 | 512)
            && supports_sequences(sequences) && sequences <= slots && sms > 0,
            "unsupported shared Qwen3 block drafter arithmetic/capacity");
        let capacity = GlmDraftCapacity::new(slots, sequences, config.block)?;
        let window = config.window.context("block runtime layout requires checkpoint sliding window")?;
        let cache = config.paged_cache_layout(slots, window, slots.checked_mul(window)
            .context("block ring capacity overflow")?, kv_cache)?;
        let weight_requests = block_dspark_arena_requests(snapshot, representation)?;
        let weights = crate::serving_capacity::WeightArenaLayout::new(arena, weight_requests.clone())?;
        let mul = |values: &[usize]| values.iter().try_fold(1usize, |n, value| n.checked_mul(*value))
            .context("block runtime owner extent overflow");
        let add = |values: &[usize]| values.iter().try_fold(0usize, |n, value| n.checked_add(*value))
            .context("block runtime width overflow");
        let (h, inter, rows) = (config.hidden, config.intermediate, capacity.block_rows);
        let q = mul(&[config.heads, config.head_dim])?;
        let kv = mul(&[config.kv_heads, config.head_dim])?;
        let qkv = add(&[q, mul(&[2,kv])?])?;
        let mut buffers = Vec::new();
        let mut push = |name: String, bytes: usize| buffers.push((name,crate::serving_capacity::WeightArenaLayout::allocation_bytes(bytes)));
        for layer in 0..config.layers {
            for plane in ["k_ring", "v_ring"] {
                push(format!("layer.{layer}.{plane}"), cache.layer_bytes[layer] / 2);
            }
        }
        for (name,width,bytes) in [("taps",mul(&[config.taps.len(),h])?,2), ("fused",h,2),
            ("fused_norm",h,2), ("context_kv",mul(&[2,kv])?,2),
            ("context_positions",1,8), ("context_slots",1,4)] {
            push(name.into(),mul(&[2048,width,bytes])?);
        }
        for (name,width,bytes) in [("h",h,2),("n",h,2),("qkv",qkv,2),("q",q,2),
            ("k",kv,2),("v",kv,2),("attn",q,2),("delta",h,2),
            ("gate_up",mul(&[2,inter])?,2),("act",inter,2),("logits",config.vocab,4),
            ("ids",1,4),("tokens",1,4),("confidence",1,4),("positions",1,8)] {
            push(format!("workspace.{name}"),mul(&[rows,width,bytes])?);
        }
        push("workspace.anchors".into(),mul(&[sequences,4])?);
        push("workspace.tables".into(),mul(&[sequences,12])?);
        push("workspace.head".into(),cuteafd_ffi::programs::VOCABULARY_HEAD_WORKSPACE);
        push("workspace.markov".into(),add(&[mul(&[sequences,296,8])?,mul(&[rows,4])?])?);
        push("workspace.attention".into(),usize::try_from(crate::serving_capacity::draft_attention_workspace_bytes(
            sequences as u64, config.kv_heads as u64,
            mul(&[config.heads / config.kv_heads, config.block])?.div_ceil(32) as u64,
            add(&[window, config.block])? as u64, config.head_dim as u64, sms as u64)?)?);
        let mut layout = Self { representation, config, kv_cache, weight_requests, capacity, weights, buffers,
            host_landing:mul(&[rows,8])?, reduction_bytes:0,reduction_peer:false,exchange_buffers:Vec::new() };
        layout.set_fp8_scratch(sms)?;
        Ok(layout)
    }

    /// Rank-local body workspace sized from the source checkpoint. Each peer
    /// retains full residual rows but only its own heads, neurons and KV planes.
    pub fn for_rank(snapshot:&Path,slots:usize,sequences:usize,sms:usize,arena:&str,
        kv_cache:crate::families::qwen4::Qwen4KvCache,rank:usize,ranks:usize,lead:usize)->anyhow::Result<Self> {
        Self::for_rank_representation(snapshot, slots, sequences, sms, arena, kv_cache, rank, ranks, lead,
            crate::families::glm5::draft_representation::GlmDraftRepresentation::Bf16Only)
    }
    pub fn for_rank_representation(snapshot:&Path,slots:usize,sequences:usize,sms:usize,arena:&str,
        kv_cache:crate::families::qwen4::Qwen4KvCache,rank:usize,ranks:usize,lead:usize,
        representation:crate::families::glm5::draft_representation::GlmDraftRepresentation)->anyhow::Result<Self> {
        use anyhow::{ensure,Context};
        ensure!(matches!(ranks,1|2) && rank<ranks && lead<ranks,"invalid block draft rank");
        let mut layout=Self::with_representation(snapshot,slots,sequences,sms,arena,kv_cache,representation)?;
        if ranks==1 { return Ok(layout); }
        let source=layout.config.clone();
        ensure!(source.heads%ranks==0 && source.kv_heads%ranks==0 && source.intermediate%ranks==0,
            "block draft rank split cuts checkpoint dimensions");
        layout.config.heads/=ranks; layout.config.kv_heads/=ranks; layout.config.intermediate/=ranks;
        layout.weight_requests=block_dspark_rank_arena_requests_with_representation(snapshot,rank,ranks,lead,representation)?;
        layout.weights=crate::serving_capacity::WeightArenaLayout::new(arena,layout.weight_requests.clone())?;
        let (h,q,kv,inter,rows)=(source.hidden,layout.config.heads*source.head_dim,
            layout.config.kv_heads*source.head_dim,layout.config.intermediate,layout.capacity.block_rows);
        let mul=|v:&[usize]|v.iter().try_fold(1usize,|n,x|n.checked_mul(*x)).context("block rank owner overflow");
        let window=source.window.context("block rank window missing")?;
        for (name,bytes) in &mut layout.buffers {
            let resize=match name.as_str() {
                n if n.ends_with(".k_ring") || n.ends_with(".v_ring")=>Some(mul(&[slots,window,
                    kv_cache.plane_record_bytes(layout.config.kv_heads,source.head_dim)])?),
                "context_kv"=>Some(mul(&[2048,2,kv,2])?),
                "workspace.qkv"=>Some(mul(&[rows,q+2*kv,2])?),
                "workspace.q"|"workspace.attn"=>Some(mul(&[rows,q,2])?),
                "workspace.k"|"workspace.v"=>Some(mul(&[rows,kv,2])?),
                "workspace.gate_up"=>Some(mul(&[rows,2,inter,2])?),
                "workspace.act"=>Some(mul(&[rows,inter,2])?),
                "workspace.attention"=>Some(usize::try_from(crate::serving_capacity::draft_attention_workspace_bytes(
                    sequences as u64,layout.config.kv_heads as u64,
                    (layout.config.heads/layout.config.kv_heads*source.block).div_ceil(32) as u64,
                    (window+source.block) as u64,source.head_dim as u64,sms as u64)?)?),
                _=>None,
            };
            if let Some(value)=resize { *bytes=crate::serving_capacity::WeightArenaLayout::allocation_bytes(value); }
        }
        if rank!=lead {
            layout.buffers.retain(|(name,_)|!matches!(name.as_str(),"workspace.logits"|"workspace.tokens"
                |"workspace.confidence"|"workspace.head"|"workspace.markov"));
            layout.host_landing=0;
        }
        layout.reduction_bytes=crate::serving_capacity::WeightArenaLayout::allocation_bytes(mul(&[rows,h,2])?);
        layout.reduction_peer=rank!=lead;
        let transport=crate::serving_capacity::PeerStorageLayout::new(1,BLOCK_TP2_EXCHANGE_BYTES,
            crate::serving_capacity::peer_wait_diagnostics())?;
        layout.exchange_buffers.extend([("tp2.exchange.recv".into(),transport.receive_bytes),
            ("tp2.exchange.control".into(),transport.control_bytes),
            ("tp2.exchange.abort".into(),crate::serving_capacity::WeightArenaLayout::ALIGNMENT)]);
        if transport.progress_bytes>0 { layout.exchange_buffers.push(("tp2.exchange.progress".into(),transport.progress_bytes)); }
        layout.set_fp8_scratch(sms)?;
        Ok(layout)
    }

    fn set_fp8_scratch(&mut self, sms: usize) -> anyhow::Result<()> {
        self.buffers.retain(|(name,_)| name != "workspace.fp8-linear");
        if self.representation == crate::families::glm5::draft_representation::GlmDraftRepresentation::Fp8Only {
            let rows = 2048usize.max(self.capacity.block_rows);
            let bytes = self.projection_shapes().into_iter().map(|(k,n)|
                crate::families::glm5::draft_representation::draft_fp8_scratch_bytes(rows as u64,k as u64,n as u64,sms as u64,0))
                .max().unwrap_or(256);
            self.buffers.push(("workspace.fp8-linear".into(), usize::try_from(bytes)?.max(256)));
        }
        Ok(())
    }

    pub fn projection_shapes(&self)->Vec<(usize,usize)> {
        let c=&self.config;
        let (h,q,kv)=(c.hidden,c.heads*c.head_dim,c.kv_heads*c.head_dim);
        vec![(c.taps.len()*h,h),(h,q+2*kv),(h,2*kv),(q,h),(h,2*c.intermediate),(c.intermediate,h)]
    }

    pub fn validate_projection_programs(&self,manifest:&serde_json::Value)->anyhow::Result<Vec<String>> {
        if self.representation == crate::families::glm5::draft_representation::GlmDraftRepresentation::Fp8Only {
            return Ok(Vec::new());
        }
        let rows=2048usize.max(self.capacity.block_rows).max(64);
        self.projection_shapes().into_iter().map(|(k,n)|Ok(cuteafd_ffi::draft_projection::validate_package_geometry(
            manifest,k,n,rows)?)).collect()
    }

    pub fn window_mark_bytes(&self) -> anyhow::Result<usize> {
        use anyhow::Context;
        let window = self.config.window.context("block runtime window missing")?;
        [self.config.layers,2,window,self.kv_cache.plane_record_bytes(self.config.kv_heads,self.config.head_dim)]
            .into_iter().try_fold(1usize, |n,v| n.checked_mul(v)).context("block snapshot extent overflow")?
            .checked_add(16).and_then(|n| n.checked_add(255)).map(|n| n / 256 * 256)
            .context("block snapshot alignment overflow")
    }

    pub fn demands(&self, gpu: u8) -> Vec<crate::placement::Demand> {
        use cuteafd_core::memory_layout::{Basis, Category};
        let mut demands=self.buffers.iter().map(|(name, bytes)| crate::placement::Demand::new(gpu,
            Category::Drafter, format!("block dSpark {name}"), *bytes as u64, Basis::Exact)).collect::<Vec<_>>();
        demands.extend(self.exchange_buffers.iter().map(|(name,bytes)|crate::placement::Demand::new(gpu,
            Category::Transport,format!("block dSpark {name}"),*bytes as u64,Basis::Exact)));
        if self.reduction_bytes>0 {
            demands.push(crate::placement::Demand::new(gpu,if self.reduction_peer { Category::Transport } else { Category::Drafter },
                "block dSpark shared TP2 reduction",self.reduction_bytes as u64,Basis::Exact));
        }
        demands
    }
}

#[derive(Debug, thiserror::Error)]
pub enum DsparkConfigError {
    #[error(transparent)]
    Topk(#[from] super::dflash::DflashTopkVocabError),
    #[error("reading dSpark config: {0}")]
    Io(#[from] std::io::Error),
    #[error("parsing dSpark config: {0}")]
    Json(#[from] serde_json::Error),
    #[error("unsupported dSpark checkpoint: {0}")]
    Unsupported(String),
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(tag = "rope_type")]
pub enum DsparkRope {
    #[serde(rename = "default")]
    Default { rope_theta: f64 },
    #[serde(rename = "yarn")]
    Yarn {
        rope_theta: f64,
        factor: f64,
        original_max_position_embeddings: usize,
        #[serde(default = "default_beta_fast")]
        beta_fast: f64,
        #[serde(default = "default_beta_slow")]
        beta_slow: f64,
        #[serde(default)]
        mscale: Option<f64>,
        #[serde(default)]
        mscale_all_dim: Option<f64>,
    },
}

fn default_beta_fast() -> f64 { 32.0 }
fn default_beta_slow() -> f64 { 1.0 }

/// Immutable draft-only rotary inputs. The legacy 64-wide default route
/// computes its frequencies in the kernel instead of consuming this table.
#[derive(Debug, Clone, PartialEq)]
pub struct DsparkRotaryPlan {
    pub inv_freq: Vec<f32>,
    /// Magnitude applied to rotary query/key coordinates.
    pub attention_factor: f32,
    /// Additional MLA softmax multiplier, separate from rotary magnitude.
    pub softmax_factor: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DsparkNorm { Rms, Gemma }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DsparkResidual { Bf16, F32 }

/// Arithmetic belongs to the checkpoint, independently of the draft lifecycle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DsparkBackbone {
    pub norm: DsparkNorm,
    pub residual: DsparkResidual,
    pub rotary_dim: usize,
    pub activation: cuteafd_core::FfnActivation,
}

impl DsparkBackbone {
    pub fn from_layers(raw: &Value, head_dim: usize) -> Result<Self, DsparkConfigError> {
        use DsparkConfigError::Unsupported;
        let norm = match raw.get("use_gemma_norm") {
            None | Some(Value::Bool(false)) => DsparkNorm::Rms,
            Some(Value::Bool(true)) => DsparkNorm::Gemma,
            _ => return Err(Unsupported("use_gemma_norm must be boolean".into())),
        };
        let rotary_dim = match raw.get("rotary_dim") {
            None => head_dim,
            Some(value) => value.as_u64().and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| Unsupported("invalid rotary_dim".into()))?,
        };
        if rotary_dim == 0 || rotary_dim > head_dim || rotary_dim % 2 != 0 {
            return Err(Unsupported("rotary_dim must be positive, even and no wider than a head".into()));
        }
        if let Some(value) = raw.get("partial_rotary_factor") {
            let factor = value.as_f64().filter(|n| n.is_finite())
                .ok_or_else(|| Unsupported("invalid partial_rotary_factor".into()))?;
            if (factor * head_dim as f64 - rotary_dim as f64).abs() > 1e-9 {
                return Err(Unsupported("partial_rotary_factor disagrees with rotary_dim".into()));
            }
        }
        let activation = match raw["hidden_act"].as_str() {
            Some("silu") => cuteafd_core::FfnActivation::SiluBf16Product { limit: None },
            Some("swigluoai") => {
                let number = |name: &str| raw[name].as_f64().map(|n| n as f32)
                    .ok_or_else(|| Unsupported(format!("missing or invalid {name}")));
                cuteafd_core::FfnActivation::SwigluOai {
                    alpha: number("swiglu_alpha")?, beta: 1.0, limit: number("swiglu_limit")?,
                }
            }
            _ => return Err(Unsupported("unsupported dSpark FFN activation".into())),
        }.validate().map_err(|error| Unsupported(error.into()))?;
        let residual = match raw["residual_dtype"].as_str() {
            None if raw.get("residual_dtype").is_none() => DsparkResidual::Bf16,
            Some("bfloat16") => DsparkResidual::Bf16,
            Some("float32") => DsparkResidual::F32,
            _ => return Err(Unsupported("invalid residual_dtype".into())),
        };
        Ok(Self { norm, residual, rotary_dim, activation })
    }

    pub fn is_qwen3(self, head_dim: usize) -> bool {
        self.norm == DsparkNorm::Rms && self.residual == DsparkResidual::Bf16 && self.rotary_dim == head_dim
            && self.activation == (cuteafd_core::FfnActivation::SiluBf16Product { limit: None })
    }
}

impl DsparkRope {
    pub fn plan(&self, head_dim: usize) -> Result<DsparkRotaryPlan, DsparkConfigError> {
        use DsparkConfigError::Unsupported;
        if head_dim == 0 || head_dim % 2 != 0 {
            return Err(Unsupported(
                "rotary head dimension must be positive and even".into(),
            ));
        }
        let (theta, yarn) = match *self {
            Self::Default { rope_theta } => (rope_theta, None),
            Self::Yarn {
                rope_theta,
                factor,
                original_max_position_embeddings,
                beta_fast,
                beta_slow,
                mscale,
                mscale_all_dim,
            } => {
                if !factor.is_finite()
                    || factor < 1.0
                    || original_max_position_embeddings == 0
                    || !beta_fast.is_finite()
                    || !beta_slow.is_finite()
                    || beta_fast <= beta_slow
                    || beta_slow <= 0.0
                    || [mscale, mscale_all_dim].into_iter().flatten()
                        .any(|scale| !scale.is_finite() || scale < 0.0)
                {
                    return Err(Unsupported("invalid YaRN rotary parameters".into()));
                }
                (
                    rope_theta,
                    Some((
                        factor,
                        original_max_position_embeddings,
                        beta_fast,
                        beta_slow,
                    )),
                )
            }
        };
        if !theta.is_finite() || theta <= 1.0 || !(theta as f32).is_finite() {
            return Err(Unsupported(
                "rotary theta must be finite and greater than one in FP32".into(),
            ));
        }
        let (attention_factor, softmax_factor) = match *self {
            Self::Yarn { factor, mscale, mscale_all_dim, .. } => {
                let magnitude = |scale: f64| if factor <= 1.0 { 1.0 }
                    else { 1.0 + 0.1 * scale * factor.ln() };
                // MLA separates the rotary-coordinate magnitude from the
                // all-dimension query scale. GQA without these fields keeps HF YaRN.
                let all = magnitude(mscale_all_dim.unwrap_or(0.0));
                ((magnitude(mscale.unwrap_or(1.0)) / all) as f32, (all * all) as f32)
            }
            Self::Default { .. } => (1.0, 1.0),
        };
        let correction = yarn.map(|(_, original, fast, slow)| {
            let dimension = |rotations: f64| {
                head_dim as f64 * (original as f64 / (rotations * 2.0 * std::f64::consts::PI)).ln()
                    / (2.0 * theta.ln())
            };
            let low = dimension(fast).floor().max(0.0) as f32;
            let mut high = dimension(slow).ceil().min((head_dim - 1) as f64) as f32;
            if low == high {
                high += 0.001;
            }
            (low, high)
        });
        let inv_freq = (0..head_dim / 2)
            .map(|i| {
                // Preserve the reference's FP32 operation boundaries, including
                // interpolation before blending (not division of extrapolation).
                let frequency = (theta as f32).powf((2 * i) as f32 / head_dim as f32);
                let extrapolation = 1.0f32 / frequency;
                match (yarn, correction) {
                    (Some((factor, _, _, _)), Some((low, high))) => {
                        let weight = 1.0 - ((i as f32 - low) / (high - low)).clamp(0.0, 1.0);
                        let interpolation = 1.0f32 / (factor as f32 * frequency);
                        interpolation * (1.0 - weight) + extrapolation * weight
                    }
                    _ => extrapolation,
                }
            })
            .collect::<Vec<_>>();
        if !attention_factor.is_finite() || !softmax_factor.is_finite()
            || inv_freq.iter().any(|x| !x.is_finite() || *x <= 0.0) {
            return Err(Unsupported(
                "rotary plan is not representable in FP32".into(),
            ));
        }
        Ok(DsparkRotaryPlan {
            inv_freq,
            attention_factor,
            softmax_factor,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DsparkDialect { Qwen3, MiniMax }

impl DsparkDialect {
    pub fn tensor(self, name: &str) -> String {
        match self {
            Self::Qwen3 => name.into(),
            Self::MiniMax => {
                let name = if name == "norm.weight" { "final_norm.weight" } else { name };
                let name = if let Some(tail) = name.strip_prefix("layers.") {
                    let (layer, suffix) = tail.split_once('.').expect("recipe layer suffix");
                    format!("layers.{layer}.decoder_layer.{suffix}")
                } else { name.into() };
                format!("language_model.model.dspark.{name}")
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BlockDsparkConfig {
    pub backbone: DsparkBackbone,
    pub dialect: DsparkDialect,
    pub hidden: usize,
    pub intermediate: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub eps: f64,
    pub rope: DsparkRope,
    pub block: usize,
    pub mask_token: u32,
    pub rank: usize,
    /// Zero-based target post-block outputs, in projector concatenation order.
    pub taps: Vec<usize>,
    pub vocab: usize,
    /// None means full context, never a silently truncated sliding ring.
    pub window: Option<usize>,
    pub causal: bool,
}

#[derive(Deserialize)]
struct Layers {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rms_norm_eps: f64,
    rope_parameters: DsparkRope,
    vocab_size: usize,
    hidden_act: String,
    layer_types: Vec<String>,
    sliding_window: Option<usize>,
}

impl BlockDsparkConfig {
    pub fn tap_sites(&self) -> Vec<cuteafd_core::TapSite> {
        self.taps.iter().copied().map(cuteafd_core::TapSite::LayerExit).collect()
    }

    fn from_minimax(raw: &Value) -> Result<Self, DsparkConfigError> {
        use DsparkConfigError::Unsupported;
        let parsed = crate::families::minimax_m3::MiniMaxM3DsparkConfig::from_hf(raw)
            .map_err(|error| Unsupported(error.to_string()))?;
        let value = raw.get("text_config").unwrap_or(raw);
        let integer = |key: &str| value[key].as_u64().filter(|&n| n > 0 && n <= 1 << 24)
            .and_then(|n| usize::try_from(n).ok()).ok_or_else(|| Unsupported(format!("invalid {key}")));
        let hidden = integer("hidden_size")?;
        let heads = integer("num_attention_heads")?;
        let kv_heads = integer("num_key_value_heads")?;
        let head_dim = integer("head_dim")?;
        let vocab = integer("vocab_size")?;
        let eps = value["rms_norm_eps"].as_f64().filter(|n| n.is_finite() && *n > 0.)
            .ok_or_else(|| Unsupported("invalid rms_norm_eps".into()))?;
        let theta = value["rope_theta"].as_f64().ok_or_else(|| Unsupported("invalid rope_theta".into()))?;
        let backbone = DsparkBackbone::from_layers(value, head_dim)?;
        let rope = DsparkRope::Default { rope_theta: theta };
        rope.plan(backbone.rotary_dim)?;
        if heads % kv_heads != 0 || parsed.mask_token as usize >= vocab
            || value["confidence_head_with_markov"] != true || !parsed.confidence_head
            || value["use_qk_norm"] != true || value["qk_norm_type"] != "per_head"
            || value["attention_output_gate"] != false || parsed.markov_rank != 256
            || parsed.target_layers.iter().any(|&layer| layer >= value["dflash_config"]["num_target_layers"].as_u64().unwrap_or(0) as usize)
        { return Err(Unsupported("invalid MiniMax draft geometry or arithmetic".into())); }
        Ok(Self { backbone, dialect: DsparkDialect::MiniMax, hidden,
            intermediate: parsed.ffn_intermediate, layers: parsed.layers, heads, kv_heads, head_dim,
            eps, rope, block: parsed.block, mask_token: parsed.mask_token, rank: parsed.markov_rank,
            taps: parsed.target_layers, vocab, window: Some(parsed.sliding_window), causal: false })
    }

    /// Full layers share the target token budget; windowed layers keep their
    /// original slot-local rings. Both use the shared draft cache inventory.
    pub fn paged_cache_layout(&self, slots: usize, context: usize, target_tokens: usize,
        kv: crate::families::qwen4::Qwen4KvCache) -> Result<super::dflash::DraftCacheLayout, DsparkConfigError> {
        let invalid = || DsparkConfigError::Unsupported("invalid dSpark paged cache admission".into());
        if slots == 0 || context == 0 || target_tokens == 0
            || context.checked_add(self.block).is_none_or(|n| n > i32::MAX as usize)
            || self.window.is_some_and(|window| window != context) { return Err(invalid()); }
        let pages = target_tokens.div_ceil(64);
        let rows = pages.checked_mul(64).ok_or_else(invalid)?;
        if rows > i32::MAX as usize { return Err(invalid()); }
        let plane = kv.plane_record_bytes(self.kv_heads, self.head_dim);
        let record_bytes = plane.checked_mul(2).ok_or_else(invalid)?;
        let full = self.window.is_none();
        let cache_rows = if full { rows } else { slots.checked_mul(context).ok_or_else(invalid)? };
        let layer_bytes = cache_rows.checked_mul(record_bytes).ok_or_else(invalid)?;
        let total = layer_bytes.checked_mul(self.layers).ok_or_else(invalid)? as u64;
        let page_table_bytes = if full { slots.checked_mul(context.div_ceil(64))
            .and_then(|n| n.checked_mul(4)).ok_or_else(invalid)? as u64 } else { 0 };
        Ok(super::dflash::DraftCacheLayout { target_tokens, page_rows: 64, pages, record_bytes,
            full_layers: if full { self.layers } else { 0 },
            window_bytes: if full { 0 } else { total }, full_bytes: if full { total } else { 0 },
            page_table_bytes, layer_bytes: vec![layer_bytes; self.layers] })
    }

    pub fn read(snapshot: &Path) -> Result<Self, DsparkConfigError> {
        Self::from_json(&serde_json::from_slice(&std::fs::read(
            snapshot.join("config.json"),
        )?)?)
    }

    pub fn from_json(raw: &Value) -> Result<Self, DsparkConfigError> {
        use DsparkConfigError::Unsupported;
        let minimax = raw["architectures"].as_array().is_some_and(|list|
            list.iter().any(|name| name == "DSparkMiniMaxDraftModel"));
        if minimax { return Self::from_minimax(raw); }
        let speculators = raw["speculators_model_type"] == "dspark";
        // Speculators also advertises DSparkDraftModel; its explicit dialect wins.
        let specforge = !speculators
            && raw["architectures"]
                .as_array()
                .is_some_and(|a| a.iter().any(|v| v == "DSparkDraftModel"));
        if !speculators && !specforge {
            return Err(Unsupported(
                "expected DSparkDraftModel or speculators_model_type=dspark".into(),
            ));
        }
        let layer_config = if speculators { &raw["transformer_layer_config"] } else { raw };
        let t: Layers = serde_json::from_value(layer_config.clone())?;
        let backbone = DsparkBackbone::from_layers(layer_config, t.head_dim)?;
        let d = if speculators {
            raw
        } else {
            &raw["dflash_config"]
        };
        let integer = |value: &Value, name: &str| {
            value
                .as_u64()
                .and_then(|v| usize::try_from(v).ok())
                .ok_or_else(|| Unsupported(format!("missing or invalid {name}")))
        };
        let block = integer(&raw["block_size"], "block_size")?;
        let rank = integer(&raw["markov_rank"], "markov_rank")?;
        let mask_token = u32::try_from(integer(&d["mask_token_id"], "mask_token_id")?)
            .map_err(|_| Unsupported("mask_token_id exceeds u32".into()))?;
        let layer_ids: Vec<usize> = serde_json::from_value(
            d[if speculators {
                "aux_hidden_state_layer_ids"
            } else {
                "target_layer_ids"
            }]
            .clone(),
        )?;
        let taps = if speculators {
            layer_ids
                .into_iter()
                .map(|id| {
                    id.checked_sub(1).ok_or_else(|| {
                        Unsupported("embedding aux id 0 is not a post-block tap".into())
                    })
                })
                .collect::<Result<Vec<_>, _>>()?
        } else {
            layer_ids
        };
        if t.hidden_size == 0
            || t.intermediate_size == 0
            || t.num_hidden_layers == 0
            || t.num_key_value_heads == 0
            || t.num_attention_heads == 0
            || t.num_attention_heads % t.num_key_value_heads != 0
            || t.head_dim == 0
            || t.head_dim % 2 != 0
            || block == 0
            || rank == 0
            || taps.is_empty()
            || t.vocab_size == 0
            || mask_token as usize >= t.vocab_size
            || !t.rms_norm_eps.is_finite()
            || t.rms_norm_eps <= 0.0
        {
            return Err(Unsupported(
                "invalid dimensions, GQA grouping, epsilon, taps or mask token".into(),
            ));
        }
        if raw["markov_head_type"] != "vanilla"
            || raw["enable_confidence_head"] != true
            || raw["confidence_head_with_markov"] != true
            || t.hidden_act != "silu"
            || !backbone.is_qwen3(t.head_dim)
            || (speculators
                && (raw["sample_from_anchor"] != true
                    || raw["draft_vocab_size"].as_u64() != Some(t.vocab_size as u64)))
            || (specforge && d["projector_type"] != "dspark")
        {
            return Err(Unsupported("requires SiLU, vanilla Markov, Markov confidence, anchor sampling and the full target vocabulary".into()));
        }
        let sliding = t.layer_types.len() == t.num_hidden_layers
            && t.layer_types.iter().all(|v| v == "sliding_attention");
        let full = t.layer_types.len() == t.num_hidden_layers
            && t.layer_types.iter().all(|v| v == "full_attention");
        let window = match (sliding, full, t.sliding_window) {
            (true, false, Some(n)) if n > 0 => Some(n),
            (false, true, None) => None,
            _ => {
                return Err(Unsupported(
                    "requires uniform full attention or a positive uniform sliding window".into(),
                ))
            }
        };
        if t.rope_parameters.plan(t.head_dim)?.softmax_factor != 1.0 {
            return Err(Unsupported("GQA dSpark has no all-dimension YaRN softmax scale".into()));
        }
        if let Some(target_layers) = raw["num_target_layers"].as_u64() {
            if taps.iter().any(|&l| l as u64 >= target_layers) {
                return Err(Unsupported("tap exceeds num_target_layers".into()));
            }
        }
        let mut unique = taps.clone();
        unique.sort_unstable();
        unique.dedup();
        if unique.len() != taps.len() {
            return Err(Unsupported("duplicate target tap".into()));
        }
        Ok(Self {
            backbone,
            dialect: DsparkDialect::Qwen3,
            hidden: t.hidden_size,
            intermediate: t.intermediate_size,
            layers: t.num_hidden_layers,
            heads: t.num_attention_heads,
            kv_heads: t.num_key_value_heads,
            head_dim: t.head_dim,
            eps: t.rms_norm_eps,
            rope: t.rope_parameters,
            block,
            mask_token,
            rank,
            taps,
            vocab: t.vocab_size,
            window,
            causal: sliding && raw["sliding_window_non_causal"] != true,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backbone_descriptor_preserves_qwen_and_reads_native_minimax() {
        use cuteafd_core::FfnActivation;
        for raw in [qwen(), glm_flash_speculators()["transformer_layer_config"].clone()] {
            let head = raw["head_dim"].as_u64().unwrap() as usize;
            let descriptor = DsparkBackbone::from_layers(&raw, head).unwrap();
            assert!(descriptor.is_qwen3(head));
            assert_eq!(descriptor.norm, DsparkNorm::Rms);
            let old = FfnActivation::SiluBf16Product { limit: None };
            for gate in -128..128 {
                for up in [-19.0, -1.125, 0.0, 0.03125, 7.0, 19.0] {
                    let gate = gate as f32 / 16.0;
                    assert_eq!(descriptor.activation.evaluate(gate, up).to_bits(), old.evaluate(gate, up).to_bits());
                }
            }
        }
        let raw: Value = serde_json::from_str(include_str!("../families/minimax_m3/fixtures/dspark-config.json")).unwrap();
        let descriptor = DsparkBackbone::from_layers(&raw["text_config"], 128).unwrap();
        assert_eq!(descriptor, DsparkBackbone { norm: DsparkNorm::Gemma, residual: DsparkResidual::Bf16, rotary_dim: 64,
            activation: FfnActivation::SwigluOai { alpha: 1.702, beta: 1.0, limit: 7.0 } });
        assert!(!descriptor.is_qwen3(128));
        let config = BlockDsparkConfig::from_json(&raw).unwrap();
        assert_eq!(config.dialect, DsparkDialect::MiniMax);
        assert_eq!((config.hidden, config.intermediate, config.heads, config.kv_heads), (6144, 12288, 64, 4));
        assert_eq!((config.block, config.window, config.causal), (7, Some(4096), false));
        assert_eq!(config.taps, [3, 17, 31, 45, 59]);
        assert_eq!(config.dialect.tensor("layers.2.self_attn.q_proj.weight"),
            "language_model.model.dspark.layers.2.decoder_layer.self_attn.q_proj.weight");
        assert_eq!(config.dialect.tensor("norm.weight"), "language_model.model.dspark.final_norm.weight");
    }

    #[test]
    fn native_minimax_arena_follows_header_formats_without_requantization() {
        let t = |name: &str, dtype: &str, shape: &[usize]| crate::SafetensorsTensorMetadata {
            name: name.into(), dtype: cuteafd_core::DType::from_safetensors(dtype), shape: shape.to_vec(),
            byte_offset: 0, byte_length: shape.iter().product::<usize>() as u64 * if dtype == "BF16" { 2 } else { 1 },
        };
        let raw: Value = serde_json::from_str(include_str!("../families/minimax_m3/fixtures/dspark-config.json")).unwrap();
        let config = BlockDsparkConfig::from_json(&raw).unwrap();
        let headers = std::cell::RefCell::new(std::collections::HashMap::new());
        block_dspark_weight_recipe(&config, |name, shape| {
            let header = t(name, "BF16", shape);
            headers.borrow_mut().insert(name.to_owned(), header);
            Ok::<_, ()>(())
        }, |parts, columns| {
            for &(name, rows) in parts {
                let bf16 = name.ends_with("dspark.fc.weight");
                let header = t(name, if bf16 { "BF16" } else { "F8_E4M3" }, &[rows, columns]);
                headers.borrow_mut().insert(name.to_owned(), header);
                if !bf16 {
                    let scales = format!("{name}_scale_inv");
                    headers.borrow_mut().insert(scales.clone(), t(&scales, "U8", &[rows, columns / 32]));
                }
            }
            Ok(())
        }).unwrap();
        let mut headers = headers.into_inner();
        let requests = native_arena_requests(&config, &headers).unwrap();
        assert_eq!(requests.len(), 98);
        let fc = "language_model.model.dspark.fc.weight";
        assert!(requests.contains(&(fc.into(), 6144 * 30720 * 2)));
        let query = "language_model.model.dspark.layers.0.decoder_layer.self_attn.q_proj.weight";
        assert!(requests.contains(&(query.into(), 8192 * 6144)));
        assert!(requests.contains(&(format!("{query}_scale_inv"), 8192 * 192)));
        assert!(!requests.iter().any(|(name, _)| name.ends_with(".payload") || name.ends_with(".scales")));
        headers.get_mut(&format!("{query}_scale_inv")).unwrap().dtype = cuteafd_core::DType::F32;
        assert!(native_arena_requests(&config, &headers).is_err());
    }

    #[test]
    fn backbone_descriptor_rejects_ambiguous_arithmetic() {
        for (key, value) in [("use_gemma_norm", json!(1)), ("rotary_dim", json!(0)),
            ("rotary_dim", json!(127)), ("rotary_dim", json!(130)), ("rotary_dim", json!(1.5)),
            ("partial_rotary_factor", json!(0.5)), ("hidden_act", json!("gelu"))] {
            let mut raw = qwen();
            raw[key] = value;
            assert!(DsparkBackbone::from_layers(&raw, 128).is_err(), "{key}");
        }
        for key in ["swiglu_alpha", "swiglu_limit"] {
            let mut raw = json!({"hidden_act":"swigluoai", "swiglu_alpha":1.702, "swiglu_limit":7.0});
            raw[key] = json!(0.0);
            assert!(DsparkBackbone::from_layers(&raw, 128).is_err(), "{key}");
        }
    }

    #[test]
    fn arena_requests_follow_the_shared_recipe_for_both_representations() {
        use crate::families::glm5::draft_representation::GlmDraftRepresentation;
        use crate::plan::testing::{t, write_safetensors};
        let snapshot = tempfile::tempdir().unwrap();
        let raw = serde_json::json!({"architectures":["DSparkDraftModel"], "speculators_model_type":"dspark",
            "aux_hidden_state_layer_ids":[1,2], "block_size":2, "confidence_head_with_markov":true,
            "draft_vocab_size":16, "enable_confidence_head":true, "markov_head_type":"vanilla", "markov_rank":256,
            "mask_token_id":0, "sample_from_anchor":true, "sliding_window_non_causal":false,
            "transformer_layer_config":{"attention_bias":false, "head_dim":128, "hidden_act":"silu",
                "hidden_size":256, "intermediate_size":256, "layer_types":["full_attention"],
                "max_position_embeddings":64, "model_type":"qwen3", "num_attention_heads":2, "num_hidden_layers":1,
                "num_key_value_heads":1, "rms_norm_eps":1e-5, "rope_parameters":{"rope_theta":10000.0,"rope_type":"default"},
                "sliding_window":null, "vocab_size":16}});
        std::fs::write(snapshot.path().join("config.json"), serde_json::to_vec(&raw).unwrap()).unwrap();
        let cfg = BlockDsparkConfig::read(snapshot.path()).unwrap();
        let shapes = std::cell::RefCell::new(Vec::new());
        block_dspark_weight_recipe(&cfg, |name, shape| { shapes.borrow_mut().push(t(name, "BF16", shape)); Ok::<_, ()>(()) },
            |parts, cols| { for (name, rows) in parts { shapes.borrow_mut().push(t(*name, "BF16", &[*rows, cols])); } Ok(()) })
            .unwrap();
        write_safetensors(&snapshot.path().join("model.safetensors"), &shapes.into_inner());
        let bf16 = block_dspark_arena_requests(snapshot.path(), GlmDraftRepresentation::Bf16Only).unwrap();
        let qkv = "layers.0.self_attn.q_proj.weight+layers.0.self_attn.k_proj.weight+layers.0.self_attn.v_proj.weight";
        assert_eq!(bf16.iter().find(|(n, _)| n == qkv).unwrap().1, (256 + 128 + 128) * 256 * 2);
        assert_eq!(bf16.iter().find(|(n, _)| n == "layers.0.self_attn.q_norm.weight").unwrap().1, 256);
        assert_eq!(bf16.iter().find(|(n, _)| n == "fc.weight").unwrap().1, 256 * 512 * 2);
        assert_eq!(bf16[bf16.len() - 2], ("markov_norms".to_string(), 16 * 4));
        assert_eq!(bf16.last().unwrap(), &("inv_freq".to_string(), 256));
        let fp8 = block_dspark_arena_requests(snapshot.path(), GlmDraftRepresentation::Fp8Only).unwrap();
        let payload = (256 + 128 + 128) * 256;
        assert!(fp8.contains(&(format!("{qkv}.payload"), payload)) && fp8.contains(&(format!("{qkv}.scales"), payload / 128 * 4)));
        // Every matrix splits into one payload and one scales plane; tensors keep their single request.
        assert_eq!(fp8.len(), bf16.len() + 5);
        let names = bf16.iter().map(|(name, _)| name).collect::<std::collections::BTreeSet<_>>();
        assert_eq!(names.len(), bf16.len(), "request names must be unique");
    }
    #[test]
    fn shared_window_runtime_has_exact_named_buffers_and_one_arena() -> anyhow::Result<()> {
        use crate::plan::testing::{t, write_safetensors};
        let root = tempfile::tempdir()?;
        let mut raw = qwen();
        raw["layer_types"] = json!(vec!["sliding_attention";5]);
        raw["sliding_window"] = json!(2048);
        raw["markov_rank"] = json!(512);
        std::fs::write(root.path().join("config.json"),serde_json::to_vec(&raw)?)?;
        let cfg = BlockDsparkConfig::read(root.path())?;
        let shapes = std::cell::RefCell::new(Vec::new());
        block_dspark_weight_recipe(&cfg, |name, shape| {
            shapes.borrow_mut().push(t(name,"BF16",shape)); Ok::<_,anyhow::Error>(())
        }, |parts, cols| { for (name, rows) in parts {
            shapes.borrow_mut().push(t(*name,"BF16",&[*rows,cols])); } Ok(()) })?;
        write_safetensors(&root.path().join("model.safetensors"),&shapes.into_inner());
        let one = BlockDsparkRuntimeLayout::new(root.path(),1,1,188,"test/draft")?;
        let two = BlockDsparkRuntimeLayout::new(root.path(),2,1,188,"test/draft")?;
        assert_eq!(one.weights.views(),two.weights.views());
        assert_eq!(one.buffers.len(),two.buffers.len());
        let total = |layout: &BlockDsparkRuntimeLayout| layout.buffers.iter().map(|(_,n)| *n).sum::<usize>();
        assert_eq!(total(&two)-total(&one),5*2*2048*8*128*2);
        assert_eq!(one.window_mark_bytes()?,(16usize+5*2*2048*8*128*2).div_ceil(256)*256);
        let fp8=BlockDsparkRuntimeLayout::with_kv(root.path(),1,1,188,"test/draft",
            crate::families::qwen4::Qwen4KvCache::Fp8)?;
        let plane=crate::families::qwen4::Qwen4KvCache::Fp8.plane_record_bytes(8,128);
        assert_eq!(total(&one)-total(&fp8),5*2*2048*(8*128*2-plane));
        assert_eq!(fp8.window_mark_bytes()?,(16usize+5*2*2048*plane).div_ceil(256)*256);
        assert_eq!(one.weights,fp8.weights);
        use crate::families::glm5::draft_representation::GlmDraftRepresentation;
        let quant=BlockDsparkRuntimeLayout::with_representation(root.path(),1,1,188,"test/quant",
            crate::families::qwen4::Qwen4KvCache::Bf16,GlmDraftRepresentation::Fp8Only)?;
        assert!(quant.weights.payload_bytes() < one.weights.payload_bytes());
        assert_eq!(quant.window_mark_bytes()?,one.window_mark_bytes()?);
        assert!(quant.buffers.iter().any(|(name,bytes)|name=="workspace.fp8-linear" && *bytes>0));
        for rank in 0..2 {
            let quant=BlockDsparkRuntimeLayout::for_rank_representation(root.path(),1,1,188,"test/quant",
                crate::families::qwen4::Qwen4KvCache::Fp8,rank,2,0,GlmDraftRepresentation::Fp8Only)?;
            let native=BlockDsparkRuntimeLayout::for_rank(root.path(),1,1,188,"test/native",
                crate::families::qwen4::Qwen4KvCache::Fp8,rank,2,0)?;
            assert!(quant.weights.payload_bytes()<native.weights.payload_bytes());
            assert_eq!(quant.weight_requests.iter().filter(|(name,_)|name.ends_with(".payload")).count(),21);
            assert_eq!(quant.window_mark_bytes()?,native.window_mark_bytes()?);
            assert!(quant.buffers.iter().any(|(name,_)|name=="workspace.fp8-linear"));
        }
        let lead=BlockDsparkRuntimeLayout::for_rank(root.path(),1,1,188,"test/draft/1",
            crate::families::qwen4::Qwen4KvCache::Bf16,0,2,0)?;
        let peer=BlockDsparkRuntimeLayout::for_rank(root.path(),1,1,188,"test/draft/0",
            crate::families::qwen4::Qwen4KvCache::Bf16,1,2,0)?;
        assert_eq!((lead.config.heads,lead.config.kv_heads,lead.config.intermediate),(20,4,5120));
        assert!(lead.weights.view("markov_head.markov_w1.weight").is_some());
        assert!(peer.weights.view("markov_head.markov_w1.weight").is_none());
        assert!(peer.buffers.iter().all(|(name,_)|!matches!(name.as_str(),"workspace.logits"|"workspace.markov")));
        assert_eq!(lead.weights.view("fc.weight").unwrap().bytes(),one.weights.view("fc.weight").unwrap().bytes());
        assert_eq!(peer.weights.view("fc.weight").unwrap().bytes(),one.weights.view("fc.weight").unwrap().bytes());
        let qkv="layers.0.self_attn.q_proj.weight+layers.0.self_attn.k_proj.weight+layers.0.self_attn.v_proj.weight";
        assert_eq!(lead.weights.view(qkv).unwrap().bytes()*2,one.weights.view(qkv).unwrap().bytes());
        assert_eq!(peer.weights.view("layers.0.self_attn.o_proj.weight").unwrap().bytes()*2,
            one.weights.view("layers.0.self_attn.o_proj.weight").unwrap().bytes());
        for local in [&lead,&peer] {
            let reduction=local.capacity.block_rows*local.config.hidden*2;
            assert_eq!(local.reduction_bytes,reduction);
            assert!(!local.buffers.iter().any(|(name,_)|name.contains("reduc")));
            let demands=local.demands(1);
            assert_eq!(demands.iter().filter(|d|d.group=="block dSpark shared TP2 reduction").count(),1);
            assert_eq!(demands.iter().map(|d|d.bytes).sum::<u64>(),
                local.buffers.iter().map(|(_,n)|*n as u64).sum::<u64>()+reduction as u64
                    +local.exchange_buffers.iter().map(|(_,n)|*n as u64).sum::<u64>());
        }
        for physical_lead in [0,1] {
            for rank in 0..2 {
                let local=BlockDsparkRuntimeLayout::for_rank(root.path(),1,1,188,&format!("test/draft/{rank}"),
                    crate::families::qwen4::Qwen4KvCache::Bf16,rank,2,physical_lead)?;
                assert_eq!(local.weights.view("markov_head.markov_w1.weight").is_some(),rank==physical_lead);
                assert_eq!(local.buffers.iter().any(|(n,_)|n=="workspace.logits"),rank==physical_lead);
            }
        }
        let paired=crate::families::kimi_k3::draft::DraftLayout::BlockTp2 {
            ranks:Box::new([peer.clone(),lead.clone()]),lead:1 };
        let marks=paired.window_mark_demands(1)?;
        assert_eq!(marks.len(),2);
        assert_eq!(marks[0].gpu,0); assert_eq!(marks[1].gpu,1);
        assert_eq!(marks[0].bytes,peer.window_mark_bytes()? as u64);
        assert_eq!(marks[1].bytes,lead.window_mark_bytes()? as u64);
        assert_eq!(marks.iter().map(|m|m.bytes).sum::<u64>(),paired.window_mark_bytes()? as u64);
        use crate::families::glm5::draft_physical::ShardAxis;
        for rank in 0..2 {
            let mut parts=std::collections::BTreeMap::new();
            block_dspark_rank_recipe(&cfg,rank,2,1,|_,_|Ok(()),|source,cols,axis| {
                for &(name,rows) in source { parts.insert(name.to_owned(),block_dspark_part_ranges(rows,cols,axis,rank,2)?); }
                Ok(())
            })?;
            assert_eq!(parts["layers.0.self_attn.q_proj.weight"].0,rank*2560..(rank+1)*2560);
            for name in ["k_proj","v_proj"] {
                assert_eq!(parts[&format!("layers.0.self_attn.{name}.weight")].0,rank*512..(rank+1)*512);
            }
            for name in ["gate_proj","up_proj"] {
                assert_eq!(parts[&format!("layers.0.mlp.{name}.weight")].0,rank*5120..(rank+1)*5120);
            }
            assert_eq!(parts["layers.0.self_attn.o_proj.weight"].1,rank*2560..(rank+1)*2560);
            assert!(block_dspark_part_ranges(3,4,Some(ShardAxis::Rows),rank,2).is_err());
        }
        assert!(one.demands(1).iter().all(|d| d.gpu == 1 && d.resident_weights.0 == 0));
        assert_eq!(one.buffers.iter().map(|(n,_)| n).collect::<std::collections::BTreeSet<_>>().len(),one.buffers.len());
        assert!(BlockDsparkRuntimeLayout::new(root.path(),1,2,188,"test/draft").is_err());
        Ok(())
    }

    use serde_json::json;

    fn qwen() -> Value {
        json!({"architectures":["DSparkDraftModel"], "hidden_size":5120, "intermediate_size":10240,
            "num_hidden_layers":5,"num_attention_heads":40,"num_key_value_heads":8,"head_dim":128,
            "rms_norm_eps":1e-6,"rope_parameters":{"rope_type":"yarn","rope_theta":10000000.0,
                "factor":32.0,"original_max_position_embeddings":8192,"beta_fast":32.0,"beta_slow":1.0},
            "vocab_size":248320,"hidden_act":"silu","layer_types":["full_attention","full_attention",
                "full_attention","full_attention","full_attention"],"sliding_window":null,"block_size":7,
            "markov_rank":256,"markov_head_type":"vanilla","enable_confidence_head":true,
            "confidence_head_with_markov":true,"num_target_layers":64,
            "dflash_config":{"projector_type":"dspark","mask_token_id":248077,"target_layer_ids":[4,16,28,40,52]}})
    }

    // RedHatAI/GLM-5.3-Flash-speculator.dspark-preview, revision 1972f1f0:
    // both dialect markers are present in the checkpoint's own config.
    fn glm_flash_speculators() -> Value {
        json!({"architectures":["DSparkDraftModel"],
            "speculators_model_type":"dspark","target_hidden_size":null,
            "aux_hidden_state_layer_ids":[20,28,32,36,40,44],"block_size":8,
            "confidence_head_with_markov":true,"draft_vocab_size":154880,
            "enable_confidence_head":true,"markov_head_type":"vanilla","markov_rank":256,
            "mask_token_id":154856,"sample_from_anchor":true,"sliding_window_non_causal":false,
            "transformer_layer_config":{"attention_bias":false,"head_dim":64,
                "hidden_act":"silu","hidden_size":4096,"intermediate_size":12288,
                "layer_types":["sliding_attention","sliding_attention","sliding_attention",
                    "sliding_attention","sliding_attention"],"max_position_embeddings":1048576,
                "model_type":"qwen3","num_attention_heads":64,"num_hidden_layers":5,
                "num_key_value_heads":64,"rms_norm_eps":1e-5,
                "rope_parameters":{"rope_theta":10000.0,"rope_type":"default"},
                "sliding_window":2048,"use_sliding_window":true,"vocab_size":154880}})
    }

    #[test]
    fn typed_layer_exit_sites_preserve_every_existing_block_family_order() {
        use cuteafd_core::TapSite;
        let mut glm = qwen();
        glm["dflash_config"]["target_layer_ids"] = json!([3, 15, 27, 39, 51]);
        let mut kimi = qwen();
        kimi["dflash_config"]["target_layer_ids"] = json!([2, 14, 26, 38, 50]);
        let minimax: Value = serde_json::from_str(include_str!("../families/minimax_m3/fixtures/dspark-config.json")).unwrap();
        for (family, raw) in [("GLM", glm), ("GLM Flash", glm_flash_speculators()),
            ("Kimi", kimi), ("MiniMax", minimax), ("Qwen 3.5", qwen())] {
            let cfg = BlockDsparkConfig::from_json(&raw).unwrap();
            let before = cfg.taps.clone();
            let after = cfg.tap_sites().into_iter().map(|site| match site {
                TapSite::LayerExit(layer) => layer, _ => panic!("{family} changed tap arithmetic"),
            }).collect::<Vec<_>>();
            assert_eq!(after, before, "{family}");
        }
    }

    #[test]
    fn speculators_architecture_marker_does_not_require_specforge_projector() {
        let c = BlockDsparkConfig::from_json(&glm_flash_speculators()).unwrap();
        assert_eq!((c.hidden, c.intermediate, c.layers), (4096, 12288, 5));
        assert_eq!((c.heads, c.kv_heads, c.head_dim), (64, 64, 64));
        assert_eq!(
            (c.block, c.rank, c.vocab, c.mask_token),
            (8, 256, 154880, 154856)
        );
        assert_eq!(c.taps, vec![19, 27, 31, 35, 39, 43]);
        assert_eq!(c.window, Some(2048));
        assert!(c.causal);
        assert_eq!(
            c.rope,
            DsparkRope::Default {
                rope_theta: 10000.0
            }
        );

        // Dialect selection must not bypass Speculators' semantic checks.
        for (field, value) in [
            ("sample_from_anchor", json!(false)),
            ("draft_vocab_size", json!(154879)),
            ("confidence_head_with_markov", json!(false)),
        ] {
            let mut raw = glm_flash_speculators();
            raw[field] = value;
            assert!(BlockDsparkConfig::from_json(&raw).is_err(), "{field}");
        }
        let mut raw = glm_flash_speculators();
        raw["transformer_layer_config"]["hidden_act"] = json!("gelu");
        assert!(BlockDsparkConfig::from_json(&raw).is_err());
    }

    #[test]
    fn immutable_yarn_plan_uses_reference_fp32_blending() {
        let c = BlockDsparkConfig::from_json(&qwen()).unwrap();
        let plan = c.rope.plan(c.head_dim).unwrap();
        assert_eq!(plan.inv_freq.len(), 64);
        assert!((plan.attention_factor - (1.0f64 + 0.1 * 32.0f64.ln()) as f32).abs() < 1e-7);
        assert_eq!(plan.inv_freq[0], 1.0);
        // Checkpoint correction dimensions floor/ceil to [14, 29].
        for i in [0usize, 14, 20, 29, 63] {
            let frequency = 10_000_000f32.powf((2 * i) as f32 / 128.0);
            let weight = 1.0 - ((i as f32 - 14.0) / 15.0).clamp(0.0, 1.0);
            let expected = (1.0 / (32.0 * frequency)) * (1.0 - weight) + (1.0 / frequency) * weight;
            assert_eq!(plan.inv_freq[i], expected, "frequency {i}");
        }
        // Pinned Qwen3RotaryEmbedding CPU FP32 table, full checkpoint geometry.
        let reference_bits = [
            1065353216u32,
            1061618021,
            1058714411,
            1055949884,
            1052440603,
            1049712609,
            1046607925,
            1043310893,
            1040747895,
            1037323628,
            1034226006,
            1031818025,
            1028093503,
            1025183233,
            1022431607,
            1018121107,
            1014872157,
            1010211608,
            1006802083,
            1002002126,
            998397281,
            993346669,
            989370984,
            984051384,
            979118631,
            973860955,
            967379848,
            960192872,
            950929914,
            934311859,
            931737090,
            928335622,
            925223776,
            922804734,
            919101633,
            916177998,
            913452258,
            909918774,
            907171966,
            904103724,
            900783954,
            898203279,
            894813249,
            891694265,
            889269675,
            885577321,
            882646980,
            879934230,
            876392641,
            873639533,
            870583494,
            867256108,
            864669514,
            861290950,
            858164812,
            855734661,
            852053079,
            849116016,
            846416285,
            842866572,
            840107150,
            837063342,
            833728324,
            831135798,
        ];
        for (&actual, expected) in plan.inv_freq.iter().zip(reference_bits) {
            assert!(
                actual.to_bits().abs_diff(expected) <= 2,
                "frequency {actual} vs {}",
                f32::from_bits(expected)
            );
        }
        assert_eq!(plan, c.rope.plan(c.head_dim).unwrap());
        assert!(plan.inv_freq.windows(2).all(|p| p[0] > p[1]));
    }

    #[test]
    fn mla_yarn_separates_rotary_magnitude_and_softmax_scale() {
        let raw = json!({"rope_type":"yarn", "rope_theta":50000.0,
            "factor":32.0, "original_max_position_embeddings":32768,
            "mscale":1.0, "mscale_all_dim":1.0});
        let rope: DsparkRope = serde_json::from_value(raw.clone()).unwrap();
        let plan = rope.plan(64).unwrap();
        assert_eq!(plan.attention_factor, 1.0);
        assert_eq!(plan.softmax_factor, (1.0 + 0.1 * 32.0f64.ln()).powi(2) as f32);
        let mut explicit = raw.clone();
        explicit["beta_fast"] = json!(32.0);
        explicit["beta_slow"] = json!(1.0);
        assert_eq!(plan, serde_json::from_value::<DsparkRope>(explicit).unwrap().plan(64).unwrap());
        let mut scaled = raw.clone();
        scaled["mscale"] = json!(0.5);
        let scaled = serde_json::from_value::<DsparkRope>(scaled).unwrap().plan(64).unwrap();
        assert_eq!(scaled.inv_freq, plan.inv_freq);
        assert_eq!(scaled.softmax_factor, plan.softmax_factor);
        assert_eq!(scaled.attention_factor,
            ((1.0 + 0.05 * 32.0f64.ln()) / (1.0 + 0.1 * 32.0f64.ln())) as f32);
        for field in ["mscale", "mscale_all_dim"] {
            let mut invalid = raw.clone();
            invalid[field] = json!(-1.0);
            assert!(serde_json::from_value::<DsparkRope>(invalid).unwrap().plan(64).is_err());
        }
        let mut unit = raw;
        unit["factor"] = json!(1.0);
        let unit = serde_json::from_value::<DsparkRope>(unit).unwrap().plan(64).unwrap();
        assert_eq!((unit.attention_factor, unit.softmax_factor), (1.0, 1.0));
    }

    #[test]
    fn rotary_plan_rejects_invalid_direct_construction() {
        assert!(DsparkRope::Default {
            rope_theta: f64::NAN
        }
        .plan(128)
        .is_err());
        assert!(DsparkRope::Default {
            rope_theta: 10000.0
        }
        .plan(127)
        .is_err());
        let plan = DsparkRope::Default {
            rope_theta: 10000.0,
        }
        .plan(64)
        .unwrap();
        assert_eq!(plan.attention_factor, 1.0);
        assert_eq!(plan.inv_freq[0], 1.0);
    }

    #[test]
    fn shared_paged_inventory_is_token_budget_not_slot_context_product() {
        use crate::families::qwen4::Qwen4KvCache;
        let cfg = BlockDsparkConfig::from_json(&qwen()).unwrap();
        for kv in [Qwen4KvCache::Bf16, Qwen4KvCache::Fp8] {
            let one = cfg.paged_cache_layout(1, 262144, 8193, kv).unwrap();
            let many = cfg.paged_cache_layout(16, 262144, 8193, kv).unwrap();
            assert_eq!(one.full_bytes, many.full_bytes);
            assert_eq!(one.pages, 129);
            assert_eq!(one.full_bytes, (129 * 64 * 2 * kv.plane_record_bytes(8, 128) * 5) as u64);
            assert_eq!(many.page_table_bytes, 16 * 4096 * 4);
            assert_eq!(many.window_bytes, 0);
            assert!(cfg.paged_cache_layout(16, 262144, 0, kv).is_err());
        }
    }

    #[test]
    fn specforge_post_block_gqa_full_yarn() {
        let c = BlockDsparkConfig::from_json(&qwen()).unwrap();
        assert_eq!(c.taps, vec![4, 16, 28, 40, 52]);
        assert_eq!((c.head_dim, c.heads / c.kv_heads * c.block), (128, 35));
        assert_eq!(c.window, None);
        assert!(!c.causal);
    }

    #[test]
    fn speculators_input_ids_become_post_block_ids() {
        let q = qwen();
        let mut t = q.clone();
        t["sliding_window"] = json!(2048);
        t["layer_types"] = json!(vec!["sliding_attention"; 5]);
        t["rope_parameters"] = json!({"rope_type":"default","rope_theta":10000.0});
        let c = BlockDsparkConfig::from_json(&json!({"speculators_model_type":"dspark",
            "transformer_layer_config":t,"aux_hidden_state_layer_ids":[20,28,32,36,40,44],
            "block_size":8,"markov_rank":256,"markov_head_type":"vanilla","sample_from_anchor":true,
            "enable_confidence_head":true,"confidence_head_with_markov":true,"draft_vocab_size":248320,
            "mask_token_id":248077})).unwrap();
        assert_eq!(c.taps, vec![19, 27, 31, 35, 39, 43]);
        assert_eq!(c.window, Some(2048));
        assert!(c.causal);
    }

    #[test]
    fn rejects_invalid_geometry_and_semantics() {
        for (field, value) in [
            ("num_key_value_heads", json!(0)),
            ("num_key_value_heads", json!(7)),
            ("block_size", json!(0)),
            ("head_dim", json!(127)),
            ("markov_head_type", json!("other")),
            ("enable_confidence_head", json!(false)),
            ("layer_types", json!(["full_attention"])),
        ] {
            let mut c = qwen();
            c[field] = value;
            assert!(BlockDsparkConfig::from_json(&c).is_err(), "{field}");
        }
        let mut c = qwen();
        c["dflash_config"]["target_layer_ids"] = json!([4, 64]);
        assert!(BlockDsparkConfig::from_json(&c).is_err());
    }
}

/// One draft layer's checkpoint operands, in loader order.
pub struct BlockDsparkLayer<T, M> {
    pub input_norm: T, pub post_norm: T, pub qkv: M, pub q_norm: T, pub k_norm: T,
    pub o: M, pub gate_up: M, pub down: M,
}

/// Every checkpoint-sourced drafter weight. `tensor` binds one tensor of a
/// shape; `matrix` binds row-concatenated parts sharing `cols`.
pub struct BlockDsparkWeights<T, M> {
    pub layers: Vec<BlockDsparkLayer<T, M>>, pub fc: M, pub hidden_norm: T, pub norm: T,
    pub markov_w1: T, pub markov_w2: T, pub confidence_w: T, pub confidence_b: T,
}

/// The single recipe for runtime loading and admission: names, shapes and the
/// order in which the loader requests each owner.
pub fn block_dspark_weight_recipe<T, M, E>(cfg: &BlockDsparkConfig,
    mut tensor_native: impl FnMut(&str, &[usize]) -> Result<T, E>,
    mut matrix_native: impl FnMut(&[(&str, usize)], usize) -> Result<M, E>) -> Result<BlockDsparkWeights<T, M>, E> {
    let mut tensor = |name: &str, shape: &[usize]| tensor_native(&cfg.dialect.tensor(name), shape);
    let mut matrix = |parts: &[(&str, usize)], columns| {
        let names = parts.iter().map(|(name, _)| cfg.dialect.tensor(name)).collect::<Vec<_>>();
        let mapped = parts.iter().zip(&names).map(|((_, rows), name)| (name.as_str(), *rows)).collect::<Vec<_>>();
        matrix_native(&mapped, columns)
    };
    let (h, inter) = (cfg.hidden, cfg.intermediate);
    let (q, kv) = (cfg.heads * cfg.head_dim, cfg.kv_heads * cfg.head_dim);
    let layers = (0..cfg.layers).map(|l| -> Result<_, E> {
        let p = format!("layers.{l}");
        let a = format!("{p}.self_attn");
        Ok(BlockDsparkLayer {
            input_norm: tensor(&format!("{p}.input_layernorm.weight"), &[h])?,
            post_norm: tensor(&format!("{p}.post_attention_layernorm.weight"), &[h])?,
            qkv: matrix(&[(&format!("{a}.q_proj.weight"), q), (&format!("{a}.k_proj.weight"), kv),
                (&format!("{a}.v_proj.weight"), kv)], h)?,
            q_norm: tensor(&format!("{a}.q_norm.weight"), &[cfg.head_dim])?,
            k_norm: tensor(&format!("{a}.k_norm.weight"), &[cfg.head_dim])?,
            o: matrix(&[(&format!("{a}.o_proj.weight"), h)], q)?,
            gate_up: matrix(&[(&format!("{p}.mlp.gate_proj.weight"), inter),
                (&format!("{p}.mlp.up_proj.weight"), inter)], h)?,
            down: matrix(&[(&format!("{p}.mlp.down_proj.weight"), h)], inter)?,
        })
    }).collect::<Result<Vec<_>, E>>()?;
    let markov_w2 = tensor("markov_head.markov_w2.weight", &[cfg.vocab, cfg.rank])?;
    Ok(BlockDsparkWeights { layers, markov_w2,
        fc: matrix(&[("fc.weight", h)], cfg.taps.len() * h)?,
        hidden_norm: tensor("hidden_norm.weight", &[h])?, norm: tensor("norm.weight", &[h])?,
        markov_w1: tensor("markov_head.markov_w1.weight", &[cfg.vocab, cfg.rank])?,
        confidence_w: tensor("confidence_head.proj.weight", &[1, h + cfg.rank])?,
        confidence_b: tensor("confidence_head.proj.bias", &[1])?,
    })
}

pub fn block_dspark_part_ranges(rows:usize,columns:usize,
    axis:Option<crate::families::glm5::draft_physical::ShardAxis>,rank:usize,ranks:usize)
    ->anyhow::Result<(std::ops::Range<usize>,std::ops::Range<usize>)> {
    use crate::families::glm5::draft_physical::ShardAxis;
    anyhow::ensure!(matches!(ranks,1|2) && rank<ranks && rows>0 && columns>0,"invalid block matrix part");
    let mut row=0..rows; let mut column=0..columns;
    if matches!(axis,Some(ShardAxis::Rows)) {
        anyhow::ensure!(rows%ranks==0,"block split cuts part rows");
        let width=rows/ranks; row=rank*width..(rank+1)*width;
    }
    if matches!(axis,Some(ShardAxis::Columns)) {
        anyhow::ensure!(columns%ranks==0,"block split cuts part columns");
        let width=columns/ranks; column=rank*width..(rank+1)*width;
    }
    Ok((row,column))
}

/// Rank-local checkpoint recipe. FC/norms are replicated; Markov/confidence
/// exist only on the explicitly selected physical lead rank.
pub struct BlockDsparkRankWeights<T,M> {
    pub layers: Vec<BlockDsparkLayer<T,M>>,
    pub fc:M, pub hidden_norm:T, pub norm:T,
    pub head:Option<(T,T,T,T)>,
}

pub fn block_dspark_rank_recipe<T,M>(cfg:&BlockDsparkConfig,rank:usize,ranks:usize,lead:usize,
    mut tensor:impl FnMut(&str,&[usize])->anyhow::Result<T>,
    mut matrix:impl FnMut(&[(&str,usize)],usize,Option<crate::families::glm5::draft_physical::ShardAxis>)->anyhow::Result<M>)
    ->anyhow::Result<BlockDsparkRankWeights<T,M>> {
    use crate::families::glm5::draft_physical::ShardAxis;
    anyhow::ensure!(matches!(ranks,1|2) && rank<ranks && lead<ranks && cfg.heads%ranks==0
        && cfg.kv_heads%ranks==0 && cfg.intermediate%ranks==0 && cfg.dialect==DsparkDialect::Qwen3,
        "block draft rank split cuts checkpoint geometry");
    let (h,q,kv,inter)=(cfg.hidden,cfg.heads*cfg.head_dim,cfg.kv_heads*cfg.head_dim,cfg.intermediate);
    let layers=(0..cfg.layers).map(|layer| {
        let p=format!("layers.{layer}"); let a=format!("{p}.self_attn");
        Ok(BlockDsparkLayer {
            input_norm:tensor(&format!("{p}.input_layernorm.weight"),&[h])?,
            post_norm:tensor(&format!("{p}.post_attention_layernorm.weight"),&[h])?,
            qkv:matrix(&[(&format!("{a}.q_proj.weight"),q),(&format!("{a}.k_proj.weight"),kv),
                (&format!("{a}.v_proj.weight"),kv)],h,Some(ShardAxis::Rows))?,
            q_norm:tensor(&format!("{a}.q_norm.weight"),&[cfg.head_dim])?,
            k_norm:tensor(&format!("{a}.k_norm.weight"),&[cfg.head_dim])?,
            o:matrix(&[(&format!("{a}.o_proj.weight"),h)],q,Some(ShardAxis::Columns))?,
            gate_up:matrix(&[(&format!("{p}.mlp.gate_proj.weight"),inter),(&format!("{p}.mlp.up_proj.weight"),inter)],
                h,Some(ShardAxis::Rows))?,
            down:matrix(&[(&format!("{p}.mlp.down_proj.weight"),h)],inter,Some(ShardAxis::Columns))?,
        })
    }).collect::<anyhow::Result<Vec<_>>>()?;
    let head=if rank==lead { Some((tensor("markov_head.markov_w1.weight",&[cfg.vocab,cfg.rank])?,
        tensor("markov_head.markov_w2.weight",&[cfg.vocab,cfg.rank])?,
        tensor("confidence_head.proj.weight",&[1,h+cfg.rank])?,tensor("confidence_head.proj.bias",&[1])?)) } else { None };
    Ok(BlockDsparkRankWeights { layers,head,fc:matrix(&[("fc.weight",h)],cfg.taps.len()*h,None)?,
        hidden_norm:tensor("hidden_norm.weight",&[h])?,norm:tensor("norm.weight",&[h])? })
}

/// Header-validated final BF16 rank destinations; source tensors never convert.
pub fn block_dspark_rank_arena_requests(snapshot:&Path,rank:usize,ranks:usize,lead:usize)
    ->anyhow::Result<Vec<(String,usize)>> {
    block_dspark_rank_arena_requests_with_representation(snapshot,rank,ranks,lead,
        crate::families::glm5::draft_representation::GlmDraftRepresentation::Bf16Only)
}
pub fn block_dspark_rank_arena_requests_with_representation(snapshot:&Path,rank:usize,ranks:usize,lead:usize,
    representation:crate::families::glm5::draft_representation::GlmDraftRepresentation)
    ->anyhow::Result<Vec<(String,usize)>> {
    use anyhow::{ensure,Context};
    let cfg=BlockDsparkConfig::read(snapshot)?;
    let checkpoint=crate::plan::Checkpoint::open(snapshot)?;
    let headers=checkpoint.tensors.iter().map(|t|(t.meta.name.as_str(),&t.meta)).collect::<std::collections::BTreeMap<_,_>>();
    let validate=|name:&str,shape:&[usize]|->anyhow::Result<usize> {
        let header=headers.get(name).with_context(||format!("block draft tensor missing: {name}"))?;
        let bytes=shape.iter().try_fold(2usize,|n,&d|n.checked_mul(d)).context("block rank tensor overflow")?;
        ensure!(header.dtype==cuteafd_core::DType::Bf16 && header.shape==shape && header.byte_length==bytes as u64,
            "block draft source header differs for {name}");
        Ok(bytes)
    };
    let requests=std::cell::RefCell::new(Vec::new());
    block_dspark_rank_recipe(&cfg,rank,ranks,lead,|name,shape| {
        requests.borrow_mut().push((name.to_owned(),crate::serving_capacity::WeightArenaLayout::allocation_bytes(validate(name,shape)?))); Ok(())
    },|parts,cols,axis| {
        let mut bytes=0usize;
        for &(name,rows) in parts {
            validate(name,&[rows,cols])?;
            let (row,column)=block_dspark_part_ranges(rows,cols,axis,rank,ranks)?;
            let local=row.len().checked_mul(column.len()).and_then(|n|n.checked_mul(2))
                .context("block rank part extent overflow")?;
            bytes=bytes.checked_add(local).context("block rank matrix overflow")?;
        }
        let name=parts.iter().map(|(name,_)|*name).collect::<Vec<_>>().join("+");
        if representation == crate::families::glm5::draft_representation::GlmDraftRepresentation::Fp8Only {
            requests.borrow_mut().extend([(format!("{name}.payload"),bytes/2),
                (format!("{name}.scales"),bytes/2/128*4)]);
        } else {
            requests.borrow_mut().push((name,crate::serving_capacity::WeightArenaLayout::allocation_bytes(bytes)));
        }
        Ok(())
    })?;
    if rank==lead { requests.borrow_mut().push(("markov_norms".into(),cfg.vocab*4)); }
    if cfg.head_dim!=64 { requests.borrow_mut().push(("inv_freq".into(),crate::serving_capacity::WeightArenaLayout::allocation_bytes(std::mem::size_of_val(cfg.rope.plan(cfg.head_dim)?.inv_freq.as_slice())))); }
    Ok(requests.into_inner())
}

/// Native mixed-format operands remain in their own checkpoint encoding.
fn native_arena_requests(cfg: &BlockDsparkConfig,
    headers: &std::collections::HashMap<String, crate::SafetensorsTensorMetadata>)
    -> anyhow::Result<Vec<(String, usize)>> {
    use anyhow::{ensure, Context};
    use cuteafd_core::DType;
    let requests = std::cell::RefCell::new(Vec::new());
    let tensor = |name: &str, shape: &[usize], dtype: DType, element_bytes: usize| -> anyhow::Result<usize> {
        let header = headers.get(name).with_context(|| format!("drafter checkpoint has no {name}"))?;
        let bytes = shape.iter().try_fold(element_bytes, |sum, &n| sum.checked_mul(n))
            .context("native draft tensor extent overflow")?;
        ensure!(header.dtype == dtype && header.shape == shape && header.byte_length == bytes as u64,
            "native drafter {name}: {:?} {:?}, expected {dtype:?} {shape:?}", header.dtype, header.shape);
        Ok(bytes)
    };
    block_dspark_weight_recipe(cfg, |name, shape| {
        let bytes = tensor(name, shape, DType::Bf16, 2)?;
        requests.borrow_mut().push((name.to_owned(), crate::serving_capacity::WeightArenaLayout::allocation_bytes(bytes)));
        Ok::<_, anyhow::Error>(())
    }, |parts, columns| {
        for &(name, rows) in parts {
            let header = headers.get(name).with_context(|| format!("drafter checkpoint has no {name}"))?;
            match &header.dtype {
                DType::Bf16 => {
                    let bytes = tensor(name, &[rows, columns], DType::Bf16, 2)?;
                    requests.borrow_mut().push((name.to_owned(), crate::serving_capacity::WeightArenaLayout::allocation_bytes(bytes)));
                }
                DType::F8E4M3 => {
                    ensure!(columns % 32 == 0, "native MXFP8 draft input must contain complete group32 scales");
                    let bytes = tensor(name, &[rows, columns], DType::F8E4M3, 1)?;
                    let scales = format!("{name}_scale_inv");
                    let scale_bytes = tensor(&scales, &[rows, columns / 32], DType::U8, 1)?;
                    requests.borrow_mut().extend([(name.to_owned(), bytes), (scales, scale_bytes)]);
                }
                other => anyhow::bail!("native drafter {name}: unsupported projection format {other:?}"),
            }
        }
        Ok(())
    })?;
    requests.borrow_mut().push(("markov_norms".into(), cfg.vocab.checked_mul(4).context("Markov norms extent overflow")?));
    let requests = requests.into_inner();
    ensure!(requests.iter().map(|(name, _)| name).collect::<std::collections::BTreeSet<_>>().len() == requests.len(),
        "native draft recipe repeats a physical owner");
    Ok(requests)
}

/// Final device owners the loader binds, in order, from checkpoint headers:
/// one request per tensor, row-concatenated BF16 matrix, or FP8 payload/scales
/// plane pair, plus the Markov norms and the RoPE table. A runtime arena and
/// admission both size from exactly this list.
pub fn block_dspark_arena_requests(snapshot: &Path,
    representation: crate::families::glm5::draft_representation::GlmDraftRepresentation)
    -> anyhow::Result<Vec<(String, usize)>> {
    use crate::families::glm5::draft_representation::GlmDraftRepresentation;
    use anyhow::{ensure, Context};
    let cfg = BlockDsparkConfig::read(snapshot)?;
    let checkpoint = crate::plan::checkpoint::Checkpoint::open(snapshot)?;
    let headers = checkpoint.tensors.iter().map(|t| (t.meta.name.clone(), t.meta.clone()))
        .collect::<std::collections::HashMap<_, _>>();
    if cfg.dialect == DsparkDialect::MiniMax {
        return native_arena_requests(&cfg, &headers);
    }
    let bf16 = |name: &str, shape: &[usize]| -> anyhow::Result<usize> {
        let tensor = headers.get(name).with_context(|| format!("drafter checkpoint has no {name}"))?;
        ensure!(tensor.dtype == cuteafd_core::DType::Bf16, "{name}: drafter weights must be BF16, found {:?}", tensor.dtype);
        let bytes = shape.iter().try_fold(2usize, |n, &d| n.checked_mul(d))
            .with_context(|| format!("{name}: BF16 tensor byte count overflow"))?;
        ensure!(tensor.shape == shape && tensor.byte_length == bytes as u64,
            "{name}: shape {:?}/{} bytes, expected BF16 {shape:?}/{bytes} bytes", tensor.shape, tensor.byte_length);
        Ok(bytes)
    };
    let requests = std::cell::RefCell::new(Vec::new());
    block_dspark_weight_recipe(&cfg, |name, shape| {
        requests.borrow_mut().push((name.to_owned(), bf16(name, shape)?.max(256)));
        Ok::<_, anyhow::Error>(())
    }, |parts, cols| {
        let bytes = parts.iter().try_fold(0usize, |n, (name, rows)| -> anyhow::Result<usize> {
            n.checked_add(bf16(name, &[*rows, cols])?).context("block dSpark matrix extent overflow")
        })?;
        let name = parts.iter().map(|(name, _)| *name).collect::<Vec<_>>().join("+");
        if representation == GlmDraftRepresentation::Bf16Only {
            requests.borrow_mut().push((name, crate::serving_capacity::WeightArenaLayout::allocation_bytes(bytes)));
        } else {
            let rows = parts.iter().try_fold(0usize, |n, (_, rows)| n.checked_add(*rows))
                .context("block dSpark matrix rows overflow")?;
            ensure!(rows > 0 && rows % 16 == 0 && cols > 0 && cols % 128 == 0,
                "block dSpark FP8 shape {rows}/{cols}");
            let payload = bytes / 2;
            requests.borrow_mut().extend([(format!("{name}.payload"), payload),
                (format!("{name}.scales"), payload / 128 * 4)]);
        }
        Ok(())
    })?;
    let mut requests = requests.into_inner();
    requests.push(("markov_norms".into(), cfg.vocab.checked_mul(4).context("block dSpark norms extent overflow")?));
    if cfg.head_dim != 64 {
        let plan = cfg.rope.plan(cfg.head_dim)?;
        requests.push(("inv_freq".into(), std::mem::size_of_val(plan.inv_freq.as_slice()).max(256)));
    }
    Ok(requests)
}
