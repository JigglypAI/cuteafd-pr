//! Execution constraints are family facts; checkpoint shard sides describe storage only.
use super::{DenseRecord, DenseEncoding};
use crate::families::{glm5::GlmDsaConfig, glm5_flash::{GlmNextConfig, GlmNextAttention}};
use crate::formats::trellis_core::CoreSplit;
use crate::kda::KdaRole;
use anyhow::{ensure, Context};
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum DenseAxis { Output, Input }

impl From<CoreSplit> for DenseExecution {
    fn from(split: CoreSplit) -> Self {
        match split {
            CoreSplit::Whole => Self::Replicated,
            CoreSplit::Output => Self::Split(DenseAxis::Output),
            CoreSplit::Input => Self::Split(DenseAxis::Input),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct KvBGeometry {
    pub heads: usize,
    pub nope: usize,
    pub values: usize,
    pub latent: usize,
}

/// The four resident operands produced by the existing native FP8 absorption job.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvBLayout {
    pub first_head: usize,
    pub heads: usize,
    pub geometry: KvBGeometry,
    pub allocations: [usize; 4],
}

impl KvBGeometry {
    pub fn layout(self, world: usize, rank: usize) -> anyhow::Result<KvBLayout> {
        ensure!(world > 0 && rank < world && self.heads > 0 && self.heads % world == 0
            && self.nope > 0 && self.nope % 64 == 0 && self.values > 0
            && self.latent > 0 && self.latent % 64 == 0, "invalid kv_b head/scale geometry");
        let heads = self.heads / world;
        let bytes = |dims: &[usize]| super::product(dims);
        Ok(KvBLayout { first_head: rank * heads, heads, geometry: self, allocations: [
            bytes(&[heads, self.latent, self.nope])?,
            bytes(&[heads, self.latent, self.nope / 64, 4])?,
            bytes(&[heads, self.values, self.latent])?,
            bytes(&[heads, self.values, self.latent / 64, 4])?,
        ] })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum DenseExecution {
    RankZero,
    Replicated,
    Split(DenseAxis),
    AbsorbHeads(KvBGeometry),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct DenseOwnership {
    pub execution: DenseExecution,
    /// Only this MLA layer's attention records may follow a planner-selected whole-layer owner.
    pub whole_mla: Option<usize>,
}

impl DenseOwnership {
    pub const fn new(execution: DenseExecution) -> Self { Self { execution, whole_mla: None } }
}

pub(super) enum DenseFamily { Glm(GlmDsaConfig), Flash(GlmNextConfig) }

impl DenseFamily {
    pub(super) fn from_hf(config: &serde_json::Value) -> anyhow::Result<Self> {
        let text = config.get("text_config").unwrap_or(config);
        match text["model_type"].as_str() {
            Some("glm_moe_dsa") => Ok(Self::Glm(GlmDsaConfig::from_hf(config)?)),
            Some("glm5_next_text") => Ok(Self::Flash(GlmNextConfig::from_hf(config)?)),
            _ => anyhow::bail!("unsupported native dense family config"),
        }
    }

    pub(super) fn ownership(&self, tensor: &str) -> anyhow::Result<DenseOwnership> {
        let (prefix, flash) = match self {
            Self::Glm(_) => ("model.", false),
            Self::Flash(_) => ("model.language_model.", true),
        };
        if tensor == "lm_head.weight" || ["embed_tokens.weight", "norm.weight"].iter()
            .any(|leaf| tensor.strip_prefix(prefix) == Some(*leaf)) {
            return Ok(DenseOwnership::new(DenseExecution::RankZero));
        }
        let root = format!("{prefix}layers.");
        let (layer, leaf) = tensor.strip_prefix(&root).and_then(|s| s.split_once('.'))
            .with_context(|| format!("{tensor}: no native dense execution constraint"))?;
        let layer: usize = layer.parse()?;
        let (layers, mla, kv) = match self {
            Self::Glm(cfg) => (cfg.layers, true, KvBGeometry { heads: cfg.heads,
                nope: cfg.qk_nope_head_dim, values: cfg.v_head_dim, latent: cfg.kv_lora_rank }),
            Self::Flash(cfg) => (cfg.layers, cfg.attention.get(layer) == Some(&GlmNextAttention::Mla),
                KvBGeometry { heads: cfg.heads, nope: cfg.qk_nope_dim,
                    values: cfg.v_head_dim, latent: cfg.kv_lora_rank }),
        };
        ensure!(layer < layers, "{tensor}: dense layer outside backbone");
        let stem = tensor.strip_suffix(".weight").unwrap_or(tensor);
        let split = match self {
            Self::Glm(cfg) => crate::families::glm5::core::split(cfg, stem),
            Self::Flash(cfg) => crate::families::glm5_flash::core::split(cfg, stem),
        };
        let execution = if leaf == "self_attn.kv_b_proj.weight" {
            ensure!(mla, "{tensor}: kv_b requires MLA");
            DenseExecution::AbsorbHeads(kv)
        } else if leaf.starts_with("self_attn.indexer.") {
            // All index heads execute on each participating attention rank.
            DenseExecution::Replicated
        } else if leaf == "mlp.gate.weight" || leaf == "mlp.gate.e_score_correction_bias" {
            if flash { DenseExecution::Replicated } else { DenseExecution::RankZero }
        } else if let Some(split) = split {
            split.into()
        } else if flash && !mla && leaf.starts_with("self_attn.") {
            let role = match leaf.trim_start_matches("self_attn.") {
                "f_a_proj.weight" => KdaRole::Fa, "g_a_proj.weight" => KdaRole::Ga,
                "f_b_proj.weight" => KdaRole::Fb, "g_b_proj.weight" => KdaRole::Gb,
                "b_proj.weight" => KdaRole::Beta,
                "q_conv1d.weight" => KdaRole::ConvQ, "k_conv1d.weight" => KdaRole::ConvK,
                "v_conv1d.weight" => KdaRole::ConvV, "A_log" => KdaRole::ALog,
                "dt_bias" => KdaRole::DtBias, "o_norm.weight" => KdaRole::ONorm,
                _ => anyhow::bail!("{tensor}: no KDA execution constraint"),
            };
            role.split().into()
        } else {
            ensure!(!leaf.contains("_proj.") && !leaf.contains(".experts."),
                "{tensor}: no native projection execution constraint");
            DenseExecution::Replicated
        };
        Ok(DenseOwnership { execution, whole_mla: (flash && mla && leaf.starts_with("self_attn.")).then_some(layer) })
    }
}

impl DenseRecord {
    pub(super) fn validate_ownership(&self) -> anyhow::Result<()> {
        if let DenseExecution::AbsorbHeads(g) = self.ownership.execution {
            ensure!(self.encoding == DenseEncoding::BlockFp8
                && self.shape == [super::product(&[g.heads, g.nope.checked_add(g.values).context("kv_b rows overflow")?])?, g.latent],
                "{}: kv_b absorption shape/encoding mismatch", self.tensor);
            g.layout(1, 0)?;
        }
        Ok(())
    }
}
