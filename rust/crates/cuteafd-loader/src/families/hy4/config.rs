//! Hy4 (`hy_v4_vl`, `HYV4VLForConditionalGeneration`) configuration reader
//! shared by `cuteafd plan`, placement admission, golden and serving (#2550).
//!
//! Verified against `tencent/Hy4@c10c0de5b6f850703fb0a90e65d1c32b558b9167` and
//! `python/reference/families/hy4/model.py` / `sources.py`.
use anyhow::{bail, ensure, Context, Result};
use serde_json::Value;
use std::path::Path;

use crate::families::hy4_common::{
    json_bool, json_float, json_int, json_token_id, json_token_ids, SPARSE_ATTENTION,
};
use crate::plan::checkpoint::read_json;

const FAMILY: &str = "Hy4";

/// Per-layer DSA IndexCache role (`text_config.indexer_types`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hy4IndexerKind {
    /// Computes per-token indexer Q/K/weights, updates its index K cache, and
    /// selects top-2048 physical slots.
    Full,
    /// Reuses the nearest preceding `Full` layer's top-2048 selection; owns no
    /// indexer weights and no index K cache pages.
    Shared,
}

/// Validated IndexCache mapping across all decoder layers.
///
/// In `tencent/Hy4` (78 layers):
/// - 21 full indexer layers: `0, 1, 5, 9, 13, ..., 77`
/// - 57 shared layers: layer `L` reuses `source_layer[L]` (the preceding full layer).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexCacheSchedule {
    pub kinds: Vec<Hy4IndexerKind>,
    /// Nearest preceding `Full` layer index for every layer `0..layers`.
    pub source_layer: Vec<usize>,
    /// Ordered 0-based indices of every `Full` indexer layer.
    pub full_layers: Vec<usize>,
}

impl IndexCacheSchedule {
    pub fn from_kinds(kinds: Vec<Hy4IndexerKind>) -> Result<Self> {
        ensure!(!kinds.is_empty(), "Hy4 indexer_types must not be empty");
        ensure!(
            kinds[0] == Hy4IndexerKind::Full,
            "Hy4 layer 0 must have indexer_types=\"full\" so every shared layer has a preceding full source"
        );
        let mut source_layer = Vec::with_capacity(kinds.len());
        let mut full_layers = Vec::new();
        let mut current_full = 0usize;
        for (layer, &kind) in kinds.iter().enumerate() {
            if kind == Hy4IndexerKind::Full {
                current_full = layer;
                full_layers.push(layer);
            }
            source_layer.push(current_full);
        }
        Ok(Self {
            kinds,
            source_layer,
            full_layers,
        })
    }

    pub fn is_full(&self, layer: usize) -> bool {
        self.kinds.get(layer) == Some(&Hy4IndexerKind::Full)
    }

    pub fn source_for(&self, layer: usize) -> usize {
        self.source_layer[layer]
    }

    pub fn full_count(&self) -> usize {
        self.full_layers.len()
    }

    pub fn shared_count(&self) -> usize {
        self.kinds.len() - self.full_layers.len()
    }
}

/// Text backbone configuration parsed from `config.json`.
#[derive(Debug, Clone, PartialEq)]
pub struct Hy4Config {
    pub vocab_size: usize,
    pub hidden: usize,
    pub layers: usize,
    pub intermediate: usize,
    pub max_position: usize,
    /// `config.rms_norm_eps` (`1e-5`): used by `input_layernorm`,
    /// `post_attention_layernorm`, `model.norm`, iHC RMSNorm, and indexer `k_norm`.
    pub rms_norm_eps: f64,
    /// `q_a_layernorm` / `kv_a_layernorm` epsilon (`1e-6`, matching Transformers main
    /// and the published F32 fidelity golden on #2550).
    pub qa_kva_eps: f64,
    // iHC (4-stream identity hyper-connections, FP32 mixing, no Sinkhorn)
    pub hc_mult: usize,
    pub hc_eps: f64,
    pub hc_magnitude: f64,
    // Uniform gated DSA MLA across all 78 layers
    pub heads: usize,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_dim: usize,
    pub qk_rope_dim: usize,
    pub v_head_dim: usize,
    pub rope_theta: f64,
    // 32-head per-token DSA indexer + IndexCache
    pub indexer_heads: usize,
    pub indexer_head_dim: usize,
    pub indexer_topk: usize,
    pub index_cache: IndexCacheSchedule,
    // FFN / direct MoE (no LatentMoE, no Engram, no SWA, no sandwich norms)
    pub dense_layer: Vec<bool>,
    pub experts: usize,
    pub topk: usize,
    pub moe_intermediate: usize,
    pub shared_experts: usize,
    pub n_group: usize,
    pub topk_group: usize,
    pub norm_topk_prob: bool,
    pub routed_scaling_factor: f64,
    pub swiglu_limit: f64,
    // Special tokens
    pub image_token_id: u32,
    pub video_token_id: u32,
    pub eos: Vec<u32>,
}

impl Hy4Config {
    pub fn read(snapshot: &Path) -> Result<Self> {
        Self::from_hf(&read_json(&snapshot.join("config.json"))?)
    }

    pub fn from_hf(root: &Value) -> Result<Self> {
        ensure!(
            root["model_type"] == "hy_v4_vl",
            "Hy4 requires root model_type=hy_v4_vl, found {}",
            root["model_type"]
        );
        let v = root
            .get("text_config")
            .filter(|v| v.is_object())
            .context("Hy4 requires text_config")?;
        ensure!(
            v["model_type"] == "hy_v4",
            "Hy4 requires text_config.model_type=hy_v4, found {}",
            v["model_type"]
        );
        ensure!(
            v["hidden_act"] == "silu",
            "Hy4 requires hidden_act=silu, found {}",
            v["hidden_act"]
        );
        json_bool(v, FAMILY, "attention_bias", false)?;
        for key in ["enable_ihc", "gated_mla", "enable_lm_head_fp32", "learnable_sink"] {
            json_bool(v, FAMILY, key, true)?;
        }
        ensure!(
            v["gating_type"] == "elementwise",
            "Hy4 requires gating_type=elementwise, found {}",
            v["gating_type"]
        );
        ensure!(
            root.get("tie_word_embeddings").is_none_or(|b| b.as_bool() == Some(false))
                && v.get("tie_word_embeddings").is_none_or(|b| b.as_bool() == Some(false)),
            "Hy4 requires tie_word_embeddings=false"
        );
        // Reject Hy4-flash-only features if ever accidentally mixed into a hy_v4_vl config.
        ensure!(
            v.get("engram_layer_ids").is_none(),
            "Hy4 does not have Engram layers (engram_layer_ids belongs to hy4_flash)"
        );
        ensure!(
            v.get("routed_expert_hidden_size").is_none(),
            "Hy4 uses direct MoE, not LatentMoE (routed_expert_hidden_size belongs to hy4_flash)"
        );
        ensure!(
            v.get("use_compressed_indexer").is_none_or(|b| b.as_bool() == Some(false)),
            "Hy4 uses per-token DSA IndexCache, not compressed_indexer"
        );

        let layers = json_int(v, FAMILY, "num_hidden_layers")?;
        let vocab_size = json_int(v, FAMILY, "vocab_size")?;
        let hidden = json_int(v, FAMILY, "hidden_size")?;
        let intermediate = json_int(v, FAMILY, "intermediate_size")?;

        let raw_layer_types = v["layer_types"]
            .as_array()
            .context("Hy4 config lacks layer_types")?;
        ensure!(
            raw_layer_types.len() == layers,
            "Hy4 layer_types length {} differs from num_hidden_layers {layers}",
            raw_layer_types.len()
        );
        for (idx, entry) in raw_layer_types.iter().enumerate() {
            ensure!(
                entry.as_str() == Some(SPARSE_ATTENTION),
                "Hy4 layer {idx} requires {SPARSE_ATTENTION}, found {entry}"
            );
        }

        let raw_mlp_types = v["mlp_layer_types"]
            .as_array()
            .context("Hy4 config lacks mlp_layer_types")?;
        ensure!(
            raw_mlp_types.len() == layers,
            "Hy4 mlp_layer_types length {} differs from num_hidden_layers {layers}",
            raw_mlp_types.len()
        );
        let mut dense_layer = Vec::with_capacity(layers);
        for entry in raw_mlp_types {
            match entry.as_str() {
                Some("dense") => dense_layer.push(true),
                Some("sparse") => dense_layer.push(false),
                _ => bail!("invalid Hy4 mlp_layer_types entry {entry}"),
            }
        }
        if let Some(first_k) = v.get("first_k_dense_replace").and_then(Value::as_u64) {
            let expected: Vec<bool> = (0..layers).map(|l| (l as u64) < first_k).collect();
            ensure!(
                dense_layer == expected,
                "Hy4 mlp_layer_types disagrees with first_k_dense_replace={first_k}"
            );
        }

        let raw_indexer_types = v["indexer_types"]
            .as_array()
            .context("Hy4 config lacks indexer_types")?;
        ensure!(
            raw_indexer_types.len() == layers,
            "Hy4 indexer_types length {} differs from num_hidden_layers {layers}",
            raw_indexer_types.len()
        );
        let mut indexer_kinds = Vec::with_capacity(layers);
        for entry in raw_indexer_types {
            match entry.as_str() {
                Some("full") => indexer_kinds.push(Hy4IndexerKind::Full),
                Some("shared") => indexer_kinds.push(Hy4IndexerKind::Shared),
                _ => bail!("invalid Hy4 indexer_types entry {entry}"),
            }
        }
        let index_cache = IndexCacheSchedule::from_kinds(indexer_kinds)?;

        let rope = v
            .get("rope_parameters")
            .filter(|r| r.is_object())
            .context("Hy4 config lacks rope_parameters")?;
        ensure!(
            rope["rope_type"] == "default" && rope["mrope_interleaved"] == true,
            "Hy4 requires default interleaved RoPE (settled #2550), found {rope}"
        );
        let rope_theta = json_float(rope, FAMILY, "rope_theta")?;

        let heads = json_int(v, FAMILY, "num_attention_heads")?;
        let q_lora_rank = json_int(v, FAMILY, "q_lora_rank")?;
        let kv_lora_rank = json_int(v, FAMILY, "kv_lora_rank")?;
        let qk_nope_dim = json_int(v, FAMILY, "qk_nope_head_dim")?;
        let qk_rope_dim = json_int(v, FAMILY, "qk_rope_head_dim")?;
        let v_head_dim = json_int(v, FAMILY, "v_head_dim")?;
        if let Some(qk_head_dim) = v.get("qk_head_dim").and_then(Value::as_u64) {
            ensure!(
                qk_head_dim as usize == qk_nope_dim + qk_rope_dim,
                "Hy4 qk_head_dim {qk_head_dim} differs from qk_nope+qk_rope {}",
                qk_nope_dim + qk_rope_dim
            );
        }

        let indexer_heads = json_int(v, FAMILY, "index_n_heads")?;
        let indexer_head_dim = json_int(v, FAMILY, "index_head_dim")?;
        let indexer_topk = json_int(v, FAMILY, "index_topk")?;

        let experts = json_int(v, FAMILY, "n_routed_experts")?;
        let topk = json_int(v, FAMILY, "num_experts_per_tok")?;
        ensure!(topk <= experts, "Hy4 topk {topk} exceeds experts {experts}");
        let moe_intermediate = json_int(v, FAMILY, "moe_intermediate_size")?;
        let shared_experts = json_int(v, FAMILY, "n_shared_experts")?;
        let n_group = json_int(v, FAMILY, "n_group")?;
        let topk_group = json_int(v, FAMILY, "topk_group")?;
        ensure!(
            n_group <= experts && experts % n_group == 0 && topk_group <= n_group,
            "invalid Hy4 router groups ({n_group}, {topk_group}) for {experts} experts"
        );
        let norm_topk_prob = v["norm_topk_prob"]
            .as_bool()
            .context("Hy4 config lacks norm_topk_prob")?;

        let image_token_id = json_token_id(root, FAMILY, "image_token_id", vocab_size)?;
        let video_token_id = json_token_id(root, FAMILY, "video_token_id", vocab_size)?;
        let eos = json_token_ids(
            root.get("eos_token_id")
                .or_else(|| v.get("eos_token_id"))
                .context("Hy4 config lacks eos_token_id")?,
            FAMILY,
            vocab_size,
            "eos_token_id",
        )?;

        Ok(Self {
            vocab_size,
            hidden,
            layers,
            intermediate,
            max_position: json_int(v, FAMILY, "max_position_embeddings")?,
            rms_norm_eps: json_float(v, FAMILY, "rms_norm_eps")?,
            qa_kva_eps: 1e-6,
            hc_mult: json_int(v, FAMILY, "hc_mult")?,
            hc_eps: json_float(v, FAMILY, "hc_eps")?,
            hc_magnitude: json_float(v, FAMILY, "hc_magnitude")?,
            heads,
            q_lora_rank,
            kv_lora_rank,
            qk_nope_dim,
            qk_rope_dim,
            v_head_dim,
            rope_theta,
            indexer_heads,
            indexer_head_dim,
            indexer_topk,
            index_cache,
            dense_layer,
            experts,
            topk,
            moe_intermediate,
            shared_experts,
            n_group,
            topk_group,
            norm_topk_prob,
            routed_scaling_factor: json_float(v, FAMILY, "routed_scaling_factor")?,
            swiglu_limit: json_float(v, FAMILY, "swiglu_limit")?,
            image_token_id,
            video_token_id,
            eos,
        })
    }

    pub const fn qk_head_dim(&self) -> usize {
        self.qk_nope_dim + self.qk_rope_dim
    }

    pub fn is_full_indexer(&self, layer: usize) -> bool {
        self.index_cache.is_full(layer)
    }

    pub fn index_source_layer(&self, layer: usize) -> usize {
        self.index_cache.source_for(layer)
    }

    pub fn full_indexer_layers(&self) -> &[usize] {
        &self.index_cache.full_layers
    }

    pub fn routed_layers(&self) -> Vec<usize> {
        (0..self.layers).filter(|&i| !self.dense_layer[i]).collect()
    }

    /// SwiGLU clamp (`swiglu_limit = 10.0`) applies to routed experts only;
    /// dense layer 0 and shared experts are unclamped (matches
    /// `python/reference/families/hy4/model.py`).
    pub fn activation(&self, layer: usize, shared: bool) -> cuteafd_core::FfnActivation {
        let limit = (!self.dense_layer[layer] && !shared).then_some(self.swiglu_limit as f32);
        cuteafd_core::FfnActivation::SiluBf16Product { limit }
    }

    /// Verify exact release geometry of `tencent/Hy4` (#2550).
    pub fn release_geometry(&self) -> Result<()> {
        let shape = (
            self.hidden,
            self.layers,
            self.hc_mult,
            self.heads,
            self.q_lora_rank,
            self.kv_lora_rank,
            self.qk_nope_dim,
            self.qk_rope_dim,
            self.v_head_dim,
        );
        ensure!(
            shape == (6144, 78, 4, 64, 2048, 512, 192, 64, 256),
            "Hy4 attention/backbone geometry {shape:?} differs from the supported Hy4 release"
        );
        let indexer = (self.indexer_heads, self.indexer_head_dim, self.indexer_topk);
        ensure!(
            indexer == (32, 128, 2048),
            "Hy4 DSA indexer geometry {indexer:?} differs from the supported Hy4 release"
        );
        let expected_full: Vec<usize> = std::iter::once(0)
            .chain((1..78).step_by(4))
            .collect();
        ensure!(
            self.index_cache.full_layers == expected_full,
            "Hy4 IndexCache full layers {:?} differ from expected 21-layer schedule (0, 1, 5, ..., 77)",
            self.index_cache.full_layers
        );
        let moe = (
            self.experts,
            self.topk,
            self.moe_intermediate,
            self.shared_experts,
            self.intermediate,
            self.vocab_size,
        );
        ensure!(
            moe == (256, 8, 2048, 1, 18432, 120_832),
            "Hy4 MoE/head geometry {moe:?} differs from the supported Hy4 release"
        );
        Ok(())
    }
}
