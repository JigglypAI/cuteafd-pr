//! Hy4-flash (`hy_v4_flash`, `HYV4FlashForConditionalGeneration`) configuration
//! reader shared by `cuteafd plan`, placement admission, golden and serving.
//!
//! Verified against `tencent/Hy4-flash@d88c9dfd` / `wrldsuksgo2mars/Hy4-flash-EXL3-K4-MC@cc721791`
//! and `python/reference/families/hy4_flash/model.py`.
use anyhow::{bail, ensure, Context, Result};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::Path;

use crate::plan::checkpoint::read_json;

pub const SPARSE_ATTENTION: &str = "deepseek_sparse_attention";
pub const SLIDING_ATTENTION: &str = "sliding_attention";

/// Attention geometry for one Hy4-flash decoder layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Hy4LayerAttention {
    pub sparse: bool,
    pub heads: usize,
    pub qk_nope_dim: usize,
    pub qk_rope_dim: usize,
    pub v_head_dim: usize,
}

impl Hy4LayerAttention {
    pub const fn qk_head_dim(self) -> usize {
        self.qk_nope_dim + self.qk_rope_dim
    }
}

/// Text backbone configuration parsed from `config.json`.
#[derive(Debug, Clone, PartialEq)]
pub struct Hy4FlashConfig {
    pub vocab_size: usize,
    pub hidden: usize,
    pub layers: usize,
    pub intermediate: usize,
    pub max_position: usize,
    pub rms_norm_eps: f64,
    // iHC (identity hyper-connections)
    pub hc_mult: usize,
    pub hc_eps: f64,
    pub hc_magnitude: f64,
    // MLA shared dimensions
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_dim: usize,
    pub qk_rope_dim: usize,
    pub v_head_dim: usize,
    pub sparse_heads: usize,
    pub swa_heads: usize,
    pub swa_qk_nope_dim: usize,
    pub swa_qk_rope_dim: usize,
    pub swa_v_head_dim: usize,
    pub sliding_window: usize,
    pub sparse_rope_theta: f64,
    pub swa_rope_theta: f64,
    /// `sparse_layer[i] == true` for DSA layers, `false` for SWA-512 layers.
    pub sparse_layer: Vec<bool>,
    // Compressed indexer (on DSA layers)
    pub indexer_heads: usize,
    pub indexer_head_dim: usize,
    pub indexer_compress_ratio: usize,
    pub indexer_group_topk: usize,
    pub indexer_candidate_topk: usize,
    // FFN / LatentMoE
    /// `dense_layer[i] == true` for dense FFN layers, `false` for LatentMoE layers.
    pub dense_layer: Vec<bool>,
    pub experts: usize,
    pub topk: usize,
    pub moe_intermediate: usize,
    pub routed_expert_hidden: usize,
    pub shared_experts: usize,
    pub n_group: usize,
    pub topk_group: usize,
    pub norm_topk_prob: bool,
    pub routed_scaling_factor: f64,
    pub swiglu_limit: f64,
    // Engram
    /// 0-based decoder layer indices carrying an Engram layer (config's `engram_layer_ids` is 1-based).
    pub engram_layers: Vec<usize>,
    /// Flat per-head vocab sizes (length `(engram_max_ngram_size - 1) * engram_n_head_per_ngram`)
    /// keyed by 0-based decoder layer index.
    pub engram_vocab_sizes: BTreeMap<usize, Vec<u64>>,
    pub engram_max_ngram_size: usize,
    pub engram_n_embed_per_ngram: usize,
    pub engram_n_head_per_ngram: usize,
    pub engram_pad_id: u32,
    pub engram_seed: u64,
    pub engram_fp8_embedding: bool,
    pub engram_lookup_table_size: usize,
    // Special tokens
    pub image_token_id: u32,
    pub video_token_id: u32,
    pub eos: Vec<u32>,
}

/// Bundled 5-layer DFly2 drafter config (`dfly_*` top-level fields).
#[derive(Debug, Clone, PartialEq)]
pub struct Hy4DflyConfig {
    pub layers: usize,
    pub target_kv_layers: Vec<usize>,
    pub target_index_layers: Vec<usize>,
    pub target_index_topk: usize,
    pub mask_token_id: u32,
    pub enable_hidden_correction: bool,
    pub mtp_fusion: String,
    pub num_speculative_tokens: usize,
}

/// Bundled 27-layer ViT vision config (`vision_config`).
#[derive(Debug, Clone, PartialEq)]
pub struct Hy4VisionConfig {
    pub hidden: usize,
    pub intermediate: usize,
    pub heads: usize,
    pub layers: usize,
    pub out_hidden: usize,
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    pub rms_norm_eps: f64,
}

fn int(value: &Value, key: &str) -> Result<usize> {
    let n = value[key].as_u64().with_context(|| format!("Hy4-flash config lacks {key}"))?;
    ensure!(n > 0 && n <= (1 << 24), "invalid Hy4-flash {key}: {n}");
    Ok(n as usize)
}

fn float(value: &Value, key: &str) -> Result<f64> {
    let x = value[key].as_f64().filter(|x| x.is_finite())
        .with_context(|| format!("Hy4-flash config lacks {key}"))?;
    ensure!(x > 0.0, "Hy4-flash {key} must be positive, found {x}");
    Ok(x)
}

fn boolean(value: &Value, key: &str, expected: bool) -> Result<()> {
    ensure!(value[key].as_bool() == Some(expected),
        "Hy4-flash {key} must be {expected}, found {}", value[key]);
    Ok(())
}

fn token_ids(value: &Value, vocab: usize, key: &str) -> Result<Vec<u32>> {
    let one = |id: &Value| -> Result<u32> {
        let id = id.as_u64().with_context(|| format!("Hy4-flash {key} must be a token id"))?;
        ensure!(id < vocab as u64, "Hy4-flash {key} {id} is outside vocabulary {vocab}");
        Ok(u32::try_from(id)?)
    };
    match value {
        Value::Array(ids) if !ids.is_empty() => ids.iter().map(one).collect(),
        Value::Array(_) => bail!("Hy4-flash {key} must not be empty"),
        other => Ok(vec![one(other)?]),
    }
}

fn token_id(value: &Value, key: &str, vocab: usize) -> Result<u32> {
    let id = value[key].as_u64().with_context(|| format!("Hy4-flash config lacks {key}"))?;
    ensure!(id < vocab as u64, "Hy4-flash {key} {id} is outside vocabulary {vocab}");
    Ok(u32::try_from(id)?)
}

impl Hy4FlashConfig {
    pub fn read(snapshot: &Path) -> Result<Self> {
        Self::from_hf(&read_json(&snapshot.join("config.json"))?)
    }

    pub fn from_hf(root: &Value) -> Result<Self> {
        ensure!(root["model_type"] == "hy_v4_flash",
            "Hy4-flash requires model_type=hy_v4_flash, found {}", root["model_type"]);
        let v = root.get("text_config").filter(|v| v.is_object())
            .context("Hy4-flash requires text_config")?;
        ensure!(v["hidden_act"] == "silu", "Hy4-flash requires hidden_act=silu, found {}", v["hidden_act"]);
        boolean(v, "attention_bias", false)?;
        for key in ["enable_ihc", "use_mla", "gated_mla", "enable_lm_head_fp32",
                    "moe_router_enable_expert_bias", "moe_router_use_sigmoid",
                    "compressed_indexer_skip_fine_select", "learnable_sink", "sandwich_norm",
                    "use_compressed_indexer", "use_dsa"] {
            boolean(v, key, true)?;
        }
        boolean(v, "dsa_skip_attn_post_norm", false)?;
        ensure!(v["compressed_indexer_key_pooling"] == "gated",
            "Hy4-flash requires compressed_indexer_key_pooling=gated");
        ensure!(root.get("tie_word_embeddings").is_none_or(|b| b.as_bool() == Some(false))
            && v.get("tie_word_embeddings").is_none_or(|b| b.as_bool() == Some(false)),
            "Hy4-flash requires tie_word_embeddings=false");

        let layers = int(v, "num_hidden_layers")?;
        let vocab_size = int(v, "vocab_size")?;
        let hidden = int(v, "hidden_size")?;
        let intermediate = int(v, "intermediate_size")?;

        let raw_layer_types = v["layer_types"].as_array()
            .context("Hy4-flash config lacks layer_types")?;
        ensure!(raw_layer_types.len() == layers,
            "Hy4-flash layer_types length {} differs from num_hidden_layers {layers}", raw_layer_types.len());
        let mut sparse_layer = Vec::with_capacity(layers);
        for entry in raw_layer_types {
            match entry.as_str() {
                Some(SPARSE_ATTENTION) => sparse_layer.push(true),
                Some(SLIDING_ATTENTION) => sparse_layer.push(false),
                _ => bail!("invalid Hy4-flash layer_types entry {entry}"),
            }
        }

        let raw_mlp_types = v["mlp_layer_types"].as_array()
            .context("Hy4-flash config lacks mlp_layer_types")?;
        ensure!(raw_mlp_types.len() == layers,
            "Hy4-flash mlp_layer_types length {} differs from num_hidden_layers {layers}", raw_mlp_types.len());
        let mut dense_layer = Vec::with_capacity(layers);
        for entry in raw_mlp_types {
            match entry.as_str() {
                Some("dense") => dense_layer.push(true),
                Some("sparse") => dense_layer.push(false),
                _ => bail!("invalid Hy4-flash mlp_layer_types entry {entry}"),
            }
        }

        let rope = v.get("rope_parameters").and_then(Value::as_object)
            .context("Hy4-flash config lacks rope_parameters")?;
        ensure!(rope.len() == 2 && rope.contains_key(SPARSE_ATTENTION) && rope.contains_key(SLIDING_ATTENTION),
            "Hy4-flash rope_parameters must name exactly {SPARSE_ATTENTION} and {SLIDING_ATTENTION}");
        let parse_rope = |kind: &str| -> Result<f64> {
            let entry = &rope[kind];
            ensure!(entry["rope_type"] == "default" && entry["mrope_interleaved"] == true,
                "unsupported Hy4-flash rope for {kind}");
            float(entry, "rope_theta")
        };
        let sparse_rope_theta = parse_rope(SPARSE_ATTENTION)?;
        let swa_rope_theta = parse_rope(SLIDING_ATTENTION)?;

        let q_lora_rank = int(v, "q_lora_rank")?;
        let kv_lora_rank = int(v, "kv_lora_rank")?;
        let qk_nope_dim = int(v, "qk_nope_head_dim")?;
        let qk_rope_dim = int(v, "qk_rope_head_dim")?;
        let v_head_dim = int(v, "v_head_dim")?;
        let sparse_heads = int(v, "num_attention_heads")?;
        let swa_heads = int(v, "swa_num_attention_heads")?;
        let swa_qk_nope_dim = int(v, "swa_qk_nope_head_dim")?;
        let swa_qk_rope_dim = int(v, "swa_qk_rope_head_dim")?;
        let swa_v_head_dim = int(v, "swa_v_head_dim")?;
        let sliding_window = int(v, "sliding_window")?;

        let indexer_heads = int(v, "compressed_indexer_num_attention_heads")?;
        let indexer_head_dim = int(v, "compressed_indexer_head_dim")?;
        let indexer_compress_ratio = int(v, "compressed_indexer_compress_ratio")?;
        let indexer_group_topk = int(v, "compressed_indexer_group_topk")?;
        let indexer_candidate_topk = int(v, "compressed_indexer_candidate_topk")?;
        ensure!(indexer_group_topk * indexer_compress_ratio == indexer_candidate_topk,
            "Hy4-flash indexer group_topk * compress_ratio must equal candidate_topk");

        let experts = int(v, "n_routed_experts")?;
        let topk = int(v, "num_experts_per_tok")?;
        ensure!(topk <= experts, "Hy4-flash topk {topk} exceeds experts {experts}");
        let moe_intermediate = int(v, "moe_intermediate_size")?;
        let routed_expert_hidden = int(v, "routed_expert_hidden_size")?;
        let shared_experts = int(v, "n_shared_experts")?;
        let n_group = int(v, "n_group")?;
        let topk_group = int(v, "topk_group")?;
        ensure!(n_group <= experts && experts % n_group == 0 && topk_group <= n_group,
            "invalid Hy4-flash router groups ({n_group}, {topk_group}) for {experts} experts");
        let norm_topk_prob = v["norm_topk_prob"].as_bool()
            .context("Hy4-flash config lacks norm_topk_prob")?;

        // Engram: config `engram_layer_ids` are 1-based decoder layer ids.
        let engram_max_ngram_size = int(v, "engram_max_ngram_size")?;
        ensure!(engram_max_ngram_size >= 2, "Hy4-flash engram_max_ngram_size must be >= 2");
        let engram_n_embed_per_ngram = int(v, "engram_n_embed_per_ngram")?;
        let engram_n_head_per_ngram = int(v, "engram_n_head_per_ngram")?;
        ensure!(engram_n_embed_per_ngram % engram_n_head_per_ngram == 0,
            "Hy4-flash engram_n_embed_per_ngram must divide by engram_n_head_per_ngram");
        let expected_heads = (engram_max_ngram_size - 1) * engram_n_head_per_ngram;
        let raw_engram_ids = v["engram_layer_ids"].as_array()
            .context("Hy4-flash config lacks engram_layer_ids")?;
        let vocab_map = v["engram_layer_head_vocab_sizes"].as_object()
            .context("Hy4-flash config lacks engram_layer_head_vocab_sizes")?;
        let mut engram_layers = Vec::with_capacity(raw_engram_ids.len());
        let mut engram_vocab_sizes = BTreeMap::new();
        for entry in raw_engram_ids {
            let one_based = entry.as_u64()
                .filter(|&id| id >= 1 && (id as usize) <= layers)
                .with_context(|| format!("invalid Hy4-flash 1-based engram_layer_id {entry}"))? as usize;
            let zero_based = one_based - 1;
            ensure!(!engram_layers.contains(&zero_based), "duplicate Hy4-flash engram layer {zero_based}");
            let key = one_based.to_string();
            let orders = vocab_map.get(&key).and_then(Value::as_array)
                .with_context(|| format!("Hy4-flash engram_layer_head_vocab_sizes lacks {key}"))?;
            ensure!(orders.len() == engram_max_ngram_size - 1,
                "Hy4-flash engram vocab orders for layer {zero_based} mismatch");
            let mut flat = Vec::with_capacity(expected_heads);
            for order in orders {
                let heads = order.as_array().context("Hy4-flash engram order sizes must be array")?;
                ensure!(heads.len() == engram_n_head_per_ngram,
                    "Hy4-flash engram head count mismatch for layer {zero_based}");
                for size in heads {
                    let s = size.as_u64().filter(|&s| s > 0)
                        .context("Hy4-flash engram head vocab size must be positive")?;
                    flat.push(s);
                }
            }
            engram_layers.push(zero_based);
            engram_vocab_sizes.insert(zero_based, flat);
        }
        engram_layers.sort_unstable();

        let engram_pad_id = token_id(v, "engram_pad_id", vocab_size)?;
        let engram_seed = v["engram_seed"].as_u64()
            .context("Hy4-flash config lacks engram_seed")?;
        let engram_fp8_embedding = v["engram_fp8_embedding"].as_bool()
            .context("Hy4-flash config lacks engram_fp8_embedding")?;
        ensure!(engram_fp8_embedding,
            "Hy4-flash requires text_config.engram_fp8_embedding=true (both engram_fp8 and BF16 fallback apply FP8 QDQ)");
        let engram_lookup_table_size = v.get("engram_lookup_table_size")
            .and_then(Value::as_u64)
            .map(|n| n as usize)
            .unwrap_or(120_818);
        ensure!(engram_lookup_table_size > 0 && engram_lookup_table_size <= vocab_size,
            "invalid Hy4-flash engram_lookup_table_size {engram_lookup_table_size}");

        let image_token_id = token_id(root, "image_token_id", vocab_size)?;
        let video_token_id = token_id(root, "video_token_id", vocab_size)?;
        let eos = token_ids(
            root.get("eos_token_id").or_else(|| v.get("eos_token_id"))
                .context("Hy4-flash config lacks eos_token_id")?,
            vocab_size, "eos_token_id",
        )?;

        Ok(Self {
            vocab_size,
            hidden,
            layers,
            intermediate,
            max_position: int(v, "max_position_embeddings")?,
            rms_norm_eps: float(v, "rms_norm_eps")?,
            hc_mult: int(v, "hc_mult")?,
            hc_eps: float(v, "hc_eps")?,
            hc_magnitude: float(v, "hc_magnitude")?,
            q_lora_rank,
            kv_lora_rank,
            qk_nope_dim,
            qk_rope_dim,
            v_head_dim,
            sparse_heads,
            swa_heads,
            swa_qk_nope_dim,
            swa_qk_rope_dim,
            swa_v_head_dim,
            sliding_window,
            sparse_rope_theta,
            swa_rope_theta,
            sparse_layer,
            indexer_heads,
            indexer_head_dim,
            indexer_compress_ratio,
            indexer_group_topk,
            indexer_candidate_topk,
            dense_layer,
            experts,
            topk,
            moe_intermediate,
            routed_expert_hidden,
            shared_experts,
            n_group,
            topk_group,
            norm_topk_prob,
            routed_scaling_factor: float(v, "routed_scaling_factor")?,
            swiglu_limit: float(v, "swiglu_limit")?,
            engram_layers,
            engram_vocab_sizes,
            engram_max_ngram_size,
            engram_n_embed_per_ngram,
            engram_n_head_per_ngram,
            engram_pad_id,
            engram_seed,
            engram_fp8_embedding,
            engram_lookup_table_size,
            image_token_id,
            video_token_id,
            eos,
        })
    }

    pub fn attention(&self, layer: usize) -> Hy4LayerAttention {
        if self.sparse_layer[layer] {
            Hy4LayerAttention {
                sparse: true,
                heads: self.sparse_heads,
                qk_nope_dim: self.qk_nope_dim,
                qk_rope_dim: self.qk_rope_dim,
                v_head_dim: self.v_head_dim,
            }
        } else {
            Hy4LayerAttention {
                sparse: false,
                heads: self.swa_heads,
                qk_nope_dim: self.swa_qk_nope_dim,
                qk_rope_dim: self.swa_qk_rope_dim,
                v_head_dim: self.swa_v_head_dim,
            }
        }
    }

    pub fn sparse_layers(&self) -> Vec<usize> {
        (0..self.layers).filter(|&i| self.sparse_layer[i]).collect()
    }

    pub fn swa_layers(&self) -> Vec<usize> {
        (0..self.layers).filter(|&i| !self.sparse_layer[i]).collect()
    }

    pub fn routed_layers(&self) -> Vec<usize> {
        (0..self.layers).filter(|&i| !self.dense_layer[i]).collect()
    }

    pub fn engram_head_width(&self) -> usize {
        self.engram_n_embed_per_ngram / self.engram_n_head_per_ngram
    }

    pub fn engram_total_heads(&self) -> usize {
        (self.engram_max_ngram_size - 1) * self.engram_n_head_per_ngram
    }

    pub fn engram_concat_width(&self) -> usize {
        self.engram_head_width() * self.engram_total_heads()
    }

    pub fn engram_rows(&self, layer: usize) -> Option<u64> {
        self.engram_vocab_sizes.get(&layer).map(|sizes| sizes.iter().sum())
    }

    pub fn masked_engram_tokens(&self) -> [u32; 3] {
        [self.image_token_id, self.video_token_id, self.engram_pad_id]
    }

    /// SwiGLU clamp (`swiglu_limit`) applies to routed experts only; dense layer 0
    /// and shared experts are unclamped (matches `python/reference/families/hy4_flash/model.py`).
    pub fn activation(&self, layer: usize, shared: bool) -> cuteafd_core::FfnActivation {
        let limit = (!self.dense_layer[layer] && !shared).then_some(self.swiglu_limit as f32);
        cuteafd_core::FfnActivation::SiluBf16Product { limit }
    }

    pub fn release_geometry(&self) -> Result<()> {
        let shape = (self.hidden, self.layers, self.hc_mult, self.q_lora_rank, self.kv_lora_rank,
                     self.qk_nope_dim, self.qk_rope_dim, self.v_head_dim, self.sparse_heads,
                     self.swa_heads, self.sliding_window);
        ensure!(shape == (3072, 42, 4, 1024, 512, 192, 64, 256, 32, 64, 512),
            "Hy4-flash attention/backbone geometry {shape:?} differs from the supported Hy4-flash release");
        let moe = (self.experts, self.topk, self.moe_intermediate, self.routed_expert_hidden,
                   self.shared_experts, self.intermediate, self.vocab_size);
        ensure!(moe == (512, 16, 1536, 1536, 1, 13824, 120_832),
            "Hy4-flash MoE/head geometry {moe:?} differs from the supported Hy4-flash release");
        ensure!(self.engram_layers == [1, 11, 21, 31] && self.engram_head_width() == 96
            && self.engram_total_heads() == 16,
            "Hy4-flash Engram geometry differs from the supported Hy4-flash release");
        Ok(())
    }
}

impl Hy4DflyConfig {
    /// The block contains the uncached bonus token followed by mask slots.
    pub fn block_rows(&self) -> usize {
        self.num_speculative_tokens + 1
    }

    pub fn validate_serving(&self, root: &Value, text: &Hy4FlashConfig) -> Result<()> {
        let unique = self.target_kv_layers.iter().copied().collect::<std::collections::BTreeSet<_>>();
        ensure!(self.layers == 5 && unique.len() == self.layers
            && self.target_kv_layers == self.target_index_layers
            && unique.iter().all(|&layer| text.sparse_layer.get(layer) == Some(&true)),
            "Hy4 DFly2 requires five distinct paired sparse KV and indexer taps");
        ensure!(self.target_index_topk == text.indexer_candidate_topk
            && self.enable_hidden_correction && self.mtp_fusion == "slot0"
            && root.get("dfly_context_local_window").is_none_or(|w| w.is_null() || w.as_u64() == Some(0)),
            "Hy4 DFly2 requires shared full recall, slot0 fusion and hidden correction");
        ensure!((1..=5).contains(&self.num_speculative_tokens),
            "Hy4 DFly2 supports 1 to 5 speculative tokens in config, found {}", self.num_speculative_tokens);
        Ok(())
    }

    pub fn from_hf(root: &Value, text: &Hy4FlashConfig) -> Result<Option<Self>> {
        let Some(layers) = root.get("dfly_num_hidden_layers").and_then(Value::as_u64) else {
            return Ok(None);
        };
        let layers = usize::try_from(layers)?;
        ensure!(layers > 0 && layers <= 16, "invalid Hy4-flash dfly_num_hidden_layers {layers}");
        let parse_layers = |key: &str| -> Result<Vec<usize>> {
            let arr = root[key].as_array().with_context(|| format!("Hy4-flash lacks {key}"))?;
            ensure!(arr.len() == layers, "Hy4-flash {key} length mismatch");
            arr.iter().map(|entry| {
                let id = entry.as_u64().filter(|&id| (id as usize) < text.layers)
                    .with_context(|| format!("invalid Hy4-flash {key} entry {entry}"))?;
                Ok(id as usize)
            }).collect()
        };
        let target_kv_layers = parse_layers("dfly_target_kv_layers")?;
        let target_index_layers = parse_layers("dfly_target_index_layer")?;
        let target_index_topk = int(root, "dfly_target_index_topk")?;
        let mask_token_id = token_id(root, "dfly_mask_token_id", text.vocab_size)?;
        let enable_hidden_correction = root["dfly_enable_hidden_correction"].as_bool()
            .context("Hy4-flash lacks dfly_enable_hidden_correction")?;
        let mtp_fusion = root["dfly_mtp_fusion"].as_str()
            .context("Hy4-flash lacks dfly_mtp_fusion")?.to_owned();
        let num_speculative_tokens = root.get("speculative_config")
            .and_then(|s| s.get("num_speculative_tokens"))
            .and_then(Value::as_u64)
            .map(|n| n as usize)
            .unwrap_or(layers);
        Ok(Some(Self {
            layers,
            target_kv_layers,
            target_index_layers,
            target_index_topk,
            mask_token_id,
            enable_hidden_correction,
            mtp_fusion,
            num_speculative_tokens,
        }))
    }
}

impl Hy4VisionConfig {
    pub fn from_hf(root: &Value) -> Result<Option<Self>> {
        let Some(v) = root.get("vision_config").filter(|v| v.is_object()) else {
            return Ok(None);
        };
        Ok(Some(Self {
            hidden: int(v, "hidden_size")?,
            intermediate: int(v, "intermediate_size")?,
            heads: int(v, "num_attention_heads")?,
            layers: int(v, "num_hidden_layers")?,
            out_hidden: int(v, "out_hidden_size")?,
            patch_size: int(v, "patch_size")?,
            spatial_merge_size: int(v, "spatial_merge_size")?,
            rms_norm_eps: float(v, "rms_norm_eps")?,
        }))
    }
}
