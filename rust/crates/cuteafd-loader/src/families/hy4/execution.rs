//! Named physical cache and per-unit workspace layouts for Hy4 (`hy_v4_vl`).
//!
//! Unlike `hy4_flash`, Hy4 has no SWA rings, no compressed-indexer pending
//! tokens, and no Engram tables: every layer is paged DSA MLA and IndexCache
//! stores per-token 132-byte E4M3+F32 keys on the 21 full indexer layers only
//! (`8448 B/page`, matching `glm_index_topk`). Exact prefix restore is
//! pages-only (zero per-slot device mark bytes).
use std::collections::BTreeMap;

use anyhow::{ensure, Context, Result};

use crate::families::hy4_common::PAGE_ROWS;

use super::{Hy4Config, Hy4KvCache};

/// Per-token DSA index key record bytes: 128 E4M3 dims + 4B F32 scale
/// (matching `glm_index_topk` `index_page_bytes = 64 * 132 = 8448`).
pub const INDEX_RECORD_BYTES: usize = 132;
pub const INDEX_PAGE_BYTES: usize = PAGE_ROWS * INDEX_RECORD_BYTES;

/// Paged cache page strides for one Hy4 coordinator GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheLayout {
    /// Per-page bytes of one DSA MLA layer's KV records (`64 * record_bytes`).
    pub kv_page_bytes: usize,
    /// Per-page bytes of one full-indexer layer's index K cache (`64 * 132 = 8448 B`).
    pub index_page_bytes: usize,
    /// Number of MLA layers owned by this GPU.
    pub owned_kv_layers: usize,
    /// Number of full IndexCache layers owned by this GPU (shared layers own 0 index pages).
    pub owned_full_index_layers: usize,
}

impl CacheLayout {
    pub fn new(
        kv_format: Hy4KvCache,
        owned_kv_layers: usize,
        owned_full_index_layers: usize,
    ) -> Result<Self> {
        let kv_page_bytes = PAGE_ROWS
            .checked_mul(kv_format.record_bytes())
            .context("Hy4 KV page overflow")?;
        Ok(Self {
            kv_page_bytes,
            index_page_bytes: INDEX_PAGE_BYTES,
            owned_kv_layers,
            owned_full_index_layers,
        })
    }

    /// Total bytes per physical page owned by this GPU across all its MLA and full-indexer layers.
    pub fn persistent_unit_bytes(&self) -> Option<u64> {
        let kv = (self.owned_kv_layers as u64).checked_mul(self.kv_page_bytes as u64)?;
        let idx = (self.owned_full_index_layers as u64).checked_mul(self.index_page_bytes as u64)?;
        kv.checked_add(idx)
    }
}

/// Named scratch buffers owned by one in-flight execution unit (one decode lane
/// or one prefill wave slot) on a coordinator GPU.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnitLayout {
    pub buffers: BTreeMap<String, usize>,
}

impl UnitLayout {
    pub fn new(cfg: &Hy4Config, rows: usize, head_parts: usize) -> Result<Self> {
        ensure!(rows > 0 && matches!(head_parts, 1 | 2), "invalid Hy4 UnitLayout shape");
        let h = cfg.hidden;
        let hc = cfg.hc_mult;
        let local_heads = cfg.heads / head_parts;
        let mut buffers = BTreeMap::new();

        // 4-stream iHC residual [rows, 4, H] BF16 + pre/post FP32 mixing coefficients.
        buffers.insert("ihc_residual".into(), rows * hc * h * 2);
        buffers.insert("ihc_post".into(), rows * hc * 4);
        buffers.insert("layer_in".into(), rows * h * 2);
        buffers.insert("layer_out".into(), rows * h * 2);

        // Absorbed MLA scratch (no SWA ring scratch):
        // q_a [rows, 2048], kv_a [rows, 576], absorbed_q [rows, local_heads, 576],
        // attn_latent [rows, local_heads, 512], attn_lse [rows, local_heads] F32,
        // attn_gate/attn_uv [rows, local_heads, 256] BF16.
        buffers.insert("q_a".into(), rows * cfg.q_lora_rank * 2);
        buffers.insert("kv_a".into(), rows * (cfg.kv_lora_rank + cfg.qk_rope_dim) * 2);
        buffers.insert("absorbed_q".into(), rows * local_heads * (cfg.kv_lora_rank + cfg.qk_rope_dim) * 2);
        buffers.insert("attn_latent".into(), rows * local_heads * cfg.kv_lora_rank * 2);
        buffers.insert("attn_lse".into(), rows * local_heads * 4);
        buffers.insert("attn_gate".into(), rows * local_heads * cfg.v_head_dim * 2);
        buffers.insert("attn_uv".into(), rows * local_heads * cfg.v_head_dim * 2);
        // Split-K partials for sparse MLA (up to 16 splits x local_heads x 512 BF16 + LSE).
        buffers.insert(
            "mla_split_partials".into(),
            rows.min(64) * 16 * local_heads * (cfg.kv_lora_rank * 2 + 4),
        );

        // 32-head per-token DSA indexer + IndexCache reuse buffer:
        // full layers produce q_fp8 [rows, 32, 128] E4M3 + weights [rows, 32] F32
        // and write indexer_candidates [rows, 2048] I32; the next 3 shared layers
        // read indexer_candidates directly without re-running the indexer.
        buffers.insert("indexer_q_fp8".into(), rows * cfg.indexer_heads * cfg.indexer_head_dim);
        buffers.insert("indexer_weights".into(), rows * cfg.indexer_heads * 4);
        buffers.insert("indexer_candidates".into(), rows * cfg.indexer_topk * 4);

        // Dense layer-0 / shared expert / direct MoE router scratch (no LatentMoE buffers).
        let dense_cols = cfg.intermediate / head_parts;
        buffers.insert("ffn_gate_up".into(), rows * dense_cols * 2 * 2);
        buffers.insert("router_scores".into(), rows * cfg.experts * 4);
        buffers.insert("router_ids".into(), rows * cfg.topk * 4);
        buffers.insert("router_weights".into(), rows * cfg.topk * 4);

        Ok(Self { buffers })
    }

    pub fn total_bytes(&self) -> usize {
        self.buffers.values().sum()
    }
}
