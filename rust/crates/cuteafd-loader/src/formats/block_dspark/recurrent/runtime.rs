//! Exact mixed-context cache inventory for recurrent-hidden block drafting.
use super::{ContextPolicy, RecurrentBlockConfig};
use crate::{families::qwen4::Qwen4KvCache, formats::dflash::DraftCacheLayout};
use anyhow::{ensure, Context, Result};

impl RecurrentBlockConfig {
    /// Full-layer pages follow the admitted target token pool, not concurrency
    /// times max context. Window layers retain their own exact per-slot rings.
    pub fn cache_layout(&self, slots: usize, context: usize, target_tokens: usize,
        format: Qwen4KvCache, rank: usize, ranks: usize, lead: usize) -> Result<DraftCacheLayout> {
        let local = self.rank(rank, ranks, lead)?;
        ensure!(slots > 0 && super::super::supports_sequences(slots) && context > 0
            && context <= self.transformer.max_position && target_tokens > 0,
            "recurrent draft cache capacity exceeds checkpoint");
        let pages = target_tokens.div_ceil(64);
        let full_rows = pages.checked_mul(64).context("recurrent draft page rows overflow")?;
        ensure!(full_rows <= i32::MAX as usize, "recurrent draft page indices exceed native extent");
        let plane = format.plane_record_bytes(local.kv_heads, self.transformer.head_dim);
        let record_bytes = plane.checked_mul(2).context("recurrent draft record bytes overflow")?;
        let mut full_bytes = 0usize;
        let mut window_bytes = 0usize;
        let mut full_layers = 0;
        let layer_bytes = self.context.iter().map(|policy| {
            let rows = match policy {
                ContextPolicy::Full => full_rows,
                ContextPolicy::BlockAnchoredWindow(window) => slots.checked_mul(*window)
                    .context("recurrent draft ring rows overflow")?,
            };
            let bytes = rows.checked_mul(record_bytes).context("recurrent draft layer bytes overflow")?;
            match policy {
                ContextPolicy::Full => { full_layers += 1;
                    full_bytes = full_bytes.checked_add(bytes).context("recurrent full cache bytes overflow")?; }
                ContextPolicy::BlockAnchoredWindow(_) => {
                    window_bytes = window_bytes.checked_add(bytes).context("recurrent window cache bytes overflow")?; }
            }
            Ok(bytes)
        }).collect::<Result<Vec<_>>>()?;
        let page_table_bytes = if full_layers > 0 {
            slots.checked_mul(context.div_ceil(64)).and_then(|n| n.checked_mul(4))
                .context("recurrent page table bytes overflow")?
        } else { 0 };
        Ok(DraftCacheLayout { target_tokens, page_rows: 64, pages, record_bytes, full_layers,
            window_bytes: window_bytes as u64, full_bytes: full_bytes as u64,
            page_table_bytes: page_table_bytes as u64, layer_bytes })
    }

    /// Only bounded local planes are snapshotted; full-context pages are retained
    /// through the shared prefix attachment/page pool, never copied into a ring.
    pub fn window_mark_bytes(&self, format: Qwen4KvCache, rank: usize, ranks: usize, lead: usize) -> Result<usize> {
        let local = self.rank(rank, ranks, lead)?;
        let record = format.plane_record_bytes(local.kv_heads, self.transformer.head_dim)
            .checked_mul(2).context("recurrent mark record overflow")?;
        let feature=self.transformer.hidden.checked_mul(2).context("recurrent mark feature overflow")?;
        self.context.iter().try_fold(16usize.checked_add(feature).context("recurrent mark metadata overflow")?, |sum, policy| {
            let bytes = match policy { ContextPolicy::Full => 0,
                ContextPolicy::BlockAnchoredWindow(window) => window.checked_mul(record).context("recurrent mark plane overflow")? };
            sum.checked_add(bytes).context("recurrent mark bytes overflow")
        })?.checked_add(255).map(|n| n/256*256).context("recurrent mark alignment overflow")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn release() -> RecurrentBlockConfig {
        RecurrentBlockConfig::from_hf(&serde_json::from_str(include_str!(
            "../../../families/inkling2/fixtures/config.json")).unwrap()).unwrap()
    }
    #[test]
    fn recurrent_cache_preserves_one_full_layer_and_four_bounded_rings() -> Result<()> {
        let c = release();
        for format in [Qwen4KvCache::Fp8, Qwen4KvCache::Bf16] {
            for ranks in [1, 2] {
                for rank in 0..ranks {
                    let cache = c.cache_layout(16, 32768, 131072, format, rank, ranks, 0)?;
                    let record = 2*format.plane_record_bytes(8/ranks, 128);
                    assert_eq!(cache.record_bytes, record);
                    assert_eq!(cache.full_layers, 1);
                    assert_eq!(cache.layer_bytes[2], 131072*record);
                    assert_eq!(cache.full_bytes, (131072*record) as u64);
                    assert_eq!(cache.window_bytes, (4*16*1024*record) as u64);
                    assert_eq!(cache.page_table_bytes, 16*512*4);
                    assert_eq!(c.window_mark_bytes(format, rank, ranks, 0)?, (16+4608*2+4*1024*record).div_ceil(256)*256);
                    let wider = c.cache_layout(32, 32768, 131072, format, rank, ranks, 0)?;
                    assert_eq!(cache.full_bytes, wider.full_bytes);
                    assert_eq!(2*cache.window_bytes, wider.window_bytes);
                }
            }
        }
        assert!(c.cache_layout(0, 1, 1, Qwen4KvCache::Fp8, 0, 1, 0).is_err());
        assert!(c.cache_layout(1, usize::MAX, 1, Qwen4KvCache::Fp8, 0, 1, 0).is_err());
        assert!(c.cache_layout(1, 1, usize::MAX, Qwen4KvCache::Fp8, 0, 1, 0).is_err());
        Ok(())
    }
}

/// Physical rank-local mutable owners. Projections and the target adapter use
/// the same inventory; full context cache bytes are a pool-dependent curve.
#[derive(Debug, Clone)]
pub struct RecurrentRuntimeLayout {
    pub rank: super::RecurrentRank,
    pub slots: usize,
    pub sequences: usize,
    pub context_rows: usize,
    pub buffers: Vec<(String, usize)>,
    pub kv_cache: Qwen4KvCache,
    pub cache: DraftCacheLayout,
    pub window_mark_bytes: usize,
}

impl RecurrentRuntimeLayout {
    pub fn new(config: &RecurrentBlockConfig, slots: usize, sequences: usize,
        context_rows: usize, context: usize, target_tokens: usize, sms: usize,
        kv_cache: Qwen4KvCache, rank: usize, ranks: usize, lead: usize) -> Result<Self> {
        ensure!(sequences > 0 && sequences <= slots && context_rows > 0 && context_rows <= 4096 && sms > 0,
            "recurrent draft work capacity outside native exports");
        let local = config.rank(rank, ranks, lead)?;
        let cache = config.cache_layout(slots, context, target_tokens, kv_cache, rank, ranks, lead)?;
        let c = &config.transformer;
        let (h, r) = (c.hidden, c.markov_rank);
        let rows = sequences.checked_mul(c.block).context("draft rows overflow")?;
        let q = local.heads.checked_mul(c.head_dim).context("draft q overflow")?;
        let kv = local.kv_heads.checked_mul(c.head_dim).context("draft kv overflow")?;
        let mul = |values: &[usize]| values.iter().try_fold(1usize, |n, value| n.checked_mul(*value))
            .context("recurrent runtime owner extent overflow");
        let mut buffers = Vec::new();
        let mut add = |name: String, bytes: usize| {
            buffers.push((name, crate::serving_capacity::WeightArenaLayout::allocation_bytes(bytes)));
        };
        for (layer, bytes) in cache.layer_bytes.iter().enumerate() {
            for plane in ["k", "v"] { add(format!("cache.layer.{layer}.{plane}"), bytes/2); }
        }
        if cache.page_table_bytes > 0 { add("cache.tables".into(), usize::try_from(cache.page_table_bytes)?); }
        add("cache.last_features".into(), mul(&[slots, h, 2])?);
        for (name, width, element) in [("taps", mul(&[config.taps.len(), h])?, 2),
            ("mixed", h, 2), ("kv", 2*kv, 2), ("key", kv, 2), ("value", kv, 2),
            ("positions", 1, 8), ("destinations", 1, 8)] {
            add(format!("context.{name}"), mul(&[context_rows, width, element])?);
        }
        for (name, width, element) in [("residual", h, 2), ("residual_next", h, 2),
            ("normalized", h, 2), ("qkv", q+2*kv, 2), ("query", q, 2), ("key", kv, 2),
            ("value", kv, 2), ("attention", q, 2), ("partial", h, 2),
            ("gate_up", 2*local.intermediate, 2), ("activated", local.intermediate, 2),
            ("positions", 1, 8)] {
            add(format!("step.{name}"), mul(&[rows, width, element])?);
        }
        add("projection.bf16".into(), mul(&[context_rows.max(rows), local.intermediate.max(q).max(h), 2])?);
        add("step.anchors".into(), mul(&[sequences, h, 2])?);
        add("step.ids".into(), mul(&[sequences, 4])?);
        add("step.tables".into(), mul(&[sequences, 3, 4])?);
        let attention_keys = context.checked_add(c.block).context("draft attention keys overflow")?;
        add("step.attention_workspace".into(), usize::try_from(crate::serving_capacity::draft_attention_workspace_bytes(
            sequences as u64, local.kv_heads as u64,
            (local.heads/local.kv_heads*c.block).div_ceil(32) as u64,
            attention_keys as u64, c.head_dim as u64, sms as u64)?)?);
        // Immutable rotary frequencies are tiny, unlike a max-position cos/sin table.
        add("rotary.frequencies".into(), c.head_dim/2*4);
        if rank == lead {
            for (name, width, element) in [("hidden", h, 2), ("f_rnn", h, 2), ("state", r, 2),
                ("next_state", r, 2), ("latent", r, 2), ("previous_embedding", r, 2),
                ("inputs", config.recurrent_input_width()?, 2), ("projection", 3*r, 4),
                ("raw", 3*r, 4), ("base", config.padded_vocab, 2),
                ("correction_f32", config.padded_vocab, 4), ("correction", config.padded_vocab, 2),
                ("logits_bf16", config.valid_vocab, 2), ("logits", config.valid_vocab, 4),
                ("tokens", c.block-1, 4), ("previous", 1, 4), ("status", 1, 4)] {
                add(format!("head.{name}"), mul(&[sequences, width, element])?);
            }
        }
        Ok(Self { rank: local, slots, sequences, context_rows, buffers, kv_cache, cache,
            window_mark_bytes: config.window_mark_bytes(kv_cache, rank, ranks, lead)? })
    }
}
