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
        self.context.iter().try_fold(16usize, |sum, policy| {
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
                    assert_eq!(c.window_mark_bytes(format, rank, ranks, 0)?, (16+4*1024*record).div_ceil(256)*256);
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
