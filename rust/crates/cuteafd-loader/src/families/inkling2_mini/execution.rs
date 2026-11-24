//! Native TP1 execution geometry, 48-slot cache bank, and residual schedule for Inkling 2 Mini.
use super::{Inkling2MiniConfig, InklingKvCache};
use crate::families::inkling2::{conv::ConvLayout, execution::{ResidualSite, SequenceLayout}};
use anyhow::{ensure, Context, Result};
use std::collections::BTreeMap;

/// Prefill attention tile size for Inkling 2 Mini (24 Q / 8 KV, group=3 -> 21 tokens/tile).
pub const PREFILL_TILE_ROWS: usize = 21;
/// Decode/verify attention tile size (1 row/tile).
pub const DECODE_TILE_ROWS: usize = 1;
/// Decode/verify bounded batch width (C16 or C8 x 8-token MTP block).
pub const DECODE_ROWS: usize = 64;

/// Named per-unit GPU scratch owners for Inkling 2 Mini TP1.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MiniUnitLayout {
    pub rows: usize,
    pub blocks: usize,
    pub kv_format: InklingKvCache,
    pub buffers: BTreeMap<String, usize>,
}

impl MiniUnitLayout {
    pub fn with_format(cfg: &Inkling2MiniConfig, rows: usize, context: usize, kv_format: InklingKvCache) -> Result<Self> {
        let t = &cfg.text;
        let lp = &cfg.loop_cfg;
        ensure!(
            (1..=4096).contains(&rows) && rows <= context && context <= t.max_position,
            "Inkling 2 Mini unit rows/context outside checkpoint/export"
        );
        cfg.release_geometry()?;
        let blocks = t.layers.div_ceil(t.res_block) + 1; // 24 / 6 + 1 = 5
        let mut buffers = BTreeMap::new();
        let mut add = |name: &str, width: usize, bytes: usize| -> Result<()> {
            let extent = rows
                .checked_mul(width)
                .and_then(|n| n.checked_mul(bytes))
                .context("Inkling 2 Mini unit extent overflow")?;
            ensure!(buffers.insert(name.into(), extent).is_none(), "duplicate Mini unit owner {name}");
            Ok(())
        };

        // F32 hidden planes (exact BF16 values widened at layer boundaries, F32 inside residual/conv).
        for name in ["partial", "attention", "mlp", "branch", "mixed", "pre_core_partial"] {
            add(name, t.hidden, 4)?;
        }
        // Multi-block residual & loop cross-pass planes.
        add("residual_arena", blocks * t.hidden, 4)?;
        add("pre_core_blocks", t.hidden, 4)?; // 1 completed block before core entry (embed_normed)
        add("prev_pass_arena", 2 * t.hidden, 4)?; // [h_after_L11, h_after_L17] from previous pass
        add("global_mixes", 3 * t.hidden, 4)?;
        add("cross_pass_msg", 4 * t.hidden, 4)?;

        // Looped cross-pass controller workspaces (slots=1 for MLP, slots=3 for attn/k/v).
        add("ctrl_paired", 3 * 2 * t.hidden, 2)?; // [3, rows, 4608] BF16
        add("ctrl_rank", 3 * lp.gate_rank, 2)?; // [3, rows, 64] BF16 (exact f1940f40eb down_proj output)
        add("ctrl_fg", 3 * 2 * t.hidden, 4)?; // [3, rows, 4608] F32 (exact f1940f40eb forget/gain)
        add("ctrl_out", 3 * t.hidden, 2)?; // [3, rows, 2304] BF16 ([xq, xk, xv] after blend+norm)

        // Attention & dense projection planes (f1940f40eb BF16 dense projections, F32 conv/router/sinks).
        let q_width = t.heads * t.head_dim; // 3072
        let kv_width = t.kv_heads * t.head_dim; // 1024
        let rel_width = t.heads * t.d_rel; // 384
        for (name, width, bytes) in [
            ("projection_input", t.hidden, 2),
            ("branch_bf16", t.hidden, 2),
            ("q", q_width, 2),
            ("r", rel_width, 2),
            ("k_bf16", kv_width, 2),
            ("v_bf16", kv_width, 2),
            ("k_f32", kv_width, 4),
            ("v_f32", kv_width, 4),
            ("conv_k", kv_width, 4),
            ("conv_v", kv_width, 4),
            ("q_clipped", q_width, 2),
            ("r_clipped", rel_width, 2),
            ("k_clipped", kv_width, 2),
            ("v_norm", kv_width, 2),
            ("relative_input", t.heads * 64, 2),
            ("relative", t.heads * t.rel_extent, 2),
            ("gate", q_width, 2),
            ("sinks", t.heads, 4),
            ("attended", q_width, 2),
            ("gated", q_width, 2),
            ("kv_step", 2 * kv_width, 2),
            ("k_dynamic", t.kv_heads * t.sconv_kernel, 4),
            ("v_dynamic", t.kv_heads * t.sconv_kernel, 4),
        ] {
            add(name, width, bytes)?;
        }

        // MoE routing, MXFP8 wire quantizer (layers 20..24), and metadata planes.
        for (name, width, bytes) in [
            ("route_logits", t.experts, 4),
            ("route_ids", t.topk, 4),
            ("route_weights", t.topk, 4),
            ("expert_wire", t.hidden + t.hidden / 32, 1), // 2376 B/row
            ("expert_output", t.hidden, 2),
            ("table_indices", 1, 4),
            ("tile_rows", 1, 4),
            ("positions", 1, 8),
            ("first", 1, 4),
            ("conv_slots", 1, 8),
            ("ring_slots", 1, 8),
            ("cache_slots", 1, 8),
        ] {
            add(name, width, bytes)?;
        }

        buffers.insert("attention_scratch".into(), 64 * 32 * t.heads * (128 * 4 + 4));
        buffers.insert("ring".into(), 1088 * 4096);
        buffers.insert(
            "global_cache".into(),
            context
                .div_ceil(64)
                .checked_mul(64 * kv_format.record_bytes(t.kv_heads, t.head_dim))
                .context("Inkling 2 Mini global cache overflow")?,
        );
        buffers.insert("page_table".into(), context.div_ceil(64) * 4);
        for (name, width) in [
            ("attn_conv_history", t.hidden),
            ("mlp_conv_history", t.hidden),
            ("k_conv_history", kv_width),
            ("v_conv_history", kv_width),
        ] {
            buffers.insert(name.into(), 3 * width * 4);
        }
        Ok(Self {
            rows,
            blocks,
            kv_format,
            buffers,
        })
    }

    pub fn serving(
        cfg: &Inkling2MiniConfig,
        rows: usize,
        context: usize,
        kv_format: InklingKvCache,
        sequences: usize,
    ) -> Result<Self> {
        let mut layout = Self::with_format(cfg, rows, context, kv_format)?;
        layout.buffers.insert(
            "page_table".into(),
            sequences
                .checked_mul(context.div_ceil(64) * 4)
                .context("Inkling 2 Mini request page table bank overflow")?,
        );
        for name in [
            "ring",
            "global_cache",
            "attn_conv_history",
            "mlp_conv_history",
            "k_conv_history",
            "v_conv_history",
        ] {
            layout.buffers.remove(name);
        }
        layout.buffers.insert(
            "global_slots".into(),
            rows.checked_mul(8).context("Inkling 2 Mini global slot extent overflow")?,
        );
        Ok(layout)
    }

    pub fn serving_decode(
        cfg: &Inkling2MiniConfig,
        context: usize,
        kv_format: InklingKvCache,
        sequences: usize,
    ) -> Result<Self> {
        let mut layout = Self::serving(cfg, DECODE_ROWS, context, kv_format, sequences)?;
        layout.buffers.insert(
            "decode_zero_bf16".into(),
            layout.buffers["attended"].max(layout.buffers["expert_output"]),
        );
        Ok(layout)
    }

    /// Bounded pinned upload source schedule for one unit's full 48-visit walk.
    pub fn upload_shape(
        &self,
        cfg: &Inkling2MiniConfig,
        members: usize,
    ) -> Result<crate::shared::token_table::UploadShape> {
        ensure!(members > 0 && members <= self.rows, "Inkling 2 Mini upload members outside unit");
        let mut shape = crate::shared::token_table::UploadShape::default();
        let visits = cfg.total_visits();
        let repeats = visits.checked_mul(2).context("Mini upload repetitions overflow")?;
        shape.append(
            &[
                self.rows * 8,
                self.rows * 8,
                self.rows * 8,
                self.rows * 8,
                self.rows * 8,
                self.rows * 4,
                self.rows * 4,
                self.rows * 4,
            ],
            repeats,
        )?;
        shape.append(
            &[self.buffers["page_table"]],
            cfg.global_cache_slots()
                .len()
                .checked_mul(2)
                .context("Mini global page upload repetitions overflow")?,
        )?;
        shape.append(&[self.rows * 4], 1)?;
        Ok(shape)
    }

    pub fn bytes(&self) -> Result<usize> {
        self.buffers
            .values()
            .try_fold(0usize, |sum, bytes| sum.checked_add(*bytes).context("Mini unit sum overflow"))
    }
}

/// Persistent 48-slot request state on 1x RTX PRO 6000:
/// - 40 local SWA-1024 append rings (slots whose physical layer is local)
/// - 8 global paged KV planes (`[5, 11, 17, 23, 29, 35, 41, 47]`)
/// - 48 contiguous 3-lag FP32 short-conv histories (`ConvLayout`, width 6656)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MiniCacheLayout {
    pub slots: Vec<SequenceLayout>,
    pub sequence_bytes: usize,
    pub ring_sequence_bytes: usize,
    pub ring_offsets: Vec<usize>,
    pub conv: ConvLayout,
    pub sequences: usize,
    pub pool_pages: usize,
    pub page_bytes: usize,
    pub kv_format: InklingKvCache,
    pub global_slots: Vec<usize>,
}

impl MiniCacheLayout {
    pub fn with_format(
        cfg: &Inkling2MiniConfig,
        sequences: usize,
        pool_pages: usize,
        kv_format: InklingKvCache,
    ) -> Result<Self> {
        cfg.release_geometry()?;
        let t = &cfg.text;
        ensure!(
            sequences > 0 && sequences <= i32::MAX as usize && pool_pages > 0 && pool_pages <= i32::MAX as usize,
            "Inkling 2 Mini cache capacity"
        );
        let cache_slots = cfg.cache_slots();
        let slots = (0..cache_slots)
            .map(|slot| SequenceLayout::new(t, cfg.cache_slot_is_local(slot), 1))
            .collect::<Result<Vec<_>>>()?;
        let mut sequence_bytes = 0usize;
        for slot in &slots {
            sequence_bytes = sequence_bytes
                .checked_add(slot.bytes)
                .context("Inkling 2 Mini cache mark overflow")?;
        }
        let page_bytes = kv_format
            .record_bytes(t.kv_heads, t.head_dim)
            .checked_mul(64)
            .context("Inkling 2 Mini global page overflow")?;
        let global_slots = cfg.global_cache_slots();
        let mut ring_sequence_bytes = 0usize;
        let ring_offsets = slots
            .iter()
            .map(|slot| {
                let offset = ring_sequence_bytes;
                ring_sequence_bytes = ring_sequence_bytes
                    .checked_add(slot.fields["ring"].len())
                    .context("Inkling 2 Mini ring state bytes overflow")?;
                Ok(offset)
            })
            .collect::<Result<Vec<_>>>()?;
        let conv = ConvLayout::new(sequences, cache_slots, t.hidden, t.kv_heads * t.head_dim)?;
        ensure!(
            sequence_bytes == ring_sequence_bytes + conv.slot_stride * 4,
            "Inkling 2 Mini serialized 48-slot state inventory differs"
        );
        Ok(Self {
            slots,
            sequence_bytes,
            ring_sequence_bytes,
            ring_offsets,
            conv,
            sequences,
            pool_pages,
            page_bytes,
            kv_format,
            global_slots,
        })
    }

    pub fn ring_record_bytes(&self) -> usize {
        4096
    }
    pub fn ring_stride_records(&self) -> usize {
        self.ring_sequence_bytes / self.ring_record_bytes()
    }
    pub fn active_bytes(&self) -> Result<usize> {
        self.ring_sequence_bytes
            .checked_mul(self.sequences)
            .context("Inkling 2 Mini active ring cache overflow")
    }
    pub fn retained_bytes(&self, marks: usize) -> Result<usize> {
        self.sequence_bytes
            .checked_mul(marks)
            .context("Inkling 2 Mini retained marks overflow")
    }
    pub fn layer_page_bytes(&self) -> Result<usize> {
        self.page_bytes
            .checked_mul(self.pool_pages)
            .context("Inkling 2 Mini global cache overflow")
    }
    pub fn ring_field(&self, sequence: usize, cache_slot: usize) -> Result<std::ops::Range<usize>> {
        ensure!(
            sequence < self.sequences && cache_slot < self.slots.len(),
            "Inkling 2 Mini ring field outside bank"
        );
        let at = sequence
            .checked_mul(self.ring_sequence_bytes)
            .and_then(|n| n.checked_add(self.ring_offsets[cache_slot]))
            .context("Inkling 2 Mini ring address overflow")?;
        let bytes = self.slots[cache_slot].fields["ring"].len();
        ensure!(bytes > 0, "Inkling 2 Mini global cache slot {cache_slot} has no sliding ring");
        Ok(at..at.checked_add(bytes).context("Inkling 2 Mini ring field end overflow")?)
    }
}

/// Residual candidate counts for physical layer `0..24` (identical across core passes
/// because passes 1 & 2 restart from pre-core `(partial = h_L5, blocks = [embed_normed])`).
pub fn residual_site(cfg: &Inkling2MiniConfig, physical_layer: usize) -> Result<ResidualSite> {
    ResidualSite::layer(&cfg.text, physical_layer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::families::inkling2_mini::config::tests::release_config;

    #[test]
    fn mini_48_slot_cache_and_unit_layouts_match_release_geometry() -> Result<()> {
        let cfg = Inkling2MiniConfig::from_hf(&release_config())?;
        let cache = MiniCacheLayout::with_format(&cfg, 16, 1024, InklingKvCache::Fp8)?;
        assert_eq!(cache.slots.len(), 48);
        assert_eq!(cache.global_slots, vec![5, 11, 17, 23, 29, 35, 41, 47]);
        assert_eq!(cache.conv.width, 6656);
        assert_eq!(cache.conv.layers, 48);
        assert_eq!(cache.active_bytes()? + cache.conv.bytes()?, cache.sequence_bytes * 16);
        assert_eq!(cache.page_bytes, 64 * 2112);

        let decode = MiniUnitLayout::serving_decode(&cfg, 262_144, InklingKvCache::Fp8, 16)?;
        assert_eq!(decode.blocks, 5);
        assert_eq!(decode.buffers["residual_arena"], 5 * 64 * 2304 * 4);
        assert_eq!(decode.buffers["prev_pass_arena"], 2 * 64 * 2304 * 4);
        assert_eq!(decode.buffers["cross_pass_msg"], 4 * 64 * 2304 * 4);
        assert_eq!(decode.buffers["expert_wire"], 64 * 2376);
        assert!(decode.buffers.contains_key("decode_zero_bf16"));

        let prefill = MiniUnitLayout::serving(&cfg, 2048, 262_144, InklingKvCache::Fp8, 16)?;
        assert!(!prefill.buffers.contains_key("ring"));
        assert!(!prefill.buffers.contains_key("decode_zero_bf16"));
        let upload = prefill.upload_shape(&cfg, 16)?;
        assert!(upload.capacity_bytes()? >= upload.bytes);
        Ok(())
    }

    #[test]
    fn mini_residual_candidates_never_exceed_five_across_all_48_visits() -> Result<()> {
        let cfg = Inkling2MiniConfig::from_hf(&release_config())?;
        for visit in cfg.visits() {
            let site = residual_site(&cfg, visit.layer)?;
            assert!((1..=5).contains(&site.attention_candidates));
            assert!((2..=5).contains(&site.mlp_candidates));
            assert_eq!(site.queries, if visit.local { 1 } else { 3 });
        }
        assert_eq!(residual_site(&cfg, 23)?.mlp_candidates, 5);
        Ok(())
    }
}
