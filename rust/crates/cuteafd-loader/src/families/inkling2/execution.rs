//! Native stage2 arithmetic geometry and immutable program/owner contracts.
//! No scheduling or placement decisions live here.
use super::{Inkling2Config, InklingKvCache};
use anyhow::{ensure, Context, Result};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnitLayout {
    pub rows: usize,
    pub blocks: usize,
    pub kv_format: InklingKvCache,
    pub buffers: BTreeMap<String, usize>,
}

impl UnitLayout {
    pub fn new(cfg: &Inkling2Config, rows: usize, context: usize) -> Result<Self> {
        Self::with_format(cfg, rows, context, InklingKvCache::Bf16)
    }
    pub fn with_format(cfg: &Inkling2Config, rows: usize, context: usize, kv_format: InklingKvCache) -> Result<Self> {
        ensure!((1..=4096).contains(&rows) && rows <= context && context <= cfg.max_position,
            "Inkling arithmetic rows/context outside checkpoint/export");
        ensure!((cfg.hidden, cfg.heads, cfg.kv_heads, cfg.head_dim, cfg.d_rel, cfg.sconv_kernel)
            == (4608, 48, 8, 128, 16, 4) && cfg.res_block > 0,
            "Inkling native arithmetic requires released Small widths/convolution geometry");
        let blocks = cfg.layers.div_ceil(cfg.res_block) + 1;
        let mut buffers = BTreeMap::new();
        let mut add = |name: &str, width: usize, bytes: usize| -> Result<()> {
            let extent = rows.checked_mul(width).and_then(|n| n.checked_mul(bytes))
                .context("Inkling unit extent overflow")?;
            ensure!(buffers.insert(name.into(), extent).is_none(), "duplicate Inkling arithmetic owner");
            Ok(())
        };
        for name in ["partial", "attention", "mlp", "branch", "mixed", "projection_input"] {
            add(name, cfg.hidden, if name == "projection_input" { 2 } else { 4 })?;
        }
        add("residual_arena", blocks * cfg.hidden, 4)?;
        add("global_mixes", 3 * cfg.hidden, 4)?;
        for (name, width, bytes) in [("q", 6144, 4), ("r", 768, 4), ("k", 1024, 4), ("v", 1024, 4),
            ("q_clipped", 6144, 4), ("r_clipped", 768, 4), ("k_clipped", 1024, 4), ("v_norm", 1024, 4),
            ("q_bf16", 6144, 2), ("relative_source", 768, 2), ("relative_input", 48*64, 2), ("relative", 48 * 1024, 2),
            ("relative_f32", 48*1024, 4), ("conv_k", 1024, 4), ("conv_v", 1024, 4),
            ("gate", 6144, 4), ("sinks", 48, 4), ("attended", 6144, 2), ("gated", 6144, 2),
            ("kv_step", 2048, 2), ("k_dynamic", 32, 4), ("v_dynamic", 32, 4),
            ("route_logits", cfg.experts, 4), ("route_ids", cfg.topk, 4), ("route_weights", cfg.topk, 4),
            ("expert_wire", cfg.hidden + cfg.hidden/32, 1), ("expert_output", cfg.hidden, 2),
            ("table_indices", 1, 4), ("tile_rows", 1, 4), ("positions", 1, 8), ("first", 1, 4), ("conv_slots", 1, 8), ("ring_slots", 1, 8), ("cache_slots", 1, 8)] {
            add(name, width, bytes)?;
        }
        buffers.insert("attention_scratch".into(), 64 * 32 * 48 * (128*4+4));
        buffers.insert("ring".into(), 1088 * 4096);
        buffers.insert("global_cache".into(), context.div_ceil(64).checked_mul(64*kv_format.record_bytes(cfg.kv_heads, cfg.head_dim)).context("Inkling cache overflow")?);
        buffers.insert("page_table".into(), context.div_ceil(64)*4);
        // Cold golden reuses one layer's four conv histories serially. Serving
        // history belongs to each layer's cache owner, not this scratch layout.
        for (name, width) in [("attn_conv_history", 4608), ("mlp_conv_history", 4608),
            ("k_conv_history", 1024), ("v_conv_history", 1024)] {
            buffers.insert(name.into(), 3*width*4);
        }
        Ok(Self { rows, blocks, kv_format, buffers })
    }
    /// Serving binds per-layer state from the persistent request bank. Scratch
    /// must not also allocate cold-golden copies of those physical owners.
    pub fn serving(cfg: &Inkling2Config, rows: usize, context: usize) -> Result<Self> {
        Self::serving_format(cfg, rows, context, InklingKvCache::Bf16, 1)
    }
    pub fn serving_format(cfg: &Inkling2Config, rows: usize, context: usize, kv_format: InklingKvCache,
        sequences: usize) -> Result<Self> {
        let mut layout = Self::with_format(cfg, rows, context, kv_format)?;
        layout.buffers.insert("page_table".into(), sequences.checked_mul(context.div_ceil(64)*4)
            .context("Inkling request table bank overflow")?);
        for name in ["ring", "global_cache", "attn_conv_history", "mlp_conv_history",
            "k_conv_history", "v_conv_history"] { layout.buffers.remove(name); }
        layout.buffers.insert("global_slots".into(), rows.checked_mul(8).context("Inkling physical slot extent overflow")?);
        Ok(layout)
    }

    /// Exact bounded source schedule for one unit's full layer walk: five
    /// packed i64 planes plus starts, table selectors and CTA starts. The
    /// second allowance covers deferred MLP joins; global tables are one bank.
    pub fn upload_shape(&self, cfg: &Inkling2Config, members: usize) -> Result<crate::shared::token_table::UploadShape> {
        ensure!(members > 0 && members <= self.rows, "Inkling upload member count outside unit");
        let mut shape = crate::shared::token_table::UploadShape::default();
        let repeats = cfg.layers.checked_mul(2).context("Inkling upload repetitions overflow")?;
        shape.append(&[self.rows*8, self.rows*8, self.rows*8, self.rows*8, self.rows*8,
            self.rows*4, self.rows*4, self.rows*4], repeats)?;
        shape.append(&[self.buffers["page_table"]], cfg.global_layers().len().checked_mul(2)
            .context("Inkling page upload repetitions overflow")?)?;
        shape.append(&[self.rows*4], 1)?;
        Ok(shape)
    }

    pub fn bytes(&self) -> Result<usize> {
        self.buffers.values().try_fold(0usize, |sum, bytes| sum.checked_add(*bytes).context("Inkling owner sum overflow"))
    }
}

/// Byte layout of one layer's exact per-sequence mark. KV records are BF16;
/// convolution histories preserve the native producers' FP32 inputs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SequenceLayout {
    pub ring_rows: usize,
    pub kv_width: usize,
    pub fields: BTreeMap<String, std::ops::Range<usize>>,
    pub bytes: usize,
}

impl SequenceLayout {
    /// `kv_parts` splits KV heads only. Hidden-stream convolution inputs are
    /// replicated under both head and context splitting.
    pub fn new(cfg: &Inkling2Config, local: bool, kv_parts: usize) -> Result<Self> {
        ensure!(matches!(kv_parts, 1 | 2) && cfg.kv_heads > 0
            && cfg.kv_heads % kv_parts == 0 && cfg.head_dim > 0 && cfg.hidden > 0
            && cfg.sconv_kernel > 1, "Inkling sequence state geometry");
        let kv_width = (cfg.kv_heads / kv_parts).checked_mul(cfg.head_dim)
            .context("Inkling sequence KV width overflow")?;
        let ring_rows = if local {
            usize::try_from(cuteafd_core::prefix::swa_ring(u64::try_from(cfg.window)?, 64, 64)
                .context("Inkling sequence ring geometry")?.rows)?
        } else { 0 };
        let lags = cfg.sconv_kernel - 1;
        let mut fields = BTreeMap::new();
        let mut bytes = 0usize;
        for (name, rows, width, element) in [
            ("ring", ring_rows, kv_width, 4),
            ("k_conv_history", lags, kv_width, 4),
            ("v_conv_history", lags, kv_width, 4),
            ("attn_conv_history", lags, cfg.hidden, 4),
            ("mlp_conv_history", lags, cfg.hidden, 4),
        ] {
            let size = rows.checked_mul(width).and_then(|n| n.checked_mul(element))
                .context("Inkling sequence field overflow")?;
            let end = bytes.checked_add(size).context("Inkling sequence mark overflow")?;
            fields.insert(name.into(), bytes..end);
            bytes = end;
        }
        Ok(Self { ring_rows, kv_width, fields, bytes })
    }

    pub fn field(&self, name: &str) -> Result<std::ops::Range<usize>> {
        self.fields.get(name).cloned().with_context(|| format!("Inkling sequence lacks {name}"))
    }
}

/// Persistent whole-head state on one physical rank. Logical pool pages come
/// from the shared prefix pool; each global layer retains its own page plane.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheLayout {
    pub layers: Vec<SequenceLayout>,
    /// Serialized mark bytes: all rings followed by the contiguous FIR bank.
    pub sequence_bytes: usize,
    pub ring_sequence_bytes: usize,
    pub ring_offsets: Vec<usize>,
    pub conv: super::conv::ConvLayout,
    pub sequences: usize,
    pub pool_pages: usize,
    pub page_bytes: usize,
    pub kv_format: InklingKvCache,
    pub global_layers: Vec<usize>,
}
impl CacheLayout {
    pub fn new(cfg: &Inkling2Config, sequences: usize, pool_pages: usize) -> Result<Self> {
        Self::with_format(cfg, sequences, pool_pages, InklingKvCache::Bf16)
    }
    pub fn with_format(cfg: &Inkling2Config, sequences: usize, pool_pages: usize, kv_format: InklingKvCache) -> Result<Self> {
        ensure!(sequences > 0 && sequences <= i32::MAX as usize && pool_pages > 0
            && pool_pages <= i32::MAX as usize && cfg.layers == cfg.local.len(), "Inkling cache capacity");
        let layers = cfg.local.iter().map(|&local| SequenceLayout::new(cfg, local, 1))
            .collect::<Result<Vec<_>>>()?;
        let mut sequence_bytes = 0usize;
        for layer in &layers {
            sequence_bytes = sequence_bytes.checked_add(layer.bytes).context("Inkling cache mark overflow")?;
        }
        let page_bytes = kv_format.record_bytes(cfg.kv_heads, cfg.head_dim).checked_mul(64)
            .context("Inkling cache page overflow")?;
        let global_layers = cfg.local.iter().enumerate().filter_map(|(layer, &local)| (!local).then_some(layer)).collect();
        let mut ring_sequence_bytes = 0usize;
        let ring_offsets = layers.iter().map(|layer| {
            let offset = ring_sequence_bytes;
            ring_sequence_bytes = ring_sequence_bytes.checked_add(layer.fields["ring"].len())
                .context("Inkling ring state bytes overflow")?;
            Ok(offset)
        }).collect::<Result<Vec<_>>>()?;
        let conv = super::conv::ConvLayout::new(sequences, cfg.layers, cfg.hidden, cfg.kv_heads*cfg.head_dim)?;
        ensure!(sequence_bytes == ring_sequence_bytes + conv.slot_stride*4, "Inkling serialized state inventory differs");
        Ok(Self { layers, sequence_bytes, ring_sequence_bytes, ring_offsets, conv,
            sequences, pool_pages, page_bytes, kv_format, global_layers })
    }
    pub fn active_bytes(&self) -> Result<usize> {
        self.ring_sequence_bytes.checked_mul(self.sequences).context("Inkling active ring cache overflow")
    }
    pub fn retained_bytes(&self, marks: usize) -> Result<usize> {
        self.sequence_bytes.checked_mul(marks).context("Inkling retained marks overflow")
    }
    pub fn layer_page_bytes(&self) -> Result<usize> {
        self.page_bytes.checked_mul(self.pool_pages).context("Inkling global cache overflow")
    }
    pub fn ring_field(&self, sequence: usize, layer: usize) -> Result<std::ops::Range<usize>> {
        ensure!(sequence < self.sequences && layer < self.layers.len(), "Inkling ring field outside bank");
        let at = sequence.checked_mul(self.ring_sequence_bytes).and_then(|n|n.checked_add(self.ring_offsets[layer]))
            .context("Inkling ring address overflow")?;
        let bytes = self.layers[layer].fields["ring"].len();
        ensure!(bytes > 0, "Inkling global layer has no sliding ring");
        Ok(at..at.checked_add(bytes).context("Inkling ring field end overflow")?)
    }

}

/// One contiguous request segment. Physical pages are supplied by the shared
/// refcounted pool, never inferred from logical positions or sequence IDs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnitMetadata {
    pub positions: Vec<i64>,
    pub local_slots: Vec<i64>,
    pub global_slots: Vec<i64>,
    pub ring_slots: Vec<i64>,
    pub first: Vec<i32>,
    pub conv_slots: Vec<i64>,
    pub page_table: Vec<i32>,
}
impl UnitMetadata {
    pub fn new(cfg: &Inkling2Config, span: std::ops::Range<usize>, pages: &[u32],
        pool_pages: usize) -> Result<Self> {
        ensure!(span.start < span.end && span.len() <= 4096 && span.end <= cfg.max_position,
            "Inkling segment outside context/export");
        ensure!(pages.len() >= span.end.div_ceil(64)
            && pages.iter().all(|&page| (page as usize) < pool_pages && page <= i32::MAX as u32),
            "Inkling segment lacks valid physical pages");
        let unique = pages.iter().copied().collect::<std::collections::BTreeSet<_>>();
        ensure!(unique.len() == pages.len(), "Inkling request aliases physical KV pages");
        let ring = SequenceLayout::new(cfg, true, 1)?.ring_rows;
        Ok(Self {
            positions: span.clone().map(|row| row as i64).collect(),
            local_slots: (0..span.len()).map(|row| row as i64).collect(),
            global_slots: span.clone().map(|row| i64::from(pages[row/64])*64 + (row%64) as i64).collect(),
            ring_slots: span.clone().map(|row| (row%ring) as i64).collect(),
            first: vec![0; span.len()], conv_slots: vec![0; span.len()],
            page_table: pages.iter().map(|&page| page as i32).collect(),
        })
    }
    pub fn rows(&self) -> usize { self.positions.len() }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidualSite {
    /// Completed blocks present before attention, including the embedding block.
    pub attention_candidates: usize,
    pub mlp_candidates: usize,
    pub queries: usize,
    pub starts_block: bool,
}
impl ResidualSite {
    pub fn layer(cfg: &Inkling2Config, layer: usize) -> Result<Self> {
        ensure!(layer < cfg.layers, "Inkling residual layer outside checkpoint");
        let completed = layer.div_ceil(cfg.res_block);
        let starts_block = layer % cfg.res_block == 0;
        Ok(Self { attention_candidates: completed+1,
            mlp_candidates: completed+1+usize::from(starts_block),
            queries: if cfg.local[layer] { 1 } else { 3 }, starts_block })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn cfg() -> Inkling2Config {
        Inkling2Config::from_hf(&serde_json::from_str(include_str!("fixtures/config.json")).unwrap()).unwrap()
    }
    #[test]
    fn fp8_records_charge_payload_and_head_scales_once() -> Result<()> {
        let cfg = cfg();
        assert_eq!(InklingKvCache::Fp8.record_bytes(8, 128), 2112);
        assert_eq!(InklingKvCache::Bf16.record_bytes(8, 128), 4096);
        let bf16 = CacheLayout::new(&cfg, 16, 32768)?;
        let fp8 = CacheLayout::with_format(&cfg, 16, 32768, InklingKvCache::Fp8)?;
        assert_eq!(fp8.page_bytes, 64*2112);
        assert_eq!(bf16.sequence_bytes, fp8.sequence_bytes);
        assert_eq!(bf16.layer_page_bytes()? - fp8.layer_page_bytes()?, 32768*64*(4096-2112));
        assert_ne!(fp8.kv_format.record_format(), bf16.kv_format.record_format());
        Ok(())
    }
    #[test]
    fn metadata_upload_shape_covers_every_segment_and_physical_source_once() -> Result<()> {
        let cfg = cfg();
        let layout = UnitLayout::serving(&cfg, 2752, 1 << 20)?;
        for members in [1usize, 2, 16] {
            let shape = layout.upload_shape(&cfg, members)?;
            let mut expected = crate::shared::token_table::UploadShape::default();
            expected.append(&[2752*8, 2752*8, 2752*8, 2752*8, 2752*8, 2752*4, 2752*4, 2752*4], cfg.layers*2)?;
            expected.append(&[65536], cfg.global_layers().len()*2)?;
            expected.append(&[2752*4], 1)?;
            assert_eq!(shape, expected);
            assert!(shape.capacity_bytes()? >= shape.bytes);
        }
        assert!(layout.upload_shape(&cfg, 0).is_err());
        assert!(layout.upload_shape(&cfg, 2753).is_err());
        Ok(())
    }
    #[test]
    fn unit_metadata_uses_physical_pages_and_resets_only_chunk_local_indices() -> Result<()> {
        let cfg = cfg();
        let mut pages = (0..24).rev().collect::<Vec<u32>>();
        let first = UnitMetadata::new(&cfg, 63..67, &pages, 24)?;
        assert_eq!(first.positions, [63, 64, 65, 66]);
        assert_eq!(first.global_slots, [23*64+63, 22*64, 22*64+1, 22*64+2]);
        assert_eq!(first.local_slots, [0, 1, 2, 3]);
        assert_eq!(first.first, [0; 4]);
        assert_eq!(first.conv_slots, [0; 4]);
        let wrapped = UnitMetadata::new(&cfg, 1087..1090, &pages, 24)?;
        assert_eq!(wrapped.ring_slots, [1087, 0, 1]);
        assert_eq!(wrapped.positions, [1087, 1088, 1089]);
        pages[1] = pages[0];
        assert!(UnitMetadata::new(&cfg, 0..2, &pages, 24).is_err());
        assert!(UnitMetadata::new(&cfg, 64..66, &[0], 24).is_err());
        assert!(UnitMetadata::new(&cfg, 0..2, &[24], 24).is_err());
        assert!(UnitMetadata::new(&cfg, 0..4097, &(0..65).collect::<Vec<_>>(), 65).is_err());
        Ok(())
    }
    #[test]
    fn persistent_cache_layout_separates_layers_requests_and_unit_scratch() -> Result<()> {
        let cfg = cfg();
        let cache = CacheLayout::new(&cfg, 3, 19)?;
        assert_eq!(cache.global_layers.len(), 8);
        assert_eq!(cache.page_bytes, 64*4096);
        assert_eq!(cache.active_bytes()? + cache.conv.bytes()?, cache.sequence_bytes*3);
        assert_eq!(cache.retained_bytes(7)?, cache.sequence_bytes*7);
        let mut spans = Vec::new();
        for sequence in 0..3 {
            for layer in cfg.local.iter().enumerate().filter_map(|(i,&local)|local.then_some(i)) {
                spans.push(cache.ring_field(sequence, layer)?);
            }
        }
        spans.sort_by_key(|span|span.start);
        let mut end=0;
        for span in spans { assert_eq!(span.start,end);end=span.end; }
        assert_eq!(end,cache.active_bytes()?);
        assert!(cache.ring_field(3,0).is_err());
        assert!(cache.ring_field(0,cfg.layers).is_err());
        assert_eq!(cache.conv.width,11264);
        let unit = UnitLayout::serving(&cfg, 64, 8192)?;
        for name in ["ring", "global_cache", "attn_conv_history", "mlp_conv_history", "k_conv_history", "v_conv_history"] {
            assert!(!unit.buffers.contains_key(name));
        }
        Ok(())
    }

    #[test]
    fn six_layer_residual_boundaries_preserve_embedding_block() {
        let cfg = cfg();
        for (layer, attn, mlp, global) in [(0, 1, 2, false), (1, 2, 2, false), (5, 2, 2, true),
            (6, 2, 3, false), (7, 3, 3, false), (47, 9, 9, true)] {
            let site = ResidualSite::layer(&cfg, layer).unwrap();
            assert_eq!((site.attention_candidates, site.mlp_candidates, site.queries),
                (attn, mlp, if global { 3 } else { 1 }));
        }
    }
    #[test]
    fn sequence_marks_preserve_fp32_lags_and_split_only_kv_heads() {
        let cfg = cfg();
        for local in [false, true] {
            for parts in [1, 2] {
                let layout = SequenceLayout::new(&cfg, local, parts).unwrap();
                let kv = 1024 / parts;
                assert_eq!(layout.ring_rows, if local { 1088 } else { 0 });
                assert_eq!(layout.kv_width, kv);
                assert_eq!(layout.field("ring").unwrap().len(), layout.ring_rows * kv * 4);
                for name in ["k_conv_history", "v_conv_history"] {
                    assert_eq!(layout.field(name).unwrap().len(), 3 * kv * 4);
                }
                for name in ["attn_conv_history", "mlp_conv_history"] {
                    assert_eq!(layout.field(name).unwrap().len(), 3 * 4608 * 4);
                }
                let mut spans = layout.fields.values().cloned().collect::<Vec<_>>();
                spans.sort_by_key(|span| (span.start, span.end));
                let mut end = 0;
                for span in spans { assert_eq!(span.start, end); end = span.end; }
                assert_eq!(end, layout.bytes);
                assert!(layout.field("missing").is_err());
            }
        }
        for parts in [0, 3, usize::MAX] { assert!(SequenceLayout::new(&cfg, true, parts).is_err()); }
        let mut invalid = cfg;
        invalid.sconv_kernel = 0;
        assert!(SequenceLayout::new(&invalid, true, 1).is_err());
        invalid.sconv_kernel = 4;
        invalid.hidden = usize::MAX;
        assert!(SequenceLayout::new(&invalid, true, 1).is_err());
    }

    #[test]
    fn every_unit_owner_has_checked_named_extent() {
        let cfg = cfg();
        for rows in [1, 64, 2048, 2752, 4096] {
            let layout = UnitLayout::new(&cfg, rows, 8192).unwrap();
            assert_eq!(layout.buffers["residual_arena"], 9*rows*4608*4);
            assert_eq!(layout.buffers["global_cache"], 8192*4096);
            assert_eq!(layout.buffers["global_mixes"], 3*rows*4608*4);
            assert_eq!(layout.buffers["expert_wire"], rows*(4608+144));
            assert!(layout.bytes().unwrap()>0);
        }
        for rows in [0, 4097, usize::MAX] { assert!(UnitLayout::new(&cfg, rows, 8192).is_err()); }
    }
}
