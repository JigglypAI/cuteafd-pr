//! Qwen 3.8 Flash Next's exact-row decode graph geometry and working-set
//! reserve: every decode and verify step captures lazily on first use at its
//! exact live row count (`1..=QWEN_DECODE_ROWS`) on `qwen4.segment` (`Scope::Local`),
//! with empty startup prewarm and the shared default donor rule.
use serde::{Deserialize, Serialize};

/// Rows of the decode-shaped programs (`_m64`).
pub const QWEN_DECODE_ROWS: usize = 64;
/// Narrowest decode page-table stride, in 64-row KV pages (4,096 tokens):
/// short sequences up to 4,096 tokens share one table stride instead of
/// recapturing full-attention graphs as sequences cross 256/512/1024/2048 tokens.
pub const QWEN_MIN_PAGE_STRIDE: usize = 64;
/// Tokens per allocation unit (four 64-row record pages) and pages per unit.
pub const QWEN_UNIT_ROWS: usize = 256;
pub const QWEN_UNIT_PAGES: usize = 4;
/// Measured driver bytes of one graph executable, rounded up: the 1-RTX min
/// config (`qwen38-exl3-min`, EXL3 K4.25 PLE FP8, MTP depth 3) captured 33,418
/// executables for 5,102,370,816 B on SM120 PRO (152,683.31 B each). The device
/// cache replaces this initial bootstrap estimate with observed topology growth.
pub const QWEN_GRAPH_BYTES_PER_GRAPH: u64 = 152_684;

/// The resolved capture shape: how many devices own layers, and whether Sparks
/// are attached. Routed experts run outside `qwen4.segment` replay, so 0-Spark
/// and Spark layouts share the same rank-local segment topology per owner count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QwenGraphShapeKey {
    #[serde(rename = "single_0spark")] Single0Spark,
    #[serde(rename = "single_sparks")] SingleSparks,
    #[serde(rename = "dual_0spark")] Dual0Spark,
    #[serde(rename = "dual_sparks")] DualSparks,
}

impl QwenGraphShapeKey {
    pub fn resolve(owners: usize, sparks: bool) -> Self {
        match (owners > 1, sparks) {
            (false, false) => Self::Single0Spark,
            (false, true) => Self::SingleSparks,
            (true, false) => Self::Dual0Spark,
            (true, true) => Self::DualSparks,
        }
    }
    /// Layer-owning devices this shape captures on.
    pub fn owners(self) -> usize { if self.is_dual() { 2 } else { 1 } }
    pub fn sparks(self) -> bool { matches!(self, Self::SingleSparks | Self::DualSparks) }
    fn is_dual(self) -> bool { matches!(self, Self::Dual0Spark | Self::DualSparks) }
}

/// Exact-row C1 working-set census inputs per rank (bytes), fed directly into
/// `graph_reserve::resolve_working_set`:
/// - `soft = ceil_2MiB(W1 + max(20% * W1, 100 MB, 2 * step_growth))`
/// - `growth_headroom = ceil_2MiB(max(2 * capture_growth, 2 MiB))`
/// - `hard = soft + growth_headroom`, `spanning = 0` (`qwen4.segment` is `Scope::Local`).
///
/// Calibrated from the donor-off census (`GPU_GRAPH_RESERVE=4GiB`, pressure-only
/// donors, `baseline + spec_lossless_set + 32K context_load + C1..C16`):
/// - `qwen38-exl3-min` (1 RTX, 49 segments):
///   * Full live shape set high-water: 392.00 MiB (411,041,792 B; 2,756 execs, 46,128 nodes).
///   * W1 (`rows <= 4`, short + long): 30.08 MiB (31,544,571 B; 208 execs, 3,540 nodes);
///     max single-step growth: 6.96 MiB (7,298,024 B; 819 nodes / 49 segs);
///     largest single capture: 4.00 MiB (4,194,304 B).
///   * Resolved: `soft = 126 MiB`, `growth_headroom = 8 MiB`, `hard = 134 MiB`.
/// - `qwen38-exl3-2rtx` (2 RTX; rank 0 = 25 segments, rank 1 = 24 segments):
///   * Full live shape set high-water: 206.00 / 198.00 MiB (216,006,656 / 207,618,048 B;
///     1,456 / 1,398 execs, 24,260 / 23,448 nodes).
///   * W1 (`rows <= 4`, short + long): 15.25 / 14.69 MiB (15,991,260 / 15,406,662 B;
///     106 / 102 execs, 1,796 / 1,740 nodes);
///     max single-step growth: 3.53 / 3.39 MiB (3,703,989 / 3,559,471 B; 416 / 402 nodes);
///     largest single capture: 2.00 / 4.00 MiB (2,097,152 / 4,194,304 B).
///   * Resolved: `soft = 112 / 112 MiB`, `growth_headroom = 4 / 8 MiB`, `hard = 116 / 120 MiB`.
const SINGLE_WORKING_SET: crate::placement::graph_reserve::WorkingSet =
    crate::placement::graph_reserve::WorkingSet {
        w1: 31_544_571,
        step_growth: 7_298_024,
        capture_growth: 4 << 20,
    };
const DUAL_WORKING_SET: [crate::placement::graph_reserve::WorkingSet; 2] = [
    crate::placement::graph_reserve::WorkingSet {
        w1: 15_991_260,
        step_growth: 3_703_989,
        capture_growth: 2 << 20,
    },
    crate::placement::graph_reserve::WorkingSet {
        w1: 15_406_662,
        step_growth: 3_559_471,
        capture_growth: 4 << 20,
    },
];

/// Measured exact-row C1 working-set census inputs of `rank` on `key`.
pub fn qwen_working_set(key: QwenGraphShapeKey, rank: usize)
    -> Option<crate::placement::graph_reserve::WorkingSet> {
    if rank >= key.owners() { return None; }
    if key.is_dual() { Some(DUAL_WORKING_SET[rank]) } else { Some(SINGLE_WORKING_SET) }
}

/// Resolve Qwen4's per-rank reserve through the shared `graph_reserve::current_working_set`.
pub fn qwen_resolve_reserve(key: QwenGraphShapeKey, rank: usize)
    -> Result<crate::placement::graph_reserve::Resolved, String> {
    let measured = qwen_working_set(key, rank)
        .ok_or_else(|| format!("Qwen {key:?} graph reserve has no rank {rank}"))?;
    crate::placement::graph_reserve::current_working_set(rank, measured)
        .map_err(|error| error.to_string())
}

/// Preserve the established short-context allocation buckets, but do not key a
/// long decode by unused pages reserved beyond its live power-of-two bucket.
pub fn decode_allocation_units(allocated: usize, live_tokens: usize, unit: usize) -> usize {
    allocated.min(live_tokens.max(131_072).div_ceil(unit).next_power_of_two())
}

/// Record pages of a pool of `pool_tokens` tokens.
pub fn qwen_pool_pages(pool_tokens: usize) -> Option<usize> {
    pool_tokens.div_ceil(QWEN_UNIT_ROWS).checked_mul(QWEN_UNIT_PAGES)
}

/// Owner-local layer fronts and owner0 head. Cutover and exit posts run eagerly;
/// each physical rank owns only its layer fronts, not a duplicated whole set.
pub fn qwen_graph_segments(owners: &[usize]) -> Option<Vec<u64>> {
    if owners.is_empty() || owners[0] != 0 || owners.iter().any(|&owner| owner > 1) { return None; }
    let dual = owners.contains(&1);
    if dual && (owners.last() != Some(&1)
        || owners.windows(2).filter(|pair| pair[0] != pair[1]).count() != 1) { return None; }
    let mut counts = vec![0u64; if dual { 2 } else { 1 }];
    for &owner in owners { counts[owner] = counts[owner].checked_add(1)?; }
    counts[0] = counts[0].checked_add(1)?;
    Some(counts)
}

#[cfg(test)]
mod owner_graph_tests {
    use super::*;

    #[test]
    fn qwen_owner_segments_match_cutover_and_exit() {
        assert_eq!(qwen_graph_segments(&[0; 48]), Some(vec![49]));
        for cut in [1, 24, 47] {
            let owners: Vec<_> = (0..48).map(|layer| usize::from(layer >= cut)).collect();
            let segments = qwen_graph_segments(&owners).unwrap();
            assert_eq!(segments, vec![cut as u64 + 1, (48 - cut) as u64]);
            assert_eq!(segments.iter().sum::<u64>(), 49);
        }
    }

    #[test]
    fn qwen_exact_row_soft_caps_cover_every_resolved_shape() {
        use crate::placement::graph_reserve::resolve_working_set;
        assert_eq!(QwenGraphShapeKey::resolve(1, false), QwenGraphShapeKey::Single0Spark);
        assert_eq!(QwenGraphShapeKey::resolve(1, true), QwenGraphShapeKey::SingleSparks);
        assert_eq!(QwenGraphShapeKey::resolve(2, false), QwenGraphShapeKey::Dual0Spark);
        assert_eq!(QwenGraphShapeKey::resolve(2, true), QwenGraphShapeKey::DualSparks);
        for key in [QwenGraphShapeKey::Single0Spark, QwenGraphShapeKey::SingleSparks] {
            let ws = qwen_working_set(key, 0).expect("single rank 0");
            assert_eq!(ws, SINGLE_WORKING_SET, "{key:?}");
            let r = resolve_working_set(None, None, None, 0, ws).unwrap();
            assert_eq!((r.base_bytes, r.growth_headroom, r.bytes, r.source),
                (126 << 20, Some(8 << 20), 134 << 20, "working-set reserve"), "{key:?}");
            assert_eq!(qwen_working_set(key, 1), None, "{key:?}");
        }
        for key in [QwenGraphShapeKey::Dual0Spark, QwenGraphShapeKey::DualSparks] {
            let ws0 = qwen_working_set(key, 0).expect("dual rank 0");
            let ws1 = qwen_working_set(key, 1).expect("dual rank 1");
            assert_eq!((ws0, ws1), (DUAL_WORKING_SET[0], DUAL_WORKING_SET[1]), "{key:?}");
            let r0 = resolve_working_set(None, None, None, 0, ws0).unwrap();
            let r1 = resolve_working_set(None, None, None, 1, ws1).unwrap();
            assert_eq!((r0.base_bytes, r0.growth_headroom, r0.bytes, r0.source),
                (112 << 20, Some(4 << 20), 116 << 20, "working-set reserve"), "{key:?} rank 0");
            assert_eq!((r1.base_bytes, r1.growth_headroom, r1.bytes, r1.source),
                (112 << 20, Some(8 << 20), 120 << 20, "working-set reserve"), "{key:?} rank 1");
            assert_eq!(qwen_working_set(key, 2), None, "{key:?}");
        }
    }

    #[test]
    fn qwen_owner_cache_preserves_global_totals_and_mtp_home() {
        let mut config = crate::plan::testing::qwen4_config(48);
        config["text_config"]["mtp_num_hidden_layers"] = serde_json::json!(1);
        let mut cfg = crate::families::qwen4::Qwen4Config::from_hf(&config).unwrap();
        cfg.ple_layers = vec![1];
        for cut in [1, 2, 24, 47] {
            let owners: Vec<_> = (0..48).map(|layer| usize::from(layer >= cut)).collect();
            for (mtp, kv_format) in [false, true].into_iter().flat_map(|mtp|
                [crate::families::qwen4::Qwen4KvCache::Bf16, crate::families::qwen4::Qwen4KvCache::Fp8]
                    .into_iter().map(move |format| (mtp, format))) {
                let whole = super::super::qwen_cache_geometry(&cfg, 48, mtp, kv_format).unwrap();
                let placed = qwen_cache_geometry_placed(&cfg, &owners, mtp, kv_format).unwrap();
                assert_eq!(placed.placement, super::super::KvPlacement::PartitionedLayers);
                assert_eq!(placed.ranks.iter().map(|rank| rank.persistent_unit_bytes).sum::<u64>(), whole.ranks[0].persistent_unit_bytes);
                assert_eq!(placed.ranks.iter().map(|rank| rank.active_state_per_sequence_bytes).sum::<u64>(), whole.ranks[0].active_state_per_sequence_bytes);
                assert_eq!(placed.ranks.iter().map(|rank| rank.retained_mark_bytes).sum::<u64>(), whole.ranks[0].retained_mark_bytes);
                assert_eq!(placed.ranks.iter().map(|rank| rank.speculative_replay_bytes).sum::<u64>(), whole.ranks[0].speculative_replay_bytes);
                assert_eq!(placed.ranks[0].fixed_state_bytes, whole.ranks[0].fixed_state_bytes);
                assert_eq!(placed.ranks[1].fixed_state_bytes, 3 * 64 * 4);
                let no_mtp = qwen_cache_geometry_placed(&cfg, &owners, false, kv_format).unwrap();
                assert_eq!(placed.ranks[1], no_mtp.ranks[1]);
            }
        }
    }

    #[test]
    fn qwen_owner_segments_refuse_unsupported_topologies() {
        for owners in [vec![], vec![1], vec![0, 2], vec![0, 1, 0], vec![0, 1, 0, 1]] {
            assert!(qwen_graph_segments(&owners).is_none());
        }
    }
}

/// Partition the authoritative cache geometry by whole-layer owner, while
/// retaining logical global unit ids. Commit tables exist on each owner;
/// the deferred MTP id buffer and MTP pools remain on the head's owner0.
pub fn qwen_cache_geometry_placed(cfg: &crate::families::qwen4::Qwen4Config, owners: &[usize], mtp: bool,
    kv_format: crate::families::qwen4::Qwen4KvCache)
    -> anyhow::Result<super::FamilyCacheGeometry> {
    use super::{KvPlacement, RankCacheGeometry};
    let segments = qwen_graph_segments(owners).ok_or_else(|| anyhow::anyhow!("invalid Qwen cache owners"))?;
    anyhow::ensure!(owners.len() <= cfg.layers, "Qwen cache owner count exceeds the backbone");
    let mut geometry = super::qwen_cache_geometry(cfg, owners.len(), mtp, kv_format)?;
    if segments.len() == 1 { return Ok(geometry); }
    let mut ranks = vec![RankCacheGeometry::default(); 2];
    let mut previous = RankCacheGeometry::default();
    let add = |dst: &mut u64, bytes: u64| -> anyhow::Result<()> {
        *dst = dst.checked_add(bytes).ok_or_else(|| anyhow::anyhow!("Qwen owner cache overflow"))?;
        Ok(())
    };
    for (layer, &owner) in owners.iter().enumerate() {
        let current = super::qwen_cache_geometry(cfg, layer + 1, false, kv_format)?.ranks[0];
        let rank = &mut ranks[owner];
        add(&mut rank.persistent_unit_bytes, current.persistent_unit_bytes - previous.persistent_unit_bytes)?;
        add(&mut rank.active_state_per_sequence_bytes,
            current.active_state_per_sequence_bytes - previous.active_state_per_sequence_bytes)?;
        add(&mut rank.retained_mark_bytes, current.retained_mark_bytes - previous.retained_mark_bytes)?;
        add(&mut rank.speculative_replay_bytes, current.speculative_replay_bytes - previous.speculative_replay_bytes)?;
        previous = current;
    }
    let all = geometry.ranks[0];
    add(&mut ranks[0].persistent_unit_bytes, all.persistent_unit_bytes - previous.persistent_unit_bytes)?;
    add(&mut ranks[0].active_state_per_sequence_bytes,
        all.active_state_per_sequence_bytes - previous.active_state_per_sequence_bytes)?;
    add(&mut ranks[0].retained_mark_bytes, all.retained_mark_bytes - previous.retained_mark_bytes)?;
    ranks[0].pool_metadata_unit_bytes = all.pool_metadata_unit_bytes;
    ranks[0].fixed_state_bytes = all.fixed_state_bytes;
    ranks[1].fixed_state_bytes = 3 * QWEN_DECODE_ROWS as u64 * 4;
    geometry.placement = KvPlacement::PartitionedLayers;
    geometry.ranks = ranks;
    Ok(geometry)
}

/// Device bytes of one Qwen step workspace (`Qwen4Engine::workspace`) of `t`
/// rows: every buffer has a 256-byte allocation minimum, the shared program scratch
/// (`scratch`, the largest of the step's programs) and the index top-k
/// scratch, `logit_rows` vocabulary rows and the 4 MiB head workspace. `pages`
/// is the pool's record pages (decode tables hold one row per step row).
#[allow(clippy::too_many_arguments)]
pub fn qwen_workspace_bytes(cfg: &crate::families::qwen4::Qwen4Config, t: u64, decode: bool, logit_rows: u64,
    pages: u64, mapped_ple_row_bytes: Option<u64>, scratch: u64, topk_scratch: u64) -> u64 {
    let a = |bytes: u64| bytes.max(256);
    let (h, hc) = (cfg.hidden as u64, 4u64);
    let table_rows = if decode { t } else { 1 };
    let pool_pages = pages / QWEN_UNIT_PAGES as u64;
    let ple_rows = cfg.ple_rows() as u64;
    let (heads, hd) = (cfg.heads as u64, cfg.head_dim as u64);
    let blocks = (cfg.index_budget / 4) as u64;
    let topk = cfg.topk as u64;
    let mut total = [a(t * hc * h * 2), a(t * hc * h * 2), a(t * hc * 2), a(t * h * 2), a(t * h * 2), a(t * h * 2),
        a(t * h * 2), a(t * 8), a(t * 12), a(t * 12), a(t * 8), a(t * 4), a(t * 4), a(t * 8), a(t * 4),
        a(table_rows * pages * 4), a(table_rows * pool_pages * 4), a(t * ple_rows * 8),
        a(t * heads * hd * 2), a(t * heads * hd * 2), a(t * cfg.index_heads as u64 * 128 * 2), a(t * heads * hd * 2),
        a(t * blocks * 4), a(t * 2112 * 4), a(t * 4), a(scratch), a(topk_scratch),
        a(logit_rows * cfg.vocab_size as u64 * 4), a(t * cfg.experts as u64 * 4), a(t * topk * 4), a(t * topk * 4),
        a(t * (h + h / 32)), a(t * 4), a(logit_rows * 8), a(t * 4), a(logit_rows * 8), 4 << 20].iter().sum::<u64>();
    // Packed attention uses a zero-origin member view; PLE retains global origins.
    if !decode { total += a(t * 4); }
    if let Some(row_bytes) = mapped_ple_row_bytes {
        total += a(t * ple_rows * 8) + a(t * ple_rows * row_bytes);
    }
    total
}

/// Unit-owned live rows plus hook-local scratch; retained terminal rows feed
/// the canonical MTP without keeping a whole span of every intermediate plane.
pub fn qwen_unit_workspace_bytes(cfg: &crate::families::qwen4::Qwen4Config, shape: super::PrefillShape,
    terminal_rows: u64, logit_rows: u64, mapped: Option<u64>, scratch: u64, topk: u64) -> u64 {
    let unit = shape.unit_rows;
    let h = cfg.hidden as u64;
    let slot_bytes = |rows: u64| 2 * rows * 4 * h * 2 + rows * 4 * 2 + 3 * rows * h * 2;
    qwen_workspace_bytes(cfg, unit, false, logit_rows, 0, mapped, scratch, topk)
        - slot_bytes(unit) + slot_bytes(shape.slot_rows()) + terminal_rows * 4 * h * 2
}

/// The largest scratch among the programs one Qwen step launches at `cap`
/// (`m64` / `m4096`), and the index top-k scratch, from the program manifest.
pub fn qwen_step_scratch(manifest: &serde_json::Value, decode: bool, ple_fp8: bool,
    kv_format: crate::families::qwen4::Qwen4KvCache) -> (u64, u64) {
    let cap = if decode { "m64" } else { "m4096" };
    let suffix = if kv_format == crate::families::qwen4::Qwen4KvCache::Fp8 { "_kv_fp8" } else { "" };
    let scratch = |name: &str| manifest["programs"].as_array().into_iter().flatten()
        .find(|p| p["name"] == name).and_then(|p| p["scratch_bytes_at_capacity"]["scratch"].as_u64()).unwrap_or(0);
    let names = ["qwen4_hc_pre".to_string(), "qwen4_hc_post_pre".into(), "qwen4_head".into(), "qwen4_shared".into(),
        if ple_fp8 { "qwen4_ple_fp8" } else { "qwen4_ple_bf16" }.into(), "qwen4_mtp_feedback".into(),
        format!("qwen4_gdn_{cap}"), format!("qwen4_attn_producer{suffix}_{cap}"), format!("qwen4_gdn_w8_{cap}"),
        format!("qwen4_attn_producer_w8{suffix}_{cap}"), format!("qwen4_attn_o_w8_{cap}"), format!("qwen4_sparse_gqa{suffix}_{cap}"),
        format!("qwen4_attn_o_{cap}")];
    (names.iter().map(|n| scratch(n)).max().unwrap_or(0), scratch(&format!("qwen4_index_topk_{cap}")))
}

/// Qwen's KV admission inputs, shared by serve-qwen4 and `cuteafd plan`:
/// pool bytes per token and every byte the coordinator GPU still allocates
/// beside the pool after its weights, PLE table and resident experts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QwenAdmission {
    pub per_token: u64,
    /// (category, group, bytes): step workspaces, state, replay, marks,
    /// probe logits, lazily loaded experts.
    pub items: Vec<(cuteafd_core::memory_layout::Category, &'static str, u64)>,
    /// Kept free: allocator slack and growth (`max(--headroom, 3 GiB)`).
    pub headroom: u64,
}

impl QwenAdmission {
    pub fn fixed(&self) -> u64 { self.items.iter().map(|i| i.2).sum::<u64>() + self.headroom }
}

pub struct QwenAdmissionInputs<'a> {
    pub cfg: &'a crate::families::qwen4::Qwen4Config,
    pub layers: usize,
    pub mtp: bool,
    pub kv_format: crate::families::qwen4::Qwen4KvCache,
    pub manifest: Option<&'a serde_json::Value>,
    pub prefill_rows: u64,
    pub prefill_shape: Option<super::PrefillShape>,
    pub slots: u64,
    pub mark_bytes: u64,
    /// Mapped PLE table: row bytes and whether rows are FP8.
    pub ple: Option<(u64, bool)>,
    /// Lazily loaded EXL3 expert window (peak), 0 when experts are resident.
    pub future_expert_bytes: u64,
    pub headroom: u64,
}

pub fn qwen_admission(inputs: &QwenAdmissionInputs<'_>) -> Result<QwenAdmission, super::CacheGeometryError> {
    use cuteafd_core::memory_layout::Category;
    let geometry = super::qwen_cache_geometry(inputs.cfg, inputs.layers, inputs.mtp, inputs.kv_format)?;
    let rank = &geometry.ranks[0];
    let unit = geometry.logical_unit_rows;
    // Prefill owns one page table and decode one per row, each with four
    // record page ids plus one pool page id per allocation unit.
    let separate_mtp = inputs.prefill_shape.is_some() && inputs.mtp;
    let tables_per_unit = (1 + QWEN_DECODE_ROWS as u64 + u64::from(separate_mtp)) * 5 * 4;
    let per_token = (rank.persistent_unit_bytes + rank.pool_metadata_unit_bytes + tables_per_unit).div_ceil(unit);
    let ple_fp8 = inputs.ple.is_some_and(|p| p.1);
    let mapped = inputs.ple.map(|p| p.0);
    let workspace = |decode: bool| {
        let (scratch, topk) = inputs.manifest.map_or((0, 0), |m| qwen_step_scratch(m, decode, ple_fp8, inputs.kv_format));
        let (t, logits) = if decode { (QWEN_DECODE_ROWS as u64, QWEN_DECODE_ROWS as u64) } else { (inputs.prefill_rows.max(1), 1) };
        // Page tables follow the pool (per_token above).
        if !decode {
            if let Some(shape) = inputs.prefill_shape {
                let terminal = if inputs.mtp { inputs.prefill_rows } else { 0 };
                return qwen_unit_workspace_bytes(inputs.cfg, shape, terminal, logits, mapped, scratch, topk);
            }
        }
        qwen_workspace_bytes(inputs.cfg, t, decode, logits, 0, mapped, scratch, topk)
    };
    let mtp_workspace = if separate_mtp {
        let (scratch, topk) = inputs.manifest.map_or((0, 0), |m| qwen_step_scratch(m, false, ple_fp8, inputs.kv_format));
        qwen_workspace_bytes(inputs.cfg, inputs.prefill_shape.unwrap().unit_rows, false, 1, 0, mapped, scratch, topk)
    } else { 0 };
    let items = vec![
        (Category::Workspace, "decode step", workspace(true)),
        (Category::Workspace, "prefill step", workspace(false)),
        (Category::Workspace, "prefill MTP step", mtp_workspace),
        (Category::Kv, "state", rank.active_state_per_sequence_bytes * inputs.slots + rank.fixed_state_bytes
            + rank.speculative_replay_bytes),
        (Category::Prefix, "marks", inputs.mark_bytes),
        (Category::Experts, "lazy EXL3 window", inputs.future_expert_bytes),
    ].into_iter().filter(|i| i.2 > 0).collect();
    Ok(QwenAdmission { per_token, items, headroom: inputs.headroom })
}

#[cfg(test)]
mod tests {
    use super::qwen_step_scratch;
    use crate::families::qwen4::Qwen4KvCache;

    #[test]
    fn serving_head_charge_remains_one_prefill_and_64_decode_rows() {
        use super::{qwen_admission, qwen_workspace_bytes, QwenAdmissionInputs};
        let cfg = crate::families::qwen4::Qwen4Config::from_hf(&crate::plan::testing::qwen4_config(48)).unwrap();
        for rows in [64, 512, 4096, 8256] {
            let admission = qwen_admission(&QwenAdmissionInputs {
                cfg: &cfg, layers: 48, mtp: false, kv_format: Qwen4KvCache::Fp8,
                manifest: None, prefill_rows: rows, prefill_shape: None, slots: 8,
                mark_bytes: 0, ple: None, future_expert_bytes: 0, headroom: 0,
            }).unwrap();
            for (name, capacity, decode, head_rows) in [
                ("prefill step", rows, false, 1), ("decode step", 64, true, 64),
            ] {
                let actual = admission.items.iter().find(|item| item.1 == name).unwrap().2;
                assert_eq!(actual, qwen_workspace_bytes(&cfg, capacity, decode, head_rows, 0, None, 0, 0));
            }
        }
    }

    #[test]
    fn unit_workspace_scales_only_live_planes_and_retained_terminal_rows() {
        use super::{qwen_unit_workspace_bytes, qwen_workspace_bytes};
        use crate::serving_capacity::PrefillShape;
        let cfg = crate::families::qwen4::Qwen4Config::from_hf(&crate::plan::testing::qwen4_config(48)).unwrap();
        let h = cfg.hidden as u64;
        for rows in [512, 1024, 2048, 2752, 4096] {
            let base = qwen_workspace_bytes(&cfg, rows, false, 1, 0, Some(160), 12345, 6789);
            let extra_slot = rows * (2 * 4 * h * 2 + 4 * 2 + 3 * h * 2);
            for waves in 1..=4 {
                let shape = PrefillShape { unit_rows: rows, waves }.validate().unwrap();
                let slot_bytes = qwen_unit_workspace_bytes(&cfg, shape, 0, 1, Some(160), 12345, 6789);
                assert_eq!(slot_bytes, base + (waves - 1) * extra_slot);
                for terminal in [4096, shape.slot_rows(), 16384] {
                    assert_eq!(qwen_unit_workspace_bytes(&cfg, shape, terminal, 1, Some(160), 12345, 6789),
                        slot_bytes + terminal * 4 * h * 2);
                }
            }
        }
        assert_eq!(crate::serving_capacity::prefill_return_bytes(2, 2048, 4, 2048), 4
            * cuteafd_core::serving_capacity::spark_intake_allocation_bytes(2 * 2048 * 2048 * 2).unwrap());
    }

    #[test]
    fn unit_mtp_scratch_and_page_tables_are_admitted_separately() {
        use super::{qwen_admission, qwen_workspace_bytes, QwenAdmissionInputs};
        use crate::serving_capacity::PrefillShape;
        let mut config = crate::plan::testing::qwen4_config(48);
        config["text_config"]["mtp_num_hidden_layers"] = serde_json::json!(1);
        let cfg = crate::families::qwen4::Qwen4Config::from_hf(&config).unwrap();
        let mut inputs = QwenAdmissionInputs { cfg: &cfg, layers: 48, mtp: true, kv_format: Qwen4KvCache::Fp8,
            manifest: None, prefill_rows: 4096, prefill_shape: None, slots: 8, mark_bytes: 0,
             ple: Some((160, true)), future_expert_bytes: 0, headroom: 0 };
        let legacy = qwen_admission(&inputs).unwrap();
        inputs.prefill_shape = Some(PrefillShape { unit_rows: 2048, waves: 2 });
        let units = qwen_admission(&inputs).unwrap();
        let bytes = |name| units.items.iter().find(|item| item.1 == name).unwrap().2;
        assert_eq!(bytes("prefill MTP step"), qwen_workspace_bytes(&cfg, 2048, false, 1, 0, Some(160), 0, 0));
        let geometry = crate::serving_capacity::qwen_cache_geometry(&cfg, 48, true, Qwen4KvCache::Fp8).unwrap();
        let rank = &geometry.ranks[0];
        let pool_bytes = rank.persistent_unit_bytes + rank.pool_metadata_unit_bytes;
        assert_eq!(legacy.per_token, (pool_bytes + 65 * 5 * 4).div_ceil(geometry.logical_unit_rows));
        assert_eq!(units.per_token, (pool_bytes + 66 * 5 * 4).div_ceil(geometry.logical_unit_rows));
        inputs.mtp = false;
        let no_mtp = qwen_admission(&inputs).unwrap();
        assert!(!no_mtp.items.iter().any(|item| item.1 == "prefill MTP step"));
        assert_eq!(bytes("prefill step") - no_mtp.items.iter().find(|item| item.1 == "prefill step").unwrap().2,
            4096 * 4 * cfg.hidden as u64 * 2);
    }

    #[test]
    fn scratch_admission_selects_the_kv_format_and_capacity() {
        let mut programs = Vec::new();
        for (cap, factor) in [("m64", 1_u64), ("m4096", 10)] {
            for (stem, bytes) in [("attn_producer", 300), ("attn_producer_w8", 400),
                ("sparse_gqa", 500), ("attn_producer_kv_fp8", 600),
                ("attn_producer_w8_kv_fp8", 700), ("sparse_gqa_kv_fp8", 800),
                ("index_topk", 200)] {
                programs.push(serde_json::json!({"name": format!("qwen4_{stem}_{cap}"),
                    "scratch_bytes_at_capacity": {"scratch": bytes * factor}}));
            }
        }
        let manifest = serde_json::json!({"programs": programs});
        for (decode, factor) in [(true, 1), (false, 10)] {
            assert_eq!(qwen_step_scratch(&manifest, decode, false, Qwen4KvCache::Bf16),
                (500 * factor, 200 * factor));
            assert_eq!(qwen_step_scratch(&manifest, decode, false, Qwen4KvCache::Fp8),
                (800 * factor, 200 * factor));
        }
    }
}
