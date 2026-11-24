//! Inkling 2 Mini (`inkling2_mini`) `cuteafd plan --layout` TP1 memory admission.
//!
//! Models every named physical owner on 1x RTX PRO 6000 (0 Sparks):
//! - Packed target weight arena (globals + cross-pass queries + 24 layers + 144 controllers + padded rel proj)
//! - Packed routed-expert arenas (NVFP4 layers 0..20 and MXFP8 layers 20..24)
//! - Optional bundled 3-layer recurrent dSpark drafter arena + runtime/KV
//! - Optional bundled 23-layer PE-Lang vision tower arena + scratch
//! - 48-slot KV + short-conv state: 8 global paged slots (slots 5, 11, 17, 23, 29, 35, 41, 47)
//!   plus 40 local SWA-1024 append rings and 48x3 lag conv histories in the per-sequence + prefix-mark arena
//! - Prefill/decode unit scratch, CUDA context + loaded code, #2697 soft-cap working-set graph reserve,
//!   and shared host prefix cache.
use super::Model;
use crate::families::inkling2::execution::SequenceLayout;
use crate::families::inkling2_mini::weight_layout::{layer_experts, MiniTargetWeights};
use crate::formats::block_dspark::recurrent::{native_rank_arena, RecurrentBlockConfig};
use crate::media::still_tower::StillTowerInventory;
use crate::placement::graph_reserve::{self, WorkingSet};
use crate::placement::planner::Context;
use crate::placement::{ArchContext, AttentionPlacement, PoolFacts, PoolMemory};
use crate::plan::families::inkling2::ExpertStorage;
use crate::plan::{ExpertPlacement, MediaMode, PlanReport};
use cuteafd_core::memory_layout::{Basis, Category, DeviceKind, DeviceLayout, Item, MemoryLayout};

const PAGE_ROWS: u64 = 64;
const GIB: f64 = (1u64 << 30) as f64;
const MIB: u64 = 1 << 20;

fn empty(notes: Vec<String>) -> MemoryLayout {
    MemoryLayout {
        devices: Vec::new(),
        pool_tokens: 0,
        waste: Vec::new(),
        notes,
    }
}

pub(super) fn layout(
    model: &Model,
    input: &crate::plan::PlanInput,
    report: &mut PlanReport,
    opts: &mut crate::plan::layout::LayoutOptions,
    requested: ExpertPlacement,
) -> MemoryLayout {
    let cfg = &model.cfg;
    let t = &cfg.text;
    let gpus = opts.rtx_bytes.len();
    let ranks = requested.spark_ranks();
    report.bytes_by_owner.clear();

    let mut notes = vec![
        "Inkling 2 Mini TP1 layout: 1x RTX, 0 Sparks, 48 distinct KV + short-conv cache slots (8 global paged + 40 SWA-1024)".to_string(),
    ];
    if gpus != 1 {
        report.refuse_placement(
            format!("Inkling 2 Mini is TP1-only (1 coordinator GPU), not {gpus}"),
            "Use --rtx 1 --sparks 0.",
        );
        return empty(notes);
    }
    if ranks != 0 {
        report.refuse_placement(
            format!("Inkling 2 Mini runs locally on 1x RTX with 0 Sparks, not {ranks} Spark ranks"),
            "Use --rtx 1 --sparks 0.",
        );
        return empty(notes);
    }
    if opts.host_embedding {
        report.refuse_placement(
            "Inkling 2 Mini keeps embed on GPU; --embedding-placement host is unsupported".into(),
            "Use --embedding-placement gpu.",
        );
        return empty(notes);
    }
    report.attention_placement = AttentionPlacement::Heads;

    let limits = crate::plan::Family::prefill_limits(&super::INKLING2_MINI);
    let prefill = match crate::plan::layout::resolve_prefill(input, report, opts, 0, limits) {
        Ok(p) => p,
        Err(error) => return crate::plan::layout::unsupported_prefill(report, error),
    };
    report.prefill_reserve_rows = Some(prefill.span_rows);

    let target = match MiniTargetWeights::plan(input.checkpoint(), cfg) {
        Ok(w) => w,
        Err(error) => {
            report.refuse_placement(
                format!("Inkling 2 Mini target arena refused: {error:#}"),
                "Check checkpoint headers for all 24 layers, cross-pass queries, and 144 controllers.",
            );
            return empty(notes);
        }
    };
    let exec_arena = match target.execution_arena(t.layers) {
        Ok(a) => a,
        Err(error) => {
            report.refuse_placement(format!("Inkling 2 Mini execution arena overflow: {error:#}"), "Check layout.");
            return empty(notes);
        }
    };
    let experts = match layer_experts(input.checkpoint(), cfg) {
        Ok(e) => e,
        Err(error) => {
            report.refuse_placement(
                format!("Inkling 2 Mini routed expert inventory refused: {error:#}"),
                "Provide complete NVFP4/MXFP8 expert headers for layers 0..24.",
            );
            return empty(notes);
        }
    };

    let nvfp4_layers: Vec<_> = experts.iter().filter(|e| e.storage == ExpertStorage::Nvfp4).collect();
    let mxfp8_layers: Vec<_> = experts.iter().filter(|e| e.storage == ExpertStorage::Mxfp8).collect();
    let nvfp4_arena_bytes: u64 = nvfp4_layers.iter().map(|e| e.resident_bytes).sum();
    let mxfp8_arena_bytes: u64 = mxfp8_layers.iter().map(|e| e.resident_bytes).sum();
    let expert_total_bytes = nvfp4_arena_bytes + mxfp8_arena_bytes;
    let target_arena_bytes = exec_arena.bytes() as u64;

    report.bytes_by_owner.insert("coordinator target arena".into(), target_arena_bytes);
    report.bytes_by_owner.insert("routed expert arenas (NVFP4 + MXFP8)".into(), expert_total_bytes);

    let mut items = vec![
        Item::new(
            Category::Weights,
            "target packed arena (globals + 24 layers + 144 controllers + cross-pass queries)",
            "bf16/f32",
            target_arena_bytes,
            Basis::Exact,
        ),
        Item::new(
            Category::Experts,
            format!("NVFP4 W4A4 routed expert arena ({} layers)", nvfp4_layers.len()),
            "nvfp4-g16-amax",
            nvfp4_arena_bytes,
            Basis::Exact,
        ),
        Item::new(
            Category::Experts,
            format!("MXFP8 W8A8 routed expert arena ({} layers)", mxfp8_layers.len()),
            "fp8-mxfp8-g32",
            mxfp8_arena_bytes,
            Basis::Exact,
        ),
    ];

    // Optional bundled 3-layer recurrent dSpark drafter arena.
    let mut drafter_bytes = 0u64;
    if opts.inkling_dspark && model.has_mtp {
        if let Ok(rcfg) = RecurrentBlockConfig::from_hf(&input.checkpoint().config) {
            if let Ok(arena) = native_rank_arena(&rcfg, 0, 1, 0) {
                drafter_bytes = arena.bytes() as u64;
                items.push(Item::new(
                    Category::Drafter,
                    "bundled 3-layer recurrent dSpark drafter arena",
                    "bf16",
                    drafter_bytes,
                    Basis::Exact,
                ));
            }
        }
    }

    // Optional bundled 23-layer PE-Lang vision tower arena + scratch.
    let mut vision_bytes = 0u64;
    if report.vision != MediaMode::Off && model.has_vision {
        if let Ok(inv) = StillTowerInventory::from_checkpoint(input.checkpoint(), 4096) {
            vision_bytes = inv.arena.bytes() as u64 + inv.scratch;
            items.push(Item::new(
                Category::Weights,
                "vision packed weight arena (PE-Lang 23L ViT + 2x2 merge projector)",
                "bf16",
                inv.arena.bytes() as u64,
                Basis::Exact,
            ));
            items.push(Item::new(
                Category::Workspace,
                "vision tower scratch (max 4096 image tokens)",
                "bf16/f32",
                inv.scratch,
                Basis::Formula,
            ));
        }
    }

    let sequences = opts.resolved_concurrency("inkling2_mini");
    let context_tokens = if opts.context_tokens == 0 {
        t.max_position as u64
    } else {
        opts.context_tokens
    };
    let context = Context::resolve(context_tokens, (opts.context_tokens > 0).then_some(opts.context_tokens));

    // 48-slot per-sequence state (40 local SWA-1024 append rings + 48 x 3-lag conv histories).
    let local_seq = match SequenceLayout::new(t, true, 1) {
        Ok(l) => l,
        Err(error) => {
            report.refuse_placement(format!("Inkling 2 Mini local sequence layout refused: {error:#}"), "Check window.");
            return empty(notes);
        }
    };
    let global_seq = match SequenceLayout::new(t, false, 1) {
        Ok(l) => l,
        Err(error) => {
            report.refuse_placement(format!("Inkling 2 Mini global sequence layout refused: {error:#}"), "Check conv.");
            return empty(notes);
        }
    };
    let mark_bytes_per_seq =
        cfg.local_cache_slots() as u64 * local_seq.bytes as u64
            + cfg.global_cache_slots().len() as u64 * global_seq.bytes as u64;
    let active_state_bytes = mark_bytes_per_seq * sequences;
    let prefix_entries = opts.prefix_entries("inkling2_mini");
    let mark_slots = cuteafd_core::prefix::mark_slots_with_host(
        sequences,
        prefix_entries,
        mark_bytes_per_seq,
        opts.mimo_prefix_mark_bytes,
        opts.host_cache_bytes != Some(0),
    );
    let retained_mark_bytes = mark_bytes_per_seq * mark_slots.saturating_sub(sequences);

    items.push(Item::new(
        Category::Kv,
        format!("active 48-slot SWA-1024 rings (40 slots) + conv lags (48 slots) x C{sequences}"),
        "bf16/f32",
        active_state_bytes,
        Basis::Exact,
    ));
    if retained_mark_bytes > 0 {
        items.push(Item::new(
            Category::Prefix,
            format!("retained 48-slot prefix marks ({} slots)", mark_slots.saturating_sub(sequences)),
            "bf16/f32",
            retained_mark_bytes,
            Basis::Formula,
        ));
    }

    // Unit + MoE scratch workspace estimate (prefill unit_rows x waves + decode 64 rows).
    let unit_rows = prefill.unit_rows;
    let scratch_bytes: u64 = (unit_rows * prefill.waves * (t.hidden as u64 * 4 * 24 + 24 * 1024 * 6))
        .max(256 * MIB);
    items.push(Item::new(
        Category::Workspace,
        format!("prefill ({unit_rows}x{}) + decode (64) + MoE execution workspaces", prefill.waves),
        "bf16/f32",
        scratch_bytes,
        Basis::Formula,
    ));

    // CUDA context + loaded code + #2697 soft-cap working-set graph reserve estimate.
    let cap = opts.rtx_bytes[0];
    let arch = ArchContext::for_device("sm_120", cap);
    let loaded_code = crate::placement::inventory::loaded_code_allowance(
        "inkling2_mini",
        "experts",
        false,
        0,
        arch,
        None,
    )
    .bytes;
    let graph_policy = match graph_reserve::current_working_set(
        0,
        WorkingSet {
            w1: 220 * MIB,
            step_growth: 16 * MIB,
            capture_growth: 16 * MIB,
        },
    ) {
        Ok(p) => p,
        Err(error) => {
            report.refuse_placement(format!("Inkling 2 Mini graph reserve refused: {error}"), "Check GPU_GRAPH_RESERVE.");
            return empty(notes);
        }
    };
    items.push(Item::new(
        Category::Runtime,
        "CUDA context and loaded modules",
        "",
        arch.context_bytes + loaded_code,
        Basis::Calibrated,
    ));
    items.push(Item::new(
        Category::Runtime,
        "graph reserve (#2697 working-set soft-cap forecast; calibrated at serve)",
        "",
        graph_policy.bytes,
        Basis::Formula,
    ));
    items.extend(crate::placement::reserve::gpu_item("inkling2_mini"));

    // Global paged KV pool (8 global slots: 5, 11, 17, 23, 29, 35, 41, 47).
    let global_slots = cfg.global_cache_slots().len() as u64;
    let record_bytes = opts.inkling_kv.record_bytes(t.kv_heads, t.head_dim) as u64;
    let page_bytes = PAGE_ROWS * record_bytes * global_slots;
    let fixed_used: u64 = items.iter().map(|i| i.bytes).sum();
    let kv_available = cap.saturating_sub(fixed_used);

    let mut facts = PoolFacts::new(context, opts.pool_tokens, PAGE_ROWS);
    facts.memory = vec![PoolMemory {
        available_bytes: kv_available,
        bytes_per_unit: page_bytes,
    }];
    facts.target = opts.target_pool_tokens;
    let pool = match crate::placement::prepare_pool(facts) {
        Ok(p) => p,
        Err(error) => {
            report.refuse_placement(format!("Inkling 2 Mini KV pool policy refused: {error}"), "Adjust KV min/target/max.");
            return empty(notes);
        }
    };
    let max_units_fitting = kv_available / page_bytes;
    let target_units = pool.target() / PAGE_ROWS;
    let min_units = pool.floor() / PAGE_ROWS;
    let max_units = pool.maximum_tokens() / PAGE_ROWS;
    if max_units_fitting < min_units {
        report.refuse_placement(
            format!(
                "Inkling 2 Mini KV pool does not fit minimum {} tokens ({:.2} GiB free for KV, needs {:.2} GiB)",
                pool.floor(),
                kv_available as f64 / GIB,
                (min_units * page_bytes) as f64 / GIB,
            ),
            "Lower --kv-min-tokens or --max-context-tokens, or use --kv-cache fp8.",
        );
        return empty(notes);
    }
    let admitted_units = max_units_fitting.min(target_units.max(min_units)).min(max_units);
    let pool_tokens = admitted_units * PAGE_ROWS;
    let pool_bytes = admitted_units * page_bytes;

    items.push(Item::new(
        Category::Kv,
        format!(
            "global KV pool ({global_slots} global slots {:?}, {}, {pool_tokens} tokens)",
            cfg.global_cache_slots(),
            opts.inkling_kv.name()
        ),
        opts.inkling_kv.name(),
        pool_bytes,
        Basis::Formula,
    ));

    notes.push(format!(
        "rtx0: target arena {:.2} GiB ({} views); routed experts {:.2} GiB ({} NVFP4 + {} MXFP8); \
         drafter {:.2} GiB; vision {:.2} GiB; 48-slot per-seq state {:.2} MiB/seq (active {:.2} GiB @ C{sequences}); \
         global KV pool ({global_slots} slots, {}) {pool_tokens} tokens ({:.2} GiB)",
        target_arena_bytes as f64 / GIB,
        exec_arena.views().len(),
        expert_total_bytes as f64 / GIB,
        nvfp4_layers.len(),
        mxfp8_layers.len(),
        drafter_bytes as f64 / GIB,
        vision_bytes as f64 / GIB,
        mark_bytes_per_seq as f64 / MIB as f64,
        active_state_bytes as f64 / GIB,
        opts.inkling_kv.name(),
        pool_bytes as f64 / GIB,
    ));

    let device = DeviceLayout {
        kind: DeviceKind::Rtx,
        index: 0,
        capacity_bytes: cap,
        worker_budget_bytes: 0,
        items,
        kv_tokens: pool_tokens,
    };

    MemoryLayout {
        devices: vec![device],
        pool_tokens,
        waste: Vec::new(),
        notes,
    }
}
