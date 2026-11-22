//! Hy4-flash `cuteafd plan --layout` over the shared contract planner (`Hy4Inputs`).
//!
//! Computes exact header-derived packed target arenas, RTX onboarding arenas
//! (whole on 1 GPU, TP2 768+768 halves on 2 GPUs), Spark per-rank expert slices,
//! 12-layer DSA paged KV + indexer group cache, 30-layer SWA-512 rings + exact
//! ~16.9 MiB prefix marks, and host Engram staging across 5090 / 1x RTX / 2x RTX.
use super::{ExpertStorage, Model};
use crate::families::hy4_flash::execution::{CacheLayout, INDEX_GROUP_RECORD_BYTES, PAGE_ROWS};
use crate::families::hy4_flash::weight_layout::{fp8_role, target_arenas};
use crate::families::hy4_flash::Hy4KvCache;
use crate::placement::families::hy4_flash::{
    self, Hy4Inputs, LocalBacking, LocalPackage, SparkCharge, DECODE_ROWS, HY4_FLASH, LANES,
};
use crate::placement::planner::{Baseline, Budget, Context, Device, Domain, Location, Operation, PrefillShape};
use crate::placement::{ArchContext, Onboard, PoolFacts, PoolMemory, ResidentWeights};
use crate::plan::{ExpertPlacement, PlanReport};
use crate::serving_capacity::WeightArenaLayout;
use cuteafd_core::memory_layout::{DeviceKind, DeviceLayout, MemoryLayout};

const GIB: f64 = (1u64 << 30) as f64;

fn empty(notes: Vec<String>) -> MemoryLayout {
    MemoryLayout { devices: Vec::new(), pool_tokens: 0, waste: Vec::new(), notes }
}

/// Header-derived EXL3 / BF16 resident bytes for one routed layer on one rank of `tp`.
fn layer_expert_slice_views(
    cfg: &crate::families::hy4_flash::Hy4FlashConfig,
    layer: usize,
    storage: ExpertStorage,
    tp: usize,
) -> anyhow::Result<Vec<(String, usize)>> {
    anyhow::ensure!(tp > 0 && cfg.moe_intermediate % tp == 0,
        "Hy4-flash moe_intermediate {} does not divide TP{tp}", cfg.moe_intermediate);
    let h = cfg.routed_expert_hidden;
    let slice = cfg.moe_intermediate / tp;
    let e = cfg.experts;
    let mut views = Vec::new();
    match storage {
        ExpertStorage::Exl3 { bits, .. } => {
            // Packed stacked EXL3 tensors per layer (trellis + suh + svh; mul1 stays on host).
            let gate_up_trellis = e * (slice * h * bits / 8);
            let down_trellis = e * (h * slice * bits / 8);
            views.push((format!("layer.{layer}.w13.trellis"), 2 * gate_up_trellis));
            views.push((format!("layer.{layer}.w13.suh"), 2 * e * h * 2));
            views.push((format!("layer.{layer}.w13.svh"), 2 * e * slice * 2));
            views.push((format!("layer.{layer}.w2.trellis"), down_trellis));
            views.push((format!("layer.{layer}.w2.suh"), e * slice * 2));
            views.push((format!("layer.{layer}.w2.svh"), e * h * 2));
        }
        ExpertStorage::Bf16 => {
            views.push((format!("layer.{layer}.w13.weight"), 2 * e * slice * h * 2));
            views.push((format!("layer.{layer}.w2.weight"), e * h * slice * 2));
        }
    }
    Ok(views)
}

/// Build packed RTX onboarding arenas for every prefix count `0..=routed.len()`.
fn build_local_backings(
    model: &Model,
    gpus: usize,
    prefill_rows: usize,
) -> anyhow::Result<LocalPackage> {
    let routed = model.routed();
    let mut backings = Vec::with_capacity(routed.len() + 1);
    backings.push(Vec::new()); // count == 0
    for count in 1..=routed.len() {
        let prefix = routed[..count].to_vec();
        let mut group = Vec::with_capacity(gpus);
        for gpu in 0..gpus {
            let mut views = Vec::new();
            for &layer in &prefix {
                let storage = model.expert_storage[&layer];
                views.extend(layer_expert_slice_views(&model.cfg, layer, storage, gpus)?);
            }
            let arena = WeightArenaLayout::new(format!("hy4_flash/experts/gpu{gpu}"), views)?;
            let slice = model.cfg.moe_intermediate / gpus;
            let rows = prefill_rows.max(DECODE_ROWS as usize);
            let ws_bytes = (rows * model.cfg.topk * slice * 2 * 2) as u64;
            let tp2_rows = if gpus == 2 {
                Some(crate::serving_capacity::OwnerTp2Rows::new(
                    model.cfg.routed_expert_hidden as u64,
                    model.cfg.topk as u64,
                    rows as u64,
                    true,
                )?)
            } else {
                None
            };
            group.push(LocalBacking {
                gpu,
                layers: prefix.clone(),
                arena,
                workspace: vec![("exl3_intermediate".into(), ws_bytes)],
                tp2_rows,
            });
        }
        backings.push(group);
    }
    Ok(LocalPackage::Installed { backings })
}

pub(super) fn layout(
    model: &Model,
    input: &crate::plan::PlanInput,
    report: &mut PlanReport,
    opts: &mut crate::plan::layout::LayoutOptions,
    requested: ExpertPlacement,
) -> MemoryLayout {
    let cfg = &model.cfg;
    let gpus = opts.rtx_bytes.len();
    let ranks = requested.spark_ranks();
    report.bytes_by_owner.clear();

    let mut notes = vec![
        "Hy4-flash shared contract: header-exact packed arenas over named physical owners (#2478)".to_string(),
    ];
    if !matches!(gpus, 1 | 2) {
        report.refuse_placement(
            format!("Hy4-flash needs one or two coordinator GPUs, not {gpus}"),
            "Use --rtx 1 or --rtx 2.",
        );
        return empty(notes);
    }
    if opts.host_embedding {
        report.refuse_placement(
            "Hy4-flash keeps embed_tokens on coordinator GPU; --embedding-placement host is unsupported".into(),
            "Use --embedding-placement gpu.",
        );
        return empty(notes);
    }
    if let Err(error) = input.checkpoint().validate_fp8_roles("hy4_flash", fp8_role) {
        report.refuse_placement(
            format!("Hy4-flash fp8-compact validation refused: {error:#}"),
            "Ensure every fp8-compact core weight is a BF16 matrix with whole 128x128 blocks.",
        );
        return empty(notes);
    }

    let limits = crate::plan::Family::prefill_limits(&super::HY4_FLASH);
    let prefill = match crate::plan::layout::resolve_prefill(input, report, opts, ranks, limits) {
        Ok(prefill) => prefill,
        Err(error) => return crate::plan::layout::unsupported_prefill(report, error),
    };
    report.prefill_reserve_rows = Some(prefill.span_rows);
    let shape = PrefillShape { rows: prefill.unit_rows, waves: prefill.waves };

    let attention = match HY4_FLASH.check_attention(report.requested_attention, gpus, gpus == 2 && opts.head_split) {
        Ok(mode) => mode,
        Err(error) => {
            report.refuse_placement(
                format!("Hy4-flash attention placement refused: {error}"),
                "Use ATTENTION_PLACEMENT=auto, heads, context, or layers.",
            );
            return empty(notes);
        }
    };
    report.attention_placement = attention;
    let modes = hy4_flash::layer_modes(cfg, gpus, attention);

    let targets = match target_arenas(input.checkpoint(), cfg, &model.core, &modes, gpus) {
        Ok(targets) => targets,
        Err(error) => {
            report.refuse_placement(
                format!("Hy4-flash packed target arena refused: {error:#}"),
                "Check checkpoint core headers and selected MC view.",
            );
            return empty(notes);
        }
    };

    let kv_format = match opts.mla_kv {
        cuteafd_core::MlaCacheFormat::Bf16 => Hy4KvCache::Bf16,
        cuteafd_core::MlaCacheFormat::Fp8 => Hy4KvCache::Fp8,
    };
    let sequences = opts.resolved_concurrency("hy4_flash");
    let context_tokens = if opts.context_tokens == 0 {
        cfg.max_position as u64
    } else {
        opts.context_tokens
    };
    let context = Context::resolve(context_tokens, (opts.context_tokens > 0).then_some(opts.context_tokens));

    let page_rows = if modes.iter().any(|m| *m == crate::placement::LayerMode::ContextSplit) {
        2 * PAGE_ROWS as u64
    } else {
        PAGE_ROWS as u64
    };
    let dsa_layers = cfg.sparse_layers().len() as u64;
    let bytes_per_dsa_page = dsa_layers
        * (PAGE_ROWS as u64 * kv_format.record_bytes() as u64
            + (PAGE_ROWS / cfg.indexer_compress_ratio) as u64 * INDEX_GROUP_RECORD_BYTES as u64);
    let mut facts = PoolFacts::new(context, opts.pool_tokens.filter(|&n| n > 0), PAGE_ROWS as u64);
    facts.memory = vec![PoolMemory {
        available_bytes: opts.rtx_bytes[..gpus].iter().sum(),
        bytes_per_unit: bytes_per_dsa_page,
    }];
    facts.target = opts.target_pool_tokens;
    let pool = match crate::placement::prepare_pool(facts) {
        Ok(pool) => pool,
        Err(error) => {
            report.refuse_placement(
                format!("Hy4-flash pool policy refused: {error}"),
                "Supply consistent KV target, min and max bounds.",
            );
            return empty(notes);
        }
    };

    let routed = model.routed();
    let local = match build_local_backings(model, gpus, shape.rows as usize) {
        Ok(local) => local,
        Err(error) => {
            report.refuse_placement(
                format!("Hy4-flash local expert backing refused: {error:#}"),
                "Check routed expert dimensions and TP divisibility.",
            );
            return empty(notes);
        }
    };

    // Per-prefix Spark charges (for 0..=routed.len() onboarded layers).
    let spark_runtime = cuteafd_core::memory_layout::spark_runtime_reserve_bytes(0)
        .expect("zero-arena reserve cannot overflow");
    let mut spark = Vec::with_capacity(routed.len() + 1);
    let spark_ring_bytes = if ranks > 0 {
        let rings = crate::plan::layout::spark_ring_geometry(LANES as usize);
        cuteafd_core::SparkEndpoint {
            capacity: shape.rows as u32,
            ingress: cuteafd_core::SparkIngress::Fp8K32,
            ring_depth: None,
            partial_reply_bytes: None,
        }
        .ring_bytes(
            cuteafd_core::ExpertGeometry::HY4_FLASH,
            cuteafd_core::SparkRings {
                max_frame_bytes: rings.max_frame_bytes,
                depth: rings.depth,
                slot_bytes: rings.slot_bytes,
                alignment: rings.alignment,
            },
        )
        .unwrap_or(0) as u64
    } else {
        0
    };
    let spark_ws = (shape.rows * cfg.topk as u64 * cfg.moe_intermediate as u64 * 4) / ranks.max(1) as u64;
    for count in 0..=routed.len() {
        let mut per_rank = Vec::with_capacity(ranks);
        let active = count < routed.len();
        if ranks > 0 {
            let mut rank_expert_bytes = 0u64;
            if active {
                for &layer in &routed[count..] {
                    let views = match layer_expert_slice_views(cfg, layer, model.expert_storage[&layer], ranks) {
                        Ok(v) => v,
                        Err(error) => {
                            report.refuse_placement(
                                format!("Hy4-flash Spark TP{ranks} slice refused: {error:#}"),
                                "Hy4-flash intermediate 1536 divides TP2, TP3, TP4, or TP6 (128-wide blocks).",
                            );
                            return empty(notes);
                        }
                    };
                    let arena = match WeightArenaLayout::new(format!("hy4_flash/spark/layer{layer}"), views) {
                        Ok(a) => a,
                        Err(error) => {
                            report.refuse_placement(format!("Hy4-flash Spark arena overflow: {error:#}"), "Check expert headers.");
                            return empty(notes);
                        }
                    };
                    rank_expert_bytes = rank_expert_bytes.saturating_add(arena.bytes() as u64);
                }
            }
            for _rank in 0..ranks {
                per_rank.push(SparkCharge {
                    experts: rank_expert_bytes,
                    owners: if active { spark_ring_bytes + spark_ws } else { 0 },
                    runtime: if active { spark_runtime } else { 0 },
                });
            }
        }
        spark.push(per_rank);
    }

    let head_gpu = if modes.iter().all(|m| matches!(m, crate::placement::LayerMode::HeadSplit | crate::placement::LayerMode::ContextSplit)) {
        0u8
    } else {
        (gpus - 1) as u8
    };
    let links = match hy4_flash::spark_link(head_gpu, ranks, shape, opts.spark_reply_formats) {
        Ok(links) => links,
        Err(error) => {
            report.refuse_placement(
                format!("Hy4-flash Spark transport refused: {error}"),
                "Use supported Spark rank count and prefill shape.",
            );
            return empty(notes);
        }
    };

    let (host_total, host_available) = match crate::plan::layout::planner_host_memory(
        opts.host_memory,
        crate::plan::layout::HostMemorySample::Raw,
    ) {
        Ok(host) => host,
        Err(error) => {
            report.refuse_placement(
                format!("Hy4-flash host inventory refused: {error:#}"),
                "Provide a host memory sample.",
            );
            return empty(notes);
        }
    };

    // Exact prefix mark count and host prefix quota.
    let cache0 = match CacheLayout::new(cfg, kv_format, sequences as usize, cfg.sparse_layers().len(), cfg.swa_layers().len()) {
        Ok(c) => c,
        Err(error) => {
            report.refuse_placement(format!("Hy4-flash cache layout refused: {error:#}"), "Check concurrency and config.");
            return empty(notes);
        }
    };
    let prefix_mark_slots = opts.prefix_slots.unwrap_or_else(|| {
        cuteafd_core::prefix::mark_slots_with_host(
            sequences,
            opts.prefix_entries("hy4_flash"),
            cache0.mark_bytes_per_slot as u64,
            opts.mimo_prefix_mark_bytes,
            opts.host_cache_bytes != Some(0),
        )
    }) as usize;

    let host_chunk_bytes = {
        let geom = cuteafd_core::prefix::HostCacheGeometry {
            unit_rows: page_rows,
            page_bytes: bytes_per_dsa_page,
            mark_bytes: cache0.mark_bytes_per_slot as u64,
            draft_bytes: 0,
        };
        geom.chunk_bytes()
    };
    let host_quota = if opts.host_cache_bytes == Some(0) || opts.prefix_entries("hy4_flash") == 0 {
        0u64
    } else {
        let geom = cuteafd_core::prefix::HostCacheGeometry {
            unit_rows: page_rows,
            page_bytes: bytes_per_dsa_page,
            mark_bytes: cache0.mark_bytes_per_slot as u64,
            draft_bytes: 0,
        };
        match geom.resolve(opts.host_cache_bytes, context.maximum, sequences, host_total, host_chunk_bytes) {
            Some(q) => q,
            None => {
                report.refuse_placement(
                    "Hy4-flash host prefix quota overflow".into(),
                    "Disable host retention or supply a supported quota.",
                );
                return empty(notes);
            }
        }
    };
    let host_prefix = crate::placement::planner::serving::HostPrefixOwners {
        quota_bytes: host_quota,
        chunk_bytes: host_chunk_bytes.min(host_quota.max(1)),
    };

    // Host-mapped Engram tables charged strictly to raptor host RAM (never GPU memory).
    let engram_table_bytes = if model.has_engram_fp8 {
        input.checkpoint().view.as_ref().and_then(|v| v.engram_fp8())
            .map_or(0, |e| e.total_bytes().unwrap_or(0))
    } else {
        cfg.engram_layers.iter().filter_map(|&l| cfg.engram_rows(l)).sum::<u64>()
            * cfg.engram_head_width() as u64 * 2
    };
    // Engram host pinned staging + row cache (bounded, 4 layers × 16 heads × 96 × 2B × rows).
    let engram_host_bytes = (64u64 << 20) + shape.rows * cfg.engram_concat_width() as u64 * 2 * 4;

    let mut budgets: Vec<Budget> = Vec::with_capacity(gpus + ranks + 1);
    let mut code = Vec::with_capacity(gpus);
    let mut graph_reserve = Vec::with_capacity(gpus);
    for (gpu, &cap) in opts.rtx_bytes.iter().enumerate() {
        let arch = ArchContext::for_device("sm_120", cap);
        let reserve = crate::placement::reserve::gpu("hy4_flash");
        budgets.push(Budget {
            id: format!("rtx{gpu}"),
            locations: vec![Location::coordinator(gpu as u8)],
            physical_bytes: cap,
            logical_bytes: cap,
            baseline: Baseline::Forecast {
                occupied: arch.context_bytes,
                source: "CUDA context and modules on the coordinator GPU".into(),
                assumed: true,
            },
            headroom: reserve,
        });
        code.push(crate::placement::inventory::loaded_code_allowance(
            "hy4_flash",
            "exl3",
            gpus == 2,
            gpu as u8,
            arch,
            None,
        ).bytes);
        let gr = match crate::placement::graph_reserve::resolve(None, gpu, false, 0, 0) {
            Ok(resolved) => resolved.bytes,
            Err(error) => {
                report.refuse_placement(format!("Hy4-flash graph reserve refused: {error}"), "Check GPU_GRAPH_RESERVE.");
                return empty(notes);
            }
        };
        graph_reserve.push(gr);
    }
    for rank in 0..ranks {
        let cap = opts.spark_capacity(rank);
        budgets.push(Budget {
            id: format!("spark{rank}"),
            locations: vec![Location { device: Device::Spark(rank as u8), domain: Domain::Managed }],
            physical_bytes: cap,
            logical_bytes: cap,
            baseline: Baseline::Forecast {
                occupied: 0,
                source: "Spark worker arena owned by the serving process".into(),
                assumed: false,
            },
            headroom: crate::placement::reserve::spark(),
        });
    }
    budgets.push(Budget {
        id: "host".into(),
        locations: vec![Location::HOST, Location::PINNED],
        physical_bytes: host_total,
        logical_bytes: host_total,
        baseline: Baseline::Forecast {
            occupied: host_total.saturating_sub(host_available),
            source: "host memory occupied outside cuteafd".into(),
            assumed: false,
        },
        headroom: (8 << 30).max(host_total / 10),
    });

    let inputs = Hy4Inputs {
        cfg: cfg.clone(),
        kv_format,
        gpus,
        spark_ranks: ranks,
        context,
        pool,
        prefill: shape,
        sequences,
        prefix_mark_slots,
        targets,
        placement: attention,
        modes,
        routed: routed.clone(),
        local,
        spark,
        links,
        budgets,
        code,
        graph_reserve,
        host_prefix,
        engram_table_bytes,
        engram_host_bytes,
        resident_weight_limit: Some(ResidentWeights(opts.weight_budget_bytes)),
    };

    let onboard = opts.resolve_onboard(routed.len()).unwrap_or(Onboard::Auto);
    let solve_req = hy4_flash::request(
        inputs.context,
        opts.pool_tokens,
        onboard,
        shape,
        Operation::Serving,
        inputs.resident_weight_limit,
        opts.spark_reply_formats,
    );
    let admitted = match inputs.admit(solve_req) {
        Ok(admitted) => admitted,
        Err(error) => {
            report.refuse_placement(
                format!("Hy4-flash joint admission refused: {error}"),
                "Use more coordinator memory, a compact core view (k8-compact/k6-compact), or 2+ Spark ranks.",
            );
            return empty(notes);
        }
    };

    report.phase_ledger = Some(crate::plan::SolvedPhaseLedger::from(&admitted.solved));
    report.host_budget_bytes = host_quota;
    report.host_ledger = match admitted.host_ledger(host_quota) {
        Ok(ledger) => Some(ledger),
        Err(error) => {
            report.refuse_placement(format!("Hy4-flash host ledger refused: {error}"), "Check host RAM budget.");
            return empty(notes);
        }
    };
    let placed = admitted.placement;
    let onboarded = placed.onboard_layers;
    report.bytes_by_owner.insert(
        "coordinator target arenas".into(),
        inputs.targets.iter().map(|t| t.arena.bytes() as u64).sum(),
    );
    report.bytes_by_owner.insert("routed expert weights".into(), model.routed_bytes);
    report.bytes_by_owner.insert("host-mapped engram tables".into(), engram_table_bytes);

    notes.push(format!(
        "Hy4-flash attention {}: pool {} tokens (12 DSA layers paged, 30 SWA-512 fixed per-sequence rings); \
         {} of {} routed layers onboarded ({}); Engram {}; unit {} x {} waves; C{}",
        placed.attention_summary(),
        placed.pool_tokens,
        onboarded,
        routed.len(),
        if gpus == 2 { "TP2 768+768" } else { "whole 1536" },
        if model.has_engram_fp8 { "engram_fp8 (61.5 GiB host-mapped)" } else { "BF16 (118.0 GiB host-mapped)" },
        shape.rows,
        shape.waves,
        sequences
    ));

    let mut devices = Vec::new();
    for gpu in 0..gpus {
        let mut device = DeviceLayout {
            kind: DeviceKind::Rtx,
            index: gpu as u32,
            capacity_bytes: opts.rtx_bytes[gpu],
            worker_budget_bytes: 0,
            items: placed.items[gpu].clone(),
            kv_tokens: placed.pool_tokens,
        };
        device.items.extend(crate::placement::reserve::gpu_item("hy4_flash"));
        notes.push(format!(
            "rtx{gpu}: target arena {:.2} GiB ({} views)",
            inputs.targets[gpu].arena.bytes() as f64 / GIB,
            inputs.targets[gpu].arena.views().len()
        ));
        devices.push(device);
    }
    let mut widest = 0u64;
    for rank in 0..ranks {
        let charge = inputs.spark[onboarded][rank];
        widest = widest.max(charge.experts);
        let location = Location { device: Device::Spark(rank as u8), domain: Domain::Managed };
        let items = match admitted.solved.items(location) {
            Ok(items) => items,
            Err(error) => {
                report.refuse_placement(format!("Hy4-flash Spark ledger refused: {error}"), "Check Spark budgets.");
                return empty(notes);
            }
        };
        devices.push(DeviceLayout {
            kind: DeviceKind::Spark,
            index: rank as u32,
            capacity_bytes: opts.spark_capacity(rank),
            worker_budget_bytes: opts.spark_capacity(rank),
            kv_tokens: 0,
            items,
        });
    }
    report.spark_rank_share = if model.routed_bytes == 0 {
        0.0
    } else {
        widest as f64 / model.routed_bytes as f64
    };
    if ranks > 0 && onboarded < routed.len() {
        notes.push(format!(
            "Spark: TP{ranks} slices of routed layers {}..={}; widest rank {:.2} GiB",
            routed[onboarded],
            cfg.layers - 1,
            widest as f64 / GIB
        ));
    }
    report.solver_placement = Some(placed);
    MemoryLayout {
        devices,
        pool_tokens: report.solver_placement.as_ref().map_or(0, |p| p.pool_tokens),
        waste: Vec::new(),
        notes,
    }
}
