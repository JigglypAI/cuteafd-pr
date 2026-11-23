//! Hy4 (`hy_v4_vl`) `cuteafd plan --layout` over the shared contract planner (`Hy4Inputs`).
//!
//! Uses the shared expert catalog (`OfficialV41Catalog`), shared RTX onboarding
//! (`rtx_onboard::expert_workspace_requests` + `V41Exl3Manifest::residency`),
//! and shared Spark I/O & execution owner admission (`SparkWorkingSet::rank_endpoints_io`,
//! `spark_executions::admit_family`, `spark_slots::catalog_owner_bytes`,
//! `exl3_spark_arena_bytes`). Pages-only cache (78 MLA layers + 21 full IndexCache
//! layers; zero SWA rings, zero prefix marks, zero Engram).
use super::Model;
use crate::families::hy4::execution::INDEX_PAGE_BYTES;
use crate::families::hy4::weight_layout::{fp8_role, target_arenas};
use crate::families::hy4::Hy4KvCache;
use crate::families::hy4_common::PAGE_ROWS;
use crate::placement::families::hy4::{
    self, Hy4Inputs, LocalBacking, LocalPackage, SparkCharge, HY4, LANES,
};
use crate::placement::planner::{Baseline, Budget, Context, Device, Domain, Location, Operation, PrefillShape};
use crate::placement::{ArchContext, LayerMode, Onboard, PoolFacts, PoolMemory, ResidentWeights, SparkWorkingSet};
use crate::plan::{ExpertPlacement, PlanReport};
use crate::serving_capacity::{PackedWeightRegions, WeightArenaLayout};
use anyhow::Context as _;
use cuteafd_core::memory_layout::{DeviceKind, DeviceLayout, MemoryLayout};

const GIB: f64 = (1u64 << 30) as f64;

fn empty(notes: Vec<String>) -> MemoryLayout {
    MemoryLayout {
        devices: Vec::new(),
        pool_tokens: 0,
        waste: Vec::new(),
        notes,
    }
}

fn build_local_backings(
    catalog: &crate::OfficialV41Catalog,
    manifest: Option<&std::path::Path>,
    gpus: usize,
    rows: u64,
) -> LocalPackage {
    let workspace = || -> anyhow::Result<LocalPackage> {
        anyhow::ensure!((1..=2).contains(&gpus), "Hy4 local package requires one or two ranks");
        let exl3 = catalog
            .exl3()
            .context("Hy4 RTX onboarding requires an EXL3 MUL1 expert catalog")?;
        let routed = catalog.routed_experts();
        let layers: Vec<usize> = (routed.first_layer..routed.layers).collect();
        let requests = crate::placement::rtx_onboard::expert_workspace_requests(
            catalog,
            manifest,
            rows,
            gpus,
            "hy4",
        )?;
        let exl3_regions: Vec<Vec<(String, usize)>> = (0..gpus)
            .map(|gpu| {
                layers
                    .iter()
                    .map(|&layer| {
                        let layout = exl3.residency(crate::V41Exl3Layer::Backbone(layer), gpus, gpu)?;
                        let name = if gpus == 1 {
                            format!("BackboneFull {{ layer: {layer} }}.0")
                        } else {
                            format!("BackboneTp2 {{ layer: {layer}, rank: {gpu} }}.0")
                        };
                        Ok((name, layout.device_arena_layout()?.1))
                    })
                    .collect::<anyhow::Result<Vec<_>>>()
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let tp2_rows = if gpus == 2 {
            Some(crate::serving_capacity::OwnerTp2Rows::with_partial(
                routed.hidden as u64,
                routed.topk as u64,
                rows,
                true,
                4,
            )?)
        } else {
            None
        };
        let last = requests
            .len()
            .checked_sub(1)
            .context("Hy4 EXL3 output owner absent")?;
        let mut backings = Vec::with_capacity(layers.len() + 1);
        backings.push(Vec::new()); // count == 0
        for count in 1..=layers.len() {
            let selected = layers[..count].to_vec();
            let mut owners = Vec::with_capacity(gpus);
            for gpu in 0..gpus {
                let regions = exl3_regions[gpu][..count].to_vec();
                let id = crate::serving_capacity::expert_weight_arena_id("exl3", gpu as i32, &regions)?;
                let workspace = requests
                    .iter()
                    .enumerate()
                    .map(|(index, &bytes)| {
                        (
                            if index == last {
                                "output".into()
                            } else {
                                format!("scratch.{index}")
                            },
                            bytes,
                        )
                    })
                    .collect();
                owners.push(LocalBacking {
                    gpu,
                    layers: selected.clone(),
                    arena: WeightArenaLayout::new(id, regions)?,
                    workspace,
                    tp2_rows,
                });
            }
            backings.push(owners);
        }
        Ok(LocalPackage::Installed { backings })
    };
    workspace().unwrap_or_else(|error| LocalPackage::Missing(format!("{error:#}")))
}

fn build_spark_charges(
    catalog: &crate::OfficialV41Catalog,
    routed: &[usize],
    ranks: usize,
    prefill: PrefillShape,
) -> anyhow::Result<Vec<Vec<SparkCharge>>> {
    if ranks == 0 {
        return Ok(vec![Vec::new(); routed.len() + 1]);
    }
    let backend = crate::plan::spark_executions::SparkBackend::for_catalog(catalog)?;
    let executions = crate::plan::spark_executions::admit_family(
        "hy4",
        &backend.exports(),
        None,
        prefill.rows,
        prefill.waves,
    )?;
    let endpoint = cuteafd_core::SparkEndpoint {
        ring_depth: None,
        capacity: u32::try_from(prefill.rows)?,
        ingress: cuteafd_core::SparkIngress::Fp8K32,
        partial_reply_bytes: None,
    };
    let rings = crate::plan::layout::spark_ring_geometry(LANES as usize);
    let rings = cuteafd_core::SparkRings {
        max_frame_bytes: rings.max_frame_bytes,
        depth: rings.depth,
        slot_bytes: rings.slot_bytes,
        alignment: rings.alignment,
    };
    let runtime = cuteafd_core::memory_layout::spark_runtime_reserve_bytes(0)
        .context("Spark runtime residual")?;
    let geometry = cuteafd_core::ExpertGeometry::HY4;

    (0..=routed.len())
        .map(|count| {
            let working = SparkWorkingSet {
                rtx_layers: routed[..count].to_vec(),
                spark_layers: routed[count..].to_vec(),
                spark_ranks: ranks,
                prefill_lanes: LANES,
            };
            (0..ranks)
                .map(|rank| {
                    if count == routed.len() {
                        return Ok(SparkCharge {
                            experts: 0,
                            owners: 0,
                            runtime: 0,
                        });
                    }
                    let io = working.rank_endpoints_io(rank, geometry, &[endpoint], rings)?;
                    let owner_ws = executions
                        .workspace_bytes(|extent| {
                            crate::plan::layout::spark_slots::catalog_owner_bytes(
                                catalog, "hy4", &working, rank, extent,
                            )
                        })?
                        .and_then(|bytes| bytes.total())
                        .context("Hy4 Spark owner sum overflow")?;
                    let owners = [
                        io.intake,
                        io.host_exchange,
                        io.row_indices,
                        io.mapped_rings,
                        io.endpoint_staging,
                        owner_ws,
                    ]
                    .into_iter()
                    .try_fold(0u64, |sum, bytes| sum.checked_add(bytes))
                    .context("Hy4 Spark owners overflow")?;
                    let regions = working
                        .spark_layers
                        .iter()
                        .map(|&layer| catalog.exl3_spark_arena_bytes(layer, ranks).map(|b| b[rank]))
                        .collect::<anyhow::Result<Vec<_>>>()?;
                    let resident = PackedWeightRegions::new(
                        regions
                            .into_iter()
                            .map(usize::try_from)
                            .collect::<Result<Vec<_>, _>>()?,
                    )?
                    .bytes as u64;
                    Ok(SparkCharge {
                        experts: resident,
                        owners,
                        runtime,
                    })
                })
                .collect()
        })
        .collect()
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
        "Hy4 shared contract: header-exact packed arenas over named physical owners (#2550)".to_string(),
    ];
    if !matches!(gpus, 1 | 2) {
        report.refuse_placement(
            format!("Hy4 needs one or two coordinator GPUs, not {gpus}"),
            "Use --rtx 1 or --rtx 2.",
        );
        return empty(notes);
    }
    if opts.host_embedding {
        report.refuse_placement(
            "Hy4 keeps embed_tokens on coordinator GPU; --embedding-placement host is unsupported".into(),
            "Use --embedding-placement gpu.",
        );
        return empty(notes);
    }
    if let Err(error) = input.checkpoint().validate_fp8_roles_where(
        "hy4",
        |name| !name.ends_with(".self_attn.kv_b_proj.weight"),
        fp8_role,
    ) {
        report.refuse_placement(
            format!("Hy4 fp8-compact validation refused: {error:#}"),
            "Ensure every fp8-compact core weight is a BF16 matrix with whole 128x128 blocks.",
        );
        return empty(notes);
    }

    let attention = match HY4.check_attention(report.requested_attention, gpus, gpus == 2 && opts.head_split) {
        Ok(mode) => mode,
        Err(error) => {
            report.refuse_placement(
                format!("Hy4 attention placement refused: {error}"),
                "Use ATTENTION_PLACEMENT=auto, context, layers, or heads.",
            );
            return empty(notes);
        }
    };
    report.attention_placement = attention;
    let modes = hy4::layer_modes(cfg, gpus, attention);

    let targets = match target_arenas(input.checkpoint(), cfg, &model.core, &modes, gpus) {
        Ok(targets) => targets,
        Err(error) => {
            report.refuse_placement(
                format!("Hy4 packed target arena refused: {error:#}"),
                "Check checkpoint core headers and selected MC view.",
            );
            return empty(notes);
        }
    };
    report.bytes_by_owner.insert(
        "coordinator target arenas".into(),
        targets.iter().map(|t| t.arena.bytes() as u64).sum(),
    );
    report.bytes_by_owner.insert("routed expert weights".into(), model.routed_bytes);

    let catalog = match input.expert_catalog() {
        Ok(catalog) => catalog,
        Err(error) => {
            report.refuse_placement(
                format!("Hy4 expert catalog refused: {error:#}"),
                "Use an EXL3 K4 MUL1 Hy4 MC checkpoint (wrldsuksgo2mars/Hy4-EXL3-K4-MC); BF16 fused source experts have no Spark/RTX serving package.",
            );
            return empty(notes);
        }
    };

    let limits = crate::plan::Family::prefill_limits(&super::HY4);
    let prefill = match crate::plan::layout::resolve_prefill(input, report, opts, ranks, limits) {
        Ok(prefill) => prefill,
        Err(error) => return crate::plan::layout::unsupported_prefill(report, error),
    };
    report.prefill_reserve_rows = Some(prefill.span_rows);
    let shape = PrefillShape {
        rows: prefill.unit_rows,
        waves: prefill.waves,
    };

    let kv_format = match opts.mla_kv {
        cuteafd_core::MlaCacheFormat::Bf16 => Hy4KvCache::Bf16,
        cuteafd_core::MlaCacheFormat::Fp8 => Hy4KvCache::Fp8,
    };
    let sequences = opts.resolved_concurrency("hy4");
    let context_tokens = if opts.context_tokens == 0 {
        cfg.max_position as u64
    } else {
        opts.context_tokens
    };
    let context = Context::resolve(context_tokens, (opts.context_tokens > 0).then_some(opts.context_tokens));

    let page_rows = if modes.iter().any(|m| *m == LayerMode::ContextSplit) {
        2 * PAGE_ROWS as u64
    } else {
        PAGE_ROWS as u64
    };
    let kv_page_bytes = PAGE_ROWS as u64 * kv_format.record_bytes() as u64;
    let idx_page_bytes = INDEX_PAGE_BYTES as u64;
    let bytes_per_page_on = |gpu: usize| -> u64 {
        let mut owned_kv = 0u64;
        let mut owned_full_idx = 0u64;
        for (layer, &mode) in modes.iter().enumerate() {
            let owns = match mode {
                LayerMode::Whole { gpu: owner, .. } => usize::from(owner) == gpu,
                LayerMode::HeadSplit | LayerMode::ContextSplit => true,
            };
            if !owns {
                continue;
            }
            owned_kv += 1;
            if cfg.is_full_indexer(layer) {
                owned_full_idx += 1;
            }
        }
        owned_kv * kv_page_bytes + owned_full_idx * idx_page_bytes
    };
    let bytes_per_page_total: u64 = (0..gpus).map(bytes_per_page_on).sum();
    let mut facts = PoolFacts::new(context, opts.pool_tokens.filter(|&n| n > 0), page_rows);
    facts.memory = (0..gpus)
        .map(|gpu| PoolMemory {
            available_bytes: opts.rtx_bytes[gpu],
            bytes_per_unit: bytes_per_page_on(gpu),
        })
        .collect();
    facts.target = opts.target_pool_tokens;
    let pool = match crate::placement::prepare_pool(facts) {
        Ok(pool) => pool,
        Err(error) => {
            report.refuse_placement(
                format!("Hy4 pool policy refused: {error}"),
                "Supply consistent KV target, min and max bounds.",
            );
            return empty(notes);
        }
    };

    let routed = model.routed();
    let manifest_path = opts.workspace_manifest.clone().or_else(|| {
        let path = std::path::PathBuf::from("/opt/cuteafd/share/PROGRAMS.json");
        path.is_file().then_some(path)
    });
    let local = build_local_backings(catalog, manifest_path.as_deref(), gpus, shape.rows);
    if let LocalPackage::Missing(reason) = &local {
        notes.push(format!("Hy4 RTX expert onboarding unavailable: {reason}"));
    }

    let spark = match build_spark_charges(catalog, &routed, ranks, shape) {
        Ok(spark) => spark,
        Err(error) => {
            report.refuse_placement(
                format!("Hy4 Spark charge census refused: {error:#}"),
                "Use a supported Spark world (TP3..=TP6) with an EXL3 K4 MUL1 checkpoint.",
            );
            return empty(notes);
        }
    };

    let head_gpu = if modes
        .iter()
        .all(|m| matches!(m, LayerMode::HeadSplit | LayerMode::ContextSplit))
    {
        0u8
    } else {
        (gpus - 1) as u8
    };
    let links = match hy4::spark_link(head_gpu, ranks, shape, opts.spark_reply_formats) {
        Ok(links) => links,
        Err(error) => {
            report.refuse_placement(
                format!("Hy4 Spark transport refused: {error}"),
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
                format!("Hy4 host inventory refused: {error:#}"),
                "Provide a host memory sample.",
            );
            return empty(notes);
        }
    };

    // Pages-only host prefix cache (zero per-slot device mark bytes).
    let host_chunk_bytes = {
        let geom = cuteafd_core::prefix::HostCacheGeometry {
            unit_rows: page_rows,
            page_bytes: bytes_per_page_total,
            mark_bytes: 0,
            draft_bytes: 0,
        };
        geom.chunk_bytes()
    };
    let host_quota = if opts.host_cache_bytes == Some(0) || opts.prefix_entries("hy4") == 0 {
        0u64
    } else {
        let geom = cuteafd_core::prefix::HostCacheGeometry {
            unit_rows: page_rows,
            page_bytes: bytes_per_page_total,
            mark_bytes: 0,
            draft_bytes: 0,
        };
        match geom.resolve(
            opts.host_cache_bytes,
            context.maximum,
            sequences,
            host_total,
            host_chunk_bytes,
        ) {
            Some(q) => q,
            None => {
                report.refuse_placement(
                    "Hy4 host prefix quota overflow".into(),
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
    let host_headroom = (8u64 << 30).max(host_total / 10);

    let mut budgets: Vec<Budget> = Vec::with_capacity(gpus + ranks + 1);
    let mut code = Vec::with_capacity(gpus);
    let mut graph_reserve = Vec::with_capacity(gpus);
    for (gpu, &cap) in opts.rtx_bytes.iter().enumerate() {
        let arch = ArchContext::for_device("sm_120", cap);
        let reserve = crate::placement::reserve::gpu("hy4");
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
        code.push(
            crate::placement::inventory::loaded_code_allowance(
                "hy4",
                "exl3",
                gpus == 2,
                gpu as u8,
                arch,
                None,
            )
            .bytes,
        );
        let gr = match crate::placement::graph_reserve::resolve(None, gpu, false, 0, 0) {
            Ok(resolved) => resolved.bytes,
            Err(error) => {
                report.refuse_placement(
                    format!("Hy4 graph reserve refused: {error}"),
                    "Check GPU_GRAPH_RESERVE.",
                );
                return empty(notes);
            }
        };
        graph_reserve.push(gr);
    }
    for rank in 0..ranks {
        let cap = opts.spark_capacity(rank);
        budgets.push(Budget {
            id: format!("spark{rank}"),
            locations: vec![Location {
                device: Device::Spark(rank as u8),
                domain: Domain::Managed,
            }],
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
        headroom: host_headroom,
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
        resident_weight_limit: Some(ResidentWeights(opts.weight_budget_bytes)),
    };

    let onboard = opts.resolve_onboard(routed.len()).unwrap_or(Onboard::Auto);
    let solve_req = hy4::request(
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
            let detail = match &inputs.local {
                LocalPackage::Missing(reason) if ranks == 0 => {
                    format!("Hy4 0-Spark placement requires installed RTX expert package: {reason} ({error})")
                }
                _ => format!("Hy4 joint admission refused: {error}"),
            };
            report.refuse_placement(
                detail,
                "Install exl3-hy4-k45-mul1 beside PROGRAMS.json, use a compact core view (k8/k6) or FP8 KV, or add Spark ranks (TP4/TP6).",
            );
            return empty(notes);
        }
    };

    report.phase_ledger = Some(crate::plan::SolvedPhaseLedger::from(&admitted.solved));
    report.host_budget_bytes = host_quota;
    report.host_ledger = match admitted.host_ledger(host_quota) {
        Ok(ledger) => Some(ledger),
        Err(error) => {
            report.refuse_placement(
                format!("Hy4 host ledger refused: {error}"),
                "Check host RAM budget.",
            );
            return empty(notes);
        }
    };
    let placed = admitted.placement;
    let onboarded = placed.onboard_layers;

    notes.push(format!(
        "Hy4 attention {}: pool {} tokens (78 DSA MLA layers + {} full IndexCache layers paged, pages-only prefix cache); \
         {} of {} routed layers onboarded ({}); unit {} x {} waves; C{}",
        placed.attention_summary(),
        placed.pool_tokens,
        cfg.index_cache.full_count(),
        onboarded,
        routed.len(),
        if gpus == 2 { "TP2 1024+1024" } else { "whole 2048" },
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
        device.items.extend(crate::placement::reserve::gpu_item("hy4"));
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
        let location = Location {
            device: Device::Spark(rank as u8),
            domain: Domain::Managed,
        };
        let items = match admitted.solved.items(location) {
            Ok(items) => items,
            Err(error) => {
                report.refuse_placement(
                    format!("Hy4 Spark ledger refused: {error}"),
                    "Check Spark budgets.",
                );
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
