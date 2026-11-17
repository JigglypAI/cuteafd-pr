//! One fact gatherer for Inkling 2 admission (#1930 stage 1). `cuteafd plan
//! --layout` and serving startup both call [`gather`]; they differ only in
//! [`Coordinator`] (a planned card forecast vs one measured CUDA sample per
//! GPU). The result is the shared recipe
//! [`InklingInputs`](crate::placement::families::inkling2::InklingInputs),
//! which `placement::planner::solve` admits. No family code samples memory or
//! sizes a pool here.
use crate::placement::families::inkling2::{self as shared, InklingInputs, LocalPackage, SparkCharge};
use crate::placement::planner::{serving, Budget, Context, PrefillShape};
use crate::placement::{ArchContext, ResidentWeights, SparkWorkingSet};
use crate::plan::families::inkling2::ExpertStorage;
use anyhow::Context as _;
use std::{path::Path, sync::Arc};

use super::{experts::{layer_experts, LayerExperts}, weight_layout::CoordinatorWeights, Inkling2Config};

/// A refusal both edges report the same way: what failed and how to fix it.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{what}")]
pub struct Refused {
    pub what: String,
    pub how: String,
}

fn refused(what: impl Into<String>, how: impl Into<String>) -> Refused { Refused { what: what.into(), how: how.into() } }

/// The coordinator edge: planned cards or measured samples.
#[derive(Debug, Clone)]
pub enum Coordinator {
    /// Logical caps per GPU and the planner's physical card class.
    Planned { cards: Vec<crate::placement::CoordinatorCard> },
    /// One CUDA sample per GPU before weights: (physical total, logical cap, free).
    Measured { samples: Vec<(u64, u64, u64)>, loaded_code: Vec<u64> },
}

impl Coordinator {
    pub fn gpus(&self) -> usize {
        match self { Self::Planned { cards } => cards.len(), Self::Measured { samples, .. } => samples.len() }
    }
    fn physical(&self, gpu: usize) -> u64 {
        match self { Self::Planned { cards } => cards[gpu].physical_bytes, Self::Measured { samples, .. } => samples[gpu].0 }
    }
    fn budget(&self, gpu: usize) -> Budget {
        match self {
            Self::Planned { cards } => shared::planned_budget(gpu as u8, cards[gpu].physical_bytes,
                cards[gpu].logical_bytes, ArchContext::for_device("sm_120", cards[gpu].physical_bytes).context_bytes),
            Self::Measured { samples, .. } => shared::measured_budget(gpu as u8, samples[gpu].0, samples[gpu].1,
                samples[gpu].2),
        }
    }
}

/// Every operator input admission reads, resolved once by the caller.
pub struct FactsRequest<'a> {
    pub vision: crate::plan::MediaMode,
    pub vision_replicas: usize,
    pub media_cache_bytes: Option<u64>,
    pub checkpoint: &'a crate::plan::Checkpoint,
    pub catalog: &'a crate::OfficialV41Catalog,
    pub cfg: &'a Inkling2Config,
    pub kv_format: super::InklingKvCache,
    pub manifest_path: Option<&'a Path>,
    pub manifest: Option<Arc<serde_json::Value>>,
    pub coordinator: Coordinator,
    /// The resolved attention placement (`check_attention` on the plan edge,
    /// the launcher's resolved value at serve).
    pub attention: crate::placement::AttentionPlacement,
    pub spark_ranks: usize,
    pub spark_bytes: u64,
    pub prefill: PrefillShape,
    /// Explicit context, or 0 for the checkpoint's maximum.
    pub context_tokens: u64,
    pub pool_tokens: Option<u64>,
    pub target_pool_tokens: Option<u64>,
    pub sequences: u64,
    pub host: (u64, u64),
    pub host_cache_bytes: Option<u64>,
    pub prefix_entries: u64,
    pub prefix_mark_budget: u64,
    pub replies: cuteafd_core::spark_reply::SparkReplyFormats,
    pub head: serving::BlasHead,
    pub expert_usage: bool,
    pub resident_weight_limit: ResidentWeights,
}

/// Shared installed executor owners, never a family scratch estimate. Each
/// independently retained native backend owns its scratch/output once.
pub fn local_package(catalog: &crate::OfficialV41Catalog, manifest: Option<&Path>,
    gpus: usize, rows: u64) -> LocalPackage {
    let workspace = || -> anyhow::Result<LocalPackage> {
        anyhow::ensure!((1..=2).contains(&gpus), "Inkling local package requires one or two ranks");
        let routed = catalog.routed_experts();
        let layers = (routed.first_layer..routed.layers).collect::<Vec<_>>();
        let exl3_requests = if catalog.exl3().is_some() { Some(crate::placement::rtx_onboard::expert_workspace_requests(
            catalog, manifest, rows, gpus, "inkling2")?) } else { None };
        // Residency constructs descriptor/load tables; compute every physical
        // layer/rank once, not once for every onboarding prefix containing it.
        let exl3_regions = if let Some(exl3) = catalog.exl3() {
            (0..gpus).map(|gpu| layers.iter().map(|&layer| {
                let layout = exl3.residency(crate::V41Exl3Layer::Backbone(layer), gpus, gpu)?;
                let name = if gpus == 1 { format!("BackboneFull {{ layer: {layer} }}.0") }
                    else { format!("BackboneTp2 {{ layer: {layer}, rank: {gpu} }}.0") };
                Ok((name, layout.device_arena_layout()?.1))
            }).collect::<anyhow::Result<Vec<_>>>()).collect::<anyhow::Result<Vec<_>>>()?
        } else { Vec::new() };
        let mut native_requests = std::collections::BTreeMap::<String, (usize, usize)>::new();
        let mut backings = vec![Vec::new()];
        for count in 1..=layers.len() {
            let selected = &layers[..count];
            let mut owners = Vec::new();
            if let Some(tensors) = catalog.fp8() {
                for (view, group_layers) in tensors.homogeneous_layers(selected)? {
                    let key = format!("{:?}", view.format());
                    let (scratch, output) = if let Some(&requests) = native_requests.get(&key) { requests } else {
                        let requests = crate::placement::rtx_onboard::fp8_workspace_requests(&view,
                            manifest, rows, gpus, "inkling2")?;
                        let scratch = *requests.first().context("native package scratch absent")?;
                        let (scratch, output, _) = crate::serving_capacity::fp8moe_tp2_workspace(usize::try_from(scratch)?,
                            routed.hidden, usize::try_from(rows)?)?;
                        native_requests.insert(key, (scratch, output));
                        (scratch, output)
                    };
                    for gpu in 0..gpus {
                        let regions = crate::serving_capacity::fp8_expert_regions(&view.for_coordinator(),
                            &group_layers, gpus, gpu, crate::formats::fp8_experts::Slicing::Blocks(128))?;
                        let id = crate::serving_capacity::expert_weight_arena_id("fp8", gpu as i32, &regions)?;
                        owners.push(crate::placement::families::inkling2::LocalBacking { gpu,
                            layers: group_layers.clone(), arena: crate::serving_capacity::WeightArenaLayout::new(id, regions)?,
                            workspace: vec![("scratch".into(), scratch as u64), ("output".into(), output as u64)] });
                    }
                }
            } else {
                let requests = exl3_requests.as_ref().context("Inkling EXL3 workspace absent")?;
                for gpu in 0..gpus {
                    let regions = exl3_regions[gpu][..count].to_vec();
                    let id = crate::serving_capacity::expert_weight_arena_id("exl3", gpu as i32, &regions)?;
                    let last = requests.len().checked_sub(1).context("EXL3 output owner absent")?;
                    let workspace = requests.iter().enumerate().map(|(index, &bytes)|
                        (if index == last { "output".into() } else { format!("scratch.{index}") }, bytes)).collect();
                    owners.push(crate::placement::families::inkling2::LocalBacking { gpu, layers: selected.to_vec(),
                        arena: crate::serving_capacity::WeightArenaLayout::new(id, regions)?, workspace });
                }
            }
            backings.push(owners);
        }
        Ok(LocalPackage::Installed { backings })
    };
    workspace().unwrap_or_else(|error| LocalPackage::Missing(format!("{error:#}")))
}

/// Spark charges use the shared catalog's physical resident layout, the
/// shared typed-owner census, and endpoint I/O once per rank. Missing census
/// inputs remain explicit errors; no checkpoint-byte fraction sizes scratch.
pub fn spark_charges(catalog: &crate::OfficialV41Catalog, experts: &[LayerExperts], routed: &[usize], ranks: usize,
    prefill: PrefillShape, ingress: cuteafd_core::SparkIngress) -> anyhow::Result<Vec<Vec<SparkCharge>>> {
    if ranks == 0 { return Ok(vec![Vec::new(); routed.len() + 1]); }
    anyhow::ensure!((2..=8).contains(&ranks), "Inkling 2 Spark worlds are 2..=8 ranks, not {ranks}");
    let exl3 = experts.iter().any(|layer| matches!(layer.storage, Some(ExpertStorage::Exl3 { .. })));
    let executions = shared::spark_executions(prefill, exl3)?;
    let endpoint = cuteafd_core::SparkEndpoint { ring_depth: None, capacity: u32::try_from(prefill.rows)?,
        ingress, partial_reply_bytes: None };
    let rings = crate::plan::layout::spark_ring_geometry(shared::LANES as usize);
    let rings = cuteafd_core::SparkRings { max_frame_bytes: rings.max_frame_bytes, depth: rings.depth,
        slot_bytes: rings.slot_bytes, alignment: rings.alignment };
    let runtime = cuteafd_core::memory_layout::spark_runtime_reserve_bytes(0).context("Spark runtime residual")?;
    let geometry = cuteafd_core::ExpertGeometry::INKLING2;
    (0..=routed.len()).map(|count| {
        let working = SparkWorkingSet { rtx_layers: routed[..count].to_vec(), spark_layers: routed[count..].to_vec(),
            spark_ranks: ranks, prefill_lanes: shared::LANES };
        (0..ranks).map(|rank| {
            if count == routed.len() { return Ok(SparkCharge { experts: 0, owners: 0, runtime: 0 }); }
            let io = working.rank_endpoints_io(rank, geometry, &[endpoint], rings)?;
            let owners = executions.workspace_bytes(|extent|
                crate::plan::layout::spark_slots::catalog_owner_bytes(catalog, "inkling2", &working, rank, extent))?
                .and_then(|bytes| bytes.total()).context("Inkling 2 Spark owner sum overflow")?;
            let owners = [io.intake, io.host_exchange, io.row_indices, io.mapped_rings, io.endpoint_staging, owners]
                .into_iter().try_fold(0u64, |sum, bytes| sum.checked_add(bytes)).context("Inkling 2 Spark owners")?;
            let resident = if let Some(tensors) = catalog.fp8() {
                // The shared native worker packs all homogeneous groups in
                // one physical arena, in group/projection order.
                let regions = tensors.homogeneous_layers(&working.spark_layers)?.into_iter()
                    .map(|(view, layers)| -> anyhow::Result<Vec<(String, usize)>> {
                        let format = if view.dynamic_nvfp4() { "nvfp4a4" } else { "mxfp8" };
                        let recipe = crate::plan::experts::spark_recipe(&format!("inkling2:{format}"),
                            catalog.routed_experts().intermediate, ranks)?;
                        crate::serving_capacity::fp8_expert_regions(&view, &layers, ranks, rank, recipe.slicing)
                    }).collect::<anyhow::Result<Vec<_>>>()?.into_iter().flatten();
                crate::serving_capacity::WeightArenaLayout::new("inkling2/spark-residency", regions)?.bytes() as u64
            } else {
                let regions = working.spark_layers.iter().map(|&layer|
                    catalog.exl3_spark_arena_bytes(layer, ranks).map(|bytes| bytes[rank]))
                    .collect::<anyhow::Result<Vec<_>>>()?;
                crate::serving_capacity::PackedWeightRegions::new(regions.into_iter()
                    .map(usize::try_from).collect::<Result<Vec<_>, _>>()?)?.bytes as u64
            };
            Ok(SparkCharge { experts: resident, owners, runtime })
        }).collect()
    }).collect()
}

/// Gather every admitted owner fact into the shared recipe.
pub fn gather(request: &FactsRequest<'_>) -> Result<InklingInputs, Refused> {
    let cfg = request.cfg;
    let gpus = request.coordinator.gpus();
    let ranks = request.spark_ranks;
    if !(1..=2).contains(&gpus) {
        return Err(refused(format!("Inkling 2 runs on one RTX or an RTX pair, not {gpus}"), "Use --rtx 1 or --rtx 2."));
    }
    let context = if request.context_tokens == 0 { cfg.max_position as u64 } else { request.context_tokens };
    if context > cfg.max_position as u64 {
        return Err(refused(format!("Inkling 2 context {context} exceeds the checkpoint's {}", cfg.max_position),
            "Lower --max-context-tokens."));
    }
    let shape = request.prefill;
    let routed = (0..cfg.layers).collect::<Vec<_>>();
    let placement = if gpus == 1 { crate::placement::AttentionPlacement::Heads } else { request.attention };
    let modes = shared::layer_modes(cfg, gpus, placement);
    let targets = CoordinatorWeights::plan(request.checkpoint, cfg, gpus, &modes)
        .map_err(|error| refused(format!("Inkling 2 target arena inventory refused: {error:#}"),
            "Provide complete BF16 core headers (every shard of the selected view)."))?;
    let mount = targets.iter().map(|target| target.mount_load(request.checkpoint)).collect::<anyhow::Result<Vec<_>>>()
        .map_err(|error| refused(format!("Inkling 2 mount inventory refused: {error:#}"), "Check checkpoint headers."))?;
    let experts = layer_experts(request.checkpoint, cfg).map_err(|error| refused(
        format!("Inkling 2 routed expert inventory refused: {error:#}"),
        "Provide every routed layer's fused MXFP8/NVFP4 headers or per-expert EXL3 modules."))?;
    let storage = experts.iter().map(|layer| layer.storage.expect("inventory names every layer's storage")).collect::<Vec<_>>();
    let catalog = request.catalog;
    let local = local_package(&catalog, request.manifest_path, gpus, shape.rows.max(u64::from(shared::DECODE_ROWS)));
    // Mixed native checkpoints (NVFP4 then MXFP8) keep the per-layer native
    // ingress; the link is sized for its widest request row (BF16).
    let first_wire = shared::ingress(storage.first().copied());
    let ingress = if storage.iter().all(|&storage| shared::ingress(Some(storage)) == first_wire) { first_wire }
        else { cuteafd_core::SparkIngress::Native };
    let spark = spark_charges(&catalog, &experts, &routed, ranks, shape, ingress)
        .map_err(|error| refused(format!("Inkling 2 Spark inventory refused: {error:#}"),
            "Use a Spark world of 2..=8 ranks."))?;
    let links = shared::spark_link((gpus - 1) as u8, ranks, shape, request.replies, ingress)
        .map_err(|error| refused(format!("Inkling 2 Spark transport owners refused: {error}"), "Use a supported unit."))?;
    let programs = request.manifest.as_ref().map(|manifest|
        crate::placement::ProgramSet::from_manifest(manifest, &["inkling2", "exlr8"]));
    let mut code = Vec::new();
    let mut graph_reserve = Vec::new();
    for gpu in 0..gpus {
        code.push(match &request.coordinator {
            Coordinator::Planned { .. } => crate::placement::inventory::loaded_code_allowance("inkling2", "experts",
                gpus == 2, gpu as u8, ArchContext::for_device("sm_120", request.coordinator.physical(gpu)),
                programs.as_ref()).bytes,
            // The measured sample follows module loading: only what it has not seen.
            Coordinator::Measured { loaded_code, .. } => loaded_code.get(gpu).copied().unwrap_or(0),
        });
        // No traced Inkling 2 prewarm layout yet: base zero, the shared runtime
        // headroom is the whole canonical reserve (graph-cache stage 4).
        graph_reserve.push(crate::placement::graph_reserve::current(gpu, false, 0, 0)
            .map_err(|error| refused(format!("Inkling 2 graph reserve refused: {error}"), "Fix GPU_GRAPH_RESERVE."))?.bytes);
    }
    let mut budgets = (0..gpus).map(|gpu| request.coordinator.budget(gpu)).collect::<Vec<_>>();
    budgets.extend((0..ranks).map(|rank| shared::spark_budget(rank as u8, request.spark_bytes)));
    let (host_total, host_available) = request.host;
    budgets.push(crate::placement::families::kimi::host_budget(host_total, host_available));
    let page_rows = if modes.iter().any(|mode| *mode == crate::placement::LayerMode::ContextSplit) { 2 * shared::PAGE_ROWS }
        else { shared::PAGE_ROWS };
    // Host prefix pages hold the global layers' K/V; marks hold the rings and
    // conv lags of every layer (byte-identical restores).
    let kv_row = (cfg.kv_heads * cfg.head_dim * 2) as u64;
    let page_bytes = shared::PAGE_ROWS * 2 * kv_row * cfg.global_layers().len() as u64;
    let mark_bytes = (0..cfg.layers).map(|layer| shared::SequenceState::new(cfg, cfg.local[layer])
        .map(|state| state.ring_bytes + state.lag_bytes)).sum::<Result<u64, _>>()
        .map_err(|error| refused(format!("Inkling 2 sequence state refused: {error}"), "Check the window geometry."))?;
    let host_prefix = if request.host_cache_bytes == Some(0) || request.prefix_entries == 0 {
        serving::HostPrefixOwners { quota_bytes: 0, chunk_bytes: 0 }
    } else {
        let geometry = cuteafd_core::prefix::HostCacheGeometry { unit_rows: page_rows,
            page_bytes: page_bytes * page_rows / shared::PAGE_ROWS, mark_bytes, draft_bytes: 0 };
        let quota = geometry.resolve(request.host_cache_bytes, context, request.sequences, host_total,
            geometry.chunk_bytes()).unwrap_or(0);
        serving::HostPrefixOwners { quota_bytes: quota, chunk_bytes: geometry.chunk_bytes().min(quota.max(1)) }
    };
    let push_required = match crate::push_load::namespace(&request.checkpoint.snapshot) {
        Ok(_) => true,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound && error.raw_os_error().is_none()
            && request.checkpoint.snapshot.exists() => false,
        Err(error) => return Err(refused(format!("Inkling checkpoint namespace refused: {error}"),
            "Restore SparkNest namespace metadata; required push cannot fall back to mount.")),
    };
    let push_host = (0..=routed.len()).map(|count| {
        if !push_required { return Ok(None); }
        let backings = match &local {
            LocalPackage::Installed { backings } => backings[count].as_slice(),
            LocalPackage::Missing(_) if count == 0 => &[],
            LocalPackage::Missing(_) => return Ok(None),
        };
        super::weight_layout::push_host_envelope(&catalog, backings, gpus).map(Some)
    }).collect::<anyhow::Result<Vec<_>>>().map_err(|error| refused(format!("Inkling push host inventory refused: {error:#}"),
        "Provide complete native push source/descriptor headers."))?;
    let capacities = (0..gpus).map(|gpu| request.coordinator.budget(gpu).logical_bytes).collect::<Vec<_>>();
    let mut pool_facts = crate::placement::PoolFacts::new(&capacities,
        Context::resolve(context, (request.context_tokens > 0).then_some(request.context_tokens)), request.pool_tokens, page_rows);
    pool_facts.target = request.target_pool_tokens;
    let pool = crate::placement::prepare_pool(pool_facts).map_err(|error|
        refused(format!("invalid KV pool policy: {error}"), "Set KV min <= target <= max."))?;
    let vision = crate::placement::still_encoder::StillEncoder::planned(request.checkpoint,
        request.vision, request.vision_replicas, &budgets, request.media_cache_bytes.unwrap_or(256 << 20))
        .map_err(|error| refused(format!("Inkling vision inventory refused: {error:#}"), "Provide the bundled PE tensors."))?;
    Ok(InklingInputs { cfg: cfg.clone(), kv_format: request.kv_format, gpus, spark_ranks: ranks,
        context: Context::resolve(context, (request.context_tokens > 0).then_some(request.context_tokens)),
        pool, prefill: shape, sequences: request.sequences, prefix_entries: request.prefix_entries,
        prefix_mark_budget: request.prefix_mark_budget, targets, placement, modes,
        routed: routed.clone(), experts, local, spark, links,
        spark_reply_formats: request.replies, drafter: None, vision, budgets, code, graph_reserve,
        head: request.head, host_prefix, device_sampling: true,
        expert_usage: request.expert_usage.then(|| cuteafd_core::expert_usage::Geometry { layers: cfg.layers,
            experts: cfg.experts, topk: cfg.topk, routed_layers: routed }),
        resident_weight_limit: Some(request.resident_weight_limit), mount, push_host })
}
