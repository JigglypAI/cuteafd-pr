//! Native startup facts on the same shared solver as `cuteafd plan`.
//! Weight delivery stays unpublished until its admitted push transaction lands.
use super::layers::Kernels;
use crate::shared::{inventory::{InventoryPhase, RuntimeInventory}, prefix::{budget::HostBudget, PrefixArgs}};
use anyhow::{ensure, Context, Result};
use cuteafd_api::openai::{chat::inkling2::InklingEncoding, ModelEncoding};
use cuteafd_ffi::{programs::Programs, NativeLibrary};
use cuteafd_loader::{families::inkling2::{admission::{gather, Coordinator, FactsRequest}, Inkling2Config},
    placement::{families::inkling2::{self, AdmittedInkling, InklingInputs}, planner::PrefillShape,
        AttentionPlacement, Onboard, ResidentWeights}, plan::Checkpoint};
use std::{net::SocketAddr, path::Path, sync::Arc};

#[derive(Clone, Copy)]
pub(crate) struct Options<'a> {
    pub device: i32,
    pub peers: &'a [SocketAddr],
    pub manifest: &'a Path,
    pub native_lib: &'a Path,
    pub prefill: PrefillShape,
    pub sequences: u64,
    pub context_tokens: u64,
    pub pool_tokens: Option<u64>,
    pub kv_format: cuteafd_loader::families::inkling2::InklingKvCache,
    pub onboard: Onboard,
    pub prefix: &'a PrefixArgs,
    pub resident_weights: ResidentWeights,
    pub expert_usage: bool,
}

/// Prepared owner facts, not readiness. No target/cache/transport allocation
/// escapes this edge, and no measured sample is taken after loading weights.
pub(crate) struct Prepared<'a> {
    pub kernels: Kernels<'a>,
    pub encoding: ModelEncoding,
    pub eos: Vec<u32>,
    pub inputs: InklingInputs,
    pub admitted: AdmittedInkling,
    pub runtime: RuntimeInventory,
    /// Expert code belongs to the sampled baseline until its runtime binds.
    pub module_pins: Vec<cuteafd_ffi::fp8_moe::Fp8MoeModule>,
    pub exl3_modules: Vec<crate::shared::experts::exl3::execution::Exl3Modules<'a>>,
}

fn validate(options: &Options<'_>, cfg: &Inkling2Config) -> Result<()> {
    ensure!(options.device >= 0 && (2..=8).contains(&options.peers.len()),
        "Inkling initial distributed startup requires one coordinator GPU and 2..8 Spark endpoints");
    ensure!(options.peers.iter().collect::<std::collections::BTreeSet<_>>().len() == options.peers.len(),
        "Inkling Spark endpoints must be distinct");
    cuteafd_core::concurrency::validate("inkling2", options.sequences)?;
    let ring = cuteafd_loader::families::inkling2::execution::SequenceLayout::new(cfg, true, 1)?.ring_rows;
    cuteafd_loader::families::inkling2::verify::verify_ring(cfg.window, ring, inkling2::DECODE_ROWS as usize)?;
    ensure!(options.context_tokens == 0 || options.context_tokens <= cfg.max_position as u64,
        "Inkling context exceeds checkpoint");
    ensure!(options.prefill.rows > 0 && options.prefill.rows <= 4096 && options.prefill.rows % 64 == 0
        && (1..=3).contains(&options.prefill.waves), "Inkling prefill differs from shared export/wave capacity");
    ensure!(options.prefix.prefix_partial == crate::shared::prefix::Toggle::Off,
        "Inkling requires exact captured SWA/convolution frontiers");
    Ok(())
}

pub(crate) fn prepare<'a>(library: &'a NativeLibrary, programs: &'a Programs<'a>, checkpoint: &Checkpoint,
    cfg: &Inkling2Config, options: Options<'_>) -> Result<Prepared<'a>> {
    validate(&options, cfg)?;
    cfg.release_geometry()?;
    // Checkpoint-native chat resolution happens before CUDA or target loading.
    let encoding = ModelEncoding::Inkling(Arc::new(InklingEncoding::from_snapshot(&checkpoint.snapshot)?));
    let eos = encoding.eos_token_ids();
    ensure!(eos == [200006], "Inkling checkpoint EOS differs from its native sampling-end marker");
    cuteafd_core::set_expert_geometry(cuteafd_core::ExpertGeometry::INKLING2)
        .map_err(|geometry| anyhow::anyhow!("Inkling requires isolated process geometry, found {geometry:?}"))?;
    let catalog = match &checkpoint.view {
        Some(view) => cuteafd_loader::read_expert_catalog_view(view)?,
        None => cuteafd_loader::read_expert_catalog(&checkpoint.snapshot)?,
    };
    let mut module_pins = Vec::new();
    let mut exl3_modules = Vec::new();
    let mut kernels = None;
    let runtime = RuntimeInventory::measure(library, &[options.device], InventoryPhase::PreWeight, |_| {
        // Static load finish is explicitly checked before any push owner exists.
        // Missing pin/export is a named startup error, never a host fallback.
        programs.program("inkling2_padded_transpose_bf16_r16_c1024_p64", &["source", "output"])?;
        kernels = Some(Kernels::bind(programs, options.kv_format)?);
        if let Some(tensors) = catalog.fp8() {
            let layers = (0..cfg.layers).collect::<Vec<_>>();
            for (view, _) in tensors.homogeneous_layers(&layers)? {
                let directory = crate::shared::experts::fp8::package_directory(options.native_lib, 1, view.format());
                // SAFETY: trusted installed static expert package; retained
                // across sampling and final backend binding, no weight storage.
                module_pins.push(unsafe { cuteafd_ffi::fp8_moe::Fp8MoeModule::load(&directory)? });
            }
        } else if catalog.exl3().is_some() {
            let directory = options.native_lib.parent().context("Inkling native library directory absent")?
                .join("exl3/exl3-inkling2-k45/rtx-tp1");
            exl3_modules = crate::shared::experts::rtx::exl3::Exl3Tp2::preload_whole_modules(
                crate::shared::memory::device::Device { library, id: options.device }, &directory,
                usize::try_from(options.prefill.rows.max(64))?, cuteafd_core::ExpertGeometry::INKLING2,
                crate::shared::experts::rtx::RtxSelectionPolicy::FirstCapacity)?;
        }
        Ok(())
    })?;
    let policy = cuteafd_core::memory_policy::MemoryPolicy::current();
    let samples = runtime.gpus.iter().map(|sample| (sample.total_bytes,
        policy.gpu_cap_bytes(sample.total_bytes), sample.free_bytes)).collect();
    let host = cuteafd_loader::formats::mapped_preload::host_memory_before_process()?;
    let host_cache_bytes = match options.prefix.host_cache_bytes {
        HostBudget::Auto => None, HostBudget::Bytes(bytes) => Some(bytes),
    };
    let manifest = Arc::new(serde_json::from_slice(&std::fs::read(options.manifest)?)?);
    let request = FactsRequest { kv_format: options.kv_format, drafter: None, checkpoint, catalog: &catalog, cfg, manifest_path: Some(options.manifest), manifest: Some(manifest),
        coordinator: Coordinator::Measured { samples, loaded_code: vec![0] }, attention: AttentionPlacement::Heads,
        spark_ranks: options.peers.len(), spark_bytes: cuteafd_loader::serving_capacity::spark_memory::runtime_bytes(&options.peers.iter().map(ToString::to_string).collect::<Vec<_>>())?.into_iter().max().unwrap_or(0),
        prefill: options.prefill, context_tokens: options.context_tokens, pool_tokens: options.pool_tokens,
        sequences: options.sequences, host, host_cache_bytes,
        prefix_entries: options.prefix.prefix_cache_entries as u64,
        prefix_mark_budget: options.prefix.prefix_cache_mark_mib as u64 * (1 << 20),
        replies: cuteafd_core::spark_reply::requested(),
        head: crate::shared::target_head::TargetHead::configured().blas_head(true),
        expert_usage: options.expert_usage, resident_weight_limit: options.resident_weights };
    let mut inputs = gather(&request)?;
    cuteafd_loader::serving_capacity::spark_memory::apply_rank_budgets(&mut inputs.budgets,
        &cuteafd_loader::serving_capacity::spark_memory::runtime_bytes(&options.peers.iter().map(ToString::to_string).collect::<Vec<_>>())?)?;
    let admitted = inputs.admit(inkling2::request(inputs.context, options.pool_tokens, options.onboard,
        options.prefill, cuteafd_loader::placement::planner::Operation::Serving,
        inputs.resident_weight_limit, request.replies))?;
    runtime.record("inkling2", &runtime.gpus.iter().map(|sample| sample.free_bytes).collect::<Vec<_>>());
    cuteafd_bench::context::ResolvedSettings::placement(&admitted.placement).publish();
    cuteafd_bench::context::set_resolved("max-context-tokens", &admitted.solved.served_context().to_string());
    Ok(Prepared { kernels: kernels.context("Inkling kernel preload did not run")?, encoding, eos, inputs, admitted, runtime, module_pins, exl3_modules })
}

/// Admitted grouped delivery entry, still not a public serving executor. The
/// endpoint is selected before target allocation and native views remain private
/// until both push participants and Spark workers commit their exact assignment.
pub(crate) fn deliver<'a>(library: &'a NativeLibrary,
    view: &cuteafd_loader::plan::checkpoint::CheckpointView, checkpoint: &Checkpoint,
    prepared: &Prepared<'a>, options: &Options<'_>)
    -> Result<(super::weights::TargetWeights<'a>, Vec<Vec<crate::shared::experts::rtx::push::PushedExperts<'a>>>,
        Option<crate::shared::speculation::recurrent_weights::Weights<'a>>)> {
    use crate::shared::{memory::device::Device, placement::{spark_rings, SparkAdmission}};
    use cuteafd_core::{ExpertGeometry, SparkEndpoint};
    use cuteafd_loader::placement::SparkWorkingSet;
    use cuteafd_transport::{worker_selection::ModelAdmission, TcpTransportConfig};
    let catalog = cuteafd_loader::read_expert_catalog_view(view)?;
    let inputs = &prepared.inputs;
    let admitted = &prepared.admitted;
    let working = SparkWorkingSet { rtx_layers: inputs.routed[..admitted.onboard].to_vec(),
        spark_layers: inputs.routed[admitted.onboard..].to_vec(), spark_ranks: inputs.spark_ranks,
        prefill_lanes: inkling2::LANES };
    let ingress = inkling2::ingress(inputs.experts.first().and_then(|layer| layer.storage));
    let ingress = if inputs.experts.iter().all(|layer| inkling2::ingress(layer.storage) == ingress) { ingress }
        else { cuteafd_core::SparkIngress::Native };
    let endpoint = SparkEndpoint { ring_depth: None, capacity: u32::try_from(inputs.prefill.rows)?,
        ingress, partial_reply_bytes: None };
    let config = TcpTransportConfig::default();
    let model = ModelAdmission::for_clients(ExpertGeometry::INKLING2, endpoint,
        spark_rings(config.max_frame_bytes)?, 1,
        inkling2::spark_executions(inputs.prefill, catalog.exl3().is_some())?)?;
    let spark = SparkAdmission::plan_view(view, ExpertGeometry::INKLING2, Some(&catalog),
        &working, inputs.spark_ranks, model, None)?;
    let (target, membership) = super::weights::PendingTarget::prepare(Device { library, id: options.device },
        view, checkpoint, &catalog, inputs, admitted)?;
    let pending = spark.start_with(options.peers, Some(membership))?;
    target.finish(&catalog, pending)
}

/// Construct only the initial whole-rank execution after exact shared
/// admission and required grouped delivery. The public scheduler/readiness
/// edge is intentionally separate and remains unqualified until native gates.
pub(crate) fn whole<'a>(library: &'a Arc<NativeLibrary>, programs: &'a Programs<'a>,
    view: &cuteafd_loader::plan::checkpoint::CheckpointView, checkpoint: &Checkpoint,
    cfg: &Inkling2Config, options: Options<'_>) -> Result<(super::engine::Execution<'a>, InklingInputs, AdmittedInkling, ModelEncoding)> {
    use crate::shared::{memory::device::{Device, Stream}, spark_intake::SparkLink};
    use cuteafd_transport::{expert::{SparkWorld, EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16},
        ExpertV2Dtype, SparkSessionShape, TcpTransportConfig};
    let phase = crate::shared::memory::load_timing::Split::start();
    let prepared = prepare(library, programs, checkpoint, cfg, Options { ..options })?;
    phase.log("inkling2 code and admission");
    cuteafd_bench::context::phase("Inkling code and admission");
    let phase = crate::shared::memory::load_timing::Split::start();
    let (target, pushed, draft_weights) = deliver(library, view, checkpoint, &prepared, &options)?;
    ensure!(draft_weights.is_none(), "bundled drafter delivered before its serving executor binding");
    phase.log("inkling2 grouped weight delivery");
    cuteafd_bench::context::phase("Inkling grouped weight delivery");
    let phase = crate::shared::memory::load_timing::Split::start();
    let Prepared { kernels, encoding, inputs, admitted, module_pins, exl3_modules, .. } = prepared;
    let device = Device { library, id: options.device };
    let stream = std::rc::Rc::new(Stream::new(device)?);
    let requests = super::owners::RequestOwners::whole(stream.clone(), &inputs, &admitted)?;
    let head = super::head::Head::new(stream.clone(), cfg, 64)?;
    let catalog = cuteafd_loader::read_expert_catalog_view(view)?;
    let backings = match &inputs.local {
        inkling2::LocalPackage::Installed { backings } => backings[admitted.onboard].as_slice(),
        _ if admitted.onboard == 0 => &[],
        _ => anyhow::bail!("Inkling admitted resident backend absent"),
    };
    let mut ranks = pushed.into_iter();
    let sets = ranks.next().context("Inkling committed coordinator rank absent")?;
    ensure!(ranks.next().is_none() && sets.len() == backings.len(), "Inkling pushed expert set/rank count differs");
    let mut local = Vec::<Box<dyn super::experts::Dispatcher + 'a>>::new();
    for (backing, set) in backings.iter().zip(sets) {
        let handles = super::experts::Resident::from_pushed(stream.clone(), &catalog, options.native_lib,
            backing, usize::try_from(inputs.prefill.rows.max(64))?, set)?;
        local.extend(handles.into_iter().map(|owner| Box::new(owner) as Box<dyn super::experts::Dispatcher>));
    }
    ensure!(local.len() == admitted.onboard, "Inkling resident layer handle count differs");
    let geometry = inputs.links.first().context("Inkling admitted Spark link absent")?.clone();
    let mut link = device.run(|| SparkLink::prepared(library, SparkWorld::canonical(options.peers.len())?,
        options.peers, TcpTransportConfig::default(), geometry))?;
    link.enable_terminal_ownership(Arc::clone(library))?;
    let mut session = SparkSessionShape::new(u32::try_from(inputs.prefill.rows)?, u32::try_from(cfg.hidden)?, u32::try_from(cfg.topk)?,
        if inputs.experts.iter().all(|layer| inkling2::ingress(layer.storage) == cuteafd_core::SparkIngress::Fp8K32) {
            ExpertV2Dtype::Fp8E4m3Ue8m0K32
        } else { ExpertV2Dtype::Bf16 })?;
    session.flags |= EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
    device.run(|| link.bootstrap(&session))?;
    let execution = super::engine::Execution::new(link, requests, head, target, kernels, local, cfg.clone(), inputs.graph_reserve[0],
        inputs.experts.iter().map(|layer| inkling2::ingress(layer.storage)).collect())?;
    let physical = [options.device];
    let devices = cuteafd_loader::placement::planner::CoordinatorDevices::new(&physical)?;
    let mut inventory = crate::shared::memory::receipts::Inventory::new(devices, "retained",
        cuteafd_loader::placement::planner::Lifetime::TARGET, cuteafd_core::memory_layout::Category::Workspace, "Inkling requests");
    let mut census = crate::shared::memory::receipts::Census::default();
    execution.requests.allocation_receipts(&mut inventory, &mut census, devices)?;
    super::admission::reconcile_requests(&admitted.solved, &inputs, &inventory.receipts)?;
    // Runtime backends now retain the same code sampled before weights.
    drop(module_pins); drop(exl3_modules);
    phase.log("inkling2 request owners and transport bootstrap");
    cuteafd_bench::context::phase("Inkling request owners and transport bootstrap");
    Ok((execution, inputs, admitted, encoding))
}

#[cfg(test)]
mod tests {
    #[test]
    fn startup_uses_native_chat_and_preweight_shared_fact_gather_without_readiness() {
        let source = include_str!("startup.rs").split("#[cfg(test)]").next().unwrap();
        assert!(source.contains("InklingEncoding::from_snapshot(&checkpoint.snapshot)"));
        assert!(source.contains("eos == [200006]"));
        assert!(source.contains("InventoryPhase::PreWeight"));
        assert!(source.contains("Coordinator::Measured"));
        assert!(source.contains("gather(&request)"));
        assert!(source.contains("inputs.admit(inkling2::request("));
        assert!(!source.contains("ready_tx"));
        assert!(!source.contains("TargetWeights::load"));
        assert!(source.contains("SparkAdmission::plan_view("));
        assert!(source.contains("PendingTarget::prepare("));
        assert!(source.contains("spark.start_with(options.peers, Some(membership))"));
        assert!(source.contains("target.finish(&catalog, pending)"));
    }
}
