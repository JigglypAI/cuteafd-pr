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

pub(crate) struct Options<'a> {
    pub device: i32,
    pub peers: &'a [SocketAddr],
    pub manifest: &'a Path,
    pub prefill: PrefillShape,
    pub sequences: u64,
    pub context_tokens: u64,
    pub pool_tokens: Option<u64>,
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
}

fn validate(options: &Options<'_>, cfg: &Inkling2Config) -> Result<()> {
    ensure!(options.device >= 0 && (2..=8).contains(&options.peers.len()),
        "Inkling initial distributed startup requires one coordinator GPU and 2..8 Spark endpoints");
    ensure!(options.peers.iter().collect::<std::collections::BTreeSet<_>>().len() == options.peers.len(),
        "Inkling Spark endpoints must be distinct");
    cuteafd_core::concurrency::validate("inkling2", options.sequences)?;
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
    let mut kernels = None;
    let runtime = RuntimeInventory::measure(library, &[options.device], InventoryPhase::PreWeight, |_| {
        // Static load finish is explicitly checked before any push owner exists.
        // Missing pin/export is a named startup error, never a host fallback.
        programs.program("inkling2_padded_transpose_bf16_r16_c1024_p64", &["source", "output"])?;
        kernels = Some(Kernels::bind(programs)?);
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
    let request = FactsRequest { checkpoint, cfg, manifest_path: Some(options.manifest), manifest: Some(manifest),
        coordinator: Coordinator::Measured { samples, loaded_code: vec![0] }, attention: AttentionPlacement::Heads,
        spark_ranks: options.peers.len(), spark_bytes: policy.spark_cap_bytes(100 << 30),
        prefill: options.prefill, context_tokens: options.context_tokens, pool_tokens: options.pool_tokens,
        sequences: options.sequences, host, host_cache_bytes,
        prefix_entries: options.prefix.prefix_cache_entries as u64,
        prefix_mark_budget: options.prefix.prefix_cache_mark_mib as u64 * (1 << 20),
        replies: cuteafd_core::spark_reply::requested(),
        head: crate::shared::target_head::TargetHead::configured().blas_head(true),
        expert_usage: options.expert_usage, resident_weight_limit: options.resident_weights };
    let inputs = gather(&request)?;
    let admitted = inputs.admit(inkling2::request(inputs.context, options.pool_tokens, options.onboard,
        options.prefill, cuteafd_loader::placement::planner::Operation::Serving,
        inputs.resident_weight_limit, request.replies))?;
    runtime.record("inkling2", &runtime.gpus.iter().map(|sample| sample.free_bytes).collect::<Vec<_>>());
    cuteafd_bench::context::ResolvedSettings::placement(&admitted.placement).publish();
    cuteafd_bench::context::set_resolved("max-context-tokens", &admitted.solved.served_context().to_string());
    Ok(Prepared { kernels: kernels.context("Inkling kernel preload did not run")?, encoding, eos, inputs, admitted, runtime })
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
    }
}
