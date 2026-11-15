//! Inkling HTTP/request lifecycle over the shared scheduler. CLI dispatch is
//! intentionally not registered until the v52f native load pin promotes.
use crate::shared::{prefix::PrefixArgs, token_io::TokenIoArgs};
use anyhow::{ensure, Context, Result};
use cuteafd_loader::{families::inkling2::Inkling2Config, placement::planner::PrefillShape, plan::Checkpoint};
use std::{net::SocketAddr, path::PathBuf, sync::{Arc, Mutex}};

#[derive(Debug, Clone, clap::Args)]
pub(crate) struct ServeArgs {
    #[command(flatten)] pub attention: crate::cli::CoreViewArgs,
    #[arg(long)] pub snapshot: PathBuf,
    #[arg(long, env="CUTEAFD_NATIVE_LIB")] pub native_lib: PathBuf,
    #[arg(long, default_value="/opt/cuteafd/share/PROGRAMS.json")] pub manifest: PathBuf,
    #[arg(long, default_value_t=0)] pub device: i32,
    #[arg(long, value_delimiter=',')] pub peers: Vec<SocketAddr>,
    #[arg(long)] pub pool_tokens: Option<u64>,
    #[arg(long="max-context-tokens")] pub max_context: Option<usize>,
    #[arg(long, default_value="auto")] pub rtx_expert_layers: cuteafd_loader::placement::OnboardSetting,
    #[arg(long, default_value_t=16)] pub max_sequences: usize,
    #[arg(long, default_value="0.0.0.0:8000")] pub listen: String,
    #[arg(long)] pub model_id: Option<String>,
    #[arg(long)] pub prefill_rows: Option<usize>,
    #[arg(long)] pub prefill_waves: Option<u8>,
    #[command(flatten)] pub prefix: PrefixArgs,
    #[command(flatten)] pub draft_length: crate::shared::draft::length::DraftLength,
    #[command(flatten)] pub decode_share: crate::shared::prefill_share::DecodeShareArgs,
    #[command(flatten)] pub loop_abort: crate::shared::loop_abort::LoopAbortArgs,
    #[command(flatten)] pub token_io: TokenIoArgs,
    #[command(flatten)] pub api: crate::shared::api::ApiArgs,
}
impl ServeArgs {
    fn prefill(&self) -> Result<cuteafd_loader::serving_capacity::prefill_waves::PrefillPlan> {
        use cuteafd_loader::serving_capacity::prefill_waves::{prefill_shape, PrefillOverrides};
        Ok(prefill_shape(self.peers.len(), cuteafd_loader::plan::Family::prefill_limits(
            &cuteafd_loader::plan::families::inkling2::INKLING2), PrefillOverrides {
            unit_rows: self.prefill_rows.map(|rows| rows as u64), waves: self.prefill_waves.map(u64::from),
            span_rows: None, reserve_rows: None })?)
    }
}

pub(crate) async fn run(mut args: ServeArgs) -> Result<()> {
    use crate::shared::{qwen::scheduler::{self, Backend}, inventory::{InventoryPhase, RuntimeInventory}};
    use cuteafd_api::openai::{ModelEncoding, ModelProfile, NativeRequest, chat::inkling2::InklingEncoding};
    cuteafd_core::concurrency::validate("inkling2", args.max_sequences as u64)?;
    ensure!(args.token_io.embed_placement == crate::shared::token_io::EmbedPlacement::Gpu
        && args.token_io.token_select == crate::shared::token_io::SelectPlacement::Device,
        "Inkling initial serving requires admitted GPU embedding and selection");
    args.prefix.resolve_entries(args.max_sequences);
    let prefill = args.prefill()?;
    let encoding = ModelEncoding::Inkling(Arc::new(InklingEncoding::from_snapshot(&args.snapshot)?));
    let eos = encoding.eos_token_ids();
    ensure!(eos == [200006], "Inkling native EOS differs");
    let model = args.model_id.clone().or_else(|| cuteafd_api::openai::snapshot_model_id(&args.snapshot))
        .context("Inkling snapshot without HF identity requires --model-id")?;
    let profile = ModelProfile::new(model, encoding);
    let api = args.api.load()?;
    let (queue, mut receive) = tokio::sync::mpsc::channel::<NativeRequest>(
        cuteafd_core::concurrency::queue_depth(args.max_sequences as u64) as usize);
    let stats = Arc::new(Mutex::new(serde_json::Value::Null));
    let worker_stats = stats.clone();
    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel::<Result<usize>>();
    let worker_args = args.clone();
    let worker = tokio::task::spawn_blocking(move || -> Result<()> {
        let mut ready = Some(ready_tx);
        let result = (|| {
            let view = worker_args.attention.selected_view.clone().map(Ok)
                .unwrap_or_else(|| worker_args.attention.select(&worker_args.snapshot))?;
            worker_args.attention.validate_view(&view)?;
            let checkpoint = Checkpoint::coordinator_view(&view, false, false)?;
            let cfg = Inkling2Config::from_hf(&checkpoint.config)?;
            cuteafd_core::set_expert_geometry(cuteafd_core::ExpertGeometry::INKLING2)
                .map_err(|previous| anyhow::anyhow!("Inkling process geometry already fixed to {previous:?}"))?;
            // SAFETY: every device, transport and graph owner retires inside
            // this worker before its native library and program table drop.
            let library = Arc::new(unsafe { cuteafd_ffi::NativeLibrary::load(&worker_args.native_lib)? });
            let programs = library.programs()?.with_manifest(&worker_args.manifest)?;
            let (execution, inputs, admitted, _) = super::startup::whole(&library, &programs, &view, &checkpoint, &cfg,
                super::startup::Options { device: worker_args.device, peers: &worker_args.peers,
                    native_lib: &worker_args.native_lib, manifest: &worker_args.manifest,
                    prefill: PrefillShape { rows: prefill.unit_rows, waves: prefill.waves },
                    sequences: worker_args.max_sequences as u64, context_tokens: worker_args.max_context.unwrap_or(0) as u64,
                    pool_tokens: worker_args.pool_tokens, onboard: worker_args.rtx_expert_layers.resolve(cfg.layers),
                    prefix: &worker_args.prefix, resident_weights: crate::shared::memory::budget::resident_weight_limit(),
                    expert_usage: false })?;
            let context = usize::try_from(admitted.solved.served_context())?;
            let family = super::cache::InklingPrefix::new(execution.requests.cache.clone());
            let host = (inputs.host_prefix.quota_bytes > 0).then(|| {
                use cuteafd_engine::prefix::PrefixFamily;
                worker_args.prefix.host_config_for(family.layout(), context, inputs.host_prefix.quota_bytes)
            }).transpose()?;
            let backend = super::backend::InklingBackend::new(execution, context, host, &admitted.solved)?;
            library.cuda_set_device(worker_args.device)?;
            scheduler::schedule(&backend, &worker_args.snapshot, &mut receive, &worker_stats,
                worker_args.max_sequences, scheduler::Drafts::None, worker_args.draft_length, eos,
                worker_args.decode_share, &worker_args.prefix, worker_args.loop_abort.config(),
                worker_args.token_io.token_select, &mut crate::families::qwen3_5::serve::text_media(), None, || {
                    backend.require_live()?;
                    RuntimeInventory::measure(&library, &[worker_args.device], InventoryPhase::PostLoad,
                        |_| Ok(()))?.record("inkling2-serving-loaded", &[]);
                    if let Some(ready) = ready.take() { let _ = ready.send(Ok(context)); }
                    Ok(())
                })
        })();
        crate::shared::serve::forward_initialization(result, ready.is_some(), "Inkling initialization failed", |error|
            ready.take().expect("pending readiness").send(Err(error)).map_err(|result| match result {
                Err(error) => error, Ok(_) => unreachable!("sent error") }))
    });
    let context = ready_rx.await.context("Inkling worker exited before readiness")??;
    let limits = args.prefix.reserve.limits(context)?;
    let snapshot = args.snapshot.clone();
    let hub = crate::shared::console::hub(false, || Ok(crate::shared::console::Layout::new("inkling2", profile.id.clone(), snapshot.clone())));
    let router = cuteafd_api::openai::router_for_model(queue, limits, stats, std::time::Duration::from_secs(25),
        hub.clone(), api.serve(profile, &snapshot)?);
    let listener = tokio::net::TcpListener::bind(&args.listen).await?;
    cuteafd_bench::ready(&listener);
    crate::shared::memory_report::log("ready");
    let mut worker = Some(worker);
    tokio::select! {
        served = crate::shared::api::serve_http(listener, api.app(router, hub), "inkling2") => served?,
        _ = crate::shared::api::watch_scheduler(&mut worker) => unreachable!(),
    }
    crate::shared::api::finish_scheduler(&mut worker, "inkling2").await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn shared_scheduler_owns_lanes_commit_and_last_fallible_readiness() {
        let source = include_str!("serve.rs").split("#[cfg(test)]").next().unwrap();
        assert!(source.contains("scheduler::schedule(&backend"));
        assert!(source.contains("forward_initialization("));
        assert!(source.contains("finish_scheduler(&mut worker"));
        assert!(source.find("InklingEncoding::from_snapshot").unwrap() < source.find("NativeLibrary::load").unwrap());
        let ready = source.find("ready.send(Ok(context))").unwrap();
        assert!(source.find("startup::whole(").unwrap() < ready);
        assert!(source.find("InklingBackend::new(").unwrap() < ready);
        assert!(source.find("backend.require_live()?").unwrap() < ready);
    }
}
