//! Hy4-flash HTTP/request lifecycle (`serve-hy4f`) over the shared scheduler.
use super::backend::Hy4Backend;
use super::execution::Hy4Engine;
use crate::shared::{
    experts::rtx::{exl3::Exl3Tp2, RtxSelectionPolicy},
    inventory::{InventoryPhase, RuntimeInventory},
    memory::device::Device,
    prefix::{budget::HostBudget, PrefixArgs},
    qwen::scheduler::{self, Backend},
    token_io::{EmbedPlacement, SelectPlacement, TokenIoArgs},
};
use anyhow::{ensure, Context, Result};
use cuteafd_api::openai::{chat::hy4_flash::Hy4Encoding, ModelEncoding, ModelProfile, NativeRequest};
use cuteafd_engine::prefix::PrefixFamily;
use cuteafd_loader::{
    families::hy4_flash::{execution::PAGE_ROWS, Hy4FlashConfig, Hy4KvCache},
    plan::Checkpoint,
};
use std::{
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

#[derive(Debug, Clone, clap::Args)]
pub(crate) struct ServeArgs {
    #[command(flatten)]
    pub attention: crate::cli::CoreViewArgs,
    #[arg(long)]
    pub snapshot: PathBuf,
    #[arg(long, env = "CUTEAFD_NATIVE_LIB")]
    pub native_lib: PathBuf,
    #[arg(long, default_value = "/opt/cuteafd/share/PROGRAMS.json")]
    pub manifest: PathBuf,
    #[arg(long, default_value_t = 0)]
    pub device: i32,
    #[arg(long)]
    pub split_device: Option<i32>,
    #[arg(long, value_delimiter = ',')]
    pub peers: Vec<SocketAddr>,
    #[arg(long)]
    pub pool_tokens: Option<u64>,
    #[arg(long = "kv-cache", default_value = "fp8")]
    pub kv_format: Hy4KvCache,
    #[arg(long = "max-context-tokens", value_parser = crate::shared::context::parse_tokens::<usize>)]
    pub max_context: Option<usize>,
    #[arg(long, default_value = "auto")]
    pub rtx_expert_layers: cuteafd_loader::placement::OnboardSetting,
    #[arg(long, default_value_t = 16)]
    pub max_sequences: usize,
    #[arg(long, default_value = "0.0.0.0:8000")]
    pub listen: String,
    #[arg(long)]
    pub model_id: Option<String>,
    #[arg(long)]
    pub prefill_rows: Option<usize>,
    #[arg(long, value_parser = clap::value_parser!(u8).range(1..=3))]
    pub prefill_waves: Option<u8>,
    /// Force BF16 Engram fallback tables instead of `engram_fp8`.
    #[arg(long)]
    pub engram_bf16: bool,
    /// Optional explicit path to `exl3-hy4f-k45-mul1/rtx-tp{1,2}` package directory.
    #[arg(long)]
    pub expert_package: Option<PathBuf>,
    #[command(flatten)]
    pub prefix: PrefixArgs,
    #[command(flatten)]
    pub draft_length: crate::shared::draft::length::DraftLength,
    #[command(flatten)]
    pub decode_share: crate::shared::prefill_share::DecodeShareArgs,
    #[command(flatten)]
    pub loop_abort: crate::shared::loop_abort::LoopAbortArgs,
    #[command(flatten)]
    pub token_io: TokenIoArgs,
    #[command(flatten)]
    pub api: crate::shared::api::ApiArgs,
}

impl ServeArgs {
    fn prefill(&self) -> Result<cuteafd_loader::serving_capacity::prefill_waves::PrefillPlan> {
        use cuteafd_loader::serving_capacity::prefill_waves::{prefill_shape, PrefillOverrides};
        Ok(prefill_shape(
            self.peers.len(),
            cuteafd_loader::plan::Family::prefill_limits(&cuteafd_loader::plan::families::hy4_flash::HY4_FLASH),
            PrefillOverrides {
                unit_rows: self.prefill_rows.map(|rows| rows as u64),
                waves: self.prefill_waves.map(u64::from),
                span_rows: None,
                reserve_rows: None,
            },
        )?)
    }
}

fn resolve_expert_package(native_lib: &Path, explicit: Option<&Path>, gpus: usize, catalog: &cuteafd_loader::OfficialV41Catalog) -> Result<PathBuf> {
    if let Some(path) = explicit {
        return Ok(path.to_path_buf());
    }
    let exl3 = catalog.exl3().context("Hy4-flash checkpoint has no EXL3 routed expert catalog")?;
    let base = native_lib.parent().context("Hy4 native library directory absent")?;
    Ok(base.join("exl3").join(exl3.package_name("hy4f")).join(format!("rtx-tp{gpus}")))
}

pub(crate) async fn run(mut args: ServeArgs) -> Result<()> {
    cuteafd_core::concurrency::validate("hy4_flash", args.max_sequences as u64)?;
    ensure!(
        args.token_io.embed_placement == EmbedPlacement::Gpu && args.token_io.token_select == SelectPlacement::Device,
        "Hy4-flash serving requires GPU embedding and device token selection"
    );
    ensure!(
        args.peers.is_empty(),
        "Hy4-flash Spark --peers remote expert serving is wired in Step 4D; 0-Spark RTX cards (1x or 2x RTX PRO 6000) are active"
    );
    args.prefix.resolve_entries(args.max_sequences);
    let prefill = args.prefill()?;
    cuteafd_bench::context::ResolvedSettings::prefill(prefill).publish();

    let encoding = ModelEncoding::Hy4Flash(Arc::new(Hy4Encoding::from_snapshot(&args.snapshot)?));
    let eos = encoding.eos_token_ids();
    ensure!(!eos.is_empty(), "Hy4-flash checkpoint resolved empty EOS set");
    let model = args
        .model_id
        .clone()
        .or_else(|| cuteafd_api::openai::snapshot_model_id(&args.snapshot))
        .context("Hy4-flash snapshot without HF identity requires --model-id")?;
    let profile = ModelProfile::new(model, encoding);
    let api = args.api.load()?;
    let (queue, mut receive) = tokio::sync::mpsc::channel::<NativeRequest>(
        cuteafd_core::concurrency::queue_depth(args.max_sequences as u64) as usize,
    );
    let stats = Arc::new(Mutex::new(serde_json::Value::Null));
    let worker_stats = stats.clone();
    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel::<Result<usize>>();
    let worker_args = args.clone();

    let worker = tokio::task::spawn_blocking(move || -> Result<()> {
        let mut ready = Some(ready_tx);
        let result = (|| {
            let view = worker_args
                .attention
                .selected_view
                .clone()
                .map(Ok)
                .unwrap_or_else(|| worker_args.attention.select(&worker_args.snapshot))?;
            worker_args.attention.validate_view(&view)?;
            let checkpoint = Checkpoint::open_view(&view)?;
            let cfg = Hy4FlashConfig::from_hf(&checkpoint.config)?;
            cfg.release_geometry()?;
            cuteafd_bench::context::set_family("hy4_flash");
            if let Err(prev) = cuteafd_core::set_expert_geometry(cuteafd_core::ExpertGeometry::HY4_FLASH) {
                ensure!(
                    prev == cuteafd_core::ExpertGeometry::HY4_FLASH,
                    "Hy4-flash process geometry already fixed to {prev:?}"
                );
            }

            // SAFETY: every device owner retires inside this worker before NativeLibrary drops.
            let library = Arc::new(unsafe { cuteafd_ffi::NativeLibrary::load(&worker_args.native_lib)? });
            let programs = library.programs()?.with_manifest(&worker_args.manifest)?;

            let mut device_ids = vec![worker_args.device];
            if let Some(second) = worker_args.split_device {
                ensure!(second != worker_args.device, "--split-device must differ from --device");
                device_ids.push(second);
            }
            let gpus = device_ids.len();
            let devices: Vec<Device<'_>> = device_ids.iter().map(|&id| Device { library: &library, id }).collect();
            let unit_rows = usize::try_from(prefill.unit_rows)?;
            let max_context = match worker_args.max_context {
                Some(0) | None => cfg.max_position,
                Some(n) => n.min(cfg.max_position),
            };
            let catalog = cuteafd_loader::read_expert_catalog_view(&view)?;
            let expert_pkg = resolve_expert_package(
                &worker_args.native_lib,
                worker_args.expert_package.as_deref(),
                gpus,
                &catalog,
            )?;
            let routed_layers = cfg.routed_layers();
            let budget = usize::try_from(crate::shared::memory::budget::resident_weight_limit().0)
                .unwrap_or(80usize << 30);

            let pre = RuntimeInventory::measure(&library, &device_ids, InventoryPhase::PreWeight, |_| Ok(()))?;
            let min_free = pre.gpus.iter().map(|g| g.free_bytes).min().unwrap_or(0);
            let policy = cuteafd_core::memory_policy::MemoryPolicy::current();
            let effective_cap = pre
                .gpus
                .iter()
                .map(|g| policy.gpu_cap_bytes(g.total_bytes).min(g.free_bytes))
                .min()
                .unwrap_or(min_free);

            let onboarded: Vec<Option<Exl3Tp2<'_>>> = if gpus == 1 {
                let w = Exl3Tp2::load_whole_weights(
                    devices[0],
                    &catalog,
                    &expert_pkg,
                    routed_layers.clone(),
                    unit_rows,
                    budget,
                    RtxSelectionPolicy::FirstCapacity,
                )?;
                vec![Some(w.execution()?)]
            } else {
                let [w0, w1] = Exl3Tp2::load_pair_weights(
                    [devices[0], devices[1]],
                    &catalog,
                    cuteafd_core::ExpertGeometry::HY4_FLASH,
                    [expert_pkg.as_path(), expert_pkg.as_path()],
                    routed_layers.clone(),
                    unit_rows,
                    [budget, budget],
                    RtxSelectionPolicy::FirstCapacity,
                )?;
                vec![Some(w0.execution()?), Some(w1.execution()?)]
            };

            // Measure remaining GPU free memory after routed experts, then size DSA KV pool safely.
            let post_exp = RuntimeInventory::measure(&library, &device_ids, InventoryPhase::PreWeight, |_| Ok(()))?;
            let after_exp_free = post_exp
                .gpus
                .iter()
                .map(|g| g.free_bytes.min(effective_cap))
                .min()
                .unwrap_or(0);
            let model = cuteafd_loader::plan::families::hy4_flash::Model::open(&checkpoint)?;
            let modes = cuteafd_loader::placement::families::hy4_flash::layer_modes(
                &cfg,
                gpus,
                cuteafd_loader::placement::AttentionPlacement::Heads,
            );
            let target_layouts = cuteafd_loader::families::hy4_flash::weight_layout::target_arenas(
                &checkpoint,
                &cfg,
                &model.core,
                &modes,
                gpus,
            )?;
            let target_bytes = target_layouts.iter().map(|l| l.arena.bytes() as u64).max().unwrap_or(0);
            let swa_ring_pages = ((cfg.sliding_window + unit_rows).div_ceil(PAGE_ROWS) + 1)
                .max(cuteafd_loader::families::hy4_flash::execution::SWA_RING_PAGES);
            let swa_seq_bytes = (cfg.swa_layers().len() * swa_ring_pages * PAGE_ROWS * worker_args.kv_format.record_bytes()) as u64;
            let mark_bytes = swa_seq_bytes + (cfg.sparse_layers().len() * super::cache::PENDING_KEY_SLOT_BYTES) as u64;
            let mark_slots = if worker_args.prefix.prefix_cache_entries == 0 {
                0
            } else {
                cuteafd_core::prefix::mark_slots_with_host(
                    worker_args.max_sequences as u64,
                    worker_args.prefix.prefix_cache_entries as u64,
                    mark_bytes,
                    worker_args.prefix.prefix_cache_mark_mib as u64 * (1 << 20),
                    worker_args.prefix.host_cache_bytes != HostBudget::Bytes(0),
                ) as usize
            };
            let reserve_and_ws: u64 = target_bytes
                + swa_seq_bytes * worker_args.max_sequences as u64
                + mark_bytes * mark_slots as u64
                + (4u64 << 30); // workspaces + selector + head + CUDA reserve
            let kv_avail = after_exp_free.saturating_sub(reserve_and_ws);
            let page_bytes_per_rank = (cfg.sparse_layers().len()
                * (PAGE_ROWS * worker_args.kv_format.record_bytes()
                    + (PAGE_ROWS / cfg.indexer_compress_ratio) * cuteafd_loader::families::hy4_flash::execution::INDEX_GROUP_RECORD_BYTES)) as u64;
            let max_fit_pages = (kv_avail / page_bytes_per_rank.max(1)) as usize;
            let min_pages = max_context.div_ceil(PAGE_ROWS);
            let chosen_pages = match worker_args.pool_tokens {
                Some(tokens) if tokens > 0 => {
                    let req = usize::try_from(tokens.div_ceil(PAGE_ROWS as u64))?;
                    ensure!(
                        req <= max_fit_pages,
                        "Hy4 requested pool {tokens} tokens ({req} pages) exceeds fitted GPU capacity ({max_fit_pages} pages)"
                    );
                    req
                }
                _ => {
                    let cap_pages = policy
                        .kv_max
                        .map(|t| usize::try_from(t.div_ceil(PAGE_ROWS as u64)).unwrap_or(max_fit_pages))
                        .unwrap_or(max_fit_pages);
                    max_fit_pages.min(cap_pages)
                }
            };
            ensure!(
                chosen_pages >= min_pages.min(256),
                "Hy4 fitted KV pool ({chosen_pages} pages) is below minimum usable context"
            );
            let served_context = max_context.min(chosen_pages * PAGE_ROWS);
            let pool_tokens = (chosen_pages * PAGE_ROWS) as u64;

            let engine = Hy4Engine::new_serving(
                &devices,
                &programs,
                &checkpoint,
                cfg.clone(),
                worker_args.kv_format,
                unit_rows,
                chosen_pages,
                worker_args.max_sequences,
                served_context,
                mark_slots,
                worker_args.engram_bf16,
                false,
                onboarded,
            )?;

            let family = engine.prefix()?;
            let host_total = cuteafd_loader::formats::mapped_preload::host_memory_before_process()
                .map(|(total, _)| total)
                .unwrap_or(0);
            let host_quota = match worker_args.prefix.host_cache_bytes {
                HostBudget::Bytes(0) => 0,
                HostBudget::Bytes(b) => b,
                HostBudget::Auto => {
                    let layout = family.layout();
                    let geom = cuteafd_core::prefix::HostCacheGeometry {
                        unit_rows: PAGE_ROWS as u64,
                        page_bytes: layout.page_bytes as u64,
                        mark_bytes: layout.mark_bytes as u64,
                        draft_bytes: 0,
                    };
                    geom.resolve(None, served_context as u64, worker_args.max_sequences as u64, host_total, geom.chunk_bytes())
                        .unwrap_or(0)
                }
            };
            let host = (host_quota > 0 && worker_args.prefix.prefix_cache_entries > 0)
                .then(|| worker_args.prefix.host_config_for(family.layout(), served_context, host_quota))
                .transpose()?;

            cuteafd_bench::context::ResolvedSettings::default()
                .value("max-context-tokens", served_context)
                .value("pool-tokens", pool_tokens)
                .value("rtx-expert-layers", routed_layers.len())
                .value("prefill-rows", unit_rows)
                .value("prefill-waves", prefill.waves)
                .publish();

            let backend = Hy4Backend::new(engine, served_context, unit_rows, host)?;
            let mut media_admission = cuteafd_engine::media::MediaAdmission::new(
                cuteafd_engine::media::EmbeddingCache::new(0),
                crate::shared::qwen::Encoder::Off,
                worker_args.max_sequences,
            );
            library.cuda_set_device(worker_args.device)?;
            scheduler::schedule(
                &backend,
                &worker_args.snapshot,
                &mut receive,
                &worker_stats,
                worker_args.max_sequences,
                scheduler::Drafts::None,
                worker_args.draft_length,
                eos,
                worker_args.decode_share,
                &worker_args.prefix,
                worker_args.loop_abort.config(),
                worker_args.token_io.token_select,
                &mut media_admission,
                None,
                || {
                    backend.require_live()?;
                    RuntimeInventory::measure(&library, &device_ids, InventoryPhase::PostLoad, |_| Ok(()))?
                        .record("hy4_flash-serving-loaded", &[]);
                    if let Some(ready) = ready.take() {
                        let _ = ready.send(Ok(served_context));
                    }
                    Ok(())
                },
            )
        })();
        crate::shared::serve::forward_initialization(
            result,
            ready.is_some(),
            "Hy4-flash initialization failed",
            |error| {
                ready
                    .take()
                    .expect("pending readiness")
                    .send(Err(error))
                    .map_err(|r| match r {
                        Err(e) => e,
                        Ok(_) => unreachable!("sent error"),
                    })
            },
        )
    });

    let context = ready_rx
        .await
        .context("Hy4-flash worker exited before readiness")??;
    let limits = args.prefix.reserve.limits(context)?;
    let snapshot = args.snapshot.clone();
    let hub = crate::shared::console::hub(false, || {
        Ok(crate::shared::console::Layout::new(
            "hy4_flash",
            profile.id.clone(),
            snapshot.clone(),
        ))
    });
    let router = cuteafd_api::openai::router_for_model(
        queue,
        limits,
        stats,
        std::time::Duration::from_secs(25),
        hub.clone(),
        api.serve(profile, &snapshot)?,
    );
    let listener = tokio::net::TcpListener::bind(&args.listen).await?;
    cuteafd_bench::ready(&listener);
    crate::shared::memory_report::log("ready");
    let mut worker = Some(worker);
    tokio::select! {
        served = crate::shared::api::serve_http(listener, api.app(router, hub), "hy4_flash") => served?,
        _ = crate::shared::api::watch_scheduler(&mut worker) => unreachable!(),
    }
    crate::shared::api::finish_scheduler(&mut worker, "hy4_flash").await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    #[test]
    fn serve_hy4f_cli_and_resolved_seam_match() -> anyhow::Result<()> {
        let argv = [
            "cuteafd",
            "--gpu-memory-cap",
            "97250MiB",
            "serve-hy4f",
            "--snapshot",
            "/model",
            "--native-lib",
            "/opt/cuteafd/lib/libcuteafd_native.so",
            "--device",
            "0",
            "--split-device",
            "1",
            "--listen",
            "0.0.0.0:8823",
            "--max-sequences",
            "16",
            "--max-context-tokens",
            "auto",
            "--core-view",
            "original",
            "--prefill-rows",
            "2048",
            "--prefill-waves",
            "1",
        ];
        let crate::cli::Commands::ServeHy4f(mut args) = crate::cli::Cli::try_parse_from(argv)?.command else {
            anyhow::bail!("Hy4-flash command not parsed");
        };
        assert_eq!((args.max_context, args.max_sequences, args.split_device), (Some(0), 16, Some(1)));
        crate::commands::bench::assert_resolved_family_seam("serve-hy4f", &[], || {
            args.prefix.resolve_entries(args.max_sequences);
            let prefill = args.prefill()?;
            cuteafd_bench::context::ResolvedSettings::default()
                .value("max-context-tokens", 262144)
                .value("pool-tokens", 262144)
                .value("rtx-expert-layers", 41)
                .value("prefill-rows", prefill.unit_rows)
                .value("prefill-waves", prefill.waves)
                .publish();
            Ok(())
        })?;
        Ok(())
    }

    #[test]
    fn shared_scheduler_owns_lanes_and_typed_readiness() {
        let source = include_str!("serve.rs").split("#[cfg(test)]").next().unwrap();
        assert!(source.contains("scheduler::schedule("));
        assert!(source.contains("Checkpoint::open_view(&view)?"));
        assert!(source.contains("context::set_family(\"hy4_flash\")"));
        assert!(source.contains("shared::serve::forward_initialization("));
        assert!(source.contains("finish_scheduler(&mut worker"));
        assert!(source.find("Hy4Encoding::from_snapshot").unwrap() < source.find("NativeLibrary::load").unwrap());
    }
}
