//! Hy4-flash coordinator golden runner (`hy4f-golden`) for raptor RTX GPUs (#2478).
//!
//! Compares against the published Hy4-flash fidelity set (`0b1a914773`,
//! `hy4-flash-v1_20261118_own-text-bf16-f32`, agg. NLL `1.283684105386`) and
//! `python/reference/families/hy4_flash`, verifies Engram FP8 vs BF16 bitwise
//! identity and unit N=1 vs N byte-exactness.
use super::execution::Hy4Engine;
use crate::shared::golden_scores::{scored_nll, ReferenceRows};
use crate::shared::memory::device::Device;
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::NativeLibrary;
use cuteafd_loader::families::hy4_flash::{Hy4FlashConfig, Hy4KvCache};
use cuteafd_loader::plan::checkpoint::{Checkpoint, CheckpointView};
use cuteafd_loader::plan::families::hy4_flash as hy4_plan;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, clap::Args)]
pub(crate) struct GoldenArgs {
    #[command(flatten)]
    pub(crate) attention: crate::cli::CoreViewArgs,
    /// KV cache storage format (matches serving default `fp8`).
    #[arg(long, default_value = "fp8")]
    pub(crate) kv_cache: Hy4KvCache,
    /// Checkpoint snapshot directory (e.g. Hy4-flash-EXL3-K4-MC).
    #[arg(long)]
    pub(crate) snapshot: PathBuf,
    #[arg(long, env = "CUTEAFD_NATIVE_LIB")]
    pub(crate) native_lib: Option<PathBuf>,
    #[arg(long, default_value = "/opt/cuteafd/share/PROGRAMS.json")]
    pub(crate) manifest: PathBuf,
    /// Directory containing `tokens.bin` (and optional `logits.bin` reference export).
    #[arg(long)]
    pub(crate) golden: Option<PathBuf>,
    /// Optional window subdirectory under `--golden`.
    #[arg(long)]
    pub(crate) window: Option<String>,
    /// Last target layer inclusive (0 for layer-0 dense check; 41 for full logits).
    #[arg(long, default_value_t = 41)]
    pub(crate) last_layer: usize,
    /// Primary CUDA device index (0 or 1 on raptor).
    #[arg(long, default_value_t = 0)]
    pub(crate) device: i32,
    /// Optional second CUDA device for 2x RTX TP2 coordinator execution.
    #[arg(long)]
    pub(crate) split_device: Option<i32>,
    /// Validate checkpoint headers, MC views, Engram FP8 shards, and target arenas without CUDA.
    #[arg(long)]
    pub(crate) check_headers: bool,
    /// Compute NLL/KL/top-1 against reference tokens and optional logits.
    #[arg(long)]
    pub(crate) nll: bool,
    /// Force BF16 Engram fallback tables instead of `engram_fp8` (for bitwise-identity verification).
    #[arg(long)]
    pub(crate) engram_bf16: bool,
    /// Chunk size (rows per unit) for unit-exactness verification (`N=1` vs chunked).
    #[arg(long)]
    pub(crate) prefill_units: Option<usize>,
    /// Save FP32 logits (`logits.bin`) to this directory.
    #[arg(long, hide = true)]
    pub(crate) save_logits: Option<PathBuf>,
    /// Diagnostic: zero routed experts to isolate non-expert backbone (iHC, Engram, MLA, indexer, shared FFN).
    #[arg(long, hide = true)]
    pub(crate) skip_routed_experts: bool,
    /// Optional path to `hy4f:exl3-k45-mul1` routed expert package directory.
    #[arg(long)]
    pub(crate) expert_package: Option<PathBuf>,
}

fn read_u32_bin(path: &Path) -> Result<Vec<u32>> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    ensure!(bytes.len() % 4 == 0 && !bytes.is_empty(), "invalid u32 binary {}", path.display());
    Ok(bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}

fn digest_f32(values: &[f32]) -> String {
    let mut h = Sha256::new();
    for v in values {
        h.update(v.to_le_bytes());
    }
    format!("{:x}", h.finalize())
}

fn load_published_hy4_reference(root: &Path, vocab: usize) -> Result<cuteafd_bench::reference::Reference> {
    use cuteafd_bench::fidelity_dataset;
    let bytes = std::fs::read(root.join("manifest.json"))
        .with_context(|| format!("reading {}", root.join("manifest.json").display()))?;
    let manifest: serde_json::Value = serde_json::from_slice(&bytes)?;
    let digest = format!("{:x}", Sha256::digest(&bytes));
    ensure!(
        manifest["config"].as_str() == Some(fidelity_dataset::HY4_CONFIG)
            && digest == fidelity_dataset::HY4_MANIFEST_SHA256,
        "Hy4-flash fidelity manifest mismatch: config={:?} sha256={digest}",
        manifest["config"]
    );
    let reference = fidelity_dataset::load(root, &manifest)?;
    ensure!(reference.vocab == vocab, "Hy4-flash fidelity vocab {} != {vocab}", reference.vocab);
    Ok(reference)
}

fn build_probe_row(position: usize, logits: &[f32], wanted_ids: &[u32]) -> cuteafd_api::openai::probe::ProbeRow {
    let finite = logits.iter().all(|v| v.is_finite());
    let mut max_v = f32::NEG_INFINITY;
    let mut argmax = 0u32;
    for (i, &v) in logits.iter().enumerate() {
        if v > max_v {
            max_v = v;
            argmax = i as u32;
        }
    }
    let max_f64 = f64::from(max_v);
    let sum_exp: f64 = logits.iter().map(|&v| (f64::from(v) - max_f64).exp()).sum();
    let lse = max_f64 + sum_exp.ln();
    let wanted: Vec<(u32, f32)> = wanted_ids
        .iter()
        .map(|&id| (id, (f64::from(logits[id as usize]) - lse) as f32))
        .collect();
    let mut fnv = 0xcbf2_9ce4_8422_2325u64;
    for &v in logits {
        for b in v.to_le_bytes() {
            fnv ^= u64::from(b);
            fnv = fnv.wrapping_mul(0x0100_0000_01b3);
        }
    }
    cuteafd_api::openai::probe::ProbeRow {
        position,
        hash: format!("{fnv:016x}"),
        argmax,
        top: Vec::new(),
        wanted,
        finite,
    }
}

fn check_headers_only(args: &GoldenArgs) -> Result<()> {
    let view = args.attention.core_view;
    let selected = CheckpointView::select(&args.snapshot, view)?;
    let checkpoint = Checkpoint::open_view(&selected)?;
    let cfg = Hy4FlashConfig::read(&args.snapshot)?;
    let model = hy4_plan::Model::open(&checkpoint)?;
    let gpus = if args.split_device.is_some() { 2 } else { 1 };
    let modes = cuteafd_loader::placement::families::hy4_flash::layer_modes(
        &cfg,
        gpus,
        cuteafd_loader::placement::AttentionPlacement::Heads,
    );
    let arenas = cuteafd_loader::families::hy4_flash::weight_layout::target_arenas(
        &checkpoint,
        &cfg,
        &model.core,
        &modes,
        gpus,
    )?;
    let total_arena_bytes: usize = arenas.iter().map(|a| a.arena.bytes()).sum();
    println!(
        "hy4f-golden check-headers OK: view={view:?} gpus={} layers={} dsa={} swa={} engram_fp8={} target_arena_bytes={}",
        gpus,
        cfg.layers,
        cfg.sparse_layers().len(),
        cfg.swa_layers().len(),
        model.has_engram_fp8,
        total_arena_bytes,
    );
    Ok(())
}

pub(crate) async fn run_golden(args: GoldenArgs) -> Result<()> {
    if args.check_headers {
        return check_headers_only(&args);
    }
    let view = args.attention.core_view;
    let selected = CheckpointView::select(&args.snapshot, view)?;
    let checkpoint = Checkpoint::open_view(&selected)?;
    let cfg = Hy4FlashConfig::read(&args.snapshot)?;

    let golden_dir = args
        .golden
        .as_ref()
        .context("hy4f-golden requires --golden <DIR> (unless --check-headers)")?;

    // Two reference layouts: published cuteafd-fidelity dataset (manifest.json + *.safetensors)
    // or raw window directory (tokens.bin + optional logits.bin).
    let published = if golden_dir.join("manifest.json").is_file() {
        Some(load_published_hy4_reference(golden_dir, cfg.vocab_size)?)
    } else {
        None
    };
    let (selected_windows, raw_tokens, raw_window_dir) = if let Some(ref_set) = &published {
        let wanted: Vec<String> = match &args.window {
            Some(w) => w.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect(),
            None => ref_set.quick_windows.clone(),
        };
        let mut wins = Vec::with_capacity(wanted.len());
        for id in &wanted {
            let win = ref_set
                .windows
                .iter()
                .find(|w| w.id == *id)
                .with_context(|| format!("Hy4-flash fidelity window {id:?} not found"))?;
            wins.push(win.clone());
        }
        ensure!(!wins.is_empty(), "Hy4-flash fidelity run selected zero windows");
        (Some(wins), Vec::new(), golden_dir.clone())
    } else {
        let window_dir = match &args.window {
            Some(w) => golden_dir.join(w),
            None => golden_dir.clone(),
        };
        let tokens = read_u32_bin(&window_dir.join("tokens.bin"))?;
        (None, tokens, window_dir)
    };

    let max_seq = selected_windows
        .as_ref()
        .map(|ws| ws.iter().map(|w| w.tokens.len()).max().unwrap_or(1))
        .unwrap_or(raw_tokens.len());
    let unit = args.prefill_units.unwrap_or(max_seq).min(4096);
    let max_pages = max_seq.div_ceil(64).max(16);

    let lib_path = args
        .native_lib
        .as_deref()
        .unwrap_or_else(|| Path::new("/opt/cuteafd/lib/libcuteafd_native.so"));
    // SAFETY: trusted native library built by this repository's toolchain.
    let library = unsafe { NativeLibrary::load(lib_path)? };
    let programs = library.programs()?.with_manifest(&args.manifest)?;

    let mut device_ids = vec![args.device];
    if let Some(second) = args.split_device {
        ensure!(second != args.device, "--split-device must differ from --device");
        device_ids.push(second);
    }
    let devices: Vec<Device<'_>> = device_ids.iter().map(|&id| Device { library: &library, id }).collect();
    ensure!(
        args.expert_package.is_none() || !args.skip_routed_experts,
        "--expert-package conflicts with --skip-routed-experts"
    );
    let onboarded: Vec<Option<crate::shared::experts::rtx::exl3::Exl3Tp2<'_>>> =
        if let Some(pkg) = &args.expert_package {
            use crate::shared::experts::rtx::{exl3::Exl3Tp2, RtxSelectionPolicy};
            cuteafd_core::set_expert_geometry(cuteafd_core::ExpertGeometry::HY4_FLASH)
                .map_err(|prev| anyhow::anyhow!("expert geometry already set to {prev:?}"))?;
            let catalog = cuteafd_loader::read_expert_catalog_view(
                checkpoint.view.as_ref().context("Hy4-flash checkpoint view required for routed experts")?,
            )?;
            let layers: Vec<usize> = cfg
                .routed_layers()
                .into_iter()
                .filter(|&l| l <= args.last_layer)
                .collect();
            let budget = 80usize << 30;
            if devices.len() == 1 {
                let w = Exl3Tp2::load_whole_weights(
                    devices[0],
                    &catalog,
                    pkg,
                    layers,
                    unit,
                    budget,
                    RtxSelectionPolicy::FirstCapacity,
                )?;
                vec![Some(w.execution()?)]
            } else {
                let [w0, w1] = Exl3Tp2::load_pair_weights(
                    [devices[0], devices[1]],
                    &catalog,
                    cuteafd_core::ExpertGeometry::HY4_FLASH,
                    [pkg.as_path(), pkg.as_path()],
                    layers,
                    unit,
                    [budget, budget],
                    RtxSelectionPolicy::FirstCapacity,
                )?;
                vec![Some(w0.execution()?), Some(w1.execution()?)]
            }
        } else {
            (0..devices.len()).map(|_| None).collect()
        };

    let mut engine = Hy4Engine::new(
        &devices,
        &programs,
        &checkpoint,
        cfg.clone(),
        args.kv_cache,
        unit,
        max_pages,
        args.engram_bf16,
        args.skip_routed_experts,
        onboarded,
    )?;

    let vocab = cfg.vocab_size;
    if let Some(windows) = selected_windows {
        let mut all_records = Vec::new();
        for win in &windows {
            engine.reset()?;
            let mut logits = Vec::with_capacity(win.tokens.len() * vocab);
            for chunk in win.tokens.chunks(unit) {
                let part = engine.forward_chunk(chunk, args.last_layer)?;
                logits.extend_from_slice(&part);
            }
            let want = win.want();
            let mut rows = Vec::with_capacity(win.positions.len());
            for pos in &win.positions {
                let row_idx = pos.pos - 1;
                let slice = &logits[row_idx * vocab..(row_idx + 1) * vocab];
                rows.push(build_probe_row(pos.pos, slice, &want[&pos.pos]));
            }
            let score = win.score(&rows);
            println!(
                "hy4f-golden window={} rows={} digest={} mean NLL={:.6} (ref {:.6}) mean KL={:.6} top-1={:.2}% top-3={:.2}%",
                win.id,
                score.positions,
                digest_f32(&logits),
                score.nll,
                score.ref_nll,
                score.kl,
                score.top1 * 100.0,
                score.top3_contained * 100.0,
            );
            all_records.extend(score.records);
        }
        let agg = cuteafd_bench::reference::Fidelity::from_records(all_records);
        println!(
            "hy4f-golden aggregate ({} windows, {} rows): mean NLL={:.6} (ref {:.6}, delta {:+.6}) mean KL={:.6} top-1={:.2}% confident-top-1={:.2}% top-3={:.2}%",
            windows.len(),
            agg.positions,
            agg.nll,
            agg.ref_nll,
            agg.nll - agg.ref_nll,
            agg.kl,
            agg.top1 * 100.0,
            agg.confident_top1.unwrap_or(0.0) * 100.0,
            agg.top3_contained * 100.0,
        );
        return Ok(());
    }

    let tokens = raw_tokens;
    let window_dir = raw_window_dir;
    let mut logits = Vec::new();
    for chunk in tokens.chunks(unit) {
        let part = engine.forward_chunk(chunk, args.last_layer)?;
        logits.extend_from_slice(&part);
    }

    if args.last_layer + 1 < cfg.layers {
        let res = engine.read_residual_f32(tokens.len().min(unit))?;
        println!(
            "hy4f-golden layer {} complete: rows={} residual_digest={}",
            args.last_layer,
            tokens.len().min(unit),
            digest_f32(&res)
        );
        return Ok(());
    }

    let digest = digest_f32(&logits);
    println!(
        "hy4f-golden logits digest={} rows={} vocab={} unit={} engram={}",
        digest,
        tokens.len(),
        vocab,
        unit,
        if args.engram_bf16 { "bf16" } else { "engram_fp8" }
    );

    if let Some(out_dir) = &args.save_logits {
        std::fs::create_dir_all(out_dir)?;
        let out_bytes: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(out_dir.join("logits.bin"), out_bytes)?;
    }

    if args.nll {
        let reference = ReferenceRows::open(&window_dir.join("logits.bin"), tokens.len(), vocab, true)?;
        let mean_nll = if tokens.len() > 1 {
            scored_nll(&logits[..(tokens.len() - 1) * vocab], &tokens[1..], vocab)?
        } else {
            0.0
        };
        let nll_rows = tokens.len().saturating_sub(1);
        let mut top1_matches = 0usize;
        let mut ref_rows = 0usize;
        if let Some(ref_reader) = &reference {
            for row in 0..tokens.len() {
                let slice = &logits[row * vocab..(row + 1) * vocab];
                let ref_row = ref_reader.row(row)?;
                let got_top = slice
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                let ref_top = ref_row
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                if got_top == ref_top {
                    top1_matches += 1;
                }
                ref_rows += 1;
            }
        }
        println!(
            "hy4f-golden mean NLL={mean_nll:.6} ({nll_rows} scored rows); top-1 agreement={top1_matches}/{ref_rows}"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    #[test]
    fn hy4f_golden_cli_parses() {
        let cli = crate::cli::Cli::try_parse_from([
            "cuteafd",
            "hy4f-golden",
            "--snapshot",
            "/snapshot",
            "--golden",
            "/golden",
            "--nll",
            "--split-device",
            "1",
            "--engram-bf16",
            "--skip-routed-experts",
        ])
        .expect("hy4f-golden CLI parse");
        let crate::cli::Commands::Hy4fGolden(args) = cli.command else {
            panic!("expected Hy4fGolden");
        };
        assert!(args.nll);
        assert_eq!(args.split_device, Some(1));
        assert!(args.engram_bf16);
        assert!(args.skip_routed_experts);
        assert_eq!(args.last_layer, 41);
    }

    #[test]
    #[ignore = "requires published Hy4-flash fidelity dataset in ~/.cache/cuteafd/fidelity"]
    fn installed_hy4_published_fidelity_dataset_validates() -> anyhow::Result<()> {
        let root = std::path::PathBuf::from(std::env::var("HOME")?)
            .join(".cache/cuteafd/fidelity/wrldsuksgo2mars--cuteafd-fidelity")
            .join(cuteafd_bench::fidelity_dataset::HY4_REVISION)
            .join(cuteafd_bench::fidelity_dataset::HY4_CONFIG);
        let reference = super::load_published_hy4_reference(&root, 120_832)?;
        assert_eq!(reference.windows.len(), 70);
        assert!(reference.quick_windows.iter().any(|w| w == "a00"));
        let mut nll_sum = 0.0f64;
        let mut nll_rows = 0usize;
        for w in &reference.windows {
            for p in &w.positions {
                nll_sum += -p.next_lp;
                nll_rows += 1;
            }
        }
        assert_eq!(nll_rows, 35_840);
        assert!((nll_sum / nll_rows as f64 - 1.283684105385722).abs() < 1e-6);
        Ok(())
    }
}
