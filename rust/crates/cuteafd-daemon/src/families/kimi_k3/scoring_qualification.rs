//! Ignored full-model proof: ordinary C16 admission, raw head rows versus the
//! serving backend's bounded scoring, with exact committed-state comparison.
use super::{backend::KimiBackend, device::SequenceRows, prefix::Placement, startup};
use crate::shared::{probe::{ProbeRef, ScoreLogits, ScoringRows}, qwen::scheduler::Backend};
use anyhow::{ensure, Context, Result};
use clap::Parser;
use cuteafd_api::openai::probe::{Probe, ProbeSpec};
use cuteafd_engine::prefix::PrefixFamily;
use cuteafd_loader::families::kimi_k3::config::KimiK3Config;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{path::Path, sync::Arc};

fn steps(tokens: usize, from: usize, capacity: usize) -> Result<Vec<std::ops::Range<usize>>> {
    ensure!(tokens >= 2 && from > 0 && from < tokens && capacity > 0, "invalid paired score cuts");
    let mut spans = Vec::new();
    // Same prompt/scored boundary as shared::probe::score_plan; head tiling
    // must not change either transformer's causal unit cuts.
    for (first, end) in [(0, from), (from, tokens - 1)] {
        for start in (first..end).step_by(capacity) {
            spans.push(start..start.saturating_add(capacity).min(end));
        }
    }
    Ok(spans)
}

fn compare_rows(raw: &Path, streamed: &Path, positions: &[usize], vocab: usize) -> Result<Value> {
    let a = std::fs::read(raw.join("manifest.jsonl"))?;
    let b = std::fs::read(streamed.join("manifest.jsonl"))?;
    ensure!(a == b, "paired row positions/metadata differ");
    let rows = std::str::from_utf8(&a)?.lines().map(serde_json::from_str::<Value>)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    ensure!(rows.len() == positions.len(), "paired row coverage differs");
    let mut hash = Sha256::new();
    for (row, &position) in rows.iter().zip(positions) {
        ensure!(row["position"].as_u64() == Some(position as u64)
            && row["vocab_size"].as_u64() == Some(vocab as u64), "paired row identity differs");
        let name = row["file"].as_str().context("paired row filename")?;
        ensure!(Path::new(name).components().count() == 1 && name.starts_with("row-")
            && name.ends_with(".safetensors"), "unsafe paired row path");
        let a = std::fs::read(raw.join(name))?;
        let b = std::fs::read(streamed.join(name))?;
        ensure!(a == b, "paired full-vocabulary F32 row differs at position {position}");
        let header = usize::try_from(u64::from_le_bytes(a.get(..8).context("row header")?.try_into()?))?;
        let payload = a.get(8 + header..).context("row payload")?;
        ensure!(payload.len() == vocab * 4 && payload.chunks_exact(4).all(|bytes|
            f32::from_le_bytes(bytes.try_into().unwrap()).is_finite()), "paired nonfinite/short row");
        hash.update((position as u64).to_le_bytes());
        hash.update(payload);
    }
    Ok(json!({"rows":rows.len(), "vocab":vocab, "byte_identical":true,
        "full_vocab_kl":0.0, "top1":1.0, "position_payload_sha256":format!("{:x}", hash.finalize())}))
}

fn serving_args() -> Result<super::serve::ServeArgs> {
    let snapshot = std::env::var("CUTEAFD_GOLDEN_SNAPSHOT")?;
    let library = std::env::var("CUTEAFD_NATIVE_LIB")?;
    let manifest = std::env::var("CUTEAFD_PROGRAM_MANIFEST")?;
    let peers = std::env::var("CUTEAFD_GOLDEN_PEERS")?;
    let attention = std::env::var("CUTEAFD_ATTENTION_PLACEMENT").unwrap_or_else(|_| "auto".into());
    let core = std::env::var("CUTEAFD_KIMI_PROOF_CORE_VIEW").unwrap_or_else(|_| "original".into());
    let crate::cli::Commands::ServeKimi(args) = crate::cli::Cli::try_parse_from([
        "cuteafd", "serve-kimi", "--snapshot", &snapshot, "--native-lib", &library,
        "--manifest", &manifest, "--split-device", "1", "--peers", &peers,
        "--attention-placement", &attention, "--core-view", &core, "--host-cache-bytes", "auto",
    ])?.command else { unreachable!() };
    ensure!(args.max_sequences == 16 && args.pool_tokens.is_none() && args.max_context.is_none()
        && args.draft.is_none() && args.prefill_rows.is_none() && args.prefill_waves.is_none()
        && args.peers.len() == 6, "paired proof must use ordinary C16/auto admission on six Sparks");
    Ok(args)
}

#[test]
#[ignore = "requires sealed K3.1 full model, two RTX GPUs, six Sparks and immutable Standard reference"]
fn k31_raw_and_streamed_scoring_match_ordinary_c16() -> Result<()> {
    cuteafd_core::set_expert_geometry(cuteafd_core::ExpertGeometry::KIMI_K3)
        .map_err(|_| anyhow::anyhow!("paired proof needs an isolated process"))?;
    let mut args = serving_args()?;
    args.prefix.resolve_entries(16);
    let view = args.attention.select(&args.snapshot)?;
    args.attention.validate_view(&view)?;
    let checkpoint = cuteafd_loader::plan::Checkpoint::coordinator_view(&view, false, false)?;
    let cfg = KimiK3Config::from_hf(&checkpoint.config)?;
    ensure!(cfg.dsa.is_some(), "paired K3.1 proof requires Gated DSA");
    cfg.release_geometry()?;
    cfg.validate_speculation(false)?;
    let prefill = args.prefill()?;
    ensure!((prefill.unit_rows, prefill.waves, prefill.span_rows) == (2752, 3, 8256),
        "paired proof must retain ordinary prefill reserve");
    let slots = cuteafd_core::concurrency::state_slots(16) as usize;
    let mark_bytes = super::state::KdaLayout::for_ranks(&cfg, slots, 1, 1, 2)?.mark_bytes()
        .checked_mul(2).context("proof mark extent")?;
    let marks = args.prefix.mark_rule(16).slots(mark_bytes);
    let mut options = super::serve::startup_options(&args, &cfg, [0, 1], slots, marks, prefill)?;
    // Only the prequalification runtime refusal differs from serving. Every
    // allocation, graph, pool, context and onboarding input stays identical.
    options.role = startup::Role::Golden { skip_routed_experts: false };
    options.hosted_layers = Some(super::golden::hosted_layers(&std::env::var("CUTEAFD_GOLDEN_EXPERT_LAYERS")?)?);
    startup::preflight(&args.manifest, &args.native_lib)?;
    // SAFETY: golden-run supplies the sealed library; all borrowers drop first.
    let library = Arc::new(unsafe { cuteafd_ffi::NativeLibrary::load(&args.native_lib)? });
    let programs = library.programs()?.with_manifest(&args.manifest)?;
    let loaded = startup::load(&library, &programs, &checkpoint, &cfg, &cfg, options)?;
    let engine = &loaded.engine;
    let backend = KimiBackend::new(engine)?;
    let family = engine.prefix_family()?;
    let geometry = engine.geometry()?;
    ensure!(geometry.slots == 16 && geometry.prefill_rows == prefill.span_rows as usize,
        "paired runtime differs from ordinary admission");
    let mut pool = super::prefix::diagnostic_pool(family.layout());
    let reference_root = std::path::PathBuf::from(std::env::var("CUTEAFD_GOLDEN_FIXTURES")?);
    let manifest_bytes = std::fs::read(reference_root.join("manifest.json"))?;
    ensure!(format!("{:x}", Sha256::digest(&manifest_bytes))
        == cuteafd_bench::fidelity_dataset::KIMI_K31_MANIFEST_SHA256, "paired reference publication differs");
    let manifest: Value = serde_json::from_slice(&manifest_bytes)?;
    let reference = cuteafd_bench::fidelity_dataset::load(&reference_root, &manifest)?;
    let root = std::path::PathBuf::from(std::env::var("CUTEAFD_GOLDEN_OUT")?).join("paired-scoring");
    std::fs::create_dir(&root)?;
    let mut results = Vec::new();
    for half in ["decode", "prefill"] {
        let windows = cuteafd_bench::fidelity_dataset::standard_windows(&reference, half, true)?;
        ensure!(windows.len() == 32, "paired proof needs full Standard halves");
        for window in windows {
            ensure!(window.media.is_empty() && window.tokens.len() <= geometry.max_context,
                "paired window exceeds ordinary text context");
            let pages = pool.alloc((window.tokens.len() - 1).div_ceil(64))?;
            let cuts = steps(window.tokens.len(), window.score_from, geometry.prefill_rows)?;
            let positions = window.positions.iter().map(|p| p.pos).collect::<Vec<_>>();
            let mut states = Vec::new();
            let mut records = Vec::new();
            for arm in ["raw", "streamed"] {
                engine.reset_cold_state()?;
                let mut placement = Placement { units: pages.clone(), slot: 0, len: 0, state_len: 0 };
                family.bind(&mut placement).map_err(|e| anyhow::anyhow!(e.to_string()))?;
                let probe = Probe::new(ProbeSpec { cold: true, no_speculation: true,
                    prompt_ids: Some(window.tokens.clone()), score_from: Some(window.score_from),
                    score_positions: Some(positions.clone()), top_k: 32, want: window.want(),
                    dump_rows: Some(root.join(arm).join(&window.id)), ..Default::default() });
                probe.admitted(cfg.family_id(), &window.tokens, 0);
                let probe_ref: ProbeRef = Some(probe.clone());
                for cut in &cuts {
                    let tokens = &window.tokens[cut.clone()];
                    if arm == "raw" {
                        engine.forward(&[tokens.to_vec()], &[vec![SequenceRows { slot: 0,
                            start: cut.start, rows: cut.len(), pages: &pages }]], &pool,
                            |lane, first, rows, logits| {
                                ensure!(lane == 0 && logits.len() == rows * cfg.vocab_size, "raw row geometry");
                                for (row, values) in logits.chunks_exact(cfg.vocab_size).enumerate() {
                                    if probe.wants(cut.start + first + row + 1) {
                                        ensure!(values.iter().all(|v| v.is_finite()), "nonfinite raw logits");
                                        probe.row(cut.start + first + row + 1, values);
                                    }
                                }
                                Ok(())
                            })?;
                        family.commit(&mut placement, cut.start, cut.len())?;
                    } else {
                        let count = if cut.end < window.score_from { 0 }
                            else if cut.end == window.score_from { 1 } else { cut.len() };
                        let position = if cut.end <= window.score_from { window.score_from } else { cut.start + 1 };
                        match backend.prefill_scoring(&mut placement, tokens, count, position, &probe_ref, &pool)? {
                            Some(ScoreLogits::Device(logits)) if count > 0 => {
                                ScoringRows { consumer: &probe_ref, position }.record(&library, 0, &logits)?;
                            }
                            None | Some(ScoreLogits::Recorded) | Some(ScoreLogits::Device(_)) if count == 0 => {}
                            Some(ScoreLogits::Recorded) => {}
                            _ => anyhow::bail!("unexpected streamed score result"),
                        }
                    }
                }
                let record = probe.record();
                ensure!(record.error.is_none() && record.rows.len() == positions.len(),
                    "paired {arm} probe missing/failed rows: {:?}", record.error);
                ensure!(placement.len == window.tokens.len() - 1 && placement.state_len == placement.len,
                    "paired commit frontier differs");
                states.push(family.live_digest(&placement)?);
                records.push(record);
                family.discard(&mut placement).map_err(|e| anyhow::anyhow!(e.to_string()))?;
            }
            ensure!(states[0] == states[1], "paired committed KDA/MLA/index state differs: {}", window.id);
            let mut result = compare_rows(&root.join("raw").join(&window.id),
                &root.join("streamed").join(&window.id), &positions, cfg.vocab_size)?;
            result["id"] = json!(window.id);
            result["half"] = json!(half);
            result["state_sha256"] = json!(states[0]);
            result["cuts"] = json!(cuts.iter().map(|c| [c.start, c.end]).collect::<Vec<_>>());
            result["probes"] = json!(records);
            serde_json::to_writer(std::fs::OpenOptions::new().write(true).create_new(true)
                .open(root.join(format!("{}-{}.json", half, window.id)))?, &result)?;
            println!("paired {half} {}: {} exact F32 rows and exact committed state", window.id, positions.len());
            results.push(result);
            engine.reset_cold_state()?;
            pool.release(&pages);
        }
    }
    ensure!(results.len() == 64, "paired Standard coverage incomplete");
    serde_json::to_writer(std::fs::OpenOptions::new().write(true).create_new(true).open(root.join("proof.json"))?,
        &json!({"schema":"cuteafd.kimi.paired-scoring/1", "passed":true,
            "view":crate::cli::core_view_settings(engine.target_view()), "attention":args.attention_placement,
            "concurrency":16, "state_slots":slots, "marks":marks,
            "prefill":{"unit_rows":prefill.unit_rows,"waves":prefill.waves,"span_rows":prefill.span_rows},
            "pool_pages":family.layout().pages, "context":geometry.max_context,
            "ownership":loaded.ownership, "reference_manifest_sha256":cuteafd_bench::fidelity_dataset::KIMI_K31_MANIFEST_SHA256,
            "windows":results}))?;
    Ok(())
}

#[test]
fn paired_row_comparison_rejects_missing_nonfinite_and_changed_rows() -> Result<()> {
    let root = tempfile::tempdir()?;
    let raw = root.path().join("raw");
    let streamed = root.path().join("streamed");
    std::fs::create_dir(&raw)?;
    std::fs::create_dir(&streamed)?;
    let metadata = b"{\"position\":7,\"vocab_size\":2,\"file\":\"row-00000000.safetensors\"}\n";
    let image = |values: [f32; 2]| {
        let mut bytes = 2u64.to_le_bytes().to_vec();
        bytes.extend_from_slice(b"{}");
        for value in values { bytes.extend_from_slice(&value.to_le_bytes()); }
        bytes
    };
    for dir in [&raw, &streamed] {
        std::fs::write(dir.join("manifest.jsonl"), metadata)?;
        std::fs::write(dir.join("row-00000000.safetensors"), image([-1.0, -2.0]))?;
    }
    assert_eq!(compare_rows(&raw, &streamed, &[7], 2)?["rows"], 1);
    assert!(compare_rows(&raw, &streamed, &[7, 8], 2).is_err());
    assert!(compare_rows(&raw, &streamed, &[8], 2).is_err());
    assert!(compare_rows(&raw, &streamed, &[7], 3).is_err());
    std::fs::write(streamed.join("row-00000000.safetensors"), image([-1.0, -3.0]))?;
    assert!(compare_rows(&raw, &streamed, &[7], 2).is_err());
    for dir in [&raw, &streamed] {
        std::fs::write(dir.join("row-00000000.safetensors"), image([f32::NAN, -2.0]))?;
    }
    assert!(compare_rows(&raw, &streamed, &[7], 2).is_err());
    std::fs::write(streamed.join("manifest.jsonl"), b"")?;
    assert!(compare_rows(&raw, &streamed, &[7], 2).is_err());
    Ok(())
}

#[test]
fn paired_cuts_preserve_prompt_boundary_and_ordinary_capacity() -> Result<()> {
    assert_eq!(steps(17000, 8500, 8256)?, vec![0..8256, 8256..8500, 8500..16756, 16756..16999]);
    assert_eq!(steps(66, 64, 8256)?, vec![0..64, 64..65]);
    assert!(steps(10, 10, 8256).is_err());
    assert!(steps(10, 0, 8256).is_err());
    assert!(steps(10, 3, 0).is_err());
    Ok(())
}
