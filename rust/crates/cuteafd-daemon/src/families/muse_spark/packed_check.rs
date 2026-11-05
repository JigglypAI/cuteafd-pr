//! Family-local qualification on the serving backend and shared genuine-row gate.
use super::{backend::{DraftState, MuseBackend}, prefix::Placement, serve::ServeArgs};
use crate::shared::{prefill_packing::PackingMode, qwen::{packed_check, scheduler::{Backend, PrefillSeq}}};
use anyhow::{ensure, Context, Result};
use cuteafd_engine::prefix::{MarkSlot, PrefixFamily, RefPagePool};
use cuteafd_ffi::CuteafdDeviceBuffer as Buffer;
use sha2::{Digest, Sha256};

#[derive(Debug, clap::Args)]
pub(crate) struct GoldenArgs {
    #[command(flatten)] pub engine: ServeArgs,
    /// Immutable Muse fidelity publication directory.
    #[arg(long)] pub golden: std::path::PathBuf,
    #[arg(long, default_value_t = 4)] pub packed_check: usize,
    #[arg(long)] pub window: Option<String>,
}

pub(crate) async fn run(args: GoldenArgs) -> Result<()> {
    tokio::task::spawn_blocking(move || qualify(args)).await?
}

fn qualify(mut args: GoldenArgs) -> Result<()> {
    use cuteafd_bench::fidelity_dataset as dataset;
    let members = args.packed_check;
    ensure!((2..=8).contains(&members), "Muse --packed-check requires 2..=8 members");
    args.engine.max_sequences = args.engine.max_sequences.max(u8::try_from(2 * members + 1)?);
    args.engine.prefix.resolve_entries(usize::from(args.engine.max_sequences));
    args.engine.validate()?;
    let bytes = std::fs::read(args.golden.join("manifest.json"))?;
    ensure!(format!("{:x}", Sha256::digest(&bytes)) == dataset::MUSE_SPARK_MANIFEST_SHA256,
        "Muse fidelity manifest differs from the immutable publication");
    let manifest: serde_json::Value = serde_json::from_slice(&bytes)?;
    ensure!(manifest["config"] == dataset::MUSE_SPARK_CONFIG, "Muse fidelity config differs");
    let reference = dataset::load(&args.golden, &manifest)?;
    let windows = &reference.windows;
    let first = args.window.as_deref().unwrap_or(&windows.first().context("empty Muse reference")?.id);
    let selected = packed_check::reference_windows(windows, first, members,
        args.engine.prefill_packing == PackingMode::Unrestricted)?;
    ensure!(selected.iter().all(|w| w.media.is_empty()), "Muse packing gate requires original text windows");
    super::serve::with_backend(&args.engine, |backend, slots, _| {
        clear_rings(backend)?;
        ensure!(slots > 2 * members, "Muse packing check needs a spare restore slot");
        let (family, mut cache) = backend.prefix_cache(&args.engine.prefix, slots)?;
        let prompts = packed_check::prompts(&selected[0].tokens, members)?;
        let mut allocate = |lengths: Vec<usize>, first_slot: usize| -> Result<Vec<Placement>> {
            lengths.into_iter().enumerate().map(|(i, len)| Ok(cache.admit_cold(&family, len, len + 1,
                |pages| Placement { pages, slot: first_slot + i, ..Default::default() })?.placement)).collect()
        };
        let packed = allocate(prompts.iter().map(Vec::len).collect(), 0)?;
        let alone = allocate(prompts.iter().map(Vec::len).collect(), members)?;
        let limits = backend.checked_packed_limits()?.context("Muse packed limits")?;
        let neural = args.engine.draft.is_some();
        ensure!(family.slots() >= 2, "Muse packing restore gate requires two admitted prefix mark slots");
        packed_check::check::<_, DraftState>(backend.library(), &prompts, packed, alone, limits, neural,
            args.engine.prefill_packing == PackingMode::Exact,
            |s| singleton(backend, s, cache.pool()),
            |s, callback| backend.prefill_packed(s, cache.pool(), callback),
            |p| snapshot_and_restore(backend, &family, p, 2 * members))?;
        if args.engine.prefill_packing == PackingMode::Exact {
            println!("exact Muse packing fails closed on EXL3 at this pin: {}", backend.packed_fallback_reason);
            return Ok(());
        }
        // A fresh pool owns fresh physical pages; no stale synthetic prefix becomes a control.
        drop(cache);
        drop(family);
        clear_rings(backend)?;
        let (family, mut cache) = backend.prefix_cache(&args.engine.prefix, slots)?;
        let mut allocate = |first_slot: usize| -> Result<Vec<Placement>> {
            selected.iter().enumerate().map(|(i, w)| Ok(cache.admit_cold(&family, w.tokens.len(), w.tokens.len(),
                |pages| Placement { pages, slot: first_slot + i, ..Default::default() })?.placement)).collect()
        };
        let placements = allocate(0)?;
        let controls = allocate(members)?;
        let precision = cuteafd_bench::fidelity_precision::resolve("muse-packed-check", Some(&args.engine.snapshot),
            &serde_json::json!([]), "meta-models/Muse-Spark-Open-816B-A42B", None)?;
        packed_check::reference::<_, DraftState>(backend.library(), &selected, placements, controls, limits, neural, &precision,
            |s| singleton(backend, s, cache.pool()),
            |s, callback| backend.prefill_packed(s, cache.pool(), callback))
    })
}

fn singleton(backend: &MuseBackend<'_, '_>, s: &mut PrefillSeq<'_, Placement, DraftState>, pool: &RefPagePool)
    -> Result<Vec<f32>> {
    let logits = backend.prefill_device(s.placement, s.tokens, 1, pool)?.context("Muse singleton head")?
        .row_host(backend.library(), 0)?;
    if s.neural { backend.prefill_draft(s.placement, s.start, s.tokens, s.next, s.draft)?; }
    Ok(logits)
}

fn clear_rings(backend: &MuseBackend<'_, '_>) -> Result<()> {
    backend.diagnostic_sync()?;
    // Bound host scratch even for the max-context gate; untouched ring rows
    // must have the same sentinel in singleton and packed diagnostic slots.
    let zero = vec![0; 1 << 20];
    for rank in backend.ranks {
        rank.device.run(|| {
            let ring = rank.kv.rings.buffer;
            for at in (0..ring.bytes).step_by(zero.len()) {
                let bytes = (ring.bytes - at).min(zero.len());
                backend.library().copy_h2d(view(ring, at, bytes), &zero[..bytes])?;
            }
            Ok(())
        })?;
    }
    Ok(())
}

fn read(backend: &MuseBackend<'_, '_>, buffer: Buffer) -> Result<Vec<u8>> {
    let mut bytes = vec![0; buffer.bytes];
    backend.library().copy_d2h(&mut bytes, buffer)?;
    Ok(bytes)
}

fn view(buffer: Buffer, offset: usize, bytes: usize) -> Buffer {
    Buffer { ptr: buffer.ptr.cast::<u8>().wrapping_add(offset).cast(), bytes, ..buffer }
}

/// Compare initialized full records and whole rings, and round-trip each packed
/// prompt's exact visible state through a mark into a different private slot.
fn snapshot_and_restore(backend: &MuseBackend<'_, '_>, family: &super::prefix::MusePrefix<'_, '_>,
    placement: &Placement, restore_slot: usize) -> Result<Vec<u8>> {
    let boxed = |error: cuteafd_engine::prefix::BoxError| anyhow::anyhow!(error.to_string());
    backend.diagnostic_sync()?;
    family.capture(MarkSlot(0), placement, placement.len).map_err(boxed)?;
    family.drain().map_err(boxed)?;
    let mut restored = placement.clone();
    restored.slot = restore_slot;
    restored.len = 0;
    family.restore(Some(MarkSlot(0)), &mut restored, placement.len).map_err(boxed)?;
    family.capture(MarkSlot(1), &restored, placement.len).map_err(boxed)?;
    family.drain().map_err(boxed)?;
    let cache = cuteafd_loader::families::muse_spark::cache::MuseCacheLayout::new(backend.cfg, backend.ranks.len())?;
    let record = usize::try_from(cache.record_bytes)?;
    let ring_bytes = usize::try_from(cache.ring.rows)? * record;
    let before = family.mark_segments(MarkSlot(0));
    let after = family.mark_segments(MarkSlot(1));
    let mut snapshot = Vec::new();
    for (rank_index, rank) in backend.ranks.iter().enumerate() {
        rank.device.run(|| {
            let mark = |range: cuteafd_hostcache::copy::DeviceRange| Buffer {
                ptr: range.addr as *mut std::ffi::c_void, bytes: range.bytes, device_id: rank.device.id, flags: 0 };
            let a = read(backend, mark(before[rank_index]))?;
            let b = read(backend, mark(after[rank_index]))?;
            let mark_layer_bytes = backend.cfg.window * record;
            let visible_bytes = placement.len.min(backend.cfg.window) * record;
            // Short prompts initialize only their visible rows, not mark padding.
            for (a, b) in a.chunks_exact(mark_layer_bytes).zip(b.chunks_exact(mark_layer_bytes)) {
                ensure!(a[..visible_bytes] == b[..visible_bytes],
                    "Muse prompt-end restore changed the visible ring/mark on rank {rank_index}");
                snapshot.extend_from_slice(&a[..visible_bytes]);
            }
            for layer in &rank.kv.layers {
                match *layer {
                    super::device::LayerKv::Paged { cache } => {
                        for (i, &page) in placement.pages.iter().enumerate() {
                            let rows = placement.len.saturating_sub(i * 64).min(64);
                            if rows > 0 { snapshot.extend(read(backend, view(cache, page as usize * 64 * record, rows * record))?); }
                        }
                    }
                    super::device::LayerKv::Ring { ring } => {
                        snapshot.extend(read(backend, view(ring, placement.slot * ring_bytes, ring_bytes))?);
                    }
                }
            }
            Ok(())
        })?;
    }
    println!("Muse prompt-end restore: slot {} -> {}, {} rows, full-page ownership and visible SWA marks byte-identical",
        placement.slot, restore_slot, placement.len);
    Ok(snapshot)
}
