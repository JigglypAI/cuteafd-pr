//! Header-driven real serving loads against the allocation-only CPU native DSO.
use super::*;
use cuteafd_loader::{families::glm5::draft_physical::GlmDflashPhysicalLayout,
    families::qwen4::Qwen4KvCache, placement::planner::{CoordinatorDevices, Location}};
use std::{collections::BTreeMap, path::PathBuf};

fn installed_snapshot(repo: &str) -> Result<PathBuf> {
    let home = std::env::var_os("HF_HOME").map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(std::env::var_os("HOME").expect("HOME")).join(".cache/huggingface"));
    let model = home.join("hub").join(format!("models--{}", repo.replace('/', "--")));
    let revision = std::fs::read_to_string(model.join("refs/main"))
        .with_context(|| format!("required real draft checkpoint {repo} missing"))?;
    let snapshot = model.join("snapshots").join(revision.trim());
    ensure!(snapshot.join("model.safetensors").is_file(), "required real draft headers missing: {}", snapshot.display());
    Ok(snapshot)
}

#[test]
fn installed_standard_dflash_physical_layout_equals_serving_receipts() -> Result<()> {
    // Isolate the RSS contract from other CPU gate tests running in parallel.
    const CHILD: &str = "CUTEAFD_DFLASH_PHYSICAL_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe()?)
            .args(["shared::speculation::dflash::physical_tests::installed_standard_dflash_physical_layout_equals_serving_receipts",
                "--exact", "--nocapture"])
            .env(CHILD, "1").output()?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        ensure!(output.status.success() && stdout.contains("1 passed; 0 failed"),
            "isolated physical equality failed: {stdout}\n{stderr}");
        println!("{stdout}");
        return Ok(());
    }
    let peak_rss = || -> Result<u64> {
        let status = std::fs::read_to_string("/proc/self/status")?;
        status.lines().find_map(|line| line.strip_prefix("VmHWM:"))
            .and_then(|value| value.split_whitespace().next()).context("missing RSS high-water mark")?
            .parse::<u64>().map_err(Into::into)
    };
    let baseline_rss = peak_rss()?;
    let mut arms = 0;
    let root = std::env::var_os("CUTEAFD_NATIVE_LIFETIME_FIXTURE_DIR").map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(std::env::var_os("HOME").expect("HOME"))
            .join(".cache/cuteafd/builds/glm-dflash-physical-components/fixtures"));
    let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build_in(&root)?;
    let library = fixture.load()?;
    fixture.extent_only(&library)?;
    fixture.quiet_events(&library, true)?;
    fixture.configure_memory(&library, 0, 96usize << 30)?;
    for repo in ["incoai/GLM-5.3-DFlash2", "incoai/GLM-5.3-Flash-DFlash2"] {
        let snapshot = installed_snapshot(repo)?;
        let cfg = DflashConfig::read(&snapshot)?;
        let shapes: std::collections::BTreeSet<_> = [
            (cfg.hidden, 2 * cfg.intermediate), (cfg.intermediate, cfg.hidden),
            (cfg.hidden * cfg.taps.len(), cfg.hidden), (cfg.hidden, cfg.conv_width()),
            (cfg.hidden, cfg.rank), (cfg.hidden, cfg.qkv_width()),
            (cfg.hidden, 2 * cfg.kv_width()), (cfg.heads * cfg.head_dim, cfg.hidden),
        ].into_iter().collect();
        let entries = shapes.into_iter().enumerate().map(|(index, (k, n))| {
            fixture.draft_projection(&library, index.try_into()?, k.try_into()?, n.try_into()?)?;
            Ok(serde_json::json!({"name": format!("draft_bf16_k{k}_n{n}"),
                "pointers": [{"name":"x"},{"name":"w"},{"name":"out"}], "capacity_rows": TAP_ROWS,
                "geometry": {"width":k,"outputs":n,"output_dtype":"torch.bfloat16",
                    "reduction":"ordered_k16","tiles_by_rows":[[16,1,4,16,64,6]]}}))
        }).collect::<Result<Vec<_>>>()?;
        let manifest = fixture.directory().join("draft-programs.json");
        std::fs::write(&manifest, serde_json::to_vec(&serde_json::json!({"programs":entries}))?)?;
        let programs = library.programs()?.with_manifest(&manifest)?;
        for (slots, sequences) in [(2, 1), (20, 16)] {
            for kv in [Qwen4KvCache::Bf16, Qwen4KvCache::Fp8] {
                for representation in [GlmDraftRepresentation::Bf16Only, GlmDraftRepresentation::Fp8Only] {
                for mode in [fp8_linear::Fp8Rows::W8a16, fp8_linear::Fp8Rows::Wide, fp8_linear::Fp8Rows::W8a8] {
                    let capacity = GlmDraftCapacity::new(slots, sequences, cfg.block)?;
                    let forecast = GlmDflashPhysicalLayout::read(&snapshot, representation,
                        capacity, library.sm_count()?, mode.code() as u8, kv)?;
                    // Real loader, matrix pack destinations and workspace constructor; recording
                    // suppresses checkpoint payload reads, not any device owner allocations.
                    let (drafter, records) = crate::shared::experts::rtx::dense::recording(|| {
                        DflashDrafter::load(&library, &programs, &snapshot, Vec::new(), std::ptr::null_mut(),
                            slots, sequences, vec![0; cfg.hidden * 2], false, representation,
                            fp8_linear::Fp8Scales::Amax, mode, kv)
                    })?;
                    ensure!(!records.operands.is_empty(), "real weight load did not run");
                    ensure!(representation == GlmDraftRepresentation::Bf16Only || !records.steps.is_empty(),
                        "real FP8 packing destinations did not run");
                    *drafter.workspace.borrow_mut() = Some(drafter.workspace(sequences)?);
                    drafter.prime_metadata()?;
                    let actual: BTreeMap<_, _> = drafter.allocation_receipts(CoordinatorDevices::new(&[0])?, "draft")?
                        .into_iter().map(|receipt| (receipt.allocation.id.clone(), receipt.allocation)).collect();
                    let expected: BTreeMap<_, _> = forecast.allocations("draft", Location::coordinator(0))
                        .into_iter().map(|allocation| (allocation.id.clone(), allocation)).collect();
                    assert_eq!(actual, expected, "{repo}, slots={slots}, C={sequences}, {kv}, {mode:?}, {representation:?}");
                    assert!(forecast.resident_weights().0 > 0);
                    assert!(forecast.weight_owners().all(|owner| owner.location == Location::coordinator(0)));
                    drop(drafter);
                    fixture.finish_ordered_uploads(&library)?;
                    assert_eq!(fixture.mapped_bytes(&library)?, 0, "virtual device mapping leaked");
                    ensure!(peak_rss()?.saturating_sub(baseline_rss) < 128 * 1024,
                        "allocation equality RSS grew by more than 128 MiB");
                    arms += 1;
                }
                }
            }
        }
    }
    assert_eq!(arms, 48);
    println!("{arms} real-load arms; peak RSS growth {} KiB; no retained virtual mappings",
        peak_rss()?.saturating_sub(baseline_rss));
    Ok(())
}
