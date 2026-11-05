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
    let root = std::env::var_os("CUTEAFD_NATIVE_LIFETIME_FIXTURE_DIR").map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(std::env::var_os("HOME").expect("HOME"))
            .join(".cache/cuteafd/builds/glm-dflash-physical-components/fixtures"));
    let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build_in(&root)?;
    let library = fixture.load()?;
    fixture.extent_only(&library)?;
    fixture.quiet_events(&library, true)?;
    fixture.configure_memory(&library, 0, 96usize << 30)?;
    let programs = library.programs()?;
    for repo in ["incoai/GLM-5.3-DFlash2", "incoai/GLM-5.3-Flash-DFlash2"] {
        let snapshot = installed_snapshot(repo)?;
        let cfg = DflashConfig::read(&snapshot)?;
        for (slots, sequences) in [(2, 1), (20, 16)] {
            for kv in [Qwen4KvCache::Bf16, Qwen4KvCache::Fp8] {
                for mode in [fp8_linear::Fp8Rows::W8a16, fp8_linear::Fp8Rows::Wide, fp8_linear::Fp8Rows::W8a8] {
                    let capacity = GlmDraftCapacity::new(slots, sequences, cfg.block)?;
                    let forecast = GlmDflashPhysicalLayout::read(&snapshot, GlmDraftRepresentation::Fp8Only,
                        capacity, library.sm_count()?, mode.code() as u8, kv)?;
                    // Real loader, matrix pack destinations and workspace constructor; recording
                    // suppresses checkpoint payload reads, not any device owner allocations.
                    let (drafter, records) = crate::shared::experts::rtx::dense::recording(|| {
                        DflashDrafter::load(&library, &programs, &snapshot, Vec::new(), std::ptr::null_mut(),
                            slots, sequences, vec![0; cfg.hidden * 2], false, GlmDraftRepresentation::Fp8Only,
                            fp8_linear::Fp8Scales::Amax, mode, kv)
                    })?;
                    ensure!(!records.operands.is_empty() && !records.steps.is_empty(), "real weight load did not run");
                    *drafter.workspace.borrow_mut() = Some(drafter.workspace(sequences)?);
                    drafter.prime_metadata()?;
                    let actual: BTreeMap<_, _> = drafter.allocation_receipts(CoordinatorDevices::new(&[0])?, "draft")?
                        .into_iter().map(|receipt| (receipt.allocation.id.clone(), receipt.allocation)).collect();
                    let expected: BTreeMap<_, _> = forecast.allocations("draft", Location::coordinator(0))
                        .into_iter().map(|allocation| (allocation.id.clone(), allocation)).collect();
                    assert_eq!(actual, expected, "{repo}, slots={slots}, C={sequences}, {kv}, {mode:?}");
                    assert!(forecast.resident_weights().0 > 0);
                    assert!(forecast.weight_owners().all(|owner| owner.location == Location::coordinator(0)));
                    drop(drafter);
                    fixture.finish_ordered_uploads(&library)?;
                }
            }
        }
    }
    Ok(())
}
