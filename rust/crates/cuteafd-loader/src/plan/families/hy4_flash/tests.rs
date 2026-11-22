use super::*;
use crate::plan::{
    checkpoint::{CheckpointView, CoreView},
    layout::LayoutOptions,
    ExpertPlacement, PlanInput, PlanOptions,
};

fn hy4_snapshot() -> Option<std::path::PathBuf> {
    let hf = std::env::var_os("CUTEAFD_DEPLOYMENT_TEST_HF_HOME")
        .or_else(|| std::env::var_os("HF_HOME"))
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("/mnt/sparknest/hf-home"));
    let snap = hf.join(
        "hub/models--wrldsuksgo2mars--Hy4-flash-EXL3-K4-MC/snapshots/cc72179113573d751ed3245e3e58d3640694378d",
    );
    snap.is_dir().then_some(snap)
}

#[test]
fn hy4_flash_family_descriptor_invariants() {
    assert_eq!(HY4_FLASH.id(), "hy4_flash");
    assert_eq!(HY4_FLASH.runtime(), RuntimeStatus::Planned);
    assert!(HY4_FLASH.optional(Component::Vision));
    assert!(HY4_FLASH.optional(Component::Speculator));
    assert!(!HY4_FLASH.optional(Component::Attention));
    assert!(!HY4_FLASH.optional(Component::RoutedExpert));
    assert!(!HY4_FLASH.optional(Component::MappedTable));
}

#[test]
#[ignore = "reads installed wrldsuksgo2mars/Hy4-flash-EXL3-K4-MC snapshot on raptor"]
fn installed_hy4_flash_plan_across_all_mc_views_and_two_cards() {
    let snap = hy4_snapshot().expect("Hy4-flash snapshot not installed");

    // Verify all 4 MC views classify every single tensor with zero unclassified and zero rejections.
    for view in [CoreView::Original, CoreView::K8Compact, CoreView::K6Compact, CoreView::Fp8Compact] {
        let selected = CheckpointView::select(&snap, view).expect("select Hy4 view");
        let input = PlanInput::open_view(&selected).expect("open Hy4 view");
        let report = input.plan(&PlanOptions::default()).expect("plan Hy4 view");
        assert_eq!(report.family.as_deref(), Some("hy4_flash"), "view={view}");
        assert!(report.unclassified.is_empty(), "view={view} unclassified: {:?}", report.unclassified);
        for comp in &report.components {
            assert_eq!(
                comp.rejected, 0,
                "view={view} component {:?} had rejections: {:?}",
                comp.component, comp.rejections
            );
        }
    }

    // Verify 2-column layout placement across MC views (5090 column retired):
    // 1. 1x RTX PRO 6000 (96 GiB) + 2 Sparks (min)
    // 2. 2x RTX PRO 6000 (192 GiB) + 0 Sparks (max: 0-Spark card, all 41 layers onboarded TP2)
    for view in [CoreView::Original, CoreView::K8Compact, CoreView::K6Compact, CoreView::Fp8Compact] {
        let selected = CheckpointView::select(&snap, view).expect("select view");
        let input = PlanInput::open_view(&selected).expect("open view");
        let cases = [
            ("1xPRO+2sparks", vec![96u64 << 30], ExpertPlacement::Sparks { ranks: 2 }, 30usize),
            ("2xPRO+0sparks", vec![96u64 << 30, 96u64 << 30], ExpertPlacement::Local, 41usize),
        ];
        for (label, rtx_bytes, placement, min_onboard) in cases {
            let mut layout = LayoutOptions::default();
            layout.rtx_bytes = rtx_bytes;
            layout.host_memory = Some((512 << 30, 400 << 30));
            let mut opts = PlanOptions::default();
            opts.placement = placement;
            opts.layout = Some(layout);
            let report = input.plan(&opts).unwrap_or_else(|e| panic!("{view}/{label}: {e:#}"));
            assert!(report.config_error.is_none(), "{view}/{label}: {:?}", report.config_error);
            // While RuntimeStatus::Planned, `plan` sets `fits = false` ("planned family has no
            // executable serving admission"), while still producing the full memory layout & host ledger.
            let mem = report.memory_layout.as_ref().expect("memory layout");
            assert!(!mem.devices.is_empty(), "{view}/{label}: empty devices, hints: {:?}", report.hints);
            assert!(mem.pool_tokens >= 2_000_000, "{view}/{label}: pool {} < 2M", mem.pool_tokens);
            let onboard = report.solver_placement.as_ref().unwrap().onboard_layers;
            assert!(
                onboard >= min_onboard,
                "{view}/{label}: expected >= {min_onboard} onboarded layers, got {onboard}"
            );
            // Engram FP8 tables (~61.47 GiB) must be charged to raptor host RAM, never GPU memory.
            let host = report.host_ledger.as_ref().expect("host ledger");
            let engram_host: u64 = host.items.iter()
                .filter(|item| item.group.contains("host-mapped Engram tables"))
                .map(|item| item.bytes)
                .sum();
            assert_eq!(engram_host, 66_000_641_600, "{view}/{label}: expected exact 61.47 GiB engram_fp8 on host");
            for dev in &mem.devices {
                assert!(
                    !dev.items.iter().any(|item| item.group.contains("host-mapped Engram tables")),
                    "{view}/{label}: Engram tables must not be charged to GPU/Spark {}",
                    dev.name()
                );
            }
        }
    }
}
