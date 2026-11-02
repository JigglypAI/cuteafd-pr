use super::*;
use crate::families::qwen3_5::tests::native_planner_fixture;
use crate::plan::checkpoint::Checkpoint;

fn request() -> Request {
    Request {
        context: 262144,
        pool: Pin::Exact(262144),
        onboard: Pin::Auto,
        prefill: Pin::Auto,
        operation: Operation::Serving,
    }
}
fn fixture() -> (Qwen35Config, NativeWeights, NativeBuffers) {
    let (dir, _, buffers, _) = native_planner_fixture();
    let checkpoint = Checkpoint::open(dir.path()).unwrap();
    let cfg = Qwen35Config::from_hf(&checkpoint.config).unwrap();
    let weights = NativeWeights::native(&checkpoint, &cfg, cuteafd_core::DType::F32).unwrap();
    (cfg, weights, buffers)
}
fn budgets() -> Vec<Budget> {
    forecast_budgets(
        &crate::placement::ARCH_CONTEXTS[1],
        96 << 30,
        0,
        (128 << 30, 128 << 30),
    )
    .unwrap()
}
fn facts<'a>(
    cfg: &'a Qwen35Config,
    weights: &'a NativeWeights,
    buffers: &'a NativeBuffers,
) -> DenseFacts<'a> {
    DenseFacts {
        cfg,
        weights,
        buffers,
        slots: 4,
        record_bytes: 4096,
        drafter: None,
    }
}

#[test]
fn loading_heap_allowance_is_host_only_and_cannot_shrink_the_pool() {
    let (cfg, mut weights, buffers) = fixture();
    let mut req = request();
    req.pool = Pin::Auto;
    let first = admit(req.clone(), budgets(), facts(&cfg, &weights, &buffers)).unwrap();
    weights.loading_host_bytes += 8 << 30;
    let larger = admit(req, budgets(), facts(&cfg, &weights, &buffers)).unwrap();
    assert_eq!(larger.pool_tokens, first.pool_tokens);
    assert_eq!(
        larger
            .phases
            .iter()
            .filter(|p| p.budget == "coordinator.0")
            .collect::<Vec<_>>(),
        first
            .phases
            .iter()
            .filter(|p| p.budget == "coordinator.0")
            .collect::<Vec<_>>()
    );
    for row in larger.phases.iter().filter(|p| p.budget == "host") {
        assert_eq!(
            row.opaque,
            if row.phase == Phase::Load {
                weights.loading_host_bytes
            } else {
                0
            }
        );
    }
    assert!(larger
        .facts
        .allowances
        .iter()
        .any(|a| a.location == Location::HOST
            && a.assumed
            && a.lifetime
                == Lifetime {
                    first: Phase::Load,
                    last: Phase::Load
                }));
}

#[test]
fn native_scratch_prefix_and_pinned_staging_are_individual_owners() {
    let (cfg, weights, buffers) = fixture();
    let plan = admit(request(), budgets(), facts(&cfg, &weights, &buffers)).unwrap();
    let owner = |id: &str| plan.facts.allocations.iter().find(|a| a.id == id).unwrap();
    assert_eq!(owner("planes.6").bytes, 1024);
    assert_eq!(owner("prefix").bytes, buffers.prefix as u64);
    assert_eq!(
        owner("upload.sync").bytes,
        weights
            .allocations
            .iter()
            .map(|a| a.upload_bytes)
            .max()
            .unwrap()
    );
    for index in 7..13 {
        assert_eq!(
            owner(&format!("planes.{index}.upload")).bytes,
            buffers.planes[index] as u64
        );
        assert_eq!(
            owner(&format!("planes.{index}.upload")).location,
            Location::PINNED
        );
    }
    assert_eq!(owner("ids.upload").bytes, buffers.ids as u64);
    assert_eq!(
        owner("embedding.upload").lifetime,
        Lifetime {
            first: Phase::Load,
            last: Phase::Load
        }
    );
    assert!(plan
        .facts
        .allowances
        .iter()
        .any(|a| a.bytes == 512 << 20 && a.assumed));
    let conv_view = owner("gdn.conv.0.slot.1");
    assert_eq!(conv_view.slot, Some(1));
    assert_eq!(conv_view.view.as_ref().unwrap().backing, "gdn.conv");
    assert_eq!(conv_view.view.as_ref().unwrap().offset, conv_view.bytes);
    assert!(owner("prefix.mark.0").view.is_some());
    let mut larger = buffers.clone();
    larger.planes[6] += 1 << 30;
    let result = admit(request(), budgets(), facts(&cfg, &weights, &larger)).unwrap();
    let ready = |p: &ResolvedPlan| {
        p.phases
            .iter()
            .find(|r| r.budget == "coordinator.0" && r.phase == Phase::Ready)
            .unwrap()
            .owned
    };
    assert_eq!(ready(&result) - ready(&plan), 1 << 30);
}

#[test]
fn deferred_drafter_pool_demand_is_once_per_page_and_never_ready() {
    let (cfg, weights, buffers) = fixture();
    let mut recipe = facts(&cfg, &weights, &buffers);
    recipe.drafter = Some(DrafterBytes {
        owned: 1024,
        scratch: 2048,
        pool_unit: 64,
    });
    let small = recipe
        .evaluate(&request(), &recipe.candidates()[0], 262144)
        .unwrap();
    let larger = recipe
        .evaluate(&request(), &recipe.candidates()[0], 524288)
        .unwrap();
    assert_eq!(small.deferred.len(), 1);
    assert_eq!(small.deferred[0].bytes, 1024 + 2048 + 262144);
    assert_eq!(larger.deferred[0].bytes - small.deferred[0].bytes, 262144);
    let plan = admit(request(), budgets(), recipe).unwrap();
    assert!(
        matches!(reconcile(&plan, Phase::Ready, &[]), Err(Refusal::Reconciliation(errors))
        if errors.contains(&Mismatch::Missing("deferred.drafter".into())))
    );
}

#[test]
fn malformed_native_inventory_and_operation_fail_before_admission() {
    let (cfg, mut weights, buffers) = fixture();
    assert!(matches!(
        admit(
            Request {
                operation: Operation::Scoring,
                ..request()
            },
            budgets(),
            facts(&cfg, &weights, &buffers)
        ),
        Err(Refusal::InvalidFact(_))
    ));
    weights.layers.pop();
    assert!(matches!(
        admit(request(), budgets(), facts(&cfg, &weights, &buffers)),
        Err(Refusal::InvalidFact(_))
    ));
}
