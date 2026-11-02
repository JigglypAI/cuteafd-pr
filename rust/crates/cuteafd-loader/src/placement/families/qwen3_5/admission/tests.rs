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
        serving: None,
    }
}

#[test]
fn dense_curves_match_independent_exact_inventory_and_monotone_page_neighbors() {
    let (cfg, weights, buffers) = fixture();
    for record_bytes in [2048, 4096, 8192] {
        for draft in [
            None,
            Some(DrafterBytes {
                owned: 1024,
                scratch: 2048,
                pool_unit: 4096,
            }),
        ] {
            for sampling in [false, true] {
                let mut recipe = facts(&cfg, &weights, &buffers);
                recipe.record_bytes = record_bytes;
                recipe.drafter = draft;
                recipe.serving = Some(
                    ServingFacts::new(
                        &cfg,
                        &buffers,
                        record_bytes,
                        ServingRequest {
                            concurrency: 4,
                            prefix_entries: 8,
                            host_cache_bytes: Some(512 << 20),
                            host_headroom: 8 << 30,
                            device_sampling: sampling,
                        },
                        128 << 30,
                    )
                    .unwrap(),
                );
                let contract = recipe.prepare(&request()).unwrap();
                let selected = vec![0; contract.variables.len()];
                let mut pages = std::collections::BTreeSet::from([
                    1,
                    2,
                    4095,
                    4096,
                    4097,
                    8191,
                    8192,
                    8193,
                    32767,
                    32768,
                    32769,
                    contract.maximum_pages - 1,
                    contract.maximum_pages,
                ]);
                for component in &contract.components {
                    for piece in &component.bytes.0 {
                        for boundary in [piece.first, piece.last] {
                            for p in boundary.saturating_sub(1)..=boundary.saturating_add(1) {
                                if p > 0 && p <= contract.maximum_pages {
                                    pages.insert(p);
                                }
                            }
                        }
                    }
                }
                let mut generous = budgets();
                for budget in &mut generous {
                    budget.physical_bytes = 1 << 60;
                    budget.logical_bytes = 1 << 60;
                    budget.baseline = Baseline::Forecast {
                        occupied: 0,
                        source: "curve oracle".into(),
                        assumed: false,
                    };
                    budget.headroom = 0;
                }
                for pages in pages {
                    let actual = contract.facts(&selected, pages).unwrap();
                    let expected = recipe.owner_templates(pages * PAGE_ROWS).unwrap();
                    assert_eq!(
                        actual, expected,
                        "page {pages}, record {record_bytes}, sampling {sampling}"
                    );
                    let rows = evaluate(&actual, &generous).unwrap();
                    if pages < contract.maximum_pages {
                        let next = contract.facts(&selected, pages + 1).unwrap();
                        let next_rows = evaluate(&next, &generous).unwrap();
                        assert!(rows
                            .iter()
                            .zip(next_rows)
                            .all(|(p, next)| next.owned + next.opaque >= p.owned + p.opaque));
                    }
                    if pages >= request().context / PAGE_ROWS {
                        let resolved = admit(
                            Request {
                                pool: Pin::Exact(pages * PAGE_ROWS),
                                ..request()
                            },
                            generous.clone(),
                            DenseFacts { ..recipe },
                        )
                        .unwrap();
                        assert_eq!(resolved.phases, rows);
                    }
                }
            }
        }
    }
}

#[test]
fn logical_thirty_gib_cap_preserves_pro_overheads_and_real_5090_identity() {
    let pro = &crate::placement::ARCH_CONTEXTS[1];
    let small = &crate::placement::ARCH_CONTEXTS[0];
    let host = (128 << 30, 128 << 30);
    let modules = 37 << 20;
    let capped = forecast_budgets(pro, 30 << 30, modules, host).unwrap();
    assert_eq!(capped[0].physical_bytes, pro.total_bytes);
    assert_eq!(capped[0].logical_bytes, 30 << 30);
    assert_eq!(
        capped[0].baseline,
        Baseline::Forecast {
            occupied: pro.context_bytes + modules,
            source: pro.source.into(),
            assumed: false,
        }
    );
    let live = sampled_budgets(
        pro.total_bytes - pro.context_bytes,
        pro.total_bytes,
        Some(30 << 30),
        host,
    )
    .unwrap();
    assert_eq!(live[0].physical_bytes, pro.total_bytes);
    assert_eq!(
        live[0].available().unwrap(),
        (30 << 30) - pro.context_bytes - (1 << 30)
    );
    assert!(matches!(&live[0].baseline, Baseline::Sampled { source, .. }
        if source.contains(pro.source) && !source.contains("5090")));
    let total = 32 << 30;
    let actual_5090 =
        sampled_budgets(total - small.context_bytes, total, Some(30 << 30), host).unwrap();
    assert_eq!(actual_5090[0].physical_bytes, total);
    assert!(
        matches!(&actual_5090[0].baseline, Baseline::Sampled { source, .. }
        if source.contains(small.source) && source.contains("5090"))
    );
    let (cfg, weights, buffers) = fixture();
    let mut req = request();
    req.pool = Pin::Auto;
    let large = admit(req.clone(), live, facts(&cfg, &weights, &buffers)).unwrap();
    let small = admit(req, actual_5090, facts(&cfg, &weights, &buffers)).unwrap();
    assert_eq!((large.pool_tokens, small.pool_tokens), (2 << 20, 1 << 20));
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
    let contract = recipe.prepare(&request()).unwrap();
    let selected = vec![0; contract.variables.len()];
    let small = contract.facts(&selected, 262144 / PAGE_ROWS).unwrap();
    let larger = contract.facts(&selected, 524288 / PAGE_ROWS).unwrap();
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

#[test]
fn serving_ready_requires_exact_scheduler_sampler_and_whole_host_chunks() {
    let (cfg, weights, buffers) = fixture();
    let chunk = 256 << 20;
    let serving = ServingFacts::new(
        &cfg,
        &buffers,
        4096,
        ServingRequest {
            concurrency: 4,
            prefix_entries: 8,
            host_cache_bytes: Some(2 * chunk + 17),
            host_headroom: 8 << 30,
            device_sampling: true,
        },
        128 << 30,
    )
    .unwrap();
    assert_eq!(
        (serving.host_quota, serving.host_chunk),
        (2 * chunk + 17, chunk)
    );
    let mut recipe = facts(&cfg, &weights, &buffers);
    recipe.serving = Some(serving);
    let plan = admit(request(), budgets(), recipe).unwrap();
    let warm = plan
        .facts
        .allocations
        .iter()
        .filter(|a| a.lifetime.first == Phase::Warmup)
        .collect::<Vec<_>>();
    assert_eq!(warm.len(), 2 + 22 + 2);
    assert_eq!(
        warm.iter()
            .filter(|a| a.group == "retained host prefix chunk")
            .map(|a| a.bytes)
            .sum::<u64>(),
        2 * chunk
    );
    let row = |phase| {
        plan.phases
            .iter()
            .find(|p| p.budget == "host" && p.phase == phase)
            .unwrap()
    };
    assert_eq!(row(Phase::Ready).opaque, serving.host_runtime);
    assert_eq!(row(Phase::Load).opaque, weights.loading_host_bytes);
    let mut addresses = std::collections::BTreeMap::new();
    let mut cursor = 4096u64;
    let mut all = Vec::new();
    for allocation in plan.facts.allocations.iter().filter(|a| a.view.is_none()) {
        addresses.insert(allocation.id.as_str(), cursor);
        all.push(Receipt {
            allocation: allocation.clone(),
            address: cursor,
        });
        cursor += allocation.bytes + 4096;
    }
    for allocation in plan.facts.allocations.iter().filter(|a| a.view.is_some()) {
        let view = allocation.view.as_ref().unwrap();
        all.push(Receipt {
            allocation: allocation.clone(),
            address: addresses[view.backing.as_str()] + view.offset,
        });
    }
    let receipts = all
        .iter()
        .filter(|r| r.allocation.lifetime.contains(Phase::Ready))
        .cloned()
        .collect::<Vec<_>>();
    let target = all
        .iter()
        .filter(|r| r.allocation.lifetime.contains(Phase::PostTargetKv))
        .cloned()
        .collect::<Vec<_>>();
    reconcile(&plan, Phase::PostTargetKv, &target).unwrap();
    reconcile(&plan, Phase::Ready, &receipts).unwrap();
    let observations = all
        .iter()
        .map(|receipt| Observation {
            receipt: receipt.clone(),
            created: receipt.allocation.lifetime.first,
            freed: (receipt.allocation.lifetime.last < Phase::Ready)
                .then_some(receipt.allocation.lifetime.last),
        })
        .collect::<Vec<_>>();
    reconcile_history(&plan, Phase::Ready, &observations).unwrap();
    for id in [
        "selector.output",
        "selector.landing",
        "selector.sampler.mask_pinned",
        "selector.sampler.bitmap",
        "host.prefix.1",
    ] {
        let missing = receipts
            .iter()
            .filter(|r| r.allocation.id != id)
            .cloned()
            .collect::<Vec<_>>();
        assert!(
            matches!(reconcile(&plan, Phase::Ready, &missing), Err(Refusal::Reconciliation(errors))
            if errors.contains(&Mismatch::Missing(id.into())))
        );
        for domain in [false, true] {
            let mut wrong = receipts.clone();
            let owner = wrong.iter_mut().find(|r| r.allocation.id == id).unwrap();
            if domain {
                owner.allocation.location = Location::HOST;
            } else {
                owner.allocation.bytes += 1;
            }
            assert!(
                matches!(reconcile(&plan, Phase::Ready, &wrong), Err(Refusal::Reconciliation(errors))
                if errors.contains(&Mismatch::Contract(id.into())))
            );
        }
    }
    let mut premature = observations;
    premature
        .iter_mut()
        .find(|o| o.receipt.allocation.id == "selector.output")
        .unwrap()
        .created = Phase::PostTargetKv;
    assert!(
        matches!(reconcile_history(&plan, Phase::Ready, &premature), Err(Refusal::Reconciliation(errors))
        if errors.contains(&Mismatch::Lifetime("selector.output".into())))
    );
    assert!(reconcile(&plan, Phase::Ready, &target).is_err());
}

#[test]
fn host_selector_and_disabled_prefix_have_no_sampler_or_chunk_owners() {
    let (cfg, weights, buffers) = fixture();
    for (entries, bytes) in [(0, None), (4, Some(0))] {
        let serving = ServingFacts::new(
            &cfg,
            &buffers,
            4096,
            ServingRequest {
                concurrency: 4,
                prefix_entries: entries,
                host_cache_bytes: bytes,
                host_headroom: 8 << 30,
                device_sampling: false,
            },
            128 << 30,
        )
        .unwrap();
        let mut recipe = facts(&cfg, &weights, &buffers);
        recipe.serving = Some(serving);
        let plan = admit(request(), budgets(), recipe).unwrap();
        assert_eq!(
            plan.facts
                .allocations
                .iter()
                .filter(|a| a.lifetime.first == Phase::Warmup)
                .count(),
            2
        );
        assert!(!plan
            .facts
            .allocations
            .iter()
            .any(|a| a.id.starts_with("host.prefix.") || a.id.starts_with("selector.sampler.")));
    }
}
