use super::*;
use crate::placement::FfnMode;

fn budget(bytes: u64) -> Budget {
    Budget {
        id: "gpu".into(),
        locations: vec![Location::coordinator(0)],
        physical_bytes: bytes,
        logical_bytes: bytes,
        baseline: Baseline::Forecast {
            occupied: 0,
            source: "test witness".into(),
            assumed: false,
        },
        headroom: 0,
    }
}
fn allocation(id: &str, bytes: u64) -> Allocation {
    Allocation::buffer(
        id,
        Location::coordinator(0),
        bytes,
        Lifetime::TARGET,
        Category::Workspace,
        "test",
    )
}
fn facts(allocations: Vec<Allocation>) -> Facts {
    Facts {
        allocations,
        deferred: vec![],
        allowances: vec![],
    }
}
fn request() -> Request {
    Request {
        context: 64,
        pool: Pin::Auto,
        onboard: Pin::Auto,
        prefill: Pin::Auto,
        operation: Operation::Serving,
    }
}
fn decisions(onboard: u64) -> Decisions {
    Decisions {
        attention: vec![LayerMode::Whole {
            gpu: 0,
            ffn: FfnMode::Owner,
        }],
        onboard_layers: vec![(0..onboard as usize).collect()],
        prefill: PrefillShape {
            rows: 4096,
            waves: 1,
        },
        lanes: 1,
    }
}
struct Linear {
    onboard: u64,
    maximum: u64,
    fixed: u64,
}
impl Linear {
    fn new() -> Self {
        Self {
            onboard: 0,
            maximum: 2 << 20,
            fixed: 0,
        }
    }
}
impl Recipe for Linear {
    fn prepare(&self, _: &Request) -> Result<Contract, Refusal> {
        let maximum = self.maximum / 64;
        let mut contract = Contract::fixed(
            64,
            maximum,
            1,
            facts(vec![allocation("kv", 64 + self.fixed)]),
        );
        contract.set_byte_extent("kv", ByteCurve::linear(64 + self.fixed, 64, maximum))?;
        contract.variables = vec![
            Variable::Attention {
                layer: 0,
                values: decisions(0).attention,
            },
            Variable::Prefill(vec![decisions(0).prefill]),
            Variable::Lanes(vec![1]),
        ];
        for layer in 0..self.onboard as usize {
            let variable = contract.variables.len();
            contract.variables.push(Variable::Onboard {
                layers: vec![layer],
                destinations: vec![vec![], vec![0]],
                mandatory: false,
            });
            let mut component = Component::allocation(
                allocation(&format!("expert.{layer}"), 64),
                ByteCurve::fixed(64, maximum),
                false,
                maximum,
            );
            component.activation = Activation::choice(variable, 1);
            contract.components.push(component);
        }
        Ok(contract)
    }
}
fn plan(allocations: Vec<Allocation>) -> ResolvedPlan {
    let budgets = vec![budget(4096)];
    let facts = facts(allocations);
    let phases = evaluate(&facts, &budgets).unwrap();
    ResolvedPlan {
        request: request(),
        decisions: decisions(0),
        pool_tokens: 64,
        budgets,
        facts,
        phases,
    }
}
fn receipts(plan: &ResolvedPlan) -> Vec<Receipt> {
    plan.facts
        .allocations
        .iter()
        .enumerate()
        .map(|(index, a)| Receipt {
            allocation: a.clone(),
            address: 4096 + index as u64 * 4096,
        })
        .collect()
}
fn errors(result: Result<(), Refusal>) -> Vec<Mismatch> {
    match result {
        Err(Refusal::Reconciliation(errors)) => errors,
        other => panic!("expected reconciliation refusal: {other:?}"),
    }
}

struct Prepared(Contract);
impl Recipe for Prepared {
    fn prepare(&self, _: &Request) -> Result<Contract, Refusal> {
        Ok(self.0.clone())
    }
}

fn assert_curve_census(contract: &Contract, budgets: &[Budget], selected: &[usize]) {
    contract.validate().unwrap();
    let costs = search::Compiled::new(contract, budgets).unwrap();
    let mut pages = BTreeSet::from([1, contract.maximum_pages]);
    for component in &contract.components {
        for piece in &component.bytes.0 {
            for boundary in [piece.first, piece.last] {
                for page in boundary.saturating_sub(1)..=boundary.saturating_add(1) {
                    if page >= 1 && page <= contract.maximum_pages {
                        pages.insert(page);
                    }
                }
            }
        }
    }
    for pages in pages {
        let exact = evaluate(&contract.facts(selected, pages).unwrap(), budgets).unwrap();
        assert_eq!(
            costs.phases(budgets, selected, pages).unwrap(),
            exact,
            "page {pages}"
        );
    }
}

#[test]
fn piecewise_cost_vectors_match_exact_owners_at_every_breakpoint_neighbor() {
    let mut contract = Linear {
        maximum: 64 * 20,
        fixed: 37,
        ..Linear::new()
    }
    .prepare(&request())
    .unwrap();
    let curve = ByteCurve(vec![
        BytePiece {
            first: 1,
            last: 4,
            bytes_at_first: 0,
            bytes_per_page: 16,
        },
        BytePiece {
            first: 5,
            last: 8,
            bytes_at_first: 96,
            bytes_per_page: 0,
        },
        BytePiece {
            first: 9,
            last: 20,
            bytes_at_first: 128,
            bytes_per_page: 3,
        },
    ]);
    let mut bounded = Component::allocation(allocation("index", 16), curve.clone(), false, 20);
    bounded.byte_extent(curve).unwrap();
    contract.components.push(bounded);
    let mut backing = allocation("backing", 512);
    backing.lifetime = Lifetime::RESIDENT;
    contract.components.push(Component::allocation(
        backing,
        ByteCurve::fixed(512, 20),
        false,
        20,
    ));
    let mut view = allocation("view", 64);
    view.view = Some(View {
        backing: "backing".into(),
        offset: 0,
    });
    contract.components.push(Component::allocation(
        view,
        ByteCurve::fixed(64, 20),
        false,
        20,
    ));
    let mut staging = allocation("load.staging", 65);
    staging.location = Location::PINNED;
    staging.lifetime = Lifetime {
        first: Phase::Load,
        last: Phase::Load,
    };
    contract.components.push(Component::allocation(
        staging,
        ByteCurve::fixed(65, 20),
        false,
        20,
    ));
    contract.components.push(Component {
        activation: Activation::always(),
        bytes: ByteCurve::linear(2, 1, 20),
        kind: ComponentKind::Allowance(Allowance {
            location: Location::HOST,
            bytes: 2,
            lifetime: Lifetime::TARGET,
            source: "test allowance".into(),
            assumed: true,
        }),
    });
    let mut host = budget(1 << 30);
    host.id = "host".into();
    host.locations = vec![Location::HOST, Location::PINNED];
    assert_curve_census(&contract, &[budget(1 << 30), host], &[0, 0, 0]);
}

#[test]
fn analytic_pool_solves_discontinuous_nonmonotone_and_zero_slope_pieces() {
    let mut contract = Linear {
        maximum: 64 * 12,
        ..Linear::new()
    }
    .prepare(&request())
    .unwrap();
    contract
        .set_byte_extent(
            "kv",
            ByteCurve(vec![
                BytePiece {
                    first: 1,
                    last: 3,
                    bytes_at_first: 100,
                    bytes_per_page: 10,
                },
                BytePiece {
                    first: 4,
                    last: 6,
                    bytes_at_first: 1000,
                    bytes_per_page: 0,
                },
                BytePiece {
                    first: 7,
                    last: 12,
                    bytes_at_first: 50,
                    bytes_per_page: 10,
                },
            ]),
        )
        .unwrap();
    assert_curve_census(&contract, &[budget(1 << 30)], &[0, 0, 0]);
    let result = solve(request(), vec![budget(75)], &Prepared(contract.clone())).unwrap();
    assert_eq!(result.pool_tokens, 9 * 64);
    assert!(matches!(
        solve(
            Request {
                pool: Pin::Exact(10 * 64),
                ..request()
            },
            vec![budget(75)],
            &Prepared(contract)
        ),
        Err(Refusal::Capacity { .. })
    ));
    let mut fixed = Linear {
        maximum: 1024,
        ..Linear::new()
    }
    .prepare(&request())
    .unwrap();
    fixed
        .set_byte_extent("kv", ByteCurve::fixed(75, 16))
        .unwrap();
    assert_eq!(
        solve(request(), vec![budget(75)], &Prepared(fixed))
            .unwrap()
            .pool_tokens,
        1024
    );
}

#[test]
fn exact_pins_propagate_compatibility_and_fixed_capacity_before_branching() {
    let mut contract = Linear {
        maximum: 256,
        ..Linear::new()
    }
    .prepare(&request())
    .unwrap();
    contract.variables[0] = Variable::Attention {
        layer: 0,
        values: vec![
            LayerMode::Whole {
                gpu: 0,
                ffn: FfnMode::Owner,
            },
            LayerMode::Whole {
                gpu: 0,
                ffn: FfnMode::Split,
            },
        ],
    };
    contract.variables[1] = Variable::Prefill(vec![
        decisions(0).prefill,
        PrefillShape {
            rows: 2048,
            waves: 2,
        },
    ]);
    contract.variables[2] = Variable::Lanes(vec![1, 2]);
    contract.requires = vec![
        Requires {
            when: Choice {
                variable: 1,
                option: 1,
            },
            any_of: vec![Choice {
                variable: 2,
                option: 1,
            }],
        },
        Requires {
            when: Choice {
                variable: 2,
                option: 1,
            },
            any_of: vec![Choice {
                variable: 0,
                option: 1,
            }],
        },
    ];
    let mut impossible = Component::allocation(
        allocation("impossible", 4096),
        ByteCurve::fixed(4096, 4),
        false,
        4,
    );
    impossible.activation = Activation::choice(0, 0);
    contract.components.push(impossible);
    let result = solve(
        Request {
            prefill: Pin::Exact(PrefillShape {
                rows: 2048,
                waves: 2,
            }),
            ..request()
        },
        vec![budget(1024)],
        &Prepared(contract),
    )
    .unwrap();
    assert_eq!(
        (result.decisions.prefill.rows, result.decisions.lanes),
        (2048, 2)
    );
    assert!(matches!(
        result.decisions.attention[0],
        LayerMode::Whole {
            ffn: FfnMode::Split,
            ..
        }
    ));
    assert!(!result
        .facts
        .allocations
        .iter()
        .any(|a| a.id == "impossible"));
}

#[test]
fn analytical_solve_prepares_once_and_matches_exact_small_domain_oracle() {
    struct Counted<'a>(&'a std::cell::Cell<u64>, Contract);
    impl Recipe for Counted<'_> {
        fn prepare(&self, _: &Request) -> Result<Contract, Refusal> {
            self.0.set(self.0.get() + 1);
            Ok(self.1.clone())
        }
    }
    let count = std::cell::Cell::new(0);
    let contract = Linear {
        maximum: 64 * 8,
        fixed: 3,
        ..Linear::new()
    }
    .prepare(&request())
    .unwrap();
    for available in 60..600 {
        let result = solve(
            request(),
            vec![budget(available)],
            &Counted(&count, contract.clone()),
        );
        let oracle = (1..=8)
            .filter(|&p| {
                evaluate(
                    &contract.facts(&[0, 0, 0], p).unwrap(),
                    &[budget(available)],
                )
                .is_ok()
            })
            .max();
        assert_eq!(
            result.ok().map(|p| p.pool_tokens / 64),
            oracle,
            "budget {available}"
        );
    }
    assert_eq!(count.get(), 540);
}

#[test]
fn shared_onboarding_balances_whole_and_tp2_units_and_charges_workspace_once() {
    let mut contract = Contract::fixed(64, 4, 2, facts(vec![]));
    contract.variables = vec![
        Variable::Attention {
            layer: 0,
            values: decisions(0).attention,
        },
        Variable::Prefill(vec![decisions(0).prefill]),
        Variable::Lanes(vec![1]),
    ];
    for layer in 0..4 {
        let variable = contract.variables.len();
        contract.variables.push(Variable::Onboard {
            layers: vec![layer],
            destinations: vec![vec![], vec![0], vec![1]],
            mandatory: false,
        });
        for gpu in 0..2 {
            let mut owner = allocation(&format!("layer.{layer}"), 32);
            owner.location = Location::coordinator(gpu);
            let mut term = Component::allocation(owner, ByteCurve::fixed(32, 4), false, 4);
            term.activation = Activation::choice(variable, gpu as usize + 1);
            contract.components.push(term);
        }
    }
    for gpu in 0..2 {
        let mut owner = allocation(&format!("workspace.{gpu}"), 16);
        owner.location = Location::coordinator(gpu);
        let mut term = Component::allocation(owner, ByteCurve::fixed(16, 4), false, 4);
        term.activation = Activation {
            clauses: (3..7)
                .map(|variable| {
                    vec![Choice {
                        variable,
                        option: gpu as usize + 1,
                    }]
                })
                .collect(),
        };
        contract.components.push(term);
    }
    let mut peer = budget(128);
    peer.id = "peer".into();
    peer.locations = vec![Location::coordinator(1)];
    let req = Request {
        pool: Pin::Exact(256),
        ..request()
    };
    let result = solve(
        req.clone(),
        vec![budget(128), peer.clone()],
        &Prepared(contract.clone()),
    )
    .unwrap();
    assert_eq!(
        result
            .decisions
            .onboard_layers
            .iter()
            .map(Vec::len)
            .collect::<Vec<_>>(),
        vec![2, 2]
    );
    assert_eq!(result.decisions.onboard(), 4);
    assert_eq!(
        result
            .facts
            .allocations
            .iter()
            .filter(|a| a.id.starts_with("workspace"))
            .count(),
        2
    );
    let variables = contract.variables.len();
    contract.variables.push(Variable::Onboard {
        layers: vec![4],
        destinations: vec![vec![0, 1]],
        mandatory: true,
    });
    for gpu in 0..2 {
        let mut owner = allocation(&format!("tp2.{gpu}"), 32);
        owner.location = Location::coordinator(gpu);
        let mut term = Component::allocation(owner, ByteCurve::fixed(32, 4), false, 4);
        term.activation = Activation::choice(variables, 0);
        contract.components.push(term);
    }
    let result = solve(
        Request {
            onboard: Pin::Exact(3),
            ..req
        },
        vec![budget(128), peer],
        &Prepared(contract),
    )
    .unwrap();
    assert_eq!(result.decisions.onboard(), 3);
    assert!(result
        .decisions
        .onboard_layers
        .iter()
        .all(|layers| layers.contains(&4)));
}

#[test]
fn malformed_curves_and_unrepresentable_recipes_are_typed_refusals() {
    let mut contract = Linear {
        maximum: 256,
        ..Linear::new()
    }
    .prepare(&request())
    .unwrap();
    contract
        .set_byte_extent(
            "kv",
            ByteCurve(vec![BytePiece {
                first: 2,
                last: 4,
                bytes_at_first: 1,
                bytes_per_page: 1,
            }]),
        )
        .unwrap();
    assert!(matches!(
        solve(request(), vec![budget(4096)], &Prepared(contract)),
        Err(Refusal::InvalidFact(_))
    ));
    struct Unsupported;
    impl Recipe for Unsupported {
        fn prepare(&self, _: &Request) -> Result<Contract, Refusal> {
            Err(Refusal::Unsupported)
        }
    }
    assert_eq!(
        solve(request(), vec![budget(4096)], &Unsupported).unwrap_err(),
        Refusal::Unsupported
    );
}

#[test]
fn pool_pins_are_exact_and_context_is_never_clamped() {
    let recipe = Linear::new();
    for (pin, expected) in [
        (
            65,
            Refusal::MisalignedPool {
                tokens: 65,
                page_rows: 64,
            },
        ),
        (
            0,
            Refusal::PoolBelowContext {
                tokens: 0,
                context: 64,
            },
        ),
    ] {
        assert_eq!(
            solve(
                Request {
                    pool: Pin::Exact(pin),
                    ..request()
                },
                vec![budget(1 << 30)],
                &recipe
            )
            .unwrap_err(),
            expected
        );
    }
    let exact = solve(
        Request {
            pool: Pin::Exact(256),
            ..request()
        },
        vec![budget(1 << 30)],
        &recipe,
    )
    .unwrap();
    assert_eq!(exact.pool_tokens, 256);
    assert!(matches!(
        solve(
            Request {
                pool: Pin::Exact(256),
                ..request()
            },
            vec![budget(255)],
            &recipe
        ),
        Err(Refusal::Capacity { .. })
    ));
    let automatic = solve(
        Request {
            context: 129,
            ..request()
        },
        vec![budget(192)],
        &recipe,
    )
    .unwrap();
    assert_eq!(
        (automatic.request.context, automatic.pool_tokens),
        (129, 192)
    );
    assert!(matches!(
        solve(
            Request {
                context: 193,
                ..request()
            },
            vec![budget(192)],
            &recipe
        ),
        Err(Refusal::Capacity { .. })
    ));
}

#[test]
fn automatic_target_uses_physical_card_not_logical_cap() {
    let recipe = Linear::new();
    let mut card = budget(96 << 30);
    card.logical_bytes = 24 << 30;
    assert_eq!(
        solve(request(), vec![card], &recipe).unwrap().pool_tokens,
        2 << 20
    );
    assert_eq!(
        solve(request(), vec![budget(32 << 30)], &recipe)
            .unwrap()
            .pool_tokens,
        1 << 20
    );
    let context = 3 << 20;
    let recipe = Linear {
        maximum: context,
        ..Linear::new()
    };
    assert_eq!(
        solve(
            Request {
                context,
                ..request()
            },
            vec![budget(32 << 30)],
            &recipe
        )
        .unwrap()
        .pool_tokens,
        context
    );
}

#[test]
fn pool_first_then_onboarding_and_exact_constraints() {
    let target = 1 << 20;
    let recipe = Linear {
        onboard: 2,
        ..Linear::new()
    };
    let result = solve(request(), vec![budget(target + 64)], &recipe).unwrap();
    assert_eq!(
        (result.pool_tokens, result.decisions.onboard()),
        (target, 1)
    );
    let result = solve(request(), vec![budget(target - 64)], &recipe).unwrap();
    assert_eq!(
        (result.pool_tokens, result.decisions.onboard()),
        (target - 64, 0)
    );
    let result = solve(
        Request {
            onboard: Pin::Exact(2),
            ..request()
        },
        vec![budget(target)],
        &recipe,
    )
    .unwrap();
    assert_eq!(
        (result.pool_tokens, result.decisions.onboard()),
        (target - 128, 2)
    );
    let exact = Request {
        pool: Pin::Exact(256),
        ..request()
    };
    assert_eq!(
        solve(exact, vec![budget(384)], &recipe)
            .unwrap()
            .decisions
            .onboard(),
        2
    );
    for req in [
        Request {
            onboard: Pin::Exact(3),
            ..request()
        },
        Request {
            prefill: Pin::Exact(PrefillShape {
                rows: 2048,
                waves: 1,
            }),
            ..request()
        },
    ] {
        assert_eq!(
            solve(req, vec![budget(target)], &recipe).unwrap_err(),
            Refusal::PinConflict
        );
    }
}

#[test]
fn compiled_ceiling_is_not_a_lower_automatic_onboarding_target() {
    let recipe = Linear {
        maximum: 256,
        onboard: 1,
        fixed: 0,
    };
    let result = solve(request(), vec![budget(4096)], &recipe).unwrap();
    assert_eq!((result.pool_tokens, result.decisions.onboard()), (256, 0));
}

#[test]
fn per_device_balance_uses_the_tightest_complete_cost() {
    struct Balanced;
    impl Recipe for Balanced {
        fn prepare(&self, _: &Request) -> Result<Contract, Refusal> {
            let mut contract = Contract::fixed(64, 4, 2, facts(vec![]));
            contract.variables = vec![
                Variable::Attention {
                    layer: 0,
                    values: [0, 1]
                        .map(|gpu| LayerMode::Whole {
                            gpu,
                            ffn: FfnMode::Split,
                        })
                        .to_vec(),
                },
                Variable::Prefill(vec![decisions(0).prefill]),
                Variable::Lanes(vec![1]),
            ];
            for gpu in 0..2 {
                let mut a = allocation("attention+kv", 256);
                a.location = Location::coordinator(gpu);
                let mut component =
                    Component::allocation(a, ByteCurve::linear(256, 64, 4), false, 4);
                component.byte_extent(ByteCurve::linear(256, 64, 4))?;
                component.activation = Activation::choice(0, gpu as usize);
                contract.components.push(component);
            }
            Ok(contract)
        }
    }
    let mut peer = budget(512);
    peer.id = "peer".into();
    peer.locations = vec![Location::coordinator(1)];
    let result = solve(request(), vec![budget(256), peer], &Balanced).unwrap();
    assert_eq!(result.pool_tokens, 256);
    assert_eq!(
        result.decisions.attention,
        vec![LayerMode::Whole {
            gpu: 1,
            ffn: FfnMode::Split
        }]
    );
}

#[test]
fn loading_and_ready_peaks_are_separate_and_host_domains_share_one_budget() {
    let mut load = allocation("loading", 100);
    load.lifetime = Lifetime {
        first: Phase::Load,
        last: Phase::Load,
    };
    let mut pinned = allocation("pinned", 60);
    pinned.location = Location::PINNED;
    let mut host = allocation("host", 50);
    host.location = Location::HOST;
    let mut host_budget = budget(109);
    host_budget.id = "host".into();
    host_budget.locations = vec![Location::HOST, Location::PINNED];
    let allocations = facts(vec![load, allocation("ready", 100), host, pinned]);
    assert!(
        matches!(evaluate(&allocations, &[budget(100), host_budget.clone()]), Err(Refusal::Capacity { budget, phase: Phase::PostTargetKv, required: 110, .. }) if budget == "host")
    );
    host_budget.physical_bytes = 110;
    host_budget.logical_bytes = 110;
    let rows = evaluate(&allocations, &[budget(100), host_budget]).unwrap();
    assert!(rows
        .iter()
        .filter(|r| r.budget == "gpu")
        .all(|r| r.owned == 100));
}

#[test]
fn shared_backing_views_charge_once_and_enforce_slots_alignment_and_bounds() {
    let mut backing = allocation("arena", 1024);
    backing.alignment = 256;
    let mut a = allocation("slot.0", 128);
    a.slot = Some(0);
    a.view = Some(View {
        backing: "arena".into(),
        offset: 0,
    });
    let mut b = allocation("slot.1", 128);
    b.slot = Some(1);
    b.view = Some(View {
        backing: "arena".into(),
        offset: 128,
    });
    let mut p = plan(vec![backing, a, b]);
    assert_eq!(
        p.phases
            .iter()
            .find(|p| p.phase == Phase::Ready)
            .unwrap()
            .owned,
        1024
    );
    let r = |p: &ResolvedPlan| {
        p.facts
            .allocations
            .iter()
            .map(|a| Receipt {
                address: 4096 + a.view.as_ref().map_or(0, |v| v.offset),
                allocation: a.clone(),
            })
            .collect::<Vec<_>>()
    };
    reconcile(&p, Phase::Ready, &r(&p)).unwrap();
    p.facts.allocations[2].view.as_mut().unwrap().offset = 64;
    assert!(errors(reconcile(&p, Phase::Ready, &r(&p)))
        .iter()
        .any(|e| matches!(e, Mismatch::Address(_))));
    p.facts.allocations[2].view.as_mut().unwrap().offset = 1000;
    assert!(matches!(
        evaluate(&p.facts, &p.budgets),
        Err(Refusal::InvalidFact(_))
    ));
    let p = plan(vec![allocation("owner", 100)]);
    let mut r = receipts(&p);
    r[0].address = u64::MAX;
    assert!(
        errors(reconcile(&p, Phase::Ready, &r))
            .iter()
            .any(|e| matches!(e, Mismatch::Address(_))),
        "overflowing pointer extents must fail"
    );
}

#[test]
fn missing_extra_wrong_bytes_and_equal_total_mismatches_never_cancel() {
    let p = plan(vec![allocation("a", 100), allocation("b", 100)]);
    let r = receipts(&p);
    reconcile(&p, Phase::Ready, &r).unwrap();
    assert!(errors(reconcile(&p, Phase::Ready, &r[..1])).contains(&Mismatch::Missing("b".into())));
    let mut extra = r.clone();
    let mut owner = allocation("extra", 100);
    owner.location = Location::coordinator(0);
    extra.push(Receipt {
        allocation: owner,
        address: 16384,
    });
    assert!(errors(reconcile(&p, Phase::Ready, &extra)).contains(&Mismatch::Extra("extra".into())));
    let mut wrong = r.clone();
    wrong[0].allocation.bytes += 1;
    wrong[1].allocation.bytes -= 1;
    let e = errors(reconcile(&p, Phase::Ready, &wrong));
    assert!(
        e.contains(&Mismatch::Contract("a".into())) && e.contains(&Mismatch::Contract("b".into()))
    );
    let mut wrong_device = r.clone();
    wrong_device[0].allocation.location = Location::coordinator(1);
    assert!(errors(reconcile(&p, Phase::Ready, &wrong_device))
        .contains(&Mismatch::Contract("a".into())));
    let mut duplicate = r.clone();
    duplicate.push(r[0].clone());
    assert!(
        errors(reconcile(&p, Phase::Ready, &duplicate)).contains(&Mismatch::Duplicate("a".into()))
    );
}

#[test]
fn observed_lifetimes_require_loading_owner_release_before_ready() {
    let mut staging = allocation("load", 100);
    staging.lifetime = Lifetime {
        first: Phase::Load,
        last: Phase::Load,
    };
    let p = plan(vec![staging, allocation("target", 100)]);
    let r = receipts(&p);
    let mut history = vec![
        Observation {
            receipt: r[0].clone(),
            created: Phase::Load,
            freed: Some(Phase::Load),
        },
        Observation {
            receipt: r[1].clone(),
            created: Phase::PostTargetKv,
            freed: None,
        },
    ];
    reconcile_history(&p, Phase::Ready, &history).unwrap();
    history[0].freed = None;
    assert!(errors(reconcile_history(&p, Phase::Ready, &history))
        .contains(&Mismatch::Lifetime("load".into())));
    history[0].freed = Some(Phase::PostTargetKv);
    assert!(errors(reconcile_history(&p, Phase::Ready, &history))
        .contains(&Mismatch::Lifetime("load".into())));
    history[0].freed = Some(Phase::Load);
    history[1].created = Phase::Load;
    assert!(errors(reconcile_history(&p, Phase::Ready, &history))
        .contains(&Mismatch::Lifetime("target".into())));
}

#[test]
fn deferred_reservations_cannot_authorize_readiness_and_headroom_is_not_opaque() {
    let mut p = plan(vec![allocation("target", 100)]);
    p.facts.deferred.push(allocation("draft.pending", 100));
    let r = receipts(&p);
    reconcile(&p, Phase::PostTargetKv, &r).unwrap();
    assert!(errors(reconcile(&p, Phase::Ready, &r))
        .contains(&Mismatch::Missing("draft.pending".into())));
    reconcile_usage(&p, "gpu", Phase::Ready, 100).unwrap();
    assert!(matches!(
        reconcile_usage(&p, "gpu", Phase::Ready, 101),
        Err(Refusal::Capacity { .. })
    ));
}

#[test]
fn invalid_inputs_return_typed_errors() {
    let p = plan(vec![allocation("a", 100)]);
    let mut budgets = p.budgets.clone();
    budgets[0].baseline = Baseline::Sampled {
        free: 5000,
        phase: Phase::Load,
        source: "sample".into(),
    };
    assert!(matches!(
        evaluate(&p.facts, &budgets),
        Err(Refusal::InvalidFact(_))
    ));
    let mut invalid = p.facts.clone();
    invalid.allocations.push(invalid.allocations[0].clone());
    assert!(matches!(
        evaluate(&invalid, &p.budgets),
        Err(Refusal::InvalidFact(_))
    ));
    let mut duplicate_budget = p.budgets[0].clone();
    duplicate_budget.id = "duplicate".into();
    assert!(matches!(
        evaluate(&p.facts, &[p.budgets[0].clone(), duplicate_budget]),
        Err(Refusal::InvalidFact(_))
    ));
    let mut invalid = p.facts.clone();
    invalid.allocations[0].alignment = 3;
    assert!(matches!(
        evaluate(&invalid, &p.budgets),
        Err(Refusal::InvalidFact(_))
    ));
    let mut invalid = p.facts.clone();
    invalid.allocations[0].bytes = u64::MAX;
    invalid.allocations.push(allocation("b", 1));
    assert!(matches!(
        evaluate(&invalid, &p.budgets),
        Err(Refusal::Overflow(_))
    ));
}
