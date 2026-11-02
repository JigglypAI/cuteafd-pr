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
        sequential: vec![],
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
        if self.onboard != 0 {
            let variable = contract.variables.len();
            contract.variables.push(Variable::Onboard {
                layers: (0..self.onboard as usize).collect(),
                destination: vec![0],
                counts: (0..=self.onboard as usize).collect(),
            });
            for layer in 0..self.onboard as usize {
                let mut component = Component::allocation(
                    allocation(&format!("expert.{layer}"), 64),
                    ByteCurve::fixed(64, maximum),
                    false,
                    maximum,
                );
                component.activation = Activation {
                    clauses: (layer + 1..=self.onboard as usize)
                        .map(|option| vec![Choice { variable, option }])
                        .collect(),
                };
                contract.components.push(component);
            }
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

fn two_gpus(first: u64, second: u64) -> Vec<Budget> {
    let mut peer = budget(second);
    peer.id = "peer".into();
    peer.locations = vec![Location::coordinator(1)];
    vec![budget(first), peer]
}
fn profile_contract(maximum: u64) -> Contract {
    let mut contract = Contract::fixed(64, maximum, 2, facts(vec![]));
    contract.variables = vec![
        Variable::Attention {
            layer: 0,
            values: decisions(0).attention,
        },
        Variable::Prefill(vec![decisions(0).prefill]),
        Variable::Lanes(vec![1]),
    ];
    contract
}
fn add_profile(
    contract: &mut Contract,
    layers: Vec<usize>,
    destination: Vec<u8>,
    counts: Vec<usize>,
    bytes: &[u64],
) -> usize {
    assert_eq!(counts.len(), bytes.len());
    let variable = contract.variables.len();
    for &gpu in &destination {
        for (option, &bytes) in bytes.iter().enumerate().filter(|(_, bytes)| **bytes != 0) {
            let mut owner = allocation(&format!("profile.{variable}.{option}.{gpu}"), bytes);
            owner.location = Location::coordinator(gpu);
            let mut component = Component::allocation(
                owner,
                ByteCurve::fixed(bytes, contract.maximum_pages),
                false,
                contract.maximum_pages,
            );
            component.activation = Activation::choice(variable, option);
            contract.components.push(component);
        }
    }
    contract.variables.push(Variable::Onboard {
        layers,
        destination,
        counts,
    });
    variable
}

#[test]
fn shared_onboarding_balances_ordered_profiles_and_charges_workspace_once() {
    let mut contract = profile_contract(4);
    for (gpu, layers) in [(0, vec![0, 1, 2, 3]), (1, vec![3, 2, 1, 0])] {
        let variable = add_profile(
            &mut contract,
            layers,
            vec![gpu],
            (0..=4).collect(),
            &[0, 32, 64, 96, 128],
        );
        let mut owner = allocation(&format!("workspace.{gpu}"), 16);
        owner.location = Location::coordinator(gpu);
        let mut term = Component::allocation(owner, ByteCurve::fixed(16, 4), false, 4);
        term.activation = Activation {
            clauses: (1..=4)
                .map(|option| vec![Choice { variable, option }])
                .collect(),
        };
        contract.components.push(term);
    }
    let req = Request {
        pool: Pin::Exact(256),
        ..request()
    };
    let result = solve(req.clone(), two_gpus(128, 128), &Prepared(contract.clone())).unwrap();
    assert_eq!(
        result.decisions.onboard_layers,
        vec![vec![0, 1], vec![2, 3]]
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
    add_profile(&mut contract, vec![4], vec![0, 1], vec![1], &[32]);
    let result = solve(
        Request {
            onboard: Pin::Exact(3),
            ..req
        },
        two_gpus(128, 128),
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
fn required_onboarding_domains_survive_target_propagation() {
    let mut contract = profile_contract(8);
    let variable = add_profile(&mut contract, vec![7], vec![1], vec![0, 1], &[0, 32]);
    let mut remote = allocation("remote.experts", 300);
    remote.location = Location {
        device: Device::Spark(0),
        domain: Domain::Device,
    };
    let mut remote = Component::allocation(remote, ByteCurve::fixed(300, 8), false, 8);
    remote.activation = Activation::choice(variable, 0);
    contract.components.push(remote);
    contract.requires.push(Requires {
        when: Choice {
            variable: 0,
            option: 0,
        },
        any_of: vec![Choice {
            variable,
            option: 1,
        }],
    });
    let mut spark = budget(128);
    spark.id = "spark".into();
    spark.locations = vec![Location {
        device: Device::Spark(0),
        domain: Domain::Device,
    }];
    let mut budgets = two_gpus(128, 128);
    budgets.push(spark);
    let result = solve(request(), budgets, &Prepared(contract)).unwrap();
    assert_eq!(result.pool_tokens, 8 * 64);
    assert_eq!(result.decisions.onboard_layers, vec![vec![], vec![7]]);
    assert!(!result
        .facts
        .allocations
        .iter()
        .any(|a| a.id == "remote.experts"));
}

#[test]
fn below_target_onboards_on_nonbinding_device_without_spending_one_page() {
    let mut contract = profile_contract(8);
    let mut kv =
        Component::allocation(allocation("kv", 64), ByteCurve::linear(64, 64, 8), false, 8);
    kv.byte_extent(ByteCurve::linear(64, 64, 8)).unwrap();
    contract.components.push(kv);
    add_profile(
        &mut contract,
        vec![0, 1],
        vec![0],
        vec![0, 1, 2],
        &[0, 1, 2],
    );
    add_profile(
        &mut contract,
        vec![2, 3],
        vec![1],
        vec![0, 1, 2],
        &[0, 17, 49],
    );
    let budgets = two_gpus(256, 50);
    let automatic = solve(request(), budgets.clone(), &Prepared(contract.clone())).unwrap();
    let pinned = solve(
        Request {
            pool: Pin::Exact(256),
            ..request()
        },
        budgets,
        &Prepared(contract),
    )
    .unwrap();
    assert_eq!(automatic.pool_tokens, 256);
    assert_eq!(automatic.decisions.onboard_layers, vec![vec![], vec![2, 3]]);
    assert_eq!(automatic.decisions, pinned.decisions);
}

#[test]
fn exact_completion_prefers_two_whole_layers_over_one_tp2_layer() {
    let mut contract = profile_contract(4);
    add_profile(&mut contract, vec![0], vec![0, 1], vec![0, 1], &[0, 50]);
    add_profile(&mut contract, vec![0], vec![0], vec![0, 1], &[0, 60]);
    add_profile(&mut contract, vec![1], vec![1], vec![0, 1], &[0, 60]);
    let result = solve(request(), two_gpus(100, 100), &Prepared(contract)).unwrap();
    assert_eq!(result.decisions.onboard_layers, vec![vec![0], vec![1]]);
    assert_eq!(result.decisions.onboard(), 2);
}

#[test]
fn active_gpu_balance_is_not_masked_by_host_or_load_slack() {
    let mut contract = profile_contract(4);
    add_profile(
        &mut contract,
        vec![0, 1, 2, 3],
        vec![0],
        (0..=4).collect(),
        &[0, 32, 64, 96, 128],
    );
    add_profile(
        &mut contract,
        vec![3, 2, 1, 0],
        vec![1],
        (0..=4).collect(),
        &[0, 32, 64, 96, 128],
    );
    for gpu in 0..2 {
        let mut load = allocation(&format!("load.{gpu}"), 128);
        load.location = Location::coordinator(gpu);
        load.lifetime = Lifetime {
            first: Phase::Load,
            last: Phase::Load,
        };
        contract.components.push(Component::allocation(
            load,
            ByteCurve::fixed(128, 4),
            false,
            4,
        ));
    }
    let mut budgets = two_gpus(128, 128);
    let mut host = budget(1);
    host.id = "host".into();
    host.locations = vec![Location::HOST, Location::PINNED];
    budgets.push(host);
    let first = solve(request(), budgets.clone(), &Prepared(contract.clone())).unwrap();
    assert_eq!(first.decisions.onboard_layers, vec![vec![0, 1], vec![2, 3]]);
    for _ in 0..3 {
        assert_eq!(
            solve(request(), budgets.clone(), &Prepared(contract.clone())).unwrap(),
            first
        );
    }
}

#[test]
fn capacity_shaving_keeps_resource_failure_distinct_from_pin_conflict() {
    let mut contract = profile_contract(1);
    contract.variables[2] = Variable::Lanes(vec![1, 2]);
    for gpu in 0..2 {
        let mut owner = allocation(&format!("choice.{gpu}"), 65);
        owner.location = Location::coordinator(gpu);
        let mut component = Component::allocation(owner, ByteCurve::fixed(65, 1), false, 1);
        component.activation = Activation::choice(2, gpu as usize);
        contract.components.push(component);
    }
    assert!(matches!(
        solve(request(), two_gpus(64, 64), &Prepared(contract.clone())),
        Err(Refusal::Capacity { .. })
    ));
    assert_eq!(
        solve(
            Request {
                onboard: Pin::Exact(1),
                ..request()
            },
            two_gpus(64, 64),
            &Prepared(contract)
        )
        .unwrap_err(),
        Refusal::PinConflict
    );
}

#[test]
fn correlated_count_profiles_match_exhaustive_full_score_oracle() {
    let mut contract = profile_contract(6);
    contract.variables[0] = Variable::Attention {
        layer: 0,
        values: vec![
            LayerMode::Whole {
                gpu: 0,
                ffn: FfnMode::Owner,
            },
            LayerMode::Whole {
                gpu: 1,
                ffn: FfnMode::Owner,
            },
        ],
    };
    let left = add_profile(
        &mut contract,
        vec![0, 1],
        vec![0],
        vec![0, 1, 2],
        &[0, 31, 79],
    );
    let right = add_profile(
        &mut contract,
        vec![1, 0],
        vec![1],
        vec![0, 1, 2],
        &[0, 23, 63],
    );
    let tp2 = add_profile(&mut contract, vec![2], vec![0, 1], vec![0, 1], &[0, 17]);
    contract.requires.push(Requires {
        when: Choice {
            variable: 0,
            option: 1,
        },
        any_of: vec![
            Choice {
                variable: right,
                option: 1,
            },
            Choice {
                variable: right,
                option: 2,
            },
        ],
    });
    let curve = ByteCurve(vec![
        BytePiece {
            first: 1,
            last: 3,
            bytes_at_first: 20,
            bytes_per_page: 19,
        },
        BytePiece {
            first: 4,
            last: 6,
            bytes_at_first: 12,
            bytes_per_page: 13,
        },
    ]);
    for gpu in 0..2 {
        let mut owner = allocation("kv", 20);
        owner.location = Location::coordinator(gpu);
        let mut component = Component::allocation(owner, curve.clone(), false, 6);
        component.byte_extent(curve.clone()).unwrap();
        component.activation = Activation::choice(0, gpu as usize);
        contract.components.push(component);
    }
    let mut joint = Component::allocation(
        allocation("joint.scratch", 27),
        ByteCurve::fixed(27, 6),
        false,
        6,
    );
    joint.activation = Activation {
        clauses: vec![vec![
            Choice {
                variable: left,
                option: 1,
            },
            Choice {
                variable: tp2,
                option: 1,
            },
        ]],
    };
    contract.components.push(joint);
    for a in [40, 70, 100] {
        for b in [35, 65, 95] {
            for onboard in [Pin::Auto, Pin::Exact(0), Pin::Exact(2), Pin::Exact(3)] {
                for pool in [Pin::Auto, Pin::Exact(128), Pin::Exact(384)] {
                    let budgets = two_gpus(a, b);
                    let req = Request {
                        pool,
                        onboard,
                        ..request()
                    };
                    let mut oracle = None;
                    for attention in 0..2 {
                        for left_count in 0..3 {
                            for right_count in 0..3 {
                                for tp_count in 0..2 {
                                    let assignment =
                                        vec![attention, 0, 0, left_count, right_count, tp_count];
                                    if contract.requires.iter().any(|r| {
                                        assignment[r.when.variable] == r.when.option
                                            && !r
                                                .any_of
                                                .iter()
                                                .any(|c| assignment[c.variable] == c.option)
                                    }) {
                                        continue;
                                    }
                                    let Ok(decisions) = contract.decisions(&assignment) else {
                                        continue;
                                    };
                                    if matches!(onboard, Pin::Exact(n) if decisions.onboard() != n)
                                    {
                                        continue;
                                    }
                                    for pages in 1..=6 {
                                        if matches!(pool, Pin::Exact(n) if pages * 64 != n) {
                                            continue;
                                        }
                                        let Ok(phases) = evaluate(
                                            &contract.facts(&assignment, pages).unwrap(),
                                            &budgets,
                                        ) else {
                                            continue;
                                        };
                                        let active = phases
                                            .iter()
                                            .filter(|p| p.phase >= Phase::Warmup)
                                            .map(|p| p.available - p.owned - p.opaque)
                                            .min()
                                            .unwrap();
                                        let global = phases
                                            .iter()
                                            .map(|p| p.available - p.owned - p.opaque)
                                            .min()
                                            .unwrap();
                                        let score = (pages, decisions.onboard(), active, global);
                                        if oracle.as_ref().is_none_or(
                                            |(old, choices): &(
                                                (u64, u64, u64, u64),
                                                Vec<usize>,
                                            )| {
                                                score > *old
                                                    || (score == *old && assignment < *choices)
                                            },
                                        ) {
                                            oracle = Some((score, assignment.clone()));
                                        }
                                    }
                                }
                            }
                        }
                    }
                    let result = solve(req.clone(), budgets, &Prepared(contract.clone()));
                    match oracle {
                        Some((score, assignment)) => {
                            let result =
                                result.unwrap_or_else(|e| panic!("{a}/{b} {req:?}: {e:?}"));
                            assert_eq!(result.pool_tokens / 64, score.0, "{a}/{b} {req:?}");
                            assert_eq!(
                                result.decisions,
                                contract.decisions(&assignment).unwrap(),
                                "{a}/{b} {req:?}"
                            );
                        }
                        None => assert!(result.is_err(), "{a}/{b} {req:?}"),
                    }
                }
            }
        }
    }
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
fn compiled_ceiling_uses_spare_memory_for_onboarding() {
    let recipe = Linear {
        maximum: 256,
        onboard: 1,
        fixed: 0,
    };
    let result = solve(request(), vec![budget(4096)], &recipe).unwrap();
    assert_eq!((result.pool_tokens, result.decisions.onboard()), (256, 1));
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

fn packing_plan() -> ResolvedPlan {
    let mut p = plan(vec![allocation("target", 100)]);
    let mut a = allocation("pack.a", 300);
    a.lifetime = Lifetime {
        first: Phase::DraftLoad,
        last: Phase::DraftLoad,
    };
    let mut b = a.clone();
    b.id = "pack.b".into();
    b.bytes = 200;
    b.shape = vec![200];
    p.facts.sequential.push(Sequential {
        id: "packing".into(),
        members: vec![a, b],
    });
    p.phases = evaluate(&p.facts, &p.budgets).unwrap();
    p
}

fn packing_history(p: &ResolvedPlan) -> (Vec<Observation>, Vec<OwnerEvent>) {
    let ordinary = vec![Observation {
        receipt: receipts(p)[0].clone(),
        created: Phase::PostTargetKv,
        freed: None,
    }];
    let mut edges = vec![OwnerEvent::Allocated {
        receipt: ordinary[0].receipt.clone(),
        phase: Phase::PostTargetKv,
    }];
    for member in &p.facts.sequential[0].members {
        let receipt = Receipt {
            allocation: member.clone(),
            address: 32768,
        };
        edges.push(OwnerEvent::Allocated {
            receipt: receipt.clone(),
            phase: Phase::DraftLoad,
        });
        edges.push(OwnerEvent::Freed {
            receipt,
            phase: Phase::DraftLoad,
        });
    }
    (ordinary, edges)
}

#[test]
fn sequential_sources_charge_maximum_but_require_every_actual_pair() {
    let p = packing_plan();
    let at = |phase| p.phases.iter().find(|r| r.phase == phase).unwrap().owned;
    assert_eq!(at(Phase::DraftLoad), 400);
    assert_eq!(at(Phase::Ready), 100);
    let (ordinary, edges) = packing_history(&p);
    reconcile_events(&p, Phase::Ready, &edges, &[]).unwrap();
    assert!(errors(reconcile_history(&p, Phase::Ready, &ordinary))
        .contains(&Mismatch::Missing("pack.a".into())));
    let mut missing = edges.clone();
    missing.truncate(3);
    assert!(errors(reconcile_events(&p, Phase::Ready, &missing, &[]))
        .contains(&Mismatch::Missing("pack.b".into())));
    let mut retained = edges.clone();
    retained.pop();
    assert!(errors(reconcile_events(&p, Phase::Ready, &retained, &[]))
        .contains(&Mismatch::Lifetime("pack.b".into())));
    let overlap = [
        edges[0].clone(),
        edges[1].clone(),
        edges[3].clone(),
        edges[2].clone(),
        edges[4].clone(),
    ];
    assert!(errors(reconcile_events(&p, Phase::Ready, &overlap, &[]))
        .contains(&Mismatch::Lifetime("packing".into())));
    let wrong = edges
        .iter()
        .cloned()
        .map(|mut event| {
            let receipt = match &mut event {
                OwnerEvent::Allocated { receipt, .. } | OwnerEvent::Freed { receipt, .. } => {
                    receipt
                }
            };
            receipt.allocation.bytes += 1;
            event
        })
        .collect::<Vec<_>>();
    let errors = errors(reconcile_events(&p, Phase::Ready, &wrong, &[]));
    assert!(errors.contains(&Mismatch::Contract("pack.a".into())));
    assert!(errors.contains(&Mismatch::Contract("pack.b".into())));
}

#[test]
fn sequential_sources_reject_duplicate_extra_wrong_device_address_and_phase() {
    let p = packing_plan();
    let (_, edges) = packing_history(&p);
    for mutation in 0..8 {
        let mut bad = edges.clone();
        match mutation {
            0 => bad.extend_from_slice(&edges[1..3]),
            1 => {
                if let OwnerEvent::Allocated { receipt, .. } = &mut bad[1] {
                    receipt.allocation.id = "unplanned".into();
                }
            }
            2 => {
                if let OwnerEvent::Allocated { receipt, .. } = &mut bad[1] {
                    receipt.allocation.location = Location::coordinator(1);
                }
            }
            3 => {
                if let OwnerEvent::Allocated { receipt, .. } = &mut bad[1] {
                    receipt.address = u64::MAX;
                }
            }
            4 => {
                if let OwnerEvent::Freed { phase, .. } = &mut bad[2] {
                    *phase = Phase::Warmup;
                }
            }
            5 => {
                if let OwnerEvent::Freed { receipt, .. } = &mut bad[2] {
                    receipt.address += 1;
                }
            }
            6 => bad.insert(3, edges[2].clone()),
            7 => {
                bad.remove(0);
            }
            _ => unreachable!(),
        }
        assert!(
            reconcile_events(&p, Phase::Ready, &bad, &[]).is_err(),
            "mutation {mutation}"
        );
    }
}

#[test]
fn sequential_history_proves_same_phase_owner_order_and_address_reuse() {
    let mut p = packing_plan();
    p.facts.allocations[0].lifetime.first = Phase::DraftLoad;
    p.phases = evaluate(&p.facts, &p.budgets).unwrap();
    let (_, mut events) = packing_history(&p);
    let mut owner = events.remove(0);
    if let OwnerEvent::Allocated { receipt, phase } = &mut owner {
        *phase = Phase::DraftLoad;
        receipt.address = 32768;
    }
    // A retained destination can reuse the source address only after its free.
    events.push(owner.clone());
    reconcile_events(&p, Phase::Ready, &events, &[]).unwrap();
    events.pop();
    events.insert(1, owner);
    assert!(errors(reconcile_events(&p, Phase::Ready, &events, &[]))
        .iter()
        .any(|e| matches!(e, Mismatch::Address(_))));
    let (ordinary, _) = packing_history(&p);
    assert!(
        errors(reconcile(&p, Phase::Ready, &[ordinary[0].receipt.clone()]))
            .contains(&Mismatch::Missing("pack.a".into()))
    );
}

#[test]
fn sequential_groups_may_coexist_but_their_live_addresses_cannot_overlap() {
    let mut p = packing_plan();
    let mut other = p.facts.sequential[0].members[0].clone();
    other.id = "other.source".into();
    p.facts.sequential.push(Sequential {
        id: "other.packing".into(),
        members: vec![other.clone()],
    });
    p.phases = evaluate(&p.facts, &p.budgets).unwrap();
    let (_, original) = packing_history(&p);
    for address in [65536, 32768] {
        let r = Receipt {
            allocation: other.clone(),
            address,
        };
        let events = [
            original[0].clone(),
            original[1].clone(),
            OwnerEvent::Allocated {
                receipt: r.clone(),
                phase: Phase::DraftLoad,
            },
            OwnerEvent::Freed {
                receipt: r,
                phase: Phase::DraftLoad,
            },
            original[2].clone(),
            original[3].clone(),
            original[4].clone(),
        ];
        let result = reconcile_events(&p, Phase::Ready, &events, &[]);
        if address == 65536 {
            result.unwrap();
        } else {
            assert!(errors(result)
                .iter()
                .any(|e| matches!(e, Mismatch::Address(_))));
        }
    }
}

#[test]
fn sequential_contract_costs_match_exact_evaluator_and_capacity() {
    let p = packing_plan();
    let mut contract = Contract::fixed(64, 1, 1, p.facts.clone());
    contract.variables = vec![
        Variable::Prefill(vec![decisions(0).prefill]),
        Variable::Lanes(vec![1]),
    ];
    let mut req = request();
    req.pool = Pin::Exact(64);
    let mut b = budget(400);
    let solved = solve(req.clone(), vec![b.clone()], &Prepared(contract.clone())).unwrap();
    assert_eq!(solved.facts, p.facts);
    assert_eq!(solved.phases, evaluate(&p.facts, &[b.clone()]).unwrap());
    b.physical_bytes = 399;
    b.logical_bytes = 399;
    assert!(matches!(
        solve(req, vec![b], &Prepared(contract)),
        Err(Refusal::Capacity {
            phase: Phase::DraftLoad,
            ..
        })
    ));
    for malformed in 0..4 {
        let mut bad = p.facts.clone();
        match malformed {
            0 => bad.sequential[0].members.clear(),
            1 => bad.sequential[0].members[1].location = Location::PINNED,
            2 => bad.sequential[0].members[1].lifetime.last = Phase::Ready,
            3 => bad.sequential[0].members[1].id = "target".into(),
            _ => unreachable!(),
        }
        assert!(matches!(
            evaluate(&bad, &p.budgets),
            Err(Refusal::InvalidFact(_))
        ));
    }
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
