use super::*;
use crate::placement::{FfnMode, planner};
use cuteafd_core::memory_layout::Category;

fn cfg() -> GlmDsaConfig {
    GlmDsaConfig::from_hf(&crate::plan::testing::glm5_config()).unwrap()
}

fn request() -> Request {
    Request {
        context: 4096,
        pool: Pin::Exact(4096),
        onboard: Pin::Auto,
        prefill: Pin::Auto,
        operation: Operation::Serving,
    }
}

fn recipe(cfg: &GlmDsaConfig) -> GlmRecipe<'_> {
    let prefill = vec![
        PrefillShape { rows: 64, waves: 1 },
        PrefillShape {
            rows: 128,
            waves: 3,
        },
    ];
    let mut recipe = GlmRecipe {
        cfg,
        context: 4096,
        page_tokens: 128,
        maximum_pages: 64,
        attention: vec![LayerMode::ContextSplit; cfg.layers],
        gpus: 2,
        prefill,
        lanes: vec![1],
        embedding: vec![EmbeddingHome::Device(0), EmbeddingHome::HostPinned],
        experts: Some(ExpertProfile {
            layers: vec![1],
            destination: vec![0, 1],
            counts: vec![0, 1],
            rank_sets: vec![vec![0, 1]],
        }),
        targets: vec![],
        hooks: vec![],
        exchanges: vec![],
        weights: vec![],
        owners: vec![],
    };
    for &shape in &recipe.prefill {
        for local in [false, true] {
            let when = vec![
                When::Prefill(shape),
                if local {
                    When::LocalLayer(1)
                } else {
                    When::RemoteLayer(1)
                },
            ];
            for device in 0..2 {
                recipe.targets.push(Case {
                    when: when.clone(),
                    value: TargetBuffers {
                        device,
                        lead: device == 0,
                        page_tokens: recipe.page_tokens,
                        maximum_pages: recipe.maximum_pages,
                        context: recipe.context,
                        partitioned: true,
                        owned_layers: vec![true; cfg.layers],
                        split_heads: true,
                        router: local,
                        native_head: false,
                        decode: Some(super::super::GlmScratch {
                            programs: 1024,
                            topk: 4096,
                        }),
                        prefill: Some((
                            shape,
                            super::super::GlmScratch {
                                programs: 2048,
                                topk: 8192,
                            },
                        )),
                        full_prefill_logits: false,
                        payload_row_bytes: if local { cfg.hidden as u64 * 4 } else { 0 },
                        core_rows: false,
                    },
                });
            }
        }
        recipe.exchanges.push(Case {
            when: vec![When::Prefill(shape)],
            value: ExchangeBuffers {
                devices: [0, 1],
                maximum_pages: recipe.maximum_pages,
                receive_bytes: vec![shape.rows * cfg.hidden as u64 * 2; (4 * shape.waves) as usize],
                terminal_abort: true,
                diagnostics: true,
                context: Some(super::super::exchanges::ContextOwners {
                    extent: recipe.context,
                    prefill: shape,
                    scratch: super::super::GlmContext {
                        topk_scratch: 512,
                        partial_scratch: 1024,
                    },
                }),
            },
        });
    }
    for home in &recipe.embedding {
        recipe.weights.push(Case {
            when: vec![When::Embedding(*home)],
            value: WeightInventory {
                ranks: 2,
                owners: vec![super::super::weights::WeightOwner {
                    allocation: Allocation::buffer(
                        "glm.weights.embedding",
                        home.location(),
                        8192,
                        Lifetime::RESIDENT,
                        Category::Embedding,
                        "test physical embedding",
                    ),
                    group: "embedding".into(),
                    format: "bf16".into(),
                }],
            },
        });
    }
    for gpu in 0..2 {
        recipe.owners.push(Case {
            when: vec![When::LocalLayer(1)],
            value: vec![Component::allocation(
                Allocation::buffer(
                    format!("expert.local.{gpu}.1"),
                    Location::coordinator(gpu),
                    16384,
                    Lifetime::RESIDENT,
                    Category::Weights,
                    "test local expert arena",
                ),
                ByteCurve::fixed(16384, recipe.maximum_pages),
                false,
                recipe.maximum_pages,
            )],
        });
    }
    for rank in 0..2 {
        recipe.owners.push(Case {
            when: vec![When::RemoteLayer(1), When::SparkRanks(vec![0, 1])],
            value: vec![Component::allocation(
                Allocation::buffer(
                    format!("expert.spark.{rank}.1"),
                    Location {
                        device: Device::Spark(rank),
                        domain: Domain::Managed,
                    },
                    12288,
                    Lifetime::RESIDENT,
                    Category::Weights,
                    "test Spark expert arena",
                ),
                ByteCurve::fixed(12288, recipe.maximum_pages),
                false,
                recipe.maximum_pages,
            )],
        });
    }
    recipe
}

fn budgets() -> Vec<Budget> {
    vec![
        vec![Location::coordinator(0)],
        vec![Location::coordinator(1)],
        vec![Location::HOST, Location::PINNED],
        vec![Location {
            device: Device::Spark(0),
            domain: Domain::Managed,
        }],
        vec![Location {
            device: Device::Spark(1),
            domain: Domain::Managed,
        }],
    ]
    .into_iter()
    .enumerate()
    .map(|(i, locations)| Budget {
        id: format!("physical.{i}"),
        locations,
        physical_bytes: 1 << 40,
        logical_bytes: 1 << 40,
        baseline: Baseline::Forecast {
            occupied: 0,
            source: "test physical witness".into(),
            assumed: false,
        },
        headroom: 0,
    })
    .collect()
}

#[test]
fn one_contract_activates_physical_owners_by_shape_home_and_unique_target_count() {
    let cfg = cfg();
    let recipe = recipe(&cfg);
    let contract = recipe.prepare(&request()).unwrap();
    for prefill in 0..2 {
        for embedding in 0..2 {
            for onboard in 0..2 {
                let selected = [0, 0, prefill, 0, embedding, onboard, 0];
                let facts = contract.facts(&selected, 32).unwrap();
                planner::evaluate(&facts, &budgets()).unwrap();
                let decisions = contract.decisions(&selected).unwrap();
                assert_eq!(decisions.prefill, recipe.prefill[prefill]);
                assert_eq!(decisions.embedding, Some(recipe.embedding[embedding]));
                assert_eq!(
                    decisions.onboard_layers,
                    if onboard == 0 {
                        vec![vec![], vec![]]
                    } else {
                        vec![vec![1], vec![1]]
                    }
                );
                let has = |name: &str| facts.allocations.iter().any(|owner| owner.id == name);
                assert_eq!(has("expert.local.0.1"), onboard != 0);
                assert_eq!(has("expert.local.1.1"), onboard != 0);
                assert_eq!(has("expert.spark.0.1"), onboard == 0);
                assert_eq!(has("expert.spark.1.1"), onboard == 0);
                let owner = |name: &str| facts.allocations.iter().find(|a| a.id == name).unwrap();
                assert_eq!(
                    owner("glm.weights.embedding").location,
                    recipe.embedding[embedding].location()
                );
                assert_eq!(
                    owner("glm.target.gpu.0.kv.0").bytes,
                    33 * super::super::RECORD_PAGE_BYTES
                );
                assert_eq!(
                    owner("glm.target.gpu.0.prefill.rows.h").bytes,
                    recipe.prefill[prefill].rows
                        * recipe.prefill[prefill].waves
                        * cfg.hidden as u64
                        * 2
                );
                assert_eq!(has("glm.target.gpu.0.prefill.rows.payload"), onboard != 0);
                assert_eq!(has("glm.target.gpu.1.prefill.slot.2.h"), prefill == 1);
            }
        }
    }
    // The common solver, not a family count loop, enforces the exact pins.
    let mut exact = request();
    exact.onboard = Pin::Exact(1);
    exact.prefill = Pin::Exact(recipe.prefill[0]);
    let plan = planner::solve(exact.clone(), budgets(), &recipe).unwrap();
    assert_eq!(plan.decisions.onboard_layers, vec![vec![1], vec![1]]);
    exact.onboard = Pin::Exact(2);
    assert!(matches!(
        planner::solve(exact, budgets(), &recipe),
        Err(Refusal::PinConflict)
    ));
}

#[test]
fn shared_serving_owners_compose_without_family_private_geometry_or_policy() {
    use planner::serving::{HostPrefixOwners, SelectorOwners, ServingOwners};
    let cfg = cfg();
    for device_sampling in [false, true] {
        let mut recipe = recipe(&cfg);
        let serving = ServingOwners {
            selector: SelectorOwners { location: Location::coordinator(0),
                capacity: super::super::DECODE_ROWS as usize, vocab: cfg.vocab_size, device_sampling },
            prefix: HostPrefixOwners { quota_bytes: 2 * 4096 + 17, chunk_bytes: 4096 },
        };
        let expected = serving.allocations().unwrap();
        recipe.owners.push(Case {when: vec![], value: serving.components(recipe.maximum_pages).unwrap()});
        let contract = recipe.prepare(&request()).unwrap();
        for prefill in 0..2 {
            for embedding in 0..2 {
                for onboard in 0..2 {
                    let selected = [0, 0, prefill, 0, embedding, onboard, 0];
                    let facts = contract.facts(&selected, 32).unwrap();
                    planner::evaluate(&facts, &budgets()).unwrap();
                    assert_eq!(facts.allocations.iter().filter(|a| a.id.starts_with("selector.")
                        || a.id.starts_with("host.prefix.")).cloned().collect::<Vec<_>>(), expected);
                    assert!(!facts.allocations.iter().any(|a| a.id == "host.prefix.2"));
                }
            }
        }
    }
}

#[test]
fn named_conditions_intersect_count_domains_and_reject_private_indexes() {
    let cfg = cfg();
    let mut recipe = recipe(&cfg);
    recipe.owners.push(Case {
        when: vec![When::Onboard { first: 0, last: 1 }, When::LocalLayer(1)],
        value: vec![],
    });
    recipe.prepare(&request()).unwrap();
    recipe
        .owners
        .last_mut()
        .unwrap()
        .when
        .push(When::RemoteLayer(1));
    assert!(matches!(
        recipe.prepare(&request()),
        Err(Refusal::InvalidFact(_))
    ));
    recipe.owners.pop();
    recipe.owners[0].value[0].activation = Activation::choice(99, 0);
    assert!(matches!(
        recipe.prepare(&request()),
        Err(Refusal::InvalidFact(_))
    ));
}

#[test]
fn composition_refuses_mismatched_geometry_domains_and_non_target_experts() {
    let cfg = cfg();
    let base = recipe(&cfg);
    let refuses = |recipe: GlmRecipe<'_>| {
        assert!(matches!(
            recipe.prepare(&request()),
            Err(Refusal::InvalidFact(_))
        ))
    };
    let mut bad = base.clone();
    bad.targets[0].value.page_tokens = 64;
    refuses(bad);
    let mut bad = base.clone();
    bad.targets[0].value.owned_layers[0] = false;
    refuses(bad);
    let mut bad = base.clone();
    bad.exchanges[0].value.context.as_mut().unwrap().extent = 2048;
    refuses(bad);
    let mut bad = base.clone();
    bad.experts.as_mut().unwrap().layers = vec![0];
    refuses(bad);
    let mut bad = base.clone();
    bad.experts.as_mut().unwrap().layers = vec![2];
    refuses(bad);
    let mut bad = base.clone();
    bad.experts.as_mut().unwrap().rank_sets.clear();
    refuses(bad);
    let mut bad = base.clone();
    bad.prefill.push(bad.prefill[0]);
    refuses(bad);
    let mut bad = base.clone();
    bad.weights[0].when = vec![When::Embedding(EmbeddingHome::Device(1))];
    refuses(bad);
    let mut bad = base.clone();
    bad.targets[0].value.full_prefill_logits = true;
    refuses(bad);
    let mut bad = base.clone();
    bad.attention = vec![
        LayerMode::Whole {
            gpu: 0,
            ffn: FfnMode::Split
        };
        cfg.layers
    ];
    refuses(bad);
    let mut beyond = request();
    beyond.context = 8192;
    assert!(matches!(
        base.prepare(&beyond),
        Err(Refusal::InvalidFact(_))
    ));
}
