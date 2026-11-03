//! Finite GLM component composition. This is not yet wired to live admission:
//! expert/serving owners and startup history must join before Ready is allowed.
//! Loading provenance never changes an owner's charge or its activation.
use super::{PAGE_ROWS, weights::WeightInventory};
use super::{buffers::TargetBuffers, exchanges::ExchangeBuffers, hooks::HookBuffers};
use crate::families::glm5::GlmDsaConfig;
use crate::placement::{LayerMode, planner::*};
use std::collections::{BTreeMap, BTreeSet};

/// A condition names a domain value, never a solver variable index. Conditions
/// in one case are conjunctive; membership in a count interval is disjunctive.
#[derive(Debug, Clone)]
pub enum When {
    Prefill(PrefillShape),
    Lanes(u64),
    Embedding(EmbeddingHome),
    Onboard { first: usize, last: usize },
    LocalLayer(usize),
    RemoteLayer(usize),
    SparkRanks(Vec<u8>),
}

#[derive(Debug, Clone)]
pub struct Case<T> {
    pub when: Vec<When>,
    pub value: T,
}

/// One canonical target-only prefix. TP2 names both destinations but counts a
/// routed layer once. A fallback rank domain is absent only for all-local work.
#[derive(Debug, Clone)]
pub struct ExpertProfile {
    pub layers: Vec<usize>,
    pub destination: Vec<u8>,
    pub counts: Vec<usize>,
    pub rank_sets: Vec<Vec<u8>>,
}

/// Physical subrecipes prepared from headers and selected program tables, not
/// pre-subtracted free-memory totals. This builder does not select a pool or
/// solve once per count. Additional owners use the same named finite domains.
#[derive(Debug, Clone)]
pub struct GlmRecipe<'a> {
    pub cfg: &'a GlmDsaConfig,
    pub context: u64,
    pub page_tokens: u64,
    pub maximum_pages: u64,
    /// The executor's existing attention ownership, unchanged by this port.
    pub attention: Vec<LayerMode>,
    pub gpus: usize,
    pub prefill: Vec<PrefillShape>,
    pub lanes: Vec<u64>,
    pub embedding: Vec<EmbeddingHome>,
    pub experts: Option<ExpertProfile>,
    pub targets: Vec<Case<TargetBuffers>>,
    pub hooks: Vec<Case<HookBuffers>>,
    pub exchanges: Vec<Case<ExchangeBuffers>>,
    pub weights: Vec<Case<WeightInventory>>,
    /// Expert, drafter, serving, code/graph and temporary physical components.
    /// Their constructors remain responsible for independent actual receipts.
    pub owners: Vec<Case<Vec<Component>>>,
}

struct Domains<'a> {
    prefill: (usize, &'a [PrefillShape]),
    lanes: (usize, &'a [u64]),
    embedding: (usize, &'a [EmbeddingHome]),
    onboard: Option<(usize, &'a ExpertProfile)>,
    remote: Option<usize>,
}

fn invalid(detail: &str) -> Refusal {
    Refusal::InvalidFact(format!("GLM component contract: {detail}"))
}

impl Domains<'_> {
    fn activation(&self, conditions: &[When]) -> Result<Activation, Refusal> {
        let mut domains = BTreeMap::<usize, BTreeSet<usize>>::new();
        for condition in conditions {
            let (variable, options): (_, Vec<_>) = match condition {
                When::Prefill(shape) => (
                    self.prefill.0,
                    self.prefill
                        .1
                        .iter()
                        .enumerate()
                        .filter_map(|(i, value)| (value == shape).then_some(i))
                        .collect(),
                ),
                When::Lanes(lanes) => (
                    self.lanes.0,
                    self.lanes
                        .1
                        .iter()
                        .enumerate()
                        .filter_map(|(i, value)| (value == lanes).then_some(i))
                        .collect(),
                ),
                When::Embedding(home) => (
                    self.embedding.0,
                    self.embedding
                        .1
                        .iter()
                        .enumerate()
                        .filter_map(|(i, value)| (value == home).then_some(i))
                        .collect(),
                ),
                When::Onboard { first, last } => {
                    let (variable, profile) = self
                        .onboard
                        .ok_or_else(|| invalid("count without routed experts"))?;
                    if first > last || *last > profile.layers.len() {
                        return Err(invalid("invalid onboarding interval"));
                    }
                    (
                        variable,
                        profile
                            .counts
                            .iter()
                            .enumerate()
                            .filter_map(|(i, count)| (first <= count && count <= last).then_some(i))
                            .collect(),
                    )
                }
                When::LocalLayer(layer) | When::RemoteLayer(layer) => {
                    let (variable, profile) = self
                        .onboard
                        .ok_or_else(|| invalid("layer without routed experts"))?;
                    let position = profile
                        .layers
                        .iter()
                        .position(|value| value == layer)
                        .ok_or_else(|| invalid("condition names a non-target routed layer"))?;
                    let local = matches!(condition, When::LocalLayer(_));
                    (
                        variable,
                        profile
                            .counts
                            .iter()
                            .enumerate()
                            .filter_map(|(i, &count)| ((position < count) == local).then_some(i))
                            .collect(),
                    )
                }
                When::SparkRanks(ranks) => {
                    let (_, profile) = self
                        .onboard
                        .ok_or_else(|| invalid("ranks without routed experts"))?;
                    (
                        self.remote
                            .ok_or_else(|| invalid("ranks without remote homes"))?,
                        profile
                            .rank_sets
                            .iter()
                            .enumerate()
                            .filter_map(|(i, value)| (value == ranks).then_some(i))
                            .collect(),
                    )
                }
            };
            if options.is_empty() {
                return Err(invalid("condition has no domain value"));
            }
            let options: BTreeSet<_> = options.into_iter().collect();
            if let Some(existing) = domains.get_mut(&variable) {
                existing.retain(|option| options.contains(option));
                if existing.is_empty() {
                    return Err(invalid("contradictory component conditions"));
                }
            } else {
                domains.insert(variable, options);
            }
        }
        let mut clauses = vec![vec![]];
        for (variable, options) in domains {
            clauses = clauses
                .into_iter()
                .flat_map(|clause| {
                    options.iter().map(move |&option| {
                        let mut clause = clause.clone();
                        clause.push(Choice { variable, option });
                        clause
                    })
                })
                .collect();
        }
        Ok(Activation { clauses })
    }

    fn append(
        &self,
        contract: &mut Contract,
        conditions: &[When],
        components: Vec<Component>,
    ) -> Result<(), Refusal> {
        let activation = self.activation(conditions)?;
        for mut component in components {
            // Subrecipes cannot smuggle private variable indexes into the family contract.
            if component.activation != Activation::always() {
                return Err(invalid("subrecipe supplied a private activation"));
            }
            component.activation = activation.clone();
            contract.components.push(component);
        }
        Ok(())
    }
}

impl Recipe for GlmRecipe<'_> {
    fn prepare(&self, request: &Request) -> Result<Contract, Refusal> {
        if !(1..=2).contains(&self.gpus)
            || self.attention.is_empty()
            || self.attention.len() > self.cfg.layers
            || self.context == 0
            || request.context == 0
            || request.context > self.context
            || self.maximum_pages == 0
            || self.maximum_pages == u64::MAX
            || self.page_tokens == 0
            || self.page_tokens % PAGE_ROWS != 0
            || self.prefill.is_empty()
            || self.lanes.is_empty()
            || self.embedding.is_empty()
        {
            return Err(invalid("invalid extent or empty domain"));
        }
        self.page_tokens
            .checked_mul(self.maximum_pages)
            .ok_or(Refusal::Overflow("GLM pool extent"))?;
        for (i, shape) in self.prefill.iter().enumerate() {
            if !(64..=4096).contains(&shape.rows)
                || shape.rows % PAGE_ROWS != 0
                || !(1..=4).contains(&shape.waves)
                || self.prefill[..i].contains(shape)
            {
                return Err(invalid("invalid or duplicate prefill shape"));
            }
        }
        if self
            .lanes
            .iter()
            .enumerate()
            .any(|(i, lane)| *lane == 0 || self.lanes[..i].contains(lane))
            || self.embedding.iter().enumerate().any(|(i, home)| {
                self.embedding[..i].contains(home)
                    || matches!(home, EmbeddingHome::Device(gpu) if usize::from(*gpu) >= self.gpus)
            })
        {
            return Err(invalid("invalid or duplicate lane/embedding value"));
        }
        let partitioned = self
            .attention
            .iter()
            .any(|mode| *mode == LayerMode::ContextSplit);
        if self.attention.iter().any(|mode| match mode {
            LayerMode::HeadSplit | LayerMode::ContextSplit => self.gpus != 2,
            LayerMode::Whole { gpu, .. } => usize::from(*gpu) >= self.gpus,
        }) || partitioned && self.page_tokens % (2 * PAGE_ROWS) != 0
        {
            return Err(invalid("attention ownership or context page quantum"));
        }
        let mut contract = Contract {
            page_rows: self.page_tokens,
            maximum_pages: self.maximum_pages,
            gpus: self.gpus,
            variables: self
                .attention
                .iter()
                .enumerate()
                .map(|(layer, &mode)| Variable::Attention {
                    layer,
                    values: vec![mode],
                })
                .collect(),
            requires: vec![],
            components: vec![],
        };
        let prefill = contract.variables.len();
        contract
            .variables
            .push(Variable::Prefill(self.prefill.clone()));
        let lanes = contract.variables.len();
        contract.variables.push(Variable::Lanes(self.lanes.clone()));
        let embedding = contract.variables.len();
        contract
            .variables
            .push(Variable::Embedding(self.embedding.clone()));
        let onboard = self
            .experts
            .as_ref()
            .map(|profile| -> Result<_, Refusal> {
                if profile.layers.is_empty()
                    || profile.counts.is_empty()
                    || profile.layers
                        != (self.cfg.first_moe_layer..self.attention.len()).collect::<Vec<_>>()
                    || profile.layers.iter().collect::<BTreeSet<_>>().len() != profile.layers.len()
                    || profile.destination.is_empty()
                    || profile
                        .destination
                        .iter()
                        .any(|&gpu| usize::from(gpu) >= self.gpus)
                    || profile
                        .destination
                        .windows(2)
                        .any(|pair| pair[0] >= pair[1])
                    || profile
                        .counts
                        .iter()
                        .any(|&count| count > profile.layers.len())
                    || profile.counts.windows(2).any(|pair| pair[0] >= pair[1])
                    || profile.rank_sets.iter().enumerate().any(|(i, ranks)| {
                        ranks.is_empty()
                            || ranks.windows(2).any(|pair| pair[0] >= pair[1])
                            || profile.rank_sets[..i].contains(ranks)
                    })
                    || profile.rank_sets.is_empty() && profile.counts != [profile.layers.len()]
                {
                    return Err(invalid("invalid canonical expert profile"));
                }
                let variable = contract.variables.len();
                contract.variables.push(Variable::Onboard {
                    layers: profile.layers.clone(),
                    destination: profile.destination.clone(),
                    counts: profile.counts.clone(),
                });
                Ok((variable, profile))
            })
            .transpose()?;
        let remote = onboard.and_then(|(_, profile)| {
            (!profile.rank_sets.is_empty()).then(|| {
                let variable = contract.variables.len();
                contract.variables.push(Variable::ExpertHome {
                    layers: profile.layers.clone(),
                    rank_sets: profile.rank_sets.clone(),
                });
                variable
            })
        });
        let domains = Domains {
            prefill: (prefill, &self.prefill),
            lanes: (lanes, &self.lanes),
            embedding: (embedding, &self.embedding),
            onboard,
            remote,
        };
        for case in &self.targets {
            let target = &case.value;
            let expected: Vec<_> = self
                .attention
                .iter()
                .map(|mode| match mode {
                    LayerMode::Whole { gpu, .. } => *gpu == target.device,
                    _ => true,
                })
                .collect();
            let whole = self
                .attention
                .iter()
                .any(|mode| matches!(mode, LayerMode::Whole { .. }));
            if usize::from(target.device) >= self.gpus
                || target.lead != (target.device == 0)
                || target.page_tokens != self.page_tokens
                || target.maximum_pages != self.maximum_pages
                || target.context != self.context
                || target.partitioned != partitioned
                || target.owned_layers != expected
                || target.split_heads != (self.gpus == 2 && !whole)
                || target.full_prefill_logits != (request.operation == Operation::Scoring)
                || target.prefill.is_some_and(|(shape, _)| {
                    !case
                        .when
                        .iter()
                        .any(|when| matches!(when, When::Prefill(value) if *value == shape))
                })
            {
                return Err(invalid(
                    "target geometry differs from declared execution domains",
                ));
            }
            domains.append(&mut contract, &case.when, target.components(self.cfg)?)?;
        }
        for case in &self.hooks {
            if usize::from(case.value.device) >= self.gpus
                || case.value.maximum_pages != self.maximum_pages
            {
                return Err(invalid("hook geometry differs from contract"));
            }
            domains.append(&mut contract, &case.when, case.value.components()?)?;
        }
        for case in &self.exchanges {
            if self.gpus != 2 || case.value.maximum_pages != self.maximum_pages
                || case.value.context.is_some() != partitioned
                || case.value.context.is_some_and(|context| context.extent != self.context
                    || !case.when.iter().any(|when| matches!(when, When::Prefill(value) if *value == context.prefill))) {
                return Err(invalid("exchange geometry differs from contract"));
            }
            domains.append(&mut contract, &case.when, case.value.components(self.cfg)?)?;
        }
        for case in &self.weights {
            if case.value.ranks != self.gpus {
                return Err(invalid("weight rank count differs from contract"));
            }
            domains.append(
                &mut contract,
                &case.when,
                case.value.components(self.maximum_pages)?,
            )?;
        }
        for case in &self.owners {
            domains.append(&mut contract, &case.when, case.value.clone())?;
        }
        Ok(contract)
    }
}

#[cfg(test)]
mod tests;
