//! Pure, phase-aware allocation planning. Recipes describe one legal candidate;
//! policy, exact pins and allocation reconciliation belong here, not in families.
use super::LayerMode;
use cuteafd_core::memory_layout::{Basis, Category, Item};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Phase {
    Load,
    PostTargetKv,
    DraftLoad,
    Warmup,
    Ready,
    Growth,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Lifetime {
    pub first: Phase,
    pub last: Phase,
}
impl Lifetime {
    pub const RESIDENT: Self = Self {
        first: Phase::Load,
        last: Phase::Growth,
    };
    pub const TARGET: Self = Self {
        first: Phase::PostTargetKv,
        last: Phase::Growth,
    };
    pub fn contains(self, phase: Phase) -> bool {
        self.first <= phase && phase <= self.last
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Device {
    Coordinator(u8),
    Spark(u8),
    Host,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Domain {
    Device,
    Host,
    Pinned,
    Managed,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Location {
    pub device: Device,
    pub domain: Domain,
}
impl Location {
    pub fn coordinator(index: u8) -> Self {
        Self {
            device: Device::Coordinator(index),
            domain: Domain::Device,
        }
    }
    pub const HOST: Self = Self {
        device: Device::Host,
        domain: Domain::Host,
    };
    pub const PINNED: Self = Self {
        device: Device::Host,
        domain: Domain::Pinned,
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Operation {
    Serving,
    Scoring,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Pin<T> {
    Auto,
    Exact(T),
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrefillShape {
    pub rows: u64,
    pub waves: u64,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Request {
    pub context: u64,
    pub pool: Pin<u64>,
    pub onboard: Pin<u64>,
    pub prefill: Pin<PrefillShape>,
    pub operation: Operation,
}

/// The physical witness stays independent of a logical memory ceiling.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Baseline {
    Forecast {
        occupied: u64,
        source: String,
        assumed: bool,
    },
    Sampled {
        free: u64,
        phase: Phase,
        source: String,
    },
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Budget {
    pub id: String,
    /// Domains sharing physical RAM (e.g. GB10 host, pinned and device) appear
    /// in the same budget. A location must belong to exactly one budget.
    pub locations: Vec<Location>,
    pub physical_bytes: u64,
    pub logical_bytes: u64,
    pub baseline: Baseline,
    pub headroom: u64,
}
impl Budget {
    pub fn available(&self) -> Result<u64, Refusal> {
        let occupied = match &self.baseline {
            Baseline::Forecast { occupied, .. } => *occupied,
            Baseline::Sampled { free, .. } => {
                self.physical_bytes.checked_sub(*free).ok_or_else(|| {
                    Refusal::InvalidFact(format!("{} sample exceeds physical capacity", self.id))
                })?
            }
        };
        self.physical_bytes
            .min(self.logical_bytes)
            .checked_sub(occupied)
            .and_then(|n| n.checked_sub(self.headroom))
            .ok_or_else(|| Refusal::Capacity {
                budget: self.id.clone(),
                phase: Phase::Load,
                required: occupied.saturating_add(self.headroom),
                available: self.physical_bytes.min(self.logical_bytes),
                owners: Vec::new(),
            })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct View {
    pub backing: String,
    pub offset: u64,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Allocation {
    pub id: String,
    pub location: Location,
    pub bytes: u64,
    pub alignment: u64,
    /// Raw allocations are byte arrays; tensor views may give a native shape.
    pub shape: Vec<u64>,
    pub dtype: String,
    pub lifetime: Lifetime,
    pub slot: Option<u32>,
    pub view: Option<View>,
    pub category: Category,
    pub group: String,
}
impl Allocation {
    pub fn buffer(
        id: impl Into<String>,
        location: Location,
        bytes: u64,
        lifetime: Lifetime,
        category: Category,
        group: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            location,
            bytes,
            alignment: 1,
            shape: vec![bytes],
            dtype: "bytes".into(),
            lifetime,
            slot: None,
            view: None,
            category,
            group: group.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Allowance {
    pub location: Location,
    pub bytes: u64,
    pub lifetime: Lifetime,
    pub source: String,
    pub assumed: bool,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Decisions {
    pub attention: Vec<LayerMode>,
    pub onboard_layers: Vec<Vec<usize>>,
    pub prefill: PrefillShape,
}
impl Decisions {
    fn onboard(&self) -> u64 {
        self.onboard_layers
            .iter()
            .flatten()
            .collect::<BTreeSet<_>>()
            .len() as u64
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Facts {
    pub allocations: Vec<Allocation>,
    /// Reserved by a later owner port; never evidence that it was allocated.
    pub deferred: Vec<Allocation>,
    pub allowances: Vec<Allowance>,
}

/// Implementations supply exact costs at the selected mode, rows and pool.
/// They must not change decisions or apply private fit/fallback policy.
pub trait Recipe {
    fn page_rows(&self) -> u64;
    fn maximum_pool(&self) -> u64;
    fn candidates(&self) -> Vec<Decisions>;
    fn evaluate(
        &self,
        request: &Request,
        decisions: &Decisions,
        pool: u64,
    ) -> Result<Facts, Refusal>;
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhaseBudget {
    pub budget: String,
    pub phase: Phase,
    pub owned: u64,
    pub opaque: u64,
    pub available: u64,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedPlan {
    pub request: Request,
    pub decisions: Decisions,
    pub pool_tokens: u64,
    pub budgets: Vec<Budget>,
    pub facts: Facts,
    pub phases: Vec<PhaseBudget>,
}
impl ResolvedPlan {
    pub fn items(&self, location: Location) -> Result<Vec<Item>, Refusal> {
        let mut groups = BTreeMap::new();
        for allocation in &self.facts.allocations {
            if allocation.location != location
                || allocation.view.is_some()
                || !allocation.lifetime.contains(Phase::Ready)
            {
                continue;
            }
            let total = groups
                .entry((allocation.group.clone(), allocation.category))
                .or_insert(0u64);
            *total = total
                .checked_add(allocation.bytes)
                .ok_or(Refusal::Overflow("layout items"))?;
        }
        Ok(groups
            .into_iter()
            .map(|((group, category), bytes)| Item::new(category, group, "", bytes, Basis::Exact))
            .collect())
    }
    pub fn remaining(&self, budget: &str, phase: Phase) -> Option<u64> {
        self.phases
            .iter()
            .find(|row| row.budget == budget && row.phase == phase)
            .map(|row| row.available - row.owned - row.opaque)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum Refusal {
    #[error("invalid planner fact: {0}")]
    InvalidFact(String),
    #[error("invalid exact pin: {0}")]
    InvalidPin(&'static str),
    #[error("pool pin {tokens} is not aligned to {page_rows} rows")]
    MisalignedPool { tokens: u64, page_rows: u64 },
    #[error("pool {tokens} cannot hold selected context {context}")]
    PoolBelowContext { tokens: u64, context: u64 },
    #[error("no supported candidate satisfies the exact pins")]
    PinConflict,
    #[error("no supported candidate")]
    Unsupported,
    #[error("{budget} at {phase:?} needs {required} bytes, has {available}; owners {owners:?}")]
    Capacity {
        budget: String,
        phase: Phase,
        required: u64,
        available: u64,
        owners: Vec<String>,
    },
    #[error("planner arithmetic overflow: {0}")]
    Overflow(&'static str),
    #[error("allocation reconciliation: {0:?}")]
    Reconciliation(Vec<Mismatch>),
}

const PHASES: [Phase; 6] = [
    Phase::Load,
    Phase::PostTargetKv,
    Phase::DraftLoad,
    Phase::Warmup,
    Phase::Ready,
    Phase::Growth,
];

fn validate(facts: &Facts, budgets: &[Budget]) -> Result<(), Refusal> {
    let mut locations = BTreeSet::new();
    let mut names = BTreeSet::new();
    for budget in budgets {
        if budget.id.is_empty()
            || budget.locations.is_empty()
            || !names.insert(&budget.id)
            || match &budget.baseline {
                Baseline::Forecast { source, .. } => source.is_empty(),
                Baseline::Sampled { phase, source, .. } => {
                    *phase != Phase::Load || source.is_empty()
                }
            }
        {
            return Err(Refusal::InvalidFact(
                "empty or duplicate memory budget".into(),
            ));
        }
        budget.available()?;
        for location in &budget.locations {
            if !locations.insert(*location) {
                return Err(Refusal::InvalidFact(
                    "location charged to multiple budgets".into(),
                ));
            }
        }
    }
    let mut ids = BTreeMap::new();
    for allocation in facts.allocations.iter().chain(&facts.deferred) {
        if allocation.id.is_empty()
            || ids.insert(allocation.id.clone(), allocation).is_some()
            || allocation.bytes == 0
            || allocation.alignment == 0
            || !allocation.alignment.is_power_of_two()
            || allocation.lifetime.first > allocation.lifetime.last
            || !locations.contains(&allocation.location)
        {
            return Err(Refusal::InvalidFact(format!(
                "invalid allocation {}",
                allocation.id
            )));
        }
    }
    for allocation in facts.allocations.iter().chain(&facts.deferred) {
        if let Some(view) = &allocation.view {
            let backing = ids
                .get(&view.backing)
                .ok_or_else(|| Refusal::InvalidFact(format!("missing backing {}", view.backing)))?;
            if backing.view.is_some()
                || backing.location != allocation.location
                || view
                    .offset
                    .checked_add(allocation.bytes)
                    .is_none_or(|end| end > backing.bytes)
                || backing.lifetime.first > allocation.lifetime.first
                || backing.lifetime.last < allocation.lifetime.last
            {
                return Err(Refusal::InvalidFact(format!(
                    "invalid backing view {}",
                    allocation.id
                )));
            }
        }
    }
    for allowance in &facts.allowances {
        if !locations.contains(&allowance.location)
            || allowance.source.is_empty()
            || allowance.lifetime.first > allowance.lifetime.last
        {
            return Err(Refusal::InvalidFact("invalid opaque allowance".into()));
        }
    }
    Ok(())
}

pub fn evaluate(facts: &Facts, budgets: &[Budget]) -> Result<Vec<PhaseBudget>, Refusal> {
    validate(facts, budgets)?;
    let mut rows = Vec::new();
    for budget in budgets {
        let available = budget.available()?;
        for phase in PHASES {
            let owners: Vec<_> = facts
                .allocations
                .iter()
                .chain(&facts.deferred)
                .filter(|a| {
                    a.view.is_none()
                        && a.lifetime.contains(phase)
                        && budget.locations.contains(&a.location)
                })
                .collect();
            let owned = owners.iter().try_fold(0u64, |n, a| {
                n.checked_add(a.bytes)
                    .ok_or(Refusal::Overflow("phase owners"))
            })?;
            let opaque = facts
                .allowances
                .iter()
                .filter(|a| a.lifetime.contains(phase) && budget.locations.contains(&a.location))
                .try_fold(0u64, |n, a| {
                    n.checked_add(a.bytes)
                        .ok_or(Refusal::Overflow("phase allowances"))
                })?;
            let required = owned
                .checked_add(opaque)
                .ok_or(Refusal::Overflow("phase peak"))?;
            if required > available {
                return Err(Refusal::Capacity {
                    budget: budget.id.clone(),
                    phase,
                    required,
                    available,
                    owners: owners.iter().map(|a| a.id.clone()).collect(),
                });
            }
            rows.push(PhaseBudget {
                budget: budget.id.clone(),
                phase,
                owned,
                opaque,
                available,
            });
        }
    }
    Ok(rows)
}

pub fn solve(
    request: Request,
    budgets: Vec<Budget>,
    recipe: &impl Recipe,
) -> Result<ResolvedPlan, Refusal> {
    let page = recipe.page_rows();
    if page == 0 || request.context == 0 {
        return Err(Refusal::InvalidFact("zero page/context".into()));
    }
    if let Pin::Exact(tokens) = request.pool {
        if tokens % page != 0 {
            return Err(Refusal::MisalignedPool {
                tokens,
                page_rows: page,
            });
        }
        if tokens < request.context {
            return Err(Refusal::PoolBelowContext {
                tokens,
                context: request.context,
            });
        }
    }
    let minimum = request.context.div_ceil(page);
    let ceiling = recipe.maximum_pool() / page;
    if matches!(request.prefill, Pin::Exact(shape) if shape.rows == 0 || shape.waves == 0) {
        return Err(Refusal::InvalidPin("zero prefill shape"));
    }
    let small = budgets
        .iter()
        .filter(|b| {
            b.locations
                .iter()
                .any(|l| matches!(l.device, Device::Coordinator(_)))
        })
        .any(|b| b.physical_bytes <= 34 << 30);
    let target = (if small { 1u64 << 20 } else { 2u64 << 20 })
        .max(request.context)
        .div_ceil(page);
    let wanted = match request.pool {
        Pin::Auto => target.min(ceiling),
        Pin::Exact(tokens) => tokens / page,
    };
    let onboarding_target = match request.pool {
        Pin::Auto => target,
        Pin::Exact(_) => wanted,
    };
    if wanted < minimum || wanted > ceiling {
        return Err(Refusal::InvalidPin("pool exceeds compiled extent"));
    }
    let mut best: Option<ResolvedPlan> = None;
    let mut last = Refusal::Unsupported;
    let mut matched = false;
    for decisions in recipe.candidates() {
        if matches!(request.prefill, Pin::Exact(shape) if shape != decisions.prefill)
            || matches!(request.onboard, Pin::Exact(n) if n != decisions.onboard())
        {
            continue;
        }
        matched = true;
        if decisions.prefill.rows == 0 || decisions.prefill.waves == 0 {
            return Err(Refusal::InvalidFact("zero prefill shape".into()));
        }
        if decisions.onboard_layers.iter().any(|layers| {
            let mut onboarded = BTreeSet::new();
            layers.iter().any(|layer| !onboarded.insert(*layer))
        }) {
            return Err(Refusal::InvalidFact("duplicate onboarded layer".into()));
        }
        if matches!(request.onboard, Pin::Auto)
            && decisions.onboard() != 0
            && wanted < onboarding_target
        {
            continue;
        }
        let lower = if matches!(request.pool, Pin::Exact(_)) {
            wanted
        } else {
            minimum
        };
        for pages in (lower..=wanted).rev() {
            if matches!(request.onboard, Pin::Auto)
                && decisions.onboard() != 0
                && pages < onboarding_target
            {
                break;
            }
            let pool = pages
                .checked_mul(page)
                .ok_or(Refusal::Overflow("pool tokens"))?;
            let facts = match recipe.evaluate(&request, &decisions, pool) {
                Ok(facts) => facts,
                Err(error @ Refusal::Capacity { .. }) => {
                    last = error;
                    continue;
                }
                Err(Refusal::Unsupported) => continue,
                Err(error) => return Err(error),
            };
            match evaluate(&facts, &budgets) {
                Ok(phases) => {
                    let candidate = ResolvedPlan {
                        request: request.clone(),
                        decisions: decisions.clone(),
                        pool_tokens: pool,
                        budgets: budgets.clone(),
                        facts,
                        phases,
                    };
                    let score = |p: &ResolvedPlan| {
                        (
                            p.pool_tokens,
                            p.decisions.onboard(),
                            p.phases
                                .iter()
                                .map(|row| row.available - row.owned - row.opaque)
                                .min()
                                .unwrap_or(0),
                        )
                    };
                    if best
                        .as_ref()
                        .is_none_or(|old| score(&candidate) > score(old))
                    {
                        best = Some(candidate);
                    }
                    break;
                }
                Err(error @ Refusal::Capacity { .. }) => last = error,
                Err(error) => return Err(error),
            }
        }
    }
    best.ok_or(if matched { last } else { Refusal::PinConflict })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Mismatch {
    Missing(String),
    Extra(String),
    Duplicate(String),
    Contract(String),
    Address(String),
    Lifetime(String),
}
#[derive(Debug, Clone)]
pub struct Receipt {
    pub allocation: Allocation,
    pub address: u64,
}
#[derive(Debug, Clone)]
pub struct Observation {
    pub receipt: Receipt,
    pub created: Phase,
    pub freed: Option<Phase>,
}

/// Startup observations carry actual creation/free phases, not merely the
/// planned lifetime copied onto a live owner's receipt.
pub fn reconcile_history(
    plan: &ResolvedPlan,
    phase: Phase,
    observations: &[Observation],
) -> Result<(), Refusal> {
    validate(&plan.facts, &plan.budgets)?;
    let mut seen = BTreeSet::new();
    let mut errors = Vec::new();
    for observation in observations {
        let id = &observation.receipt.allocation.id;
        if !seen.insert(id.as_str()) {
            errors.push(Mismatch::Duplicate(id.clone()));
        }
        match plan.facts.allocations.iter().find(|a| &a.id == id) {
            None => errors.push(Mismatch::Extra(id.clone())),
            Some(spec) => {
                if *spec != observation.receipt.allocation {
                    errors.push(Mismatch::Contract(id.clone()));
                }
                if observation.created != spec.lifetime.first
                    || observation.created > phase
                    || observation.freed.is_some_and(|freed| {
                        freed < observation.created || freed > phase || freed != spec.lifetime.last
                    })
                    || (spec.lifetime.last < phase && observation.freed.is_none())
                {
                    errors.push(Mismatch::Lifetime(id.clone()));
                }
            }
        }
    }
    for spec in &plan.facts.allocations {
        if spec.lifetime.first <= phase && !seen.contains(spec.id.as_str()) {
            errors.push(Mismatch::Missing(spec.id.clone()));
        }
    }
    if !errors.is_empty() {
        return Err(Refusal::Reconciliation(errors));
    }
    for at in PHASES.into_iter().take_while(|at| *at <= phase) {
        let receipts = observations
            .iter()
            .filter(|o| o.created <= at && o.freed.is_none_or(|freed| at <= freed))
            .map(|o| o.receipt.clone())
            .collect::<Vec<_>>();
        reconcile(plan, at, &receipts)?;
    }
    Ok(())
}

/// Observed device use since the before-weights sample must stay within owned
/// allocations plus the explicit opaque bound, never consume reserved headroom.
pub fn reconcile_usage(
    plan: &ResolvedPlan,
    budget: &str,
    phase: Phase,
    consumed: u64,
) -> Result<(), Refusal> {
    let row = plan
        .phases
        .iter()
        .find(|row| row.budget == budget && row.phase == phase)
        .ok_or_else(|| Refusal::InvalidFact("missing usage budget/phase".into()))?;
    let bound = row
        .owned
        .checked_add(row.opaque)
        .ok_or(Refusal::Overflow("usage bound"))?;
    if consumed > bound {
        return Err(Refusal::Capacity {
            budget: budget.into(),
            phase,
            required: consumed,
            available: bound,
            owners: vec!["observed owned plus opaque usage".into()],
        });
    }
    Ok(())
}

/// Receipts describe actual owners. A view address must equal backing+offset;
/// overlapping physical owners and overlapping independent slot views fail.
pub fn reconcile(plan: &ResolvedPlan, phase: Phase, receipts: &[Receipt]) -> Result<(), Refusal> {
    validate(&plan.facts, &plan.budgets)?;
    let expected: BTreeMap<_, _> = plan
        .facts
        .allocations
        .iter()
        .filter(|a| a.lifetime.contains(phase))
        .map(|a| (a.id.as_str(), a))
        .collect();
    let mut actual = BTreeMap::new();
    let mut errors = Vec::new();
    if phase >= Phase::Ready {
        errors.extend(
            plan.facts
                .deferred
                .iter()
                .map(|a| Mismatch::Missing(a.id.clone())),
        );
    }
    for receipt in receipts {
        let id = receipt.allocation.id.as_str();
        if actual.insert(id, receipt).is_some() {
            errors.push(Mismatch::Duplicate(id.into()));
        }
        match expected.get(id) {
            None => errors.push(Mismatch::Extra(id.into())),
            Some(spec) if **spec != receipt.allocation => {
                errors.push(Mismatch::Contract(id.into()))
            }
            _ => (),
        }
        if receipt.address == 0
            || receipt
                .address
                .checked_add(receipt.allocation.bytes)
                .is_none()
            || receipt.allocation.alignment == 0
            || receipt.address % receipt.allocation.alignment.max(1) != 0
        {
            errors.push(Mismatch::Address(id.into()));
        }
    }
    for id in expected.keys() {
        if !actual.contains_key(id) {
            errors.push(Mismatch::Missing((*id).into()));
        }
    }
    for receipt in actual.values() {
        let a = &receipt.allocation;
        if let Some(view) = &a.view {
            if actual.get(view.backing.as_str()).is_none_or(|backing| {
                backing.address.checked_add(view.offset) != Some(receipt.address)
            }) {
                errors.push(Mismatch::Address(a.id.clone()));
            }
        }
    }
    let values: Vec<_> = actual.values().collect();
    for (i, a) in values.iter().enumerate() {
        for b in &values[i + 1..] {
            let (left, right) = (&a.allocation, &b.allocation);
            if left.location != right.location {
                continue;
            }
            let overlap = a
                .address
                .checked_add(left.bytes)
                .zip(b.address.checked_add(right.bytes))
                .is_none_or(|(end_a, end_b)| a.address < end_b && b.address < end_a);
            let distinct_backings = left.view.is_none() && right.view.is_none();
            let distinct_slots =
                left.view
                    .as_ref()
                    .zip(right.view.as_ref())
                    .is_some_and(|(a, b)| {
                        a.backing == b.backing
                            && left.slot != right.slot
                            && left.slot.is_some()
                            && right.slot.is_some()
                    });
            if overlap && (distinct_backings || distinct_slots) {
                errors.push(Mismatch::Address(format!("{} / {}", left.id, right.id)));
            }
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(Refusal::Reconciliation(errors))
    }
}

#[cfg(test)]
mod tests;
