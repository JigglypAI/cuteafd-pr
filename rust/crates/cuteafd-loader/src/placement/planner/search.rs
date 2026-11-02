//! Propagate finite domains, price surviving assignments, materialize one winner.
use super::*;

#[derive(Debug, Clone, Copy, Default)]
struct Cost {
    owned: u64,
    opaque: u64,
    owned_slope: u64,
    opaque_slope: u64,
}
impl Cost {
    fn add(&mut self, other: Self) -> Result<(), Refusal> {
        self.owned = self
            .owned
            .checked_add(other.owned)
            .ok_or(Refusal::Overflow("owned cost"))?;
        self.opaque = self
            .opaque
            .checked_add(other.opaque)
            .ok_or(Refusal::Overflow("opaque cost"))?;
        self.owned_slope = self
            .owned_slope
            .checked_add(other.owned_slope)
            .ok_or(Refusal::Overflow("owned slope"))?;
        self.opaque_slope = self
            .opaque_slope
            .checked_add(other.opaque_slope)
            .ok_or(Refusal::Overflow("opaque slope"))?;
        Ok(())
    }
    fn required(self) -> Result<u64, Refusal> {
        self.owned
            .checked_add(self.opaque)
            .ok_or(Refusal::Overflow("phase cost"))
    }
    fn slope(self) -> Result<u64, Refusal> {
        self.owned_slope
            .checked_add(self.opaque_slope)
            .ok_or(Refusal::Overflow("phase slope"))
    }
    fn at(self, delta: u64) -> Result<Self, Refusal> {
        Ok(Self {
            owned: self
                .owned_slope
                .checked_mul(delta)
                .and_then(|n| self.owned.checked_add(n))
                .ok_or(Refusal::Overflow("owned curve"))?,
            opaque: self
                .opaque_slope
                .checked_mul(delta)
                .and_then(|n| self.opaque.checked_add(n))
                .ok_or(Refusal::Overflow("opaque curve"))?,
            ..self
        })
    }
}
struct Group {
    activation: Activation,
    /// [piece][budget x phase], precomputed once, without allocation metadata.
    costs: Vec<Vec<Cost>>,
}
pub(super) struct Compiled {
    starts: Vec<u64>,
    groups: Vec<Group>,
    rows: usize,
}
impl Compiled {
    pub(super) fn new(contract: &Contract, budgets: &[Budget]) -> Result<Self, Refusal> {
        let mut starts = BTreeSet::from([1, contract.maximum_pages + 1]);
        for component in &contract.components {
            for piece in &component.bytes.0 {
                starts.insert(piece.first);
            }
        }
        let starts: Vec<_> = starts.into_iter().collect();
        let rows = budgets.len() * PHASES.len();
        let mut groups = BTreeMap::<Activation, Vec<Vec<Cost>>>::new();
        for component in &contract.components {
            let Some((location, lifetime, opaque)) = component.charge() else {
                continue;
            };
            let budget = budgets
                .iter()
                .position(|b| b.locations.contains(&location))
                .ok_or_else(|| Refusal::InvalidFact("component has no physical budget".into()))?;
            let costs = groups
                .entry(component.activation.clone())
                .or_insert_with(|| vec![vec![Cost::default(); rows]; starts.len() - 1]);
            for (index, &first) in starts[..starts.len() - 1].iter().enumerate() {
                let piece = component.bytes.piece(first)?;
                let bytes = component.bytes.at(first)?;
                for (phase, at) in PHASES.iter().enumerate() {
                    if !lifetime.contains(*at) {
                        continue;
                    }
                    let cost = if opaque {
                        Cost {
                            opaque: bytes,
                            opaque_slope: piece.bytes_per_page,
                            ..Cost::default()
                        }
                    } else {
                        Cost {
                            owned: bytes,
                            owned_slope: piece.bytes_per_page,
                            ..Cost::default()
                        }
                    };
                    costs[index][budget * PHASES.len() + phase].add(cost)?;
                }
            }
        }
        Ok(Self {
            starts,
            rows,
            groups: groups
                .into_iter()
                .map(|(activation, costs)| Group { activation, costs })
                .collect(),
        })
    }
    fn costs(&self, selected: &[usize], piece: usize) -> Result<Vec<Cost>, Refusal> {
        let mut costs = vec![Cost::default(); self.rows];
        for group in &self.groups {
            if group.activation.active(selected) {
                for (sum, &cost) in costs.iter_mut().zip(&group.costs[piece]) {
                    sum.add(cost)?;
                }
            }
        }
        Ok(costs)
    }
    fn piece(&self, pages: u64) -> usize {
        self.starts.partition_point(|&start| start <= pages) - 1
    }
    pub(super) fn phases(
        &self,
        budgets: &[Budget],
        selected: &[usize],
        pages: u64,
    ) -> Result<Vec<PhaseBudget>, Refusal> {
        let piece = self.piece(pages);
        self.costs(selected, piece)?
            .into_iter()
            .enumerate()
            .map(|(row, cost)| {
                let cost = cost.at(pages - self.starts[piece])?;
                let budget = &budgets[row / PHASES.len()];
                Ok(PhaseBudget {
                    budget: budget.id.clone(),
                    phase: PHASES[row % PHASES.len()],
                    owned: cost.owned,
                    opaque: cost.opaque,
                    available: budget.available()?,
                })
            })
            .collect()
    }
    fn limit(
        &self,
        budgets: &[Budget],
        selected: &[usize],
        lower: u64,
        upper: u64,
    ) -> Result<Option<u64>, Refusal> {
        let mut answer = None;
        for piece in 0..self.starts.len() - 1 {
            let first = self.starts[piece].max(lower);
            let mut last = (self.starts[piece + 1] - 1).min(upper);
            if first > last {
                continue;
            }
            for (row, cost) in self.costs(selected, piece)?.into_iter().enumerate() {
                let cost = cost.at(first - self.starts[piece])?;
                let available = budgets[row / PHASES.len()].available()?;
                let required = cost.required()?;
                if required > available {
                    last = 0;
                    break;
                }
                let slope = cost.slope()?;
                if slope != 0 {
                    last = last.min(first.saturating_add((available - required) / slope));
                }
            }
            if first <= last {
                answer = Some(last);
            }
        }
        Ok(answer)
    }
    fn group_minimum(&self, group: &Group, lower: u64, upper: u64) -> Result<Vec<u64>, Refusal> {
        let mut result = vec![u64::MAX; self.rows];
        for piece in 0..self.starts.len() - 1 {
            let first = self.starts[piece].max(lower);
            if first > upper || first >= self.starts[piece + 1] {
                continue;
            }
            for (value, &cost) in result.iter_mut().zip(&group.costs[piece]) {
                *value = (*value).min(cost.at(first - self.starts[piece])?.required()?);
            }
        }
        Ok(result)
    }
    fn lower_bound(
        &self,
        domains: &[Vec<usize>],
        lower: u64,
        upper: u64,
    ) -> Result<Vec<u64>, Refusal> {
        let mut result = vec![0u64; self.rows];
        let mut local = BTreeMap::<usize, Vec<Vec<u64>>>::new();
        for group in &self.groups {
            let minimum = self.group_minimum(group, lower, upper)?;
            if group.activation.guaranteed(domains) {
                add_bytes(&mut result, &minimum)?;
            } else if let Some(variable) = group.activation.local_variable() {
                let values = local
                    .entry(variable)
                    .or_insert_with(|| vec![vec![0; self.rows]; domains[variable].len()]);
                for (&option, costs) in domains[variable].iter().zip(values) {
                    if group
                        .activation
                        .clauses
                        .iter()
                        .any(|c| c[0].option == option)
                    {
                        add_bytes(costs, &minimum)?;
                    }
                }
            }
        }
        for values in local.into_values() {
            let minimum: Vec<_> = (0..self.rows)
                .map(|row| values.iter().map(|v| v[row]).min().unwrap_or(0))
                .collect();
            add_bytes(&mut result, &minimum)?;
        }
        Ok(result)
    }
}
fn add_bytes(destination: &mut [u64], bytes: &[u64]) -> Result<(), Refusal> {
    for (sum, &n) in destination.iter_mut().zip(bytes) {
        *sum = sum
            .checked_add(n)
            .ok_or(Refusal::Overflow("cost lower bound"))?;
    }
    Ok(())
}
fn capacity(rows: &[u64], budgets: &[Budget]) -> Result<(), Refusal> {
    for (row, &required) in rows.iter().enumerate() {
        let budget = &budgets[row / PHASES.len()];
        let available = budget.available()?;
        if required > available {
            return Err(Refusal::Capacity {
                budget: budget.id.clone(),
                phase: PHASES[row % PHASES.len()],
                required,
                available,
                owners: vec![],
            });
        }
    }
    Ok(())
}
fn compatible(contract: &Contract, domains: &[Vec<usize>]) -> bool {
    contract.requires.iter().all(|r| {
        domains[r.when.variable] != [r.when.option]
            || r.any_of
                .iter()
                .any(|c| domains[c.variable].contains(&c.option))
    })
}
fn logical(contract: &Contract, domains: &mut [Vec<usize>], exact: Option<u64>) -> bool {
    loop {
        let before = domains.to_vec();
        for constraint in &contract.requires {
            let available: Vec<_> = constraint
                .any_of
                .iter()
                .filter(|c| domains[c.variable].contains(&c.option))
                .copied()
                .collect();
            if available.is_empty() {
                domains[constraint.when.variable].retain(|&o| o != constraint.when.option);
            } else if domains[constraint.when.variable] == [constraint.when.option]
                && available.len() == 1
            {
                domains[available[0].variable].retain(|&o| o == available[0].option);
            }
        }
        if domains.iter().any(Vec::is_empty) {
            return false;
        }
        if let Some(exact) = exact {
            let ranges: Vec<_> = contract
                .variables
                .iter()
                .zip(domains.iter())
                .map(|(v, d)| {
                    let counts: Vec<_> = d.iter().map(|&o| v.onboard(o)).collect();
                    (*counts.iter().min().unwrap(), *counts.iter().max().unwrap())
                })
                .collect();
            let minimum: u64 = ranges.iter().map(|r| r.0).sum();
            let maximum: u64 = ranges.iter().map(|r| r.1).sum();
            if exact < minimum || exact > maximum {
                return false;
            }
            for (index, domain) in domains.iter_mut().enumerate() {
                domain.retain(|&o| {
                    let count = contract.variables[index].onboard(o);
                    minimum - ranges[index].0 + count <= exact
                        && maximum - ranges[index].1 + count >= exact
                });
            }
        }
        if domains.iter().any(Vec::is_empty) {
            return false;
        }
        if before == domains {
            return compatible(contract, domains);
        }
    }
}
fn propagate(
    contract: &Contract,
    costs: &Compiled,
    budgets: &[Budget],
    domains: &mut [Vec<usize>],
    lower: u64,
    upper: u64,
    exact: Option<u64>,
) -> Result<(), Refusal> {
    loop {
        if !logical(contract, domains, exact) {
            return Err(Refusal::PinConflict);
        }
        capacity(&costs.lower_bound(domains, lower, upper)?, budgets)?;
        let before = domains.to_vec();
        for variable in 0..domains.len() {
            let mut retained = Vec::new();
            for &option in &domains[variable] {
                let mut forced = domains.to_vec();
                forced[variable] = vec![option];
                if logical(contract, &mut forced, exact) {
                    match capacity(&costs.lower_bound(&forced, lower, upper)?, budgets) {
                        Ok(()) => retained.push(option),
                        Err(Refusal::Capacity { .. }) => {}
                        Err(error) => return Err(error),
                    }
                }
            }
            domains[variable] = retained;
            if domains[variable].is_empty() {
                return Err(Refusal::PinConflict);
            }
        }
        if before == domains {
            return Ok(());
        }
    }
}
fn enumerate(
    contract: &Contract,
    domains: &[Vec<usize>],
    index: usize,
    selected: &mut Vec<usize>,
    visit: &mut impl FnMut(&[usize]) -> Result<(), Refusal>,
) -> Result<(), Refusal> {
    if index == domains.len() {
        if contract.requires.iter().all(|r| {
            selected[r.when.variable] != r.when.option
                || r.any_of.iter().any(|c| selected[c.variable] == c.option)
        }) {
            visit(selected)?;
        }
        return Ok(());
    }
    for &option in &domains[index] {
        selected.push(option);
        enumerate(contract, domains, index + 1, selected, visit)?;
        selected.pop();
    }
    Ok(())
}
fn remaining(phases: &[PhaseBudget]) -> u64 {
    phases
        .iter()
        .map(|p| p.available - p.owned - p.opaque)
        .min()
        .unwrap_or(0)
}

/// The one optional-onboarding pass: fit whole units without spending the pool,
/// maximizing admitted unique layers per step, then the tightest-device slack.
fn onboard(
    contract: &Contract,
    costs: &Compiled,
    budgets: &[Budget],
    domains: &[Vec<usize>],
    selected: &mut Vec<usize>,
    pages: u64,
) -> Result<(), Refusal> {
    loop {
        let mut best: Option<(u64, u64, Vec<usize>)> = None;
        for (variable, definition) in contract.variables.iter().enumerate() {
            if !definition.optional_onboard() || definition.onboard(selected[variable]) != 0 {
                continue;
            }
            for &option in &domains[variable] {
                if definition.onboard(option) == 0 {
                    continue;
                }
                let mut trial = selected.clone();
                trial[variable] = option;
                let singleton: Vec<_> = trial.iter().map(|&o| vec![o]).collect();
                if !compatible(contract, &singleton) {
                    continue;
                }
                let phases = costs.phases(budgets, &trial, pages)?;
                if phases.iter().any(|p| {
                    p.owned
                        .checked_add(p.opaque)
                        .is_none_or(|n| n > p.available)
                }) {
                    continue;
                }
                let layers = contract.decisions(&trial)?.onboard();
                let slack = remaining(&phases);
                if best.as_ref().is_none_or(|b| (layers, slack) > (b.0, b.1)) {
                    best = Some((layers, slack, trial));
                }
            }
        }
        let Some((_, _, chosen)) = best else {
            return Ok(());
        };
        *selected = chosen;
    }
}

pub fn solve(
    request: Request,
    budgets: Vec<Budget>,
    recipe: &impl Recipe,
) -> Result<ResolvedPlan, Refusal> {
    let started = std::time::Instant::now();
    let contract = recipe.prepare(&request)?;
    contract.validate()?;
    // Validate physical budgets independently of conditional allocation choices.
    validate(
        &Facts {
            allocations: vec![],
            deferred: vec![],
            allowances: vec![],
        },
        &budgets,
    )?;
    if request.context == 0 {
        return Err(Refusal::InvalidFact("zero context".into()));
    }
    let page = contract.page_rows;
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
    if matches!(request.prefill, Pin::Exact(s) if s.rows == 0 || s.waves == 0) {
        return Err(Refusal::InvalidPin("zero prefill shape"));
    }
    let minimum = request.context.div_ceil(page);
    let small = budgets.iter().any(|b| {
        b.physical_bytes <= 34 << 30
            && b.locations
                .iter()
                .any(|l| matches!(l.device, Device::Coordinator(_)))
    });
    let target = (if small { 1u64 << 20 } else { 2u64 << 20 })
        .max(request.context)
        .div_ceil(page);
    let wanted = match request.pool {
        Pin::Auto => target.min(contract.maximum_pages),
        Pin::Exact(n) => n / page,
    };
    if wanted < minimum || wanted > contract.maximum_pages {
        return Err(Refusal::InvalidPin("pool exceeds compiled extent"));
    }
    let lower = if matches!(request.pool, Pin::Exact(_)) {
        wanted
    } else {
        minimum
    };
    let exact = match request.onboard {
        Pin::Auto => None,
        Pin::Exact(n) => Some(n),
    };
    let mut domains: Vec<Vec<usize>> = contract
        .variables
        .iter()
        .map(|v| (0..v.len()).collect())
        .collect();
    for (variable, domain) in contract.variables.iter().zip(&mut domains) {
        if let (Variable::Prefill(values), Pin::Exact(shape)) = (variable, request.prefill) {
            domain.retain(|&o| values[o] == shape);
        }
    }
    if domains.iter().any(Vec::is_empty) {
        return Err(Refusal::PinConflict);
    }
    let costs = Compiled::new(&contract, &budgets)?;
    propagate(
        &contract,
        &costs,
        &budgets,
        &mut domains,
        lower,
        wanted,
        exact,
    )?;
    let mut base = domains.clone();
    if exact.is_none() {
        for (variable, domain) in contract.variables.iter().zip(&mut base) {
            if variable.optional_onboard() && domain.iter().any(|&o| variable.onboard(o) == 0) {
                domain.retain(|&o| variable.onboard(o) == 0);
            }
        }
    }
    let mut best: Option<(u64, u64, u64, Vec<usize>)> = None;
    let mut assignments = 0u64;
    let mut last = Refusal::Unsupported;
    enumerate(&contract, &base, 0, &mut Vec::new(), &mut |selected| {
        if let Some(exact) = exact {
            if contract
                .variables
                .iter()
                .zip(selected)
                .map(|(v, &o)| v.onboard(o))
                .sum::<u64>()
                != exact
            {
                return Ok(());
            }
        }
        assignments = assignments
            .checked_add(1)
            .ok_or(Refusal::Overflow("assignment count"))?;
        let Some(pages) = costs.limit(&budgets, selected, lower, wanted)? else {
            let rows = costs.phases(&budgets, selected, lower)?;
            let totals: Vec<_> = rows
                .iter()
                .map(|r| {
                    r.owned
                        .checked_add(r.opaque)
                        .ok_or(Refusal::Overflow("refused cost"))
                })
                .collect::<Result<_, _>>()?;
            last = capacity(&totals, &budgets)
                .err()
                .unwrap_or(Refusal::Unsupported);
            return Ok(());
        };
        let mut selected = selected.to_vec();
        let onboarding_target = if matches!(request.pool, Pin::Exact(_)) {
            wanted
        } else {
            target
        };
        if exact.is_none() && pages >= onboarding_target {
            onboard(&contract, &costs, &budgets, &domains, &mut selected, pages)?;
        }
        let decisions = contract.decisions(&selected)?;
        let slack = remaining(&costs.phases(&budgets, &selected, pages)?);
        let score = (pages, decisions.onboard(), slack);
        if best
            .as_ref()
            .is_none_or(|old| score > (old.0, old.1, old.2))
        {
            best = Some((score.0, score.1, score.2, selected));
        }
        Ok(())
    })?;
    tracing::trace!(target: "cuteafd::planner", surviving_assignments = assignments,
        components = contract.components.len(), preparation_search_us = started.elapsed().as_micros() as u64,
        "priced propagated contract assignments");
    let (pages, _, _, selected) = best.ok_or(last)?;
    let decisions = contract.decisions(&selected)?;
    let facts = contract.facts(&selected, pages)?;
    let phases = evaluate(&facts, &budgets)?;
    if phases != costs.phases(&budgets, &selected, pages)? {
        return Err(Refusal::InvalidFact(
            "component cost vectors differ from exact phase evaluator".into(),
        ));
    }
    Ok(ResolvedPlan {
        request,
        decisions,
        pool_tokens: pages
            .checked_mul(page)
            .ok_or(Refusal::Overflow("pool tokens"))?,
        budgets,
        facts,
        phases,
    })
}
