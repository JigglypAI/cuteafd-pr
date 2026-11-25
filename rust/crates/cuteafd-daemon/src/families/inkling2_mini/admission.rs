//! Runtime allocation reconciliation against solved `MiniInputs` admission.
use anyhow::{ensure, Context as _, Result};
use cuteafd_loader::placement::{
    families::inkling2_mini::MiniInputs,
    planner::{ComponentKind, Location, Receipt, Recipe, ResolvedPlan},
};
use std::collections::BTreeMap;

pub(crate) fn reconcile_requests(plan: &ResolvedPlan, inputs: &MiniInputs, receipts: &[Receipt]) -> Result<()> {
    let contract = inputs.prepare(&plan.request)?;
    let selected = |id: &str| {
        id.starts_with("inkling2_mini.cache.")
            || id.starts_with("inkling2_mini.kv.")
            || id.starts_with("inkling2_mini.decode.")
            || id.starts_with("inkling2_mini.unit.")
            || id.starts_with("inkling2_mini.head.")
    };
    let planned: BTreeMap<_, _> = plan
        .facts
        .allocations
        .iter()
        .filter(|a| a.view.is_none() && selected(&a.id))
        .map(|a| (a.id.as_str(), a))
        .collect();
    let actual: BTreeMap<_, _> = receipts
        .iter()
        .filter(|r| r.allocation.view.is_none() && r.allocation.location != Location::HOST)
        .map(|r| {
            let id = r.allocation.id.strip_prefix("retained.").unwrap_or(&r.allocation.id);
            (id, r)
        })
        .collect();
    ensure!(
        actual.len()
            == receipts
                .iter()
                .filter(|r| r.allocation.view.is_none() && r.allocation.location != Location::HOST)
                .count(),
        "Inkling 2 Mini duplicate request backing receipt"
    );
    ensure!(
        actual.len() == planned.len(),
        "Inkling 2 Mini request owner count differs: missing {:?}, extra {:?}",
        planned.keys().filter(|id| !actual.contains_key(*id)).collect::<Vec<_>>(),
        actual.keys().filter(|id| !planned.contains_key(*id)).collect::<Vec<_>>()
    );
    let pages = plan.pool_tokens.div_ceil(plan.pool_page_rows);
    for (id, receipt) in actual {
        let owner = planned
            .get(id)
            .with_context(|| format!("Mini unadmitted request owner {id}"))?;
        let curves: Vec<_> = contract
            .components
            .iter()
            .filter(|c| matches!(&c.kind, ComponentKind::Allocation { template, .. } if template.id == id))
            .map(|c| &c.bytes)
            .collect();
        let curve = curves.first().context("Mini request owner has no component curve")?;
        ensure!(
            curves.iter().all(|other| *other == *curve),
            "Mini request owner has ambiguous curves"
        );
        let piece = curve
            .0
            .iter()
            .find(|p| (p.first..=p.last).contains(&pages))
            .context("Mini request owner curve does not cover admitted pool")?;
        let payload = pages
            .checked_sub(piece.first)
            .and_then(|p| piece.bytes_per_page.checked_mul(p))
            .and_then(|b| piece.bytes_at_first.checked_add(b))
            .context("Mini request payload overflow")?;
        ensure!(
            piece.backings == 1
                && receipt.address != 0
                && receipt.allocation.bytes == payload
                && receipt.allocation.location == owner.location
                && curve.at(pages)? == owner.bytes,
            "Inkling 2 Mini request owner {id} differs from admission: actual {}, requested {payload}, charged {}",
            receipt.allocation.bytes,
            owner.bytes
        );
    }
    Ok(())
}
