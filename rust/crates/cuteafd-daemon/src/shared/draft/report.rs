//! One projection of the chosen serving admission, never of a launch key.
use anyhow::{ensure, Context, Result};
use cuteafd_core::memory_layout::Category;
use cuteafd_loader::placement::planner::{Allocation, CoordinatorDevices, Location, Phase, ResolvedPlan};
pub(crate) use cuteafd_bench::report::{DrafterKind as Kind, DrafterLayout, DrafterPlacement as Placement, DrafterRank as Rank};

pub(crate) fn admitted(kind: Kind, placement: Placement, lead: u8, devices: &[i32], plan: &ResolvedPlan) -> Result<DrafterLayout> {
    from_plan(kind, placement, lead, devices, plan, |_| false)
}

/// Shared physical projection. `extra` identifies a drafter-exclusive owner
/// that the family admission classifies with another category (e.g. MTP KV).
pub(crate) fn from_plan(kind: Kind, placement: Placement, lead: u8, devices: &[i32], plan: &ResolvedPlan,
    extra: impl Fn(&Allocation) -> bool) -> Result<DrafterLayout> {
    let physical = CoordinatorDevices::new(devices)?;
    let lead_rank = Rank::CudaRank(physical.physical_device(lead)?);
    let split = matches!(placement, Placement::Split | Placement::PairedBlock);
    ensure!(!split || devices.len() == 2, "split drafter requires two admitted coordinator ranks");
    let mut ranks = Vec::new();
    let mut arena_bytes = Vec::new();
    let facts = &plan.facts;
    let live = |lifetime: cuteafd_loader::placement::planner::Lifetime| lifetime.contains(Phase::Ready, facts.phase_order);
    for (logical, &device) in devices.iter().enumerate() {
        if !split && logical != usize::from(lead) { continue; }
        let location = Location::coordinator(logical as u8);
        let owners = facts.allocations.iter().filter(|owner| owner.location == location && owner.view.is_none()
            && live(owner.lifetime) && (owner.category == Category::Drafter || extra(owner))).collect::<Vec<_>>();
        let totals = facts.allocation_granularity.iter().filter(|tail| tail.location == location && live(tail.lifetime))
            .map(|tail| -> Result<u64> {
                let owns = |backing: &&cuteafd_loader::placement::planner::GranularityBacking|
                    owners.iter().any(|owner| owner.id == backing.backing);
                if tail.backings.iter().all(|backing| owns(&backing)) { return Ok(tail.bytes()?); }
                Ok(cuteafd_loader::placement::planner::independent_backing_slack(
                    tail.backings.iter().filter(owns).flat_map(|backing| backing.requests.iter().copied()))?)
            });
        let bytes = owners.iter().map(|owner| Ok(owner.bytes)).chain(totals)
            .try_fold(0u64, |sum, n| sum.checked_add(n?).context("admitted drafter bytes overflow"))?;
        ranks.push(Rank::CudaRank(device));
        arena_bytes.push(bytes);
    }
    Ok(DrafterLayout { kind, placement: Some(placement), ranks, lead: Some(lead_rank), arena_bytes, gpu_uuids: Default::default() })
}

pub(crate) fn publish(layout: DrafterLayout) {
    cuteafd_bench::context::set_drafter(layout);
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use cuteafd_loader::placement::planner::{self as p, Lifetime};

    pub(crate) fn plan(allocations: Vec<Allocation>) -> ResolvedPlan {
        let shape = p::PrefillShape { rows: 64, waves: 1 };
        ResolvedPlan { pool_policy: None, encoder_diagnostics: vec![], pool_tokens: 512, pool_page_rows: 512,
            request: p::Request { context: p::Context::automatic(512), pool: p::Pin::Auto,
                onboard: p::Pin::Auto, prefill: p::Pin::Exact(shape), operation: p::Operation::Serving,
                resident_weight_limit: None, spark_reply_formats: Default::default() },
            decisions: p::Decisions { attention: vec![], onboard_layers: vec![], expert_homes: Default::default(),
                embedding: None, draft_home: None, draft_traffic_bytes_per_cycle: 0, head_home: None,
                intake_home: None, encoder: None, prefill: shape, lanes: 1, spark_reply_formats: None },
            budgets: vec![], phases: vec![], facts: p::Facts { phase_order: p::PhaseOrder::TARGET_THEN_DRAFT,
                allocations, resident_weights: vec![], deferred: vec![], sequential: vec![], reservations: vec![],
                allowances: vec![], driver_page_tables: vec![], allocation_granularity: vec![] } }
    }

    pub(crate) fn owner(rank: u8, bytes: u64) -> Allocation {
        Allocation::buffer(format!("draft.{rank}"), Location::coordinator(rank), bytes, Lifetime::RESIDENT,
            Category::Drafter, "weight arena")
    }

    #[test]
    fn rank_totals_count_retained_roots_and_exclusive_tails_once() -> Result<()> {
        let arena = owner(0, (2 << 20) + 1);
        let mut alias = owner(0, 128);
        alias.id = "alias".into();
        alias.view = Some(p::View { backing: arena.id.clone(), offset: 0 });
        let mut staging = owner(0, 1 << 20);
        staging.id = "load-staging".into();
        staging.lifetime = Lifetime { first: Phase::Load, last: Phase::DraftLoad };
        let mut plan = plan(vec![arena.clone(), alias, staging, owner(1, 256)]);
        plan.facts.allocation_granularity.push(p::AllocationGranularity { id: "arena tail".into(),
            location: arena.location, lifetime: arena.lifetime, evidence: p::GranularityEvidence::Forecast,
            backings: vec![p::GranularityBacking { backing: arena.id, requests: vec![arena.bytes] }] });
        let layout = admitted(Kind::Dflash2, Placement::Split, 1, &[5, 2], &plan)?;
        assert_eq!(layout.ranks, [Rank::CudaRank(5), Rank::CudaRank(2)]);
        assert_eq!(layout.lead, Some(Rank::CudaRank(2)));
        assert_eq!(layout.arena_bytes, [4 << 20, 256]);
        Ok(())
    }
}
