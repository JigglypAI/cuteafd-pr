//! Actual-owner receipts for no-drafter target and serving startup, including pinned owners.
use super::*;
use cuteafd_core::memory_layout::Category;
use cuteafd_ffi::memory_ledger::{self, AllocationAudit, AllocationEvent, Space};
use cuteafd_loader::placement::planner::{
    self, Allocation as Spec, Domain, Lifetime, Location, Observation, Phase, Receipt, ResolvedPlan,
};
use std::collections::BTreeMap;

fn receipt(
    id: String,
    buffer: CuteafdDeviceBuffer,
    category: Category,
    group: &str,
    lifetime: Lifetime,
) -> Receipt {
    Receipt {
        allocation: Spec::buffer(
            id,
            Location::coordinator(0),
            buffer.bytes as u64,
            lifetime,
            category,
            group,
        ),
        address: buffer.ptr as u64,
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn reconcile_target(
    plan: &ResolvedPlan,
    library: &NativeLibrary,
    device: Device<'_>,
    stream: *mut std::ffi::c_void,
    audit: &AllocationAudit,
    loaded_events: usize,
    embedding: &TokenEmbedding<'_>,
    weights: &DenseWeights<'_>,
    planes: &[Allocation<'_>],
    ids: &Allocation<'_>,
    logits: &Allocation<'_>,
    head: &Allocation<'_>,
    bank: &GdnBank<'_>,
    kv: &[Option<std::rc::Rc<Allocation<'_>>>],
    verification: &verify::Verification<'_>,
    prefix: Option<&Allocation<'_>>,
) -> Result<Vec<Receipt>> {
    let mut receipts = Vec::new();
    for (layer, weights) in weights.layers.iter().enumerate() {
        for (name, buffer) in weights.allocation_buffers() {
            receipts.push(receipt(
                format!("weights.layer.{layer}.{name}"),
                buffer,
                Category::Weights,
                "native dense weights",
                Lifetime::RESIDENT,
            ));
        }
    }
    for (id, buffer) in [
        ("weights.norm", weights.norm.buffer),
        ("weights.head", weights.head.buffer),
    ] {
        receipts.push(receipt(
            id.into(),
            buffer,
            Category::Weights,
            "native dense weights",
            Lifetime::RESIDENT,
        ));
    }
    receipts.push(receipt(
        "weights.embedding".into(),
        embedding.allocation_buffer(),
        Category::Embedding,
        "token embedding",
        Lifetime::RESIDENT,
    ));
    for (index, owner) in planes.iter().enumerate() {
        receipts.push(receipt(
            format!("planes.{index}"),
            owner.buffer,
            Category::Workspace,
            "native step planes and scratch",
            Lifetime::TARGET,
        ));
    }
    for (id, owner, group) in [
        ("ids", ids, "token ids"),
        ("logits", logits, "logits"),
        ("head", head, "vocabulary head workspace"),
    ] {
        receipts.push(receipt(
            id.into(),
            owner.buffer,
            Category::Workspace,
            group,
            Lifetime::TARGET,
        ));
    }
    for (id, buffer) in bank.allocation_buffers() {
        let (category, group) = match id {
            "gdn.replay" => (Category::Workspace, "GDN replay"),
            "gdn.commit" => (Category::Workspace, "GDN commit table"),
            _ => (Category::Kv, "GDN state"),
        };
        receipts.push(receipt(
            id.into(),
            buffer,
            category,
            group,
            Lifetime::TARGET,
        ));
    }
    for (layer, owner) in kv.iter().enumerate() {
        if let Some(owner) = owner {
            receipts.push(receipt(
                format!("kv.{layer}"),
                owner.buffer,
                Category::Kv,
                "full GQA KV",
                Lifetime::TARGET,
            ));
        }
    }
    for (index, buffer) in verification.allocation_buffers().into_iter().enumerate() {
        receipts.push(receipt(
            format!("verify.{index}"),
            buffer,
            Category::Workspace,
            "block verify marks, logits and taps",
            Lifetime::TARGET,
        ));
    }
    for (id, buffer, lifetime, group) in [
        (
            "upload.sync",
            library.sync_h2d_staging_buffer()?,
            Lifetime::RESIDENT,
            "retained sync-H2D staging",
        ),
        (
            "gdn.commit.upload",
            library.ordered_h2d_staging_buffer(bank.commit_tables.buffer, stream)?,
            Lifetime::TARGET,
            "ordered GDN commit staging",
        ),
    ] {
        receipts.push(Receipt {
            allocation: Spec::buffer(
                id,
                Location::PINNED,
                buffer.bytes as u64,
                lifetime,
                Category::Staging,
                group,
            ),
            address: buffer.ptr as u64,
        });
    }
    for (id, dst, group) in planes
        .iter()
        .enumerate()
        .skip(7)
        .map(|(index, a)| {
            (
                format!("planes.{index}.upload"),
                a.buffer,
                "ordered execution table staging",
            )
        })
        .chain(std::iter::once((
            "ids.upload".into(),
            ids.buffer,
            "ordered token id staging",
        )))
    {
        let buffer = library.ordered_h2d_staging_buffer(dst, stream)?;
        receipts.push(Receipt {
            allocation: Spec::buffer(
                id,
                Location::PINNED,
                buffer.bytes as u64,
                Lifetime::TARGET,
                Category::Staging,
                group,
            ),
            address: buffer.ptr as u64,
        });
    }
    if let Some(prefix) = prefix {
        receipts.push(receipt(
            "prefix".into(),
            prefix.buffer,
            Category::Prefix,
            "exact retained prefix marks",
            Lifetime::TARGET,
        ));
    }
    let mut view_receipts = Vec::new();
    let mut add_view = |id: String,
                        backing: &str,
                        buffer: CuteafdDeviceBuffer,
                        slot,
                        dtype: &str,
                        width|
     -> Result<()> {
        let base = receipts
            .iter()
            .find(|r| r.allocation.id == backing)
            .context("view without physical owner")?
            .address;
        let mut allocation = Spec::buffer(
            id,
            Location::coordinator(0),
            buffer.bytes as u64,
            Lifetime::TARGET,
            Category::Kv,
            "exact state view",
        );
        allocation.view = Some(planner::View {
            backing: backing.into(),
            offset: (buffer.ptr as u64)
                .checked_sub(base)
                .context("view before backing")?,
        });
        allocation.slot = Some(slot);
        allocation.dtype = dtype.into();
        allocation.shape = vec![buffer.bytes as u64 / width];
        view_receipts.push(Receipt {
            allocation,
            address: buffer.ptr as u64,
        });
        Ok(())
    };
    let verify_views = verification.state_views()?;
    for (index, (ordinal, slot, regions)) in bank.state_views().into_iter().enumerate() {
        add_view(
            format!("gdn.conv.{ordinal}.slot.{slot}"),
            "gdn.conv",
            regions[0],
            slot as u32,
            "bf16",
            2,
        )?;
        add_view(
            format!("gdn.state.{ordinal}.slot.{slot}"),
            "gdn.state",
            regions[1],
            slot as u32,
            "f32",
            4,
        )?;
        add_view(
            format!("verify.state.{ordinal}.slot.{slot}.conv"),
            "verify.0",
            verify_views[index * 2],
            slot as u32,
            "bf16",
            2,
        )?;
        add_view(
            format!("verify.state.{ordinal}.slot.{slot}.recurrent"),
            "verify.0",
            verify_views[index * 2 + 1],
            slot as u32,
            "f32",
            4,
        )?;
    }
    if let Some(prefix) = prefix {
        let mark_bytes: usize = bank
            .state_views()
            .iter()
            .filter(|(_, slot, _)| *slot == 0)
            .map(|(_, _, regions)| regions.iter().map(|a| a.bytes).sum::<usize>())
            .sum();
        ensure!(
            mark_bytes > 0 && prefix.buffer.bytes % mark_bytes == 0,
            "prefix owner does not contain whole marks"
        );
        for mark in 0..prefix.buffer.bytes / mark_bytes {
            let buffer = crate::shared::prefix::view(prefix.buffer, mark * mark_bytes, mark_bytes)?;
            add_view(
                format!("prefix.mark.{mark}"),
                "prefix",
                buffer,
                mark as u32,
                "bytes",
                1,
            )?;
        }
    }
    receipts.extend(view_receipts);
    reconcile_startup(plan, library, device, Phase::PostTargetKv, audit, loaded_events,
        audit.events().len(), &receipts)?;
    Ok(receipts)
}

pub(super) struct Startup {
    pub plan: ResolvedPlan,
    pub audit: AllocationAudit,
    pub loaded_events: usize,
    pub target_events: usize,
    pub receipts: Vec<Receipt>,
}

impl Startup {
    pub fn ready(&mut self, library: &NativeLibrary, device: Device<'_>,
        selector: &crate::shared::token_io::TokenSelector<'_>,
        cache: &mut cuteafd_engine::prefix::PrefixCache<crate::shared::prefix::CudaCopyEngine<'_>>) -> Result<()> {
        let lifetime = Lifetime { first: Phase::Warmup, last: Phase::Growth };
        let (gpu, pinned) = selector.allocation_buffers();
        let mut receipts = self.receipts.clone();
        for (id, buffer) in gpu {
            let group = if id.starts_with("selector.sampler.") { "target sampler" } else { "token selector" };
            receipts.push(receipt(id, buffer, Category::Workspace, group, lifetime));
        }
        for (id, buffer) in pinned {
            let (category, group) = if id.starts_with("selector.sampler.") {
                (Category::Workspace, "target sampler")
            } else { (Category::Staging, "token selector") };
            receipts.push(Receipt { allocation: Spec::buffer(id, Location::PINNED, buffer.bytes as u64,
                lifetime, category, group), address: buffer.ptr as u64 });
        }
        if let Some(copy) = cache.host_engine_mut() {
            for (index, buffer) in copy.allocation_buffers() {
                receipts.push(Receipt { allocation: Spec::buffer(format!("host.prefix.{index}"), Location::PINNED,
                    buffer.bytes as u64, lifetime, Category::Prefix, "retained host prefix chunk"),
                    address: buffer.ptr as u64 });
            }
        }
        reconcile_startup(&self.plan, library, device, Phase::Ready, &self.audit,
            self.loaded_events, self.target_events, &receipts)
    }
}

#[allow(clippy::too_many_arguments)]
fn reconcile_startup(plan: &ResolvedPlan, library: &NativeLibrary, device: Device<'_>, phase: Phase,
    audit: &AllocationAudit, loaded_events: usize, target_events: usize, receipts: &[Receipt]) -> Result<()> {
    planner::reconcile(plan, phase, receipts)?;
    let mut live = BTreeMap::new();
    let mut observations = Vec::new();
    for (index, event) in audit.events().into_iter().enumerate() {
        let at = if index < loaded_events {
            Phase::Load
        } else if index < target_events {
            Phase::PostTargetKv
        } else {
            Phase::Warmup
        };
        match event {
            AllocationEvent::Allocated(a) => {
                ensure!(
                    live.insert(a.address, (a, at)).is_none(),
                    "startup allocation reused a live pointer"
                );
            }
            AllocationEvent::Freed(a) => {
                let (allocated, created) = live
                    .remove(&a.address)
                    .context("free without audited allocation")?;
                ensure!(
                    allocated.key.space == Space::Pinned && allocated.key.scope == "embedding",
                    "unplanned startup transient: {allocated:?}"
                );
                observations.push(Observation {
                    receipt: Receipt {
                        allocation: Spec::buffer(
                            "embedding.upload",
                            Location::PINNED,
                            allocated.bytes as u64,
                            Lifetime {
                                first: Phase::Load,
                                last: Phase::Load,
                            },
                            Category::Staging,
                            "embedding upload staging",
                        ),
                        address: allocated.address as u64,
                    },
                    created,
                    freed: Some(at),
                });
            }
        }
    }
    // Match each actual object to an independently recorded FFI owner, including
    // domain, physical device and bytes. Unknown owners cannot cancel totals.
    let census = memory_ledger::allocations()
        .into_iter()
        .map(|a| (a.address, a))
        .collect::<BTreeMap<_, _>>();
    for receipt in receipts {
        if receipt.allocation.view.is_some() {
            observations.push(Observation {
                receipt: receipt.clone(),
                created: Phase::PostTargetKv,
                freed: None,
            });
            continue;
        }
        let (owner, created) = live
            .remove(&(receipt.address as usize))
            .context("receipt absent from startup allocation audit")?;
        let domain = match owner.key.space {
            Space::Device => Domain::Device,
            Space::Pinned => Domain::Pinned,
            Space::Managed => Domain::Managed,
        };
        ensure!(
            domain == receipt.allocation.location.domain
                && owner.bytes as u64 == receipt.allocation.bytes,
            "allocation census differs from object receipt {}",
            receipt.allocation.id
        );
        if domain == Domain::Device {
            ensure!(
                owner.key.device == device.id,
                "dense allocation on wrong physical device"
            );
        }
        ensure!(
            census
                .get(&owner.address)
                .is_some_and(|a| a.key == owner.key && a.bytes == owner.bytes),
            "retained target owner absent from live census"
        );
        observations.push(Observation {
            receipt: receipt.clone(),
            created,
            freed: None,
        });
    }
    ensure!(
        live.is_empty(),
        "dense allocations without receipts: {live:?}"
    );
    planner::reconcile_history(plan, phase, &observations)?;
    let before = match plan.budgets[0].baseline {
        planner::Baseline::Sampled { free, .. } => free,
        _ => anyhow::bail!("live reconciliation requires a before-weights sample"),
    };
    let (free, _) = library.cuda_physical_memory_info()?;
    planner::reconcile_usage(
        plan,
        "coordinator.0",
        phase,
        before.saturating_sub(free as u64),
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::{memory::device::Allocation, prefix::CudaCopyEngine,
        token_io::{SelectPlacement, TokenSelector}};
    use cuteafd_engine::prefix::{FamilyLayout, MarkStore, PageOwners, PrefixCache, PrefixConfig};
    use planner::{Baseline, Budget, Decisions, Facts, Operation, Pin, PrefillShape, Request};

    #[test]
    #[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
    fn actual_ready_census_reconciles_scheduler_owners_and_rejects_individual_mismatches() -> Result<()> {
        let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build()?;
        let library = fixture.load()?;
        let total = 1 << 30;
        fixture.configure_memory(&library, 0, total)?;
        let device = Device { library: &library, id: 0 };
        for placement in [SelectPlacement::Device, SelectPlacement::Host] {
            let audit = AllocationAudit::start();
            let target = Allocation::new(device, 256)?;
            let target_events = audit.events().len();
            let target_receipt = receipt("fixture.target".into(), target.buffer,
                Category::Workspace, "fixture target", Lifetime::TARGET);
            let lifetime = Lifetime { first: Phase::Warmup, last: Phase::Growth };
            let mut owners = vec![target_receipt.allocation.clone(),
                Spec::buffer("selector.output", Location::coordinator(0), 64 * 12,
                    lifetime, Category::Workspace, "token selector"),
                Spec::buffer("selector.landing", Location::PINNED, 64 * 12,
                    lifetime, Category::Staging, "token selector")];
            if placement == SelectPlacement::Device {
                owners.extend(cuteafd_core::target_sampler_allocations(64, 33).unwrap().into_iter().map(|a|
                    Spec::buffer(format!("selector.sampler.{}", a.id),
                        if a.pinned { Location::PINNED } else { Location::coordinator(0) },
                        a.bytes as u64, lifetime, Category::Workspace, "target sampler")));
            }
            for index in 0..2 {
                owners.push(Spec::buffer(format!("host.prefix.{index}"), Location::PINNED, 4096,
                    lifetime, Category::Prefix, "retained host prefix chunk"));
            }
            let budgets = vec![Budget { id: "coordinator.0".into(), locations: vec![Location::coordinator(0)],
                physical_bytes: total as u64, logical_bytes: total as u64,
                baseline: Baseline::Sampled { free: total as u64, phase: Phase::Load, source: "CPU fixture".into() },
                headroom: 0 }, Budget { id: "host".into(), locations: vec![Location::HOST, Location::PINNED],
                physical_bytes: total as u64, logical_bytes: total as u64,
                baseline: Baseline::Forecast { occupied: 0, source: "CPU fixture".into(), assumed: false }, headroom: 0 }];
            let facts = Facts { allocations: owners, deferred: vec![], allowances: vec![] };
            let plan = ResolvedPlan { request: Request { context: 64, pool: Pin::Exact(64), onboard: Pin::Auto,
                prefill: Pin::Auto, operation: Operation::Serving }, decisions: Decisions {
                attention: vec![], onboard_layers: vec![vec![]], prefill: PrefillShape { rows: 4096, waves: 1 } },
                pool_tokens: 64, phases: planner::evaluate(&facts, &budgets)?, budgets, facts };
            let mut startup = Startup { plan, audit, loaded_events: 0, target_events,
                receipts: vec![target_receipt] };
            let mut selector = TokenSelector::new(&library, placement, 33, 64)?;
            selector.reserve_sampler()?;
            let copy = CudaCopyEngine::registered(&library, &[target.buffer])?;
            let layout = FamilyLayout { page_rows: 64, pages: 1, page_bytes: 256, mark_bytes: 0,
                draft_bytes: 0, rule: cuteafd_core::prefix::ReuseRule::EXACT,
                mark_store: MarkStore::Arena, page_owners: PageOwners::Uniform };
            let mut cache = PrefixCache::new(layout, PrefixConfig { entries: 1, mark_slots: 0,
                keep_logits: false, min_tokens: 1 }, Some((cuteafd_hostcache::config::Config {
                bytes: 2 * 4096 + 17, chunk_bytes: 4096, min_tokens: 1, ..Default::default() }, copy)))?;
            startup.ready(&library, device, &selector, &mut cache)?;
            let expected = startup.plan.clone();
            for mismatch in 0..4 {
                startup.plan = expected.clone();
                let owner = startup.plan.facts.allocations.iter_mut().find(|a| a.id == "selector.output").unwrap();
                match mismatch {
                    0 => owner.bytes += 4,
                    1 => owner.location = Location::PINNED,
                    2 => owner.lifetime.first = Phase::Load,
                    _ => { startup.plan.facts.allocations.retain(|a| a.id != "selector.output"); }
                }
                assert!(startup.ready(&library, device, &selector, &mut cache).is_err(), "mismatch {mismatch}");
            }
            startup.plan = expected;
            let extra = Allocation::new(device, 256)?;
            assert!(startup.ready(&library, device, &selector, &mut cache).is_err(), "unreceipted owner");
            drop(extra);
        }
        Ok(())
    }
}
