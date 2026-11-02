//! Actual-owner receipts for the no-drafter target, including pinned owners.
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
) -> Result<()> {
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
    let phase = if plan.facts.deferred.is_empty() {
        Phase::Ready
    } else {
        Phase::PostTargetKv
    };
    planner::reconcile(plan, phase, &receipts)?;
    let mut live = BTreeMap::new();
    let mut observations = Vec::new();
    for (index, event) in audit.events().into_iter().enumerate() {
        let at = if index < loaded_events {
            Phase::Load
        } else {
            Phase::PostTargetKv
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
    for receipt in &receipts {
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
    let (free, _) = library.cuda_memory_info()?;
    planner::reconcile_usage(
        plan,
        "coordinator.0",
        phase,
        before.saturating_sub(free as u64),
    )?;
    Ok(())
}
