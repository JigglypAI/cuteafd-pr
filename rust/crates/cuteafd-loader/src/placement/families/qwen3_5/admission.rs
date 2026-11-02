//! Dense single-device allocation facts, independent of planner policy.
use super::{
    DrafterBytes, NativeBuffers, NativeWeights, Qwen35Config, PAGE_ROWS, PREFILL_ROWS, WHOLE,
};
use crate::families::qwen::HybridAttention;
use crate::placement::{planner::*, ArchContext};
use cuteafd_core::memory_layout::Category;

pub struct DenseFacts<'a> {
    pub cfg: &'a Qwen35Config,
    pub weights: &'a NativeWeights,
    pub buffers: &'a NativeBuffers,
    pub slots: usize,
    pub record_bytes: u64,
    /// Slice 0b supplies exact drafter owners; these reservations are explicitly
    /// deferred and cannot satisfy a Ready reconciliation in the meantime.
    pub drafter: Option<DrafterBytes>,
}

pub fn forecast_budgets(
    physical: &ArchContext,
    capacity: u64,
    module_bytes: u64,
    host: (u64, u64),
) -> Result<Vec<Budget>, Refusal> {
    let context = physical
        .context_bytes
        .checked_add(module_bytes)
        .ok_or(Refusal::Overflow("context/modules"))?;
    Ok(vec![
        Budget {
            id: "coordinator.0".into(),
            locations: vec![Location::coordinator(0)],
            physical_bytes: if physical.total_bytes == 0 {
                32 << 30
            } else {
                physical.total_bytes
            },
            logical_bytes: capacity,
            baseline: Baseline::Forecast {
                occupied: context,
                source: if physical.total_bytes == 0 {
                    format!(
                        "{}; physical capacity assumes nominal 32 GiB (CUDA total unmeasured)",
                        physical.source
                    )
                } else {
                    physical.source.into()
                },
                assumed: physical.source.contains("5090"),
            },
            headroom: 1 << 30,
        },
        Budget {
            id: "host".into(),
            locations: vec![Location::HOST, Location::PINNED],
            physical_bytes: host.0,
            logical_bytes: host.0,
            baseline: Baseline::Sampled {
                free: host.1,
                phase: Phase::Load,
                source: "host/cgroup available".into(),
            },
            headroom: host.0 / 10,
        },
    ])
}

pub fn admit(
    request: Request,
    budgets: Vec<Budget>,
    facts: DenseFacts<'_>,
) -> Result<ResolvedPlan, Refusal> {
    if facts.weights.layers.len() != facts.cfg.attention.len()
        || facts.record_bytes == 0
        || facts.slots == 0
        || facts.buffers.head_rows
            != match request.operation {
                Operation::Serving => super::VERIFY_ROWS,
                Operation::Scoring => PREFILL_ROWS,
            }
    {
        return Err(Refusal::InvalidFact(
            "dense geometry or operation differs from allocation recipe".into(),
        ));
    }
    for (layer, &expected) in facts.weights.layers.iter().enumerate() {
        let prefix = format!("layer.{layer}.");
        let actual = facts
            .weights
            .allocations
            .iter()
            .filter(|a| a.id.starts_with(&prefix))
            .try_fold(0u64, |total, a| {
                total
                    .checked_add(a.bytes)
                    .ok_or(Refusal::Overflow("dense layer weights"))
            })?;
        if actual == 0 || actual != expected {
            return Err(Refusal::InvalidFact(format!(
                "dense layer {layer} inventory differs from physical operands"
            )));
        }
    }
    solve(request, budgets, &facts)
}

impl Recipe for DenseFacts<'_> {
    fn page_rows(&self) -> u64 {
        PAGE_ROWS
    }
    fn maximum_pool(&self) -> u64 {
        (u32::MAX as u64 / PAGE_ROWS) * PAGE_ROWS
    }
    fn candidates(&self) -> Vec<Decisions> {
        vec![Decisions {
            attention: vec![WHOLE; self.cfg.attention.len()],
            onboard_layers: vec![vec![]],
            prefill: PrefillShape {
                rows: PREFILL_ROWS as u64,
                waves: 1,
            },
        }]
    }
    fn evaluate(
        &self,
        _request: &Request,
        _decisions: &Decisions,
        pool: u64,
    ) -> Result<Facts, Refusal> {
        let mut allocations = Vec::new();
        let mut add = |id: String, location, bytes, lifetime, category, group: &str| {
            if bytes > 0 {
                allocations.push(Allocation::buffer(
                    id, location, bytes, lifetime, category, group,
                ));
            }
        };
        let gpu = Location::coordinator(0);
        for operand in &self.weights.allocations {
            add(
                format!("weights.{}", operand.id),
                gpu,
                operand.bytes,
                Lifetime::RESIDENT,
                Category::Weights,
                "native dense weights",
            );
        }
        add(
            "weights.embedding".into(),
            gpu,
            self.weights.embedding,
            Lifetime::RESIDENT,
            Category::Embedding,
            "token embedding",
        );
        // Embedding loading overlaps native weight loading, but ends before KV.
        add(
            "embedding.upload".into(),
            Location::PINNED,
            64 << 20,
            Lifetime {
                first: Phase::Load,
                last: Phase::Load,
            },
            Category::Staging,
            "embedding upload staging",
        );
        let staging = self
            .weights
            .allocations
            .iter()
            .map(|a| a.upload_bytes)
            .max()
            .unwrap_or(0);
        add(
            "upload.sync".into(),
            Location::PINNED,
            staging,
            Lifetime::RESIDENT,
            Category::Staging,
            "retained sync-H2D staging",
        );
        for (index, bytes) in self.buffers.planes.into_iter().enumerate() {
            add(
                format!("planes.{index}"),
                gpu,
                bytes as u64,
                Lifetime::TARGET,
                Category::Workspace,
                "native step planes and scratch",
            );
            if index >= 7 {
                add(
                    format!("planes.{index}.upload"),
                    Location::PINNED,
                    bytes as u64,
                    Lifetime::TARGET,
                    Category::Staging,
                    "ordered execution table staging",
                );
            }
        }
        add(
            "ids.upload".into(),
            Location::PINNED,
            self.buffers.ids as u64,
            Lifetime::TARGET,
            Category::Staging,
            "ordered token id staging",
        );
        for (id, group, bytes, category) in [
            ("ids", "token ids", self.buffers.ids, Category::Workspace),
            ("logits", "logits", self.buffers.logits, Category::Workspace),
            (
                "head",
                "vocabulary head workspace",
                self.buffers.head,
                Category::Workspace,
            ),
            (
                "gdn.commit",
                "GDN commit table",
                self.buffers.commit,
                Category::Workspace,
            ),
            (
                "prefix",
                "exact retained prefix marks",
                self.buffers.prefix,
                Category::Prefix,
            ),
        ] {
            add(
                id.into(),
                gpu,
                bytes as u64,
                Lifetime::TARGET,
                category,
                group,
            );
        }
        add(
            "gdn.commit.upload".into(),
            Location::PINNED,
            self.buffers.commit as u64,
            Lifetime::TARGET,
            Category::Staging,
            "ordered GDN commit staging",
        );
        for (index, bytes) in self.buffers.verify.into_iter().enumerate() {
            add(
                format!("verify.{index}"),
                gpu,
                bytes as u64,
                Lifetime::TARGET,
                Category::Workspace,
                "block verify marks, logits and taps",
            );
        }
        let layers = self
            .cfg
            .attention
            .iter()
            .filter(|&&kind| kind == HybridAttention::Gdn)
            .count() as u64;
        let conv_slot = ((self.cfg.gdn.conv_kernel - 1) * self.cfg.gdn.conv_width() * 2) as u64;
        let state_slot = self.buffers.state_per_layer as u64 - conv_slot;
        for (id, per_layer, group) in [
            (
                "gdn.conv",
                conv_slot.checked_mul(self.slots as u64),
                "GDN state",
            ),
            (
                "gdn.state",
                state_slot.checked_mul(self.slots as u64),
                "GDN state",
            ),
            (
                "gdn.replay",
                Some(self.buffers.replay_per_layer as u64),
                "GDN replay",
            ),
        ] {
            let bytes = per_layer
                .and_then(|n| n.checked_mul(layers))
                .ok_or(Refusal::Overflow("GDN backing"))?;
            if layers > 0 {
                add(
                    id.into(),
                    gpu,
                    bytes.max(256),
                    Lifetime::TARGET,
                    if id == "gdn.replay" {
                        Category::Workspace
                    } else {
                        Category::Kv
                    },
                    group,
                );
            }
        }
        for (layer, kind) in self.cfg.attention.iter().enumerate() {
            if *kind == HybridAttention::Full {
                add(
                    format!("kv.{layer}"),
                    gpu,
                    pool.checked_mul(self.record_bytes)
                        .ok_or(Refusal::Overflow("dense KV"))?,
                    Lifetime::TARGET,
                    Category::Kv,
                    "full GQA KV",
                );
            }
        }
        drop(add);
        let mut view = |id: String, backing: &str, offset, bytes, slot, dtype: &str, width| {
            let mut a = Allocation::buffer(
                id,
                gpu,
                bytes,
                Lifetime::TARGET,
                Category::Kv,
                "exact state view",
            );
            a.view = Some(View {
                backing: backing.into(),
                offset,
            });
            a.slot = Some(slot);
            a.dtype = dtype.into();
            a.shape = vec![bytes / width];
            allocations.push(a);
        };
        for ordinal in 0..layers {
            for slot in 0..self.slots as u64 {
                let record = ordinal * self.slots as u64 + slot;
                view(
                    format!("gdn.conv.{ordinal}.slot.{slot}"),
                    "gdn.conv",
                    record * conv_slot,
                    conv_slot,
                    slot as u32,
                    "bf16",
                    2,
                );
                view(
                    format!("gdn.state.{ordinal}.slot.{slot}"),
                    "gdn.state",
                    record * state_slot,
                    state_slot,
                    slot as u32,
                    "f32",
                    4,
                );
                let mark_offset = record * self.buffers.state_per_layer as u64;
                view(
                    format!("verify.state.{ordinal}.slot.{slot}.conv"),
                    "verify.0",
                    mark_offset,
                    conv_slot,
                    slot as u32,
                    "bf16",
                    2,
                );
                view(
                    format!("verify.state.{ordinal}.slot.{slot}.recurrent"),
                    "verify.0",
                    mark_offset + conv_slot,
                    state_slot,
                    slot as u32,
                    "f32",
                    4,
                );
            }
        }
        for mark in 0..self.buffers.marks as u64 {
            view(
                format!("prefix.mark.{mark}"),
                "prefix",
                mark * self.buffers.mark_bytes as u64,
                self.buffers.mark_bytes as u64,
                mark as u32,
                "bytes",
                1,
            );
        }
        let mut deferred = Vec::new();
        if let Some(draft) = self.drafter {
            let bytes = draft
                .owned
                .checked_add(draft.scratch)
                .and_then(|n| {
                    draft
                        .pool_unit
                        .checked_mul(pool / PAGE_ROWS)
                        .and_then(|kv| n.checked_add(kv))
                })
                .ok_or(Refusal::Overflow("deferred drafter"))?;
            deferred.push(Allocation::buffer(
                "deferred.drafter",
                gpu,
                bytes,
                Lifetime::TARGET,
                Category::Drafter,
                "drafter reserved; exact owners pending Slice 0b",
            ));
        }
        // This was an ownerless Exact demand. It remains an explicit unmeasured
        // growth allowance, never a receipt or a physical-card measurement.
        let allowances = vec![Allowance { location: gpu, bytes: 512 << 20, lifetime: Lifetime::TARGET,
            source: "dense head/runtime growth: unqualified pre-existing allowance".into(), assumed: true },
            Allowance { location: Location::HOST, bytes: self.weights.loading_host_bytes,
                lifetime: Lifetime { first: Phase::Load, last: Phase::Load },
                source: "checkpoint headers: one tensor/operand read plus packed Vec capacity; read_exact_at, no shard buffer".into(),
                assumed: true }];
        Ok(Facts {
            allocations,
            deferred,
            allowances,
        })
    }
}

#[cfg(test)]
mod tests;
