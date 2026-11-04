//! GLM coordinator target and drafter records through the shared push ranks.
use crate::shared::{experts::rtx::dense as coordinator_dense, memory::device::Device};
use anyhow::Result;

/// Each GPU's dense plan and destination, ranks `sparks..`.
#[allow(clippy::type_complexity)]
pub(crate) fn plan_ranks<'a>(snapshot: &std::path::Path, records: coordinator_dense::Records,
    devices: [Device<'a>; 2], sparks: usize)
    -> Result<(Vec<crate::shared::experts::rtx::push::DensePlan>, Vec<coordinator_dense::DenseArena<'a>>)> {
    coordinator_dense::plan_records(snapshot, records, &devices, sparks)
}
