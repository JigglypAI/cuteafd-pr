//! Shared simulated hardware for planner and runtime-admission tests.
use super::SimulatedHost;
use crate::placement::{inventory::PRO_TOTAL_BYTES, Baseline, GpuBudget};
use crate::plan::{layout::LayoutOptions, ExpertPlacement, PlanOptions};
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct SimCluster {
    pub host: SimulatedHost,
    context: u64,
    concurrency: u64,
    manifest: Option<PathBuf>,
}

impl SimCluster {
    pub fn rtx_pro(gpus: usize, sparks: usize) -> Self {
        assert!((1..=2).contains(&gpus) && sparks <= 8);
        Self { host: SimulatedHost { gpu_count: gpus, gpu_bytes: PRO_TOTAL_BYTES,
            spark_count: sparks, ..SimulatedHost::raptor() }, context: 0, concurrency: 16, manifest: None }
    }

    pub fn two_rtx_256g_host(sparks: usize) -> Self {
        Self::rtx_pro(2, sparks).host_gib(256, 256)
    }

    /// Memory-capped PRO, not an assertion of a 5090's physical SM count.
    pub fn sim5090(sparks: usize) -> Self { Self::rtx_pro(1, sparks).rtx_bytes(32 << 30) }
    pub fn zero_spark(gpus: usize) -> Self { Self::rtx_pro(gpus, 0) }

    pub fn host_gib(mut self, total: u64, available: u64) -> Self {
        assert!(total > 0 && available <= total);
        self.host.host_ram = (total.checked_mul(1 << 30).expect("host total"),
            available.checked_mul(1 << 30).expect("host available"));
        self
    }
    pub fn rtx_bytes(mut self, bytes: u64) -> Self {
        assert!(bytes > 0);
        self.host.gpu_bytes = bytes;
        self
    }
    pub fn context(mut self, tokens: u64) -> Self { self.context = tokens; self }
    pub fn concurrency(mut self, count: u64) -> Self {
        assert!(count > 0);
        self.concurrency = count;
        self
    }
    pub fn manifest(mut self, path: impl Into<PathBuf>) -> Self {
        self.manifest = Some(path.into());
        self
    }
    pub fn layout_options(&self) -> LayoutOptions {
        LayoutOptions { context_tokens: self.context, concurrency: self.concurrency,
            workspace_manifest: self.manifest.clone(), head_split: self.host.gpu_count == 2,
            ..self.host.layout() }
    }
    pub fn plan_options(&self) -> PlanOptions {
        PlanOptions { placement: ExpertPlacement::from_spark_ranks(self.host.spark_count),
            layout: Some(self.layout_options()), ..Default::default() }
    }
    pub fn coordinator_budgets(&self, family: &str) -> Vec<GpuBudget> {
        let context_bytes = crate::placement::ArchContext::for_device("sm_120", self.host.gpu_bytes).context_bytes;
        vec![GpuBudget { capacity_bytes: self.host.gpu_bytes,
            reserve_bytes: crate::placement::reserve::gpu(family),
            baseline: Baseline::Planned { context_bytes, loaded_bytes: 0 } }; self.host.gpu_count]
    }
    /// Occupancy is supplied by the test's modeled owner, never sampled from CUDA.
    pub fn measured(&self, occupied: u64) -> Vec<GpuBudget> {
        let free_bytes = self.host.gpu_bytes.checked_sub(occupied).expect("fixture occupancy exceeds capacity");
        vec![GpuBudget { capacity_bytes: self.host.gpu_bytes, reserve_bytes: 0,
            baseline: Baseline::Measured { free_bytes } }; self.host.gpu_count]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_and_projections_use_one_explicit_environment() {
        for cluster in [SimCluster::rtx_pro(1, 4), SimCluster::two_rtx_256g_host(6),
            SimCluster::sim5090(4), SimCluster::zero_spark(2)] {
            let options = cluster.clone().context(65536).concurrency(17).manifest("/fixture/PROGRAMS.json");
            let plan = options.plan_options();
            let layout = plan.layout.unwrap();
            assert_eq!(layout.host_memory, Some(options.host.host_ram));
            assert_eq!(layout.rtx_bytes, vec![options.host.gpu_bytes; options.host.gpu_count]);
            assert_eq!(plan.placement.spark_ranks(), options.host.spark_count);
            assert_eq!((layout.context_tokens, layout.concurrency), (65536, 17));
            assert_eq!(layout.workspace_manifest, Some(PathBuf::from("/fixture/PROGRAMS.json")));
            let occupied = 2 << 30;
            let measured = options.measured(occupied);
            assert_eq!(measured.len(), options.host.gpu_count);
            assert!(measured.iter().all(|b| b.available() == options.host.gpu_bytes - occupied));
            let budgets = options.coordinator_budgets("muse_spark");
            assert_eq!(budgets.len(), options.host.gpu_count);
            assert!(budgets.iter().all(|b| b.capacity_bytes == options.host.gpu_bytes));
        }
    }
}
