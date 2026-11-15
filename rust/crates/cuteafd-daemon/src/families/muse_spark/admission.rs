//! What the Muse coordinator tells its Spark workers, derived from the solved
//! placement and the shared endpoint rule (plan == runtime): the Spark working
//! set (layers whose experts live on the Sparks), the sealed `ModelAdmission`
//! (one endpoint shared by decode, verify and prefill, its partial-reply frame
//! and depth, the typed owners), and the coordinator transport's frame budget.
use anyhow::{ensure, Context, Result};
use cuteafd_core::ExpertGeometry;
use cuteafd_loader::families::muse_spark::replies::Cutoff;
use cuteafd_loader::placement::{families::muse_spark as shared, ExpertHome, Placement, SparkWorkingSet};
use cuteafd_loader::serving_capacity::prefill_waves::PrefillPlan;
use cuteafd_transport::worker_selection::ModelAdmission;

/// The Spark side of one admitted placement.
#[derive(Debug, Clone)]
pub(crate) struct SparkPlan {
    pub working: SparkWorkingSet,
    pub model: ModelAdmission,
    /// The coordinator transport's frame budget: the endpoint's own frame.
    pub max_frame_bytes: usize,
}

pub(crate) fn spark_plan(placement: &Placement, prefill: PrefillPlan, cutoff: Cutoff, sequences: usize,
    spark_ranks: usize) -> Result<SparkPlan> {
    let geometry = ExpertGeometry::MUSE_SPARK;
    let layers_of = |home: fn(&ExpertHome) -> bool| placement.layers.iter().enumerate()
        .filter_map(|(layer, assignment)| home(&assignment.experts).then_some(layer)).collect::<Vec<_>>();
    let spark_layers = layers_of(|home| *home == ExpertHome::Spark);
    let rtx_layers = layers_of(|home| matches!(home, ExpertHome::RtxTp2 | ExpertHome::RtxWhole { .. }));
    ensure!(spark_layers.len() + rtx_layers.len() == placement.layers.len(),
        "Muse places every layer's experts on the Sparks or an RTX (all layers are MoE)");
    ensure!(rtx_layers.len() == placement.onboard_layers, "Muse onboarded layers differ from the placement count");
    // Current RTX capacities refuse an all-local Muse plan before this runtime path.
    ensure!(!spark_layers.is_empty(), "Muse serves with at least one Spark layer");
    let (endpoint, rings) = shared::spark_endpoint(prefill, cutoff, sequences)?;
    let max_frame_bytes = endpoint.frame_bytes(geometry, rings).context("Muse Spark endpoint frame")?;
    let rings = cuteafd_core::SparkRings { max_frame_bytes, ..rings };
    let executions = shared::spark_executions(prefill.waves, prefill.unit_rows)?;
    let model = ModelAdmission::new(vec![endpoint], rings, executions)?;
    model.validate(geometry)?;
    let working = SparkWorkingSet { rtx_layers, spark_layers, spark_ranks, prefill_lanes: shared::LANES };
    Ok(SparkPlan { working, model, max_frame_bytes })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The coordinator transport accepts the sealed frame at every wide format
    /// (the native budget admits partial-reply frames up to the all-BF16 bound).
    #[test]
    fn coordinator_frame_is_within_the_native_budget() -> Result<()> {
        use cuteafd_loader::families::muse_spark::replies::WideFormat;
        let budget = cuteafd_transport::expert::native_frame_budget(ExpertGeometry::MUSE_SPARK, 2752)?;
        for wide in [WideFormat::Nvfp4, WideFormat::Fp8] {
            for sequences in [1, 16, 64] {
                let prefill = PrefillPlan { unit_rows: 2752, waves: 3, span_rows: 8256 };
                let (endpoint, rings) = shared::spark_endpoint(prefill, Cutoff::with_wide(64, 1, wide)?, sequences)?;
                let frame = endpoint.frame_bytes(ExpertGeometry::MUSE_SPARK, rings).context("frame")?;
                assert!(frame <= budget, "{wide:?} x {sequences}: frame {frame} > budget {budget}");
            }
        }
        Ok(())
    }
}
