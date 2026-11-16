//! Unchanged core tensor bytes in one caller-admitted arena per weight set.
use crate::shared::{experts::rtx::dense::{DenseLoad, Operand, Rect},
    memory::{device::{Allocation, Device}, receipts::Inventory, weight_arena::DeviceWeightArena}};
use anyhow::{ensure, Context, Result};
use cuteafd_core::memory_layout::ArenaTag;
use cuteafd_loader::{families::inkling2::weight_layout::CoordinatorWeights, plan::Checkpoint,
    serving_capacity::WeightArenaLayout};
use std::collections::BTreeMap;

/// Views retain their physical backing independently of the loading object.
/// DenseLoad supplies the shared mount/recorded-push path and staging owner.
pub(crate) struct TargetWeights<'a> {
    pub rank: usize,
    pub layers: Vec<usize>,
    operands: BTreeMap<String, Allocation<'a>>,
}

/// Unpublished target delivery; the only consumer requires the shared Spark
/// validation barrier and whole-group committed receive before returning views.
pub(crate) struct PendingTarget<'a> {
    target: TargetWeights<'a>,
    draft: Option<crate::shared::speculation::recurrent_weights::Weights<'a>>,
    push: crate::shared::experts::rtx::push::CoordinatorPush,
    dense: (Vec<crate::shared::experts::rtx::push::DensePlan>, Vec<crate::shared::experts::rtx::dense::DenseArena<'a>>),
    device: Device<'a>,
    sparks: usize,
}

impl<'a> PendingTarget<'a> {
    pub fn prepare(device: Device<'a>, view: &cuteafd_loader::plan::checkpoint::CheckpointView,
        checkpoint: &Checkpoint, catalog: &cuteafd_loader::OfficialV41Catalog,
        inputs: &cuteafd_loader::placement::families::inkling2::InklingInputs,
        admitted: &cuteafd_loader::placement::families::inkling2::AdmittedInkling)
        -> Result<(Self, (Vec<std::net::TcpStream>, crate::shared::memory::push_exl3::CoordinatorRanks))> {
        use crate::shared::experts::{layer::ExpertLayer, rtx::push::{CoordinatorExperts, CoordinatorPush, PushPolicy, WholeOwner}};
        ensure!(inputs.gpus == 1 && inputs.spark_ranks > 0 && checkpoint.snapshot == view.snapshot()
            && PushPolicy::for_snapshot(view.snapshot())? == PushPolicy::Required,
            "Inkling target transaction requires admitted whole-rank SparkNest push");
        let target_layout = inputs.targets.first().context("Inkling target layout absent")?.execution_arena()?;
        let planned = admitted.solved.facts.allocations.iter().find(|owner| owner.id == target_layout.id())
            .context("Inkling target arena absent from solved admission")?;
        ensure!(planned.bytes == target_layout.bytes() as u64
            && planned.location == cuteafd_loader::placement::planner::Location::coordinator(0),
            "Inkling target backing differs from solved admission");
        let envelope = inputs.push_host.get(admitted.onboard).and_then(Option::as_ref)
            .context("Inkling target transaction has no admitted push host owners")?.clone();
        let backings = match &inputs.local {
            cuteafd_loader::placement::families::inkling2::LocalPackage::Installed { backings } =>
                backings.get(admitted.onboard).context("Inkling selected expert prefix missing")?.as_slice(),
            _ if admitted.onboard == 0 => &[],
            _ => anyhow::bail!("Inkling selected local expert backend missing"),
        };
        let sets = backings.iter().map(|backing| CoordinatorExperts::Whole { owners: vec![WholeOwner {
            rank: 0, budget: backing.arena.bytes(), selections: backing.layers.iter().map(|&layer|
                ExpertLayer::BackboneFull { layer }).collect(),
        }] }).collect::<Vec<_>>();
        let experts = (!sets.is_empty()).then_some(CoordinatorExperts::Sets(sets));
        let ((target, draft), records) = crate::shared::experts::rtx::dense::recording(|| {
            let target = TargetWeights::record_push(device, checkpoint, &inputs.targets[0])?;
            let draft = inputs.drafter.as_ref().map(|draft| {
                let layout = draft.layout.weights.first().context("admitted drafter rank absent")?;
                let planned = admitted.solved.facts.allocations.iter().find(|owner|owner.id==layout.arena.id())
                    .context("drafter weight arena absent from solved admission")?;
                ensure!(planned.bytes==layout.arena.bytes() as u64
                    && planned.location==cuteafd_loader::placement::planner::Location::coordinator(0),
                    "drafter backing differs from pre-allocation admission");
                let source = cuteafd_loader::formats::block_dspark::recurrent::loading::RecurrentSource::open(view)?;
                crate::shared::speculation::recurrent_weights::Weights::load(device,&source,layout.clone(),std::ptr::null_mut())
            }).transpose()?;
            Ok((target,draft))
        })?;
        let dense = crate::shared::experts::rtx::dense::plan_records(view.snapshot(), records, &[device], inputs.spark_ranks)?;
        let (push, streams, ranks) = CoordinatorPush::prepare_view_admitted(view, Some(catalog), 1, experts, envelope)?;
        Ok((Self { target, draft, push, dense, device, sparks: inputs.spark_ranks }, (streams, ranks)))
    }
    pub fn finish(self, catalog: &cuteafd_loader::OfficialV41Catalog,
        pending: crate::shared::placement::PendingSparkAdmission)
        -> Result<(TargetWeights<'a>, Vec<Vec<crate::shared::experts::rtx::push::PushedExperts<'a>>>,
            Option<crate::shared::speculation::recurrent_weights::Weights<'a>>)> {
        let Self { target, draft, push, dense, device, sparks } = self;
        crate::shared::placement::load_while_admitting(Some(pending), || {
            let (received, ()) = push.receive_sets_with(vec![device], Some(catalog), sparks, Some(Ok(dense)), || ());
            let experts = crate::shared::experts::rtx::push::PushPolicy::Required.committed(received)?
                .context("Inkling target push group did not commit")?;
            Ok((target, experts, draft))
        })
    }
}

pub(crate) fn execution_arena(layout: &CoordinatorWeights) -> Result<WeightArenaLayout> {
    layout.execution_arena()
}

fn transpose_relative(source: &[u8]) -> Result<Vec<u8>> {
    ensure!(source.len() == 16*1024*2, "Inkling relative projection must be logical [16,1024] BF16");
    let mut output = vec![0u8; 1024*64*2];
    for row in 0..1024 {
        for column in 0..16 {
            output[(row*64+column)*2..(row*64+column+1)*2]
                .copy_from_slice(&source[(column*1024+row)*2..(column*1024+row+1)*2]);
        }
    }
    Ok(output)
}

impl<'a> TargetWeights<'a> {
    pub fn load(device: Device<'a>, checkpoint: &Checkpoint, layout: &CoordinatorWeights) -> Result<Self> {
        ensure!(layout.core.is_empty(),
            "Inkling compact core is planned; shared native trellis execution binds with the serving stage");
        ensure!(layout.rank == 0 && layout.arena.views().keys().all(|name| !name.contains('#')),
            "Inkling stage2 core mount requires whole coordinator operands");
        ensure!(!crate::shared::experts::rtx::dense::is_recording(),
            "Inkling relative transpose is a cold mount transform, not yet a push finish");
        let mut rects = BTreeMap::new();
        for (name, view) in layout.arena.views() {
            let rect = Rect::tensor(checkpoint, name, None, None)?;
            ensure!(rect.bytes() == view.bytes(), "Inkling core {name}: checkpoint differs from admitted view");
            rects.insert(name.clone(), rect);
        }
        let dense = DenseLoad::mount(device.library);
        dense.admit_mount(layout.mount_load(checkpoint)?)?;
        let arena = DeviceWeightArena::new(device, execution_arena(layout)?, ArenaTag::weights("inkling2/target"))?;
        let mut operands = BTreeMap::new();
        for (name, rect) in rects {
            let bytes = rect.bytes();
            let allocation = arena.view(&name, bytes)?;
            dense.operand(Operand { device: device.id, buffer: allocation.buffer, lands: vec![(rect, 0)], bytes })?;
            operands.insert(name, allocation);
        }
        dense.release();
        for &layer in &layout.layers {
            let name = format!("model.llm.layers.{layer}.attn.rel_logits_proj");
            let mut source = vec![0u8; 16*1024*2];
            Rect::tensor(checkpoint, &format!("{name}.proj"), None, None)?.read(&mut source)?;
            let padded = transpose_relative(&source)?;
            let allocation = arena.view(&format!("{name}.padded"), padded.len())?;
            device.run(|| device.library.copy_h2d(allocation.buffer, &padded))?;
            tracing::info!(layer, logical = "[16,1024]", physical = "[1024,64]", logical_k = 16,
                physical_k = 64, source_staging_bytes = source.len(), output_staging_bytes = padded.len(),
                padding_bytes = 1024*48*2, padding_zero_exact = true, "Inkling relative raw-BF16 transpose/pad load");
            operands.insert(format!("{name}.padded"), allocation);
        }
        Ok(Self { rank: layout.rank, layers: layout.layers.clone(), operands })
    }
    /// Header-only delivery records plus generated final views. This entry is
    /// not used by serving until the padded-transpose fork export is pinned.
    pub fn record_push(device: Device<'a>, checkpoint: &Checkpoint, layout: &CoordinatorWeights) -> Result<Self> {
        use crate::shared::experts::rtx::dense::Finish;
        ensure!(crate::shared::experts::rtx::dense::is_recording(), "Inkling push weights require shared delivery recording");
        ensure!(layout.core.is_empty() && layout.rank == 0
            && layout.arena.views().keys().all(|name| !name.contains('#')),
            "Inkling initial push requires original whole-rank target operands");
        let dense = DenseLoad::mount(device.library);
        let arena = DeviceWeightArena::new(device, layout.execution_arena()?, ArenaTag::weights("inkling2/target"))?;
        let mut operands = BTreeMap::new();
        for (name, view) in layout.arena.views() {
            let rect = Rect::tensor(checkpoint, name, None, None)?;
            ensure!(rect.bytes() == view.bytes(), "Inkling push header differs from admitted view {name}");
            let owner = arena.view(name, view.bytes())?;
            dense.operand(Operand { device: device.id, buffer: owner.buffer, lands: vec![(rect, 0)], bytes: view.bytes() })?;
            operands.insert(name.clone(), owner);
        }
        for &layer in &layout.layers {
            let prefix = format!("model.llm.layers.{layer}.attn.rel_logits_proj");
            let source = operands.get(&format!("{prefix}.proj")).context("Inkling push relative source absent")?.buffer;
            let output = arena.view(&format!("{prefix}.padded"), 1024*64*2)?;
            dense.finish(Finish::PaddedTranspose { source, output: output.buffer, dtype: cuteafd_core::DType::Bf16,
                rows: 16, cols: 1024, padded_cols: 64 }, std::ptr::null_mut())?;
            operands.insert(format!("{prefix}.padded"), output);
        }
        // Generated views have no delivery Operand: shared DenseArena executes
        // every registered finish and drains before publishing rank Done.
        Ok(Self { rank: layout.rank, layers: layout.layers.clone(), operands })
    }
    pub fn operand(&self, name: &str) -> Result<&Allocation<'a>> {
        self.operands.get(name).with_context(|| format!("Inkling target lacks operand {name}"))
    }
    pub fn inventory(&self, inventory: &mut Inventory<'_>, prefix: &str) -> Result<()> {
        for (name, owner) in &self.operands { owner.inventory(inventory, &format!("{prefix}.{name}"))?; }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn push_records_source_and_generated_final_view_without_reading_or_host_transform() -> Result<()> {
        use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
        use cuteafd_loader::plan::testing::{t, write_snapshot};
        let directory = tempfile::tempdir()?;
        write_snapshot(directory.path(), &serde_json::json!({}), &[
            t("model.llm.layers.0.attn.rel_logits_proj.proj", "BF16", &[16, 1024])], None);
        let checkpoint = Checkpoint::open(directory.path())?;
        let layout = CoordinatorWeights { rank: 0, layers: vec![0], core: BTreeMap::new(),
            arena: WeightArenaLayout::new("inkling2/target/0", [(
                "model.llm.layers.0.attn.rel_logits_proj.proj".into(), 16*1024*2)])? };
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        let device = Device { library: &library, id: 0 };
        let (weights, records) = crate::shared::experts::rtx::dense::recording(||
            TargetWeights::record_push(device, &checkpoint, &layout))?;
        assert_eq!(records.operands.len(), 1, "padded view is generated, never delivered as payload");
        assert_eq!(records.finishes.len(), 1);
        let source = weights.operand("model.llm.layers.0.attn.rel_logits_proj.proj")?.buffer;
        let output = weights.operand("model.llm.layers.0.attn.rel_logits_proj.padded")?.buffer;
        assert_ne!(source.ptr, output.ptr);
        assert_eq!(output.bytes, 1024*64*2);
        assert!(matches!(records.finishes[0], crate::shared::experts::rtx::dense::Finish::PaddedTranspose {
            source: recorded, output: generated, rows: 16, cols: 1024, padded_cols: 64, ..
        } if recorded.ptr == source.ptr && generated.ptr == output.ptr));
        let rank = crate::shared::experts::rtx::dense::RankLayout::new(records.operands, records.steps, device.id)?
            .with_finishes(&records.finishes);
        assert_eq!(rank.generated.len(), 1);
        assert_eq!(rank.generated[0].ptr, output.ptr);
        assert_eq!(rank.operands.len(), 1);
        Ok(())
    }

    #[test]
    fn relative_transpose_preserves_bf16_bits_and_exact_zero_padding() {
        let source = (0..16*1024).flat_map(|n| ((n*29%65536) as u16).to_le_bytes()).collect::<Vec<_>>();
        let padded = transpose_relative(&source).unwrap();
        for row in 0..1024 {
            for col in 0..16 {
                assert_eq!(&padded[(row*64+col)*2..(row*64+col+1)*2],
                    &source[(col*1024+row)*2..(col*1024+row+1)*2]);
            }
            assert!(padded[(row*64+16)*2..(row+1)*64*2].iter().all(|&b| b == 0));
        }
        assert!(transpose_relative(&source[..100]).is_err());
    }
}
