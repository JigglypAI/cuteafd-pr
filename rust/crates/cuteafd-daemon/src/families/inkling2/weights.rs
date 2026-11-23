//! Unchanged core tensor bytes in one caller-admitted arena per weight set.
use crate::shared::{dense_trellis::{CoreWeightLayout, CoreWeights},
    experts::rtx::dense::{DenseLoad, Operand, Rect},
    native_attention::{attention_records, AttentionRecord},
    memory::{device::{Allocation, Device}, receipts::Inventory, weight_arena::DeviceWeightArena}};
use anyhow::{ensure, Context, Result};
use cuteafd_core::memory_layout::ArenaTag;
use cuteafd_loader::{families::inkling2::weight_layout::{fp8_role, projection_prefix, CoordinatorWeights, NativeSource}, plan::Checkpoint,
    serving_capacity::WeightArenaLayout};
use std::collections::BTreeMap;

/// Views retain their physical backing independently of the loading object.
/// DenseLoad supplies the shared mount/recorded-push path and staging owner.
pub(crate) struct TargetWeights<'a> {
    pub rank: usize,
    pub layers: Vec<usize>,
    operands: BTreeMap<String, Allocation<'a>>,
    core: BTreeMap<String, CoreWeights<'a>>,
    fp8: BTreeMap<String, Fp8Weight<'a>>,
}

pub(crate) struct Fp8Weight<'a> {
    pub shape: [usize; 2],
    pub values: Allocation<'a>,
    pub scales: Allocation<'a>,
}

#[derive(Clone, Copy)]
pub(crate) enum ProjectionWeight<'s, 'a> {
    Bf16(&'s Allocation<'a>),
    Trellis(&'s CoreWeights<'a>),
    Fp8(&'s Fp8Weight<'a>),
}

/// Unpublished target delivery; the only consumer requires the shared Spark
/// validation barrier and whole-group committed receive before returning views.
pub(crate) struct PendingTarget<'a> {
    targets: Vec<TargetWeights<'a>>,
    draft: Vec<crate::shared::speculation::recurrent_weights::Weights<'a>>,
    push: crate::shared::experts::rtx::push::CoordinatorPush,
    dense: (Vec<crate::shared::experts::rtx::push::DensePlan>, Vec<crate::shared::experts::rtx::dense::DenseArena<'a>>),
    devices: Vec<Device<'a>>,
    sparks: usize,
    media: crate::shared::vision::push::Contribution,
}

impl<'a> PendingTarget<'a> {
    pub fn prepare(devices: Vec<Device<'a>>, view: &cuteafd_loader::plan::checkpoint::CheckpointView,
        checkpoint: &Checkpoint, catalog: &cuteafd_loader::OfficialV41Catalog,
        inputs: &cuteafd_loader::placement::families::inkling2::InklingInputs,
        admitted: &cuteafd_loader::placement::families::inkling2::AdmittedInkling,
        media: crate::shared::vision::push::Contribution)
        -> Result<(Self, (Vec<std::net::TcpStream>, crate::shared::memory::push_exl3::CoordinatorRanks))> {
        use crate::shared::experts::{layer::ExpertLayer, rtx::push::{CoordinatorExperts, CoordinatorPush, PushPolicy, WholeOwner}};
        ensure!(devices.len() == inputs.gpus && inputs.targets.len() == inputs.gpus && inputs.spark_ranks > 0
            && checkpoint.snapshot == view.snapshot() && PushPolicy::for_snapshot(view.snapshot())? == PushPolicy::Required,
            "Inkling target transaction requires admitted SparkNest push ranks");
        let check = |rank: usize, layout: &WeightArenaLayout| -> Result<()> {
            let planned = admitted.solved.facts.allocations.iter().find(|owner| owner.id == layout.id())
                .context("Inkling target or drafter arena absent from solved admission")?;
            ensure!(planned.bytes == layout.bytes() as u64
                && planned.location == cuteafd_loader::placement::planner::Location::coordinator(rank as u8),
                "Inkling weight backing differs from solved admission: {}", layout.id());
            Ok(())
        };
        for (rank, target) in inputs.targets.iter().enumerate() { check(rank, &target.execution_arena()?)?; }
        let backings = match &inputs.local {
            cuteafd_loader::placement::families::inkling2::LocalPackage::Installed { backings } =>
                backings.get(admitted.onboard).context("Inkling selected expert prefix missing")?.as_slice(),
            _ if admitted.onboard == 0 => &[],
            _ => anyhow::bail!("Inkling selected local expert backend missing"),
        };
        let sets = local_groups(backings, inputs.gpus)?.into_iter().map(|group| {
            if inputs.gpus == 2 {
                CoordinatorExperts::Tp2 { layers: group[0].layers.clone(),
                    budgets: group.iter().map(|owner| owner.arena.bytes()).collect() }
            } else {
                CoordinatorExperts::Whole { owners: vec![WholeOwner { rank: 0, budget: group[0].arena.bytes(),
                    selections: group[0].layers.iter().map(|&layer| ExpertLayer::BackboneFull { layer }).collect() }] }
            }
        }).collect::<Vec<_>>();
        let experts = (!sets.is_empty()).then_some(CoordinatorExperts::Sets(sets));
        let ((targets, draft), mut records) = crate::shared::experts::rtx::dense::recording(|| {
            let targets = devices.iter().zip(&inputs.targets).map(|(&device, layout)|
                TargetWeights::record_push(device, checkpoint, layout)).collect::<Result<Vec<_>>>()?;
            let draft = if let Some(draft) = &inputs.drafter {
                ensure!(draft.layout.weights.len() == devices.len(), "Inkling admitted drafter ranks differ");
                let source = cuteafd_loader::formats::block_dspark::recurrent::loading::RecurrentSource::open(view)?;
                devices.iter().zip(&draft.layout.weights).enumerate().map(|(rank, (&device, layout))| {
                    check(rank, &layout.arena)?;
                    crate::shared::speculation::recurrent_weights::Weights::load(device, &source, layout.clone(), std::ptr::null_mut())
                }).collect::<Result<Vec<_>>>()?
            } else { Vec::new() };
            Ok((targets, draft))
        })?;
        records.extend(media.records());
        let dense = crate::shared::experts::rtx::dense::plan_records(view.snapshot(), records, &devices, inputs.spark_ranks)?;
        let (push, streams, ranks) = CoordinatorPush::prepare_view(view, Some(catalog), inputs.gpus, experts)?;
        Ok((Self { targets, draft, push, dense, devices, sparks: inputs.spark_ranks, media }, (streams, ranks)))
    }

    pub fn finish(self, catalog: &cuteafd_loader::OfficialV41Catalog,
        pending: crate::shared::placement::PendingSparkAdmission)
        -> Result<(Vec<TargetWeights<'a>>, Vec<Vec<crate::shared::experts::rtx::push::PushedExperts<'a>>>,
            Vec<crate::shared::speculation::recurrent_weights::Weights<'a>>)> {
        let Self { targets, draft, push, dense, devices, sparks, media } = self;
        crate::shared::placement::load_while_admitting(Some(pending), || {
            let (received, ()) = push.receive_sets_with(devices, Some(catalog), sparks, Some(Ok(dense)), || ());
            let experts = crate::shared::experts::rtx::push::PushPolicy::Required.committed(received)?
                .context("Inkling target push group did not commit")?;
            media.commit()?;
            Ok((targets, experts, draft))
        })
    }
}

pub(crate) fn local_groups(backings: &[cuteafd_loader::placement::families::inkling2::LocalBacking], gpus: usize)
    -> Result<Vec<&[cuteafd_loader::placement::families::inkling2::LocalBacking]>> {
    ensure!(matches!(gpus, 1 | 2) && backings.len().is_multiple_of(gpus), "Inkling expert backing rank count differs");
    backings.chunks_exact(gpus).map(|group| {
        ensure!(!group[0].layers.is_empty() && group.iter().enumerate().all(|(rank, owner)|
            owner.gpu == rank && owner.layers == group[0].layers), "Inkling grouped expert backing differs from rank assignment");
        Ok(group)
    }).collect()
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

fn native_rect(checkpoint: &Checkpoint, view: &str, rank: usize) -> Result<(String, Rect)> {
    let source = NativeSource::new(checkpoint, view, rank)?;
    let rect = Rect::tensor(checkpoint, source.name, Some(source.rows), Some(source.columns))?;
    Ok((source.name.to_owned(), rect))
}

impl<'a> TargetWeights<'a> {
    pub fn load(device: Device<'a>, checkpoint: &Checkpoint, layout: &CoordinatorWeights) -> Result<Self> {
        ensure!(!crate::shared::experts::rtx::dense::is_recording(),
            "Inkling relative transpose is a cold mount transform, not yet a push finish");
        let mut rects = BTreeMap::new();
        for (name, view) in layout.arena.views() {
            if layout.is_specialized_view(name) { continue; }
            let (operand, rect) = native_rect(checkpoint, name, layout.rank)?;
            ensure!(rect.bytes() == view.bytes(), "Inkling core {name}: checkpoint differs from admitted view");
            rects.insert(name.clone(), (operand, rect));
        }
        let dense = DenseLoad::mount(device.library);
        dense.admit_mount(layout.mount_load(checkpoint)?)?;
        let arena = DeviceWeightArena::new(device, execution_arena(layout)?, ArenaTag::weights("inkling2/target"))?;
        let mut operands = BTreeMap::new();
        for (name, (operand, rect)) in rects {
            let bytes = rect.bytes();
            let allocation = arena.view(&name, bytes)?;
            dense.operand(Operand { device: device.id, buffer: allocation.buffer, lands: vec![(rect, 0)], bytes })?;
            operands.insert(operand, allocation);
        }
        let (core, fp8) = load_specialized(device, checkpoint, layout, &arena, &dense)?;
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
        Ok(Self { rank: layout.rank, layers: layout.layers.clone(), operands, core, fp8 })
    }
    /// Header-only delivery records plus generated final views. This entry is
    /// not used by serving until the padded-transpose fork export is pinned.
    pub fn record_push(device: Device<'a>, checkpoint: &Checkpoint, layout: &CoordinatorWeights) -> Result<Self> {
        use crate::shared::experts::rtx::dense::Finish;
        ensure!(crate::shared::experts::rtx::dense::is_recording(), "Inkling push weights require shared delivery recording");
        let dense = DenseLoad::mount(device.library);
        let arena = DeviceWeightArena::new(device, layout.execution_arena()?, ArenaTag::weights("inkling2/target"))?;
        let mut operands = BTreeMap::new();
        for (name, view) in layout.arena.views() {
            if layout.is_specialized_view(name) { continue; }
            let (operand, rect) = native_rect(checkpoint, name, layout.rank)?;
            ensure!(rect.bytes() == view.bytes(), "Inkling push header differs from admitted view {name}");
            let owner = arena.view(name, view.bytes())?;
            dense.operand(Operand { device: device.id, buffer: owner.buffer, lands: vec![(rect, 0)], bytes: view.bytes() })?;
            operands.insert(operand, owner);
        }
        for &layer in &layout.layers {
            let prefix = format!("model.llm.layers.{layer}.attn.rel_logits_proj");
            let source = operands.get(&format!("{prefix}.proj")).context("Inkling push relative source absent")?.buffer;
            let output = arena.view(&format!("{prefix}.padded"), 1024*64*2)?;
            dense.finish(Finish::PaddedTranspose { source, output: output.buffer, dtype: cuteafd_core::DType::Bf16,
                rows: 16, cols: 1024, padded_cols: 64 }, std::ptr::null_mut())?;
            operands.insert(format!("{prefix}.padded"), output);
        }
        let (core, fp8) = load_specialized(device, checkpoint, layout, &arena, &dense)?;
        // Generated views have no delivery Operand: shared DenseArena executes
        // every registered finish and drains before publishing rank Done.
        Ok(Self { rank: layout.rank, layers: layout.layers.clone(), operands, core, fp8 })
    }
    pub fn operand(&self, name: &str) -> Result<&Allocation<'a>> {
        self.operands.get(name).with_context(|| format!("Inkling target lacks operand {name}"))
    }
    pub fn projection(&self, name: &str) -> Result<ProjectionWeight<'_, 'a>> {
        let stem = name.strip_suffix(".weight").unwrap_or(name);
        if let Some(weight) = self.core.get(stem) { return Ok(ProjectionWeight::Trellis(weight)); }
        if let Some(weight) = self.fp8.get(stem) { return Ok(ProjectionWeight::Fp8(weight)); }
        Ok(ProjectionWeight::Bf16(self.operand(name)?))
    }
    pub fn inventory(&self, inventory: &mut Inventory<'_>, prefix: &str) -> Result<()> {
        for (name, owner) in &self.operands { owner.inventory(inventory, &format!("{prefix}.{name}"))?; }
        for (name, w) in &self.core {
            for (suffix, owner) in [("trellis", &w.payload), ("suh", &w.suh), ("svh", &w.svh)] {
                owner.inventory(inventory, &format!("{prefix}.{name}.{suffix}"))?;
            }
        }
        for (name, w) in &self.fp8 {
            w.values.inventory(inventory, &format!("{prefix}.{name}.values"))?;
            w.scales.inventory(inventory, &format!("{prefix}.{name}.scales"))?;
        }
        Ok(())
    }
}

fn load_specialized<'a>(device: Device<'a>, checkpoint: &Checkpoint, layout: &CoordinatorWeights,
    arena: &DeviceWeightArena<'a>, dense: &DenseLoad<'a>)
    -> Result<(BTreeMap<String, CoreWeights<'a>>, BTreeMap<String, Fp8Weight<'a>>)> {
    let mut core = BTreeMap::new();
    let mut fp8 = BTreeMap::new();
    let stream = std::ptr::null_mut();
    for (stem, weight) in &layout.core {
        let prefix = projection_prefix(stem, weight.rank, weight.ranks);
        let native = CoreWeightLayout::from_shared(weight.clone());
        let loaded = native.load_recorded_with(device, checkpoint, dense, stream, |name, bytes| {
            let suffix = name.strip_prefix(stem).context("Inkling trellis view stem changed")?;
            arena.view(&format!("{prefix}{suffix}"), bytes)
        })?;
        core.insert(stem.clone(), loaded);
    }
    for (stem, weight) in &layout.fp8 {
        let prefix = projection_prefix(stem, weight.rank, weight.ranks);
        let [n, k] = weight.shape;
        let values = arena.view(&format!("{prefix}.values"), n*k)?;
        let scales = arena.view(&format!("{prefix}.scales"), n/128*(k/128)*4)?;
        let selected = checkpoint.projection(stem)?;
        for record in attention_records(&checkpoint.snapshot, &selected, checkpoint.projection_view(stem), weight,
            fp8_role(stem), crate::shared::fp8_linear::Fp8Scales::Amax, values.buffer, Some(scales.buffer))? {
            match record {
                AttentionRecord::Operand(op) => dense.operand(op)?,
                AttentionRecord::Step(step) => dense.step(step, stream)?,
            }
        }
        fp8.insert(stem.clone(), Fp8Weight { shape: weight.shape, values, scales });
    }
    Ok((core, fp8))
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
        let layout = CoordinatorWeights { rank: 0, layers: vec![0], core: BTreeMap::new(), fp8: BTreeMap::new(),
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
