//! Packed GPU-streamed target weights for Inkling 2 Mini (TP1).
//!
//! Streams every target tensor directly into one preplanned `DeviceWeightArena`
//! via shared `DenseLoad` + `CoordinatorPush::prepare_alone_view` (or mount outside
//! SparkNest namespaces), with GPU-side `Finish::PaddedTranspose` for all 24
//! relative-bias projections and zero CPU per-tensor transforms.
use crate::shared::{
    experts::rtx::dense::{DenseLoad, Finish, Operand, Rect},
    memory::{
        device::{Allocation, Device},
        receipts::Inventory,
        weight_arena::DeviceWeightArena,
    },
};
use anyhow::{ensure, Context, Result};
use cuteafd_core::memory_layout::ArenaTag;
use cuteafd_loader::{
    families::inkling2_mini::weight_layout::MiniTargetWeights as TargetLayout,
    placement::families::inkling2_mini::{AdmittedMini, LocalPackage, MiniInputs},
    plan::Checkpoint,
    serving_capacity::WeightArenaLayout,
};
use std::collections::BTreeMap;

pub(crate) struct MiniTargetWeights<'a> {
    pub layers: Vec<usize>,
    operands: BTreeMap<String, Allocation<'a>>,
}

impl<'a> MiniTargetWeights<'a> {
    /// Streams all target operands into one packed arena using GPU-side `Finish::PaddedTranspose`.
    /// Works identically inside `dense::recording(...)` (SparkNest push) and direct mount mode.
    pub fn stream(device: Device<'a>, checkpoint: &Checkpoint, layout: &TargetLayout) -> Result<Self> {
        let recording = crate::shared::experts::rtx::dense::is_recording();
        let dense = DenseLoad::mount(device.library);
        if !recording {
            dense.admit_mount(layout.mount_load(checkpoint)?)?;
        }
        let arena = DeviceWeightArena::new(
            device,
            layout.execution_arena()?,
            ArenaTag::weights("inkling2_mini/target"),
        )?;
        let mut operands = BTreeMap::new();
        for (name, view) in layout.arena.views() {
            let slices = layout.sources(checkpoint, name)?;
            let mut lands = Vec::with_capacity(slices.len());
            let mut total = 0usize;
            for slice in slices {
                let rect = Rect::tensor(checkpoint, &slice.tensor, Some(slice.rows), Some(slice.columns))?;
                total += rect.bytes();
                lands.push((rect, slice.dst_offset));
            }
            ensure!(
                total == view.bytes(),
                "Inkling 2 Mini target {name}: source slices ({total}) differ from admitted view ({})",
                view.bytes()
            );
            let owner = arena.view(name, view.bytes())?;
            dense.operand(Operand {
                device: device.id,
                buffer: owner.buffer,
                lands,
                bytes: view.bytes(),
            })?;
            operands.insert(name.clone(), owner);
        }
        for &layer in &layout.layers {
            let prefix = format!("model.llm.layers.{layer}.attn.rel_logits_proj");
            let source = operands
                .get(&format!("{prefix}.proj"))
                .context("Inkling 2 Mini relative projection source absent")?
                .buffer;
            let output = arena.view(&format!("{prefix}.padded"), 1024 * 64 * 2)?;
            dense.finish(
                Finish::PaddedTranspose {
                    source,
                    output: output.buffer,
                    dtype: cuteafd_core::DType::Bf16,
                    rows: 16,
                    cols: 1024,
                    padded_cols: 64,
                },
                std::ptr::null_mut(),
            )?;
            operands.insert(format!("{prefix}.padded"), output);
        }
        if !recording {
            dense.release();
        }
        Ok(Self {
            layers: layout.layers.clone(),
            operands,
        })
    }

    pub fn operand(&self, name: &str) -> Result<&Allocation<'a>> {
        self.operands
            .get(name)
            .with_context(|| format!("Inkling 2 Mini target lacks operand {name}"))
    }

    pub fn inventory(&self, inventory: &mut Inventory<'_>, prefix: &str) -> Result<()> {
        for (name, owner) in &self.operands {
            owner.inventory(inventory, &format!("{prefix}.{name}"))?;
        }
        Ok(())
    }
}

/// Delivers target, homogeneous expert sets (NVFP4 0..20 + MXFP8 20..24), optional drafter,
/// and optional vision tower in ONE standalone coordinator push transaction (when inside a
/// SparkNest push namespace) or via GPU-streaming mount outside push namespaces.
pub(crate) fn deliver_alone<'a>(
    device: Device<'a>,
    view: &cuteafd_loader::plan::checkpoint::CheckpointView,
    checkpoint: &Checkpoint,
    catalog: &cuteafd_loader::OfficialV41Catalog,
    inputs: &MiniInputs,
    admitted: &AdmittedMini,
    media: crate::shared::vision::push::Contribution,
) -> Result<(
    MiniTargetWeights<'a>,
    Option<Vec<crate::shared::experts::rtx::push::PushedExperts<'a>>>,
    Option<crate::shared::speculation::recurrent_weights::Weights<'a>>,
)> {
    use crate::shared::experts::{
        layer::ExpertLayer,
        rtx::push::{CoordinatorExperts, CoordinatorPush, PushPolicy, WholeOwner},
    };
    let check = |layout: &WeightArenaLayout| -> Result<()> {
        let planned = admitted
            .solved
            .facts
            .allocations
            .iter()
            .find(|owner| owner.id == layout.id())
            .with_context(|| format!("Inkling 2 Mini weight arena absent from admission: {}", layout.id()))?;
        ensure!(
            planned.bytes == layout.bytes() as u64
                && planned.location == cuteafd_loader::placement::planner::Location::coordinator(0),
            "Inkling 2 Mini weight backing differs from solved admission: {}",
            layout.id()
        );
        Ok(())
    };
    check(&inputs.target.execution_arena()?)?;
    if let Some(draft) = &inputs.drafter {
        check(&draft.layout.weights[0].arena)?;
    }

    let policy = PushPolicy::for_snapshot(view.snapshot())?;
    if policy == PushPolicy::Mount {
        let target = MiniTargetWeights::stream(device, checkpoint, &inputs.target)?;
        let draft = if let Some(draft) = &inputs.drafter {
            let source = cuteafd_loader::formats::block_dspark::recurrent::loading::RecurrentSource::open(view)?;
            Some(crate::shared::speculation::recurrent_weights::Weights::load(
                device,
                &source,
                draft.layout.weights[0].clone(),
                std::ptr::null_mut(),
            )?)
        } else {
            None
        };
        media.commit()?;
        return Ok((target, None, draft));
    }

    let LocalPackage::Installed { backings } = &inputs.local else {
        anyhow::bail!("Inkling 2 Mini local expert packages missing");
    };
    let sets = backings
        .iter()
        .map(|backing| CoordinatorExperts::Whole {
            owners: vec![WholeOwner {
                rank: 0,
                budget: backing.arena.bytes(),
                selections: backing
                    .layers
                    .iter()
                    .map(|&layer| ExpertLayer::BackboneFull { layer })
                    .collect(),
            }],
        })
        .collect::<Vec<_>>();
    let experts = Some(CoordinatorExperts::Sets(sets));
    let ((target, draft), mut records) = crate::shared::experts::rtx::dense::recording(|| {
        let target = MiniTargetWeights::stream(device, checkpoint, &inputs.target)?;
        let draft = if let Some(draft) = &inputs.drafter {
            let source = cuteafd_loader::formats::block_dspark::recurrent::loading::RecurrentSource::open(view)?;
            Some(crate::shared::speculation::recurrent_weights::Weights::load(
                device,
                &source,
                draft.layout.weights[0].clone(),
                std::ptr::null_mut(),
            )?)
        } else {
            None
        };
        Ok((target, draft))
    })?;
    records.extend(media.records());
    let dense = crate::shared::experts::rtx::dense::plan_records(view.snapshot(), records, &[device], 0)?;
    let push = CoordinatorPush::prepare_alone_view(view, Some(catalog), 1, experts)?;
    let (received, ()) = push.receive_sets_with(vec![device], Some(catalog), 0, Some(Ok(dense)), || ());
    let mut pushed = PushPolicy::Required
        .committed(received)?
        .context("Inkling 2 Mini standalone push group did not commit")?;
    media.commit()?;
    ensure!(pushed.len() == 1, "Inkling 2 Mini standalone push returned wrong rank count");
    Ok((target, Some(pushed.remove(0)), draft))
}
