//! Rank-local recurrent dSpark operands use the same recorded dense delivery as
//! target weights. Recording never reads native payloads through the mount.
use crate::shared::{dense_trellis::{CoreWeightLayout, CoreWeights},
    experts::rtx::dense::{DenseLoad, Operand, Rect},
    memory::{device::{Allocation, Device}, receipts::Inventory, weight_arena::DeviceWeightArena}};
use anyhow::{ensure, Context, Result};
use cuteafd_core::memory_layout::ArenaTag;
use cuteafd_loader::formats::block_dspark::recurrent::{rank_recipe, RecurrentWeights,
    loading::{RecurrentSource, RecurrentWeightLayout}};
use std::{collections::BTreeMap, ffi::c_void};

pub(crate) enum ProjectionWeight<'a> {
    Bf16(Allocation<'a>),
    Trellis(CoreWeights<'a>),
}

pub(crate) struct Weights<'a> {
    pub layout: RecurrentWeightLayout,
    pub tensors: RecurrentWeights<Allocation<'a>, Vec<ProjectionWeight<'a>>>,
    pub derived: BTreeMap<String, Allocation<'a>>,
}

impl<'a> Weights<'a> {
    /// `layout` was validated by admission. The caller wraps target+drafter
    /// together in dense::recording and publishes only after group commit.
    pub fn load(device: Device<'a>, source: &RecurrentSource, layout: RecurrentWeightLayout,
        stream: *mut c_void) -> Result<Self> {
        let checked = source.rank(layout.rank, layout.ranks, layout.lead)?;
        ensure!(checked.arena == layout.arena && checked.core == layout.core,
            "recurrent draft arena differs from header-validated admission");
        let dense = DenseLoad::mount(device.library);
        dense.admit_mount(source.mount_load(layout.ranks)?)?;
        let arena = DeviceWeightArena::new(device, layout.arena.clone(),
            ArenaTag::drafter_weights("shared/recurrent-dspark"))?;
        let tensors = rank_recipe(&source.config, layout.rank, layout.ranks, layout.lead,
            |relative, _| {
                let name = source.name(relative);
                let rect = Rect::tensor(&source.checkpoint, &name, None, None)?;
                let bytes = rect.bytes();
                let owner = arena.view(&name, bytes)?;
                dense.operand(Operand { device: device.id, buffer: owner.buffer,
                    lands: vec![(rect, 0)], bytes })?;
                Ok(owner)
            }, |parts, columns, axis| {
                parts.iter().map(|&(relative, rows)| {
                    let name = source.name(relative);
                    if let Some(core) = layout.core.get(&name) {
                        let core = CoreWeightLayout::from_shared(core.clone());
                        return core.load_recorded_with(device, &source.checkpoint, &dense, stream,
                            |name, bytes| arena.view(name, bytes)).map(ProjectionWeight::Trellis);
                    }
                    let (rows, cols) = cuteafd_loader::formats::block_dspark::block_dspark_part_ranges(
                        rows, columns, axis, layout.rank, layout.ranks)?;
                    let rect = Rect::tensor(&source.checkpoint, &name, Some(rows), Some(cols.start*2..cols.end*2))?;
                    let bytes = rect.bytes();
                    let owner = arena.view(&name, bytes)?;
                    dense.operand(Operand { device: device.id, buffer: owner.buffer,
                        lands: vec![(rect, 0)], bytes })?;
                    Ok(ProjectionWeight::Bf16(owner))
                }).collect()
            })?;
        let derived = layout.arena.views().iter().filter(|(name, _)| name.starts_with("derived."))
            .map(|(name, view)| Ok((name.clone(), arena.view(name, view.bytes())?)))
            .collect::<Result<BTreeMap<_, _>>>()?;
        dense.release();
        Ok(Self { layout, tensors, derived })
    }

    pub fn derived(&self, name: &str) -> Result<&Allocation<'a>> {
        self.derived.get(name).with_context(|| format!("recurrent draft lacks generated view {name}"))
    }

    pub fn inventory(&self, inventory: &mut Inventory<'_>, prefix: &str) -> Result<()> {
        let tensor = |inventory: &mut Inventory<'_>, name: &str, owner: &Allocation<'_>| {
            owner.inventory(inventory, &format!("{prefix}.{name}"))
        };
        let matrix = |inventory: &mut Inventory<'_>, name: &str, owners: &[ProjectionWeight<'_>]| -> Result<()> {
            for (part, weight) in owners.iter().enumerate() {
                match weight {
                    ProjectionWeight::Bf16(owner) => tensor(inventory, &format!("{name}.{part}"), owner)?,
                    ProjectionWeight::Trellis(owner) => {
                        for (suffix, owner) in [("trellis", &owner.payload), ("suh", &owner.suh), ("svh", &owner.svh)] {
                            tensor(inventory, &format!("{name}.{part}.{suffix}"), owner)?;
                        }
                    }
                }
            }
            Ok(())
        };
        for (index, layer) in self.tensors.layers.iter().enumerate() {
            for (name, owner) in [("input_norm", &layer.input_norm), ("post_norm", &layer.post_norm),
                ("dflare_norm", &layer.dflare_norm), ("q_norm", &layer.q_norm), ("k_norm", &layer.k_norm),
                ("context_k_norm", &layer.context_k_norm)] {
                tensor(inventory, &format!("layer.{index}.{name}"), owner)?;
            }
            for (name, owners) in [("qkv", &layer.qkv), ("context_kv", &layer.context_kv),
                ("output", &layer.output), ("gate_up", &layer.gate_up), ("down", &layer.down)] {
                matrix(inventory, &format!("layer.{index}.{name}"), owners)?;
            }
        }
        for (name, owner) in [("norm", &self.tensors.norm), ("fuse", &self.tensors.fuse),
            ("mask_embedding", &self.tensors.mask_embedding)] { tensor(inventory, name, owner)?; }
        if let Some(head) = &self.tensors.head {
            for (name, owner) in [("token_embedding", &head.token_embedding), ("projection_bias", &head.projection_bias),
                ("hidden_out_bias", &head.hidden_out_bias), ("dflare_norm", &head.dflare_norm)] {
                tensor(inventory, &format!("head.{name}"), owner)?;
            }
            matrix(inventory, "head.projection", &head.projection)?;
            matrix(inventory, "head.hidden_out", &head.hidden_out)?;
        }
        for (name, owner) in &self.derived { tensor(inventory, name, owner)?; }
        Ok(())
    }
}
