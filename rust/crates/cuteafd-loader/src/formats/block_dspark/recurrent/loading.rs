//! Header-native recurrent drafter sources and rank-local physical arenas.
//! Native bundles have an unindexed bare-name file; MC bundles index `mtp.*`.
use super::{rank_recipe, source_shapes, RecurrentBlockConfig};
use anyhow::{ensure, Context, Result};
use crate::{formats::{dense_trellis::TrellisSide, trellis_core::{CoreContract, CoreSplit, CoreWeightLayout}},
    families::glm5::draft_physical::ShardAxis,
    plan::checkpoint::{Checkpoint, CheckpointTensor, CheckpointView}, serving_capacity::WeightArenaLayout};
use cuteafd_core::DType;
use std::{cell::RefCell, collections::BTreeMap};

#[derive(Debug)]
pub struct RecurrentSource {
    pub config: RecurrentBlockConfig,
    pub checkpoint: Checkpoint,
    pub prefix: String,
}

impl RecurrentSource {
    pub fn mount_load(&self, ranks: usize) -> Result<crate::shared::dense_read::MountLoad> {
        crate::shared::dense_read::MountLoad::from_headers(self.checkpoint.tensors.iter().map(|t| &t.meta), ranks)
    }

    /// Does not read target/expert payloads or require our tooling's side files.
    pub fn open(view: &CheckpointView) -> Result<Self> {
        let config = RecurrentBlockConfig::from_hf(&view.config()?)?;
        let mut checkpoint = Checkpoint::open_view_for_role(view, "recurrent-draft", |name| name.starts_with("mtp."))?;
        let indexed = checkpoint.weight_map.keys().any(|name| name.starts_with("mtp."));
        let prefix = if indexed { "mtp.".to_owned() } else {
            let shard = "mtp.safetensors";
            let headers = crate::read_safetensors_metadata(&view.snapshot().join(shard))?;
            ensure!(!headers.is_empty(), "empty bundled drafter file");
            checkpoint.tensors = headers.into_iter().map(|meta| CheckpointTensor { shard: shard.into(), meta }).collect();
            checkpoint.tensors.sort_by(|a, b| a.meta.name.cmp(&b.meta.name));
            for tensor in &checkpoint.tensors {
                ensure!(checkpoint.weight_map.insert(tensor.meta.name.clone(), shard.into()).is_none(),
                    "unindexed drafter name collides with target tensor {}", tensor.meta.name);
            }
            String::new()
        };
        Ok(Self { config, checkpoint, prefix })
    }

    pub fn name(&self, relative: &str) -> String { format!("{}{relative}", self.prefix) }

    /// Validate all selected operands before admitting or allocating a backing.
    pub fn rank(&self, rank: usize, ranks: usize, lead: usize) -> Result<RecurrentWeightLayout> {
        self.config.rank(rank, ranks, lead)?;
        let requests = RefCell::new(Vec::new());
        let core = RefCell::new(BTreeMap::new());
        rank_recipe(&self.config, rank, ranks, lead, |relative, shape| {
            let name = self.name(relative);
            let header = self.checkpoint.header(&name).with_context(|| format!("drafter lacks {name}"))?;
            let bytes = shape.iter().try_fold(2usize, |n, d| n.checked_mul(*d)).context("draft bytes overflow")?;
            ensure!(header.dtype == DType::Bf16 && header.shape == shape && header.byte_length == bytes as u64,
                "drafter {name}: expected BF16 {shape:?}/{bytes} bytes, found {:?} {:?}/{}",
                header.dtype, header.shape, header.byte_length);
            requests.borrow_mut().push((name, bytes));
            Ok(())
        }, |parts, columns, axis| {
            for &(relative, rows) in parts {
                let name = self.name(relative);
                let stem = name.strip_suffix(".weight").context("draft matrix suffix")?;
                let (row, column) = super::super::block_dspark_part_ranges(rows, columns, axis, rank, ranks)?;
                if self.checkpoint.contains_tensor(&format!("{stem}.trellis")) {
                    let side = if matches!(axis, Some(ShardAxis::Columns)) { TrellisSide::Input } else { TrellisSide::Output };
                    let split = match axis { Some(ShardAxis::Rows) => CoreSplit::Output,
                        Some(ShardAxis::Columns) => CoreSplit::Input, None => CoreSplit::Whole };
                    let mut projections = crate::formats::trellis_core::read_projections(&self.checkpoint,
                        &BTreeMap::from([(name.clone(), vec![rows, columns])]),
                        |member| member.strip_prefix(stem).is_some_and(|tail| tail.starts_with('.')),
                        |candidate| (candidate == stem).then_some(CoreContract { family: "recurrent block drafter", widths: &[6, 8],
                            split, padded_outputs: false, view_varying: true }))?;
                    let projection = projections.remove(stem).context("draft compact headers missing")?;
                    let local = CoreWeightLayout::new(&projection, side,
                        if axis.is_some() { rank } else { 0 }, if axis.is_some() { ranks } else { 1 })?;
                    requests.borrow_mut().extend(local.arena_tensors(stem)?);
                    core.borrow_mut().insert(name, local);
                } else {
                    let header = self.checkpoint.header(&name).with_context(|| format!("drafter lacks {name}"))?;
                    let source_bytes = rows.checked_mul(columns).and_then(|n| n.checked_mul(2)).context("draft source bytes overflow")?;
                    ensure!(header.dtype == DType::Bf16 && header.shape == [rows, columns]
                        && header.byte_length == source_bytes as u64, "drafter {name}: expected BF16 [{rows},{columns}]");
                    let bytes = row.len().checked_mul(column.len()).and_then(|n| n.checked_mul(2))
                        .context("draft rank projection bytes overflow")?;
                    requests.borrow_mut().push((name, bytes));
                }
            }
            Ok(())
        })?;
        // Folded weights are generated once, but share the drafter backing and
        // admission. They are not a second charge for the borrowed target head.
        let native = super::native_rank_arena(&self.config, rank, ranks, lead)?;
        let mut requests = requests.into_inner();
        requests.extend(native.views().iter().filter(|(name, _)| name.starts_with("derived."))
            .map(|(name, view)| (name.clone(), view.bytes())));
        let result = RecurrentWeightLayout { rank, ranks, lead,
            arena: WeightArenaLayout::new(format!("block-dspark/recurrent/weights/{rank}"), requests)?,
            core: core.into_inner() };
        self.validate_inventory()?;
        Ok(result)
    }

    fn validate_inventory(&self) -> Result<()> {
        let expected = source_shapes(&self.config)?;
        for tensor in &self.checkpoint.tensors {
            let Some(relative) = tensor.meta.name.strip_prefix(&self.prefix) else { continue };
            if expected.contains_key(relative) { continue; }
            let valid_member = ["trellis", "suh", "svh", "mcg"].iter().any(|suffix|
                relative.strip_suffix(&format!(".{suffix}")).is_some_and(|stem| expected.contains_key(&format!("{stem}.weight"))));
            ensure!(valid_member, "unknown recurrent drafter tensor {}", tensor.meta.name);
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct RecurrentWeightLayout {
    pub rank: usize,
    pub ranks: usize,
    pub lead: usize,
    pub arena: WeightArenaLayout,
    pub core: BTreeMap<String, CoreWeightLayout>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::testing::{t, write_safetensors};
    use crate::plan::checkpoint::CoreView;

    fn fixture(root: &std::path::Path, indexed: bool) -> Result<CheckpointView> {
        let raw: serde_json::Value = serde_json::from_str(include_str!("../../../families/inkling2/fixtures/config.json"))?;
        std::fs::write(root.join("config.json"), serde_json::to_vec(&raw)?)?;
        let config = RecurrentBlockConfig::from_hf(&raw)?;
        let tensors = source_shapes(&config)?.into_iter().map(|(name, shape)|
            t(if indexed { format!("mtp.{name}") } else { name }, "BF16", &shape)).collect::<Vec<_>>();
        // Header fixtures use sparse payloads and never touch model bytes.
        write_safetensors(&root.join("mtp.safetensors"), &tensors);
        let weight_map = if indexed { tensors.iter().map(|(name, ..)| (name.clone(), "mtp.safetensors"))
            .collect::<BTreeMap<_, _>>() } else { BTreeMap::new() };
        std::fs::write(root.join("model.safetensors.index.json"), serde_json::to_vec(&serde_json::json!({"weight_map":weight_map}))?)?;
        CheckpointView::select(root, CoreView::Original)
    }

    #[test]
    fn indexed_mc_and_unindexed_native_have_same_rank_arena() -> Result<()> {
        let native = tempfile::tempdir()?;
        let indexed = tempfile::tempdir()?;
        let native = RecurrentSource::open(&fixture(native.path(), false)?)?;
        let indexed = RecurrentSource::open(&fixture(indexed.path(), true)?)?;
        for ranks in [1, 2] {
            for rank in 0..ranks {
                let a = native.rank(rank, ranks, ranks-1)?;
                let b = indexed.rank(rank, ranks, ranks-1)?;
                assert_eq!(a.arena.bytes(), b.arena.bytes());
                for (name, view) in a.arena.views() {
                    assert_eq!(*view, b.arena.view(&if name.starts_with("derived.") { name.clone() } else { format!("mtp.{name}") }).unwrap());
                }
            }
        }
        Ok(())
    }
}
