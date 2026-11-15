//! Named cold-forward working owners; admission provides the exact layout.
//! This is not a serving cache: each layer's histories are independently
//! retained by the serving cache when that stage lands.
use crate::shared::memory::{device::{Allocation, Stream}, receipts::Inventory};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::CuteafdDeviceBuffer as Buffer;
use cuteafd_core::memory_layout::Tag;
use cuteafd_loader::families::inkling2::execution::UnitLayout;
use std::{collections::BTreeMap, rc::Rc};

/// Arithmetic borrows storage; request state never becomes unit scratch.
pub(crate) trait UnitBuffers<'a> {
    fn stream(&self) -> &Stream<'a>;
    fn layout(&self) -> &UnitLayout;
    fn buffer(&self, name: &str) -> Result<Buffer>;
}

pub(crate) struct LayerUnit<'u, 'a> {
    unit: &'u UnitOwner<'a>,
    cache: &'u super::cache::CacheBank<'a>,
    sequence: usize,
    layer: usize,
}
impl<'a> UnitBuffers<'a> for LayerUnit<'_, 'a> {
    fn stream(&self) -> &Stream<'a> { &self.unit.stream }
    fn layout(&self) -> &UnitLayout { &self.unit.layout }
    fn buffer(&self, name: &str) -> Result<Buffer> {
        match name {
            "ring" | "attn_conv_history" | "mlp_conv_history" | "k_conv_history" | "v_conv_history" =>
                self.cache.field(self.sequence, self.layer, name),
            "global_cache" => self.cache.global(self.layer),
            "cache_slots" if self.cache.layout.global_layers.contains(&self.layer) => self.unit.buffer("global_slots"),
            _ => self.unit.buffer(name),
        }
    }
}

pub(crate) struct UnitOwner<'a> {
    pub stream: Rc<Stream<'a>>,
    pub layout: UnitLayout,
    buffers: BTreeMap<String, Allocation<'a>>,
}
impl<'a> UnitOwner<'a> {
    pub fn bind<'u>(&'u self, cache: &'u super::cache::CacheBank<'a>, sequence: usize,
        layer: usize) -> Result<LayerUnit<'u, 'a>> {
        ensure!(sequence < cache.layout.sequences && layer < cache.layout.layers.len(),
            "Inkling persistent layer binding outside cache");
        cache.require_reader(&self.stream)?;
        for name in ["ring", "global_cache", "attn_conv_history", "mlp_conv_history",
            "k_conv_history", "v_conv_history"] {
            ensure!(!self.layout.buffers.contains_key(name), "Inkling serving scratch duplicates persistent {name}");
        }
        ensure!(self.layout.buffers.contains_key("global_slots"), "Inkling serving unit lacks physical page slots");
        Ok(LayerUnit { unit: self, cache, sequence, layer })
    }
    /// Queues metadata through the shared completion-held pinned arena. The
    /// caller admits its shape and never changes the unit while consumers run.
    pub fn metadata(&self, metadata: &cuteafd_loader::families::inkling2::execution::UnitMetadata,
        uploads: &mut crate::shared::token_table::RetainedUploads<'a>) -> Result<()> {
        ensure!(metadata.rows() <= self.layout.rows && self.layout.buffers.contains_key("global_slots"),
            "Inkling metadata does not fit serving unit");
        self.stream.device.run(|| {
            for (name, values) in [("positions", &metadata.positions), ("cache_slots", &metadata.local_slots),
                ("global_slots", &metadata.global_slots), ("ring_slots", &metadata.ring_slots),
                ("conv_slots", &metadata.conv_slots)] {
                let bytes = values.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>();
                // SAFETY: unit and upload arena remain owned until their drains;
                // all writes and arithmetic consumers use this unit's stream.
                unsafe { uploads.upload(self.buffer(name)?, &bytes, self.stream.raw)?; }
            }
            for (name, values) in [("first", &metadata.first), ("page_table", &metadata.page_table)] {
                let bytes = values.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>();
                // SAFETY: same retained-source and ordered-destination contract.
                unsafe { uploads.upload(self.buffer(name)?, &bytes, self.stream.raw)?; }
            }
            Ok(())
        })
    }
    /// Caller admits each named physical request and its allocation tail first.
    pub fn new(stream: Rc<Stream<'a>>, layout: UnitLayout) -> Result<Self> {
        let buffers = layout.buffers.iter().map(|(name, bytes)| Ok((name.clone(),
            Allocation::new(stream.device, *bytes, Tag::workspace("inkling2/unit"))?)))
            .collect::<Result<BTreeMap<_, _>>>()?;
        let owner = Self { stream, layout, buffers };
        owner.zero()?;
        Ok(owner)
    }
    pub fn buffer(&self, name: &str) -> Result<cuteafd_ffi::CuteafdDeviceBuffer> {
        Ok(self.buffers.get(name).with_context(|| format!("Inkling unit lacks owner {name}"))?.buffer)
    }
    pub fn zero(&self) -> Result<()> {
        self.stream.device.run(|| {
            for buffer in self.buffers.values() {
                // SAFETY: these independently retained fresh owners are not yet
                // published to readers; all writes use the retained unit stream.
                unsafe { self.stream.device.library.cuda_zero_bytes_async(buffer.buffer, buffer.buffer.bytes,
                    self.stream.raw)?; }
            }
            Ok(())
        })
    }
    pub fn inventory(&self, inventory: &mut Inventory<'_>, prefix: &str) -> Result<()> {
        for (name, owner) in &self.buffers { owner.inventory(inventory, &format!("{prefix}.{name}"))?; }
        Ok(())
    }
}
impl<'a> UnitBuffers<'a> for UnitOwner<'a> {
    fn stream(&self) -> &Stream<'a> { &self.stream }
    fn layout(&self) -> &UnitLayout { &self.layout }
    fn buffer(&self, name: &str) -> Result<Buffer> { UnitOwner::buffer(self, name) }
}
impl Drop for UnitOwner<'_> {
    fn drop(&mut self) { self.stream.drain_or_abort("Inkling unit owners"); }
}
