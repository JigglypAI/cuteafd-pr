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
    fn packed(&self) -> Option<(&UnitOwner<'a>, &super::cache::CacheBank<'a>, usize, &[PackedMember])> { None }
    fn prepare_page_table(&self, _pages: &[i32]) -> Result<()> { anyhow::bail!("Inkling owner has no packed page-table uploader") }
}

#[derive(Clone)]
pub(crate) struct PackedMember {
    pub sequence: usize,
    pub rows: std::ops::Range<usize>,
    pub metadata: cuteafd_loader::families::inkling2::execution::UnitMetadata,
}

pub(crate) struct PackedLayerUnit<'u, 'a> {
    pub unit: &'u UnitOwner<'a>,
    pub cache: &'u super::cache::CacheBank<'a>,
    pub layer: usize,
    pub members: &'u [PackedMember],
    pub uploads: std::cell::RefCell<&'u mut crate::shared::token_table::RetainedUploads<'a>>,
}
impl<'a> UnitBuffers<'a> for PackedLayerUnit<'_, 'a> {
    fn stream(&self) -> &Stream<'a> { &self.unit.stream }
    fn layout(&self) -> &UnitLayout { &self.unit.layout }
    fn buffer(&self, name: &str) -> Result<Buffer> {
        match name {
            "global_cache" => self.cache.global(self.layer),
            "cache_slots" if self.cache.layout.global_layers.contains(&self.layer) => self.unit.buffer("global_slots"),
            _ => self.unit.buffer(name),
        }
    }
    fn packed(&self) -> Option<(&UnitOwner<'a>, &super::cache::CacheBank<'a>, usize, &[PackedMember])> {
        Some((self.unit, self.cache, self.layer, self.members))
    }
    fn prepare_page_table(&self, pages: &[i32]) -> Result<()> {
        let bytes = pages.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>();
        // SAFETY: the retained upload owner fences its immutable source; the
        // per-sequence cache launch consumes this table before the next upload.
        unsafe { self.uploads.borrow_mut().upload(self.unit.buffer("page_table")?, &bytes, self.unit.stream.raw) }
    }
}

pub(crate) struct LayerUnit<'u, 'a> {
    unit: &'u UnitOwner<'a>,
    cache: &'u super::cache::CacheBank<'a>,
    sequence: usize,
    layer: usize,
    rows: std::ops::Range<usize>,
}
impl<'a> UnitBuffers<'a> for LayerUnit<'_, 'a> {
    fn stream(&self) -> &Stream<'a> { &self.unit.stream }
    fn layout(&self) -> &UnitLayout { &self.unit.layout }
    fn buffer(&self, name: &str) -> Result<Buffer> {
        match name {
            "ring" | "attn_conv_history" | "mlp_conv_history" | "k_conv_history" | "v_conv_history" =>
                self.cache.field(self.sequence, self.layer, name),
            "global_cache" => self.cache.global(self.layer),
            "cache_slots" if self.cache.layout.global_layers.contains(&self.layer) => self.row_buffer("global_slots"),
            "attention_scratch" | "page_table" => self.unit.buffer(name),
            _ => self.row_buffer(name),
        }
    }
}

impl LayerUnit<'_, '_> {
    fn row_buffer(&self, name: &str) -> Result<Buffer> {
        let buffer = self.unit.buffer(name)?;
        let parts = match name { "residual_arena" => self.unit.layout.blocks, "global_mixes" => 3, _ => 1 };
        let stride = buffer.bytes.checked_div(self.unit.layout.rows * parts).context("Inkling row-plane stride")?;
        ensure!(stride > 0 && buffer.bytes == self.unit.layout.rows * parts * stride,
            "Inkling row plane has a fractional stride");
        // Multi-plane residual/global mixes retain their original plane pitch;
        // per-segment arithmetic receives that pitch explicitly.
        let offset = self.rows.start.checked_mul(stride).context("Inkling row-plane offset")?;
        let bytes = if parts == 1 { self.rows.len()*stride } else { buffer.bytes-offset };
        super::cache::view(buffer, offset, bytes)
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
        Ok(LayerUnit { unit: self, cache, sequence, layer, rows: 0..self.layout.rows })
    }
    pub fn segment<'u>(&'u self, cache: &'u super::cache::CacheBank<'a>, sequence: usize,
        layer: usize, rows: std::ops::Range<usize>) -> Result<LayerUnit<'u, 'a>> {
        ensure!(rows.start < rows.end && rows.end <= self.layout.rows, "Inkling segment outside unit");
        let mut unit = self.bind(cache, sequence, layer)?;
        unit.rows = rows;
        Ok(unit)
    }
    pub fn segment_metadata(&self, cache: &super::cache::CacheBank<'a>, sequence: usize, layer: usize,
        rows: std::ops::Range<usize>, metadata: &cuteafd_loader::families::inkling2::execution::UnitMetadata,
        uploads: &mut crate::shared::token_table::RetainedUploads<'a>) -> Result<()> {
        ensure!(metadata.rows() == rows.len(), "Inkling segment metadata row count differs");
        let segment = self.segment(cache, sequence, layer, rows)?;
        self.stream.device.run(|| {
            for (name, values) in [("positions", &metadata.positions), ("cache_slots", &metadata.local_slots),
                ("global_slots", &metadata.global_slots), ("ring_slots", &metadata.ring_slots),
                ("conv_slots", &metadata.conv_slots)] {
                let bytes = values.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>();
                // SAFETY: the segment borrows this owner; shared pinned source
                // fences retain each immutable upload through its reader.
                unsafe { uploads.upload(segment.row_buffer(name)?, &bytes, self.stream.raw)?; }
            }
            for (name, values) in [("first", &metadata.first), ("page_table", &metadata.page_table)] {
                if name == "page_table" && !cache.layout.global_layers.contains(&layer) { continue; }
                let bytes = values.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>();
                let destination = if name == "page_table" { self.buffer(name)? } else { segment.row_buffer(name)? };
                // SAFETY: same retained source and issuing-stream ownership.
                unsafe { uploads.upload(destination, &bytes, self.stream.raw)?; }
            }
            Ok(())
        })
    }
    pub fn packed_metadata(&self, layer: usize, cache: &super::cache::CacheBank<'a>, members: &[PackedMember],
        uploads: &mut crate::shared::token_table::RetainedUploads<'a>) -> Result<()> {
        let rows = members.last().context("Inkling packed metadata is empty")?.rows.end;
        let mut positions = Vec::with_capacity(rows);
        let mut local = Vec::with_capacity(rows);
        let mut global = Vec::with_capacity(rows);
        let mut ring = Vec::with_capacity(rows);
        let mut conv = Vec::with_capacity(rows);
        let mut first = Vec::with_capacity(rows);
        for member in members {
            ensure!(member.rows.start == positions.len() && member.metadata.rows() == member.rows.len(),
                "Inkling packed metadata members do not cover unit rows");
            positions.extend_from_slice(&member.metadata.positions);
            local.extend_from_slice(&member.metadata.local_slots);
            global.extend_from_slice(&member.metadata.global_slots);
            ring.extend_from_slice(&member.metadata.ring_slots);
            conv.extend_from_slice(&member.metadata.conv_slots);
            first.extend(std::iter::repeat_n(0i32, member.rows.len()));
        }
        ensure!(rows <= self.layout.rows && layer < cache.layout.layers.len(), "Inkling packed metadata extent differs");
        for (name, values) in [("positions", positions), ("cache_slots", local), ("global_slots", global),
            ("ring_slots", ring), ("conv_slots", conv)] {
            let bytes = values.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>();
            // SAFETY: each whole-row destination is admitted and the immutable
            // packed source stays fenced through every arithmetic consumer.
            unsafe { uploads.upload(self.buffer(name)?, &bytes, self.stream.raw)?; }
        }
        let bytes = first.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>();
        // SAFETY: same whole-unit metadata source lifetime.
        unsafe { uploads.upload(self.buffer("first")?, &bytes, self.stream.raw)?; }
        Ok(())
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
    pub fn uploads(&self, cfg: &cuteafd_loader::families::inkling2::Inkling2Config, members: usize)
        -> Result<crate::shared::token_table::RetainedUploads<'a>> {
        crate::shared::token_table::RetainedUploads::new(self.stream.device, self.layout.upload_shape(cfg, members)?)
    }
    pub fn inventory(&self, inventory: &mut Inventory<'_>, prefix: &str) -> Result<()> {
        for (name, owner) in &self.buffers { owner.inventory(inventory, &format!("{prefix}.{name}"))?; }
        Ok(())
    }
}
/// Runtime construction is driven by an already solved shared recipe. No
/// family pool/onboarding policy or allocation-before-admission lives here.
pub(crate) struct RequestOwners<'a> {
    pub decode_uploads: crate::shared::token_table::RetainedUploads<'a>,
    pub prefill_uploads: Vec<crate::shared::token_table::RetainedUploads<'a>>,
    pub decode: UnitOwner<'a>,
    pub prefill: Vec<UnitOwner<'a>>,
    pub cache: Rc<super::cache::CacheBank<'a>>,
}
impl<'a> RequestOwners<'a> {
    pub fn allocation_receipts(&self, inventory: &mut Inventory<'_>,
        census: &mut crate::shared::memory::receipts::Census,
        devices: cuteafd_loader::placement::planner::CoordinatorDevices<'_>) -> Result<()> {
        self.cache.inventory(inventory)?;
        self.decode.inventory(inventory, "inkling2.decode.0")?;
        self.decode_uploads.allocation_receipts(inventory, "inkling2.decode.0.uploads")?;
        for (slot, (owner, uploads)) in self.prefill.iter().zip(&self.prefill_uploads).enumerate() {
            let prefix = format!("inkling2.unit.0.{slot}");
            owner.inventory(inventory, &prefix)?;
            uploads.allocation_receipts(inventory, &format!("{prefix}.uploads"))?;
        }
        for (name, uploads) in std::iter::once(("inkling2.decode.0.uploads".to_owned(), &self.decode_uploads))
            .chain(self.prefill_uploads.iter().enumerate().map(|(slot, uploads)| (format!("inkling2.unit.0.{slot}.uploads"), uploads))) {
            let backing = inventory.receipts.iter().find(|receipt| receipt.allocation.id.ends_with(&format!(".{name}"))
                && receipt.allocation.bytes == uploads.mapped_alias().bytes as u64
                && receipt.allocation.location == cuteafd_loader::placement::planner::Location::PINNED)
                .with_context(|| format!("Inkling upload alias has no pinned owner {name}"))?.allocation.id.clone();
            census.mapped(format!("{name}.mapped"), backing, uploads.mapped_alias(), devices)?;
        }
        Ok(())
    }
    pub fn whole(stream: Rc<Stream<'a>>, inputs: &cuteafd_loader::placement::families::inkling2::InklingInputs,
        admitted: &cuteafd_loader::placement::families::inkling2::AdmittedInkling) -> Result<Self> {
        use cuteafd_loader::placement::{LayerMode, planner::Location};
        ensure!(inputs.gpus == 1 && admitted.placement.layers.iter().all(|layer|
            matches!(layer.mode, LayerMode::Whole { gpu: 0, .. })), "Inkling request owner requires admitted whole rank");
        let cfg = &inputs.cfg;
        let sequences = usize::try_from(inputs.sequences)?;
        let context = usize::try_from(inputs.context.maximum)?;
        let pages = usize::try_from(admitted.solved.pool_tokens.div_ceil(64))?;
        let cache_layout = cuteafd_loader::families::inkling2::execution::CacheLayout::new(cfg, sequences, pages)?;
        let mark_slots = usize::try_from(cuteafd_core::prefix::mark_slots_with_host(inputs.sequences, inputs.prefix_entries,
            cache_layout.sequence_bytes as u64, inputs.prefix_mark_budget, inputs.host_prefix.quota_bytes > 0))?;
        let decode_layout = UnitLayout::serving(cfg, 64, context)?;
        let prefill_layout = UnitLayout::serving(cfg, usize::try_from(inputs.prefill.rows)?, context)?;
        let owner = |id: &str, bytes: usize, location| -> Result<()> {
            let allocation = admitted.solved.facts.allocations.iter().find(|allocation| allocation.id == id)
                .with_context(|| format!("Inkling unadmitted runtime owner {id}"))?;
            ensure!(allocation.location == location && allocation.bytes == bytes as u64,
                "Inkling runtime owner differs from admission: {id} {bytes}/{}", allocation.bytes);
            Ok(())
        };
        let device = Location::coordinator(0);
        owner("inkling2.cache.0.active", cache_layout.active_bytes()?, device)?;
        if mark_slots > 0 { owner("inkling2.cache.0.marks", cache_layout.retained_bytes(mark_slots)?, device)?; }
        for &layer in &cache_layout.global_layers {
            let bytes = usize::try_from(cuteafd_loader::placement::planner::sm120_request_backing(cache_layout.layer_page_bytes()? as u64)?)?;
            owner(&format!("inkling2.kv.{layer}.0.records"), bytes, device)?;
        }
        for (prefix, layout) in std::iter::once(("inkling2.decode.0".to_owned(), &decode_layout))
            .chain((0..inputs.prefill.waves).map(|slot| (format!("inkling2.unit.0.{slot}"), &prefill_layout))) {
            for (name, bytes) in &layout.buffers { owner(&format!("{prefix}.{name}"), *bytes, device)?; }
            let uploads = layout.upload_shape(cfg, sequences.min(layout.rows))?;
            owner(&format!("{prefix}.uploads"), uploads.capacity_bytes()?, Location::PINNED)?;
        }
        // Every extent was checked before the first allocation.
        let cache = Rc::new(super::cache::CacheBank::new(vec![stream.clone()], cache_layout, mark_slots)?);
        let decode = UnitOwner::new(stream.clone(), decode_layout)?;
        let prefill = (0..inputs.prefill.waves).map(|_| UnitOwner::new(stream.clone(), prefill_layout.clone()))
            .collect::<Result<Vec<_>>>()?;
        let decode_uploads = decode.uploads(cfg, sequences.min(decode.layout.rows))?;
        let prefill_uploads = prefill.iter().map(|owner| owner.uploads(cfg, sequences.min(owner.layout.rows)))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { decode_uploads, prefill_uploads, decode, prefill, cache })
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
