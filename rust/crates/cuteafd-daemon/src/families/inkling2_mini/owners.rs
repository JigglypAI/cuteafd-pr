//! Per-unit working owners and admitted serving request bank for Inkling 2 Mini (TP1).
use crate::shared::memory::{
    device::{Allocation, Stream},
    receipts::Inventory,
};
use anyhow::{ensure, Context, Result};
use cuteafd_core::memory_layout::Tag;
use cuteafd_ffi::CuteafdDeviceBuffer as Buffer;
use cuteafd_loader::families::inkling2::execution::UnitMetadata;
use cuteafd_loader::families::inkling2_mini::execution::{
    MiniCacheLayout, MiniUnitLayout, DECODE_TILE_ROWS, PREFILL_TILE_ROWS,
};
use std::{collections::BTreeMap, rc::Rc};

pub(crate) trait UnitBuffers<'a> {
    fn stream(&self) -> &Stream<'a>;
    fn layout(&self) -> &MiniUnitLayout;
    fn buffer(&self, name: &str) -> Result<Buffer>;
    fn decode_attention(&self) -> bool {
        false
    }
    fn convolution_strides(&self) -> Option<(usize, usize, usize)> {
        None
    }
    fn convolution_replay(&self, _field: &str) -> Result<Option<(Buffer, usize)>> {
        Ok(None)
    }
    fn packed(&self) -> Option<(&UnitOwner<'a>, &super::cache::CacheBank<'a>, usize, &[PackedMember])> {
        None
    }
}

#[derive(Clone)]
pub(crate) struct PackedMember {
    pub sequence: usize,
    pub rows: std::ops::Range<usize>,
    pub metadata: UnitMetadata,
}

/// Binds one unit owner to a specific 48-slot cache slot (`0..48`) for one visit.
pub(crate) struct PackedVisitUnit<'u, 'a> {
    pub unit: &'u UnitOwner<'a>,
    pub cache: &'u super::cache::CacheBank<'a>,
    pub cache_slot: usize,
    pub members: &'u [PackedMember],
}

impl<'a> UnitBuffers<'a> for PackedVisitUnit<'_, 'a> {
    fn decode_attention(&self) -> bool {
        self.unit.decode_attention
    }

    fn convolution_replay(&self, field: &str) -> Result<Option<(Buffer, usize)>> {
        let Some((backing, width)) = self.unit.replay.get() else {
            return Ok(None);
        };
        let (range, _) = self.cache.layout.conv.field(0, 0, field)?;
        let offset = (self.cache_slot * 64 * width + range.start / 4) * 4;
        Ok(Some((super::cache::view(backing, offset, backing.bytes - offset)?, width)))
    }

    fn convolution_strides(&self) -> Option<(usize, usize, usize)> {
        Some((
            self.cache.layout.conv.slot_stride,
            self.cache.layout.conv.lag_stride,
            self.cache.layout.sequences,
        ))
    }

    fn stream(&self) -> &Stream<'a> {
        &self.unit.stream
    }
    fn layout(&self) -> &MiniUnitLayout {
        &self.unit.layout
    }

    fn buffer(&self, name: &str) -> Result<Buffer> {
        match name {
            "ring" | "attn_conv_history" | "mlp_conv_history" | "k_conv_history" | "v_conv_history" => {
                self.cache.field_bank(self.cache_slot, name)
            }
            "global_cache" => self.cache.global(self.cache_slot),
            "cache_slots" if self.cache.layout.global_slots.contains(&self.cache_slot) => {
                self.unit.buffer("global_slots")
            }
            _ => self.unit.buffer(name),
        }
    }

    fn packed(&self) -> Option<(&UnitOwner<'a>, &super::cache::CacheBank<'a>, usize, &[PackedMember])> {
        Some((self.unit, self.cache, self.cache_slot, self.members))
    }
}

pub(crate) struct VisitSegment<'u, 'a> {
    unit: &'u UnitOwner<'a>,
    cache: &'u super::cache::CacheBank<'a>,
    sequence: usize,
    cache_slot: usize,
    rows: std::ops::Range<usize>,
}

impl<'a> UnitBuffers<'a> for VisitSegment<'_, 'a> {
    fn convolution_strides(&self) -> Option<(usize, usize, usize)> {
        Some((self.cache.layout.conv.slot_stride, self.cache.layout.conv.lag_stride, 1))
    }
    fn stream(&self) -> &Stream<'a> {
        &self.unit.stream
    }
    fn layout(&self) -> &MiniUnitLayout {
        &self.unit.layout
    }
    fn buffer(&self, name: &str) -> Result<Buffer> {
        match name {
            "ring" | "attn_conv_history" | "mlp_conv_history" | "k_conv_history" | "v_conv_history" => {
                self.cache.field(self.sequence, self.cache_slot, name)
            }
            "global_cache" => self.cache.global(self.cache_slot),
            "cache_slots" if self.cache.layout.global_slots.contains(&self.cache_slot) => {
                self.row_buffer("global_slots")
            }
            "attention_scratch" | "page_table" => self.unit.buffer(name),
            _ => self.row_buffer(name),
        }
    }
}

impl VisitSegment<'_, '_> {
    fn row_buffer(&self, name: &str) -> Result<Buffer> {
        let buffer = self.unit.buffer(name)?;
        let parts = match name {
            "residual_arena" => self.unit.layout.blocks,
            "prev_pass_arena" => 2,
            "global_mixes" | "ctrl_paired" | "ctrl_rank" | "ctrl_fg" | "ctrl_out" => 3,
            "cross_pass_msg" => 4,
            _ => 1,
        };
        let stride = buffer
            .bytes
            .checked_div(self.unit.layout.rows * parts)
            .context("Inkling 2 Mini row-plane stride")?;
        ensure!(
            stride > 0 && buffer.bytes == self.unit.layout.rows * parts * stride,
            "Inkling 2 Mini row plane has fractional stride"
        );
        let offset = self
            .rows
            .start
            .checked_mul(stride)
            .context("Inkling 2 Mini row-plane offset")?;
        let bytes = if parts == 1 {
            self.rows.len() * stride
        } else {
            buffer.bytes - offset
        };
        super::cache::view(buffer, offset, bytes)
    }
}

pub(crate) struct UnitOwner<'a> {
    pub stream: Rc<Stream<'a>>,
    pub layout: MiniUnitLayout,
    buffers: BTreeMap<String, Allocation<'a>>,
    replay: std::cell::Cell<Option<(Buffer, usize)>>,
    pub decode_attention: bool,
}

impl<'a> UnitOwner<'a> {
    pub fn replaying(&self) -> bool {
        self.replay.get().is_some()
    }

    pub fn set_replay(&self, replay: Option<(Buffer, usize)>) -> Result<()> {
        ensure!(
            replay.is_none_or(|(buffer, width)| {
                width == 6656 && buffer.device_id == self.stream.device.id && buffer.bytes >= 48 * 64 * width * 4
            }),
            "Inkling 2 Mini FIR replay backing differs from admitted 48-slot width 6656"
        );
        self.replay.set(replay);
        Ok(())
    }

    pub fn bind<'u>(
        &'u self,
        cache: &'u super::cache::CacheBank<'a>,
        sequence: usize,
        cache_slot: usize,
    ) -> Result<VisitSegment<'u, 'a>> {
        ensure!(
            sequence < cache.layout.sequences && cache_slot < cache.layout.slots.len(),
            "Inkling 2 Mini persistent binding outside 48-slot cache"
        );
        cache.require_reader(&self.stream)?;
        Ok(VisitSegment {
            unit: self,
            cache,
            sequence,
            cache_slot,
            rows: 0..self.layout.rows,
        })
    }

    pub fn segment<'u>(
        &'u self,
        cache: &'u super::cache::CacheBank<'a>,
        sequence: usize,
        cache_slot: usize,
        rows: std::ops::Range<usize>,
    ) -> Result<VisitSegment<'u, 'a>> {
        ensure!(rows.start < rows.end && rows.end <= self.layout.rows, "Mini segment outside unit");
        let mut seg = self.bind(cache, sequence, cache_slot)?;
        seg.rows = rows;
        Ok(seg)
    }

    pub fn packed_metadata(
        &self,
        cache_slot: usize,
        cache: &super::cache::CacheBank<'a>,
        members: &[PackedMember],
        uploads: &mut crate::shared::token_table::RetainedUploads<'a>,
    ) -> Result<()> {
        ensure!(
            cache.layout.ring_sequence_bytes % cache.layout.ring_record_bytes() == 0,
            "Inkling 2 Mini sequence bank must have an integral record stride"
        );
        ensure!(
            members.len() <= cache.layout.sequences
                && members.iter().all(|m| m.sequence < cache.layout.sequences)
                && members.iter().map(|m| m.sequence).collect::<std::collections::BTreeSet<_>>().len() == members.len(),
            "Inkling 2 Mini packed cache members alias or exceed admitted request slots"
        );
        let rows = members.last().context("Mini packed metadata is empty")?.rows.end;
        let mut positions = Vec::with_capacity(rows);
        let mut local = Vec::with_capacity(rows);
        let mut global = Vec::with_capacity(rows);
        let mut ring = Vec::with_capacity(rows);
        let mut conv = Vec::with_capacity(rows);
        let mut first = Vec::with_capacity(rows);
        for member in members {
            ensure!(
                member.rows.start == positions.len() && member.metadata.rows() == member.rows.len(),
                "Mini packed metadata members do not cover unit rows"
            );
            positions.extend_from_slice(&member.metadata.positions);
            local.extend(member.metadata.local_slots.iter().map(|&slot| slot + member.rows.start as i64));
            global.extend_from_slice(&member.metadata.global_slots);
            ring.extend(
                member
                    .metadata
                    .ring_slots
                    .iter()
                    .map(|&slot| slot + (member.sequence * cache.layout.ring_stride_records()) as i64),
            );
            conv.extend(std::iter::repeat_n(member.sequence as i64, member.rows.len()));
            first.extend(std::iter::repeat_n(member.rows.start as i32, member.rows.len()));
        }
        ensure!(
            rows <= self.layout.rows && cache_slot < cache.layout.slots.len(),
            "Mini packed metadata extent differs"
        );
        for (name, values) in [
            ("positions", positions),
            ("cache_slots", local),
            ("global_slots", global),
            ("ring_slots", ring),
            ("conv_slots", conv),
        ] {
            let bytes = values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
            // SAFETY: admitted destinations and completion-fenced pinned upload source.
            unsafe {
                uploads.upload(self.buffer(name)?, &bytes, self.stream.raw)?;
            }
        }
        let bytes = first.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
        // SAFETY: same whole-unit metadata lifetime.
        unsafe {
            uploads.upload(self.buffer("first")?, &bytes, self.stream.raw)?;
        }
        let step = if self.decode_attention {
            DECODE_TILE_ROWS
        } else {
            PREFILL_TILE_ROWS
        };
        let tiles = members
            .iter()
            .flat_map(|m| m.rows.clone().step_by(step))
            .map(|row| row as i32)
            .collect::<Vec<_>>();
        let indices = members
            .iter()
            .enumerate()
            .flat_map(|(table, m)| std::iter::repeat_n(table as i32, m.rows.len()))
            .collect::<Vec<_>>();
        let width = self.layout.buffers["page_table"] / (cache.layout.sequences * 4);
        let mut pages = vec![0i32; members.len() * width];
        for (table, member) in members.iter().enumerate() {
            ensure!(member.metadata.page_table.len() <= width, "Mini member page table exceeds stride");
            pages[table * width..table * width + member.metadata.page_table.len()]
                .copy_from_slice(&member.metadata.page_table);
        }
        for (name, values) in [("tile_rows", tiles), ("table_indices", indices), ("page_table", pages)] {
            if name == "page_table" && !cache.layout.global_slots.contains(&cache_slot) {
                continue;
            }
            let bytes = values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
            // SAFETY: completion-fenced pinned upload source.
            unsafe {
                uploads.upload(self.buffer(name)?, &bytes, self.stream.raw)?;
            }
        }
        Ok(())
    }

    pub fn new(stream: Rc<Stream<'a>>, layout: MiniUnitLayout) -> Result<Self> {
        let buffers = layout
            .buffers
            .iter()
            .map(|(name, bytes)| {
                Ok((
                    name.clone(),
                    Allocation::new(stream.device, *bytes, Tag::workspace("inkling2_mini/unit"))?,
                ))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        let owner = Self {
            stream,
            layout,
            buffers,
            replay: std::cell::Cell::new(None),
            decode_attention: false,
        };
        owner.zero()?;
        Ok(owner)
    }

    pub fn buffer(&self, name: &str) -> Result<Buffer> {
        Ok(self
            .buffers
            .get(name)
            .with_context(|| format!("Inkling 2 Mini unit lacks owner {name}"))?
            .buffer)
    }

    pub fn zero(&self) -> Result<()> {
        self.stream.device.run(|| {
            for buffer in self.buffers.values() {
                // SAFETY: fresh private owners written on retained stream.
                unsafe {
                    self.stream
                        .device
                        .library
                        .cuda_zero_bytes_async(buffer.buffer, buffer.buffer.bytes, self.stream.raw)?;
                }
            }
            Ok(())
        })
    }

    pub fn uploads(
        &self,
        cfg: &cuteafd_loader::families::inkling2_mini::Inkling2MiniConfig,
        members: usize,
    ) -> Result<crate::shared::token_table::RetainedUploads<'a>> {
        self.stream.device.run(|| {
            crate::shared::token_table::RetainedUploads::new(self.stream.device, self.layout.upload_shape(cfg, members)?)
        })
    }

    pub fn inventory(&self, inventory: &mut Inventory<'_>, prefix: &str) -> Result<()> {
        for (name, owner) in &self.buffers {
            owner.inventory(inventory, &format!("{prefix}.{name}"))?;
        }
        Ok(())
    }
}

impl<'a> UnitBuffers<'a> for UnitOwner<'a> {
    fn stream(&self) -> &Stream<'a> {
        &self.stream
    }
    fn layout(&self) -> &MiniUnitLayout {
        &self.layout
    }
    fn buffer(&self, name: &str) -> Result<Buffer> {
        UnitOwner::buffer(self, name)
    }
}

impl Drop for UnitOwner<'_> {
    fn drop(&mut self) {
        self.stream.drain_or_abort("Inkling 2 Mini unit owners");
    }
}

pub(crate) struct RequestOwners<'a> {
    pub decode_uploads: crate::shared::token_table::RetainedUploads<'a>,
    pub prefill_uploads: Vec<crate::shared::token_table::RetainedUploads<'a>>,
    pub decode: UnitOwner<'a>,
    pub prefill: Vec<UnitOwner<'a>>,
    pub cache: Rc<super::cache::CacheBank<'a>>,
}

impl<'a> RequestOwners<'a> {
    pub fn allocation_receipts(
        &self,
        inventory: &mut Inventory<'_>,
        census: &mut crate::shared::memory::receipts::Census,
        devices: cuteafd_loader::placement::planner::CoordinatorDevices<'_>,
    ) -> Result<()> {
        self.cache.inventory(inventory)?;
        let decode = "inkling2_mini.decode.0";
        self.decode.inventory(inventory, decode)?;
        self.decode_uploads
            .allocation_receipts(inventory, &format!("{decode}.uploads"))?;
        for (slot, (owner, uploads)) in self.prefill.iter().zip(&self.prefill_uploads).enumerate() {
            let prefix = format!("inkling2_mini.unit.0.{slot}");
            owner.inventory(inventory, &prefix)?;
            uploads.allocation_receipts(inventory, &format!("{prefix}.uploads"))?;
        }
        for (name, uploads) in std::iter::once((format!("{decode}.uploads"), &self.decode_uploads)).chain(
            self.prefill_uploads
                .iter()
                .enumerate()
                .map(|(slot, uploads)| (format!("inkling2_mini.unit.0.{slot}.uploads"), uploads)),
        ) {
            let backing = inventory
                .receipts
                .iter()
                .find(|receipt| {
                    receipt.allocation.id.ends_with(&format!(".{name}"))
                        && receipt.allocation.bytes == uploads.mapped_alias().bytes as u64
                        && receipt.allocation.location == cuteafd_loader::placement::planner::Location::PINNED
                })
                .with_context(|| format!("Mini upload alias has no pinned owner {name}"))?
                .allocation
                .id
                .clone();
            census.mapped(format!("{name}.mapped"), backing, uploads.mapped_alias(), devices)?;
        }
        Ok(())
    }

    pub fn new(
        stream: Rc<Stream<'a>>,
        inputs: &cuteafd_loader::placement::families::inkling2_mini::MiniInputs,
        admitted: &cuteafd_loader::placement::families::inkling2_mini::AdmittedMini,
    ) -> Result<Self> {
        use cuteafd_loader::placement::planner::Location;
        let cfg = &inputs.cfg;
        let sequences = usize::try_from(inputs.sequences)?;
        let context = usize::try_from(inputs.context.maximum)?;
        let pages = usize::try_from(admitted.solved.pool_tokens.div_ceil(64))?;
        let cache_layout = MiniCacheLayout::with_format(cfg, sequences, pages, inputs.kv_format)?;
        let mark_slots = inputs.prefix_mark_slots;
        let decode_layout = MiniUnitLayout::serving_decode(cfg, context, inputs.kv_format, sequences)?;
        let prefill_layout =
            MiniUnitLayout::serving(cfg, usize::try_from(inputs.prefill.rows)?, context, inputs.kv_format, sequences)?;
        let owner = |id: &str, bytes: usize, location| -> Result<()> {
            let allocation = admitted
                .solved
                .facts
                .allocations
                .iter()
                .find(|a| a.id == id)
                .with_context(|| format!("Inkling 2 Mini unadmitted runtime owner {id}"))?;
            ensure!(
                allocation.location == location && allocation.bytes == bytes as u64,
                "Inkling 2 Mini runtime owner differs from admission: {id} {bytes}/{}",
                allocation.bytes
            );
            Ok(())
        };
        let device = Location::coordinator(0);
        owner("inkling2_mini.cache.0.active", cache_layout.active_bytes()?, device)?;
        owner("inkling2_mini.cache.0.conv", cache_layout.conv.bytes()?, device)?;
        if mark_slots > 0 {
            owner("inkling2_mini.cache.0.marks", cache_layout.retained_bytes(mark_slots)?, device)?;
        }
        for &slot in &cache_layout.global_slots {
            let bytes = usize::try_from(cuteafd_loader::placement::planner::sm120_request_backing(
                cache_layout.layer_page_bytes()? as u64,
            )?)?;
            owner(&format!("inkling2_mini.kv.{slot}.0.records"), bytes, device)?;
        }
        for (prefix, layout) in std::iter::once(("inkling2_mini.decode.0".to_string(), &decode_layout)).chain(
            (0..inputs.prefill.waves).map(|slot| (format!("inkling2_mini.unit.0.{slot}"), &prefill_layout)),
        ) {
            for (name, bytes) in &layout.buffers {
                owner(&format!("{prefix}.{name}"), *bytes, device)?;
            }
            let uploads = layout.upload_shape(cfg, sequences.min(layout.rows))?;
            owner(&format!("{prefix}.uploads"), uploads.capacity_bytes()?, Location::PINNED)?;
        }
        let cache = Rc::new(super::cache::CacheBank::new(vec![stream.clone()], cache_layout, mark_slots)?);
        let mut decode = UnitOwner::new(stream.clone(), decode_layout)?;
        decode.decode_attention = true;
        let prefill = (0..inputs.prefill.waves)
            .map(|_| UnitOwner::new(stream.clone(), prefill_layout.clone()))
            .collect::<Result<Vec<_>>>()?;
        let decode_uploads = decode.uploads(cfg, sequences.min(decode.layout.rows))?;
        let prefill_uploads = prefill
            .iter()
            .map(|o| o.uploads(cfg, sequences.min(o.layout.rows)))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            decode_uploads,
            prefill_uploads,
            decode,
            prefill,
            cache,
        })
    }
}
