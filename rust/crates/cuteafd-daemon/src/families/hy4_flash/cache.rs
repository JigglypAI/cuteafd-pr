//! Committed KV, compressed-indexer group cache, SWA-512 rings and exact prefix marks.
//!
//! Layout per coordinator GPU rank:
//! - `dsa_kv`: `[dsa_layers, pages, 64, record_bytes]` (RefPagePool COW physical pages)
//! - `dsa_index`: `[dsa_layers, pages, 16 * 132]` (RefPagePool COW physical pages, 4-token groups)
//! - `swa_rings`: `[swa_layers, sequences * swa_ring_rows, record_bytes]` (contiguous per layer so
//!   batched `mla_swa_decode`/`mla_swa_extend` attends all sequences in one launch via page offset)
//! - `pending_keys`: `[sequences, dsa_layers, 3, 128 * 4]` (incomplete `<4` token keys per DSA layer)
//! - `marks`: `[mark_slots, mark_bytes]` (exact SWA rings + pending keys; ~16.9 MiB/mark FP8)
use crate::shared::memory::device::{Allocation, Device, Stream};
use crate::shared::memory::receipts::Inventory;
use crate::shared::memory::Tag;
use crate::shared::prefix::cuda_copy::SnapshotOwner;
use crate::shared::prefix::{PrefixCopies, Reader};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::CuteafdDeviceBuffer;
use cuteafd_loader::families::hy4_flash::{
    execution::{CacheLayout, PAGE_ROWS, SWA_RING_PAGES},
    Hy4FlashConfig, Hy4KvCache,
};
use std::cell::RefCell;
use std::collections::BTreeSet;
use std::rc::Rc;

/// Bytes per sequence per DSA layer reserved for `0..=3` pending incomplete indexer token keys.
pub(crate) const PENDING_KEY_SLOT_BYTES: usize = 3 * 128 * 4;

fn slice_buf(buf: CuteafdDeviceBuffer, offset: usize, bytes: usize) -> Result<CuteafdDeviceBuffer> {
    ensure!(
        bytes > 0 && offset.checked_add(bytes).is_some_and(|end| end <= buf.bytes),
        "Hy4-flash cache slice [{offset}..+{bytes}] exceeds allocation {}",
        buf.bytes
    );
    Ok(CuteafdDeviceBuffer {
        ptr: buf.ptr.cast::<u8>().wrapping_add(offset).cast(),
        bytes,
        ..buf
    })
}

/// Committed per-rank KV/index/SWA storage with fenced prefix capture/restore and COW tail copies.
pub(crate) struct Hy4CacheRank<'a> {
    copies: PrefixCopies<'a>,
    readers: Vec<Rc<Stream<'a>>>,
    pub(crate) device: Device<'a>,
    pub(crate) kv_format: Hy4KvCache,
    pub(crate) layout: CacheLayout,
    pub(crate) dsa_layers: usize,
    pub(crate) swa_layers: usize,
    pub(crate) pages: usize,
    pub(crate) sequences: usize,
    pub(crate) swa_ring_pages: usize,
    pub(crate) swa_ring_rows: usize,
    pub(crate) mark_slots: usize,
    pub(crate) mark_bytes_per_slot: usize,
    /// `[dsa_layers, pages, 64, record_bytes]`
    pub(crate) dsa_kv: Allocation<'a>,
    /// `[dsa_layers, pages, 16 * 132]`
    pub(crate) dsa_index: Allocation<'a>,
    /// `[swa_layers, sequences * swa_ring_rows, record_bytes]`
    pub(crate) swa_rings: Allocation<'a>,
    /// `[sequences, dsa_layers, 3 * 128 * 4]`
    pub(crate) pending_keys: Allocation<'a>,
    /// Optional prefix-mark storage (`mark_slots * mark_bytes_per_slot`).
    pub(crate) marks: Option<Allocation<'a>>,
    /// Per-sequence count of pending token keys (`0..=3`) per DSA layer.
    pub(crate) pending_counts: RefCell<Vec<Vec<usize>>>,
}

impl<'a> Hy4CacheRank<'a> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        readers: Vec<Rc<Stream<'a>>>,
        cfg: &Hy4FlashConfig,
        kv_format: Hy4KvCache,
        pages: usize,
        sequences: usize,
        max_chunk_rows: usize,
        mark_slots: usize,
    ) -> Result<Self> {
        let first = readers.first().context("Hy4-flash cache requires at least one reader stream")?;
        let device = first.device;
        ensure!(
            readers.iter().all(|s| s.device.id == device.id && std::ptr::eq(s.device.library, device.library)),
            "Hy4-flash cache reader devices differ"
        );
        ensure!(pages > 0 && sequences > 0, "Hy4-flash cache requires positive pages and sequences");
        let dsa_layers = cfg.sparse_layers().len();
        let swa_layers = cfg.swa_layers().len();
        let layout = CacheLayout::new(cfg, kv_format, sequences, dsa_layers, swa_layers)?;
        // Ring holds the live 512-token window plus one full prefill chunk without overwriting
        // any token still needed by queries earlier in the same chunk.
        let swa_ring_pages = ((cfg.sliding_window + max_chunk_rows).div_ceil(PAGE_ROWS) + 1).max(SWA_RING_PAGES);
        let swa_ring_rows = swa_ring_pages * PAGE_ROWS;
        let swa_seq_layer_bytes = swa_ring_rows
            .checked_mul(kv_format.record_bytes())
            .context("Hy4-flash SWA ring layer overflow")?;
        let swa_rings_bytes = swa_layers
            .checked_mul(sequences)
            .and_then(|v| v.checked_mul(swa_seq_layer_bytes))
            .context("Hy4-flash SWA ring pool overflow")?;

        let dsa_kv_bytes = dsa_layers
            .checked_mul(pages)
            .and_then(|v| v.checked_mul(layout.dsa_kv_page_bytes))
            .context("Hy4-flash DSA KV pool overflow")?;
        let dsa_index_bytes = dsa_layers
            .checked_mul(pages)
            .and_then(|v| v.checked_mul(layout.dsa_index_page_bytes))
            .context("Hy4-flash DSA index pool overflow")?;
        let pending_bytes = sequences
            .checked_mul(dsa_layers)
            .and_then(|v| v.checked_mul(PENDING_KEY_SLOT_BYTES))
            .context("Hy4-flash DSA pending keys overflow")?;
        let mark_bytes_per_slot = swa_layers
            .checked_mul(swa_seq_layer_bytes)
            .and_then(|v| v.checked_add(dsa_layers * PENDING_KEY_SLOT_BYTES))
            .context("Hy4-flash mark slot size overflow")?;

        let dsa_kv = Allocation::new(device, dsa_kv_bytes, Tag::kv("hy4_flash/dsa-kv"))?;
        let dsa_index = Allocation::new(device, dsa_index_bytes, Tag::kv("hy4_flash/dsa-index"))?;
        let swa_rings = Allocation::new(device, swa_rings_bytes, Tag::kv("hy4_flash/swa-rings"))?;
        let pending_keys = Allocation::new(device, pending_bytes, Tag::kv("hy4_flash/dsa-pending-keys"))?;
        let marks = if mark_slots > 0 {
            let bytes = mark_slots
                .checked_mul(mark_bytes_per_slot)
                .context("Hy4-flash prefix marks overflow")?;
            Some(Allocation::new(device, bytes, Tag::prefix("hy4_flash/prefix-marks"))?)
        } else {
            None
        };

        let copies = PrefixCopies::new(
            device.library,
            vec![readers
                .iter()
                .map(|s| Reader { device: s.device.id, stream: s.raw })
                .collect()],
        )?;

        let owner = Self {
            copies,
            readers,
            device,
            kv_format,
            layout,
            dsa_layers,
            swa_layers,
            pages,
            sequences,
            swa_ring_pages,
            swa_ring_rows,
            mark_slots,
            mark_bytes_per_slot,
            dsa_kv,
            dsa_index,
            swa_rings,
            pending_keys,
            marks,
            pending_counts: RefCell::new(vec![vec![0; dsa_layers]; sequences]),
        };
        owner.zero_all()?;
        Ok(owner)
    }

    pub(crate) fn zero_all(&self) -> Result<()> {
        self.copies.ordered(0, |stream| {
            // SAFETY: all allocations live on `self.device` and readers are fenced on both sides.
            unsafe {
                self.device.library.cuda_zero_bytes_async(self.dsa_kv.buffer, self.dsa_kv.buffer.bytes, stream)?;
                self.device.library.cuda_zero_bytes_async(self.dsa_index.buffer, self.dsa_index.buffer.bytes, stream)?;
                self.device.library.cuda_zero_bytes_async(self.swa_rings.buffer, self.swa_rings.buffer.bytes, stream)?;
                self.device.library.cuda_zero_bytes_async(self.pending_keys.buffer, self.pending_keys.buffer.bytes, stream)?;
            }
            Ok(())
        })?;
        for seq in self.pending_counts.borrow_mut().iter_mut() {
            seq.fill(0);
        }
        self.copies.drain()
    }

    pub(crate) fn reset_sequence(&self, seq: usize) -> Result<()> {
        ensure!(seq < self.sequences, "Hy4-flash sequence {seq} out of range");
        let mut ranges = Vec::with_capacity(self.swa_layers + 1);
        for swa_idx in 0..self.swa_layers {
            ranges.push(self.swa_ring(seq, swa_idx)?);
        }
        let pending_seq_bytes = self.dsa_layers * PENDING_KEY_SLOT_BYTES;
        ranges.push(slice_buf(self.pending_keys.buffer, seq * pending_seq_bytes, pending_seq_bytes)?);
        self.copies.ordered(0, |stream| {
            // SAFETY: fenced copy stream orders all arithmetic readers before zeroing this sequence's state.
            unsafe {
                for buf in &ranges {
                    self.device.library.cuda_zero_bytes_async(*buf, buf.bytes, stream)?;
                }
            }
            Ok(())
        })?;
        self.pending_counts.borrow_mut()[seq].fill(0);
        Ok(())
    }

    pub(crate) fn dsa_kv_layer(&self, dsa_idx: usize) -> Result<CuteafdDeviceBuffer> {
        ensure!(dsa_idx < self.dsa_layers, "DSA layer index {dsa_idx} out of range");
        let layer_bytes = self.pages * self.layout.dsa_kv_page_bytes;
        slice_buf(self.dsa_kv.buffer, dsa_idx * layer_bytes, layer_bytes)
    }

    pub(crate) fn dsa_kv_page(&self, dsa_idx: usize, page: usize) -> Result<CuteafdDeviceBuffer> {
        ensure!(page < self.pages, "DSA KV page {page} outside pool {}", self.pages);
        let layer = self.dsa_kv_layer(dsa_idx)?;
        slice_buf(layer, page * self.layout.dsa_kv_page_bytes, self.layout.dsa_kv_page_bytes)
    }

    pub(crate) fn dsa_index_layer(&self, dsa_idx: usize) -> Result<CuteafdDeviceBuffer> {
        ensure!(dsa_idx < self.dsa_layers, "DSA layer index {dsa_idx} out of range");
        let layer_bytes = self.pages * self.layout.dsa_index_page_bytes;
        slice_buf(self.dsa_index.buffer, dsa_idx * layer_bytes, layer_bytes)
    }

    pub(crate) fn dsa_index_page(&self, dsa_idx: usize, page: usize) -> Result<CuteafdDeviceBuffer> {
        ensure!(page < self.pages, "DSA index page {page} outside pool {}", self.pages);
        let layer = self.dsa_index_layer(dsa_idx)?;
        slice_buf(layer, page * self.layout.dsa_index_page_bytes, self.layout.dsa_index_page_bytes)
    }

    /// Contiguous `[sequences * swa_ring_rows, record_bytes]` bank for one SWA layer (batched attention).
    pub(crate) fn swa_layer_bank(&self, swa_idx: usize) -> Result<CuteafdDeviceBuffer> {
        ensure!(swa_idx < self.swa_layers, "SWA layer index {swa_idx} out of range");
        let layer_bytes = self.sequences * self.swa_ring_rows * self.kv_format.record_bytes();
        slice_buf(self.swa_rings.buffer, swa_idx * layer_bytes, layer_bytes)
    }

    /// One sequence's `[swa_ring_rows, record_bytes]` ring slice inside `swa_layer_bank(swa_idx)`.
    pub(crate) fn swa_ring(&self, seq: usize, swa_idx: usize) -> Result<CuteafdDeviceBuffer> {
        ensure!(seq < self.sequences && swa_idx < self.swa_layers, "SWA ring ({seq}, {swa_idx}) out of range");
        let ring_bytes = self.swa_ring_rows * self.kv_format.record_bytes();
        let bank = self.swa_layer_bank(swa_idx)?;
        slice_buf(bank, seq * ring_bytes, ring_bytes)
    }

    pub(crate) fn pending_key_slot(&self, seq: usize, dsa_idx: usize) -> Result<CuteafdDeviceBuffer> {
        ensure!(seq < self.sequences && dsa_idx < self.dsa_layers, "pending key ({seq}, {dsa_idx}) out of range");
        let seq_stride = self.dsa_layers * PENDING_KEY_SLOT_BYTES;
        slice_buf(self.pending_keys.buffer, seq * seq_stride + dsa_idx * PENDING_KEY_SLOT_BYTES, PENDING_KEY_SLOT_BYTES)
    }

    pub(crate) fn mark(&self, slot: usize) -> Result<CuteafdDeviceBuffer> {
        ensure!(slot < self.mark_slots, "Hy4-flash mark slot {slot} outside arena {}", self.mark_slots);
        let arena = self.marks.as_ref().context("Hy4-flash prefix mark arena absent")?;
        slice_buf(arena.buffer, slot * self.mark_bytes_per_slot, self.mark_bytes_per_slot)
    }

    fn state_copy(&self, seq: usize, mark_slot: usize, capture: bool) -> Result<()> {
        let mark = self.mark(mark_slot)?;
        let ring_bytes = self.swa_ring_rows * self.kv_format.record_bytes();
        let mut copies = Vec::with_capacity(self.swa_layers + 1);
        let mut offset = 0usize;
        for swa_idx in 0..self.swa_layers {
            let live = self.swa_ring(seq, swa_idx)?;
            let saved = slice_buf(mark, offset, ring_bytes)?;
            offset += ring_bytes;
            copies.push(if capture { (0, saved, live) } else { (0, live, saved) });
        }
        let pending_bytes = self.dsa_layers * PENDING_KEY_SLOT_BYTES;
        let live_pending = slice_buf(self.pending_keys.buffer, seq * pending_bytes, pending_bytes)?;
        let saved_pending = slice_buf(mark, offset, pending_bytes)?;
        copies.push(if capture { (0, saved_pending, live_pending) } else { (0, live_pending, saved_pending) });
        // SAFETY: disjoint active/mark regions fenced across all registered readers.
        unsafe { self.copies.copy_d2d(&copies) }
    }

    pub(crate) fn capture(&self, seq: usize, mark_slot: usize) -> Result<()> {
        self.state_copy(seq, mark_slot, true)
    }

    pub(crate) fn restore(&self, mark_slot: usize, seq: usize, len: usize) -> Result<()> {
        self.state_copy(seq, mark_slot, false)?;
        self.pending_counts.borrow_mut()[seq].fill(len % 4);
        Ok(())
    }

    /// Exact COW partial-page copy: copies `rows` KV rows and `rows / 4` complete indexer groups across all 12 DSA layers.
    pub(crate) fn copy_page_rows(&self, from: usize, to: usize, rows: usize) -> Result<()> {
        ensure!(from != to && (1..=PAGE_ROWS).contains(&rows), "Hy4-flash page tail copy aliases or exceeds page");
        let kv_bytes = rows * self.kv_format.record_bytes();
        let idx_groups = rows / 4;
        let idx_bytes = idx_groups * (self.layout.dsa_index_page_bytes / (PAGE_ROWS / 4));
        let mut copies = Vec::with_capacity(self.dsa_layers * 2);
        for dsa_idx in 0..self.dsa_layers {
            let src_kv = slice_buf(self.dsa_kv_page(dsa_idx, from)?, 0, kv_bytes)?;
            let dst_kv = slice_buf(self.dsa_kv_page(dsa_idx, to)?, 0, kv_bytes)?;
            copies.push((0, dst_kv, src_kv));
            if idx_bytes > 0 {
                let src_idx = slice_buf(self.dsa_index_page(dsa_idx, from)?, 0, idx_bytes)?;
                let dst_idx = slice_buf(self.dsa_index_page(dsa_idx, to)?, 0, idx_bytes)?;
                copies.push((0, dst_idx, src_idx));
            }
        }
        // SAFETY: `from != to` guarantees disjoint pages; `PrefixCopies` fences all readers.
        unsafe { self.copies.copy_d2d(&copies) }
    }

    pub(crate) fn drain(&self) -> Result<()> {
        self.copies.drain()
    }

    pub(crate) fn page_bytes_total(&self) -> usize {
        self.dsa_layers * (self.layout.dsa_kv_page_bytes + self.layout.dsa_index_page_bytes)
    }

    pub(crate) fn inventory(&self, inventory: &mut Inventory<'_>) -> Result<()> {
        let location = inventory.location(self.device.id)?;
        let cuteafd_loader::placement::planner::Device::Coordinator(gpu) = location.device else {
            anyhow::bail!("Hy4-flash cache is not coordinator-owned");
        };
        self.dsa_kv.inventory(inventory, &format!("hy4_flash.cache.{gpu}.dsa_kv"))?;
        self.dsa_index.inventory(inventory, &format!("hy4_flash.cache.{gpu}.dsa_index"))?;
        self.swa_rings.inventory(inventory, &format!("hy4_flash.cache.{gpu}.swa_rings"))?;
        self.pending_keys.inventory(inventory, &format!("hy4_flash.cache.{gpu}.pending_keys"))?;
        if let Some(marks) = &self.marks {
            marks.inventory(inventory, &format!("hy4_flash.cache.{gpu}.marks"))?;
        }
        self.copies.allocation_receipts(inventory, &format!("hy4_flash.cache.{gpu}.copies"))
    }

    pub(crate) const fn page_rows() -> usize {
        PAGE_ROWS
    }

    pub(crate) const fn swa_ring_pages() -> usize {
        SWA_RING_PAGES
    }
}

impl SnapshotOwner for Hy4CacheRank<'_> {
    fn snapshot_buffers(&self) -> Result<Vec<CuteafdDeviceBuffer>> {
        let mut out = vec![self.dsa_kv.buffer, self.dsa_index.buffer];
        if let Some(marks) = &self.marks {
            out.push(marks.buffer);
        }
        Ok(out)
    }
}

/// Logical COW pages, active slot, causal frontier and Engram rolling tail for one Hy4-flash sequence.
#[derive(Clone, Debug)]
pub(crate) struct Hy4Placement {
    pub units: Vec<u32>,
    pub slot: usize,
    pub len: usize,
    pub state_len: usize,
    /// Last 2 committed token IDs (`[t_{len-2}, t_{len-1}]`, left-padded with `engram_pad_id`)
    /// so Engram 3-gram hashing resumes byte-identically from any prefix mark or decode step.
    pub engram_tail: [u32; 2],
}

impl Hy4Placement {
    pub(crate) fn new(units: Vec<u32>, slot: usize, pad_id: u32) -> Self {
        Self {
            units,
            slot,
            len: 0,
            state_len: 0,
            engram_tail: [pad_id, pad_id],
        }
    }

    pub(crate) fn set_engram_tail_from_prefix(&mut self, prefix: &[u32], pad_id: u32) {
        self.engram_tail = match prefix.len() {
            0 => [pad_id, pad_id],
            1 => [pad_id, prefix[0]],
            n => [prefix[n - 2], prefix[n - 1]],
        };
    }
}

impl crate::shared::qwen::scheduler::Placement for Hy4Placement {
    fn len(&self) -> usize {
        self.len
    }
    fn set_len(&mut self, len: usize) {
        self.len = len;
    }
    fn units(&self) -> &[u32] {
        &self.units
    }
    fn slot(&self) -> i32 {
        self.slot as i32
    }
    fn spans(&self) -> &[cuteafd_core::MediaSpan] {
        &[]
    }
}

/// Exact prefix-cache adapter across 1 or 2 Hy4-flash coordinator ranks.
#[derive(Clone)]
pub(crate) struct Hy4Prefix<'a> {
    ranks: Vec<Rc<Hy4CacheRank<'a>>>,
    pad_id: u32,
}

impl<'a> Hy4Prefix<'a> {
    pub(crate) fn new(ranks: Vec<Rc<Hy4CacheRank<'a>>>, pad_id: u32) -> Result<Self> {
        ensure!(!ranks.is_empty(), "Hy4Prefix requires at least one rank");
        Ok(Self { ranks, pad_id })
    }

    pub(crate) fn snapshot_owners(&self) -> Vec<Rc<dyn SnapshotOwner + 'a>> {
        self.ranks
            .iter()
            .map(|r| r.clone() as Rc<dyn SnapshotOwner>)
            .collect()
    }

    fn validate(&self, placement: &Hy4Placement, len: usize) -> Result<()> {
        let r0 = &self.ranks[0];
        ensure!(
            placement.slot < r0.sequences
                && len.div_ceil(PAGE_ROWS) <= placement.units.len()
                && placement.units.iter().all(|&p| (p as usize) < r0.pages),
            "Hy4-flash prefix placement outside admitted state/pages"
        );
        ensure!(
            placement.units.iter().copied().collect::<BTreeSet<_>>().len() == placement.units.len(),
            "Hy4-flash prefix placement aliases physical pages"
        );
        Ok(())
    }
}

impl cuteafd_engine::prefix::PrefixFamily for Hy4Prefix<'_> {
    type Placement = Hy4Placement;

    fn record_format(&self) -> &'static str {
        match self.ranks[0].kv_format {
            Hy4KvCache::Fp8 => "mla_fp8_ds_v1",
            Hy4KvCache::Bf16 => "mla_bf16_v1",
        }
    }

    fn layout(&self) -> cuteafd_engine::prefix::FamilyLayout {
        use cuteafd_engine::prefix::{FamilyLayout, MarkStore, ReuseRule};
        FamilyLayout {
            page_rows: PAGE_ROWS,
            pages: self.ranks[0].pages,
            page_bytes: self.ranks.iter().map(|r| r.page_bytes_total()).sum(),
            mark_bytes: self.ranks.iter().map(|r| r.mark_bytes_per_slot).sum(),
            draft_bytes: 0,
            rule: ReuseRule::EXACT,
            mark_store: MarkStore::Arena,
            page_owners: Default::default(),
        }
    }

    fn pages<'p>(&self, placement: &'p Hy4Placement) -> &'p [u32] {
        &placement.units
    }

    fn bind(&self, placement: &mut Hy4Placement) -> Result<(), cuteafd_engine::prefix::BoxError> {
        self.validate(placement, placement.len)?;
        for rank in &self.ranks {
            rank.reset_sequence(placement.slot)?;
        }
        placement.state_len = 0;
        placement.engram_tail = [self.pad_id, self.pad_id];
        Ok(())
    }

    fn discard(&self, placement: &mut Hy4Placement) -> Result<(), cuteafd_engine::prefix::BoxError> {
        for rank in &self.ranks {
            rank.reset_sequence(placement.slot)?;
        }
        placement.state_len = 0;
        placement.engram_tail = [self.pad_id, self.pad_id];
        Ok(())
    }

    fn extend_pages(&self, placement: &mut Hy4Placement, pages: &[u32]) -> Result<(), cuteafd_engine::prefix::BoxError> {
        let mut extended = placement.clone();
        extended.units.extend_from_slice(pages);
        self.validate(&extended, extended.len)?;
        placement.units = extended.units;
        Ok(())
    }

    fn commit_point(&self, placement: &Hy4Placement) -> usize {
        placement.len.min(placement.state_len)
    }

    fn capture(
        &self,
        slot: cuteafd_engine::prefix::MarkSlot,
        placement: &Hy4Placement,
        len: usize,
    ) -> Result<(), cuteafd_engine::prefix::BoxError> {
        self.validate(placement, len)?;
        if len != placement.len || len != placement.state_len {
            return Err("Hy4-flash snapshot frontier does not match committed causal state".into());
        }
        for rank in &self.ranks {
            rank.capture(placement.slot, slot.0 as usize)?;
        }
        Ok(())
    }

    fn restore(
        &self,
        mark: Option<cuteafd_engine::prefix::MarkSlot>,
        placement: &mut Hy4Placement,
        len: usize,
    ) -> Result<(), cuteafd_engine::prefix::BoxError> {
        self.validate(placement, len)?;
        let slot = mark.context("Hy4-flash exact restore requires its SWA/pending-key mark")?;
        for rank in &self.ranks {
            rank.restore(slot.0 as usize, placement.slot, len)?;
        }
        placement.len = len;
        placement.state_len = len;
        Ok(())
    }

    fn copy_rows(&self, copy: cuteafd_engine::prefix::TailCopy) -> Result<(), cuteafd_engine::prefix::BoxError> {
        if copy.owner.is_some() {
            return Err("Hy4-flash whole-rank pages have no parity owner".into());
        }
        for rank in &self.ranks {
            rank.copy_page_rows(copy.from as usize, copy.to as usize, copy.rows)?;
        }
        Ok(())
    }

    fn drain(&self) -> Result<(), cuteafd_engine::prefix::BoxError> {
        for rank in &self.ranks {
            rank.drain()?;
        }
        Ok(())
    }

    fn page_segments(&self, page: u32) -> Vec<cuteafd_hostcache::copy::DeviceRange> {
        let mut out = Vec::new();
        for rank in &self.ranks {
            for d in 0..rank.dsa_layers {
                if let Ok(b) = rank.dsa_kv_page(d, page as usize) {
                    out.push(crate::shared::prefix::cuda_copy::range(b));
                }
                if let Ok(b) = rank.dsa_index_page(d, page as usize) {
                    out.push(crate::shared::prefix::cuda_copy::range(b));
                }
            }
        }
        out
    }

    fn mark_segments(&self, slot: cuteafd_engine::prefix::MarkSlot) -> Vec<cuteafd_hostcache::copy::DeviceRange> {
        self.ranks
            .iter()
            .filter_map(|rank| rank.mark(slot.0 as usize).ok())
            .map(crate::shared::prefix::cuda_copy::range)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::prefix::CudaCopyEngine;
    use cuteafd_engine::prefix::{After, PrefixCache, PrefixConfig, PrefixFamily, SnapshotKind};
    use cuteafd_ffi::native_library_lifetime_fixture::Fixture;

    #[test]
    fn hy4_exact_prefix_cache_restores_swa_pending_and_dsa_pages_byte_identically() -> Result<()> {
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.immediate_copies(&library)?;
        let device = Device { library: &library, id: 0 };
        let reader = Rc::new(Stream::new(device)?);
        let cfg = Hy4FlashConfig {
            vocab_size: 120_832,
            hidden: 3072,
            layers: 2,
            intermediate: 13_824,
            max_position: 262_144,
            rms_norm_eps: 1e-5,
            hc_mult: 4,
            hc_eps: 1e-6,
            hc_magnitude: 2.0,
            q_lora_rank: 1024,
            kv_lora_rank: 512,
            qk_nope_dim: 192,
            qk_rope_dim: 64,
            v_head_dim: 256,
            sparse_heads: 32,
            swa_heads: 64,
            swa_qk_nope_dim: 192,
            swa_qk_rope_dim: 64,
            swa_v_head_dim: 256,
            sliding_window: 512,
            sparse_rope_theta: 10_000.0,
            swa_rope_theta: 10_000.0,
            sparse_layer: vec![true, false],
            indexer_heads: 16,
            indexer_head_dim: 128,
            indexer_compress_ratio: 4,
            indexer_group_topk: 512,
            indexer_candidate_topk: 2048,
            dense_layer: vec![true, false],
            experts: 512,
            topk: 16,
            moe_intermediate: 1536,
            routed_expert_hidden: 1536,
            shared_experts: 1,
            n_group: 1,
            topk_group: 1,
            norm_topk_prob: true,
            routed_scaling_factor: 1.0,
            swiglu_limit: 10.0,
            engram_layers: Vec::new(),
            engram_vocab_sizes: std::collections::BTreeMap::new(),
            engram_max_ngram_size: 3,
            engram_n_embed_per_ngram: 768,
            engram_n_head_per_ngram: 8,
            engram_pad_id: 120_002,
            engram_seed: 0,
            engram_fp8_embedding: true,
            engram_lookup_table_size: 120_818,
            image_token_id: 120_120,
            video_token_id: 120_124,
            eos: vec![120_025],
        };
        let rank = Rc::new(Hy4CacheRank::new(vec![reader], &cfg, Hy4KvCache::Fp8, 8, 2, 64, 3)?);
        let family = Hy4Prefix::new(vec![rank.clone()], cfg.engram_pad_id)?;
        let mut cache: PrefixCache<CudaCopyEngine<'_>> = PrefixCache::new(
            family.layout(),
            PrefixConfig { entries: 3, mark_slots: 3, keep_logits: true, min_tokens: 1 },
            None,
        )?;

        let tokens: Vec<u32> = (10..77).collect(); // 67 tokens -> 2 pages, partial tail = 3 rows, pending = 3!
        let mut src = cache
            .admit(&family, &tokens, 128, false, |units| Hy4Placement::new(units, 0, cfg.engram_pad_id))?
            .placement;
        let swa0 = rank.swa_ring(0, 0)?;
        let swa_bytes = vec![0x5Au8; swa0.bytes];
        library.copy_h2d(swa0, &swa_bytes)?;
        let pend0 = rank.pending_key_slot(0, 0)?;
        let pend_bytes = vec![0xA5u8; pend0.bytes];
        library.copy_h2d(pend0, &pend_bytes)?;
        for &pg in &src.units {
            let kv = rank.dsa_kv_page(0, pg as usize)?;
            library.copy_h2d(kv, &vec![(pg as u8).wrapping_add(0x21); kv.bytes])?;
            let idx = rank.dsa_index_page(0, pg as usize)?;
            library.copy_h2d(idx, &vec![(pg as u8).wrapping_add(0x41); idx.bytes])?;
        }
        src.len = tokens.len();
        src.state_len = tokens.len();
        assert!(cache.capture(&family, SnapshotKind::Prompt, &tokens, &src, After::from_logits(&[1.0, 0.0], true))?);
        cache.release(&family, &src.units)?;

        let mut ext_tokens = tokens.clone();
        ext_tokens.push(999);
        let restored = cache.admit(&family, &ext_tokens, 128, false, |units| {
            Hy4Placement::new(units, 1, cfg.engram_pad_id)
        })?;
        assert_eq!(restored.resume, tokens.len());
        assert_eq!(restored.placement.len, tokens.len());
        assert_eq!(rank.pending_counts.borrow()[1][0], tokens.len() % 4);
        let mut got_swa = vec![0u8; swa0.bytes];
        library.copy_d2h(&mut got_swa, rank.swa_ring(1, 0)?)?;
        assert_eq!(got_swa, swa_bytes);
        let mut got_pend = vec![0u8; pend0.bytes];
        library.copy_d2h(&mut got_pend, rank.pending_key_slot(1, 0)?)?;
        assert_eq!(got_pend, pend_bytes);
        Ok(())
    }
}
