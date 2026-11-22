//! Committed KV, compressed-indexer group cache, SWA-512 rings and exact prefix marks.
use crate::shared::memory::{
    device::{Allocation, Device, Stream},
    Tag,
};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::CuteafdDeviceBuffer;
use cuteafd_loader::families::hy4_flash::{
    execution::{CacheLayout, PAGE_ROWS, SWA_RING_PAGES, SWA_RING_ROWS},
    Hy4FlashConfig, Hy4KvCache,
};

/// Committed per-rank KV/index/SWA storage matching `CacheLayout` byte for byte.
pub(crate) struct Hy4CacheRank<'a> {
    pub(crate) device: Device<'a>,
    pub(crate) kv_format: Hy4KvCache,
    pub(crate) layout: CacheLayout,
    pub(crate) dsa_layers: usize,
    pub(crate) swa_layers: usize,
    pub(crate) pages: usize,
    pub(crate) sequences: usize,
    /// `[dsa_layers, pages, 64, record_bytes]`
    pub(crate) dsa_kv: Allocation<'a>,
    /// `[dsa_layers, pages, 16 * 132]`
    pub(crate) dsa_index: Allocation<'a>,
    /// `[sequences, swa_layers, SWA_RING_ROWS (640), record_bytes]`
    pub(crate) swa_rings: Allocation<'a>,
    /// `[sequences, dsa_layers, 3, 128]` BF16 pending incomplete indexer token keys.
    pub(crate) pending_keys: Allocation<'a>,
    /// Optional prefix-mark storage (`mark_slots * mark_bytes_per_slot`).
    pub(crate) marks: Option<Allocation<'a>>,
    /// Per-sequence count of pending token keys (`0..=3`) per DSA layer.
    pub(crate) pending_counts: Vec<Vec<usize>>,
}

fn slice_buf(buf: CuteafdDeviceBuffer, offset: usize, bytes: usize) -> Result<CuteafdDeviceBuffer> {
    ensure!(
        offset.checked_add(bytes).is_some_and(|end| end <= buf.bytes),
        "Hy4-flash cache slice [{offset}..+{bytes}] exceeds allocation {}",
        buf.bytes
    );
    Ok(CuteafdDeviceBuffer {
        ptr: buf.ptr.cast::<u8>().wrapping_add(offset).cast(),
        bytes,
        ..buf
    })
}

impl<'a> Hy4CacheRank<'a> {
    pub(crate) fn new(
        device: Device<'a>,
        cfg: &Hy4FlashConfig,
        kv_format: Hy4KvCache,
        pages: usize,
        sequences: usize,
        mark_slots: usize,
        stream: &Stream<'a>,
    ) -> Result<Self> {
        ensure!(pages > 0 && sequences > 0, "Hy4-flash cache requires positive pages and sequences");
        let dsa_layers = cfg.sparse_layers().len();
        let swa_layers = cfg.swa_layers().len();
        let layout = CacheLayout::new(cfg, kv_format, sequences, dsa_layers, swa_layers)?;

        let dsa_kv_bytes = dsa_layers
            .checked_mul(pages)
            .and_then(|v| v.checked_mul(layout.dsa_kv_page_bytes))
            .context("Hy4-flash DSA KV pool overflow")?;
        let dsa_index_bytes = dsa_layers
            .checked_mul(pages)
            .and_then(|v| v.checked_mul(layout.dsa_index_page_bytes))
            .context("Hy4-flash DSA index pool overflow")?;

        let dsa_kv = Allocation::new(device, dsa_kv_bytes, Tag::kv("hy4_flash/dsa-kv"))?;
        let dsa_index = Allocation::new(device, dsa_index_bytes, Tag::kv("hy4_flash/dsa-index"))?;
        let swa_rings = Allocation::new(device, layout.active_swa_ring_bytes, Tag::kv("hy4_flash/swa-rings"))?;
        let pending_keys = Allocation::new(
            device,
            layout.active_indexer_pending_bytes,
            Tag::kv("hy4_flash/dsa-pending-keys"),
        )?;
        let marks = if mark_slots > 0 {
            let bytes = mark_slots
                .checked_mul(layout.mark_bytes_per_slot)
                .context("Hy4-flash prefix marks overflow")?;
            Some(Allocation::new(device, bytes, Tag::prefix("hy4_flash/prefix-marks"))?)
        } else {
            None
        };

        let owner = Self {
            device,
            kv_format,
            layout,
            dsa_layers,
            swa_layers,
            pages,
            sequences,
            dsa_kv,
            dsa_index,
            swa_rings,
            pending_keys,
            marks,
            pending_counts: vec![vec![0; dsa_layers]; sequences],
        };
        owner.zero_all(stream)?;
        Ok(owner)
    }

    pub(crate) fn zero_all(&self, stream: &Stream<'a>) -> Result<()> {
        self.device.run(|| {
            // SAFETY: all allocations live on `self.device` until `stream` drains.
            unsafe {
                self.device.library.cuda_zero_bytes_async(self.dsa_kv.buffer, self.dsa_kv.buffer.bytes, stream.raw)?;
                self.device.library.cuda_zero_bytes_async(self.dsa_index.buffer, self.dsa_index.buffer.bytes, stream.raw)?;
                self.device.library.cuda_zero_bytes_async(self.swa_rings.buffer, self.swa_rings.buffer.bytes, stream.raw)?;
                self.device.library.cuda_zero_bytes_async(self.pending_keys.buffer, self.pending_keys.buffer.bytes, stream.raw)?;
            }
            Ok(())
        })?;
        stream.drain()
    }

    pub(crate) fn dsa_kv_layer(&self, dsa_idx: usize) -> Result<CuteafdDeviceBuffer> {
        ensure!(dsa_idx < self.dsa_layers, "DSA layer index {dsa_idx} out of range");
        let layer_bytes = self.pages * self.layout.dsa_kv_page_bytes;
        slice_buf(self.dsa_kv.buffer, dsa_idx * layer_bytes, layer_bytes)
    }

    pub(crate) fn dsa_index_layer(&self, dsa_idx: usize) -> Result<CuteafdDeviceBuffer> {
        ensure!(dsa_idx < self.dsa_layers, "DSA layer index {dsa_idx} out of range");
        let layer_bytes = self.pages * self.layout.dsa_index_page_bytes;
        slice_buf(self.dsa_index.buffer, dsa_idx * layer_bytes, layer_bytes)
    }

    pub(crate) fn swa_ring(&self, seq: usize, swa_idx: usize) -> Result<CuteafdDeviceBuffer> {
        ensure!(seq < self.sequences && swa_idx < self.swa_layers, "SWA ring ({seq}, {swa_idx}) out of range");
        let ring_bytes = SWA_RING_ROWS * self.kv_format.record_bytes();
        let seq_stride = self.swa_layers * ring_bytes;
        slice_buf(self.swa_rings.buffer, seq * seq_stride + swa_idx * ring_bytes, ring_bytes)
    }

    pub(crate) fn pending_key_slot(&self, seq: usize, dsa_idx: usize) -> Result<CuteafdDeviceBuffer> {
        ensure!(seq < self.sequences && dsa_idx < self.dsa_layers, "pending key ({seq}, {dsa_idx}) out of range");
        // 3 pending tokens x 128 BF16 (2B) packed inside the 3 * 128 * 4B reservation slot.
        let slot_bytes = 3 * 128 * 4;
        let seq_stride = self.dsa_layers * slot_bytes;
        slice_buf(self.pending_keys.buffer, seq * seq_stride + dsa_idx * slot_bytes, slot_bytes)
    }

    pub(crate) const fn page_rows() -> usize {
        PAGE_ROWS
    }

    pub(crate) const fn swa_ring_pages() -> usize {
        SWA_RING_PAGES
    }
}
