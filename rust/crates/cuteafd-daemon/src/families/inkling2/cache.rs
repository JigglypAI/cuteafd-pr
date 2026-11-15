//! Persistent request state, separate from in-flight row scratch. Shared prefix
//! copy fences join every reader before snapshots, restores and retirement.
use crate::shared::{memory::{device::{Allocation, Stream}, receipts::Inventory},
    prefix::{PrefixCopies, Reader}};
use anyhow::{ensure, Context, Result};
use cuteafd_core::memory_layout::Tag;
use cuteafd_ffi::CuteafdDeviceBuffer as Buffer;
use cuteafd_loader::families::inkling2::execution::CacheLayout;
use std::{collections::BTreeMap, rc::Rc};

pub(crate) fn view(buffer: Buffer, offset: usize, bytes: usize) -> Result<Buffer> {
    ensure!(bytes > 0 && offset.checked_add(bytes).is_some_and(|end| end <= buffer.bytes),
        "Inkling cache view exceeds backing");
    Ok(Buffer { ptr: buffer.ptr.cast::<u8>().wrapping_add(offset).cast(), bytes, ..buffer })
}

pub(crate) struct CacheBank<'a> {
    // Copies drain while all readers and their storage still exist.
    copies: PrefixCopies<'a>,
    readers: Vec<Rc<Stream<'a>>>,
    active: Allocation<'a>,
    marks: Option<Allocation<'a>>,
    pages: BTreeMap<usize, Allocation<'a>>,
    pub layout: CacheLayout,
    mark_slots: usize,
}
impl<'a> CacheBank<'a> {
    pub fn require_reader(&self, stream: &Stream<'_>) -> Result<()> {
        ensure!(self.readers.iter().any(|reader| reader.raw == stream.raw
            && reader.device.id == stream.device.id
            && std::ptr::eq(reader.device.library, stream.device.library)),
            "Inkling arithmetic stream is not fenced by persistent cache");
        Ok(())
    }
    /// All requests, including tails, are admitted before construction.
    pub fn new(readers: Vec<Rc<Stream<'a>>>, layout: CacheLayout, mark_slots: usize) -> Result<Self> {
        let first = readers.first().context("Inkling cache requires readers")?;
        let device = first.device;
        ensure!(readers.iter().all(|stream| stream.device.id == device.id
            && std::ptr::eq(stream.device.library, device.library)), "Inkling cache reader devices differ");
        let active = Allocation::new(device, layout.active_bytes()?, Tag::kv("inkling2/sequence-state"))?;
        let marks = (mark_slots > 0).then(|| Allocation::new(device, layout.retained_bytes(mark_slots)?,
            Tag::new(cuteafd_core::memory_layout::MemoryKind::Prefix, "inkling2/prefix-marks"))).transpose()?;
        let pages = layout.global_layers.iter().map(|&layer| Ok((layer,
            Allocation::new(device, layout.layer_page_bytes()?, Tag::kv("inkling2/global-pages"))?)))
            .collect::<Result<BTreeMap<_, _>>>()?;
        let copies = PrefixCopies::new(device.library, vec![readers.iter().map(|stream|
            Reader { device: stream.device.id, stream: stream.raw }).collect()])?;
        let bank = Self { copies, readers, active, marks, pages, layout, mark_slots };
        bank.copies.ordered(0, |stream| {
            // SAFETY: all allocations succeeded before initialization; the bank
            // retains every destination and joins readers on both sides.
            unsafe { device.library.cuda_zero_bytes_async(bank.active.buffer, bank.active.buffer.bytes, stream)?; }
            Ok(())
        })?;
        Ok(bank)
    }
    pub fn field(&self, sequence: usize, layer: usize, name: &str) -> Result<Buffer> {
        let span = self.layout.field(sequence, layer, name)?;
        view(self.active.buffer, span.start, span.len())
    }
    pub fn global(&self, layer: usize) -> Result<Buffer> {
        self.pages.get(&layer).map(|owner| owner.buffer).context("Inkling layer has no global KV pages")
    }
    pub fn page(&self, layer: usize, page: usize) -> Result<Buffer> {
        ensure!(page < self.layout.pool_pages, "Inkling physical page outside pool");
        view(self.global(layer)?, page*self.layout.page_bytes, self.layout.page_bytes)
    }
    fn sequence(&self, sequence: usize) -> Result<Buffer> {
        ensure!(sequence < self.layout.sequences, "Inkling sequence outside cache");
        view(self.active.buffer, sequence*self.layout.sequence_bytes, self.layout.sequence_bytes)
    }
    pub fn mark(&self, slot: usize) -> Result<Buffer> {
        ensure!(slot < self.mark_slots, "Inkling mark outside arena");
        view(self.marks.as_ref().context("Inkling mark arena absent")?.buffer,
            slot*self.layout.sequence_bytes, self.layout.sequence_bytes)
    }
    pub fn reset(&self, sequence: usize) -> Result<()> {
        let dst = self.sequence(sequence)?;
        self.copies.ordered(0, |stream| {
            // SAFETY: copy fences join every arithmetic reader before overwrite.
            unsafe { self.readers[0].device.library.cuda_zero_bytes_async(dst, dst.bytes, stream) }
        })
    }
    pub fn capture(&self, sequence: usize, slot: usize) -> Result<()> {
        self.copy(self.mark(slot)?, self.sequence(sequence)?)
    }
    pub fn restore(&self, slot: usize, sequence: usize) -> Result<()> {
        self.copy(self.sequence(sequence)?, self.mark(slot)?)
    }
    fn copy(&self, dst: Buffer, src: Buffer) -> Result<()> {
        ensure!(dst.bytes == src.bytes && dst.ptr != src.ptr, "Inkling cache copy extent/alias");
        // SAFETY: checked disjoint sequence/mark regions survive all fenced
        // copies; PrefixCopies orders every registered consumer on both sides.
        unsafe { self.copies.copy_d2d(&[(0, dst, src)]) }
    }
    pub fn copy_page(&self, from: usize, to: usize) -> Result<()> {
        ensure!(from != to, "Inkling page tail copy aliases itself");
        let copies = self.layout.global_layers.iter().map(|&layer|
            Ok((0, self.page(layer, to)?, self.page(layer, from)?))).collect::<Result<Vec<_>>>()?;
        // SAFETY: physical page IDs are distinct; the retained bank owns every
        // source/destination until the shared prefix fence drains.
        unsafe { self.copies.copy_d2d(&copies) }
    }
    pub fn drain(&self) -> Result<()> { self.copies.drain() }
    pub fn inventory(&self, inventory: &mut Inventory<'_>) -> Result<()> {
        self.active.inventory(inventory, "inkling2.cache.active")?;
        if let Some(marks) = &self.marks { marks.inventory(inventory, "inkling2.cache.marks")?; }
        for (layer, owner) in &self.pages { owner.inventory(inventory, &format!("inkling2.cache.layer.{layer}.pages"))?; }
        self.copies.allocation_receipts(inventory, "inkling2.cache.copies")
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::memory::device::Device;
    use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
    use cuteafd_loader::families::inkling2::{execution::UnitLayout, Inkling2Config};

    #[test]
    fn persistent_cache_marks_copy_exact_bytes_and_fence_only_registered_readers() -> Result<()> {
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.immediate_copies(&library)?;
        let device = Device { library: &library, id: 0 };
        let reader = Rc::new(Stream::new(device)?);
        let mut cfg = Inkling2Config::from_hf(&serde_json::from_str(include_str!(
            "../../../../cuteafd-loader/src/families/inkling2/fixtures/config.json"))?)?;
        cfg.layers = 2;
        cfg.local = vec![true, false];
        let cache = CacheBank::new(vec![reader.clone()], CacheLayout::new(&cfg, 2, 3)?, 1)?;
        let state = cache.sequence(0)?;
        let pattern = (0..state.bytes).map(|byte| (byte.wrapping_mul(17) % 251) as u8).collect::<Vec<_>>();
        library.copy_h2d(state, &pattern)?;
        cache.capture(0, 0)?;
        cache.reset(0)?;
        cache.restore(0, 1)?;
        cache.drain()?;
        let mut restored = vec![0; pattern.len()];
        library.copy_d2h(&mut restored, cache.sequence(1)?)?;
        assert_eq!(restored, pattern);
        library.copy_d2h(&mut restored, cache.sequence(0)?)?;
        assert!(restored.iter().all(|&byte| byte == 0));
        let page = cache.page(1, 0)?;
        let page_pattern = vec![0x5a; page.bytes];
        library.copy_h2d(page, &page_pattern)?;
        cache.copy_page(0, 2)?;
        cache.drain()?;
        let mut restored_page = vec![0; page.bytes];
        library.copy_d2h(&mut restored_page, cache.page(1, 2)?)?;
        assert_eq!(restored_page, page_pattern);
        let unit = super::super::owners::UnitOwner::new(reader, UnitLayout::serving(&cfg, 64, 256)?)?;
        assert!(unit.bind(&cache, 1, 0).is_ok());
        assert!(unit.bind(&cache, 2, 0).is_err());
        assert!(unit.bind(&cache, 0, 2).is_err());
        let stranger = Rc::new(Stream::new(device)?);
        assert!(cache.require_reader(&stranger).is_err());
        let unit = super::super::owners::UnitOwner::new(stranger, UnitLayout::serving(&cfg, 64, 256)?)?;
        assert!(unit.bind(&cache, 0, 0).is_err());
        assert!(cache.field(0, 1, "ring").is_err());
        assert!(cache.copy_page(0, 0).is_err());
        assert!(cache.copy_page(0, 3).is_err());
        Ok(())
    }
}

impl Drop for CacheBank<'_> {
    fn drop(&mut self) {
        crate::shared::decode_graph::fatal_drain(self.copies.drain(), "Inkling persistent cache retirement");
    }
}
