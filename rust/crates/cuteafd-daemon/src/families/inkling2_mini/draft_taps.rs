//! Stream-ordered target tap staging for Inkling 2 Mini's bundled 3-layer MTP (`[5, 23, 41, 48]`).
use crate::shared::memory::{
    device::{Allocation, Stream},
    Tag,
};
use anyhow::{ensure, Result};
use cuteafd_ffi::CuteafdDeviceBuffer as Buffer;
use cuteafd_loader::formats::block_dspark::recurrent::RecurrentBlockConfig;
use std::{collections::BTreeMap, rc::Rc};

pub(crate) struct TapBuffer<'a> {
    pub stream: Rc<Stream<'a>>,
    pub output: Allocation<'a>,
    narrow: Allocation<'a>,
    capacity: usize,
    hidden: usize,
    taps: usize,
    visits: BTreeMap<cuteafd_core::TapSite, usize>,
}

impl<'a> TapBuffer<'a> {
    pub fn new(stream: Rc<Stream<'a>>, config: &RecurrentBlockConfig, rows: usize) -> Result<Self> {
        let hidden = config.transformer.hidden;
        let taps = config.taps.len();
        ensure!(rows > 0 && rows <= 4096, "Inkling 2 Mini tap staging row capacity");
        let output = Allocation::new(
            stream.device,
            rows * taps * hidden * 2,
            Tag::drafter_workspace("inkling2_mini/target-taps"),
        )?;
        let narrow = Allocation::new(
            stream.device,
            rows * hidden * 2,
            Tag::drafter_workspace("inkling2_mini/target-tap-narrow"),
        )?;
        let visits = config.taps.iter().copied().enumerate().map(|(tap, site)| (site, tap)).collect();
        Ok(Self {
            stream,
            output,
            narrow,
            capacity: rows,
            hidden,
            taps,
            visits,
        })
    }

    pub fn entry(&self, visit_idx: usize, raw_partial: Buffer, rows: usize) -> Result<()> {
        let Some(&tap) = self.visits.get(&cuteafd_core::TapSite::LayerEntry(visit_idx)) else {
            return Ok(());
        };
        ensure!(
            rows > 0 && rows <= self.capacity && raw_partial.bytes >= rows * self.hidden * 4,
            "Inkling 2 Mini tap source exceeds owner"
        );
        let library = self.stream.device.library;
        // SAFETY: live F32 partial on same stream narrowed & scattered into BF16 tap buffer.
        unsafe {
            library.load_narrow_f32_bf16(raw_partial.ptr, self.narrow.buffer.ptr, rows * self.hidden, self.stream.raw)?;
            library.copy_d2d_2d_async(
                Buffer {
                    ptr: self.output.buffer.ptr.wrapping_byte_add(tap * self.hidden * 2),
                    bytes: self.output.buffer.bytes - tap * self.hidden * 2,
                    ..self.output.buffer
                },
                self.taps * self.hidden * 2,
                self.narrow.buffer,
                self.hidden * 2,
                self.hidden * 2,
                rows,
                self.stream.raw,
            )
        }
    }

    pub fn final_norm(&self, hidden: Buffer, first: usize, rows: usize) -> Result<()> {
        let Some(&tap) = self.visits.get(&cuteafd_core::TapSite::FinalNorm) else {
            return Ok(());
        };
        ensure!(
            rows > 0 && first.checked_add(rows).is_some_and(|end| end <= self.capacity) && hidden.bytes >= rows * self.hidden * 2,
            "Inkling 2 Mini final tap outside staging"
        );
        let offset = (first * self.taps + tap) * self.hidden * 2;
        // SAFETY: final normalized BF16 rows copied without rerounding.
        unsafe {
            self.stream.device.library.copy_d2d_2d_async(
                Buffer {
                    ptr: self.output.buffer.ptr.wrapping_byte_add(offset),
                    bytes: self.output.buffer.bytes - offset,
                    ..self.output.buffer
                },
                self.taps * self.hidden * 2,
                hidden,
                self.hidden * 2,
                self.hidden * 2,
                rows,
                self.stream.raw,
            )
        }
    }
}
