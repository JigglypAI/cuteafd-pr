//! Inkling 2 Mini TP1 binding of the shared 3-layer recurrent dSpark drafter & FIR verifier.
use super::{draft_taps::TapBuffer, verify::Verify};
use crate::shared::{
    memory::device::Stream,
    speculation::{
        recurrent::Drafter,
        recurrent_cache::{ContextBank, ContextState},
        recurrent_forward::{ProgramsForRank, Rank},
        recurrent_snapshots::Marks,
        recurrent_weights::Weights,
    },
};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{programs::Programs, CuteafdDeviceBuffer as Buffer};
use cuteafd_loader::{families::inkling2::draft::DraftLayout, placement::planner::ResolvedPlan};
use std::{cell::RefCell, rc::Rc};

pub(crate) struct Binding<'a> {
    pub drafter: Rc<Drafter<'a>>,
    pub marks: Vec<Marks<'a>>,
    pub verify: Verify<'a>,
    pub prefill: Vec<TapBuffer<'a>>,
    pub decode: TapBuffer<'a>,
    pub layout: DraftLayout,
}

impl crate::shared::prefix::cuda_copy::SnapshotOwner for Binding<'_> {
    fn snapshot_buffers(&self) -> Result<Vec<Buffer>> {
        let mut output = self.marks.first().map(|m| m.backing().buffer).into_iter().collect::<Vec<_>>();
        for (layer, policy) in self.layout.config.context.iter().enumerate() {
            if matches!(policy, cuteafd_loader::formats::block_dspark::recurrent::ContextPolicy::Full) {
                for plane in ["k", "v"] {
                    output.push(
                        self.drafter
                            .lead
                            .buffers
                            .get(&format!("cache.layer.{layer}.{plane}"))
                            .context("Mini snapshot full drafter KV owner missing")?
                            .buffer,
                    );
                }
            }
        }
        Ok(output)
    }
}

impl<'a> Binding<'a> {
    pub fn ranks(&self) -> impl Iterator<Item = &Rank<'a>> {
        std::iter::once(&self.drafter.lead)
    }

    pub fn new(
        programs: &'a Programs<'a>,
        stream: Rc<Stream<'a>>,
        weights: Weights<'a>,
        layout: DraftLayout,
        admitted: &ResolvedPlan,
        unembed_weight: Buffer,
        prefill_rows: usize,
        waves: usize,
        mark_slots: usize,
    ) -> Result<Self> {
        ensure!(
            layout.weights.len() == 1 && layout.runtime.len() == 1,
            "Inkling 2 Mini drafter must be single-rank TP1"
        );
        let check = |id: &str, bytes: usize| -> Result<()> {
            let owner = admitted
                .facts
                .allocations
                .iter()
                .find(|owner| owner.id == id)
                .with_context(|| format!("unadmitted Mini drafter owner {id}"))?;
            ensure!(
                owner.bytes == bytes as u64
                    && owner.location == cuteafd_loader::placement::planner::Location::coordinator(0),
                "Mini drafter runtime owner differs from admission {id}"
            );
            Ok(())
        };
        let runtime = &layout.runtime[0];
        check(layout.weights[0].arena.id(), layout.weights[0].arena.bytes())?;
        for (name, bytes) in &runtime.buffers {
            check(&format!("inkling2_mini.draft.0.{name}"), *bytes)?;
        }
        if mark_slots > 0 {
            check("inkling2_mini.draft.0.marks", runtime.window_mark_bytes * mark_slots)?;
        }
        for (name, bytes) in &layout.target {
            check(&format!("inkling2_mini.draft.target.{name}"), *bytes)?;
        }

        let lead = stream.device.run(|| {
            let kernels = ProgramsForRank::load(programs, &weights, &layout.config, runtime, stream.device)?;
            let rank = Rank::allocate(weights, Rc::clone(&stream), layout.config.clone(), runtime.clone(), kernels)?;
            rank.initialize_derived()?;
            rank.fold_head(unembed_weight)?;
            stream.drain()?;
            Ok(rank)
        })?;
        let context = ContextBank {
            state: RefCell::new(ContextState::new(
                runtime.slots,
                usize::try_from(admitted.served_context())?,
                runtime.cache.target_tokens,
                0x494e4b4d494e49,
            )?),
        };
        let drafter = Rc::new(Drafter::whole(lead, context)?);
        let marks = if mark_slots == 0 {
            Vec::new()
        } else {
            vec![drafter.lead.stream.device.run(|| Marks::new(&drafter.lead, mark_slots))?]
        };
        let verify = stream
            .device
            .run(|| Verify::new(programs, Rc::clone(&stream), runtime.slots, 6656))?;
        let prefill = (0..waves)
            .map(|_| TapBuffer::new(Rc::clone(&stream), &layout.config, prefill_rows))
            .collect::<Result<Vec<_>>>()?;
        let decode = TapBuffer::new(Rc::clone(&stream), &layout.config, 64)?;
        stream.drain()?;
        Ok(Self {
            drafter,
            marks,
            verify,
            prefill,
            decode,
            layout,
        })
    }

    pub fn select(&self, sequences: usize, anchors: Buffer, unembed_weight: Buffer) -> Result<Buffer> {
        self.drafter.lead.recurrent_tokens(sequences, anchors, unembed_weight)
    }

    pub fn taps(&self, decode: bool, unit: usize) -> Result<&TapBuffer<'a>> {
        if decode {
            Ok(&self.decode)
        } else {
            self.prefill
                .get(unit)
                .context("Mini drafter prefill tap unit outside admission")
        }
    }

    pub fn update(&self, taps: &TapBuffer<'a>, slot: usize, start: usize, first: usize, rows: usize) -> Result<()> {
        let pitch = self.layout.config.taps.len() * self.layout.config.transformer.hidden * 2;
        for offset in (0..rows).step_by(self.layout.runtime[0].context_rows) {
            let count = (rows - offset).min(self.layout.runtime[0].context_rows);
            let position = (first + offset) * pitch;
            let output = taps.output.buffer;
            ensure!(
                position.checked_add(count * pitch).is_some_and(|end| end <= output.bytes),
                "Mini drafter target taps outside owner"
            );
            self.drafter.append_context(
                slot,
                start + offset,
                Buffer {
                    ptr: output.ptr.wrapping_byte_add(position),
                    bytes: count * pitch,
                    ..output
                },
                count,
            )?;
        }
        Ok(())
    }
}
