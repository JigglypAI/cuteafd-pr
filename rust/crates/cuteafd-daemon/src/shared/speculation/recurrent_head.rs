//! Seven sequential predictions over the lead rank's fully reduced block.
use super::{recurrent_forward::Rank,recurrent_weights::ProjectionWeight};
use anyhow::{ensure,Context,Result};
use cuteafd_ffi::CuteafdDeviceBuffer as Buffer;

#[derive(Clone, Copy, Debug)]
pub(crate) enum HeadStage {
    Begin, Gather, Recurrent, Base, Correction, Bias, Finish, Widen, Argmax, Copies,
}

fn part(buffer:Buffer,offset:usize,bytes:usize)->Result<Buffer> {
    ensure!(bytes>0 && offset.checked_add(bytes).is_some_and(|end|end<=buffer.bytes),"recurrent head view outside owner");
    Ok(Buffer{ptr:buffer.ptr.wrapping_byte_add(offset),bytes,..buffer})
}

impl Rank<'_> {
    /// Called once after all push records commit and before readiness.
    pub fn fold_head(&self,target_head:Buffer)->Result<()> {
        let c=&self.config.transformer;
        let head=self.weights.tensors.head.as_ref().context("peer has no recurrent head")?;
        let programs=self.programs.head.as_ref().context("head programs not bound")?;
        let ProjectionWeight::Bf16(hidden_out)=&head.hidden_out[0] else {
            anyhow::bail!("recurrent hidden_out must remain native BF16");
        };
        ensure!(target_head.bytes==self.config.padded_vocab*c.hidden*2,"recurrent fold requires native borrowed target head");
        for first in (0..self.config.padded_vocab).step_by(4096) {
            let rows=4096.min(self.config.padded_vocab-first);
            // SAFETY: input head is borrowed native BF16; all folded output
            // views retain the admitted drafter backing through readiness/drain.
            unsafe{programs.fold.launch(&[target_head,hidden_out.buffer,head.hidden_out_bias.buffer,
                self.weights.derived("derived.fold_weight")?.buffer,self.weights.derived("derived.fold_bias")?.buffer],
                rows,c.rms_norm_eps as f32,first,self.stream.raw)?;}
        }
        Ok(())
    }

    /// Project tiles of H_out^T through the borrowed packed head, then round
    /// once into the admitted folded arena. The caller lends idle target head
    /// stages, which remain live through the startup stream drain.
    pub fn fold_compact_head(&self, programs: &cuteafd_ffi::programs::Programs<'_>, stored: usize,
        stage: Buffer, transpose: Buffer, mut project: impl FnMut(Buffer, Buffer, usize) -> Result<()>) -> Result<()> {
        use crate::shared::experts::rtx::dense::Finish;
        use cuteafd_core::DType;
        use cuteafd_loader::families::inkling2::draft::{fold_programs, FOLD_TILE as TILE};
        let c = &self.config.transformer;
        let n = self.config.padded_vocab;
        let (h, r) = (c.hidden, c.markov_rank);
        ensure!(r % TILE == 0 && stored >= n, "invalid compact fold geometry");
        let lib = self.stream.device.library;
        let stream = self.stream.raw;
        let head = self.weights.tensors.head.as_ref().context("peer has no recurrent head")?;
        let ProjectionWeight::Bf16(hidden_out) = &head.hidden_out[0] else {
            anyhow::bail!("recurrent hidden_out must remain native BF16");
        };
        let input = part(self.buffer("projection.bf16")?, 0, h*r*2)?;
        let fold = self.weights.derived("derived.fold_weight")?.buffer;
        let bias = self.weights.derived("derived.fold_bias")?.buffer;
        let narrow = lib.f32_bf16_narrow()?;
        let stage = part(stage, 0, stored*TILE*4)?;
        let transpose = part(transpose, 0, stored*TILE*4)?;
        let output = part(stage, 0, stored*TILE*2)?;
        let mut guard = self.stream.cancellation_guard("compact recurrent startup fold");
        Finish::PaddedTranspose { source: hidden_out.buffer, output: input, dtype: DType::Bf16,
            rows: h, cols: r, padded_cols: h }.run(lib, stream)?;
        let [_, egress_name] = fold_programs(h, r, stored);
        let egress = programs.program(&egress_name, &["source", "output"])?;
        ensure!(egress.spec().scalar_kinds.is_empty(), "compact fold egress ABI");
        for first in (0..r).step_by(TILE) {
            project(part(input, first*h*2, TILE*h*2)?, stage, TILE)?;
            // SAFETY: the stage and transpose spans are disjoint. The narrowed
            // tile only overwrites the stage after the ordered transpose reads it.
            unsafe {
                egress.launch(&[stage.ptr, transpose.ptr], &[], stream)?;
                narrow.launch(transpose.ptr, output.ptr, stored*TILE, stream)?;
                lib.copy_d2d_2d_async(part(fold, first*2, fold.bytes-first*2)?, r*2,
                    output, TILE*2, TILE*2, n, stream)?;
            }
        }
        project(head.hidden_out_bias.buffer, stage, 1)?;
        // SAFETY: only the logical vocabulary is written; source and bias are disjoint.
        unsafe { narrow.launch(stage.ptr, bias.ptr, n, stream)?; }
        self.stream.drain()?;
        guard.complete = true;
        Ok(())
    }

    /// Returns device U32 [sequences,predictions] after the seven Markov steps.
    /// `step.normalized` is the final full-hidden block, replicated at the lead.
    /// Request-local f_rnn rows are already gathered from committed target taps.
    pub fn recurrent_tokens(&self,sequences:usize,anchor_ids:Buffer,target_head:Buffer)->Result<Buffer> {
        self.recurrent_tokens_measured(sequences, anchor_ids, Some(sequences*self.config.predictions()), |hidden, output, rows| {
            // SAFETY: native BF16 head and admitted inputs/output outlive the stream.
            unsafe { self.programs.plain.launch_checked(hidden.ptr, target_head.ptr, output, rows,
                self.config.transformer.hidden, self.config.padded_vocab, self.stream.raw)?; }
            Ok(())
        }, |_, _| Ok(()))
    }

    pub(crate) fn recurrent_tokens_measured(&self, sequences:usize, anchor_ids:Buffer,
        base_tile: Option<usize>, base: impl Fn(Buffer, Buffer, usize) -> Result<()>, mut stamp: impl FnMut(usize, HeadStage) -> Result<()>) -> Result<Buffer> {
        let c=&self.config.transformer;let lib=self.stream.device.library;let stream=self.stream.raw;
        ensure!(sequences>0 && sequences<=self.layout.sequences,"recurrent head live sequences exceed admission");
        let head=self.weights.tensors.head.as_ref().context("peer has no recurrent head")?;
        let programs=self.programs.head.as_ref().context("head programs absent")?;
        let ProjectionWeight::Bf16(rnn_weight)=&head.projection[0] else {anyhow::bail!("RNN projection is native BF16");};
        let state=self.buffer("head.state")?;let next_state=self.buffer("head.next_state")?;
        let ids=self.buffer("head.previous")?;
        // SAFETY: only live state rows/ids initialize on the same retained
        // stream; no host bounce occurs between sequential sampling steps.
        unsafe{lib.cuda_zero_bytes_async(state,sequences*c.markov_rank*2,stream)?;
            lib.copy_d2d_async(ids,anchor_ids,sequences*4,stream)?;}
        let input_width=self.config.recurrent_input_width()?;
        let input=self.buffer("head.inputs")?;
        let predictions=self.config.predictions();
        let hidden_pitch=c.hidden*2;
        let base_pitch=self.config.padded_vocab*2;
        let hidden_rows=self.buffer("head.hidden")?;
        let base_rows=self.buffer("head.base")?;
        let source=self.buffer("step.normalized")?;
        let gather=|step:usize, output:Buffer| -> Result<()> {
            let offset=(step+1)*hidden_pitch;
            // SAFETY: skip the anchor and pack one independent mask row per
            // sequence while retaining the fully reduced source block.
            unsafe {lib.copy_d2d_2d_async(output,hidden_pitch,
                part(source,offset,source.bytes-offset)?,c.block*hidden_pitch,
                hidden_pitch,sequences,stream)}
        };
        let step_hidden=|step:usize| part(hidden_rows,step*sequences*hidden_pitch,sequences*hidden_pitch);
        let step_base=|step:usize| part(base_rows,step*sequences*base_pitch,sequences*base_pitch);
        if let Some(tile)=base_tile {
            ensure!(tile>0,"recurrent head projection tile must be nonzero");
            // The draft block is already complete: all base logits depend
            // only on mask hidden rows. Markov state and feedback start below.
            for step in 0..predictions { gather(step,step_hidden(step)?)?; }
            let live=sequences*predictions;
            for first in (0..live).step_by(tile) {
                let rows=(live-first).min(tile);
                base(part(hidden_rows,first*hidden_pitch,rows*hidden_pitch)?,
                    part(base_rows,first*base_pitch,rows*base_pitch)?,rows)?;
            }
        }
        for step in 0..self.config.predictions() {
            stamp(step, HeadStage::Begin)?;
            let row=if base_tile.is_some(){step}else{0};
            let hidden=step_hidden(row)?;
            let base_logits=step_base(row)?;
            if base_tile.is_none(){gather(step,hidden)?;}
            // SAFETY: the previous token is produced in stream order.
            unsafe{lib.cuda_embedding_lookup_bf16_async(head.token_embedding.buffer,ids,
                self.buffer("head.previous_embedding")?,sequences,self.config.padded_vocab,c.markov_rank,stream)?;}
            let mut offset=0;
            for (source,width) in [(state,c.markov_rank),(self.buffer("head.previous_embedding")?,c.markov_rank),
                (hidden,c.hidden),(self.buffer("head.f_rnn")?,c.hidden)] {
                // SAFETY: concatenation writes disjoint column windows; input
                // planes remain live through the ensuing batched GEMM.
                unsafe{lib.copy_d2d_2d_async(part(input,offset,input.bytes-offset)?,input_width*2,
                    source,width*2,width*2,sequences,stream)?;}
                offset+=width*2;
            }
            stamp(step, HeadStage::Gather)?;
            // SAFETY: admitted BF16 input/weight and disjoint F32 projection;
            // bias is added in FP32 by one batched launch before BF16 gates.
            unsafe{self.programs.f32.launch_checked(input.ptr,rnn_weight.buffer.ptr,
                self.buffer("head.projection")?,sequences,input_width,3*c.markov_rank,stream)?;
                programs.bias_raw.launch(&[self.buffer("head.projection")?,head.projection_bias.buffer,
                    self.buffer("head.raw")?],sequences,c.rms_norm_eps as f32,0,stream)?;
                programs.gate.launch(&[self.buffer("head.raw")?,state,next_state,self.buffer("head.latent")?],
                    sequences,c.rms_norm_eps as f32,0,stream)?;
                stamp(step, HeadStage::Recurrent)?;
                if base_tile.is_none(){ base(hidden,base_logits,sequences)?; }
                stamp(step, HeadStage::Base)?;
                self.programs.f32.launch_checked(self.buffer("head.latent")?.ptr,
                    self.weights.derived("derived.fold_weight")?.buffer.ptr,self.buffer("head.correction_f32")?,
                    sequences,c.markov_rank,self.config.padded_vocab,stream)?;
                stamp(step, HeadStage::Correction)?;
                programs.bias_correction.launch(&[self.buffer("head.correction_f32")?,
                    self.weights.derived("derived.fold_bias")?.buffer,self.buffer("head.correction")?],
                    sequences,c.rms_norm_eps as f32,0,stream)?;
                stamp(step, HeadStage::Bias)?;
                programs.finish.launch(&[base_logits,self.buffer("head.correction")?,
                    self.buffer("head.logits_bf16")?],sequences,c.rms_norm_eps as f32,0,stream)?;
                stamp(step, HeadStage::Finish)?;
                lib.load_widen_bf16_f32(self.buffer("head.logits_bf16")?.ptr,self.buffer("head.logits")?.ptr,
                    sequences*self.config.valid_vocab,stream)?;
                stamp(step, HeadStage::Widen)?;
                lib.cuda_logits_greedy_f32_async(self.buffer("head.logits")?.ptr,sequences,self.config.valid_vocab,
                    self.config.valid_vocab,ids.ptr,std::ptr::null_mut(),self.buffer("head.status")?.ptr,stream)?;
                stamp(step, HeadStage::Argmax)?;
                lib.copy_d2d_2d_async(part(self.buffer("head.tokens")?,step*4,
                    self.buffer("head.tokens")?.bytes-step*4)?,self.config.predictions()*4,ids,4,4,sequences,stream)?;
                lib.copy_d2d_async(state,next_state,sequences*c.markov_rank*2,stream)?;
            }
            stamp(step, HeadStage::Copies)?;
        }
        self.buffer("head.tokens")
    }
}
