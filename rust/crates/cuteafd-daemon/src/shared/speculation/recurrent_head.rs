//! Seven sequential predictions over the lead rank's fully reduced block.
use super::{recurrent_forward::Rank,recurrent_weights::ProjectionWeight};
use anyhow::{ensure,Context,Result};
use cuteafd_ffi::CuteafdDeviceBuffer as Buffer;

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

    /// Returns device U32 [sequences,predictions] after the seven Markov steps.
    /// `step.normalized` is the final full-hidden block, replicated at the lead.
    /// Request-local f_rnn rows are already gathered from committed target taps.
    pub fn recurrent_tokens(&self,sequences:usize,anchor_ids:Buffer,target_head:Buffer)->Result<Buffer> {
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
        for step in 0..self.config.predictions() {
            let hidden=self.buffer("head.hidden")?;
            let source=self.buffer("step.normalized")?;
            // SAFETY: skip anchor row; each row is one sequence's mask at the
            // same position, copied to a contiguous step-major head input.
            unsafe{lib.copy_d2d_2d_async(hidden,c.hidden*2,
                part(source,(step+1)*c.hidden*2,source.bytes-(step+1)*c.hidden*2)?,
                c.block*c.hidden*2,c.hidden*2,sequences,stream)?;
                lib.cuda_embedding_lookup_bf16_async(head.token_embedding.buffer,ids,
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
            // SAFETY: admitted BF16 input/weight and disjoint F32 projection;
            // bias is added in FP32 by one batched launch before BF16 gates.
            unsafe{self.programs.f32.launch_checked(input.ptr,rnn_weight.buffer.ptr,
                self.buffer("head.projection")?,sequences,input_width,3*c.markov_rank,stream)?;
                programs.bias_raw.launch(&[self.buffer("head.projection")?,head.projection_bias.buffer,
                    self.buffer("head.raw")?],sequences,c.rms_norm_eps as f32,0,stream)?;
                programs.gate.launch(&[self.buffer("head.raw")?,state,next_state,self.buffer("head.latent")?],
                    sequences,c.rms_norm_eps as f32,0,stream)?;
                self.programs.plain.launch_checked(hidden.ptr,target_head.ptr,self.buffer("head.base")?,sequences,
                    c.hidden,self.config.padded_vocab,stream)?;
                self.programs.f32.launch_checked(self.buffer("head.latent")?.ptr,
                    self.weights.derived("derived.fold_weight")?.buffer.ptr,self.buffer("head.correction_f32")?,
                    sequences,c.markov_rank,self.config.padded_vocab,stream)?;
                programs.bias_correction.launch(&[self.buffer("head.correction_f32")?,
                    self.weights.derived("derived.fold_bias")?.buffer,self.buffer("head.correction")?],
                    sequences,c.rms_norm_eps as f32,0,stream)?;
                programs.finish.launch(&[self.buffer("head.base")?,self.buffer("head.correction")?,
                    self.buffer("head.logits_bf16")?],sequences,c.rms_norm_eps as f32,0,stream)?;
                lib.load_widen_bf16_f32(self.buffer("head.logits_bf16")?.ptr,self.buffer("head.logits")?.ptr,
                    sequences*self.config.valid_vocab,stream)?;
                lib.cuda_logits_greedy_f32_async(self.buffer("head.logits")?.ptr,sequences,self.config.valid_vocab,
                    self.config.valid_vocab,ids.ptr,std::ptr::null_mut(),self.buffer("head.status")?.ptr,stream)?;
                lib.copy_d2d_2d_async(part(self.buffer("head.tokens")?,step*4,
                    self.buffer("head.tokens")?.bytes-step*4)?,self.config.predictions()*4,ids,4,4,sequences,stream)?;
                lib.copy_d2d_async(state,next_state,sequences*c.markov_rank*2,stream)?;
            }
        }
        self.buffer("head.tokens")
    }
}
