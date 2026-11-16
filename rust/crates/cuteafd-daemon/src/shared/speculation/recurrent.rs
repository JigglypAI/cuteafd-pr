//! One target-independent recurrent block drafter over shared cache/TP2 owners.
use super::{recurrent_forward::Rank,recurrent_cache::ContextBank,dflash::{Draft,DraftSeq},gqa_tp2::GqaHeadSplit};
use anyhow::{ensure,Result};
use cuteafd_ffi::CuteafdDeviceBuffer as Buffer;
use std::{cell::Cell,rc::Rc};

pub(crate) struct Drafter<'a> {
    // Every producer drains in Drop before rank/peer/cache storage retires.
    pub lead:Rank<'a>,
    pub split:Option<GqaHeadSplit<'a,Rank<'a>>>,
    pub context:ContextBank,
    live:Cell<bool>,
}
impl Drafter<'_> {
    pub fn draft(&self,sequences:&[DraftSeq],embedding:Buffer,embed_norm:Buffer,head:Buffer)->Result<Vec<Draft>> {
        ensure!(self.live.get(),"recurrent drafter is terminal");
        let result=(||{
            let pages=self.context.state.borrow().pages(sequences)?;
            let anchors=sequences.iter().map(|seq|seq.anchor.to_le_bytes()).flatten().collect::<Vec<_>>();
            let lib=self.lead.stream.device.library;let stream=self.lead.stream.raw;
            // SAFETY: permanent metadata owner and stream retain staging; no
            // host hop occurs for recurrent state, features or intermediate IDs.
            unsafe{lib.copy_h2d_ordered(self.lead.buffer("step.ids")?,&anchors,stream)?;}
            self.lead.initialize_block(self.lead.buffer("step.ids")?,embedding,embed_norm,sequences.len())?;
            for (index,seq) in sequences.iter().enumerate(){
                let source=self.lead.buffer("cache.last_features")?;
                let destination=self.lead.buffer("head.f_rnn")?;
                let width=self.lead.config.transformer.hidden*2;
                // SAFETY: exact persistent last-feature row copied into packed
                // live head sequence order, with both owners retained.
                unsafe{lib.copy_d2d_async(Buffer{ptr:destination.ptr.wrapping_byte_add(index*width),bytes:width,..destination},
                    Buffer{ptr:source.ptr.wrapping_byte_add(seq.slot*width),bytes:width,..source},width,stream)?;}
            }
            match &self.split {
                Some(split)=>super::recurrent_tp2::forward(&self.lead,split,sequences,&pages)?,
                None=>self.lead.forward_whole(sequences,&pages)?,
            }
            self.lead.recurrent_tokens(sequences.len(),self.lead.buffer("step.ids")?,head)?;
            self.lead.stream.drain()?;
            let mut bytes=vec![0;sequences.len()*self.lead.config.predictions()*4];
            lib.copy_d2h(&mut bytes,Buffer{bytes:bytes.len(),..self.lead.buffer("head.tokens")?})?;
            let tokens=bytes.chunks_exact(4).map(|v|u32::from_le_bytes(v.try_into().unwrap())).collect::<Vec<_>>();
            Ok(tokens.chunks_exact(self.lead.config.predictions()).map(|tokens|Draft{tokens:tokens.to_vec(),features:vec![],confidence:vec![]}).collect())
        })();
        if result.is_err(){self.live.set(false);self.lead.stream.drain_or_abort("recurrent drafter failed step");
            if let Some(split)=&self.split{split.peer.stream.drain_or_abort("recurrent drafter failed peer step");}}
        result
    }
}
impl Drop for Drafter<'_> {
    fn drop(&mut self){
        self.live.set(false);
        self.lead.stream.drain_or_abort("recurrent drafter retirement");
        if let Some(split)=&self.split{split.peer.stream.drain_or_abort("recurrent peer retirement");}
    }
}
