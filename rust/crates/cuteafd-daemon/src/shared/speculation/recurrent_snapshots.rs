//! Exact local-ring and recurrent-feature marks beside shared full-page leases.
use super::{block_dspark::WindowSnapshots,recurrent_forward::Rank,recurrent_cache::ContextBank};
use anyhow::{ensure,Context,Result};
use cuteafd_engine::prefix::DraftPageAttachment;
use cuteafd_ffi::CuteafdDeviceBuffer as Buffer;
use cuteafd_loader::formats::block_dspark::recurrent::ContextPolicy;
use std::collections::BTreeMap;

pub(crate) struct Marks<'a> {
    local:WindowSnapshots<'a>,
    pages:std::cell::RefCell<BTreeMap<usize,DraftPageAttachment>>,
}
fn part(buffer:Buffer,offset:usize,bytes:usize)->Result<Buffer>{
    ensure!(bytes>0 && offset.checked_add(bytes).is_some_and(|end|end<=buffer.bytes),"recurrent mark view exceeds backing");
    Ok(Buffer{ptr:buffer.ptr.wrapping_byte_add(offset),bytes,..buffer})
}
impl Rank<'_> {
    fn local_planes(&self,slot:usize)->Result<Vec<Buffer>> {
        ensure!(slot<self.layout.slots,"recurrent mark slot outside admission");
        let mut planes=Vec::new();
        for (index,policy) in self.config.context.iter().enumerate() {
            if let ContextPolicy::BlockAnchoredWindow(window)=policy {
                let bytes=window*self.layout.kv_cache.plane_record_bytes(self.layout.rank.kv_heads,self.config.transformer.head_dim);
                for plane in ["k","v"] {planes.push(part(self.buffer(&format!("cache.layer.{index}.{plane}"))?,slot*bytes,bytes)?);}
            }
        }
        let width=self.config.transformer.hidden*2;
        planes.push(part(self.buffer("cache.last_features")?,slot*width,width)?);
        Ok(planes)
    }
    fn full_tail_copy(&self,copy:cuteafd_engine::prefix::TailCopy)->Result<()> {
        let row=self.layout.cache.record_bytes/2;
        for (index,policy) in self.config.context.iter().enumerate() {
            if !matches!(policy,ContextPolicy::Full){continue;}
            for plane in ["k","v"] {
                let cache=self.buffer(&format!("cache.layer.{index}.{plane}"))?;
                // SAFETY: shared page-pool tail copy selects distinct retained
                // pages; the rank stream orders it before snapshot publication.
                unsafe{self.stream.device.library.copy_d2d_async(part(cache,copy.to as usize*64*row,copy.rows*row)?,
                    part(cache,copy.from as usize*64*row,copy.rows*row)?,copy.rows*row,self.stream.raw)?;}
            }
        }
        Ok(())
    }
}
impl<'a> Marks<'a> {
    pub fn new(rank:&Rank<'a>,marks:usize)->Result<Self>{
        let planes=rank.local_planes(0)?.iter().map(|plane|plane.bytes).collect::<Vec<_>>();
        let (stride,_)=WindowSnapshots::bytes(&planes,marks)?;
        ensure!(stride==rank.layout.window_mark_bytes,"recurrent mark planes differ from admission");
        Ok(Self{local:WindowSnapshots::new(&rank.stream,planes,marks)?,pages:Default::default()})
    }
    pub fn capture(&self,rank:&Rank<'a>,bank:&ContextBank,slot:usize,mark:usize)->Result<()> {
        let (attachment,copies)={
            let mut state=bank.state.borrow_mut();let binding=state.binding(slot)?;
            let (snapshot,copies)=state.pool.capture(binding)?;
            (state.pool.attachment(snapshot)?,copies)
        };
        for copy in copies{rank.full_tail_copy(copy)?;}
        // SAFETY: rank planes are disjoint retained local state; shared snapshot
        // fences join all arithmetic producers/readers before these copies.
        unsafe{self.local.capture(mark,&rank.local_planes(slot)?,attachment.origin,attachment.frontier)?;}
        self.local.drain()?;rank.stream.drain()?;
        if let Some(old)=self.pages.borrow_mut().insert(mark,attachment){
            bank.state.borrow_mut().pool.release_snapshot(old.snapshot)?;
        }
        Ok(())
    }
    pub fn restore(&self,rank:&Rank<'a>,bank:&ContextBank,slot:usize,mark:usize,frontier:usize)->Result<()> {
        let attachment=self.pages.borrow().get(&mark).cloned().context("recurrent mark has no full-context attachment")?;
        ensure!(attachment.frontier==frontier,"recurrent restored frontier differs");
        let copies={
            let mut state=bank.state.borrow_mut();
            if let Ok(binding)=state.binding(slot){state.pool.release(binding)?;}
            let (binding,copies)=state.pool.restore(slot,attachment.snapshot,frontier)?;
            state.install(slot,binding)?;copies
        };
        for copy in copies{rank.full_tail_copy(copy)?;}
        // SAFETY: initialized mark and exact slot planes survive ordered copies;
        // f_rnn restores byte-for-byte, never through a KV reconstruction.
        unsafe{self.local.restore(mark,&rank.local_planes(slot)?,frontier)?;}
        self.local.drain()?;rank.stream.drain()
    }
    pub fn discard(&self,bank:&ContextBank,mark:usize)->Result<()> {
        if let Some(old)=self.pages.borrow_mut().remove(&mark){bank.state.borrow_mut().pool.release_snapshot(old.snapshot)?;}
        Ok(())
    }
}
