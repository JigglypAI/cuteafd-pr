//! Exact local-ring/f_rnn marks. Shared PrefixFamily hooks separately own all
//! full-context page attachments, as for the vanilla block-dSpark integration.
use super::{block_dspark::WindowSnapshots,recurrent_forward::Rank};
use anyhow::{ensure,Result};
use cuteafd_ffi::CuteafdDeviceBuffer as Buffer;
use cuteafd_loader::formats::block_dspark::recurrent::ContextPolicy;

pub(crate) struct Marks<'a> { local:WindowSnapshots<'a> }
fn part(buffer:Buffer,offset:usize,bytes:usize)->Result<Buffer>{
    ensure!(bytes>0 && offset.checked_add(bytes).is_some_and(|end|end<=buffer.bytes),"recurrent mark view exceeds backing");
    Ok(Buffer{ptr:buffer.ptr.wrapping_byte_add(offset),bytes,..buffer})
}
impl Rank<'_> {
    pub(super) fn local_planes(&self,slot:usize)->Result<Vec<Buffer>> {
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
    pub(super) fn full_tail_copy(&self,copy:cuteafd_engine::prefix::TailCopy)->Result<()> {
        let row=self.layout.cache.record_bytes/2;
        for (index,policy) in self.config.context.iter().enumerate() {
            if !matches!(policy,ContextPolicy::Full){continue;}
            for plane in ["k","v"] {
                let cache=self.buffer(&format!("cache.layer.{index}.{plane}"))?;
                // SAFETY: shared page-pool tail copy selects distinct retained
                // pages, ordered before PrefixFamily snapshot publication.
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
        Ok(Self{local:WindowSnapshots::new(&rank.stream,planes,marks)?})
    }
    pub fn backing(&self)->std::rc::Rc<crate::shared::memory::device::Allocation<'a>>{self.local.backing()}
    pub fn bytes(&self)->usize {self.local.mark_bytes()}
    pub fn mark(&self,index:usize)->Result<Buffer>{self.local.mark(index)}
    pub fn capture(&self,rank:&Rank<'a>,slot:usize,mark:usize,origin:usize,frontier:usize)->Result<()> {
        // SAFETY: disjoint retained local planes include exact last-feature
        // bytes; registered arithmetic readers are fenced before these copies.
        unsafe{self.local.capture(mark,&rank.local_planes(slot)?,origin,frontier)}
    }
    pub fn restore(&self,rank:&Rank<'a>,slot:usize,mark:usize,frontier:usize)->Result<()> {
        // SAFETY: PrefixFamily selected this initialized exact mark and retains
        // both mark/live backings through publication and all ordered copies.
        unsafe{self.local.restore(mark,&rank.local_planes(slot)?,frontier)?;}
        Ok(())
    }
    pub fn drain(&self)->Result<()>{self.local.drain()}
}
