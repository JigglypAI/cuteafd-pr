//! Same page-attachment lifecycle exposed by the shared vanilla block drafter.
use super::recurrent::Drafter;
use anyhow::{ensure,Result};
use cuteafd_engine::prefix::{DraftPageAttachment,DraftPageId};
use cuteafd_ffi::CuteafdDeviceBuffer as Buffer;
use cuteafd_loader::formats::block_dspark::recurrent::ContextPolicy;

impl Drafter<'_>{
    fn copies(&self,copies:Vec<cuteafd_engine::prefix::TailCopy>)->Result<()> {
        for copy in copies {
            self.lead.full_tail_copy(copy)?;
            if let Some(split)=&self.split{split.peer.stream.device.run(||split.peer.full_tail_copy(copy))?;}
        }
        Ok(())
    }
    pub fn admit_pages(&self,slot:usize,end:usize)->Result<()>{
        let mut state=self.context.state.borrow_mut();let binding=state.binding(slot)?;
        Ok(state.pool.admit(binding,end,self.lead.config.transformer.block)?)
    }
    pub fn capture_pages(&self,slot:usize,frontier:usize)->Result<DraftPageAttachment>{
        let (attachment,copies)={
            let mut state=self.context.state.borrow_mut();let binding=state.binding(slot)?;
            ensure!(state.pool.frontier(binding)?.1==frontier,"draft capture frontier differs from target");
            let (snapshot,copies)=state.pool.capture(binding)?;
            (state.pool.attachment(snapshot)?,copies)
        };
        if let Err(error)=self.copies(copies){
            self.lead.stream.drain_or_abort("recurrent failed prefix capture");
            if let Some(split)=&self.split{split.peer.stream.drain_or_abort("recurrent failed peer prefix capture");}
            self.context.state.borrow_mut().pool.release_snapshot(attachment.snapshot)?;
            return Err(error);
        }
        Ok(attachment)
    }
    pub fn pages_valid(&self,attachment:&DraftPageAttachment)->bool{
        self.context.state.borrow().pool.check_attachment(attachment).is_ok()
    }
    pub fn pages_revocable(&self,attachment:&DraftPageAttachment)->bool{
        let state=self.context.state.borrow();state.pool.check_attachment(attachment).is_ok()
            && state.pool.snapshot_revocable(attachment.snapshot)
    }
    pub fn retain_pages(&self,attachment:&DraftPageAttachment)->Result<()>{
        let mut state=self.context.state.borrow_mut();state.pool.check_attachment(attachment)?;
        Ok(state.pool.retain_snapshot(attachment.snapshot)?)
    }
    pub fn release_pages(&self,attachment:&DraftPageAttachment)->Result<()>{
        self.lead.stream.drain()?;
        if let Some(split)=&self.split{split.peer.stream.drain()?;}
        let mut state=self.context.state.borrow_mut();state.pool.check_attachment(attachment)?;
        Ok(state.pool.release_snapshot(attachment.snapshot)?)
    }
    pub fn restore_pages(&self,slot:usize,attachment:&DraftPageAttachment,frontier:usize)->Result<()>{
        let copies={
            let mut state=self.context.state.borrow_mut();state.pool.check_attachment(attachment)?;
            ensure!(attachment.frontier==frontier,"draft restore differs from target frontier");
            if let Ok(binding)=state.binding(slot){state.pool.release(binding)?;}
            let (binding,copies)=state.pool.restore(slot,attachment.snapshot,frontier)?;
            state.install(slot,binding)?;copies
        };
        self.copies(copies)
    }
    pub fn promote_pages(&self,saved:&DraftPageAttachment)->Result<DraftPageAttachment>{
        Ok(self.context.state.borrow_mut().pool.promote(saved)?)
    }
    pub fn page_segments(&self,page:DraftPageId)->Result<Vec<Buffer>>{
        ensure!((page.index as usize)<self.lead.layout.cache.pages,"draft page outside admitted pool");
        let mut output=Vec::new();
        let mut rank=|rank:&super::recurrent_forward::Rank<'_>|->Result<()>{
            let bytes=64*rank.layout.cache.record_bytes/2;
            for (layer,policy) in rank.config.context.iter().enumerate(){
                if !matches!(policy,ContextPolicy::Full){continue;}
                for plane in ["k","v"]{
                    let b=rank.buffer(&format!("cache.layer.{layer}.{plane}"))?;
                    output.push(Buffer{ptr:b.ptr.wrapping_byte_add(page.index as usize*bytes),bytes,..b});
                }
            }Ok(())
        };
        rank(&self.lead)?;if let Some(split)=&self.split{rank(&split.peer)?;}Ok(output)
    }
}
