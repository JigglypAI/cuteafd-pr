//! Shared page-pool reservations for mixed full/window block-drafter context.
use super::{recurrent_forward::Rank,dflash::DraftSeq};
use anyhow::{ensure,Context,Result};
use cuteafd_engine::prefix::{DraftBinding,DraftPagePool,DraftReservation};
use cuteafd_loader::formats::block_dspark::recurrent::ContextPolicy;
use cuteafd_ffi::CuteafdDeviceBuffer as Buffer;
use std::cell::RefCell;

pub(crate) struct ContextState {
    pub pool:DraftPagePool,
    bindings:Vec<Option<DraftBinding>>,
    slots:usize,
    context:usize,
}
impl ContextState {
    pub fn new(slots:usize,context:usize,target_tokens:usize,identity:u64)->Result<Self>{
        Ok(Self{pool:DraftPagePool::new(target_tokens,slots,context,identity)?,
            bindings:vec![None;slots],slots,context})
    }
    pub fn reset(&mut self,slot:usize,origin:usize)->Result<()> {
        ensure!(slot<self.slots && origin<=self.context,"recurrent context reset outside admission");
        if let Some(binding)=self.bindings[slot].take(){self.pool.release(binding)?;}
        self.bindings[slot]=Some(self.pool.bind(slot,origin)?);Ok(())
    }
    pub fn install(&mut self,slot:usize,binding:DraftBinding)->Result<()> {
        ensure!(slot<self.slots,"recurrent restored binding slot exceeds admission");
        self.bindings[slot]=Some(binding);Ok(())
    }
    pub fn binding(&self,slot:usize)->Result<DraftBinding>{
        self.bindings.get(slot).copied().flatten().context("unbound recurrent context slot")
    }
    pub fn pages(&self,sequences:&[DraftSeq])->Result<Vec<Vec<i32>>>{
        sequences.iter().map(|seq|{
            let binding=self.binding(seq.slot)?;
            let (origin,frontier)=self.pool.frontier(binding)?;
            ensure!((origin,frontier)==(seq.valid_from,seq.position),"recurrent draft differs from committed context frontier");
            Ok(self.pool.committed_page_table(binding)?)
        }).collect()
    }
}

pub(crate) struct ContextBank { pub state:RefCell<ContextState> }
impl ContextBank {
    /// Context inputs are one contiguous causal sequence span. The caller
    /// tiles prefill and packed members through this shared operation.
    pub fn append(&self,rank:&Rank<'_>,slot:usize,start:usize,features:Buffer,rows:usize)->Result<()> {
        ensure!(rows>0 && rows<=rank.layout.context_rows,"recurrent context append row capacity");
        let (reservation,physical)={
            let mut state=self.state.borrow_mut();let binding=state.binding(slot)?;
            let (origin,frontier)=state.pool.frontier(binding)?;
            ensure!(frontier==start && start.checked_add(rows).is_some_and(|end|end<=state.context)
                && origin<=start,"recurrent context append is not canonical");
            let reservation=state.pool.reserve(binding,start,rows)?;
            (reservation,state.pool.destinations(reservation)?)
        };
        let positions=(start..start+rows).map(|p|p as i64).collect::<Vec<_>>();
        let destinations=rank.config.context.iter().map(|policy|match policy {
            ContextPolicy::Full=>physical.clone(),ContextPolicy::BlockAnchoredWindow(window)=>
                (start..start+rows).map(|p|(slot*window+p%window) as i64).collect(),
        }).collect::<Vec<_>>();
        let result=(||{
            rank.store_context(features,&positions,&destinations)?;
            if rank.layout.rank.rank==rank.layout.rank.lead {
                let width=rank.config.transformer.hidden*2;
                let taps=rank.config.taps.len()*width;
                let last=Buffer{ptr:features.ptr.wrapping_byte_add((rows-1)*taps),bytes:taps,..features};
                let f=rank.buffer("cache.last_features")?;
                rank.recurrent_feature(last,Buffer{ptr:f.ptr.wrapping_byte_add(slot*width),bytes:width,..f})?;
            }
            rank.stream.drain()?;
            self.state.borrow_mut().pool.commit(reservation,rows)?;Ok(())
        })();
        if result.is_err(){
            // A failed producer may still read/write the provisional pages.
            // Never return them to the pool until native stream retirement.
            rank.stream.drain_or_abort("recurrent failed context append");
            self.state.borrow_mut().pool.commit(reservation,0)?;
        }
        result
    }
}
