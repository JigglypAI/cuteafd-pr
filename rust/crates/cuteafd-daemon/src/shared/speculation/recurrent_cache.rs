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
    /// Shared slot-binding entry point for both serving (`PrefixFamily::bind`) and
    /// golden/oracle runners: resets/binds `slot` at `origin` and, when `admit` is
    /// `Some((end, block))`, admits draft pages through `end` for the given block size.
    pub fn bind_slot(&mut self,slot:usize,origin:usize,admit:Option<(usize,usize)>)->Result<DraftBinding> {
        self.reset(slot,origin)?;
        let binding=self.binding(slot)?;
        if let Some((end,block))=admit {
            self.pool.admit(binding,end,block)?;
        }
        Ok(binding)
    }
    pub fn quarantine(&mut self,slot:usize) {
        if let Some(binding)=self.bindings.get(slot).copied().flatten() {
            self.pool.quarantine(binding).expect("live recurrent binding belongs to its pool");
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn quarantined_recurrent_destinations_survive_normal_cleanup() -> Result<()> {
        let mut state=ContextState::new(2,1024,8*64,88)?;
        state.reset(0,0)?;
        let binding=state.binding(0)?;
        let pending=state.pool.reserve(binding,0,3*64)?;
        let destinations=state.pool.destinations(pending)?;
        state.quarantine(0);
        assert!(state.reset(0,0).is_err(),"quarantined slot must never be rebound");
        assert_eq!(state.pool.free(),5);
        state.reset(1,0)?;
        let next=state.pool.reserve(state.binding(1)?,0,5*64)?;
        assert!(state.pool.destinations(next)?.iter().all(|row|!destinations.contains(row)));
        assert_eq!(state.pool.free(),0);
        Ok(())
    }

    #[test]
    fn bind_slot_binds_every_slot_and_rebinds_without_leaking_pages() -> Result<()> {
        // Exercise both Inkling 2 Mini (4 slots, block=8, 48-visit golden width-invariance/graph-parity)
        // and Inkling 2 Small (16 slots, block=8, width-invariance) recurrent page-pool geometries.
        for (slots, context, target_tokens, block, end, seed) in [
            (4usize, 4096usize, 32 * 64, 8usize, 127usize, 2001u64),
            (16usize, 4096usize, 64 * 64, 8usize, 72usize, 2002u64),
        ] {
            let mut state = ContextState::new(slots, context, target_tokens, seed)?;
            let initial_free = state.pool.free();
            // Unbound slots must fail with the exact guard error seen when a golden skips binding.
            for slot in 0..slots {
                let err = state.binding(slot).unwrap_err();
                assert!(err.to_string().contains("unbound recurrent context slot"), "{err:#}");
            }
            // Binding every slot through the shared entry point admits draft pages for prefill.
            for slot in 0..slots {
                let binding = state.bind_slot(slot, 0, Some((end, block)))?;
                assert_eq!(state.binding(slot)?, binding);
                let reservation = state.pool.reserve(binding, 0, end)?;
                state.pool.commit(reservation, end)?;
                assert_eq!(state.pool.frontier(binding)?, (0, end));
            }
            // Repeatedly re-binding a slot (width-invariance & graph-parity replay loops)
            // releases its prior draft pages cleanly without leaking pool capacity.
            let free_after_bind = state.pool.free();
            for _ in 0..6 {
                let binding = state.bind_slot(1, 0, Some((end, block)))?;
                let reservation = state.pool.reserve(binding, 0, end)?;
                state.pool.commit(reservation, end)?;
            }
            assert_eq!(state.pool.free(), free_after_bind);
            // Releasing every slot returns all pool pages.
            for slot in 0..slots {
                state.bind_slot(slot, 0, None)?;
            }
            assert_eq!(state.pool.free(), initial_free);
        }
        Ok(())
    }

    #[test]
    fn golden_and_serve_share_recurrent_slot_binding_across_families() {
        // Inkling 2 Mini: serve (`MiniPrefix::bind`) and golden (`inkling2_mini/golden.rs`)
        // must both go through `MiniPrefix::bind_slot` so drafter context binding never diverges.
        let mini_cache = include_str!("../../families/inkling2_mini/cache.rs");
        let mini_golden = include_str!("../../families/inkling2_mini/golden.rs");
        assert!(mini_cache.contains("pub fn bind_slot("), "MiniPrefix must expose shared bind_slot");
        let mini_serve_bind = mini_cache.split("fn bind(&self, placement: &mut Placement)").nth(1)
            .expect("MiniPrefix::bind").split("fn discard(").next().expect("MiniPrefix::bind body");
        assert!(mini_serve_bind.contains("self.bind_slot(placement, None)"),
            "Mini serve bind must delegate to MiniPrefix::bind_slot");
        assert!(mini_golden.contains("MiniPrefix::for_execution(&execution)"),
            "Mini golden must build MiniPrefix::for_execution");
        assert!(mini_golden.contains("prefix.bind_slot("),
            "Mini golden must bind slots via shared MiniPrefix::bind_slot");
        assert!(!mini_golden.contains("requests.cache.reset(") && !mini_golden.contains("drafter.reset_context("),
            "Mini golden must not bypass MiniPrefix::bind_slot with raw reset calls");

        // Inkling 2 Small: serve (`InklingPrefix::bind`) and width-invariance golden
        // (`inkling2/width_invariance.rs`) must both go through `InklingPrefix::bind_slot`.
        let small_cache = include_str!("../../families/inkling2/cache.rs");
        let small_wi = include_str!("../../families/inkling2/width_invariance.rs");
        assert!(small_cache.contains("pub fn bind_slot("), "InklingPrefix must expose shared bind_slot");
        let small_serve_bind = small_cache.split("fn bind(&self, placement: &mut Placement)").nth(1)
            .expect("InklingPrefix::bind").split("fn discard(").next().expect("InklingPrefix::bind body");
        assert!(small_serve_bind.contains("self.bind_slot(placement, None)"),
            "Inkling 2 Small serve bind must delegate to InklingPrefix::bind_slot");
        assert!(small_wi.contains("InklingPrefix::for_execution(&execution)"),
            "Inkling 2 Small width-invariance golden must build InklingPrefix::for_execution");
        assert!(small_wi.contains("prefix.bind_slot("),
            "Inkling 2 Small width-invariance golden must bind slots via shared InklingPrefix::bind_slot");
        assert!(!small_wi.contains("requests.cache.reset(") && !small_wi.contains("drafter.reset_context(")
                && !small_wi.contains("drafter.admit_pages("),
            "Inkling 2 Small width-invariance golden must not duplicate bind/admit outside InklingPrefix::bind_slot");

        // Audited sibling families: GLM Flash KDA and Kimi K3 goldens do not use `recurrent_cache`
        // (if either adopts `recurrent_cache` in the future, this assertion flags it for shared bind_slot wiring).
        for (name, src) in [
            ("glm5_flash/mod.rs", include_str!("../../families/glm5_flash/mod.rs")),
            ("glm5_flash/speculate.rs", include_str!("../../families/glm5_flash/speculate.rs")),
            ("kimi_k3/golden.rs", include_str!("../../families/kimi_k3/golden.rs")),
        ] {
            assert!(!src.contains("recurrent_cache"),
                "{name} adopted recurrent_cache; wire it to the shared bind_slot entry point and cover it here");
        }
    }
}

pub(crate) struct ContextBank { pub state:RefCell<ContextState> }
impl ContextBank {
    /// Context inputs are one contiguous causal sequence span. The caller
    /// tiles prefill and packed members through this shared operation.
    pub fn append_ranked(&self,rank:&Rank<'_>,split:Option<&super::gqa_tp2::GqaHeadSplit<'_,Rank<'_>>>,
        slot:usize,start:usize,features:Buffer,rows:usize)->Result<()> {
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
            if let Some(split)=split {
                use crate::shared::peer_split::DIRECT;
                let input=split.peer.buffer("context.taps")?;
                let bytes=rows*rank.config.taps.len()*rank.config.transformer.hidden*2;
                ensure!(features.bytes>=bytes && input.bytes>=bytes,"recurrent peer feature extent differs");
                split.exchange.push_to(split.lead,DIRECT,features.ptr,input.ptr,bytes)?;
                split.exchange.wait(1-split.lead,DIRECT)?;
                split.peer.stream.device.run(||split.peer.store_context(input,&positions,&destinations))?;
            }
            rank.store_context(features,&positions,&destinations)?;
            if rank.layout.rank.rank==rank.layout.rank.lead {
                let width=rank.config.transformer.hidden*2;
                let taps=rank.config.taps.len()*width;
                let last=Buffer{ptr:features.ptr.wrapping_byte_add((rows-1)*taps),bytes:taps,..features};
                let f=rank.buffer("cache.last_features")?;
                rank.recurrent_feature(last,Buffer{ptr:f.ptr.wrapping_byte_add(slot*width),bytes:width,..f})?;
            }
            rank.stream.drain()?;
            if let Some(split)=split{split.peer.stream.drain()?;}
            self.state.borrow_mut().pool.commit(reservation,rows)?;Ok(())
        })();
        if result.is_err(){
            // A failed producer may still read/write the provisional pages.
            // Never return them to the pool until native stream retirement.
            if let Some(split)=split {
                let drained=(||{split.exchange.publish_abort()?;split.exchange.drain_compute()})();
                split.exchange.finish_terminal(drained.is_ok());
                crate::shared::decode_graph::fatal_drain(drained,"recurrent failed peer context append");
            }
            rank.stream.drain_or_abort("recurrent failed context append");
            self.state.borrow_mut().pool.commit(reservation,0)?;
        }
        result
    }
}
