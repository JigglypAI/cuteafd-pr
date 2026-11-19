//! Kimi binding dispatch over the shared executors; no scheduling policy lives here.
use crate::shared::speculation::{block_dspark::DsparkDrafter,dflash::{self,ContextRow,DflashDrafter,Draft,DraftSeq,TargetHead}};
use anyhow::{ensure,Context,Result};
use cuteafd_ffi::CuteafdDeviceBuffer;
use std::cell::RefCell;

type Attachment = cuteafd_engine::prefix::DraftPageAttachment;
type BoxError = cuteafd_engine::prefix::BoxError;

pub(super) enum CommonDraft<'a> {
    Mla(DflashDrafter<'a>),
    Block { draft: DsparkDrafter<'a>, frontiers: RefCell<Vec<(usize,usize)>> },
}

pub(super) struct Geometry<'g> {
    pub hidden: usize,
    pub vocab: usize,
    pub block: usize,
    pub mask_token: u32,
    pub taps: &'g [usize],
}

impl<'a> CommonDraft<'a> {
    pub fn geometry(&self) -> Geometry<'_> {
        match self {
            Self::Mla(d) => Geometry { hidden:d.cfg.hidden,vocab:d.cfg.vocab,block:d.cfg.block,
                mask_token:d.cfg.mask_token,taps:&d.cfg.taps },
            Self::Block { draft:d,.. } => Geometry { hidden:d.cfg.hidden,vocab:d.cfg.vocab,block:d.cfg.block,
                mask_token:d.cfg.mask_token,taps:&d.cfg.taps },
        }
    }
    pub fn is_block(&self) -> bool { matches!(self,Self::Block { .. }) }
    pub fn score_source(&self) -> cuteafd_core::HeadScore {
        match self {
            Self::Mla(d) => d.score_source(),
            Self::Block { .. } => cuteafd_core::HeadScore::Probability,
        }
    }
    pub fn configure_scores(&self, enabled: bool) {
        if let Self::Mla(d) = self { d.set_selector_readback(false, enabled); }
    }
    pub fn is_windowed_block(&self) -> bool { matches!(self,Self::Block { draft,.. } if draft.cfg.window.is_some()) }
    pub fn tap_index(&self, layer: usize) -> Option<usize> { self.geometry().taps.iter().position(|&l| l == layer) }
    pub fn taps(&self) -> CuteafdDeviceBuffer {
        match self { Self::Mla(d) => d.taps.buffer, Self::Block { draft:d,.. } => d.taps.buffer }
    }
    pub fn max_batch_sequences(&self) -> usize {
        match self { Self::Mla(d) => d.max_batch_sequences(), Self::Block { draft:d,.. } => d.max_sequences }
    }
    pub fn drafts(&self) -> usize {
        match self { Self::Mla(d) => d.cfg.drafts(), Self::Block { draft:d,.. } => d.cfg.drafts() }
    }
    pub fn allocation_receipts(&self, devices:cuteafd_loader::placement::planner::CoordinatorDevices<'_>)
        -> Result<Vec<cuteafd_loader::placement::planner::Receipt>> {
        match self {
            Self::Mla(d) => super::draft::common_draft_receipts(d,devices),
            Self::Block { draft:d,.. } => {
                let mut receipts=d.runtime_receipts(devices,"kimi.draft")?;
                for receipt in &mut receipts { receipt.allocation.lifetime=cuteafd_loader::placement::planner::Lifetime::RESIDENT; }
                Ok(receipts)
            }
        }
    }
    pub fn prime_remote(&self) -> Result<()> {
        match self { Self::Mla(d) => d.prime_remote(), Self::Block { draft:d,.. } => d.prepare_workspace() }
    }
    pub fn update_with(&self, rows:&[ContextRow], ready:impl FnOnce()->Result<()>) -> Result<()> {
        match self {
            Self::Mla(d) => d.update_with(rows,ready),
            Self::Block { draft:d,frontiers } => {
                let ends=append_frontiers(&frontiers.borrow(),rows)?;
                ready()?;
                d.update(rows)?;
                *frontiers.borrow_mut()=ends;
                Ok(())
            }
        }
    }
    pub fn reset_context(&self,slot:usize,origin:usize)->Result<()> {
        match self { Self::Mla(d)=>d.reset_context(slot,origin), Self::Block { draft:d,frontiers }=> {
            d.reset_context(slot,if d.cfg.window.is_some() { 0 } else { origin })?;
            *frontiers.borrow_mut().get_mut(slot).context("block draft reset slot outside admission")?=(origin,origin);
            Ok(())
        } }
    }
    pub fn discard_context(&self,slot:usize)->Result<()> {
        match self { Self::Mla(d)=>d.discard_context(slot), Self::Block { draft:d,frontiers }=> {
            d.discard_context(slot)?;
            *frontiers.borrow_mut().get_mut(slot).context("block draft discard slot outside admission")?=(0,0);
            Ok(())
        } }
    }
    pub fn mla_frontier(&self,slot:usize)->Result<(usize,usize)> {
        match self { Self::Mla(d)=>d.mla_frontier(slot), Self::Block { frontiers,.. }=>
            frontiers.borrow().get(slot).copied().context("block draft frontier slot outside admission") }
    }
    pub fn restore_frontier(&self,slot:usize,origin:usize,end:usize)->Result<()> {
        let Self::Block { frontiers,.. }=self else { anyhow::bail!("frontier metadata belongs to block draft") };
        ensure!(origin<=end,"block draft invalid restored frontier");
        *frontiers.borrow_mut().get_mut(slot).context("block draft restore slot outside admission")?=(origin,end);
        Ok(())
    }
    pub fn mla_window_segments(&self,slot:usize)->Result<Vec<cuteafd_hostcache::copy::DeviceRange>> {
        match self { Self::Mla(d)=>d.mla_window_segments(slot),Self::Block { draft:d,.. }=>
            d.window_planes(slot).map(|planes|planes.into_iter().map(crate::shared::prefix::cuda_copy::range).collect()) }
    }
    pub fn block_window_planes(&self,slot:usize)->Result<Vec<CuteafdDeviceBuffer>> {
        let Self::Block { draft,.. }=self else { anyhow::bail!("expected block window planes") };
        draft.window_planes(slot)
    }
    pub fn block_readers(&self)->Result<Vec<crate::shared::prefix::Reader>> {
        let Self::Block { draft,.. }=self else { anyhow::bail!("expected block readers") };
        Ok(draft.window_readers())
    }
    pub fn mla_snapshot_buffers(&self)->Result<Vec<CuteafdDeviceBuffer>> {
        match self { Self::Mla(d)=>d.mla_snapshot_buffers(),Self::Block { draft:d,.. } if d.cfg.window.is_none()=>Ok(Vec::new()),Self::Block { draft:d,.. }=>
            (0..d.slots).map(|slot|d.window_planes(slot)).collect::<Result<Vec<_>>>().map(|v|v.into_iter().flatten().collect()) }
    }
    pub fn bind_cooperative_producers(&mut self,producers:&[crate::shared::peer_split::RankDevice])->Result<()> {
        match self { Self::Mla(_)=>Ok(()),Self::Block { draft,.. }=>draft.bind_cooperative_producers(producers) }
    }
    pub fn abort(&self)->Result<()> {
        match self { Self::Mla(_)=>Ok(()),Self::Block { draft,.. }=>draft.abort() }
    }
    pub fn finish_terminal(&self,drained:bool) {
        if let Self::Block { draft,.. }=self { draft.finish_terminal(drained); }
    }
    pub fn require_live(&self)->Result<()> {
        match self { Self::Mla(_)=>Ok(()),Self::Block { draft,.. }=>draft.require_live() }
    }
    pub fn drain_mla(&self)->Result<()> {
        match self { Self::Mla(d)=>d.drain_mla(),Self::Block { draft:d,.. }=>d.drain() }
    }
    pub fn copy_mla_windows(&self,slot:usize,marks:&[CuteafdDeviceBuffer],capture:bool)->Result<()> {
        match self { Self::Mla(d)=>d.copy_mla_windows(slot,marks,capture),Self::Block { .. }=>
            anyhow::bail!("block windows require frontier metadata binding") }
    }
    pub fn draft_device_with(&self,seqs:&[DraftSeq],input:&dyn Fn(&[u32],CuteafdDeviceBuffer,*mut std::ffi::c_void)->Result<()>,
        head:TargetHead<'_>)->Result<Vec<Draft>> {
        match self { Self::Mla(d)=>d.draft_device_with(seqs,input,head),Self::Block { draft:d,.. }=>d.draft_device_with(seqs,input,head) }
    }
    #[cfg(test)]
    pub fn mla_readback(&self,slot:usize,rows:usize)->Result<dflash::MlaReadback> {
        match self { Self::Mla(d)=>d.mla_readback(slot,rows),Self::Block { .. }=>anyhow::bail!("MLA-specific diagnostic") }
    }
    #[cfg(test)]
    pub fn last_logits(&self,seqs:usize)->Result<Vec<f32>> {
        match self { Self::Mla(d)=>d.last_logits(seqs),Self::Block { draft:d,.. }=>d.last_logits(seqs) }
    }
    pub fn mla_windows_only(&self) -> bool { matches!(self, Self::Mla(d) if d.mla_windows_only()) }
    pub fn restore_mla_windows(&self, slot: usize, frontier: usize) -> Result<()> {
        match self { Self::Mla(d) => d.restore_mla_windows(slot, frontier),
            Self::Block { .. } => anyhow::bail!("block draft restores its own frontier metadata") }
    }
    pub fn mla_draft_page_bytes(&self)->usize { match self { Self::Mla(d)=>d.mla_draft_page_bytes(),Self::Block { draft,.. }=>draft.page_bytes() } }
    pub fn mla_reservation(&self,slot:usize)->Option<cuteafd_engine::reserve::DraftDemand> {
        match self { Self::Mla(d)=>d.mla_reservation(slot),Self::Block { draft,.. }=>draft.reservation(slot) }
    }
    pub fn mla_reservable(&self)->Option<usize> { match self { Self::Mla(d)=>d.mla_reservable(),Self::Block { draft,.. }=>draft.reservable() } }
    pub fn mla_lifetime_rows(&self)->Option<usize> { match self { Self::Mla(d)=>d.mla_lifetime_rows(),Self::Block { draft,.. }=>draft.lifetime_rows() } }
    pub fn mla_admission(&self,origin:usize)->Option<cuteafd_engine::reserve::DraftDemand> {
        match self { Self::Mla(d)=>d.mla_admission(origin),Self::Block { draft,.. }=>draft.admission(origin) }
    }
    pub fn quarantine_mla_pages(&self, slot: usize) {
        match self { Self::Mla(d) => d.quarantine_mla_pages(slot), Self::Block { draft, .. } => draft.quarantine_pages(slot) }
    }
    pub fn admit_mla_pages(&self,slot:usize,capacity:usize)->std::result::Result<(),BoxError> {
        match self { Self::Mla(d)=>d.admit_mla_pages(slot,capacity),Self::Block { draft,.. }=>draft.admit_pages(slot,capacity).map_err(Into::into) }
    }
    pub fn mla_pages_valid(&self,a:&Attachment)->bool { match self { Self::Mla(d)=>d.mla_pages_valid(a),Self::Block { draft,.. }=>draft.pages_valid(a) } }
    pub fn mla_pages_revocable(&self,a:&Attachment)->bool { match self { Self::Mla(d)=>d.mla_pages_revocable(a),Self::Block { draft,.. }=>draft.pages_revocable(a) } }
    pub fn retain_mla_pages(&self,a:&Attachment)->Result<()> { match self { Self::Mla(d)=>d.retain_mla_pages(a),Self::Block { draft,.. }=>draft.retain_pages(a) } }
    pub fn release_mla_pages(&self,a:&Attachment)->Result<()> { match self { Self::Mla(d)=>d.release_mla_pages(a),Self::Block { draft,.. }=>draft.release_pages(a) } }
    pub fn capture_mla_pages(&self,slot:usize,frontier:usize,write:usize)->std::result::Result<Attachment,BoxError> {
        match self { Self::Mla(d)=>d.capture_mla_pages(slot,frontier,write),Self::Block { draft,.. }=>draft.capture_pages(slot,frontier,write).map_err(Into::into) }
    }
    pub fn promote_mla_pages(&self,a:&Attachment)->std::result::Result<Attachment,BoxError> {
        match self { Self::Mla(d)=>d.promote_mla_pages(a),Self::Block { draft,.. }=>draft.promote_pages(a).map_err(Into::into) }
    }
    pub fn mla_page_segments(&self,p:cuteafd_engine::prefix::DraftPageId)->Result<Vec<cuteafd_hostcache::copy::DeviceRange>> {
        match self { Self::Mla(d)=>d.mla_page_segments(p),Self::Block { draft,.. }=>draft.page_segments(p) }
    }
    pub fn restore_mla_pages(&self,slot:usize,a:&Attachment,end:usize)->std::result::Result<(),BoxError> {
        match self { Self::Mla(d)=>d.restore_mla_pages(slot,a,end),Self::Block { draft,frontiers }=> {
            draft.restore_pages(slot,a,end)?;
            *frontiers.borrow_mut().get_mut(slot).ok_or("block restore slot outside admission")?=(a.origin,a.frontier);
            Ok(())
        } }
    }
}

fn append_frontiers(frontiers:&[(usize,usize)],rows:&[ContextRow])->Result<Vec<(usize,usize)>> {
    let mut ends=frontiers.to_vec();
    for row in rows {
        let end=ends.get_mut(row.slot).context("block draft slot outside admission")?;
        ensure!(row.position==end.1,"block draft context must append canonical rows");
        end.1=end.1.checked_add(1).context("block frontier overflow")?;
    }
    Ok(ends)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn window_frontiers_are_exact_across_wrap_concurrency_and_failure() -> Result<()> {
        let before=vec![(0,2047),(8192,8192)];
        let rows=[ContextRow { tap_row:0,slot:0,position:2047 },
            ContextRow { tap_row:1,slot:1,position:8192 },
            ContextRow { tap_row:2,slot:0,position:2048 }];
        assert_eq!(append_frontiers(&before,&rows)?,[(0,2049),(8192,8193)]);
        for bad in [ContextRow { tap_row:0,slot:0,position:2046 },
            ContextRow { tap_row:0,slot:0,position:2048 },
            ContextRow { tap_row:0,slot:2,position:0 }] {
            assert!(append_frontiers(&before,&[bad]).is_err());
            assert_eq!(before,[(0,2047),(8192,8192)]);
        }
        let duplicate=[rows[0],rows[0]];
        assert!(append_frontiers(&before,&duplicate).is_err());
        Ok(())
    }
}
