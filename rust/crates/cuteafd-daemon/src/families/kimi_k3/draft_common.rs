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
                let mut ends=frontiers.borrow().clone();
                for row in rows {
                    let end=ends.get_mut(row.slot).context("block draft slot outside admission")?;
                    ensure!(row.position==end.1,"block draft context must append canonical rows");
                    end.1=end.1.checked_add(1).context("block frontier overflow")?;
                }
                ready()?;
                d.update(rows)?;
                *frontiers.borrow_mut()=ends;
                Ok(())
            }
        }
    }
    pub fn reset_context(&self,slot:usize,origin:usize)->Result<()> {
        match self { Self::Mla(d)=>d.reset_context(slot,origin), Self::Block { draft:d,frontiers }=> {
            d.reset_context(slot,0)?;
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
    pub fn mla_window_segments(&self,slot:usize)->Result<Vec<CuteafdDeviceBuffer>> {
        match self { Self::Mla(d)=>d.mla_window_segments(slot),Self::Block { draft:d,.. }=>d.window_planes(slot) }
    }
    pub fn mla_snapshot_buffers(&self)->Result<Vec<CuteafdDeviceBuffer>> {
        match self { Self::Mla(d)=>d.mla_snapshot_buffers(),Self::Block { draft:d,.. }=>
            (0..d.slots).map(|slot|d.window_planes(slot)).collect::<Result<Vec<_>>>().map(|v|v.into_iter().flatten().collect()) }
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
    pub fn mla_draft_page_bytes(&self)->usize { match self { Self::Mla(d)=>d.mla_draft_page_bytes(),Self::Block { .. }=>0 } }
    pub fn mla_reservation(&self,slot:usize)->Option<cuteafd_engine::reserve::DraftDemand> {
        match self { Self::Mla(d)=>d.mla_reservation(slot),Self::Block { .. }=>None }
    }
    pub fn mla_reservable(&self)->Option<usize> { match self { Self::Mla(d)=>d.mla_reservable(),Self::Block { .. }=>None } }
    pub fn mla_lifetime_rows(&self)->Option<usize> { match self { Self::Mla(d)=>d.mla_lifetime_rows(),Self::Block { .. }=>None } }
    pub fn mla_admission(&self,origin:usize)->Option<cuteafd_engine::reserve::DraftDemand> {
        match self { Self::Mla(d)=>d.mla_admission(origin),Self::Block { .. }=>None }
    }
    pub fn admit_mla_pages(&self,slot:usize,capacity:usize)->std::result::Result<(),BoxError> {
        match self { Self::Mla(d)=>d.admit_mla_pages(slot,capacity),Self::Block { .. }=>Ok(()) }
    }
    pub fn mla_pages_valid(&self,a:&Attachment)->bool { match self { Self::Mla(d)=>d.mla_pages_valid(a),Self::Block { .. }=>false } }
    pub fn mla_pages_revocable(&self,a:&Attachment)->bool { match self { Self::Mla(d)=>d.mla_pages_revocable(a),Self::Block { .. }=>false } }
    pub fn retain_mla_pages(&self,a:&Attachment)->Result<()> { match self { Self::Mla(d)=>d.retain_mla_pages(a),Self::Block { .. }=>anyhow::bail!("window draft has no page attachments") } }
    pub fn release_mla_pages(&self,a:&Attachment)->Result<()> { match self { Self::Mla(d)=>d.release_mla_pages(a),Self::Block { .. }=>anyhow::bail!("window draft has no page attachments") } }
    pub fn capture_mla_pages(&self,slot:usize,frontier:usize,write:usize)->std::result::Result<Attachment,BoxError> {
        match self { Self::Mla(d)=>d.capture_mla_pages(slot,frontier,write),Self::Block { .. }=>Err("window draft has no page attachments".into()) }
    }
    pub fn promote_mla_pages(&self,a:&Attachment)->std::result::Result<Attachment,BoxError> {
        match self { Self::Mla(d)=>d.promote_mla_pages(a),Self::Block { .. }=>Err("window draft has no page attachments".into()) }
    }
    pub fn mla_page_segments(&self,p:cuteafd_engine::prefix::DraftPageId)->Result<Vec<cuteafd_hostcache::copy::DeviceRange>> {
        match self { Self::Mla(d)=>d.mla_page_segments(p),Self::Block { .. }=>anyhow::bail!("window draft has no page attachments") }
    }
    pub fn restore_mla_pages(&self,slot:usize,a:&Attachment,end:usize)->std::result::Result<(),BoxError> {
        match self { Self::Mla(d)=>d.restore_mla_pages(slot,a,end),Self::Block { .. }=>Err("window draft has no page attachments".into()) }
    }
}
