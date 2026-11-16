//! Bundled recurrent drafter facts; the shared solver decides pool and placement.
use crate::{formats::block_dspark::recurrent::{loading::{RecurrentSource,RecurrentWeightLayout},
    runtime::RecurrentRuntimeLayout,RecurrentBlockConfig},families::qwen4::Qwen4KvCache,
    plan::checkpoint::CheckpointView};
use anyhow::{ensure,Context,Result};

#[derive(Debug,Clone)]
pub struct DraftLayout {
    pub config:RecurrentBlockConfig,
    pub weights:Vec<RecurrentWeightLayout>,
    pub runtime:Vec<RecurrentRuntimeLayout>,
    pub marks:usize,
    /// Target-side tap and exact FIR verification storage, independent of pool.
    pub target:Vec<(String,usize)>,
}
impl DraftLayout {
    pub fn new(view:&CheckpointView,ranks:usize,slots:usize,sequences:usize,context:usize,pool:usize,
        prefill:usize,waves:usize,sms:usize,format:Qwen4KvCache,marks:usize)->Result<Self>{
        ensure!(matches!(ranks,1|2) && slots>0 && sequences>0 && prefill>0 && waves>0,"recurrent draft fact geometry");
        let source=RecurrentSource::open(view)?;
        let lead=ranks-1;
        let weights=(0..ranks).map(|rank|source.rank(rank,ranks,lead)).collect::<Result<Vec<_>>>()?;
        let mut runtime=(0..ranks).map(|rank|RecurrentRuntimeLayout::new(&source.config,slots,sequences,2048,context,pool,
            sms,format,rank,ranks,lead)).collect::<Result<Vec<_>>>()?;
        for (weight,work) in weights.iter().zip(&mut runtime){
            let rows=work.context_rows.max(work.sequences*source.config.transformer.block);
            let mut projection:Option<crate::formats::trellis_core::ProjectionScratchLayout>=None;
            for core in weight.core.values(){
                let need=crate::formats::trellis_core::ProjectionScratchLayout::new(core.shape[1],core.shape[0],rows)?;
                if let Some(projection)=&mut projection{projection.merge(&need)?;}else{projection=Some(need);}
            }
            if let Some(projection)=projection{
                work.buffers.push(("projection.trellis".into(),projection.packed_bytes()?));
            }
        }
        let h=source.config.transformer.hidden;let taps=source.config.taps.len();
        let extent=|values:&[usize]|values.iter().try_fold(1usize,|n,value|n.checked_mul(*value)).context("draft target owner overflow");
        let mut target=Vec::new();
        for wave in 0..waves{
            target.push((format!("unit.{wave}.taps"),extent(&[prefill,taps,h,2])?));
            target.push((format!("unit.{wave}.narrow"),extent(&[prefill,h,2])?));
        }
        target.push(("decode.taps".into(),extent(&[64,taps,h,2])?));
        target.push(("decode.narrow".into(),extent(&[64,h,2])?));
        let conv=super::conv::ConvLayout::new(slots,48,h,1024)?;
        target.push(("verify.checkpoint".into(),conv.bytes()?));
        target.push(("verify.replay".into(),extent(&[48,64,conv.width,4])?));
        target.push(("verify.table".into(),64*12));
        Ok(Self{config:source.config,weights,runtime,marks,target})
    }
}
