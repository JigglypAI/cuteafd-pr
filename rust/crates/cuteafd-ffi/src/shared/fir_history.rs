//! Exact accepted-prefix FIR-history commit, shared with hybrid verification.
//! Payloads are F32 bit patterns; no arithmetic or checkpoint reconstruction.
use crate::{programs::{Program,Programs,Scalar},CuteafdDeviceBuffer as Buffer};
use anyhow::{ensure,Context,Result};
use std::ffi::c_void;

#[derive(Debug,Clone,Copy,PartialEq,Eq)]
pub struct Layout {
    pub slots:usize,
    pub width:usize,
    pub lag_stride:usize,
    pub layer_stride:usize,
    pub slot_stride:usize,
    pub replay_row_stride:usize,
    pub replay_layer_stride:usize,
}
impl Layout {
    pub fn contiguous(slots:usize,width:usize)->Result<Self>{
        ensure!(slots>0 && width>0 && width<=11264,"FIR bank width/slots outside export");
        let layer_stride=width.checked_mul(3).context("FIR layer stride overflow")?;
        Ok(Self{slots,width,lag_stride:width,layer_stride,
            slot_stride:layer_stride.checked_mul(48).context("FIR slot stride overflow")?,
            replay_row_stride:width,replay_layer_stride:width.checked_mul(64).context("FIR replay stride overflow")?})
    }
    pub fn history_bytes(self)->Result<usize>{
        self.slots.checked_mul(self.slot_stride).and_then(|n|n.checked_mul(4)).context("FIR history extent overflow")
    }
    pub fn replay_bytes(self)->Result<usize>{
        self.replay_layer_stride.checked_mul(48).and_then(|n|n.checked_mul(4)).context("FIR replay extent overflow")
    }
}

pub struct Commit<'a>{program:Program<'a>,layout:Layout}
impl<'a> Commit<'a>{
    pub fn load(programs:&'a Programs<'a>,layout:Layout)->Result<Self>{
        let program=programs.program("fir_history_commit",&["checkpoint","replay","table","live"])?;
        let spec=program.spec();
        ensure!(spec.capacity_rows==64 && spec.scalar_kinds==b"iiilllll" && spec.scratch.is_empty(),
            "FIR commit manifest ABI/capacity differs");
        Ok(Self{program,layout})
    }
    /// Metadata validation precedes upload and every mutation. Segments and
    /// destination slots are unique and contain only actually verified inputs.
    pub fn table(&self,entries:&[(usize,usize,usize)],available:&[(usize,usize)])->Result<Vec<i32>>{
        ensure!(!entries.is_empty() && entries.len()<=64 && entries.len()==available.len(),"FIR commit sequence inventory differs");
        let mut slots=std::collections::BTreeSet::new();
        let mut table=vec![0;entries.len()*3];
        for (index,(&(slot,first,keep),&(start,count))) in entries.iter().zip(available).enumerate(){
            ensure!(slot<self.layout.slots && slots.insert(slot) && first==start && count>0 && count<=8
                && keep<=count && first.checked_add(count).is_some_and(|end|end<=64),"FIR commit outside staged segment or aliases slot");
            table[index]=i32::try_from(slot)?;
            table[entries.len()+index]=i32::try_from(first)?;
            table[2*entries.len()+index]=i32::try_from(keep)?;
        }
        Ok(table)
    }
    /// # Safety
    /// Table was checked by `table` and uploaded before this stream reaches the
    /// launch. Source planes and disjoint live backing survive commit/drain.
    pub unsafe fn launch(&self,checkpoint:Buffer,replay:Buffer,table:Buffer,live:Buffer,
        sequences:usize,stream:*mut c_void)->Result<()> {
        let l=self.layout;
        ensure!((1..=64).contains(&sequences),"FIR commit live sequence count");
        let spans=[(checkpoint,l.history_bytes()?),(replay,l.replay_bytes()?),(table,sequences*12),(live,l.history_bytes()?)];
        for (buffer,bytes) in spans {
            ensure!(!buffer.ptr.is_null() && buffer.device_id==live.device_id && buffer.bytes>=bytes
                && buffer.ptr as usize%4==0 && (buffer.ptr as usize).checked_add(bytes).is_some(),"FIR commit operand extent/device/alignment");
        }
        let out=live.ptr as usize;
        for (source,bytes) in spans[..3].iter().copied(){
            let from=source.ptr as usize;
            ensure!(out+l.history_bytes()?<=from || from+bytes<=out,"FIR commit source aliases live output");
        }
        let scalars=[Scalar::I32(i32::try_from(sequences)?),Scalar::I32(i32::try_from(l.slots)?),Scalar::I32(i32::try_from(l.width)?),
            Scalar::I64(i64::try_from(l.lag_stride)?),Scalar::I64(i64::try_from(l.layer_stride)?),
            Scalar::I64(i64::try_from(l.slot_stride)?),Scalar::I64(i64::try_from(l.replay_row_stride)?),Scalar::I64(i64::try_from(l.replay_layer_stride)?)];
        // SAFETY: validated exact physical planes; caller retains initialized
        // checked metadata and all storage through ordered stream completion.
        unsafe{self.program.launch(&[checkpoint.ptr,replay.ptr,table.ptr,live.ptr],&scalars,stream)}
    }
}
