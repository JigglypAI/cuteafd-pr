//! Short-convolution producer with exact F32 input persistence for shared
//! accepted-prefix history commits; the existing non-replay programs are intact.
use crate::{programs::{Program,Programs,Scalar},CuteafdDeviceBuffer as Buffer};
use anyhow::{ensure,Context,Result};
use std::ffi::c_void;

pub struct Convolution<'a>{program:Program<'a>,width:usize,heads:usize,dynamic:bool}
impl<'a> Convolution<'a>{
    pub fn load(programs:&'a Programs<'a>,width:usize,dynamic:bool)->Result<Self>{
        ensure!(matches!((width,dynamic),(4608,false)|(1024,false)|(1024,true)),"Inkling replay convolution geometry");
        let heads=if dynamic{8}else{1};
        let name=format!("inkling2_conv_{}_h{width}_f32_replay",if dynamic{"dynamic"}else{"static"});
        let program=programs.program(&name,&["x","weight","dynamic","history","first","slots","out","replay"])?;
        let spec=program.spec();let g=&spec.geometry;
        ensure!(spec.capacity_rows==64 && spec.scalar_kinds==b"illl" && spec.scratch.is_empty()
            && g["width"]==width && g["taps"]==4 && g["heads"]==heads && g["dynamic"]==dynamic
            && g["history"]=="newest_first","FIR replay convolution manifest differs");
        Ok(Self{program,width,heads,dynamic})
    }
    /// # Safety
    /// Segment starts and slots were validated against the live request bank;
    /// history/replay pointers already select this layer/field. Mutable spans
    /// are disjoint and all inputs/owners survive the retained stream.
    pub unsafe fn launch(&self,buffers:[Buffer;8],rows:usize,slots:usize,
        history_slot_stride:usize,history_lag_stride:usize,replay_row_stride:usize,
        stream:*mut c_void)->Result<()> {
        ensure!((1..=64).contains(&rows) && slots>0 && history_lag_stride>=self.width
            && history_slot_stride>=3*history_lag_stride && replay_row_stride>=self.width,
            "FIR replay convolution capacity/pitches differ");
        let history=((slots-1).checked_mul(history_slot_stride)
            .and_then(|n|n.checked_add(2*history_lag_stride+self.width))
            .and_then(|n|n.checked_mul(4))).context("FIR history view overflow")?;
        let replay=(rows-1).checked_mul(replay_row_stride).and_then(|n|n.checked_add(self.width))
            .and_then(|n|n.checked_mul(4)).context("FIR replay view overflow")?;
        let extents=[rows*self.width*4,self.width*4*2,
            if self.dynamic{rows*self.heads*4*4}else{1},history,rows*4,rows*8,rows*self.width*4,replay];
        let device=buffers[0].device_id;
        for (&buffer,&bytes) in buffers.iter().zip(&extents){
            ensure!(buffer.device_id==device && !buffer.ptr.is_null() && buffer.bytes>=bytes
                && (buffer.ptr as usize).checked_add(bytes).is_some(),"FIR replay operand extent/device");
        }
        for output in [3usize,6,7]{
            for input in 0..buffers.len(){
                if input==output || (input==2 && !self.dynamic){continue;}
                let a=buffers[output].ptr as usize;let b=buffers[input].ptr as usize;
                ensure!(a+extents[output]<=b || b+extents[input]<=a,"FIR replay mutable operand aliases another span");
            }
        }
        let scalars=[Scalar::I32(i32::try_from(rows)?),Scalar::I64(i64::try_from(history_slot_stride)?),
            Scalar::I64(i64::try_from(history_lag_stride)?),Scalar::I64(i64::try_from(replay_row_stride)?)];
        // SAFETY: extents and disjoint writable spans checked; caller validates
        // request-local metadata and retains the owning stream/storage.
        unsafe{self.program.launch(&buffers.map(|b|b.ptr),&scalars,stream)}
    }
}
