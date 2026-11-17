//! Wide prefill companion for contiguous strided FIR banks, no replay sink.
use crate::{programs::{Program,Programs,Scalar},CuteafdDeviceBuffer as Buffer};
use anyhow::{ensure,Context,Result};
use std::ffi::c_void;
pub struct Convolution<'a>{program:Program<'a>,width:usize,heads:usize,dynamic:bool}
impl<'a> Convolution<'a>{
 pub fn load(programs:&'a Programs<'a>,width:usize,dynamic:bool)->Result<Self>{
  ensure!(matches!((width,dynamic),(4608,false)|(1024,false)|(1024,true)),"Inkling strided convolution geometry");
  let heads=if dynamic{8}else{1};let name=format!("inkling2_conv_{}_h{width}_f32_strided",if dynamic{"dynamic"}else{"static"});
  let program=programs.program(&name,&["x","weight","dynamic","history","first","slots","out"])?;
  let spec=program.spec();let g=&spec.geometry;
  ensure!(spec.capacity_rows==4096 && spec.scalar_kinds==b"ill" && spec.scratch.is_empty()
   && g["width"]==width && g["taps"]==4 && g["heads"]==heads && g["dynamic"]==dynamic
   && g["history"]=="newest_first","strided FIR manifest differs");
  Ok(Self{program,width,heads,dynamic})
 }
 /// # Safety
 /// Host checked segment starts and slots before upload. All operands survive
 /// stream completion; history pointer is pre-offset to the layer/field.
 pub unsafe fn launch(&self,b:[Buffer;7],rows:usize,slots:usize,slot_stride:usize,lag_stride:usize,stream:*mut c_void)->Result<()> {
  ensure!((1..=4096).contains(&rows) && slots>0 && lag_stride>=self.width && slot_stride>=3*lag_stride,"strided FIR capacity/pitches");
  let history=(slots-1).checked_mul(slot_stride).and_then(|n|n.checked_add(2*lag_stride+self.width))
   .and_then(|n|n.checked_mul(4)).context("strided FIR history extent overflow")?;
  let bytes=[rows*self.width*4,self.width*8,if self.dynamic{rows*self.heads*16}else{1},history,rows*4,rows*8,rows*self.width*4];
  for (buffer,bytes) in b.iter().zip(bytes){ensure!(!buffer.ptr.is_null() && buffer.device_id==b[0].device_id
   && buffer.bytes>=bytes && (buffer.ptr as usize).checked_add(bytes).is_some(),"strided FIR operand extent");}
  for output in [3usize,6]{for input in 0..b.len(){if output==input || (input==2 && !self.dynamic){continue;}
   let a=b[output].ptr as usize;let c=b[input].ptr as usize;ensure!(a+bytes[output]<=c || c+bytes[input]<=a,"strided FIR output aliases operand");}}
  let scalars=[Scalar::I32(i32::try_from(rows)?),Scalar::I64(i64::try_from(slot_stride)?),Scalar::I64(i64::try_from(lag_stride)?)];
  // SAFETY: typed extents/disjoint mutable regions checked, caller retains
  // validated metadata and all buffers on this ordered stream.
  unsafe{self.program.launch(&b.map(|buffer|buffer.ptr),&scalars,stream)}
 }
}
