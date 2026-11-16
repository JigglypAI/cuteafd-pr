//! Inkling binding of shared recurrent speculation, exact local marks and FIR
//! verifier storage. No scheduling, sampling policy or private exchange here.
use super::{draft_taps::TapBuffer,verify::Verify};
use crate::shared::{memory::device::Stream,speculation::{recurrent::Drafter,recurrent_forward::{Rank,ProgramsForRank},
    recurrent_weights::Weights,recurrent_cache::{ContextBank,ContextState},recurrent_snapshots::Marks}};
use anyhow::{ensure,Context,Result};
use cuteafd_ffi::{programs::Programs,CuteafdDeviceBuffer as Buffer};
use cuteafd_loader::{families::inkling2::draft::DraftLayout,placement::planner::ResolvedPlan};
use std::{cell::RefCell,rc::Rc};

pub(crate) struct Binding<'a>{
    pub drafter:Rc<Drafter<'a>>,
    pub marks:Option<Marks<'a>>,
    pub verify:Verify<'a>,
    pub prefill:Vec<TapBuffer<'a>>,
    pub decode:TapBuffer<'a>,
    pub layout:DraftLayout,
}
impl<'a> Binding<'a>{
    pub fn new(programs:&'a Programs<'a>,stream:Rc<Stream<'a>>,weights:Weights<'a>,layout:DraftLayout,
        admitted:&ResolvedPlan,target_head:Buffer,prefill_rows:usize,waves:usize)->Result<Self>{
        ensure!(layout.weights.len()==1 && layout.runtime.len()==1,"Inkling target binding currently runs whole rank");
        let runtime=&layout.runtime[0];
        let check=|id:&str,bytes:usize|->Result<()>{
            let owner=admitted.facts.allocations.iter().find(|owner|owner.id==id).with_context(||format!("unadmitted drafter owner {id}"))?;
            ensure!(owner.bytes==bytes as u64,"drafter runtime owner differs from admission {id}");Ok(())
        };
        check(layout.weights[0].arena.id(),layout.weights[0].arena.bytes())?;
        for (name,bytes) in &runtime.buffers{check(&format!("inkling2.draft.0.{name}"),*bytes)?;}
        for (name,bytes) in &layout.target{check(&format!("inkling2.draft.target.{name}"),*bytes)?;}
        if layout.marks>0{check("inkling2.draft.0.marks",runtime.window_mark_bytes*layout.marks)?;}
        let kernels=ProgramsForRank::load(programs,&weights,&layout.config,runtime,stream.device)?;
        let rank=Rank::allocate(weights,Rc::clone(&stream),layout.config.clone(),runtime.clone(),kernels)?;
        rank.initialize_derived()?;rank.fold_head(target_head)?;
        let context=ContextBank{state:RefCell::new(ContextState::new(runtime.slots,
            usize::try_from(admitted.served_context())?,runtime.cache.target_tokens,0x494e4b4d5450)?)};
        let drafter=Rc::new(Drafter::whole(rank,context)?);
        let marks=(layout.marks>0).then(||Marks::new(&drafter.lead,layout.marks)).transpose()?;
        let verify=Verify::new(programs,Rc::clone(&stream),runtime.slots,11264)?;
        let prefill=(0..waves).map(|_|TapBuffer::new(Rc::clone(&stream),&layout.config,prefill_rows)).collect::<Result<Vec<_>>>()?;
        let decode=TapBuffer::new(Rc::clone(&stream),&layout.config,64)?;
        stream.drain()?;
        Ok(Self{drafter,marks,verify,prefill,decode,layout})
    }
    pub fn taps(&self,decode:bool,unit:usize)->Result<&TapBuffer<'a>>{
        if decode{Ok(&self.decode)}else{self.prefill.get(unit).context("drafter prefill tap unit outside admission")}
    }
    pub fn update(&self,taps:&TapBuffer<'a>,slot:usize,start:usize,first:usize,rows:usize)->Result<()> {
        let pitch=self.layout.config.taps.len()*self.layout.config.transformer.hidden*2;
        for offset in (0..rows).step_by(self.layout.runtime[0].context_rows){
            let count=(rows-offset).min(self.layout.runtime[0].context_rows);
            let position=(first+offset)*pitch;
            let output=taps.output.buffer;
            ensure!(position.checked_add(count*pitch).is_some_and(|end|end<=output.bytes),"drafter target taps outside owner");
            self.drafter.context.append(&self.drafter.lead,slot,start+offset,
                Buffer{ptr:output.ptr.wrapping_byte_add(position),bytes:count*pitch,..output},count)?;
        }Ok(())
    }
}
