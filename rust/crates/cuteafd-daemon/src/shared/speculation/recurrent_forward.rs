//! Shared recurrent block body. Target bindings own taps/verify/commit; the
//! numerical producers and row-parallel rank reductions live here.
use super::recurrent_weights::{ProjectionWeight, Weights};
use crate::shared::{dense_trellis::{Projection, ProjectionLayout, ProjectionScratch},
    memory::{device::{Allocation, Device, Stream}, Tag}};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{draft_projection::{DraftProjections, ProjectionOutput},
    dense_trellis::TrellisDType, programs::Programs, recurrent_draft::{Op, Primitive}, CuteafdDeviceBuffer as Buffer};
use cuteafd_loader::formats::block_dspark::recurrent::{RecurrentBlockConfig,
    runtime::RecurrentRuntimeLayout};
use std::{collections::BTreeMap, ffi::c_void, rc::Rc};

type Shape = (usize,usize);

pub(crate) struct ProgramsForRank<'a> {
    pub(super) plain: DraftProjections<'a>,
    pub(super) f32: DraftProjections<'a>,
    compressed: BTreeMap<String,Projection<'a>>,
    compressed_scratch: Option<ProjectionScratch<'a>>,
    pub mix: Primitive<'a>, pub init: Primitive<'a>, pub residual_norm: Primitive<'a>,
    pub rotary: Primitive<'a>, pub context_rotary: Primitive<'a>, pub silu: Primitive<'a>,
    pub store: Primitive<'a>,
    pub head:Option<HeadPrograms<'a>>,
}

pub(crate) struct HeadPrograms<'a> {
    pub fold:Primitive<'a>,pub gate:Primitive<'a>,pub finish:Primitive<'a>,
    pub bias_raw:Primitive<'a>,pub bias_correction:Primitive<'a>,
}
impl<'a> ProgramsForRank<'a> {
    pub fn load(programs:&'a Programs<'a>,weights:&Weights<'a>,config:&RecurrentBlockConfig,
        layout:&RecurrentRuntimeLayout,device:Device<'a>)->Result<Self> {
        let c=&config.transformer;
        let local=layout.rank;
        let q=local.heads*c.head_dim; let kv=local.kv_heads*c.head_dim;
        let mut shapes=vec![(c.hidden,q),(c.hidden,kv),(q,c.hidden),(c.hidden,local.intermediate),
            (local.intermediate,c.hidden)];
        if local.rank==local.lead { shapes.extend([(c.hidden,config.padded_vocab),(c.markov_rank,config.padded_vocab)]); }
        let rows=layout.context_rows.max(layout.sequences*c.block);
        let plain=DraftProjections::load(programs,&shapes,rows)?;
        let f32_shapes=if local.rank==local.lead { vec![(config.recurrent_input_width()?,3*c.markov_rank),
            (c.markov_rank,config.padded_vocab)] } else { vec![] };
        let f32=DraftProjections::load_with_output(programs,&f32_shapes,layout.sequences,ProjectionOutput::F32)?;
        let mut compressed=BTreeMap::new();
        let mut required:Option<ProjectionLayout>=None;
        for (name,core) in &weights.layout.core {
            let side=match core.side {cuteafd_loader::formats::dense_trellis::TrellisSide::Output=>1,
                cuteafd_loader::formats::dense_trellis::TrellisSide::Input=>2};
            let book=match core.codebook {cuteafd_loader::formats::exl3_storage::Exl3Codebook::Mcg=>"mcg",
                cuteafd_loader::formats::exl3_storage::Exl3Codebook::Mul1=>"mul1"};
            let projection=Projection::load_geometry_bits(programs,core.shape[1],core.shape[0],side,
                core.bits,book,rows,TrellisDType::Bf16)?;
            let need=ProjectionLayout::new(&projection.program,rows)?;
            if let Some(required)=&mut required {required.merge(&need)?;} else {required=Some(need);}
            compressed.insert(name.clone(),projection);
        }
        let compressed_scratch=required.map(|layout|layout.allocate_packed(device,Tag::drafter_workspace("shared/recurrent-projection"))).transpose()?;
        let head=if local.rank==local.lead {Some(HeadPrograms{
            fold:Primitive::load(programs,Op::Fold)?,gate:Primitive::load(programs,Op::Gate)?,
            finish:Primitive::load(programs,Op::Finish)?,
            bias_raw:Primitive::load(programs,Op::Bias{width:768,bf16:false})?,
            bias_correction:Primitive::load(programs,Op::Bias{width:201024,bf16:true})?,
        })}else{None};
        Ok(Self {plain,f32,compressed,compressed_scratch,head,
            mix:Primitive::load(programs,Op::Mix)?,init:Primitive::load(programs,Op::Mask)?,
            residual_norm:Primitive::load(programs,Op::ResidualNorm)?,
            rotary:Primitive::load(programs,Op::QkRope{query_heads:local.heads,kv_heads:local.kv_heads})?,
            context_rotary:Primitive::load(programs,Op::QkRope{query_heads:0,kv_heads:local.kv_heads})?,
            silu:Primitive::load(programs,Op::Silu{intermediate:local.intermediate})?,
            store:Primitive::load(programs,Op::Store{kv_heads:local.kv_heads,
                fp8:layout.kv_cache==cuteafd_loader::families::qwen4::Qwen4KvCache::Fp8})?})
    }

    pub fn project(&self,name:&str,weights:&ProjectionWeight<'_>,input:Buffer,output:Buffer,
        rows:usize,shape:Shape,stream:*mut c_void)->Result<()> {
        let (k,n)=shape;
        match weights {
            ProjectionWeight::Bf16(weight)=> {
                ensure!(input.bytes>=rows*k*2 && weight.buffer.bytes==n*k*2,"recurrent BF16 projection operands differ");
                // SAFETY: input and weight extents checked; caller retains disjoint
                // admitted destination and stream, and producer checks its extent.
                unsafe {self.plain.launch_checked(input.ptr,weight.buffer.ptr,output,rows,k,n,stream)?;}
            }
            ProjectionWeight::Trellis(weight)=> {
                let projection=self.compressed.iter().find(|(key,_)|key.as_str()==name || key.strip_prefix("mtp.")==Some(name))
                    .map(|(_,value)|value).context("recurrent compact projection not bound")?;
                let scratch=self.compressed_scratch.as_ref().context("recurrent compact scratch not admitted")?;
                projection.local(rows,input,weight,scratch,output,stream)?;
            }
        }
        Ok(())
    }

    /// Concatenated parts preserve independent compressed rotations/calibrations.
    pub(super) fn parts(&self,names:&[String],weights:&[ProjectionWeight<'_>],input:Buffer,output:Buffer,
        rows:usize,k:usize,widths:&[usize],library:&cuteafd_ffi::NativeLibrary,
        scratch:Buffer,stream:*mut c_void)->Result<()> {
        ensure!(weights.len()==widths.len() && names.len()==widths.len(),"recurrent grouped operand inventory differs");
        let pitch=widths.iter().sum::<usize>()*2;
        let mut offset=0;
        for ((name,weight),&width) in names.iter().zip(weights).zip(widths) {
            self.project(name,weight,input,scratch,rows,(k,width),stream)?;
            let destination=Buffer{ptr:output.ptr.wrapping_byte_add(offset),bytes:output.bytes-offset,..output};
            // SAFETY: projection scratch and packed destination are disjoint,
            // exact row pitches and widths cover their admitted spans.
            unsafe{library.copy_d2d_2d_async(destination,pitch,scratch,width*2,width*2,rows,stream)?;}
            offset+=width*2;
        }
        Ok(())
    }
}

pub(crate) struct Rank<'a> {
    pub weights:Weights<'a>,
    pub stream:Rc<Stream<'a>>,
    pub config:RecurrentBlockConfig,
    pub layout:RecurrentRuntimeLayout,
    pub programs:ProgramsForRank<'a>,
    pub buffers:BTreeMap<String,Allocation<'a>>,
}
impl<'a> Rank<'a> {
    pub fn buffer(&self,name:&str)->Result<Buffer> {
        self.buffers.get(name).map(|owner|owner.buffer).with_context(||format!("recurrent draft lacks {name}"))
    }
    pub fn allocate(weights:Weights<'a>,stream:Rc<Stream<'a>>,config:RecurrentBlockConfig,
        layout:RecurrentRuntimeLayout,programs:ProgramsForRank<'a>)->Result<Self> {
        let device=stream.device;
        let buffers=layout.buffers.iter().map(|(name,bytes)| {
            let tag=if name.starts_with("cache."){Tag::drafter_kv("shared/recurrent-cache")}
                else{Tag::drafter_workspace("shared/recurrent-workspace")};
            Ok((name.clone(),Allocation::new(device,*bytes,tag)?))
        }).collect::<Result<BTreeMap<_,_>>>()?;
        Ok(Self{weights,stream,config,layout,programs,buffers})
    }
    pub fn norm_first(&self,rows:usize)->Result<()> {
        // SAFETY: rank-local fixed-hidden BF16 planes and gamma are retained
        // through every launch; norm uses the existing single-round native path.
        unsafe{self.stream.device.library.cuda_ds4_rmsnorm_bf16_rne_async(self.buffer("step.residual")?,
            self.weights.tensors.layers[0].input_norm.buffer,self.buffer("step.normalized")?,i32::try_from(rows)?,
            i32::try_from(self.config.transformer.hidden)?,self.config.transformer.rms_norm_eps as f32,self.stream.raw)}
    }
    pub fn attention_partial(&self,index:usize,sequences:usize,max_keys:usize)->Result<()> {
        let c=&self.config.transformer; let rank=self.layout.rank;
        let layer=&self.weights.tensors.layers[index];
        let rows=sequences*c.block; let q=rank.heads*c.head_dim; let kv=rank.kv_heads*c.head_dim;
        let a=format!("layers.{index}.self_attn");
        let names=["q_proj","k_proj","v_proj"].map(|p|format!("{a}.{p}.weight"));
        self.programs.parts(&names,&layer.qkv,self.buffer("step.normalized")?,self.buffer("step.qkv")?,
            rows,c.hidden,&[q,kv,kv],self.stream.device.library,self.buffer("projection.bf16")?,self.stream.raw)?;
        let eps=c.rms_norm_eps as f32;
        // SAFETY: typed primitive checks dense spans and disjoint outputs; all
        // rank-local planes survive the retained stream and external TP meet.
        unsafe{self.programs.rotary.launch(&[self.buffer("step.qkv")?,layer.q_norm.buffer,layer.k_norm.buffer,
            self.buffer("step.positions")?,self.buffer("rotary.frequencies")?,self.buffer("step.query")?,
            self.buffer("step.key")?,self.buffer("step.value")?],rows,eps,0,self.stream.raw)?;}
        self.attend(index,sequences,max_keys)?;
        self.programs.project(&format!("{a}.o_proj.weight"),&layer.output[0],self.buffer("step.attention")?,
            self.buffer("step.partial")?,rows,(q,c.hidden),self.stream.raw)
    }
    fn attend(&self,index:usize,sequences:usize,max_keys:usize)->Result<()> {
        let c=&self.config.transformer;let rank=self.layout.rank;let lib=self.stream.device.library;
        let tables=self.buffer("step.tables")?;
        let lengths=Buffer{ptr:tables.ptr.wrapping_byte_add(sequences*4),bytes:sequences*4,..tables};
        let ends=Buffer{ptr:tables.ptr.wrapping_byte_add(sequences*8),bytes:sequences*4,..tables};
        let key=self.buffer(&format!("cache.layer.{index}.k"))?;
        let value=self.buffer(&format!("cache.layer.{index}.v"))?;
        let fp8=self.layout.kv_cache==cuteafd_loader::families::qwen4::Qwen4KvCache::Fp8;
        ensure!(matches!(self.config.context[index],cuteafd_loader::formats::block_dspark::recurrent::ContextPolicy::Full)
            || max_keys<=self.config.transformer.window+self.config.transformer.block,
            "local recurrent attention requires clamped committed lengths");
        // SAFETY: caller sets slot/length/end metadata for exact committed
        // context and retains every visible page/ring record through attention.
        unsafe{match self.config.context[index] {
            cuteafd_loader::formats::block_dspark::recurrent::ContextPolicy::BlockAnchoredWindow(window)=>
                lib.block_dspark_attention(self.buffer("step.query")?.ptr,self.buffer("step.key")?.ptr,
                    self.buffer("step.value")?.ptr,key.ptr,value.ptr,tables.ptr,lengths.ptr,ends.ptr,
                    self.buffer("step.attention")?.ptr,self.buffer("step.attention_workspace")?.ptr,
                    sequences,c.block,rank.heads,rank.kv_heads,window,max_keys.min(window+c.block),false,
                    c.head_dim,1./(c.head_dim as f32).sqrt(),fp8,self.stream.raw),
            cuteafd_loader::formats::block_dspark::recurrent::ContextPolicy::Full=>
                lib.block_dspark_attention_paged(self.buffer("step.query")?.ptr,self.buffer("step.key")?.ptr,
                    self.buffer("step.value")?.ptr,key.ptr,value.ptr,tables.ptr,lengths.ptr,ends.ptr,
                    self.buffer("cache.tables")?.ptr,self.buffer("step.attention")?.ptr,
                    self.buffer("step.attention_workspace")?.ptr,sequences,c.block,rank.heads,rank.kv_heads,
                    self.layout.cache.pages*64,max_keys,false,c.head_dim,
                    self.layout.cache.page_table_bytes as usize/self.layout.slots/4,
                    1./(c.head_dim as f32).sqrt(),fp8,self.stream.raw),
        }}
    }
    pub fn ffn_partial(&self,index:usize,sequences:usize,reduced:Buffer)->Result<()> {
        let c=&self.config.transformer;let rows=sequences*c.block;let layer=&self.weights.tensors.layers[index];
        let eps=c.rms_norm_eps as f32;
        // SAFETY: residual is read from previous stage and new disjoint residual
        // plus normalized output are retained through all subsequent projections.
        unsafe{self.programs.residual_norm.launch(&[self.buffer("step.residual")?,reduced,layer.post_norm.buffer,
            self.buffer("step.residual_next")?,self.buffer("step.normalized")?],rows,eps,0,self.stream.raw)?;
            self.stream.device.library.copy_d2d_async(self.buffer("step.residual")?,self.buffer("step.residual_next")?,
                rows*c.hidden*2,self.stream.raw)?;}
        let names=["gate_proj","up_proj"].map(|p|format!("layers.{index}.mlp.{p}.weight"));
        let i=self.layout.rank.intermediate;
        self.programs.parts(&names,&layer.gate_up,self.buffer("step.normalized")?,self.buffer("step.gate_up")?,
            rows,c.hidden,&[i,i],self.stream.device.library,self.buffer("projection.bf16")?,self.stream.raw)?;
        // SAFETY: packed gate/up and activation planes are admitted and disjoint.
        unsafe{self.programs.silu.launch(&[self.buffer("step.gate_up")?,self.buffer("step.activated")?],rows,eps,0,self.stream.raw)?;}
        self.programs.project(&format!("layers.{index}.mlp.down_proj.weight"),&layer.down[0],self.buffer("step.activated")?,
            self.buffer("step.partial")?,rows,(i,c.hidden),self.stream.raw)
    }
    pub fn finish_layer(&self,index:usize,sequences:usize,reduced:Buffer)->Result<()> {
        let rows=sequences*self.config.transformer.block;
        let gamma=self.weights.tensors.layers.get(index+1).map_or(self.weights.tensors.norm.buffer,|layer|layer.input_norm.buffer);
        // SAFETY: typed primitive and retained rank planes satisfy same disjoint
        // residual contract as ffn_partial, using next layer or final norm gamma.
        unsafe{self.programs.residual_norm.launch(&[self.buffer("step.residual")?,reduced,gamma,
            self.buffer("step.residual_next")?,self.buffer("step.normalized")?],rows,
            self.config.transformer.rms_norm_eps as f32,0,self.stream.raw)?;
            self.stream.device.library.copy_d2d_async(self.buffer("step.residual")?,self.buffer("step.residual_next")?,
                rows*self.config.transformer.hidden*2,self.stream.raw)}
    }
}
