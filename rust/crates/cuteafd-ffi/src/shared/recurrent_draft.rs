//! Checkpoint-native non-GEMM block-drafter programs. These operators preserve
//! Inkling's single-round affine norms and FP32 RoPE, unlike vanilla DSpark.
use crate::{programs::{Program, ProgramSpec, Programs, Scalar}, CuteafdDeviceBuffer as Buffer};
use anyhow::{ensure, Context, Result};
use std::ffi::c_void;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Mix, Mask, Gate, Fold, Finish, ResidualNorm,
    QkRope { query_heads: usize, kv_heads: usize },
    Silu { intermediate: usize },
    Bias { width: usize, bf16: bool },
    Store { kv_heads: usize, fp8: bool },
}
impl Op {
    pub fn name(self) -> Result<String> {
        Ok(format!("inkling_draft_{}", match self {
            Self::Mix => "dflare_mix".into(), Self::Mask => "mask_init".into(),
            Self::Gate => "recurrent_gate".into(), Self::Fold => "head_fold".into(),
            Self::Finish => "logit_finish".into(), Self::ResidualNorm => "residual_norm".into(),
            Self::QkRope { query_heads, kv_heads } => {
                ensure!([(48,8),(24,4),(0,8),(0,4)].contains(&(query_heads,kv_heads)), "invalid recurrent rotary heads");
                format!("qk_norm_rope_q{query_heads}_k{kv_heads}")
            }
            Self::Silu { intermediate } => {
                ensure!(matches!(intermediate,18432|9216), "invalid recurrent FFN width");
                format!("silu_mul_i{intermediate}")
            }
            Self::Bias { width, bf16 } => {
                ensure!(matches!((width,bf16),(768,false)|(201024,true)), "invalid recurrent biased projection");
                format!("bias_add_w{width}_{}",if bf16 {"bf16"} else {"f32"})
            }
            Self::Store { kv_heads, fp8 } => {
                ensure!(matches!(kv_heads,8|4), "invalid recurrent KV heads");
                format!("kv_store_k{kv_heads}_{}",if fp8 {"fp8"} else {"bf16"})
            }
        }))
    }
    fn pointers(self) -> &'static [&'static str] {
        match self {
            Self::Mix => &["features","alpha","gamma","mixed"],
            Self::Mask => &["normalized_anchors","mask","block"],
            Self::Gate => &["raw","state","new_state","latent"],
            Self::Fold => &["lm_head","hidden_out","hidden_bias","folded_weight","folded_bias"],
            Self::Finish => &["base","correction","output"],
            Self::ResidualNorm => &["residual","delta","gamma","residual_out","norm_out"],
            Self::QkRope { .. } => &["qkv","q_gamma","k_gamma","positions","inv_freq","q_out","k_out","v_out"],
            Self::Silu { .. } => &["gate_up","activated"],
            Self::Bias { .. } => &["x","bias","out"],
            Self::Store { .. } => &["k","v","slots","k_plane","v_plane"],
        }
    }
    fn scalar_kinds(self) -> &'static [u8] {
        match self { Self::Mix|Self::ResidualNorm|Self::QkRope {..}=>b"if", Self::Fold=>b"ii", _=>b"i" }
    }
    fn outputs(self) -> usize {
        match self { Self::Mix|Self::Mask|Self::Finish|Self::Silu{..}|Self::Bias{..}=>1,
            Self::Gate|Self::Fold|Self::ResidualNorm|Self::Store{..}=>2, Self::QkRope{..}=>3 }
    }
    fn shapes(self, rows: usize) -> Result<Vec<usize>> {
        let m = |dims: &[usize]| dims.iter().try_fold(1usize,|n,d|n.checked_mul(*d)).context("recurrent operand bytes overflow");
        Ok(match self {
            Self::Mix=>vec![m(&[rows,8,4608,2])?,16,9216,m(&[rows,4608,2])?],
            Self::Mask=>vec![m(&[rows,4608,2])?,9216,m(&[rows,8,4608,2])?],
            Self::Gate=>vec![m(&[rows,768,4])?,m(&[rows,256,2])?,m(&[rows,256,2])?,m(&[rows,256,2])?],
            Self::Fold=>vec![201024*4608*2,4608*256*2,9216,201024*256*2,201024*2],
            Self::Finish=>vec![m(&[rows,201024,2])?,m(&[rows,201024,2])?,m(&[rows,200058,2])?],
            Self::ResidualNorm=>vec![m(&[rows,4608,2])?,m(&[rows,4608,2])?,9216,m(&[rows,4608,2])?,m(&[rows,4608,2])?],
            Self::QkRope {query_heads:q,kv_heads:k}=>vec![m(&[rows,q+2*k,128,2])?,256,256,m(&[rows,8])?,256,
                m(&[rows,q,128,2])?,m(&[rows,k,128,2])?,m(&[rows,k,128,2])?],
            Self::Silu{intermediate:i}=>vec![m(&[rows,2,i,2])?,m(&[rows,i,2])?],
            Self::Bias{width:w,bf16}=>vec![m(&[rows,w,4])?,m(&[w,2])?,m(&[rows,w,if bf16 {2}else{4}])?],
            // Cache planes are checked by the admitted ring/page owner; the
            // scatter's data-dependent slots cannot prove their full extent here.
            Self::Store{kv_heads:k,..}=>vec![m(&[rows,k,128,2])?,m(&[rows,k,128,2])?,m(&[rows,8])?,1,1],
        })
    }
}

pub struct Primitive<'a> { program: Program<'a>, op: Op }
impl<'a> Primitive<'a> {
    pub fn load(programs: &'a Programs<'a>, op: Op) -> Result<Self> {
        let name=op.name()?;
        let spec=programs.spec(&name)?;
        validate_spec(op,spec)?;
        Ok(Self{program:programs.program(&name,op.pointers())?,op})
    }
    /// # Safety
    /// Inputs are initialized and ordered on `stream`; all spans survive its
    /// completion. Store slots address admitted retained ring/page planes only.
    /// Fold source head is native BF16 (never a compact trellis payload).
    pub unsafe fn launch(&self,buffers:&[Buffer],rows:usize,eps:f32,first:usize,stream:*mut c_void)->Result<()> {
        let cap=self.program.spec().capacity_rows as usize;
        ensure!(rows>0 && rows<=cap,"recurrent primitive rows exceed export capacity");
        ensure!(eps.is_finite() && eps>0.,"recurrent primitive epsilon invalid");
        if self.op==Op::Fold { ensure!(first.checked_add(rows).is_some_and(|end|end<=201024),"fold row tile exceeds head"); }
        let extents=self.op.shapes(rows)?;
        ensure!(buffers.len()==extents.len(),"recurrent primitive pointer count");
        let device=buffers[0].device_id;
        for (index,(&buffer,&bytes)) in buffers.iter().zip(&extents).enumerate() {
            if bytes==0 {continue;}
            ensure!(buffer.device_id==device && !buffer.ptr.is_null() && buffer.ptr as usize%2==0
                && buffer.bytes>=bytes && (buffer.ptr as usize).checked_add(bytes).is_some(),
                "recurrent primitive operand {index} outside admitted extent");
        }
        let first_output=buffers.len()-self.op.outputs();
        for index in first_output..buffers.len() {
            if extents[index]==0 {continue;}
            for other in 0..buffers.len() {
                if index==other || extents[other]==0 {continue;}
                let (a,b)=(buffers[index].ptr as usize,buffers[other].ptr as usize);
                ensure!(a+extents[index]<=b || b+extents[other]<=a,"recurrent primitive output aliases operand");
            }
        }
        let scalars=match self.op {
            Op::Mix|Op::ResidualNorm|Op::QkRope{..}=>vec![Scalar::I32(i32::try_from(rows)?),Scalar::F32(eps)],
            Op::Fold=>vec![Scalar::I32(i32::try_from(first)?),Scalar::I32(i32::try_from(rows)?)],
            _=>vec![Scalar::I32(i32::try_from(rows)?)],
        };
        // SAFETY: validated extents and disjoint outputs; caller retains owners,
        // initializes source/slots and orders the retained stream.
        unsafe{self.program.launch(&buffers.iter().map(|buffer|buffer.ptr).collect::<Vec<_>>(),&scalars,stream)}
    }
}

fn validate_spec(op:Op,spec:&ProgramSpec)->Result<()> {
    let capacity=match op {Op::Mask|Op::Gate=>64,Op::Finish|Op::Bias{..}=>448,_=>4096};
    ensure!(spec.name==op.name()? && spec.capacity_rows==capacity
        && spec.scalar_kinds==op.scalar_kinds() && spec.scratch.is_empty()
        && spec.pointers.iter().map(String::as_str).eq(op.pointers().iter().copied()),
        "recurrent primitive manifest ABI/capacity differs");
    let g=&spec.geometry;
    let equal=|key:&str,value:serde_json::Value|->Result<()> {
        ensure!(g[key]==value,"recurrent primitive geometry.{key} differs");Ok(())
    };
    match op {
        Op::Mix=>{equal("hidden",4608.into())?;equal("taps",8.into())?;}
        Op::Mask=>{equal("hidden",4608.into())?;equal("block",8.into())?;}
        Op::Gate=>equal("rank",256.into())?,
        Op::Fold=>{equal("vocab",201024.into())?;equal("hidden",4608.into())?;equal("rank",256.into())?;}
        Op::Finish=>{equal("vocab",201024.into())?;equal("valid_vocab",200058.into())?;equal("softcap",30.into())?;}
        Op::ResidualNorm=>equal("hidden",4608.into())?,
        Op::QkRope{query_heads,kv_heads}=>{equal("q_heads",query_heads.into())?;equal("kv_heads",kv_heads.into())?;equal("head_dim",128.into())?;}
        Op::Silu{intermediate}=>equal("intermediate",intermediate.into())?,
        Op::Bias{width,bf16}=>{equal("width",width.into())?;equal("output_dtype",if bf16{"torch.bfloat16"}else{"torch.float32"}.into())?;}
        Op::Store{kv_heads,fp8}=>{equal("kv_heads",kv_heads.into())?;equal("kv_format",if fp8{"fp8"}else{"bf16"}.into())?;
            equal("plane_bytes",if fp8{(kv_heads*132).div_ceil(16)*16}else{kv_heads*256}.into())?;}
    }
    Ok(())
}
