//! Real checkpoint forward oracle; run only through golden-run's native-test.
use super::{recurrent_forward::{Rank,ProgramsForRank},recurrent_weights::Weights,
    recurrent_cache::{ContextBank,ContextState},dflash::DraftSeq};
use crate::shared::{experts::rtx::dense::{DenseLoad,Operand,Rect},
    memory::{device::{Allocation,Device,Stream},Tag,weight_arena::DeviceWeightArena}};
use anyhow::{ensure,Context,Result};
use cuteafd_core::memory_layout::ArenaTag;
use cuteafd_loader::{formats::block_dspark::recurrent::{loading::RecurrentSource,runtime::RecurrentRuntimeLayout},
    plan::checkpoint::{CheckpointView,Checkpoint,CoreView},serving_capacity::WeightArenaLayout};
use cuteafd_ffi::CuteafdDeviceBuffer as Buffer;
use std::{collections::BTreeMap,path::Path,rc::Rc,os::unix::fs::FileExt};

fn tensor(path:&Path,name:&str)->Result<Vec<u8>> {
    let header=cuteafd_loader::read_safetensors_metadata(path)?.into_iter().find(|t|t.name==name).context("oracle fixture tensor absent")?;
    let mut bytes=vec![0;usize::try_from(header.byte_length)?];
    std::fs::File::open(path)?.read_exact_at(&mut bytes,header.byte_offset)?;Ok(bytes)
}
fn bf16_values(bytes:&[u8])->Vec<f32>{bytes.chunks_exact(2).map(|v|f32::from_bits(u32::from(u16::from_le_bytes([v[0],v[1]]))<<16)).collect()}
fn compare(actual:&[u8],expected:&[u8],name:&str)->Result<()> {
    ensure!(actual.len()==expected.len(),"oracle {name}: extent differs");
    let a=bf16_values(actual);let b=bf16_values(expected);
    let error=a.iter().zip(&b).map(|(a,b)|f64::from(a-b).powi(2)).sum::<f64>().sqrt();
    let norm=b.iter().map(|b|f64::from(*b).powi(2)).sum::<f64>().sqrt().max(1e-12);
    let equal=actual.chunks_exact(2).zip(expected.chunks_exact(2)).filter(|(a,b)|a==b).count();
    eprintln!("recurrent oracle {name}: relative_l2={} bit_equal={}/{}",error/norm,equal,a.len());
    ensure!(error/norm<0.02,"oracle {name}: relative L2 exceeds0.02");Ok(())
}

#[test]
#[ignore="requires real Inkling BF16 MTP tensors, native programs and GPU; golden-run only"]
fn installed_inkling_recurrent_forward_matches_real_weight_reference()->Result<()> {
    // SAFETY: golden-run mounts the matching sealed trusted native library.
    let library=unsafe{cuteafd_ffi::NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)?};
    let programs=library.programs()?.with_manifest(Path::new(&std::env::var("CUTEAFD_PROGRAM_MANIFEST")?))?;
    let snapshot=std::path::PathBuf::from(std::env::var("CUTEAFD_TEST_DRAFT_SNAPSHOT")?);
    let fixture=std::path::PathBuf::from(std::env::var("CUTEAFD_GOLDEN_FIXTURES")?);
    let device=Device{library:&library,id:0};
    let stream=Rc::new(Stream::new(device)?);
    let source=RecurrentSource::open(&CheckpointView::select(&snapshot,CoreView::Original)?)?;
    let layout=source.rank(0,1,0)?;
    let weights=Weights::load(device,&source,layout,stream.raw)?;
    let runtime=RecurrentRuntimeLayout::new(&source.config,1,1,2048,2048,4096,library.sm_count()? as usize,
        cuteafd_loader::families::qwen4::Qwen4KvCache::Bf16,0,1,0)?;
    let native=ProgramsForRank::load(&programs,&weights,&source.config,&runtime,device)?;
    let rank=Rank::allocate(weights,Rc::clone(&stream),source.config.clone(),runtime,native)?;
    let target=Checkpoint::open_for_role(&snapshot,"recurrent-oracle-target",|name|
        matches!(name,"model.llm.embed.weight"|"model.llm.embed_norm.weight"|"model.llm.unembed.weight"))?;
    let names=["model.llm.embed.weight","model.llm.embed_norm.weight","model.llm.unembed.weight"];
    let target_arena=DeviceWeightArena::new(device,WeightArenaLayout::new("oracle/borrowed-target",
        names.iter().map(|&name|Ok((name.to_owned(),usize::try_from(target.header(name).context("oracle target missing")?.byte_length)?)))
            .collect::<Result<Vec<_>>>()?)?,ArenaTag::weights("oracle/target"))?;
    let dense=DenseLoad::mount(&library);let mut targets=BTreeMap::new();
    for name in names {
        let rect=Rect::tensor(&target,name,None,None)?;let bytes=rect.bytes();let owner=target_arena.view(name,bytes)?;
        dense.operand(Operand{device:0,buffer:owner.buffer,lands:vec![(rect,0)],bytes})?;targets.insert(name,owner);
    }
    dense.release();rank.initialize_derived()?;rank.fold_head(targets["model.llm.unembed.weight"].buffer)?;stream.drain()?;
    for rows in [3usize,1031,2048] {
        let file=fixture.join(format!("rows-{rows}.safetensors"));
        let features=tensor(&file,"features")?;
        library.copy_h2d(rank.buffer("context.taps")?,&features)?;
        let bank=ContextBank{state:std::cell::RefCell::new(ContextState::new(1,2048,4096,rows as u64)?)};
        bank.state.borrow_mut().reset(0,0)?;
        bank.append(&rank,0,0,rank.buffer("context.taps")?,rows)?;
        let sequence=[DraftSeq{slot:0,position:rows,anchor:101,valid_from:0}];
        let pages=bank.state.borrow().pages(&sequence)?;
        library.copy_h2d(rank.buffer("step.ids")?,&101u32.to_le_bytes())?;
        rank.initialize_block(rank.buffer("step.ids")?,targets["model.llm.embed.weight"].buffer,
            targets["model.llm.embed_norm.weight"].buffer,1)?;
        rank.forward_whole(&sequence,&pages)?;
        // SAFETY: last committed feature and head feature are distinct admitted
        // BF16 planes on this ordered stream.
        unsafe{library.copy_d2d_async(rank.buffer("head.f_rnn")?,rank.buffer("cache.last_features")?,4608*2,stream.raw)?;}
        rank.recurrent_tokens(1,rank.buffer("step.ids")?,targets["model.llm.unembed.weight"].buffer)?;
        stream.drain()?;
        let expected=tensor(&file,"hidden")?;let mut actual=vec![0;expected.len()];
        library.copy_d2h(&mut actual,Buffer{bytes:expected.len(),..rank.buffer("step.normalized")?})?;
        compare(&actual,&expected,&format!("rows{rows}.hidden"))?;
        let expected=tensor(&file,"tokens")?;let mut actual=vec![0;expected.len()];
        library.copy_d2h(&mut actual,Buffer{bytes:expected.len(),..rank.buffer("head.tokens")?})?;
        ensure!(actual==expected,"recurrent oracle rows{rows}: seven predicted token ids differ");
    }
    Ok(())
}
