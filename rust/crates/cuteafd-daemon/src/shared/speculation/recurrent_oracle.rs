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

/// Real target head weights, all nine stages of the unchanged recurrent token
/// loop, and deterministic inputs to isolate the head from a served target.
#[test]
#[ignore = "requires an Inkling MC checkpoint, native programs and GPU; golden-run only"]
fn installed_inkling_recurrent_head_baseline() -> Result<()> {
    use super::recurrent_head::HeadStage;
    use crate::{families::inkling2::{layers::Kernels, weights::{ProjectionWeight, TargetWeights}},
        shared::memory::device::Event};
    use cuteafd_loader::{families::inkling2::{Inkling2Config, InklingKvCache,
        weight_layout::{CoordinatorWeights, HEAD}}, placement::{FfnMode, LayerMode}};

    let snapshot = std::path::PathBuf::from(std::env::var("CUTEAFD_GOLDEN_SNAPSHOT")?);
    let mode = std::env::var("CUTEAFD_TEST_CORE_VIEW")?.parse::<CoreView>()?;
    let manifest = std::path::PathBuf::from(std::env::var("CUTEAFD_PROGRAM_MANIFEST")?);
    // SAFETY: golden-run mounts the matching sealed trusted native library.
    let library = unsafe { cuteafd_ffi::NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
    let programs = library.programs()?.with_manifest(&manifest)?;
    let device = Device { library: &library, id: 0 };
    let stream = Rc::new(Stream::new(device)?);
    let source = RecurrentSource::open(&CheckpointView::select(&snapshot, CoreView::Original)?)?;
    let weights = Weights::load(device, &source, source.rank(0, 1, 0)?, stream.raw)?;
    let config = &source.config;
    let runtime = RecurrentRuntimeLayout::new(config, 64, 64, 512, 2048, 4096,
        library.sm_count()? as usize, cuteafd_loader::families::qwen4::Qwen4KvCache::Bf16, 0, 1, 0)?;
    let native = ProgramsForRank::load(&programs, &weights, config, &runtime, device)?;
    let rank = Rank::allocate(weights, Rc::clone(&stream), config.clone(), runtime, native)?;

    let view = CheckpointView::select(&snapshot, mode)?;
    let checkpoint = Checkpoint::open_view(&view)?;
    let cfg = Inkling2Config::from_hf(&checkpoint.config)?;
    let layers = vec![LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner }; cfg.layers];
    let mut layout = CoordinatorWeights::plan(&checkpoint, &cfg, 1, &layers)?.remove(0);
    layout.layers.clear();
    layout.core.retain(|name, _| name == HEAD);
    layout.fp8.retain(|name, _| name == HEAD);
    layout.arena = WeightArenaLayout::new("inkling2/recurrent-head-bench", layout.arena.views().iter()
        .filter(|(name, _)| name.starts_with(&format!("{HEAD}.")))
        .map(|(name, view)| (name.clone(), view.bytes())))?;
    let target = TargetWeights::load(device, &checkpoint, &layout)?;
    let stored = layout.stored_head().unwrap_or(config.padded_vocab);
    let f32_stage = Allocation::new(device, 64*stored*4, Tag::drafter_workspace("head-bench/f32"))?;
    let bf16_stage = Allocation::new(device, 64*stored*2, Tag::drafter_workspace("head-bench/bf16"))?;
    let mut kernels = Kernels::bind(&programs, InklingKvCache::Bf16)?;
    let fp8_bytes = layout.fp8_scratch(&serde_json::from_slice(&std::fs::read(&manifest)?)?, 64, 64)?;
    kernels.bind_compact(&programs, Rc::clone(&stream), &layout, 64, 64, fp8_bytes)?;

    // No fold is timed. Synthetic nonzero folded operands do equivalent work
    // without adding a second head or depending on a future compact fold.
    for (index, buffer) in [rank.buffer("step.normalized")?, rank.buffer("head.f_rnn")?,
        rank.weights.derived("derived.fold_weight")?.buffer, rank.weights.derived("derived.fold_bias")?.buffer]
        .into_iter().enumerate() {
        let pattern = [0x3c00u16, 0x3b80, 0x0000, 0xbb80, 0xbc00, 0x3c80, 0xbc80];
        let data = (0..buffer.bytes/2).flat_map(|j| pattern[(j ^ (j >> 5) ^ index) % pattern.len()].to_le_bytes())
            .collect::<Vec<_>>();
        library.copy_h2d(buffer, &data)?;
    }
    let anchors = rank.buffer("step.ids")?;
    let mut ids = vec![0; anchors.bytes];
    for (i, dst) in ids.chunks_exact_mut(4).take(64).enumerate() {
        dst.copy_from_slice(&(101 + i as u32).to_le_bytes());
    }
    library.copy_h2d(anchors, &ids)?;

    const SAMPLES: usize = 12;
    const STAGES: usize = HeadStage::Copies as usize;
    let predictions = config.predictions();
    let h = config.transformer.hidden;
    let n = config.padded_vocab;
    let narrow = library.f32_bf16_narrow()?;
    let head = || target.projection("model.llm.unembed.weight");
    let head_bytes = match head()? {
        ProjectionWeight::Bf16(w) => w.buffer.bytes,
        ProjectionWeight::Trellis(w) => w.payload.buffer.bytes + w.suh.buffer.bytes + w.svh.buffer.bytes,
        ProjectionWeight::Fp8(w) => w.values.buffer.bytes + w.scales.buffer.bytes,
    };
    for rows in [1, 4, 16, 64] {
        let base = |hidden: Buffer, output: Buffer| -> Result<()> {
            let weight = head()?;
            if let ProjectionWeight::Bf16(w) = weight {
                // SAFETY: same retained BF16 operands as the original head.
                unsafe { rank.programs.plain.launch_checked(hidden.ptr, w.buffer.ptr, output, rows, h, n, stream.raw)?; }
                return Ok(());
            }
            kernels.project_compact(&stream, hidden, weight, f32_stage.buffer, rows, h, stored)?;
            // SAFETY: typed scratch covers physical row pitch; copy strips only
            // compact padding and leaves the unchanged BF16 baseline epilogue.
            unsafe {
                narrow.launch(f32_stage.buffer.ptr, bf16_stage.buffer.ptr, rows*stored, stream.raw)?;
                library.copy_d2d_2d_async(output, n*2, bf16_stage.buffer, stored*2, n*2, rows, stream.raw)
            }
        };
        rank.recurrent_tokens_measured(rows, anchors, base, |_, _| Ok(()))?;
        stream.drain()?;
        let mut totals = [[0.0; STAGES]; 7];
        for _ in 0..SAMPLES {
            let mut events = (0..predictions*(STAGES+1)).map(|_| Event::timing(device)).collect::<Result<Vec<_>>>()?;
            rank.recurrent_tokens_measured(rows, anchors, base,
                |step, stage| events[step*(STAGES+1)+stage as usize].record(&stream))?;
            stream.drain()?;
            for (step, stages) in totals.iter_mut().enumerate() {
                for (stage, total) in stages.iter_mut().enumerate() {
                    let i = step*(STAGES+1) + stage;
                    // SAFETY: both events are recorded and drained on this stream.
                    *total += f64::from(unsafe { library.cuda_event_elapsed_ms(events[i].raw, events[i+1].raw)? }) * 1e3;
                }
            }
        }
        let mean = (0..STAGES).map(|stage| totals.iter().map(|step| step[stage]).sum::<f64>() / (SAMPLES*predictions) as f64)
            .collect::<Vec<_>>();
        eprintln!("RECURRENT_HEAD_BASELINE {}", serde_json::json!({"view": mode.name(), "rows": rows,
            "head_bytes": head_bytes, "roof_us": head_bytes as f64 / 1_792_000.0,
            "gather_us": mean[0], "recurrent_us": mean[1], "base_us": mean[2], "correction_us": mean[3],
            "bias_us": mean[4], "finish_us": mean[5], "widen_us": mean[6], "argmax_us": mean[7],
            "copies_us": mean[8], "total_us": mean.iter().sum::<f64>(),
            "per_step_total_us": totals.iter().map(|step| step.iter().sum::<f64>() / SAMPLES as f64).collect::<Vec<_>>() }));
    }
    Ok(())
}
