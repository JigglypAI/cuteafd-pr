//! Trimmed vocabulary and final softcap use the shared target-head route.
use super::{layers::Kernels, owners::{UnitBuffers, UnitOwner}, weights::{ProjectionWeight, TargetWeights}};
use crate::shared::{memory::device::{Allocation, Stream}, target_head::{self, TargetHead}};
use anyhow::{ensure, Result};
use cuteafd_core::memory_layout::Tag;
use cuteafd_ffi::{inkling2::producers::Precision, programs::VocabularyHead, CuteafdDeviceBuffer as Buffer};
use cuteafd_loader::families::inkling2::Inkling2Config;
use std::rc::Rc;

pub(crate) struct Head<'a> {
    stream: Rc<Stream<'a>>,
    bf16_head: Option<(VocabularyHead<'a>, Allocation<'a>)>,
    tokens: Allocation<'a>,
    normalized: Allocation<'a>,
    pub logits: Allocation<'a>,
    compact_logits: Option<(usize, Allocation<'a>)>,
    rows: usize,
    local_vocab: usize,
    split: Option<Split<'a>>,
}

struct Split<'a> {
    handoff: crate::shared::distributed_head::Handoff<'a>,
    peer: HeadShard<'a>,
    lead_logits: Allocation<'a>,
}

struct HeadShard<'a> {
    stream: Rc<Stream<'a>>,
    input: Allocation<'a>,
    logits: Allocation<'a>,
    weight: Option<(VocabularyHead<'a>, Allocation<'a>)>,
    draft_output: Option<Allocation<'a>>,
    valid: usize,
    pitch: usize,
}

#[derive(Clone, Copy)]
pub(super) struct PeerHead<'s, 'a> {
    pub kernels: &'s Kernels<'a>,
    pub target: &'s TargetWeights<'a>,
}

impl<'a> Head<'a> {
    /// Caller admits tokens, normalized, logits and shared BLAS workspace first.
    pub fn new(stream: Rc<Stream<'a>>, cfg: &Inkling2Config, rows: usize, stored_head: Option<usize>) -> Result<Self> {
        Self::with_local(stream, cfg, rows, stored_head, cfg.unpadded_vocab_size)
    }
    fn with_local(stream: Rc<Stream<'a>>, cfg: &Inkling2Config, rows: usize,
        stored_head: Option<usize>, local_vocab: usize) -> Result<Self> {
        ensure!(rows>0 && rows<=4096, "Inkling head rows exceed export");
        let d = stream.device;
        let alloc = |bytes, name| Allocation::new(d, bytes, Tag::workspace(name));
        let tokens = alloc(rows*4, "inkling2/head/tokens")?;
        let normalized = alloc(rows*cfg.hidden*2, "inkling2/head/normalized")?;
        let logits = alloc(rows*cfg.unpadded_vocab_size*4, "inkling2/head/logits")?;
        let compact_logits = stored_head.map(|stored| Ok::<_, anyhow::Error>((stored,
            alloc(rows.min(64)*stored*4, "inkling2/head/compact_logits")?))).transpose()?;
        let bf16_head = if stored_head.is_none() { Some(blas(&stream, cfg.hidden, rows, local_vocab)?) } else { None };
        Ok(Self { stream, bf16_head, tokens, normalized, logits, compact_logits, rows, local_vocab, split: None })
    }
    pub fn pair(stream: Rc<Stream<'a>>, peer_stream: Rc<Stream<'a>>, cfg: &Inkling2Config,
        rows: usize, stored: [Option<usize>;2], draft: bool) -> Result<Self> {
        let valid = super::vocab_span(cfg, 0)?.len();
        let peer_valid = super::vocab_span(cfg, 1)?.len();
        let mut head = Self::with_local(stream.clone(), cfg, rows, stored[0], valid)?;
        let local = |bytes, name| Allocation::new(stream.device, bytes, Tag::workspace(name));
        let remote = |bytes, name| Allocation::new(peer_stream.device, bytes, Tag::workspace(name));
        let lead_logits = local(rows*valid*4, "inkling2/head/local_logits")?;
        let pitch = stored[1].unwrap_or(peer_valid);
        let peer = HeadShard {
            input: remote(rows*cfg.hidden*2, "inkling2/head/peer_input")?,
            logits: remote(rows*pitch*4, "inkling2/head/peer_logits")?,
            weight: if stored[1].is_none() { Some(blas(&peer_stream, cfg.hidden, rows, peer_valid)?) } else { None },
            draft_output: if draft && stored[1].is_some() {
                Some(remote(rows*pitch*2,"inkling2/head/peer_draft_output")?)
            } else { None },
            stream: peer_stream.clone(), valid: peer_valid, pitch,
        };
        head.split = Some(Split { handoff: crate::shared::distributed_head::Handoff::new(stream, peer_stream)?, peer, lead_logits });
        Ok(head)
    }
    pub(super) fn peer_weight(&self) -> Option<usize> { self.split.as_ref().map(|s| s.peer.pitch) }
    pub(super) fn stream(&self) -> &Stream<'a> { &self.stream }
    pub(super) fn local_stage(&self) -> Result<Buffer> {
        self.split.as_ref().map(|s| s.lead_logits.buffer).ok_or_else(|| anyhow::anyhow!("Inkling local shard stage missing"))
    }
    pub(super) fn peer_stream(&self) -> Option<&Stream<'a>> { self.split.as_ref().map(|s| s.peer.stream.as_ref()) }
    pub(super) fn peer_draft_output(&self) -> Result<Buffer> {
        self.split.as_ref().and_then(|s| s.peer.draft_output.as_ref()).map(|a| a.buffer)
            .ok_or_else(|| anyhow::anyhow!("Inkling peer draft stage missing"))
    }
    pub(super) fn peer_stages(&self) -> Result<(Buffer,Buffer)> {
        let split = self.split.as_ref().ok_or_else(|| anyhow::anyhow!("Inkling peer head missing"))?;
        Ok((split.peer.input.buffer,split.peer.logits.buffer))
    }
    pub(super) fn handoff(&self) -> Option<&crate::shared::distributed_head::Handoff<'a>> {
        self.split.as_ref().map(|s| &s.handoff)

    }
    /// The recurrent head borrows the verifier's idle, same-stream stages.
    pub(super) fn compact_stages(&self) -> Result<(usize, Buffer, Buffer)> {
        let (stored, stage) = self.compact_logits.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Inkling compact head scratch missing"))?;
        Ok((*stored, stage.buffer, self.logits.buffer))
    }
    pub fn inventory(&self, inventory: &mut crate::shared::memory::receipts::Inventory<'_>, prefix: &str) -> Result<()> {
        for (name, owner) in [("tokens", &self.tokens), ("normalized", &self.normalized),
            ("logits", &self.logits)] {
            owner.inventory(inventory, &format!("{prefix}.{name}"))?;
        }
        if let Some((_, workspace)) = &self.bf16_head { workspace.inventory(inventory, &format!("{prefix}.workspace"))?; }
        if let Some((_, owner)) = &self.compact_logits { owner.inventory(inventory, &format!("{prefix}.compact_logits"))?; }
        if let Some(split) = &self.split {
            split.lead_logits.inventory(inventory, &format!("{prefix}.local_logits"))?;
            split.peer.input.inventory(inventory, &format!("{prefix}.peer_input"))?;
            if let Some(output) = &split.peer.draft_output { output.inventory(inventory, &format!("{prefix}.peer_draft_output"))?; }
            split.peer.logits.inventory(inventory, &format!("{prefix}.peer_logits"))?;
            if let Some((_, owner)) = &split.peer.weight { owner.inventory(inventory, &format!("{prefix}.peer_workspace"))?; }
        }
        Ok(())
    }
    pub fn embed(&self, k: &Kernels<'_>, owner: &UnitOwner<'_>, target: &TargetWeights<'a>,
        cfg: &Inkling2Config, tokens: &[u32]) -> Result<()> {
        ensure!(!tokens.is_empty() && tokens.len()<=self.rows && tokens.iter().all(|&t| (t as usize)<cfg.unpadded_vocab_size),
            "Inkling embedding token bounds");
        let d = self.stream.device;
        self.stream.drain()?;
        d.run(|| {
            d.library.copy_h2d(self.tokens.buffer, &tokens.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())?;
            // SAFETY: shared gather reads native BF16 arena; zero+accumulate is
            // an exact widening into the retained F32 residual.
            unsafe {
                d.library.cuda_embedding_lookup_bf16_async(target.operand("model.llm.embed.weight")?.buffer,
                    self.tokens.buffer, self.normalized.buffer, tokens.len(), cfg.vocab_size, cfg.hidden, self.stream.raw)?;
                d.library.cuda_zero_bytes_async(owner.buffer("branch")?, tokens.len()*cfg.hidden*4, self.stream.raw)?;
                d.library.cuda_accumulate_bf16_to_f32_async(self.normalized.buffer, owner.buffer("branch")?,
                    tokens.len()*cfg.hidden, self.stream.raw)?;
            }
            k.norm(owner, owner.buffer("branch")?, target.operand("model.llm.embed_norm.weight")?.buffer,
                owner.buffer("partial")?, false, Precision::F32, tokens.len())
        })
    }
    /// Prefill embedding uses existing unit planes, not the bounded vocabulary
    /// head's <=64-row token/normalization storage. `first` is metadata scratch
    /// until segment_metadata publishes convolution starts before attention.
    pub fn embed_unit(&self, k: &Kernels<'a>, owner: &UnitOwner<'a>, target: &TargetWeights<'a>,
        cfg: &Inkling2Config, tokens: &[u32], media: &cuteafd_engine::media::MediaChunk,
        uploads: &mut crate::shared::token_table::RetainedUploads<'a>) -> Result<()> {
        ensure!(!tokens.is_empty() && tokens.len() <= owner.layout.rows
            && tokens.iter().all(|&token| (token as usize) < cfg.unpadded_vocab_size)
            && owner.stream.raw == self.stream.raw && owner.stream.device.id == self.stream.device.id,
            "Inkling prefill embedding exceeds admitted owner/stream");
        let bytes = tokens.iter().flat_map(|token| token.to_le_bytes()).collect::<Vec<_>>();
        self.stream.device.run(|| {
            // SAFETY: shared upload retains source until the gather reads token
            // IDs; segment metadata overwrites the same plane later in order.
            unsafe {
                uploads.upload(owner.buffer("first")?, &bytes, self.stream.raw)?;
                self.stream.device.library.cuda_embedding_lookup_bf16_async(target.operand("model.llm.embed.weight")?.buffer,
                    owner.buffer("first")?, owner.buffer("projection_input")?, tokens.len(), cfg.vocab_size, cfg.hidden, self.stream.raw)?;
            }
            // SAFETY: admitted planes and the retained unit stream order gather
            // before the exact widening of the text embeddings.
            unsafe {
                self.stream.device.library.cuda_zero_bytes_async(owner.buffer("branch")?, tokens.len()*cfg.hidden*4, self.stream.raw)?;
                self.stream.device.library.cuda_accumulate_bf16_to_f32_async(owner.buffer("projection_input")?, owner.buffer("branch")?,
                    tokens.len()*cfg.hidden, self.stream.raw)?;
            }
            k.norm(owner, owner.buffer("branch")?, target.operand("model.llm.embed_norm.weight")?.buffer,
                owner.buffer("partial")?, false, Precision::F32, tokens.len())?;
            // Vendor Inkling2 embed_input_ids normalizes text before merging
            // projected image features (vLLM d664cd1e model.py:663-671).
            crate::shared::vision::still::inject_f32(&owner.stream, media, owner.buffer("branch")?,
                owner.buffer("route_ids")?, owner.buffer("projection_input")?, owner.buffer("partial")?,
                tokens.len(), cfg.hidden)
        })
    }
    pub fn project(&self, k: &Kernels<'a>, owner: &impl UnitBuffers<'a>, target: &TargetWeights<'a>,
        cfg: &Inkling2Config, rows: usize, candidates: usize) -> Result<Buffer> {
        self.project_at(k, owner, target, None, cfg, rows, candidates, 0)
    }
    pub fn project_at(&self, k: &Kernels<'a>, owner: &impl UnitBuffers<'a>, target: &TargetWeights<'a>,
        peer: Option<PeerHead<'_, 'a>>, cfg: &Inkling2Config, rows: usize, candidates: usize, output_row: usize) -> Result<Buffer> {
        ensure!(rows>0 && output_row.checked_add(rows).is_some_and(|end| end <= self.rows) && owner.stream().raw == self.stream.raw
            && owner.stream().device.id == self.stream.device.id,
            "Inkling head rows/producer stream differ from admitted owner");
        self.normalize(k,owner,target,rows,candidates)?;
        let logits = super::cache::view(self.logits.buffer, output_row*cfg.unpadded_vocab_size*4,
            rows*cfg.unpadded_vocab_size*4)?;
        if let Some(split) = &self.split {
            let peer = peer.ok_or_else(|| anyhow::anyhow!("Inkling split head must bind its peer"))?;
            let input = self.normalized(rows)?;
            let local = self.compact_logits.as_ref().map_or(split.lead_logits.buffer, |(_, b)| b.buffer);
            let pitch = self.compact_logits.as_ref().map_or(self.local_vocab, |(stored, _)| *stored);
            let remote = split.peer.logits.buffer;
            let peer_input = split.peer.input.buffer;
            split.handoff.project(input, peer_input,
                || self.project_shard(&self.stream,k,target,input,local,rows,cfg.hidden,pitch,self.bf16_head.as_ref()),
                || self.project_shard(&split.peer.stream,peer.kernels,peer.target,peer_input,remote,rows,cfg.hidden,
                    split.peer.pitch,split.peer.weight.as_ref()),
                || {
                    let dst_pitch = cfg.unpadded_vocab_size*4;
                    let local_bytes = self.local_vocab*4;
                    split.handoff.copy_columns(logits,local,rows,dst_pitch,pitch*4,local_bytes)?;
                    split.handoff.copy_columns(super::cache::view(logits,local_bytes,logits.bytes-local_bytes)?,remote,
                        rows,dst_pitch,split.peer.pitch*4,split.peer.valid*4)
                })?;
            self.softcap(logits,rows,cfg)?;
            return Ok(logits);
        }
        ensure!(peer.is_none(), "Inkling whole head got a peer");
        let weight = target.projection("model.llm.unembed.weight")?;
        let lib = self.stream.device.library;
        if let ProjectionWeight::Bf16(weight) = weight {
            let (head, _) = self.bf16_head.as_ref().ok_or_else(|| anyhow::anyhow!("Inkling BF16 head not admitted"))?;
            // SAFETY: all admitted BF16 and F32 views survive the ordered stream.
            unsafe { target_head::launch(TargetHead::configured(), head, self.normalized.buffer.ptr,
                weight.buffer.ptr, logits.ptr.cast(), rows, self.stream.raw)?; }
        } else {
            let (stored, stage) = self.compact_logits.as_ref().ok_or_else(|| anyhow::anyhow!("Inkling compact head scratch missing"))?;
            for first in (0..rows).step_by(64) {
                let count = (rows-first).min(64);
                let input = super::cache::view(self.normalized.buffer, first*cfg.hidden*2, count*cfg.hidden*2)?;
                let output = super::cache::view(logits, first*cfg.unpadded_vocab_size*4, count*cfg.unpadded_vocab_size*4)?;
                k.project_target(owner, input, target.projection("model.llm.unembed.weight")?, stage.buffer, count, cfg.hidden, *stored)?;
                // SAFETY: every staged row keeps its physical vocabulary pitch;
                // the distinct destination strips only the padded columns.
                unsafe { lib.copy_d2d_2d_async(output, cfg.unpadded_vocab_size*4,
                    stage.buffer, stored*4, cfg.unpadded_vocab_size*4, count, self.stream.raw)?; }
            }
        }
        self.softcap(logits,rows,cfg)?;
        Ok(logits)
    }
    fn project_shard(&self, stream: &Stream<'a>, kernels: &Kernels<'a>, target: &TargetWeights<'a>, input: Buffer,
        output: Buffer, rows: usize, hidden: usize, pitch: usize, blas: Option<&(VocabularyHead<'a>, Allocation<'a>)>) -> Result<()> {
        let weight = target.projection("model.llm.unembed.weight")?;
        if let ProjectionWeight::Bf16(weight) = weight {
            let (head, _) = blas.ok_or_else(|| anyhow::anyhow!("Inkling BF16 shard workspace missing"))?;
            // SAFETY: the admitted local input, weights, and output outlive the ordered stream.
            unsafe { target_head::launch(TargetHead::configured(),head,input.ptr,weight.buffer.ptr,output.ptr.cast(),rows,stream.raw) }
        } else {
            kernels.project_compact(stream,input,weight,output,rows,hidden,pitch)
        }
    }
    fn softcap(&self, logits: Buffer, rows: usize, cfg: &Inkling2Config) -> Result<()> {
        // SAFETY: logits are now a contiguous, valid-vocabulary F32 view.
        unsafe { self.stream.device.library.cuda_logits_softcap_f32_async(logits.ptr, rows,
            cfg.unpadded_vocab_size, cfg.unpadded_vocab_size, cfg.final_softcap as f32, self.stream.raw) }?;
        Ok(())
    }
    pub fn normalize(&self,k:&Kernels<'a>,owner:&impl UnitBuffers<'a>,target:&TargetWeights<'_>,
        rows:usize,candidates:usize)->Result<Buffer> {
        ensure!((1..=self.rows).contains(&rows),"Inkling normalized tap exceeds head owner");
        k.residual(owner,target.operand("model.llm.attn_res.output_query")?.buffer,
            owner.buffer("partial")?,owner.buffer("mixed")?,rows,1,candidates,false)?;
        k.norm(owner,owner.buffer("mixed")?,target.operand("model.llm.norm.weight")?.buffer,
            self.normalized.buffer,false,Precision::Bf16,rows)?;
        self.normalized(rows)
    }
    pub fn normalized(&self,rows:usize)->Result<Buffer> {
        ensure!((1..=self.rows).contains(&rows),"Inkling final norm tap rows outside head owner");
        super::cache::view(self.normalized.buffer,0,rows*4608*2)
    }
    /// Vocabulary storage remains <=64 rows regardless of prompt length. The
    /// shared consumer finishes each tile before the next overwrites it.
    pub fn project_spans(&self, k: &Kernels<'a>, owner: &UnitOwner<'a>, cache: &super::cache::CacheBank<'a>,
        sequence: usize, target: &TargetWeights<'a>, cfg: &Inkling2Config, rows: usize, candidates: usize,
        consume: &mut dyn FnMut(usize, crate::shared::token_io::DeviceLogits) -> Result<()>) -> Result<()> {
        ensure!(self.rows <= 64 && rows <= owner.layout.rows && rows > 0,
            "Inkling serving head must be bounded to <=64 rows");
        for span in crate::shared::probe::head_spans(rows, self.rows)? {
            let segment = owner.segment(cache, sequence, cfg.layers-1, span.clone())?;
            let logits = self.project(k, &segment, target, cfg, span.len(), candidates)?;
            consume(span.start, crate::shared::token_io::DeviceLogits { ptr: logits.ptr, rows: span.len(),
                vocab: cfg.unpadded_vocab_size, stride: cfg.unpadded_vocab_size,
                stream: self.stream.raw, greedy: None })?;
        }
        Ok(())
    }
    pub fn project_scoring(&self, k: &Kernels<'a>, owner: &UnitOwner<'a>, cache: &super::cache::CacheBank<'a>,
        sequence: usize, target: &TargetWeights<'a>, cfg: &Inkling2Config, rows: usize, candidates: usize,
        scoring: crate::shared::probe::ScoringRows<'_>) -> Result<crate::shared::probe::ScoreLogits> {
        self.project_spans(k, owner, cache, sequence, target, cfg, rows, candidates,
            &mut |offset, logits| scoring.record(self.stream.device.library, offset, &logits))?;
        Ok(crate::shared::probe::ScoreLogits::Recorded)
    }
}
fn blas<'a>(stream: &Stream<'a>, hidden: usize, rows: usize, vocab: usize) -> Result<(VocabularyHead<'a>,Allocation<'a>)> {
    let d = stream.device;
    let workspace = Allocation::new(d,cuteafd_ffi::programs::VOCABULARY_HEAD_WORKSPACE,Tag::workspace("inkling2/head/blas"))?;
    // SAFETY: the admitted local workspace outlives all launches.
    let head = d.run(|| unsafe { d.library.vocabulary_head_rows(workspace.buffer.ptr, hidden as u32,
        rows as u32,vocab as u32,TargetHead::configured().native_mode()) })?;
    Ok((head,workspace))
}

impl Drop for Head<'_> {
    fn drop(&mut self) {
        self.stream.drain_or_abort("Inkling head handle/workspace retirement");
        if let Some(split) = &self.split { split.peer.stream.drain_or_abort("Inkling peer head retirement"); }
    }
}


#[cfg(test)]
mod media_tests {
    #[test]
    fn vendor_image_rows_replace_normalized_text_without_a_second_norm() {
        let source = include_str!("head.rs").split("#[cfg(test)]").next().unwrap();
        let unit = source.split("pub fn embed_unit(").nth(1).unwrap().split("pub fn project(").next().unwrap();
        assert!(unit.find("k.norm(").unwrap() < unit.find("still::inject_f32(").unwrap());
        assert_eq!(unit.matches("k.norm(").count(),1);
        assert!(unit.contains("owner.buffer(\"partial\")?"));
    }
}
