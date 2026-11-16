//! Trimmed vocabulary and final softcap use the shared target-head route.
use super::{layers::Kernels, owners::{UnitBuffers, UnitOwner}, weights::TargetWeights};
use crate::shared::{memory::device::{Allocation, Stream}, target_head::{self, TargetHead}};
use anyhow::{ensure, Result};
use cuteafd_core::memory_layout::Tag;
use cuteafd_ffi::{inkling2::producers::Precision, programs::VocabularyHead, CuteafdDeviceBuffer as Buffer};
use cuteafd_loader::families::inkling2::Inkling2Config;
use std::rc::Rc;

pub(crate) struct Head<'a> {
    stream: Rc<Stream<'a>>,
    head: VocabularyHead<'a>,
    tokens: Allocation<'a>,
    normalized: Allocation<'a>,
    pub logits: Allocation<'a>,
    _workspace: Allocation<'a>,
    rows: usize,
}
impl<'a> Head<'a> {
    /// Caller admits tokens, normalized, logits and shared BLAS workspace first.
    pub fn new(stream: Rc<Stream<'a>>, cfg: &Inkling2Config, rows: usize) -> Result<Self> {
        ensure!(rows>0 && rows<=4096, "Inkling head rows exceed export");
        let d = stream.device;
        let alloc = |bytes, name| Allocation::new(d, bytes, Tag::workspace(name));
        let tokens = alloc(rows*4, "inkling2/head/tokens")?;
        let normalized = alloc(rows*cfg.hidden*2, "inkling2/head/normalized")?;
        let logits = alloc(rows*cfg.unpadded_vocab_size*4, "inkling2/head/logits")?;
        let workspace = alloc(Self::workspace_bytes(cfg, rows)?, "inkling2/head/blas")?;
        // SAFETY: admitted workspace survives the handle and stream drain;
        // passing unpadded vocabulary makes the stored tail rows unreachable.
        let head = d.run(|| unsafe { d.library.vocabulary_head_rows(workspace.buffer.ptr, cfg.hidden as u32,
            rows as u32, cfg.unpadded_vocab_size as u32, TargetHead::configured().native_mode()) })?;
        Ok(Self { stream, head, tokens, normalized, logits, _workspace: workspace, rows })
    }
    pub fn workspace_bytes(_cfg: &Inkling2Config, _rows: usize) -> Result<usize> {
        Ok(cuteafd_ffi::programs::VOCABULARY_HEAD_WORKSPACE)
    }
    pub fn inventory(&self, inventory: &mut crate::shared::memory::receipts::Inventory<'_>, prefix: &str) -> Result<()> {
        for (name, owner) in [("tokens", &self.tokens), ("normalized", &self.normalized),
            ("logits", &self.logits), ("workspace", &self._workspace)] {
            owner.inventory(inventory, &format!("{prefix}.{name}"))?;
        }
        Ok(())
    }
    pub fn embed(&self, k: &Kernels<'_>, owner: &UnitOwner<'_>, target: &TargetWeights<'_>,
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
    pub fn embed_unit(&self, k: &Kernels<'a>, owner: &UnitOwner<'a>, target: &TargetWeights<'_>,
        cfg: &Inkling2Config, tokens: &[u32], uploads: &mut crate::shared::token_table::RetainedUploads<'a>) -> Result<()> {
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
                self.stream.device.library.cuda_zero_bytes_async(owner.buffer("branch")?, tokens.len()*cfg.hidden*4, self.stream.raw)?;
                self.stream.device.library.cuda_accumulate_bf16_to_f32_async(owner.buffer("projection_input")?, owner.buffer("branch")?,
                    tokens.len()*cfg.hidden, self.stream.raw)?;
            }
            k.norm(owner, owner.buffer("branch")?, target.operand("model.llm.embed_norm.weight")?.buffer,
                owner.buffer("partial")?, false, Precision::F32, tokens.len())
        })
    }
    pub fn project(&self, k: &Kernels<'a>, owner: &impl UnitBuffers<'a>, target: &TargetWeights<'_>,
        cfg: &Inkling2Config, rows: usize, candidates: usize) -> Result<Buffer> {
        self.project_at(k, owner, target, cfg, rows, candidates, 0)
    }
    pub fn project_at(&self, k: &Kernels<'a>, owner: &impl UnitBuffers<'a>, target: &TargetWeights<'_>,
        cfg: &Inkling2Config, rows: usize, candidates: usize, output_row: usize) -> Result<Buffer> {
        ensure!(rows>0 && output_row.checked_add(rows).is_some_and(|end| end <= self.rows) && owner.stream().raw == self.stream.raw
            && owner.stream().device.id == self.stream.device.id,
            "Inkling head rows/producer stream differ from admitted owner");
        self.normalize(k,owner,target,rows,candidates)?;
        let logits = super::cache::view(self.logits.buffer, output_row*cfg.unpadded_vocab_size*4,
            rows*cfg.unpadded_vocab_size*4)?;
        // SAFETY: BF16 normalized rows and trimmed weight matrix stay live;
        // checked output-row window keeps every request's logits disjoint.
        unsafe {
            target_head::launch(TargetHead::configured(), &self.head, self.normalized.buffer.ptr,
                target.operand("model.llm.unembed.weight")?.buffer.ptr, logits.ptr.cast(), rows, self.stream.raw)?;
            self.stream.device.library.cuda_logits_softcap_f32_async(logits.ptr, rows,
                cfg.unpadded_vocab_size, cfg.unpadded_vocab_size, cfg.final_softcap as f32, self.stream.raw)?;
        }
        Ok(logits)
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
        sequence: usize, target: &TargetWeights<'_>, cfg: &Inkling2Config, rows: usize, candidates: usize,
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
        sequence: usize, target: &TargetWeights<'_>, cfg: &Inkling2Config, rows: usize, candidates: usize,
        scoring: crate::shared::probe::ScoringRows<'_>) -> Result<crate::shared::probe::ScoreLogits> {
        self.project_spans(k, owner, cache, sequence, target, cfg, rows, candidates,
            &mut |offset, logits| scoring.record(self.stream.device.library, offset, &logits))?;
        Ok(crate::shared::probe::ScoreLogits::Recorded)
    }
}
impl Drop for Head<'_> {
    fn drop(&mut self) { self.stream.drain_or_abort("Inkling head handle/workspace retirement"); }
}
