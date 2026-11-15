//! Inkling adapter for the shared request scheduler; no family scheduling loop.
use super::{cache::{InklingPrefix, Placement}, engine::Execution};
use crate::shared::{draft::routes::RoundRoutes, prefix::{CudaCopyEngine, PrefixArgs}, probe,
    prefill_packing::{Limits, MemberOutcome}, prefill_pipeline::{sequence_units, Unit, UnitSegment},
    qwen::scheduler::{self, Backend}, token_io::{DeviceLogits, TokenSelector}};
use anyhow::{ensure, Context, Result};
use cuteafd_api::openai::NativeRequest;
use cuteafd_engine::{media::{EncodeJob, EmbeddingCache, MediaKeys, RequestMedia},
    prefix::{PrefixCache, PrefixConfig, PrefixFamily, RefPagePool}};
use std::{cell::RefCell, collections::BTreeSet, path::Path};

#[derive(Default)]
pub(crate) struct Draft;
impl scheduler::DraftState for Draft {
    fn pending_rows(&self) -> usize { 0 }
    fn close(&mut self, _: u32) {}
}

pub(crate) struct InklingBackend<'a> {
    execution: RefCell<Execution<'a>>,
    family: InklingPrefix<'a>,
    library: &'a cuteafd_ffi::NativeLibrary,
    context: usize,
    concurrency: usize,
    unit_rows: usize,
    waves: usize,
    vocab: usize,
    host: Option<cuteafd_hostcache::config::Config>,
    profile: RefCell<[f64; 2]>,
    pending: RefCell<Option<Vec<(usize, usize, Vec<u32>)>>>,
}
impl<'a> InklingBackend<'a> {
    pub fn new(execution: Execution<'a>, context: usize,
        host: Option<cuteafd_hostcache::config::Config>) -> Result<Self> {
        let library = execution.requests.decode.stream.device.library;
        let cache = execution.requests.cache.clone();
        ensure!(context > 0 && context <= execution.cfg.max_position && context <= cache.layout.pool_pages*64,
            "Inkling served context outside admitted pool/checkpoint");
        let concurrency = cache.layout.sequences;
        let unit_rows = execution.requests.prefill[0].layout.rows;
        let waves = execution.requests.prefill.len();
        let vocab = execution.cfg.unpadded_vocab_size;
        Ok(Self { execution: RefCell::new(execution), family: InklingPrefix::new(cache), library,
            context, concurrency, unit_rows, waves, vocab, host, profile: RefCell::new([0.; 2]),
            pending: RefCell::new(None) })
    }
    fn check(&self, placement: &Placement, tokens: &[u32], pool: &RefPagePool) -> Result<()> {
        ensure!(!tokens.is_empty() && tokens.iter().all(|&token| (token as usize) < self.vocab)
            && placement.len == placement.state_len && placement.slot < self.concurrency,
            "Inkling append token/frontier/slot differs");
        let end = placement.len.checked_add(tokens.len()).context("Inkling append extent overflow")?;
        ensure!(end <= self.context && end.div_ceil(64) <= placement.units.len(), "Inkling append exceeds reserved pages");
        let mut writable = BTreeSet::new();
        for position in placement.len..end {
            let page = placement.units[position/64];
            ensure!(pool.refs(page) == 1 && writable.insert((page, position%64)),
                "Inkling append requires private nonaliased COW pages");
        }
        Ok(())
    }
    fn append(&self, placement: &mut Placement, tokens: &[u32], heads: bool, pool: &RefPagePool,
        callback: &mut dyn FnMut(usize, usize, DeviceLogits) -> Result<()>) -> Result<()> {
        self.check(placement, tokens, pool)?;
        let start = placement.len;
        let layers = self.execution.borrow().cfg.layers;
        let mut units = sequence_units(tokens.len(), self.unit_rows, self.waves, layers)?;
        for unit in &mut units {
            unit.rows = unit.rows.start+start..unit.rows.end+start;
            unit.segments = vec![UnitSegment { sequence: 0, rows: 0..unit.rows.len(), tokens: unit.rows.clone(),
                prompt_end: unit.rows.end == start+tokens.len(), snapshot: false, head: heads }];
        }
        let mut next = [placement.clone()];
        self.execution.borrow_mut().prefill(&mut next, &[tokens], &units, 1, callback)?;
        *placement = next.into_iter().next().unwrap();
        Ok(())
    }
    fn decode(&self, sequences: &mut [(&mut Placement, &[u32])], speculative: bool,
        pool: &RefPagePool) -> Result<Option<DeviceLogits>> {
        ensure!(!speculative && !sequences.is_empty() && sequences.len() <= self.concurrency
            && sequences.iter().all(|(_, tokens)| tokens.len() == 1), "Inkling initial decode supports one target row per request");
        let mut slots = BTreeSet::new();
        let mut members = Vec::new();
        let mut placements = Vec::new();
        let mut sources = Vec::new();
        let mut receipt = Vec::new();
        for (index, (placement, tokens)) in sequences.iter().enumerate() {
            self.check(placement, tokens, pool)?;
            ensure!(slots.insert(placement.slot), "Inkling decode aliases a request state slot");
            members.push(UnitSegment { sequence: index, rows: index..index+1, tokens: placement.len..placement.len+1,
                prompt_end: false, snapshot: false, head: true });
            receipt.push((placement.slot, placement.len, tokens.to_vec()));
            placements.push((**placement).clone()); sources.push(*tokens);
        }
        let layers = self.execution.borrow().cfg.layers;
        let unit = Unit { index: 0, sequence: 0, segments: members, rows: 0..sequences.len(),
            layers: 0..layers, slot: 0, barrier: false };
        let logits = self.execution.borrow_mut().decode(&mut placements, &sources, &unit, 1)?;
        for ((placement, _), next) in sequences.iter_mut().zip(placements) { **placement = next; }
        *self.pending.borrow_mut() = Some(receipt);
        Ok(Some(logits))
    }
}

impl<'a> Backend<'a> for InklingBackend<'a> {
    type Placement = Placement;
    type Draft = Draft;
    type History = usize;
    type Coordinates = ();
    type Encoder = crate::shared::qwen::Encoder;
    type Meta = ();
    type Prefix = InklingPrefix<'a>;
    fn geometry(&self) -> scheduler::Geometry {
        scheduler::Geometry { family: "inkling2", vocab_size: self.vocab, max_context: self.context,
            slots: self.concurrency, prefill_rows: self.unit_rows*self.waves, decode_rows: 64 }
    }
    fn message_starts(&self) -> &[&str] { &["<|message_system|>", "<|message_user|>", "<|message_model|>", "<|message_tool|>"] }
    fn library(&self) -> &'a cuteafd_ffi::NativeLibrary { self.library }
    fn profile(&self) -> &RefCell<[f64; 2]> { &self.profile }
    fn prefix_cache(&self, args: &PrefixArgs, concurrency: usize)
        -> Result<(Self::Prefix, PrefixCache<CudaCopyEngine<'a>>)> {
        ensure!(concurrency == self.concurrency && args.prefix_partial == crate::shared::prefix::Toggle::Off,
            "Inkling prefix geometry requires exact admitted frontiers");
        let bank = self.execution.borrow().requests.cache.clone();
        let host = self.host.clone().map(|config| {
            let retained: std::rc::Rc<dyn crate::shared::prefix::cuda_copy::SnapshotOwner + 'a> = bank.clone();
            CudaCopyEngine::registered_retained(self.library, vec![], vec![retained]).map(|engine| (config, engine))
        }).transpose()?;
        let cache = PrefixCache::new(self.family.layout(), PrefixConfig { entries: args.prefix_cache_entries,
            mark_slots: bank.mark_slots(), keep_logits: true, min_tokens: args.prefix_cache_min_tokens }, host)?;
        Ok((self.family.clone(), cache))
    }
    fn prepare(&self, job: NativeRequest, tokens: Vec<u32>) -> Result<(scheduler::Prompt, RequestMedia, Vec<EncodeJob>)> {
        ensure!(job.media.is_empty(), "Inkling initial text executor does not load its bundled vision tower");
        let keys = MediaKeys::new(&tokens, u32::try_from(self.vocab)?, &[])?;
        let media = RequestMedia::new(vec![], 4608, tokens.len())?;
        Ok((scheduler::Prompt { job, tokens, keys }, media, vec![]))
    }
    fn probe_features(&self, _: &scheduler::Prompt, media: &mut RequestMedia, _: &mut EmbeddingCache, _: &Path) -> Result<()> {
        ensure!(media.spans().is_empty(), "Inkling media executor absent"); Ok(())
    }
    fn coordinates(&self, _: &[u32], job: &NativeRequest, media: &RequestMedia) -> Result<()> {
        ensure!(job.media.is_empty() && media.spans().is_empty(), "Inkling media executor absent"); Ok(())
    }
    fn placement(&self, units: Vec<u32>, slot: i32) -> Placement { Placement { units, slot: slot as usize, len: 0, state_len: 0 } }
    fn bind_request(&self, placement: &mut Placement, _: (), _: RequestMedia, tokens: &[u32], resume: usize) -> Result<()> {
        ensure!(placement.len == resume && placement.state_len == resume && resume <= tokens.len(),
            "Inkling restored request frontier differs"); Ok(())
    }
    fn history(&self, placement: &Placement) -> usize { placement.slot }
    fn require_live(&self) -> Result<()> { Ok(()) }
    fn log_table_stats(&self, _: &str) {}
    fn prefill_device(&self, placement: &mut Placement, tokens: &[u32], rows: usize, pool: &RefPagePool) -> Result<Option<DeviceLogits>> {
        ensure!(rows <= 1, "Inkling prefill head retains one row; scoring streams tiles");
        let mut last = None;
        self.append(placement, tokens, rows > 0, pool, &mut |_, _, logits| {
            let offset = (logits.rows-1)*logits.stride*4;
            last = Some(DeviceLogits { ptr: logits.ptr.wrapping_byte_add(offset), rows: 1, ..logits }); Ok(())
        })?;
        Ok(last)
    }
    fn streamed_prefill_scoring(&self) -> bool { true }
    fn prefill_scoring(&self, placement: &mut Placement, tokens: &[u32], rows: usize, position: usize,
        probe: &probe::ProbeRef, pool: &RefPagePool) -> Result<Option<probe::ScoreLogits>> {
        ensure!(rows <= tokens.len(), "Inkling score suffix exceeds submitted rows");
        let first = placement.len+tokens.len()-rows+1;
        let scoring = probe::ScoringRows { consumer: probe, position };
        self.append(placement, tokens, rows > 0, pool, &mut |_, at, logits| {
            if at >= first { scoring.record(self.library, at-first, &logits)?; }
            Ok(())
        })?;
        Ok((rows > 0).then_some(probe::ScoreLogits::Recorded))
    }
    fn packed_limits(&self) -> Option<Limits> { Some(Limits { rows: self.unit_rows, chunk_rows: self.unit_rows, max_context: self.context, taps: None }) }
    fn prefill_packed(&self, sequences: &mut [scheduler::PrefillSeq<'_, Placement, Draft>], pool: &RefPagePool,
        on_logits: &mut dyn FnMut(usize, DeviceLogits) -> MemberOutcome) -> Result<Vec<Result<()>>> {
        ensure!(!sequences.is_empty(), "Inkling packed submission is empty");
        let limits = self.packed_limits().unwrap();
        let chunks = sequences.iter().map(|sequence| {
            ensure!(!sequence.neural && sequence.start == sequence.placement.len, "Inkling packed start/drafter differs");
            self.check(sequence.placement, sequence.tokens, pool)?;
            Ok((sequence.start, sequence.tokens.len()))
        }).collect::<Result<Vec<_>>>()?;
        let segments = crate::shared::prefill_packing::plan(&chunks, &limits)?;
        let rows = segments.last().context("Inkling pack has no members")?.end_row();
        let mut placements = sequences.iter().map(|sequence| sequence.placement.clone()).collect::<Vec<_>>();
        let sources = sequences.iter().map(|sequence| sequence.tokens).collect::<Vec<_>>();
        let layers = self.execution.borrow().cfg.layers;
        let unit = Unit { index: 0, sequence: 0, rows: 0..rows, layers: 0..layers, slot: 0, barrier: false,
            segments: segments.iter().enumerate().map(|(sequence, segment)| UnitSegment { sequence,
                rows: segment.first..segment.end_row(), tokens: segment.start..segment.start+segment.rows,
                prompt_end: true, snapshot: false, head: true }).collect() };
        let mut last = vec![None; sequences.len()];
        self.execution.borrow_mut().prefill(&mut placements, &sources, &[unit], 1, &mut |member, position, logits| {
            if position+logits.rows == chunks[member].0+chunks[member].1+1 {
                let offset = (logits.rows-1)*logits.stride*4;
                let logits = DeviceLogits { ptr: logits.ptr.wrapping_byte_add(offset), rows: 1, ..logits };
                last[member] = Some(on_logits(member, logits)?);
            }
            Ok(())
        })?;
        for (sequence, placement) in sequences.iter_mut().zip(placements) { *sequence.placement = placement; }
        last.into_iter().map(|outcome| outcome.context("Inkling packed member last-row callback missing")).collect()
    }
    fn verify_device(&self, sequences: &mut [(&mut Placement, &[u32])], speculative: bool, pool: &RefPagePool) -> Result<Option<DeviceLogits>> {
        self.decode(sequences, speculative, pool)
    }
    fn verify_device_ungraphed(&self, sequences: &mut [(&mut Placement, &[u32])], speculative: bool,
        _: scheduler::UngraphedKind, pool: &RefPagePool) -> Result<Option<DeviceLogits>> { self.decode(sequences, speculative, pool) }
    fn verify_bucket_rows(&self, rows: usize, _: bool, _: bool) -> usize { rows }
    fn verify_attention_trace(&self, _: &[Vec<usize>], rows: usize) -> crate::shared::speculation_trace::Deferred {
        crate::shared::speculation_trace::Deferred::new(move || serde_json::json!({"kind":"inkling2", "rows":rows, "attention":"eager"}))
    }
    fn draft_geometry(&self, _: &Path, _: usize) -> Result<cuteafd_core::PolicyGeometry> { anyhow::bail!("Inkling bundled drafter is not yet qualified") }
    fn arm_draft_probe(&self) -> Result<()> { Ok(()) }
    fn prefill_draft(&self, _: &Placement, _: usize, _: &[u32], _: Option<u32>, _: &mut Draft) -> Result<()> { Ok(()) }
    fn draft(&self, sequences: &mut [scheduler::DraftSeq<'_, Placement, Draft>], _: &mut scheduler::DraftTiming,
        _: &mut TokenSelector<'_>) -> Result<Vec<Vec<u32>>> { Ok(vec![vec![]; sequences.len()]) }
    fn accept(&self, verified: &mut [scheduler::Verified<'_, Placement, Draft, usize>], speculative: bool) -> Result<()> {
        ensure!(!speculative, "Inkling bundled drafter is not yet qualified");
        let pending = self.pending.borrow_mut().take().context("Inkling decode acceptance has no submitted receipt")?;
        ensure!(pending.len() == verified.len(), "Inkling accepted request count differs");
        for ((slot, start, tokens), request) in pending.iter().zip(verified) {
            ensure!(*slot == request.history && *slot == request.placement.slot && *start == request.start
                && tokens == request.rows && request.placement.len == start+tokens.len()
                && request.placement.state_len == request.placement.len, "Inkling accepted causal receipt differs");
        }
        Ok(())
    }
    fn cancel(&self, _: &Placement) -> Result<()> { self.execution.borrow().requests.cache.drain()?; self.pending.borrow_mut().take(); Ok(()) }
    fn probe_begin(&self, _: usize) {}
    fn probe_end(&self) {}
    fn probe_finish<T>(&self, _: impl FnOnce(&RoundRoutes, &[Option<f64>]) -> T) -> Option<T> { None }
}
