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

fn check_writable(placement: &Placement, rows: usize, context: usize, pool: &RefPagePool) -> Result<()> {
    let end = placement.len.checked_add(rows).context("Inkling append extent overflow")?;
    ensure!(rows > 0 && end <= context && end.div_ceil(64) <= placement.units.len(), "Inkling append exceeds reserved pages");
    let mut writable = BTreeSet::new();
    for position in placement.len..end {
        let page = placement.units[position/64];
        ensure!((page as usize) < pool.capacity() && pool.refs(page) == 1 && writable.insert((page, position%64)),
            "Inkling append requires private nonaliased COW pages");
    }
    Ok(())
}

fn decode_spans(widths: impl IntoIterator<Item = usize>, capacity: usize) -> Result<Vec<std::ops::Range<usize>>> {
    let mut rows = 0usize;
    widths.into_iter().map(|width| {
        ensure!(width > 0, "Inkling decode member has no rows");
        let end = rows.checked_add(width).context("Inkling decode row extent overflow")?;
        ensure!(end <= capacity, "Inkling decode exceeds admitted {capacity}-row owner");
        let span = rows..end;
        rows = end;
        Ok(span)
    }).collect()
}

#[derive(Default)]
pub(crate) struct Draft {anchor:Option<u32>}
impl scheduler::DraftState for Draft {
    fn pending_rows(&self) -> usize { usize::from(self.anchor.is_some()) }
    fn close(&mut self, token: u32) {self.anchor=Some(token);}
}

enum Pending<'a>{
    Plain(Vec<(usize,usize,Vec<u32>)>),
    Spec{mark:super::verify::TargetMark<'a>,tokens:Vec<Vec<u32>>},
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
    pending: RefCell<Option<Pending<'a>>>,
    selector: Vec<cuteafd_loader::placement::planner::Allocation>,
    selector_ready: std::cell::Cell<bool>,
}
impl<'a> InklingBackend<'a> {
    pub fn new(execution: Execution<'a>, context: usize,
        host: Option<cuteafd_hostcache::config::Config>, plan: &cuteafd_loader::placement::planner::ResolvedPlan) -> Result<Self> {
        let library = execution.requests.decode.stream.device.library;
        let cache = execution.requests.cache.clone();
        ensure!(context > 0 && context <= execution.cfg.max_position && context <= cache.layout.pool_pages*64,
            "Inkling served context outside admitted pool/checkpoint");
        let concurrency = cache.layout.sequences;
        let unit_rows = execution.requests.prefill[0].layout.rows;
        let waves = execution.requests.prefill.len();
        let vocab = execution.cfg.unpadded_vocab_size;
        let selector = plan.facts.allocations.iter().filter(|owner| owner.id.starts_with("selector.")).cloned().collect::<Vec<_>>();
        ensure!(!selector.is_empty(), "Inkling selector was not admitted before startup");
        let family=InklingPrefix::with_draft(cache,execution.draft.clone());
        Ok(Self { execution: RefCell::new(execution), family, library,
            context, concurrency, unit_rows, waves, vocab, host, profile: RefCell::new([0.; 2]),
            pending: RefCell::new(None), selector, selector_ready: std::cell::Cell::new(false) })
    }
    fn check(&self, placement: &Placement, tokens: &[u32], pool: &RefPagePool) -> Result<()> {
        ensure!(!tokens.is_empty() && tokens.iter().all(|&token| (token as usize) < self.vocab)
            && placement.len == placement.state_len && placement.slot < self.concurrency,
            "Inkling append token/frontier/slot differs");
        check_writable(placement, tokens.len(), self.context, pool)

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
        ensure!(!sequences.is_empty() && sequences.len() <= self.concurrency && self.pending.borrow().is_none(),
            "Inkling target decode requires admitted requests and prior commit");
        let spans = decode_spans(sequences.iter().map(|(_, tokens)| tokens.len()),
            self.execution.borrow().requests.decode.layout.rows)?;
        let rows = spans.last().context("Inkling decode has no rows")?.end;
        let mut slots = BTreeSet::new();
        let mut members = Vec::new();
        let mut placements = Vec::new();
        let mut sources = Vec::new();
        let mut receipt = Vec::new();
        for (index, (placement, tokens)) in sequences.iter().enumerate() {
            self.check(placement, tokens, pool)?;
            let cfg = &self.execution.borrow().cfg;
            let ring = cuteafd_loader::families::inkling2::execution::SequenceLayout::new(cfg, true, 1)?.ring_rows;
            debug_assert!(tokens.len() <= ring-cfg.window, "verify append exceeds ring dead horizon");
            cuteafd_loader::families::inkling2::verify::verify_ring(cfg.window, ring, tokens.len())?;
            ensure!(slots.insert(placement.slot), "Inkling decode aliases a request state slot");
            members.push(UnitSegment { sequence: index, rows: spans[index].clone(),
                tokens: placement.len..placement.len+tokens.len(),
                prompt_end: false, snapshot: false, head: true });
            receipt.push((placement.slot, placement.len, tokens.to_vec()));
            placements.push((**placement).clone()); sources.push(*tokens);
        }
        let layers = self.execution.borrow().cfg.layers;
        let unit = Unit { index: 0, sequence: 0, segments: members, rows: 0..rows,
            layers: 0..layers, slot: 0, barrier: false };
        let (logits,pending)=if speculative{
            use crate::shared::speculation::target::{BlockVerifyTarget,VerifyRequest,VerifyRow};
            let mut execution=self.execution.borrow_mut();
            let taps=execution.draft.as_ref().context("speculative verify without admitted drafter")?.layout.config.taps.clone();
            let ids=placements.iter().map(|p|p.slot).collect::<Vec<_>>();
            let rows=receipt.iter().flat_map(|(slot,start,tokens)|tokens.iter().enumerate().map(move |(offset,&token)|
                VerifyRow{slot:*slot,position:start+offset,token:Some(token)})).collect::<Vec<_>>();
            let mut target=super::verify_target::Target::new(&mut execution,&mut placements);
            let mark=target.checkpoint(&ids)?;
            let logits=target.verify(&mark,VerifyRequest{rows:&rows,tap_sites:&taps})?.logits;
            (logits,Pending::Spec{mark,tokens:receipt.iter().map(|(_,_,tokens)|tokens.clone()).collect()})
        }else{
            (self.execution.borrow_mut().decode(&mut placements,&sources,&unit,1)?,Pending::Plain(receipt))
        };
        for ((placement, _), next) in sequences.iter_mut().zip(placements) { **placement = next; }
        *self.pending.borrow_mut() = Some(pending);
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
    fn selector_ready(&self, selector: &mut TokenSelector<'a>) -> Result<()> {
        use cuteafd_loader::placement::planner::{CoordinatorDevices, Location};
        ensure!(!self.selector_ready.get(), "Inkling selector initialized twice");
        selector.reserve_sampler()?;
        let physical = [self.execution.borrow().requests.decode.stream.device.id];
        let mut inventory = crate::shared::memory::receipts::Inventory::new(CoordinatorDevices::new(&physical)?, "selector",
            cuteafd_loader::placement::planner::serving::ServingOwners::LIFETIME,
            cuteafd_core::memory_layout::Category::Workspace, "Inkling selector");
        selector.allocation_receipts(&mut inventory, "next")?;
        for receipt in &mut inventory.receipts {
            receipt.allocation.id = receipt.allocation.id.replacen("selector.next.", "selector.", 1);
        }
        let actual = inventory.receipts.iter().filter(|receipt| receipt.allocation.location != Location::HOST)
            .map(|receipt| (receipt.allocation.id.as_str(), receipt)).collect::<std::collections::BTreeMap<_, _>>();
        ensure!(actual.len() == self.selector.len(), "Inkling selector owner count differs from admission");
        for planned in &self.selector {
            let id = match planned.id.strip_prefix("selector.sampler.") {
                Some(id) => format!("selector.sampler.{}", cuteafd_core::target_sampler_receipt_key(id)
                    .with_context(|| format!("Inkling sampler owner has no shared receipt key {id}"))?),
                None => planned.id.clone(),
            };
            let receipt = actual.get(id.as_str()).with_context(|| format!("Inkling missing selector owner {id}"))?;
            ensure!(receipt.allocation.bytes == planned.bytes && receipt.allocation.location == planned.location
                && receipt.allocation.lifetime == planned.lifetime && receipt.address != 0,
                "Inkling selector physical request differs from admission {id}");
        }
        self.selector_ready.set(true);
        Ok(())
    }
    fn prefix_cache(&self, args: &PrefixArgs, concurrency: usize)
        -> Result<(Self::Prefix, PrefixCache<CudaCopyEngine<'a>>)> {
        ensure!(concurrency == self.concurrency && args.prefix_partial == crate::shared::prefix::Toggle::Off,
            "Inkling prefix geometry requires exact admitted frontiers");
        let bank = self.execution.borrow().requests.cache.clone();
        let host = self.host.clone().map(|config| {
            let retained: std::rc::Rc<dyn crate::shared::prefix::cuda_copy::SnapshotOwner + 'a> = bank.clone();
            CudaCopyEngine::registered_retained(self.library, vec![], vec![retained]).map(|engine| (config, engine))
        }).transpose()?;
        let host_bytes = host.as_ref().map_or(0, |(config, _)| config.bytes);
        let layout = self.family.layout();
        let cache = PrefixCache::new(layout, PrefixConfig { entries: args.prefix_cache_entries,
            mark_slots: bank.mark_slots(), keep_logits: true, min_tokens: args.prefix_cache_min_tokens }, host)?;
        cuteafd_bench::context::set_kv((layout.pages * layout.page_rows) as u64,
            Some((layout.pages * layout.page_rows) as u64), layout.pages as u64,
            self.family.record_format(), host_bytes);
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
    fn require_live(&self) -> Result<()> { self.execution.borrow().require_live() }
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
            let end = at.checked_add(logits.rows).context("Inkling score tile overflow")?;
            if end > first {
                let skip = first.saturating_sub(at);
                let selected = DeviceLogits { ptr: logits.ptr.wrapping_byte_add(skip*logits.stride*4),
                    rows: logits.rows-skip, ..logits };
                scoring.record(self.library, at+skip-first, &selected)?;
            }
            Ok(())
        })?;
        Ok((rows > 0).then_some(probe::ScoreLogits::Recorded))
    }
    fn packed_limits(&self) -> Option<Limits> { Some(Limits { rows: self.unit_rows, chunk_rows: self.unit_rows, max_context: self.context, taps: None }) }
    fn prefill_packed(&self, sequences: &mut [scheduler::PrefillSeq<'_, Placement, Draft>], pool: &RefPagePool,
        on_logits: &mut dyn FnMut(usize, DeviceLogits) -> MemberOutcome) -> Result<Vec<Result<()>>> {
        ensure!(!sequences.is_empty(), "Inkling packed submission is empty");
        let limits = self.packed_limits().unwrap();
        ensure!(sequences.iter().map(|sequence| sequence.placement.slot).collect::<BTreeSet<_>>().len() == sequences.len(),
            "Inkling packed members alias a request state slot");
        let chunks = sequences.iter().map(|sequence| {
            ensure!(sequence.start == sequence.placement.len, "Inkling packed start differs");
            ensure!(!sequence.neural || self.execution.borrow().draft.is_some(), "Inkling packed neural prefill needs admitted drafter");
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
        let mut last = (0..sequences.len()).map(|_| None).collect::<Vec<_>>();
        self.execution.borrow_mut().prefill(&mut placements, &sources, &[unit], 1, &mut |member, position, logits| {
            if position+logits.rows == chunks[member].0+chunks[member].1+1 {
                let offset = (logits.rows-1)*logits.stride*4;
                let logits = DeviceLogits { ptr: logits.ptr.wrapping_byte_add(offset), rows: 1, ..logits };
                last[member] = Some(on_logits(member, logits)?);
            }
            Ok(())
        })?;
        for (sequence, placement) in sequences.iter_mut().zip(placements) {
            *sequence.placement = placement;
            if sequence.neural {
                self.prefill_draft(sequence.placement, sequence.start, sequence.tokens, sequence.next, sequence.draft)?;
            }
        }
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
    fn draft_geometry(&self,_:&Path,width:usize)->Result<cuteafd_core::PolicyGeometry>{
        ensure!(self.execution.borrow().draft.is_some(),"Inkling drafter not admitted");
        Ok(cuteafd_core::PolicyGeometry{layers:vec![cuteafd_core::LayerResource{class:None,slice_bytes:0.,group_rows:1,timed:false};48],
            group_rows_by_rows:vec![],experts:0,topk:0,max_requests:self.concurrency,max_positions:64,
            widths:vec![width],classes:vec![cuteafd_core::ResourceClass::rtx()],regimes:1})
    }
    fn arm_draft_probe(&self)->Result<()>{Ok(())}
    fn prefill_draft(&self,p:&Placement,_:usize,_:&[u32],next:Option<u32>,state:&mut Draft)->Result<()>{
        if let Some(draft)=&self.execution.borrow().draft{ensure!(draft.drafter.frontier(p.slot)?.1==p.len,"Inkling draft prefill frontier differs");}
        state.anchor=next;Ok(())
    }
    fn draft(&self,sequences:&mut [scheduler::DraftSeq<'_,Placement,Draft>],timing:&mut scheduler::DraftTiming,
        _:&mut TokenSelector<'_>)->Result<Vec<Vec<u32>>>{
        let execution=self.execution.borrow();let Some(draft)=&execution.draft else{return Ok(vec![vec![];sequences.len()]);};
        let start=std::time::Instant::now();let mut output=vec![vec![];sequences.len()];
        for first in (0..sequences.len()).step_by(draft.layout.runtime[0].sequences){
            let end=(first+draft.layout.runtime[0].sequences).min(sequences.len());
            let selected=sequences[first..end].iter().enumerate().filter_map(|(index,s)|s.seq.anchor.map(|anchor|(index,anchor,s))).collect::<Vec<_>>();
            if selected.is_empty(){continue;}
            let rows=selected.iter().map(|(_,anchor,s)|crate::shared::speculation::dflash::DraftSeq{
                slot:s.placement.slot,anchor:*anchor,position:s.placement.len,valid_from:0}).collect::<Vec<_>>();
            let proposed=draft.drafter.draft(&rows,execution.target.operand("model.llm.embed.weight")?.buffer,
                execution.target.operand("model.llm.embed_norm.weight")?.buffer,execution.target.operand("model.llm.unembed.weight")?.buffer)?;
            for ((index,_,s),tokens) in selected.into_iter().zip(proposed){output[first+index]=tokens.tokens.into_iter().take(s.depth).collect();}
        }
        timing.steps+=1;timing.seconds+=start.elapsed().as_secs_f64();Ok(output)
    }
    fn accept(&self, verified: &mut [scheduler::Verified<'_, Placement, Draft, usize>], speculative: bool) -> Result<()> {
        let pending=self.pending.borrow_mut().take().context("Inkling accept lacks submitted receipt")?;
        let mut execution=self.execution.borrow_mut();let draft=execution.draft.clone();
        match pending{
            Pending::Plain(sent)=>{
                ensure!(!speculative && sent.len()==verified.len(),"Inkling plain accept inventory differs");
                for ((slot,start,tokens),request) in sent.iter().zip(verified.iter()){
                    ensure!(*slot==request.history && *slot==request.placement.slot && *start==request.start
                        && tokens==request.rows,"Inkling plain accept receipt differs");
                }
            }
            Pending::Spec{mark,tokens}=>{
                use crate::shared::speculation::target::{AcceptedPrefix,BlockVerifyTarget};
                ensure!(speculative && tokens.len()==verified.len() && tokens.iter().zip(verified.iter()).all(|(tokens,s)|tokens==s.rows),
                    "Inkling speculative accept receipt differs");
                let accepted=verified.iter().map(|s|AcceptedPrefix{slot:s.placement.slot,rows:s.kept.map_or(0,|(rows,_)|rows)}).collect::<Vec<_>>();
                let mut placements=verified.iter().map(|s|s.placement.clone()).collect::<Vec<_>>();
                super::verify_target::Target::new(&mut execution,&mut placements).commit(&mark,&accepted)?;
                for (s,p) in verified.iter_mut().zip(placements){*s.placement=p;}
            }
        }
        if let Some(draft)=draft{
            for s in verified.iter_mut(){
                let keep=if speculative{s.kept.map_or(0,|(rows,_)|rows)}else{s.rows.len()};
                if keep>0{draft.update(&draft.decode,s.placement.slot,s.start,s.first_row,keep)?;}
                if let Some((_,anchor))=s.kept{s.seq.anchor=Some(anchor);}
            }
        }
        Ok(())
    }
    fn cancel(&self,_:&Placement)->Result<()>{
        let execution=self.execution.borrow();
        if let Some(Pending::Spec{mark,..})=self.pending.borrow_mut().take(){
            execution.draft.as_ref().context("cancelled draft missing")?.verify.restore(&mark.conv,execution.requests.cache.conv_backing())?;
        }
        execution.requests.cache.drain()?;Ok(())
    }
    fn probe_begin(&self, _: usize) {}
    fn probe_end(&self) {}
    fn probe_finish<T>(&self, _: impl FnOnce(&RoundRoutes, &[Option<f64>]) -> T) -> Option<T> { None }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_neural_prefill_updates_context_and_anchor_before_returning() {
        let source = include_str!("backend.rs").split("#[cfg(test)]").next().unwrap();
        let packed = source.split("fn prefill_packed(").nth(1).unwrap().split("fn verify_device(").next().unwrap();
        assert!(!packed.contains("!sequence.neural &&"));
        assert!(packed.contains("self.prefill_draft(sequence.placement, sequence.start, sequence.tokens, sequence.next, sequence.draft)?"));
        assert!(packed.find("*sequence.placement = placement;").unwrap() < packed.find("self.prefill_draft(").unwrap());
    }

    #[test]
    fn decode_members_share_admitted_rows_without_overlapping_logits() -> Result<()> {
        assert_eq!(decode_spans([1, 7, 16, 40], 64)?, vec![0..1, 1..8, 8..24, 24..64]);
        assert_eq!(decode_spans([64], 64)?, vec![0..64]);
        assert!(decode_spans([1, 64], 64).is_err());
        assert!(decode_spans([0], 64).is_err());
        assert!(decode_spans([usize::MAX, 1], usize::MAX).is_err());
        Ok(())
    }
    #[test]
    fn append_writes_only_reserved_private_nonaliased_pages() -> Result<()> {
        let mut pool = RefPagePool::new(4, 64);
        let units = pool.alloc_for(0..2)?;
        let placement = Placement { units: units.clone(), slot: 0, len: 63, state_len: 63 };
        check_writable(&placement, 3, 128, &pool)?;
        pool.share(&units[..1]);
        assert!(check_writable(&placement, 3, 128, &pool).is_err());
        pool.release(&units[..1]);
        let alias = Placement { units: vec![units[0], units[0]], len: 0, state_len: 0, ..placement.clone() };
        assert!(check_writable(&alias, 128, 128, &pool).is_err());
        assert!(check_writable(&placement, 66, 128, &pool).is_err());
        assert!(check_writable(&placement, 0, 128, &pool).is_err());
        let invalid = Placement { units: vec![99], len: 0, state_len: 0, ..placement };
        assert!(check_writable(&invalid, 1, 128, &pool).is_err());
        Ok(())
    }
    #[test]
    fn adapter_uses_shared_scheduler_units_prefix_and_streamed_scoring() {
        let source = include_str!("backend.rs").split("#[cfg(test)]").next().unwrap();
        assert!(source.contains("Backend<'a> for InklingBackend"));
        assert!(source.contains("PrefixCache::new("));
        assert!(source.contains("registered_retained("));
        assert!(source.contains("prefill_packing::plan("));
        assert!(source.contains("streamed_prefill_scoring(&self) -> bool { true }"));
        assert!(!source.contains("tokio::select!"));
        assert!(!source.contains("loop {"));
        assert!(!source.contains("for layer in"));
    }
}
