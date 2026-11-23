//! Hy4-flash adapter for the shared request scheduler (`shared::qwen::scheduler::Backend`).
use super::cache::{Hy4Placement, Hy4Prefix};
use super::execution::{HeadMode, Hy4Engine};
use crate::shared::{
    prefix::{CudaCopyEngine, PrefixArgs},
    prefill_packing::{Limits, MemberOutcome},
    prefill_pipeline::{sequence_units, Unit, UnitSegment},
    probe,
    qwen::scheduler::{self, Backend},
    token_io::{DeviceLogits, TokenSelector},
};
use anyhow::{ensure, Context, Result};
use cuteafd_api::openai::NativeRequest;
use cuteafd_engine::{
    media::{EncodeJob, EmbeddingCache, RequestMedia},
    prefix::{PrefixCache, PrefixConfig, PrefixFamily, RefPagePool},
};
use cuteafd_loader::families::hy4_flash::execution::PAGE_ROWS;
use cuteafd_loader::formats::mapped_table::ReadPriority;
use std::cell::RefCell;
use std::collections::BTreeSet;
use std::path::Path;

fn check_writable(placement: &Hy4Placement, rows: usize, context: usize, pool: &RefPagePool) -> Result<()> {
    let end = placement.len.checked_add(rows).context("Hy4 append extent overflow")?;
    ensure!(
        rows > 0 && end <= context && end.div_ceil(PAGE_ROWS) <= placement.units.len(),
        "Hy4 append exceeds reserved pages"
    );
    let mut writable = BTreeSet::new();
    for position in placement.len..end {
        let page = placement.units[position / PAGE_ROWS];
        ensure!(
            (page as usize) < pool.capacity()
                && pool.refs(page) == 1
                && writable.insert((page, position % PAGE_ROWS)),
            "Hy4 append requires private nonaliased COW pages"
        );
    }
    Ok(())
}

fn decode_spans(widths: impl IntoIterator<Item = usize>, capacity: usize) -> Result<Vec<std::ops::Range<usize>>> {
    let mut rows = 0usize;
    widths
        .into_iter()
        .map(|width| {
            ensure!(width > 0, "Hy4 decode member has no rows");
            let end = rows.checked_add(width).context("Hy4 decode row extent overflow")?;
            ensure!(end <= capacity, "Hy4 decode exceeds admitted {capacity}-row owner");
            let span = rows..end;
            rows = end;
            Ok(span)
        })
        .collect()
}

#[derive(Default)]
pub(crate) struct Draft {
    anchor: Option<u32>,
}

impl scheduler::DraftState for Draft {
    fn pending_rows(&self) -> usize {
        usize::from(self.anchor.is_some())
    }
    fn close(&mut self, token: u32) {
        self.anchor = Some(token);
    }
}

fn scoring_replay(
    sequences: &mut [(&mut Hy4Placement, &[u32])],
    speculative: bool,
    decode: impl FnOnce(&mut [(&mut Hy4Placement, &[u32])]) -> Result<Option<DeviceLogits>>,
    accept: impl FnOnce(&mut [scheduler::Verified<'_, Hy4Placement, Draft, usize>]) -> Result<()>,
) -> Result<Option<DeviceLogits>> {
    ensure!(!speculative, "Hy4 scoring replay cannot speculate");
    let logits = decode(sequences)?;
    let mut drafts = (0..sequences.len()).map(|_| Draft::default()).collect::<Vec<_>>();
    let mut verified = sequences
        .iter_mut()
        .zip(&mut drafts)
        .scan(0, |first_row, ((placement, rows), seq)| {
            let verified = scheduler::Verified {
                start: placement.len - rows.len(),
                history: placement.slot,
                placement: &mut **placement,
                seq,
                rows: *rows,
                first_row: *first_row,
                kept: None,
                finished: false,
                neural: false,
            };
            *first_row += rows.len();
            Some(verified)
        })
        .collect::<Vec<_>>();
    accept(&mut verified)?;
    Ok(logits)
}

pub(crate) struct Hy4Backend<'a> {
    engine: RefCell<Hy4Engine<'a>>,
    family: Hy4Prefix<'a>,
    library: &'a cuteafd_ffi::NativeLibrary,
    context: usize,
    concurrency: usize,
    unit_rows: usize,
    vocab: usize,
    pad_id: u32,
    host: Option<cuteafd_hostcache::config::Config>,
    profile: RefCell<[f64; 2]>,
    pending: RefCell<Option<Vec<(usize, usize, Vec<u32>)>>>,
}

impl<'a> Hy4Backend<'a> {
    pub(crate) fn new(
        engine: Hy4Engine<'a>,
        context: usize,
        unit_rows: usize,
        host: Option<cuteafd_hostcache::config::Config>,
    ) -> Result<Self> {
        let library = engine.library();
        let family = engine.prefix()?;
        let concurrency = engine.max_sequences;
        let vocab = engine.cfg.vocab_size;
        let pad_id = engine.cfg.engram_pad_id;
        ensure!(
            context > 0 && context <= engine.cfg.max_position && context <= engine.max_pages * PAGE_ROWS,
            "Hy4 served context {context} outside admitted pool/checkpoint"
        );
        ensure!(unit_rows > 0 && unit_rows <= engine.max_rows, "Hy4 unit_rows {unit_rows} outside 1..={}", engine.max_rows);
        Ok(Self {
            engine: RefCell::new(engine),
            family,
            library,
            context,
            concurrency,
            unit_rows,
            vocab,
            pad_id,
            host,
            profile: RefCell::new([0.0; 2]),
            pending: RefCell::new(None),
        })
    }

    fn check(&self, placement: &Hy4Placement, tokens: &[u32], pool: &RefPagePool) -> Result<()> {
        ensure!(
            !tokens.is_empty()
                && tokens.iter().all(|&token| (token as usize) < self.vocab)
                && placement.len == placement.state_len
                && placement.slot < self.concurrency,
            "Hy4 append token/frontier/slot differs"
        );
        check_writable(placement, tokens.len(), self.context, pool)
    }

    fn decode(&self, sequences: &mut [(&mut Hy4Placement, &[u32])], speculative: bool, pool: &RefPagePool) -> Result<Option<DeviceLogits>> {
        ensure!(!speculative, "Hy4 DFly2 speculative verify is not yet enabled");
        ensure!(
            !sequences.is_empty() && sequences.len() <= self.concurrency && self.pending.borrow().is_none(),
            "Hy4 decode requires admitted requests and prior commit"
        );
        let spans = decode_spans(sequences.iter().map(|(_, tokens)| tokens.len()), 64)?;
        let total_rows = spans.last().context("Hy4 decode has no rows")?.end;
        let mut slots = BTreeSet::new();
        let mut segments = Vec::with_capacity(sequences.len());
        let mut placements = Vec::with_capacity(sequences.len());
        let mut sources = Vec::with_capacity(sequences.len());
        let mut receipt = Vec::with_capacity(sequences.len());
        for (index, (placement, tokens)) in sequences.iter().enumerate() {
            self.check(placement, tokens, pool)?;
            ensure!(slots.insert(placement.slot), "Hy4 decode aliases a request state slot");
            segments.push(UnitSegment {
                sequence: index,
                rows: spans[index].clone(),
                tokens: placement.len..placement.len + tokens.len(),
                prompt_end: false,
                snapshot: false,
                head: true,
            });
            receipt.push((placement.slot, placement.len, tokens.to_vec()));
            placements.push((**placement).clone());
            sources.push(*tokens);
        }
        let layers = self.engine.borrow().cfg.layers;
        let unit = Unit {
            index: 0,
            sequence: 0,
            segments,
            rows: 0..total_rows,
            layers: 0..layers,
            slot: 0,
            barrier: false,
        };
        let mut logits = None;
        self.engine.borrow_mut().run_units(
            &mut placements,
            &sources,
            &[unit],
            ReadPriority::Decode,
            |_| HeadMode::Suffix(0),
            &mut |_, _, tile| {
                logits = Some(tile);
                Ok(())
            },
        )?;
        for ((placement, _), next) in sequences.iter_mut().zip(placements) {
            **placement = next;
        }
        *self.pending.borrow_mut() = Some(receipt);
        Ok(logits)
    }
}

impl<'a> Backend<'a> for Hy4Backend<'a> {
    type Placement = Hy4Placement;
    type Draft = Draft;
    type History = usize;
    type Coordinates = ();
    type Encoder = crate::shared::qwen::Encoder;
    type Meta = ();
    type Prefix = Hy4Prefix<'a>;

    fn geometry(&self) -> scheduler::Geometry {
        scheduler::Geometry {
            family: "hy4_flash",
            vocab_size: self.vocab,
            max_context: self.context,
            slots: self.concurrency,
            prefill_rows: self.unit_rows,
            decode_rows: 64,
        }
    }

    fn message_starts(&self) -> &[&str] {
        &["<｜hy_start:opensource｜>"]
    }

    fn library(&self) -> &'a cuteafd_ffi::NativeLibrary {
        self.library
    }

    fn profile(&self) -> &RefCell<[f64; 2]> {
        &self.profile
    }

    fn selector_ready(&self, selector: &mut TokenSelector<'a>) -> Result<()> {
        selector.reserve_sampler()
    }

    fn prefix_cache(
        &self,
        args: &PrefixArgs,
        concurrency: usize,
    ) -> Result<(Self::Prefix, PrefixCache<CudaCopyEngine<'a>>)> {
        ensure!(
            concurrency == self.concurrency && args.prefix_partial == crate::shared::prefix::Toggle::Off,
            "Hy4 prefix geometry requires exact admitted frontiers"
        );
        let mark_slots = self.engine.borrow().mark_slots();
        let host = self
            .host
            .clone()
            .map(|config| {
                CudaCopyEngine::registered_retained(self.library, vec![], self.family.snapshot_owners())
                    .map(|engine| (config, engine))
            })
            .transpose()?;
        let host_bytes = host.as_ref().map_or(0, |(config, _)| config.bytes);
        let layout = self.family.layout();
        let cache = PrefixCache::new(
            layout,
            PrefixConfig {
                entries: args.prefix_cache_entries,
                mark_slots,
                keep_logits: true,
                min_tokens: args.prefix_cache_min_tokens,
            },
            host,
        )?;
        cuteafd_bench::context::set_kv(
            (layout.pages * layout.page_rows) as u64,
            Some((layout.pages * layout.page_rows) as u64),
            layout.pages as u64,
            self.family.record_format(),
            host_bytes,
        );
        Ok((self.family.clone(), cache))
    }

    fn prepare(&self, job: NativeRequest, tokens: Vec<u32>) -> Result<(scheduler::Prompt, RequestMedia, Vec<EncodeJob>)> {
        ensure!(job.media.is_empty() && job.image_groups.is_empty(), "Hy4 vision tower is not enabled");
        ensure!(job.audio.is_empty(), "Hy4 has no bundled audio tower");
        let keys = cuteafd_engine::media::MediaKeys::new(&tokens, u32::try_from(self.vocab)?, &[])?;
        let media = RequestMedia::new(vec![], self.engine.borrow().cfg.hidden, tokens.len())?;
        Ok((scheduler::Prompt { job, tokens, keys }, media, vec![]))
    }

    fn probe_features(&self, _: &scheduler::Prompt, _: &mut RequestMedia, _: &mut EmbeddingCache, _: &Path) -> Result<()> {
        Ok(())
    }

    fn coordinates(&self, _: &[u32], _: &NativeRequest, _: &RequestMedia) -> Result<()> {
        Ok(())
    }

    fn placement(&self, units: Vec<u32>, slot: i32) -> Hy4Placement {
        Hy4Placement::new(units, slot as usize, self.pad_id)
    }

    fn bind_request(&self, placement: &mut Hy4Placement, _: (), _: RequestMedia, tokens: &[u32], resume: usize) -> Result<()> {
        ensure!(
            placement.len == resume && placement.state_len == resume && resume <= tokens.len(),
            "Hy4 restored request frontier differs"
        );
        placement.set_engram_tail_from_prefix(&tokens[..resume], self.pad_id);
        Ok(())
    }

    fn history(&self, placement: &Hy4Placement) -> usize {
        placement.slot
    }

    fn require_live(&self) -> Result<()> {
        self.engine.borrow().require_live()
    }

    fn log_table_stats(&self, phase: &str) {
        self.engine.borrow().engram.log_stats(phase);
    }

    fn prefill_device(
        &self,
        placement: &mut Hy4Placement,
        tokens: &[u32],
        rows: usize,
        pool: &RefPagePool,
    ) -> Result<Option<DeviceLogits>> {
        ensure!(rows <= 1, "Hy4 prefill head retains one row; scoring streams tiles");
        self.check(placement, tokens, pool)?;
        let start = placement.len;
        let layers = self.engine.borrow().cfg.layers;
        let mut units = sequence_units(tokens.len(), self.unit_rows, 1, layers)?;
        let last_idx = units.len() - 1;
        for (i, unit) in units.iter_mut().enumerate() {
            let u_start = start + unit.rows.start;
            let u_end = start + unit.rows.end;
            unit.rows = u_start..u_end;
            unit.segments = vec![UnitSegment {
                sequence: 0,
                rows: 0..unit.rows.len(),
                tokens: u_start..u_end,
                prompt_end: i == last_idx,
                snapshot: false,
                head: rows > 0 && i == last_idx,
            }];
        }
        let mut next = [placement.clone()];
        let mut last = None;
        self.engine.borrow_mut().run_units(
            &mut next,
            &[tokens],
            &units,
            ReadPriority::Prefill,
            |unit| if rows > 0 && unit.index == last_idx { HeadMode::LastPerSeq } else { HeadMode::None },
            &mut |_, _, logits| {
                last = Some(logits);
                Ok(())
            },
        )?;
        *placement = next.into_iter().next().unwrap();
        Ok(last)
    }

    fn streamed_prefill_scoring(&self) -> bool { true }

    fn prefill_scoring(
        &self,
        placement: &mut Hy4Placement,
        tokens: &[u32],
        rows: usize,
        position: usize,
        probe: &probe::ProbeRef,
        pool: &RefPagePool,
    ) -> Result<Option<probe::ScoreLogits>> {
        ensure!(rows <= tokens.len(), "Hy4 score suffix exceeds submitted rows");
        self.check(placement, tokens, pool)?;
        let start = placement.len;
        let first = start + tokens.len() - rows + 1;
        let layers = self.engine.borrow().cfg.layers;
        let mut units = sequence_units(tokens.len(), self.unit_rows, 1, layers)?;
        let last_idx = units.len() - 1;
        for (i, unit) in units.iter_mut().enumerate() {
            let u_start = start + unit.rows.start;
            let u_end = start + unit.rows.end;
            unit.rows = u_start..u_end;
            unit.segments = vec![UnitSegment {
                sequence: 0,
                rows: 0..unit.rows.len(),
                tokens: u_start..u_end,
                prompt_end: i == last_idx,
                snapshot: false,
                head: rows > 0 && u_end + 1 > first,
            }];
        }
        let scoring = probe::ScoringRows { consumer: probe, position };
        let mut next = [placement.clone()];
        self.engine.borrow_mut().run_units(
            &mut next,
            &[tokens],
            &units,
            ReadPriority::Prefill,
            |unit| {
                if rows == 0 || unit.rows.end + 1 <= first {
                    HeadMode::None
                } else {
                    HeadMode::Suffix(first.saturating_sub(unit.rows.start + 1))
                }
            },
            &mut |unit, row_off, logits| {
                let at = unit.rows.start + row_off + 1;
                ensure!(at >= first, "Hy4 scoring tile started before first scored position");
                scoring.record(self.library, at - first, &logits)
            },
        )?;
        *placement = next.into_iter().next().unwrap();
        Ok((rows > 0).then_some(probe::ScoreLogits::Recorded))
    }

    fn packed_limits(&self) -> Option<Limits> {
        Some(Limits {
            rows: self.unit_rows,
            chunk_rows: self.unit_rows,
            max_context: self.context,
            taps: None,
        })
    }

    fn prefill_packed(
        &self,
        sequences: &mut [scheduler::PrefillSeq<'_, Hy4Placement, Draft>],
        pool: &RefPagePool,
        on_logits: &mut dyn FnMut(usize, DeviceLogits) -> MemberOutcome,
    ) -> Result<Vec<Result<()>>> {
        ensure!(!sequences.is_empty(), "Hy4 packed submission is empty");
        ensure!(sequences.len() <= 64, "Hy4 packed member count exceeds 64-row head");
        let limits = self.packed_limits().unwrap();
        ensure!(
            sequences.iter().map(|s| s.placement.slot).collect::<BTreeSet<_>>().len() == sequences.len(),
            "Hy4 packed members alias a request state slot"
        );
        let chunks = sequences
            .iter()
            .map(|sequence| {
                ensure!(sequence.start == sequence.placement.len, "Hy4 packed start differs");
                ensure!(!sequence.neural, "Hy4 DFly2 neural drafting is not yet enabled");
                self.check(sequence.placement, sequence.tokens, pool)?;
                Ok((sequence.start, sequence.tokens.len()))
            })
            .collect::<Result<Vec<_>>>()?;
        let segments = crate::shared::prefill_packing::plan(&chunks, &limits)?;
        let rows = segments.last().context("Hy4 pack has no members")?.end_row();
        let mut placements: Vec<Hy4Placement> = sequences.iter().map(|s| s.placement.clone()).collect();
        let sources: Vec<&[u32]> = sequences.iter().map(|s| s.tokens).collect();
        let layers = self.engine.borrow().cfg.layers;
        let unit = Unit {
            index: 0,
            sequence: 0,
            rows: 0..rows,
            layers: 0..layers,
            slot: 0,
            barrier: false,
            segments: segments
                .iter()
                .enumerate()
                .map(|(sequence, segment)| UnitSegment {
                    sequence,
                    rows: segment.first..segment.end_row(),
                    tokens: segment.start..segment.start + segment.rows,
                    prompt_end: true,
                    snapshot: false,
                    head: true,
                })
                .collect(),
        };
        let mut outcomes: Vec<Option<MemberOutcome>> = (0..sequences.len()).map(|_| None).collect();
        self.engine.borrow_mut().run_units(
            &mut placements,
            &sources,
            &[unit],
            ReadPriority::Prefill,
            |_| HeadMode::LastPerSeq,
            &mut |_, _, logits| {
                ensure!(logits.rows == sequences.len(), "Hy4 packed head rows mismatch");
                for member in 0..sequences.len() {
                    let row = DeviceLogits {
                        ptr: logits.ptr.wrapping_byte_add(member * logits.stride * 4),
                        rows: 1,
                        ..logits
                    };
                    outcomes[member] = Some(on_logits(member, row));
                }
                Ok(())
            },
        )?;
        for (sequence, placement) in sequences.iter_mut().zip(placements) {
            *sequence.placement = placement;
        }
        outcomes
            .into_iter()
            .map(|o| o.context("Hy4 packed member last-row callback missing")?)
            .collect()
    }

    fn verify_device(
        &self,
        sequences: &mut [(&mut Hy4Placement, &[u32])],
        speculative: bool,
        pool: &RefPagePool,
    ) -> Result<Option<DeviceLogits>> {
        self.decode(sequences, speculative, pool)
    }

    fn verify_device_ungraphed(
        &self,
        sequences: &mut [(&mut Hy4Placement, &[u32])],
        speculative: bool,
        kind: scheduler::UngraphedKind,
        pool: &RefPagePool,
    ) -> Result<Option<DeviceLogits>> {
        if kind == scheduler::UngraphedKind::Decode {
            return self.decode(sequences, speculative, pool);
        }
        scoring_replay(
            sequences,
            speculative,
            |sequences| self.decode(sequences, false, pool),
            |verified| self.accept(verified, false),
        )
    }

    fn verify_bucket_rows(&self, rows: usize, _: bool, _: bool) -> usize {
        rows
    }

    fn verify_attention_trace(&self, _: &[Vec<usize>], rows: usize) -> crate::shared::speculation_trace::Deferred {
        crate::shared::speculation_trace::Deferred::new(move || {
            serde_json::json!({"kind": "hy4_flash", "rows": rows, "attention": "eager"})
        })
    }

    fn prefill_draft(&self, _: &Hy4Placement, _: usize, _: &[u32], next: Option<u32>, state: &mut Draft) -> Result<()> {
        state.anchor = next;
        Ok(())
    }

    fn draft(
        &self,
        sequences: &mut [scheduler::DraftSeq<'_, Hy4Placement, Draft>],
        _: &mut scheduler::DraftTiming,
        _: &mut TokenSelector<'_>,
    ) -> Result<Vec<Vec<u32>>> {
        Ok(vec![vec![]; sequences.len()])
    }

    fn accept(&self, verified: &mut [scheduler::Verified<'_, Hy4Placement, Draft, usize>], speculative: bool) -> Result<()> {
        let sent = self
            .pending
            .borrow_mut()
            .take()
            .context("Hy4 accept lacks submitted receipt")?;
        ensure!(!speculative && sent.len() == verified.len(), "Hy4 accept inventory differs");
        for ((slot, start, tokens), request) in sent.iter().zip(verified.iter()) {
            ensure!(
                *slot == request.history
                    && *slot == request.placement.slot
                    && *start == request.start
                    && tokens == request.rows,
                "Hy4 accept receipt differs"
            );
        }
        Ok(())
    }

    fn cancel(&self, _: &Hy4Placement) -> Result<()> {
        let _ = self.pending.borrow_mut().take();
        self.engine.borrow().drain()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn append_writes_only_reserved_private_nonaliased_pages() -> Result<()> {
        let mut pool = RefPagePool::new(4, PAGE_ROWS);
        let units = pool.alloc_for(0..2)?;
        let mut p = Hy4Placement::new(units.clone(), 0, 120_002);
        p.len = 63;
        p.state_len = 63;
        check_writable(&p, 3, 128, &pool)?;
        pool.share(&units[..1]);
        assert!(check_writable(&p, 3, 128, &pool).is_err());
        pool.release(&units[..1]);
        let alias = Hy4Placement::new(vec![units[0], units[0]], 0, 120_002);
        assert!(check_writable(&alias, 128, 128, &pool).is_err());
        assert!(check_writable(&p, 66, 128, &pool).is_err());
        assert!(check_writable(&p, 0, 128, &pool).is_err());
        Ok(())
    }

    #[test]
    fn scoring_replay_commits_receipts_between_windows() -> Result<()> {
        let pending = RefCell::new(None);
        let mut first = Hy4Placement::new(vec![0], 0, 120_002);
        first.len = 11;
        first.state_len = 11;
        let mut decode = |seqs: &mut [(&mut Hy4Placement, &[u32])]| -> Result<Option<DeviceLogits>> {
            ensure!(pending.borrow().is_none(), "prior commit required");
            let r = seqs.iter().map(|(p, t)| (p.slot, p.len, t.to_vec())).collect::<Vec<_>>();
            for (p, t) in seqs {
                p.len += t.len();
                p.state_len = p.len;
            }
            *pending.borrow_mut() = Some(r);
            Ok(None)
        };
        let mut accept = |verified: &mut [scheduler::Verified<'_, Hy4Placement, Draft, usize>]| -> Result<()> {
            let sent = pending.borrow_mut().take().context("missing receipt")?;
            assert_eq!(sent.len(), verified.len());
            Ok(())
        };
        for _ in 0..3 {
            scoring_replay(&mut [(&mut first, &[10, 20])], false, &mut decode, &mut accept)?;
            assert!(pending.borrow().is_none());
        }
        assert_eq!(first.len, 17);
        assert!(scoring_replay(&mut [(&mut first, &[1])], true, &mut decode, &mut accept).is_err());
        Ok(())
    }

    #[test]
    fn adapter_uses_shared_scheduler_and_prefill_units() {
        let backend = include_str!("backend.rs").split("#[cfg(test)]").next().unwrap();
        let exec = include_str!("execution.rs").split("#[cfg(test)]").next().unwrap();
        assert!(backend.contains("Backend<'a> for Hy4Backend"));
        assert!(backend.contains("streamed_prefill_scoring(&self) -> bool { true }"));
        assert!(backend.contains("prefill_packing::plan("));
        assert!(backend.contains("registered_retained("));
        assert!(!backend.contains("tokio::select!"));
        assert!(!backend.contains("loop {"));
        assert!(exec.contains("run_prefill_units("));
    }
}
