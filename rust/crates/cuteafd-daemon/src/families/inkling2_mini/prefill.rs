//! Inkling 2 Mini 48-visit TP1 unit executor on the shared `PrefillHooks` pipeline.
use super::{
    experts::Dispatcher,
    graphs::{self, DecodeGraphs, Mode, Section},
    layers::{self, MiniKernels},
    owners::{PackedMember, PackedVisitUnit, UnitBuffers, UnitOwner},
    weights::MiniTargetWeights,
};
use crate::shared::prefill_pipeline::{run_prefill_units, FfnRoute, PrefillHooks, Unit, UnitSegment};
use anyhow::{bail, ensure, Context, Result};
use cuteafd_loader::families::inkling2::execution::UnitMetadata;
use cuteafd_loader::families::inkling2_mini::{Inkling2MiniConfig, LoopVisit};

pub(crate) trait UnitIo<'a> {
    fn begin(
        &mut self,
        unit: &Unit,
        owner: &UnitOwner<'a>,
        uploads: &mut crate::shared::token_table::RetainedUploads<'a>,
    ) -> Result<()>;
    fn end(&mut self, unit: &Unit, owner: &UnitOwner<'a>) -> Result<()>;
    fn tap(&mut self, _visit_idx: usize, _unit: &Unit, _owner: &UnitOwner<'a>) -> Result<()> {
        Ok(())
    }
}

pub(crate) struct TokenIo<'s, 'a> {
    pub kernels: &'s MiniKernels<'a>,
    pub head: &'s super::head::MiniHead<'a>,
    pub cache: &'s super::cache::CacheBank<'a>,
    pub target: &'s MiniTargetWeights<'a>,
    pub cfg: &'s Inkling2MiniConfig,
    pub tokens: &'s [&'s [u32]],
    pub starts: &'s [usize],
    pub slots: &'s [usize],
    pub retained_rows: bool,
    pub draft: Option<&'s super::draft::Binding<'a>>,
    pub publish_draft: bool,
    pub media: &'s [Option<cuteafd_engine::media::RequestMedia>],
    pub on_logits: &'s mut dyn FnMut(usize, usize, crate::shared::token_io::DeviceLogits) -> Result<()>,
}

impl<'a> UnitIo<'a> for TokenIo<'_, 'a> {
    fn tap(&mut self, visit_idx: usize, unit: &Unit, owner: &UnitOwner<'a>) -> Result<()> {
        if let Some(draft) = self.draft {
            draft
                .taps(self.retained_rows, unit.slot)?
                .entry(visit_idx, owner.buffer("partial")?, unit.rows.len())?;
        }
        Ok(())
    }

    fn begin(
        &mut self,
        unit: &Unit,
        owner: &UnitOwner<'a>,
        uploads: &mut crate::shared::token_table::RetainedUploads<'a>,
    ) -> Result<()> {
        let tokens = unit_tokens(unit, self.tokens, self.starts)?;
        let mut chunk = cuteafd_engine::media::MediaChunk::default();
        for member in members(unit) {
            if let Some(media) = self.media.get(member.sequence).and_then(Option::as_ref) {
                let mut part = cuteafd_engine::media::MediaChunk::default();
                media.write_chunk(member.tokens.start, member.tokens.end, &mut part)?;
                chunk
                    .indices
                    .extend(part.indices.into_iter().map(|row| row + member.rows.start as u32));
                chunk.features.extend(part.features);
            }
        }
        self.head
            .embed_unit(self.kernels, owner, self.target, self.cfg, &tokens, &chunk, uploads)
    }

    fn end(&mut self, unit: &Unit, owner: &UnitOwner<'a>) -> Result<()> {
        let members = members(unit);
        let head = members.iter().any(|m| m.head);
        if !head && self.draft.is_none() {
            return Ok(());
        }
        let sequence = *self
            .slots
            .get(members[0].sequence)
            .context("Inkling 2 Mini head sequence slot absent")?;
        let final_candidates = self.cfg.text.residual_blocks() + 1; // 5
        let final_slot = self.cfg.cache_slot(self.cfg.text.layers - 1, 0);
        for span in crate::shared::probe::head_spans(unit.rows.len(), 64)? {
            let state = owner.segment(self.cache, sequence, final_slot, span.clone())?;
            let output_row = if self.retained_rows { span.start } else { 0 };
            let logits = if head {
                Some(self.head.project_at(
                    self.kernels,
                    &state,
                    self.target,
                    self.cfg,
                    span.len(),
                    final_candidates,
                    output_row,
                )?)
            } else {
                self.head
                    .normalize(self.kernels, &state, self.target, span.len(), final_candidates)?;
                None
            };
            if let Some(draft) = self.draft {
                draft.taps(self.retained_rows, unit.slot)?.final_norm(
                    self.head.normalized(span.len())?,
                    span.start,
                    span.len(),
                )?;
            }
            let Some(logits) = logits else {
                continue;
            };
            for member in members.iter().filter(|m| m.head) {
                let first = span.start.max(member.rows.start);
                let end = span.end.min(member.rows.end);
                if first >= end {
                    continue;
                }
                let offset = (first - span.start) * self.cfg.text.unpadded_vocab_size * 4;
                (self.on_logits)(
                    member.sequence,
                    member.tokens.start + first - member.rows.start + 1,
                    crate::shared::token_io::DeviceLogits {
                        ptr: logits.ptr.wrapping_byte_add(offset),
                        rows: end - first,
                        vocab: self.cfg.text.unpadded_vocab_size,
                        stride: self.cfg.text.unpadded_vocab_size,
                        stream: owner.stream.raw,
                        greedy: None,
                    },
                )?;
            }
        }
        if self.publish_draft {
            if let Some(draft) = self.draft {
                let taps = draft.taps(self.retained_rows, unit.slot)?;
                for member in &members {
                    let slot = *self
                        .slots
                        .get(member.sequence)
                        .context("Mini draft member slot missing")?;
                    draft.update(taps, slot, member.tokens.start, member.rows.start, member.rows.len())?;
                }
            }
        }
        Ok(())
    }
}

pub(crate) struct Stateful<'s, 'a> {
    pub(super) graphs: Option<&'s mut DecodeGraphs<'a>>,
    pub kernels: &'s MiniKernels<'a>,
    pub owners: &'s [UnitOwner<'a>],
    pub cache: &'s super::cache::CacheBank<'a>,
    pub target: &'s MiniTargetWeights<'a>,
    pub cfg: &'s Inkling2MiniConfig,
    pub visits: &'s [LoopVisit],
    pub placements: &'s mut [super::cache::Placement],
    pub uploads: &'s mut [crate::shared::token_table::RetainedUploads<'a>],
    pub local: &'s [Box<dyn Dispatcher + 'a>],
    pub io: &'s mut dyn UnitIo<'a>,
}

impl Stateful<'_, '_> {
    pub fn run(&mut self, units: &[Unit]) -> Result<()> {
        let total_visits = self.visits.len();
        ensure!(
            !units.is_empty()
                && self.owners.len() == self.uploads.len()
                && self.local.len() == self.cfg.text.layers
                && total_visits == self.cfg.total_visits(),
            "Inkling 2 Mini stateful prefill owners/visits differ"
        );
        let stream = &self
            .owners
            .first()
            .context("Inkling 2 Mini prefill has no unit owners")?
            .stream;
        ensure!(
            self.owners.iter().all(|o| std::rc::Rc::ptr_eq(&o.stream, stream)),
            "Inkling 2 Mini causal state requires one ordered stream"
        );
        ensure!(
            units.iter().all(|u| {
                u.slot < self.owners.len()
                    && u.layers == (0..total_visits)
                    && members(u).iter().all(|m| m.sequence < self.placements.len())
            }),
            "Inkling 2 Mini prefill unit differs from admitted requests"
        );
        let waves = self.owners.len();
        let mut prepared = PreparedUnitRunner {
            cfg: self.cfg,
            visits: self.visits,
            owners: self.owners,
            cache: self.cache,
            placements: self.placements,
            uploads: self.uploads,
            members: vec![Vec::new(); waves],
            ops: StatefulOps {
                graphs: &mut self.graphs,
                kernels: self.kernels,
                target: self.target,
                local: self.local,
                io: self.io,
            },
        };
        run_prefill_units(&mut prepared, units, waves)
    }
}

trait VisitOps<'a> {
    fn begin_io(
        &mut self,
        unit: &Unit,
        owner: &UnitOwner<'a>,
        uploads: &mut crate::shared::token_table::RetainedUploads<'a>,
    ) -> Result<()>;
    fn attention(
        &mut self,
        visit_idx: usize,
        visit: LoopVisit,
        unit: &Unit,
        cfg: &Inkling2MiniConfig,
        state: &PackedVisitUnit<'_, 'a>,
    ) -> Result<()>;
    fn local_ffn(
        &mut self,
        visit_idx: usize,
        visit: LoopVisit,
        unit: &Unit,
        cfg: &Inkling2MiniConfig,
        state: &PackedVisitUnit<'_, 'a>,
    ) -> Result<()>;
    fn end_io(&mut self, unit: &Unit, owner: &UnitOwner<'a>) -> Result<()>;
}

struct StatefulOps<'p, 's, 'a> {
    graphs: &'p mut Option<&'s mut DecodeGraphs<'a>>,
    kernels: &'s MiniKernels<'a>,
    target: &'s MiniTargetWeights<'a>,
    local: &'s [Box<dyn Dispatcher + 'a>],
    io: &'p mut dyn UnitIo<'a>,
}

impl<'a> VisitOps<'a> for StatefulOps<'_, '_, 'a> {
    fn begin_io(
        &mut self,
        unit: &Unit,
        owner: &UnitOwner<'a>,
        uploads: &mut crate::shared::token_table::RetainedUploads<'a>,
    ) -> Result<()> {
        self.io.begin(unit, owner, uploads)
    }

    fn attention(
        &mut self,
        visit_idx: usize,
        visit: LoopVisit,
        unit: &Unit,
        cfg: &Inkling2MiniConfig,
        state: &PackedVisitUnit<'_, 'a>,
    ) -> Result<()> {
        self.io.tap(visit_idx, unit, state.unit)?;
        let rows = unit.rows.len();
        let mode = if state.unit.replaying() {
            Mode::Verify
        } else {
            Mode::Decode
        };
        graphs::local(
            self.graphs.as_deref_mut(),
            state.stream(),
            visit_idx,
            rows,
            Section::VisitAttention,
            mode,
            || layers::begin_visit(self.kernels, state, self.target, cfg, visit, rows),
        )
    }

    fn local_ffn(
        &mut self,
        visit_idx: usize,
        visit: LoopVisit,
        unit: &Unit,
        cfg: &Inkling2MiniConfig,
        state: &PackedVisitUnit<'_, 'a>,
    ) -> Result<()> {
        let rows = unit.rows.len();
        let experts = self.local[visit.layer].as_ref();
        graphs::local(
            self.graphs.as_deref_mut(),
            &state.unit.stream,
            visit_idx,
            rows,
            Section::Routes,
            Mode::Decode,
            || layers::prepare_routes(self.kernels, state.unit, self.target, experts, visit.layer, rows),
        )?;
        layers::dispatch_experts(state.unit, experts, rows)?;
        let mode = if state.unit.replaying() {
            Mode::Verify
        } else {
            Mode::Decode
        };
        graphs::local(
            self.graphs.as_deref_mut(),
            state.stream(),
            visit_idx,
            rows,
            Section::VisitFinish,
            mode,
            || layers::join_visit(self.kernels, state, self.target, cfg, visit, rows),
        )
    }

    fn end_io(&mut self, unit: &Unit, owner: &UnitOwner<'a>) -> Result<()> {
        self.io.end(unit, owner)
    }
}

fn build_packed_members(
    cfg: &Inkling2MiniConfig,
    cache: &super::cache::CacheBank<'_>,
    placements: &[super::cache::Placement],
    unit: &Unit,
) -> Result<Vec<PackedMember>> {
    members(unit)
        .into_iter()
        .map(|member| {
            let placement = &placements[member.sequence];
            Ok(PackedMember {
                sequence: placement.slot,
                rows: member.rows,
                metadata: UnitMetadata::new(
                    &cfg.text,
                    member.tokens,
                    &placement.units,
                    cache.layout.pool_pages,
                )?,
            })
        })
        .collect()
}

struct PreparedUnitRunner<'s, 'a, O> {
    cfg: &'s Inkling2MiniConfig,
    visits: &'s [LoopVisit],
    owners: &'s [UnitOwner<'a>],
    cache: &'s super::cache::CacheBank<'a>,
    placements: &'s mut [super::cache::Placement],
    uploads: &'s mut [crate::shared::token_table::RetainedUploads<'a>],
    members: Vec<Vec<PackedMember>>,
    ops: O,
}

fn unit_tokens(unit: &Unit, sources: &[&[u32]], starts: &[usize]) -> Result<Vec<u32>> {
    let mut tokens = Vec::with_capacity(unit.rows.len());
    for member in members(unit) {
        let source = sources
            .get(member.sequence)
            .context("Mini unit token source absent")?;
        let origin = *starts
            .get(member.sequence)
            .context("Mini token source origin absent")?;
        let first = member
            .tokens
            .start
            .checked_sub(origin)
            .context("Mini token span precedes submitted suffix")?;
        let end = member
            .tokens
            .end
            .checked_sub(origin)
            .context("Mini token end precedes submitted suffix")?;
        tokens.extend_from_slice(source.get(first..end).context("Mini unit token span outside request")?);
    }
    ensure!(tokens.len() == unit.rows.len(), "Mini token members do not cover unit");
    Ok(tokens)
}

fn members(unit: &Unit) -> Vec<UnitSegment> {
    if !unit.segments.is_empty() {
        return unit.segments.clone();
    }
    vec![UnitSegment {
        sequence: unit.sequence,
        rows: 0..unit.rows.len(),
        tokens: unit.rows.clone(),
        prompt_end: false,
        snapshot: false,
        head: false,
    }]
}

impl<'a, O: VisitOps<'a>> PrefillHooks for PreparedUnitRunner<'_, 'a, O> {
    type Ticket = ();

    fn route(&self, _visit_idx: usize) -> FfnRoute {
        FfnRoute::Local
    }

    fn begin_unit(&mut self, unit: &Unit) -> Result<()> {
        let owner = &self.owners[unit.slot];
        let uploads = &mut self.uploads[unit.slot];
        // Embed runs first (it writes token ids into `first` before reading them);
        // then upload all 9 read-only metadata planes once for all visits.
        self.ops.begin_io(unit, owner, uploads)?;
        let packed = build_packed_members(self.cfg, self.cache, self.placements, unit)?;
        owner.packed_metadata(self.cache, &packed, uploads)?;
        self.members[unit.slot] = packed;
        Ok(())
    }

    fn attention(&mut self, visit_idx: usize, unit: &Unit) -> Result<()> {
        let visit = self.visits[visit_idx];
        let owner = &self.owners[unit.slot];
        let members = &self.members[unit.slot];
        ensure!(
            !members.is_empty(),
            "Inkling 2 Mini attention ran before begin_unit prepared metadata"
        );
        let state = PackedVisitUnit {
            unit: owner,
            cache: self.cache,
            cache_slot: visit.cache_slot,
            members,
        };
        self.ops.attention(visit_idx, visit, unit, self.cfg, &state)
    }

    fn local_ffn(&mut self, visit_idx: usize, unit: &Unit) -> Result<()> {
        let visit = self.visits[visit_idx];
        let owner = &self.owners[unit.slot];
        let members = &self.members[unit.slot];
        ensure!(
            !members.is_empty(),
            "Inkling 2 Mini local_ffn ran before begin_unit prepared metadata"
        );
        let state = PackedVisitUnit {
            unit: owner,
            cache: self.cache,
            cache_slot: visit.cache_slot,
            members,
        };
        self.ops.local_ffn(visit_idx, visit, unit, self.cfg, &state)
    }

    fn send_experts(&mut self, _: usize, _: &Unit) -> Result<()> {
        bail!("Inkling 2 Mini is TP1 local-only; Spark expert offload is unused")
    }

    fn land_experts(&mut self, _: usize, _: &Unit, _: ()) -> Result<()> {
        bail!("Inkling 2 Mini is TP1 local-only; Spark expert offload is unused")
    }

    fn end_unit(&mut self, unit: &Unit) -> Result<()> {
        self.ops.end_io(unit, &self.owners[unit.slot])
    }

    fn commit_unit(&mut self, unit: &Unit) -> Result<()> {
        for member in members(unit) {
            let placement = &mut self.placements[member.sequence];
            ensure!(
                placement.state_len == member.tokens.start,
                "Inkling 2 Mini causal commit is out of order"
            );
            placement.state_len = member.tokens.end;
            placement.len = member.tokens.end;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::memory::device::{Device, Stream};
    use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
    use cuteafd_loader::families::inkling2_mini::{
        execution::{MiniCacheLayout, MiniUnitLayout},
        InklingKvCache,
    };
    use std::rc::Rc;

    fn test_config() -> Result<Inkling2MiniConfig> {
        Inkling2MiniConfig::from_hf(&serde_json::from_str(include_str!(
            "../../../../cuteafd-loader/src/families/inkling2_mini/fixtures/config.json"
        ))?)
    }

    struct StubVisitOps {
        upload_tokens: bool,
        attention_visits: usize,
        ffn_visits: usize,
    }

    impl<'a> VisitOps<'a> for StubVisitOps {
        fn begin_io(
            &mut self,
            unit: &Unit,
            owner: &UnitOwner<'a>,
            uploads: &mut crate::shared::token_table::RetainedUploads<'a>,
        ) -> Result<()> {
            if self.upload_tokens {
                let bytes = vec![0u8; unit.rows.len() * 4];
                // SAFETY: matches embed_unit uploading token ids into `first`.
                unsafe {
                    uploads.upload(owner.buffer("first")?, &bytes, owner.stream.raw)?;
                }
            }
            Ok(())
        }

        fn attention(
            &mut self,
            _visit_idx: usize,
            _visit: LoopVisit,
            _unit: &Unit,
            _cfg: &Inkling2MiniConfig,
            state: &PackedVisitUnit<'_, 'a>,
        ) -> Result<()> {
            assert!(!state.members.is_empty());
            // Verify every visit resolves the shared device metadata views.
            let _ = state.buffer("positions")?;
            let _ = state.buffer("cache_slots")?;
            let _ = state.buffer("ring_slots")?;
            let _ = state.buffer("page_table")?;
            self.attention_visits += 1;
            Ok(())
        }

        fn local_ffn(
            &mut self,
            _visit_idx: usize,
            _visit: LoopVisit,
            _unit: &Unit,
            _cfg: &Inkling2MiniConfig,
            state: &PackedVisitUnit<'_, 'a>,
        ) -> Result<()> {
            assert!(!state.members.is_empty());
            self.ffn_visits += 1;
            Ok(())
        }

        fn end_io(&mut self, _unit: &Unit, _owner: &UnitOwner<'a>) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn prefill_metadata_h2d_copies_are_bounded_and_independent_of_visit_count() -> Result<()> {
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.immediate_copies(&library)?;
        fixture.configure_device_flags(&library, 0)?;
        let device = Device {
            library: &library,
            id: 0,
        };
        let stream = Rc::new(Stream::new(device)?);
        let cfg = test_config()?;
        let all_visits = cfg.visits();
        assert_eq!(all_visits.len(), 48);
        let cache = super::super::cache::CacheBank::new(
            vec![stream.clone()],
            MiniCacheLayout::with_format(&cfg, 2, 4, InklingKvCache::Fp8)?,
            0,
        )?;
        let owners = vec![UnitOwner::new(
            stream.clone(),
            MiniUnitLayout::serving(&cfg, 4, 128, InklingKvCache::Fp8, 2)?,
        )?];
        let mut uploads = vec![owners[0].uploads(&cfg, 2)?];

        fn run_for_visits<'a>(
            fixture: &Fixture,
            cfg: &Inkling2MiniConfig,
            all_visits: &[LoopVisit],
            stream: &Stream<'a>,
            cache: &super::super::cache::CacheBank<'a>,
            owners: &[UnitOwner<'a>],
            uploads: &mut [crate::shared::token_table::RetainedUploads<'a>],
            visit_count: usize,
            upload_tokens: bool,
        ) -> Result<usize> {
            let mut placements = vec![
                super::super::cache::Placement {
                    units: vec![2],
                    slot: 1,
                    len: 63,
                    state_len: 63,
                    media: None,
                },
                super::super::cache::Placement {
                    units: vec![0, 3],
                    slot: 0,
                    len: 65,
                    state_len: 65,
                    media: None,
                },
            ];
            let unit = Unit {
                index: 0,
                sequence: 0,
                segments: vec![
                    UnitSegment {
                        sequence: 0,
                        rows: 0..1,
                        tokens: 63..64,
                        prompt_end: false,
                        snapshot: false,
                        head: true,
                    },
                    UnitSegment {
                        sequence: 1,
                        rows: 1..4,
                        tokens: 65..68,
                        prompt_end: false,
                        snapshot: false,
                        head: true,
                    },
                ],
                rows: 0..4,
                layers: 0..visit_count,
                slot: 0,
                barrier: false,
            };
            let before = fixture.events()?.chars().filter(|&c| c == 'C').count();
            let mut runner = PreparedUnitRunner {
                cfg,
                visits: &all_visits[..visit_count],
                owners,
                cache,
                placements: &mut placements,
                uploads,
                members: vec![Vec::new(); 1],
                ops: StubVisitOps {
                    upload_tokens,
                    attention_visits: 0,
                    ffn_visits: 0,
                },
            };
            run_prefill_units(&mut runner, &[unit], 1)?;
            stream.drain()?;
            assert_eq!(runner.ops.attention_visits, visit_count);
            assert_eq!(runner.ops.ffn_visits, visit_count);
            assert_eq!(placements[0].len, 64);
            assert_eq!(placements[1].len, 68);
            let after = fixture.events()?.chars().filter(|&c| c == 'C').count();
            Ok(after - before)
        }

        // Metadata-only path: 1 batched H2D copy for 1 visit AND 1 batched H2D copy for 48 visits.
        let copies_1 = run_for_visits(&fixture, &cfg, &all_visits, &stream, &cache, &owners, &mut uploads, 1, false)?;
        let copies_48 =
            run_for_visits(&fixture, &cfg, &all_visits, &stream, &cache, &owners, &mut uploads, 48, false)?;
        assert_eq!(copies_1, 1);
        assert_eq!(copies_48, 1);
        assert_eq!(copies_48, copies_1, "metadata H2D copies must not scale with visit count");

        // Full begin_unit path (token ids + batched metadata): 2 copies for 1 visit AND 2 for 48 visits.
        let full_1 = run_for_visits(&fixture, &cfg, &all_visits, &stream, &cache, &owners, &mut uploads, 1, true)?;
        let full_48 = run_for_visits(&fixture, &cfg, &all_visits, &stream, &cache, &owners, &mut uploads, 48, true)?;
        assert_eq!(full_1, 2);
        assert_eq!(full_48, 2);
        assert_eq!(full_48, full_1, "full unit H2D copies must stay bounded and independent of 48 visits");
        Ok(())
    }
}
