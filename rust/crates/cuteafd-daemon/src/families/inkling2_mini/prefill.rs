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
        run_prefill_units(self, units, waves)
    }

    fn packed_members(&self, unit: &Unit) -> Result<Vec<PackedMember>> {
        members(unit)
            .into_iter()
            .map(|member| {
                let placement = &self.placements[member.sequence];
                Ok(PackedMember {
                    sequence: placement.slot,
                    rows: member.rows,
                    metadata: UnitMetadata::new(
                        &self.cfg.text,
                        member.tokens,
                        &placement.units,
                        self.cache.layout.pool_pages,
                    )?,
                })
            })
            .collect()
    }
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

impl PrefillHooks for Stateful<'_, '_> {
    type Ticket = ();

    fn route(&self, _visit_idx: usize) -> FfnRoute {
        FfnRoute::Local
    }

    fn begin_unit(&mut self, unit: &Unit) -> Result<()> {
        self.io
            .begin(unit, &self.owners[unit.slot], &mut self.uploads[unit.slot])
    }

    fn attention(&mut self, visit_idx: usize, unit: &Unit) -> Result<()> {
        let visit = self.visits[visit_idx];
        let owner = &self.owners[unit.slot];
        self.io.tap(visit_idx, unit, owner)?;
        let members = self.packed_members(unit)?;
        owner.packed_metadata(visit.cache_slot, self.cache, &members, &mut self.uploads[unit.slot])?;
        let state = PackedVisitUnit {
            unit: owner,
            cache: self.cache,
            cache_slot: visit.cache_slot,
            members: &members,
        };
        let rows = unit.rows.len();
        let mode = if owner.replaying() { Mode::Verify } else { Mode::Decode };
        graphs::local(
            self.graphs.as_deref_mut(),
            state.stream(),
            visit_idx,
            rows,
            Section::VisitAttention,
            mode,
            || layers::begin_visit(self.kernels, &state, self.target, self.cfg, visit, rows),
        )
    }

    fn local_ffn(&mut self, visit_idx: usize, unit: &Unit) -> Result<()> {
        let visit = self.visits[visit_idx];
        let owner = &self.owners[unit.slot];
        let rows = unit.rows.len();
        let experts = self.local[visit.layer].as_ref();
        graphs::local(
            self.graphs.as_deref_mut(),
            &owner.stream,
            visit_idx,
            rows,
            Section::Routes,
            Mode::Decode,
            || layers::prepare_routes(self.kernels, owner, self.target, experts, visit.layer, rows),
        )?;
        layers::dispatch_experts(owner, experts, rows)?;
        let members = self.packed_members(unit)?;
        let state = PackedVisitUnit {
            unit: owner,
            cache: self.cache,
            cache_slot: visit.cache_slot,
            members: &members,
        };
        let mode = if owner.replaying() { Mode::Verify } else { Mode::Decode };
        graphs::local(
            self.graphs.as_deref_mut(),
            state.stream(),
            visit_idx,
            rows,
            Section::VisitFinish,
            mode,
            || layers::join_visit(self.kernels, &state, self.target, self.cfg, visit, rows),
        )
    }

    fn send_experts(&mut self, _: usize, _: &Unit) -> Result<()> {
        bail!("Inkling 2 Mini is TP1 local-only; Spark expert offload is unused")
    }

    fn land_experts(&mut self, _: usize, _: &Unit, _: ()) -> Result<()> {
        bail!("Inkling 2 Mini is TP1 local-only; Spark expert offload is unused")
    }

    fn end_unit(&mut self, unit: &Unit) -> Result<()> {
        self.io.end(unit, &self.owners[unit.slot])
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
