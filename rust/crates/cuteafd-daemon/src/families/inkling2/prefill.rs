//! Family arithmetic on the shared causal unit executor; no private schedule.
use super::{experts::Dispatcher, layers::{self, Kernels}, owners::UnitBuffers, weights::TargetWeights};
use crate::shared::prefill_pipeline::{run_prefill_units, sequence_units, FfnRoute, PrefillHooks, Unit};
use anyhow::{bail, ensure, Context, Result};
use cuteafd_loader::families::inkling2::Inkling2Config;

/// Persistent whole-rank prefill over shared Spark waves. Each in-flight unit
/// has an admitted scratch owner; global pages and causal histories are shared
/// request owners, ordered by run_prefill_units at every layer.
pub(crate) trait UnitIo<'a> {
    fn begin(&mut self, unit: &Unit, owner: &super::owners::UnitOwner<'a>,
        uploads: &mut crate::shared::token_table::RetainedUploads<'a>) -> Result<()>;
    fn end(&mut self, unit: &Unit, owner: &super::owners::UnitOwner<'a>) -> Result<()>;
    fn tap(&mut self,_layer:usize,_unit:&Unit,_owner:&super::owners::UnitOwner<'a>)->Result<()> {Ok(())}
}
/// Token/head edge for ordinary prefill and bounded streamed scoring. The
/// callback consumes scratch logits before another head tile overwrites them.
pub(crate) struct TokenIo<'s, 'a> {
    pub kernels: &'s Kernels<'a>,
    pub head: &'s super::head::Head<'a>,
    pub cache: &'s super::cache::CacheBank<'a>,
    pub target: &'s TargetWeights<'a>,
    pub cfg: &'s Inkling2Config,
    /// Submission-local token streams in the shared unit sequence order.
    pub tokens: &'s [&'s [u32]],
    pub starts: &'s [usize],
    pub slots: &'s [usize],
    pub retained_rows: bool,
    pub draft:Option<&'s super::draft::Binding<'a>>,
    pub publish_draft:bool,
    pub on_logits: &'s mut dyn FnMut(usize, usize, crate::shared::token_io::DeviceLogits) -> Result<()>,
}
impl<'a> UnitIo<'a> for TokenIo<'_, 'a> {
    fn tap(&mut self,layer:usize,unit:&Unit,owner:&super::owners::UnitOwner<'a>)->Result<()> {
        if let Some(draft)=self.draft {
            draft.taps(self.retained_rows,unit.slot)?.entry(layer,owner.buffer("partial")?,unit.rows.len())?;
        }
        Ok(())
    }
    fn begin(&mut self, unit: &Unit, owner: &super::owners::UnitOwner<'a>,
        uploads: &mut crate::shared::token_table::RetainedUploads<'a>) -> Result<()> {
        let tokens = unit_tokens(unit, self.tokens, self.starts)?;
        self.head.embed_unit(self.kernels, owner, self.target, self.cfg, &tokens, uploads)
    }
    fn end(&mut self, unit: &Unit, owner: &super::owners::UnitOwner<'a>) -> Result<()> {
        let members = members(unit);
        let head=members.iter().any(|member|member.head);
        if !head && self.draft.is_none() {return Ok(());}
        let sequence = *self.slots.get(members[0].sequence).context("Inkling head sequence slot absent")?;
        for span in crate::shared::probe::head_spans(unit.rows.len(), 64)? {
            let state = owner.segment(self.cache, sequence, self.cfg.layers-1, span.clone())?;
            let output_row = if self.retained_rows { span.start } else { 0 };
            let logits = if head {Some(self.head.project_at(self.kernels,&state,self.target,self.cfg,span.len(),
                self.cfg.residual_blocks()+1,output_row)?)} else {
                self.head.normalize(self.kernels,&state,self.target,span.len(),self.cfg.residual_blocks()+1)?;None
            };
            if let Some(draft)=self.draft {
                draft.taps(self.retained_rows,unit.slot)?.final_norm(self.head.normalized(span.len())?,span.start,span.len())?;
            }
            let Some(logits)=logits else {continue;};
            for member in members.iter().filter(|member| member.head) {
                let first = span.start.max(member.rows.start);
                let end = span.end.min(member.rows.end);
                if first >= end { continue; }
                let offset = (first-span.start)*self.cfg.unpadded_vocab_size*4;
                (self.on_logits)(member.sequence, member.tokens.start+first-member.rows.start+1,
                    crate::shared::token_io::DeviceLogits { ptr: logits.ptr.wrapping_byte_add(offset), rows: end-first,
                        vocab: self.cfg.unpadded_vocab_size, stride: self.cfg.unpadded_vocab_size,
                        stream: owner.stream.raw, greedy: None })?;
            }
        }
        if self.publish_draft {
            if let Some(draft)=self.draft {
                let taps=draft.taps(self.retained_rows,unit.slot)?;
                for member in &members {
                    let slot=*self.slots.get(member.sequence).context("draft member slot missing")?;
                    draft.update(taps,slot,member.tokens.start,member.rows.start,member.rows.len())?;
                }
            }
        }
        Ok(())
    }
}

pub(crate) struct Stateful<'s, 'a> {
    pub kernels: &'s Kernels<'a>,
    pub owners: &'s [super::owners::UnitOwner<'a>],
    pub cache: &'s super::cache::CacheBank<'a>,
    pub target: &'s TargetWeights<'a>,
    pub cfg: &'s Inkling2Config,
    pub ingress: &'s [cuteafd_core::SparkIngress],
    pub placements: &'s mut [super::cache::Placement],
    pub uploads: &'s mut [crate::shared::token_table::RetainedUploads<'a>],
    pub waves: &'s mut crate::shared::spark_waves::Waves<'a>,
    pub onboarded: usize,
    pub local: &'s [Box<dyn Dispatcher + 'a>],
    pub next_identity: &'s mut u64,
    pub placement_version: u64,
    pub kind: cuteafd_transport::ExpertV2SourceKind,
    pub io: &'s mut dyn UnitIo<'a>,
}
impl Stateful<'_, '_> {
    pub fn run(&mut self, units: &[Unit]) -> Result<()> {
        ensure!(!units.is_empty() && self.owners.len() == self.uploads.len()
            && self.owners.len() <= self.waves.waves() && self.onboarded == self.local.len()
            && self.onboarded <= self.cfg.layers && self.ingress.len() == self.cfg.layers && self.placement_version != 0,
            "Inkling stateful prefill owners/identity differ");
        let stream = &self.owners.first().context("Inkling prefill has no unit owners")?.stream;
        ensure!(self.owners.iter().all(|owner| std::rc::Rc::ptr_eq(&owner.stream, stream)),
            "Inkling causal state requires one ordered issuing stream");
        ensure!(units.iter().all(|unit| unit.slot < self.owners.len() && unit.layers == (0..self.cfg.layers)
            && members(unit).iter().all(|member| member.sequence < self.placements.len())),
            "Inkling prefill unit differs from admitted requests");
        let waves = self.owners.len();
        run_prefill_units(self, units, waves)
    }
    fn packed_members(&self, unit: &Unit) -> Result<Vec<super::owners::PackedMember>> {
        members(unit).into_iter().map(|member| {
            let placement = &self.placements[member.sequence];
            Ok(super::owners::PackedMember { sequence: placement.slot, rows: member.rows,
                metadata: cuteafd_loader::families::inkling2::execution::UnitMetadata::new(self.cfg,
                    member.tokens, &placement.units, self.cache.layout.pool_pages)? })
        }).collect()
    }
    fn join(&mut self, layer: usize, unit: &Unit) -> Result<()> {
        let owner = &self.owners[unit.slot];
        let members = self.packed_members(unit)?;
        let state = super::owners::PackedLayerUnit { unit: owner, cache: self.cache, layer, members: &members,
            uploads: std::cell::RefCell::new(&mut self.uploads[unit.slot]) };
        layers::join_experts(self.kernels, &state, self.target, self.cfg, layer, unit.rows.len())
    }
}

fn unit_tokens(unit: &Unit, sources: &[&[u32]], starts: &[usize]) -> Result<Vec<u32>> {
    let mut tokens = Vec::with_capacity(unit.rows.len());
    for member in members(unit) {
        let source = sources.get(member.sequence).context("Inkling unit token source absent")?;
        let origin = *starts.get(member.sequence).context("Inkling token source origin absent")?;
        let first = member.tokens.start.checked_sub(origin).context("Inkling token span precedes submitted suffix")?;
        let end = member.tokens.end.checked_sub(origin).context("Inkling token end precedes submitted suffix")?;
        tokens.extend_from_slice(source.get(first..end).context("Inkling unit token span outside request")?);
    }
    ensure!(tokens.len() == unit.rows.len(), "Inkling token members do not cover unit");
    Ok(tokens)
}

fn members(unit: &Unit) -> Vec<crate::shared::prefill_pipeline::UnitSegment> {
    if !unit.segments.is_empty() { return unit.segments.clone(); }
    vec![crate::shared::prefill_pipeline::UnitSegment { sequence: unit.sequence,
        rows: 0..unit.rows.len(), tokens: unit.rows.clone(), prompt_end: false, snapshot: false, head: false }]
}

impl PrefillHooks for Stateful<'_, '_> {
    type Ticket = crate::shared::spark_waves::WaveTicket;
    fn route(&self, layer: usize) -> FfnRoute {
        if layer < self.onboarded { FfnRoute::Local } else { FfnRoute::Spark }
    }
    fn begin_unit(&mut self, unit: &Unit) -> Result<()> {
        self.io.begin(unit, &self.owners[unit.slot], &mut self.uploads[unit.slot])
    }
    fn attention(&mut self, layer: usize, unit: &Unit) -> Result<()> {
        let owner = &self.owners[unit.slot];
        self.io.tap(layer,unit,owner)?;
        let members = self.packed_members(unit)?;
        owner.packed_metadata(layer, self.cache, &members, &mut self.uploads[unit.slot])?;
        let state = super::owners::PackedLayerUnit { unit: owner, cache: self.cache, layer, members: &members,
            uploads: std::cell::RefCell::new(&mut self.uploads[unit.slot]) };
        layers::begin_layer(self.kernels, &state, self.target, self.cfg, layer, unit.rows.len())
    }
    fn local_ffn(&mut self, layer: usize, unit: &Unit) -> Result<()> {
        let owner = &self.owners[unit.slot];
        layers::routed_ffn(self.kernels, owner, self.target, self.local[layer].as_ref(), self.cfg, layer, unit.rows.len())?;
        self.join(layer, unit)
    }
    fn send_experts(&mut self, layer: usize, unit: &Unit) -> Result<Self::Ticket> {
        use crate::shared::{experts::staging::RouteStaging, spark_waves};
        use cuteafd_transport::ExpertV2SourceKind;
        let owner = &self.owners[unit.slot];
        let rows = unit.rows.len();
        layers::prepare_routes(self.kernels, owner, self.target, self.cfg, layer, rows)?;
        let identity = *self.next_identity;
        ensure!(identity > 0, "Inkling Spark request identity is zero");
        *self.next_identity = identity.checked_add(1).context("Inkling Spark identity exhausted")?;
        let (source, start, version) = (unit.sequence as u64 + 1, unit.rows.start, self.placement_version);
        let dtype = match self.ingress[layer] {
            cuteafd_core::SparkIngress::Fp8K32 => cuteafd_transport::ExpertV2Dtype::Fp8E4m3Ue8m0K32,
            cuteafd_core::SparkIngress::Bf16 => cuteafd_transport::ExpertV2Dtype::Bf16,
            _ => bail!("Inkling layer ingress must name its native package input"),
        };
        let input = owner.buffer(if dtype == cuteafd_transport::ExpertV2Dtype::Bf16 { "projection_input" } else { "expert_wire" })?;
        let staging = RouteStaging::new(rows, self.cfg.topk, dtype.row_bytes(self.cfg.hidden)?)?;
        let kind = self.kind;
        // Explicit singleton members can carry a nonzero restored frontier;
        // preserve their request-local position too, not only multi-member packs.
        let origins = members(unit).iter().flat_map(|member| member.tokens.clone().map(move |position|
            (member.sequence as u64+1, position as u64))).collect::<Vec<_>>();
        let packed = unit.reply_origins()?;
        let combined = if kind == ExpertV2SourceKind::Prefill {
            packed.as_ref().map_or_else(|| spark_waves::single_reply_eligible(rows, kind), |origins| origins.combined_reply_eligible())
        } else { spark_waves::single_reply_eligible(rows, kind) };
        owner.stream.device.run(|| {
            // SAFETY: admitted unit owns routes/wire and the issuing stream;
            // shared SM staging finishes before the same unit reuses them.
            unsafe { self.waves.send(rows, rows*self.cfg.hidden*2, staging,
                [owner.buffer("route_ids")?, owner.buffer("route_weights")?, input],
                owner.stream.raw, combined, Box::new(move |routes, wire| {
                    let mut request = request(identity, version, layer, source, start, rows, dtype, routes, wire)?;
                    if let Some(packed) = packed { packed.apply(&mut request)?; }
                    ensure!(origins.len() == request.rows.len(), "Inkling Spark origin count differs");
                    for (row, &(source, position)) in request.rows.iter_mut().zip(&origins) {
                        row.source_request_id = source; row.token_position = position; row.source_kind = kind;
                    }
                    Ok(request)
                })) }
        })
    }
    fn land_experts(&mut self, layer: usize, unit: &Unit, ticket: Self::Ticket) -> Result<()> {
        let owner = &self.owners[unit.slot];
        let slot = ticket.slot();
        owner.stream.device.run(|| {
            self.waves.land(ticket, owner.stream.raw)?;
            // SAFETY: shared wave landing precedes reduction into the private
            // unit plane; the reducer also releases the shared return credit.
            unsafe { self.waves.reduce(slot, std::ptr::null(), owner.buffer("expert_output")?.ptr.cast(),
                unit.rows.len(), owner.stream.raw) }
        })?;
        self.join(layer, unit)
    }
    fn end_unit(&mut self, unit: &Unit) -> Result<()> { self.io.end(unit, &self.owners[unit.slot]) }
    fn commit_unit(&mut self, unit: &Unit) -> Result<()> {
        for member in members(unit) {
            let placement = &mut self.placements[member.sequence];
            ensure!(placement.state_len == member.tokens.start, "Inkling causal commit is out of order");
            placement.state_len = member.tokens.end;
            placement.len = member.tokens.end;
        }
        Ok(())
    }
}

fn request(identity: u64, version: u64, layer: usize, source: u64, start: usize, rows: usize,
    dtype: cuteafd_transport::ExpertV2Dtype, routes: Vec<cuteafd_transport::ExpertProtocolV2RouteEntry>, wire: bytes::Bytes)
    -> Result<cuteafd_transport::ExpertProtocolV2Request> {
    use cuteafd_transport::{ExpertProtocolV2Request, ExpertProtocolV2RowDescriptor, ExpertV2SourceKind};
    ensure!(identity > 0 && version > 0 && source > 0 && layer < 48 && (1..=4096).contains(&rows),
        "Inkling Spark request geometry/identity differs");
    let descriptors = (0..rows).map(|row| ExpertProtocolV2RowDescriptor {
        row_id: (start+row) as u64, source_kind: ExpertV2SourceKind::Prefill, source_request_id: source,
        token_position: (start+row) as u64, route_offset: (row*8) as u32, route_count: 8,
    }).collect();
    let mut request = ExpertProtocolV2Request::new_bytes(identity, version, u32::try_from(layer)?, 4608,
        dtype, descriptors, routes, wire)?;
    request.header.flags |= cuteafd_transport::expert::EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
    Ok(request)
}

pub(crate) fn layer<'a>(kernels: &Kernels<'a>, owner: &impl UnitBuffers<'a>, target: &TargetWeights<'a>,
    experts: &dyn Dispatcher, cfg: &Inkling2Config, index: usize, rows: usize) -> Result<()> {
    struct Hooks<'b, 'a, O> {
        kernels: &'b Kernels<'a>, owner: &'b O, target: &'b TargetWeights<'a>,
        experts: &'b dyn Dispatcher, cfg: &'b Inkling2Config,
    }
    impl<'a, O: UnitBuffers<'a>> PrefillHooks for Hooks<'_, 'a, O> {
        type Ticket = ();
        fn route(&self, _: usize) -> FfnRoute { FfnRoute::Local }
        fn begin_unit(&mut self, _: &Unit) -> Result<()> { Ok(()) }
        fn attention(&mut self, layer: usize, unit: &Unit) -> Result<()> {
            layers::begin_layer(self.kernels, self.owner, self.target, self.cfg, layer, unit.rows.len())
        }
        fn local_ffn(&mut self, layer: usize, unit: &Unit) -> Result<()> {
            layers::finish_layer(self.kernels, self.owner, self.target, self.experts, self.cfg, layer, unit.rows.len())
        }
        fn send_experts(&mut self, _: usize, _: &Unit) -> Result<()> { bail!("cold Inkling golden has no Spark route") }
        fn land_experts(&mut self, _: usize, _: &Unit, _: ()) -> Result<()> { bail!("cold Inkling golden has no Spark ticket") }
        fn end_unit(&mut self, _: &Unit) -> Result<()> { Ok(()) }
        fn commit_unit(&mut self, _: &Unit) -> Result<()> { Ok(()) }
    }
    let mut units = sequence_units(rows, owner.layout().rows.next_multiple_of(64), 1, cfg.layers)?;
    for unit in &mut units { unit.layers = index..index+1; }
    run_prefill_units(&mut Hooks { kernels, owner, target, experts, cfg }, &units, 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_transport::{ExpertProtocolV2RouteEntry, ExpertV2Dtype};

    #[test]
    fn requests_keep_real_request_origins_and_exact_mxfp8_wire() -> Result<()> {
        let routes = (0..16).map(|index| ExpertProtocolV2RouteEntry {
            row_index: index/8, expert_id: index, gate_weight: 0.125,
        }).collect();
        let request = request(17, 4, 5, 83, 1087, 2, ExpertV2Dtype::Fp8E4m3Ue8m0K32, routes, vec![0; 2*4752].into())?;
        assert_eq!(request.header.hidden_dtype, ExpertV2Dtype::Fp8E4m3Ue8m0K32);
        assert_eq!(request.header.hidden_row_stride_bytes, 4752);
        assert_eq!(request.rows[0].source_request_id, 83);
        assert_eq!(request.rows[0].token_position, 1087);
        assert_eq!(request.rows[1].token_position, 1088);
        assert_eq!(request.rows[1].route_offset, 8);
        assert_eq!(request.rows[1].route_count, 8);
        assert_ne!(request.header.flags & cuteafd_transport::expert::EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16, 0);
        Ok(())
    }

    #[test]
    fn native_bf16_requests_preserve_unquantized_input_bytes() -> Result<()> {
        let routes = (0..16).map(|index| ExpertProtocolV2RouteEntry {
            row_index: index/8, expert_id: index, gate_weight: 0.125,
        }).collect();
        let wire = vec![0x3d; 2*4608*2];
        let request = request(17, 4, 0, 83, 1087, 2, ExpertV2Dtype::Bf16, routes, wire.clone().into())?;
        assert_eq!(request.header.hidden_dtype, ExpertV2Dtype::Bf16);
        assert_eq!(request.header.hidden_row_stride_bytes, 9216);
        assert_eq!(request.hidden_payload.as_ref(), wire.as_slice());
        Ok(())
    }

    #[test]
    fn packed_members_keep_request_local_frontiers_and_shared_wire_origins() -> Result<()> {
        use crate::shared::prefill_pipeline::UnitSegment;
        let segments = vec![
            UnitSegment { sequence: 2, rows: 0..2, tokens: 1087..1089, prompt_end: false, snapshot: false, head: false },
            UnitSegment { sequence: 7, rows: 2..5, tokens: 63..66, prompt_end: true, snapshot: true, head: true },
        ];
        let unit = Unit { index: 0, sequence: 2, segments: segments.clone(), rows: 0..5,
            layers: 0..48, slot: 0, barrier: false };
        assert_eq!(members(&unit), segments);
        let a = (0..1090).collect::<Vec<u32>>();
        let b = (1000..1100).collect::<Vec<u32>>();
        let sources: Vec<&[u32]> = vec![&[], &[], &a, &[], &[], &[], &[], &b];
        assert_eq!(unit_tokens(&unit, &sources, &[0; 8])?, [1087, 1088, 1063, 1064, 1065]);
        assert!(unit_tokens(&unit, &[&a], &[0]).is_err());
        let routes = (0..40).map(|route| ExpertProtocolV2RouteEntry {
            row_index: route/8, expert_id: route%384, gate_weight: 0.125 }).collect();
        let mut request = request(99, 3, 5, 3, 0, 5, ExpertV2Dtype::Fp8E4m3Ue8m0K32, routes, vec![0; 5*4752].into())?;
        unit.reply_origins()?.unwrap().apply(&mut request)?;
        assert_eq!(request.rows.iter().map(|row| row.source_request_id).collect::<Vec<_>>(), [3, 3, 8, 8, 8]);
        assert_eq!(request.rows.iter().map(|row| row.token_position).collect::<Vec<_>>(), [1087, 1088, 63, 64, 65]);
        let singleton = Unit { segments: vec![], rows: 63..66, ..unit };
        assert_eq!(members(&singleton)[0].rows, 0..3);
        assert_eq!(members(&singleton)[0].tokens, 63..66);
        Ok(())
    }

    #[test]
    fn persistent_units_use_shared_waves_without_host_waits_or_a_private_scheduler() {
        let source = include_str!("prefill.rs").split("#[cfg(test)]").next().unwrap();
        assert!(source.contains("impl PrefillHooks for Stateful"));
        assert!(source.contains("run_prefill_units(self, units, waves)"));
        assert!(source.contains("self.waves.send("));
        assert!(source.contains("self.waves.land("));
        assert!(source.contains("self.waves.reduce("));
        assert!(source.contains("layers::prepare_routes("));
        assert!(source.contains("layers::join_experts("));
        assert!(!source.contains("for layer in"));
        assert!(!source.contains("cuda_stream_synchronize"));
        assert!(!source.contains(".drain("));
    }
}
