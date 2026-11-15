//! Family arithmetic on the shared causal unit executor; no private schedule.
use super::{experts::Dispatcher, layers::{self, Kernels}, owners::UnitBuffers, weights::TargetWeights};
use crate::shared::prefill_pipeline::{run_prefill_units, sequence_units, FfnRoute, PrefillHooks, Unit};
use anyhow::{bail, ensure, Context, Result};
use cuteafd_loader::families::inkling2::Inkling2Config;

/// Persistent whole-rank prefill over shared Spark waves. Each in-flight unit
/// has an admitted scratch owner; global pages and causal histories are shared
/// request owners, ordered by run_prefill_units at every layer.
pub(crate) trait UnitIo<'a> {
    fn begin(&mut self, unit: &Unit, owner: &super::owners::UnitOwner<'a>) -> Result<()>;
    fn end(&mut self, unit: &Unit, owner: &super::owners::UnitOwner<'a>) -> Result<()>;
}
pub(crate) struct Stateful<'s, 'a> {
    pub kernels: &'s Kernels<'a>,
    pub owners: &'s [super::owners::UnitOwner<'a>],
    pub cache: &'s super::cache::CacheBank<'a>,
    pub target: &'s TargetWeights<'a>,
    pub cfg: &'s Inkling2Config,
    pub placements: &'s mut [super::cache::Placement],
    pub uploads: &'s mut [crate::shared::token_table::RetainedUploads<'a>],
    pub waves: &'s mut crate::shared::spark_waves::Waves<'a>,
    pub onboarded: usize,
    pub local: &'s [Box<dyn Dispatcher + 'a>],
    pub next_identity: &'s mut u64,
    pub placement_version: u64,
    pub io: &'s mut dyn UnitIo<'a>,
}
impl Stateful<'_, '_> {
    pub fn run(&mut self, units: &[Unit]) -> Result<()> {
        ensure!(!units.is_empty() && self.owners.len() == self.uploads.len()
            && self.owners.len() <= self.waves.waves() && self.onboarded == self.local.len()
            && self.onboarded <= self.cfg.layers && self.placement_version != 0,
            "Inkling stateful prefill owners/identity differ");
        ensure!(units.iter().all(|unit| unit.segments.is_empty() && unit.sequence < self.placements.len()
            && unit.slot < self.owners.len() && unit.layers == (0..self.cfg.layers)),
            "Inkling singleton prefill unit differs from admitted request");
        let waves = self.owners.len();
        run_prefill_units(self, units, waves)
    }
}
impl PrefillHooks for Stateful<'_, '_> {
    type Ticket = crate::shared::spark_waves::WaveTicket;
    fn route(&self, layer: usize) -> FfnRoute {
        if layer < self.onboarded { FfnRoute::Local } else { FfnRoute::Spark }
    }
    fn begin_unit(&mut self, unit: &Unit) -> Result<()> {
        let placement = &self.placements[unit.sequence];
        let metadata = cuteafd_loader::families::inkling2::execution::UnitMetadata::new(self.cfg,
            unit.rows.clone(), &placement.units, self.cache.layout.pool_pages)?;
        self.owners[unit.slot].metadata(&metadata, &mut self.uploads[unit.slot])?;
        self.io.begin(unit, &self.owners[unit.slot])
    }
    fn attention(&mut self, layer: usize, unit: &Unit) -> Result<()> {
        let state = self.owners[unit.slot].bind(self.cache, self.placements[unit.sequence].slot, layer)?;
        layers::begin_layer(self.kernels, &state, self.target, self.cfg, layer, unit.rows.len())
    }
    fn local_ffn(&mut self, layer: usize, unit: &Unit) -> Result<()> {
        let state = self.owners[unit.slot].bind(self.cache, self.placements[unit.sequence].slot, layer)?;
        layers::finish_layer(self.kernels, &state, self.target, self.local[layer].as_ref(), self.cfg, layer, unit.rows.len())
    }
    fn send_experts(&mut self, layer: usize, unit: &Unit) -> Result<Self::Ticket> {
        use crate::shared::{experts::staging::RouteStaging, spark_waves};
        use cuteafd_transport::ExpertV2SourceKind;
        let owner = &self.owners[unit.slot];
        let state = owner.bind(self.cache, self.placements[unit.sequence].slot, layer)?;
        let rows = unit.rows.len();
        layers::prepare_routes(self.kernels, &state, self.target, self.cfg, layer, rows)?;
        let identity = *self.next_identity;
        ensure!(identity > 0, "Inkling Spark request identity is zero");
        *self.next_identity = identity.checked_add(1).context("Inkling Spark identity exhausted")?;
        let (source, start, version) = (unit.sequence as u64 + 1, unit.rows.start, self.placement_version);
        let staging = RouteStaging::new(rows, self.cfg.topk, self.cfg.hidden + self.cfg.hidden/32)?;
        owner.stream.device.run(|| {
            // SAFETY: admitted unit owns routes/wire and the issuing stream;
            // shared SM staging finishes before the same unit reuses them.
            unsafe { self.waves.send(rows, rows*self.cfg.hidden*2, staging,
                [owner.buffer("route_ids")?, owner.buffer("route_weights")?, owner.buffer("expert_wire")?],
                owner.stream.raw, spark_waves::single_reply_eligible(rows, ExpertV2SourceKind::Prefill),
                Box::new(move |routes, wire| request(identity, version, layer, source, start, rows, routes, wire))) }
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
        let state = owner.bind(self.cache, self.placements[unit.sequence].slot, layer)?;
        layers::join_experts(self.kernels, &state, self.target, self.cfg, layer, unit.rows.len())
    }
    fn end_unit(&mut self, unit: &Unit) -> Result<()> { self.io.end(unit, &self.owners[unit.slot]) }
    fn commit_unit(&mut self, unit: &Unit) -> Result<()> {
        let placement = &mut self.placements[unit.sequence];
        ensure!(placement.state_len == unit.rows.start, "Inkling causal commit is out of order");
        placement.state_len = unit.rows.end;
        placement.len = unit.rows.end;
        Ok(())
    }
}

fn request(identity: u64, version: u64, layer: usize, source: u64, start: usize, rows: usize,
    routes: Vec<cuteafd_transport::ExpertProtocolV2RouteEntry>, wire: bytes::Bytes)
    -> Result<cuteafd_transport::ExpertProtocolV2Request> {
    use cuteafd_transport::{ExpertProtocolV2Request, ExpertProtocolV2RowDescriptor, ExpertV2Dtype, ExpertV2SourceKind};
    ensure!(identity > 0 && version > 0 && source > 0 && layer < 48 && (1..=4096).contains(&rows),
        "Inkling Spark request geometry/identity differs");
    let descriptors = (0..rows).map(|row| ExpertProtocolV2RowDescriptor {
        row_id: (start+row) as u64, source_kind: ExpertV2SourceKind::Prefill, source_request_id: source,
        token_position: (start+row) as u64, route_offset: (row*8) as u32, route_count: 8,
    }).collect();
    let mut request = ExpertProtocolV2Request::new_bytes(identity, version, u32::try_from(layer)?, 4608,
        ExpertV2Dtype::Fp8E4m3Ue8m0K32, descriptors, routes, wire)?;
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
