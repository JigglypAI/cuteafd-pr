//! Single-rank and head-parallel arithmetic on the shared request scheduler.
use super::{cache::Placement, experts::Dispatcher, head::Head, layers::Kernels,
    owners::RequestOwners, prefill::{Stateful, TokenIo}, weights::TargetWeights};
use crate::shared::{prefill_pipeline::Unit, spark_intake::SparkLink, token_io::DeviceLogits};
use anyhow::{ensure, Result};
use cuteafd_loader::families::inkling2::Inkling2Config;

/// Keep transport ahead of every consumer's retirement. On failure the owner
/// releases device waits, drains consumers, then releases registered buffers.
pub(crate) struct DecodeGraphs<'a> {
    bank: crate::shared::decode_graph::GraphBank<(usize, usize), crate::shared::decode_graph::GraphOwner<'a, ()>>,
    stream: std::rc::Rc<crate::shared::memory::device::Stream<'a>>,
}
impl<'a> DecodeGraphs<'a> {
    pub fn new(requests: &RequestOwners<'a>, reserve: u64) -> Result<Self> {
        use crate::shared::decode_graph::{GraphBank, device::{Basis, Cycle, Preference, Site}};
        let stream = requests.decode.stream.clone();
        let mut bank = GraphBank::with_family("inkling2");
        bank.attach_device(stream.device.library, stream.device.id, reserve,
            Site { name: "inkling2.heads.output", cycle: Cycle::Decode, preference: Preference::Capture,
                bootstrap: 2 << 20, basis: Basis::Bytes, spans_devices: false })?;
        Ok(Self { bank, stream })
    }
    pub fn project(&mut self, layer: usize, rows: usize, work: impl FnOnce() -> Result<()>) -> Result<()> {
        if let Some(owner) = self.bank.cached_launch(&(layer, rows)) {
            // SAFETY: startup captured this same issuing stream, fixed unit
            // planes and immutable layer weights. No mutable cache is captured.
            unsafe { crate::shared::decode_graph::launch(self.stream.device.library, owner.raw, self.stream.raw) }
        } else { work() }
    }
    fn prewarm(&mut self, rank: usize, requests: &RequestOwners<'a>, kernels: &Kernels<'a>,
        target: &TargetWeights<'a>, cfg: &Inkling2Config) -> Result<()> {
        use crate::shared::decode_graph::{self, GraphOwner};
        if !decode_graph::prewarm_enabled() { return Ok(()); }
        let unit = &requests.decode;
        let stream = &self.stream;
        stream.drain()?;
        for rows in [1, 4, 16, 64] {
            for layer in 0..cfg.layers {
                let key = (layer, rows);
                let Some(mut ticket) = self.bank.cache_ticket(&key, false, None) else { continue; };
                stream.device.run(|| {
                    let lib = stream.device.library;
                    let work = || super::layers::project_heads(kernels, unit, target, cfg, layer, rows);
                    // SAFETY: startup has no request/peer work in flight. This
                    // records only the private epilogue on permanent buffers.
                    if let Err(error) = unsafe { decode_graph::begin_capture(lib, stream.raw, &ticket) } {
                        self.bank.cache_ticket_failed(ticket, &error);
                        return Ok(());
                    }
                    let executed = work();
                    // SAFETY: close the owned capture even when recording failed.
                    let captured = unsafe { decode_graph::end_capture(lib, stream.raw, &mut ticket) };
                    if let Err(error) = executed {
                        if let Ok((raw, ..)) = captured {
                            // SAFETY: recording failed; executable never launched.
                            unsafe { lib.cuda_graph_exec_destroy(raw)?; }
                        }
                        return Err(error);
                    }
                    let (raw, nodes, width, spans, inspection) = match captured {
                        Ok(result) => result,
                        Err(error) => { self.bank.cache_ticket_failed(ticket, &error); return Ok(()); }
                    };
                    // SAFETY: executable uses only the persistent decode unit
                    // and target arena; the executor drains before dropping them.
                    let owner = unsafe { GraphOwner::new(lib, stream.device.id, raw, ())? };
                    if nodes > 1 && !spans && inspection.is_none() {
                        self.bank.census_capture_key("heads.output", rank as i32, None, &key, raw);
                        self.bank.cache_commit(key, owner, ticket, nodes as u64, width, false);
                    }
                    Ok(())
                })?;
                stream.drain()?;
                // SAFETY: the same persistent decode owner outlives all graphs.
                unsafe { crate::shared::decode_graph::device::observe_completed(stream.device.library, |visit|
                    visit(stream.device.id, "inkling2.heads.output", false, &mut self.bank))?; }
            }
        }
        Ok(())
    }
}

pub(crate) struct Peer<'a> {
    pub graphs: DecodeGraphs<'a>,
    pub exchange: crate::shared::peer_split::PeerExchange<'a>,
    pub requests: RequestOwners<'a>,
    pub kernels: Kernels<'a>,
    pub target: TargetWeights<'a>,
}
impl<'a> Peer<'a> {
    fn units(&mut self, decode: bool) -> (super::head::PeerHead<'_, 'a>, super::prefill::PeerUnits<'_, 'a>) {
        let (owners, uploads) = if decode {
            (std::slice::from_ref(&self.requests.decode), std::slice::from_mut(&mut self.requests.decode_uploads))
        } else {
            (self.requests.prefill.as_slice(), self.requests.prefill_uploads.as_mut_slice())
        };
        (super::head::PeerHead { kernels: &self.kernels, target: &self.target },
        super::prefill::PeerUnits { exchange: &self.exchange, cache: &self.requests.cache,
            target: &self.target, kernels: &self.kernels, owners, uploads,
            graphs: decode.then_some(&mut self.graphs) })
    }
}

pub(crate) struct Execution<'a> {
    graphs: Option<DecodeGraphs<'a>>,
    pub link: SparkLink<'a>,
    pub requests: RequestOwners<'a>,
    pub peer: Option<Peer<'a>>,
    pub head: Head<'a>,
    pub target: TargetWeights<'a>,
    pub kernels: Kernels<'a>,
    pub local: Vec<Box<dyn Dispatcher + 'a>>,
    pub cfg: Inkling2Config,
    pub draft:Option<std::rc::Rc<super::draft::Binding<'a>>>,
    pub width_trace:Option<std::cell::RefCell<super::width_invariance::Trace>>,
    ingress: Vec<cuteafd_core::SparkIngress>,
    next_identity: u64,
    terminal: bool,
}
impl<'a> Execution<'a> {
    pub fn request_ranks(&self) -> impl Iterator<Item=&RequestOwners<'a>> {
        std::iter::once(&self.requests).chain(self.peer.as_ref().map(|peer|&peer.requests))
    }
    pub fn restore_verify(&self,mark:&super::verify::TargetMark<'_>)->Result<()> {
        let draft=self.draft.as_ref().ok_or_else(||anyhow::anyhow!("Inkling drafter missing"))?;
        ensure!(mark.conv.len()==draft.verify.len(),"Inkling verify mark rank count differs");
        for ((verify,mark),requests) in draft.verify.iter().zip(&mark.conv).zip(self.request_ranks()) {
            requests.decode.stream.device.run(||verify.restore(mark,requests.cache.conv_backing()))?;
        }
        Ok(())
    }
    pub fn require_live(&self) -> Result<()> { ensure!(!self.terminal, "Inkling execution is terminal"); Ok(()) }
    pub fn new(link: SparkLink<'a>, requests: RequestOwners<'a>, head: Head<'a>,
        target: TargetWeights<'a>, kernels: Kernels<'a>, local: Vec<Box<dyn Dispatcher + 'a>>,
        cfg: Inkling2Config, graph_reserve: u64, ingress: Vec<cuteafd_core::SparkIngress>) -> Result<Self> {
        ensure!(target.rank == 0 && target.layers.iter().copied().eq(0..cfg.layers)
            && !requests.prefill.is_empty() && local.len() <= cfg.layers && ingress.len() == cfg.layers,
            "Inkling whole execution differs from admitted owners");
        let graphs = (requests.decode.layout.head_parts == 2).then(|| DecodeGraphs::new(&requests, graph_reserve)).transpose()?;
        Ok(Self { graphs, link, requests, peer: None, head, target, kernels, local, cfg, ingress, next_identity: 1, terminal: false, draft:None, width_trace:None })
    }

    /// Tokens are submitted suffixes; placement frontiers define their origins
    /// while unit members retain absolute request-local causal spans. Every scratch head is consumed before another overwrites it.
    pub fn prefill(&mut self, placements: &mut [Placement], tokens: &[&[u32]], units: &[Unit],
        version: u64, on_logits: &mut dyn FnMut(usize, usize, DeviceLogits) -> Result<()>) -> Result<()> {
        ensure!(!self.terminal && placements.len() == tokens.len(), "Inkling execution terminal or request count differs");
        let slots = placements.iter().map(|placement| placement.slot).collect::<Vec<_>>();
        let starts = placements.iter().map(|placement| placement.len).collect::<Vec<_>>();
        let media = placements.iter().map(|placement| placement.media.clone()).collect::<Vec<_>>();
        let result = (|| {
            let waves = self.link.exchange()?;
            let (peer_head, peer_units) = self.peer.as_mut().map(|peer| peer.units(false)).unzip();
            let mut io = TokenIo { kernels: &self.kernels, head: &self.head, peer_head, cache: &self.requests.cache,
                target: &self.target, cfg: &self.cfg, tokens, media: &media, starts: &starts, slots: &slots, retained_rows: false, draft:self.draft.as_deref(),publish_draft:true,trace:None,on_logits };
            Stateful { graphs: None, peer: peer_units, kernels: &self.kernels, owners: &self.requests.prefill, cache: &self.requests.cache,
                target: &self.target, cfg: &self.cfg, ingress: &self.ingress, placements, uploads: &mut self.requests.prefill_uploads,
                waves, onboarded: self.local.len(), local: &self.local, next_identity: &mut self.next_identity,
                placement_version: version, kind: cuteafd_transport::ExpertV2SourceKind::Prefill, io: &mut io }.run(units)?;
            self.requests.decode.stream.drain()?;
            crate::shared::spark_waves::healthy()?;
            self.observe_completed()
        })();
        if result.is_err() { self.terminate(); }
        result
    }
    /// Eager non-speculative decode uses the admitted decode owner and shared
    /// unit executor. Each member's output row remains in the one bounded head.
    pub fn decode(&mut self, placements: &mut [Placement], tokens: &[&[u32]], unit: &Unit,
        version: u64) -> Result<DeviceLogits> {
        ensure!(!self.terminal && !placements.is_empty() && placements.len() == tokens.len()
            && unit.rows.len() <= 64 && unit.slot == 0, "Inkling decode exceeds admitted head/owner");
        let slots = placements.iter().map(|placement| placement.slot).collect::<Vec<_>>();
        let starts = placements.iter().map(|placement| placement.len).collect::<Vec<_>>();
        let media = placements.iter().map(|placement| placement.media.clone()).collect::<Vec<_>>();
        let result = (|| {
            let waves = self.link.exchange()?;
            let (peer_head, peer_units) = self.peer.as_mut().map(|peer| peer.units(true)).unzip();
            let mut io = TokenIo { kernels: &self.kernels, head: &self.head, peer_head, cache: &self.requests.cache,
                target: &self.target, cfg: &self.cfg, tokens, media: &media, starts: &starts, slots: &slots, retained_rows: true, draft:self.draft.as_deref(),publish_draft:false,trace:self.width_trace.as_ref(),
                on_logits: &mut |_, _, _| Ok(()) };
            Stateful { graphs: self.graphs.as_mut(), peer: peer_units, kernels: &self.kernels, owners: std::slice::from_ref(&self.requests.decode),
                cache: &self.requests.cache, target: &self.target, cfg: &self.cfg, ingress: &self.ingress, placements,
                uploads: std::slice::from_mut(&mut self.requests.decode_uploads), waves,
                onboarded: self.local.len(), local: &self.local, next_identity: &mut self.next_identity,
                placement_version: version, kind: cuteafd_transport::ExpertV2SourceKind::Decode, io: &mut io }.run(std::slice::from_ref(unit))?;
            self.requests.decode.stream.drain()?;
            crate::shared::spark_waves::healthy()?;
            self.observe_completed()?;
            Ok(DeviceLogits { ptr: self.head.logits.buffer.ptr, rows: unit.rows.len(),
                vocab: self.cfg.unpadded_vocab_size, stride: self.cfg.unpadded_vocab_size,
                stream: self.requests.decode.stream.raw, greedy: None })
        })();
        if result.is_err() { self.terminate(); }
        result
    }
    pub fn prewarm(&mut self) -> Result<()> {
        if let Some(graphs) = &mut self.graphs { graphs.prewarm(0, &self.requests, &self.kernels, &self.target, &self.cfg)?; }
        if let Some(peer) = &mut self.peer { peer.graphs.prewarm(1, &peer.requests, &peer.kernels, &peer.target, &self.cfg)?; }
        Ok(())
    }
    fn observe_completed(&mut self) -> Result<()> {
        let stream = &self.requests.decode.stream;
        // SAFETY: completed lead work consumes all peer partials. Both banks
        // retain only graph executables over persistent rank-local storage.
        unsafe { crate::shared::decode_graph::device::observe_completed(stream.device.library, |visit| {
            if let Some(graphs) = &mut self.graphs { visit(stream.device.id, "inkling2.heads.output", false, &mut graphs.bank)?; }
            if let Some(peer) = &mut self.peer { visit(peer.requests.decode.stream.device.id, "inkling2.heads.output", false, &mut peer.graphs.bank)?; }
            Ok(())
        }) }
    }
    pub fn terminate(&mut self) {
        if self.terminal { return; }
        self.terminal = true;
        let drained = (|| {
            if let Some(peer) = &self.peer { peer.exchange.publish_abort()?; }
            self.link.terminal_quiesce()?;
            if let Some(peer) = &self.peer {
                peer.exchange.drain_compute()?;
                peer.requests.cache.drain()?;
            }
            self.requests.decode.stream.drain()?;
            for owner in &self.requests.prefill { owner.stream.drain()?; }
            self.requests.cache.drain()?;
            self.link.terminal_drain_copies()?;
            self.link.terminal_release()?;
            if let Some(peer) = &self.peer { peer.exchange.finish_terminal(true); }
            Ok(())
        })();
        crate::shared::decode_graph::fatal_drain(drained, "Inkling request executor retirement");
        if let Some(graphs) = &mut self.graphs { graphs.bank.cache_clear_drained(); }
        if let Some(peer) = &mut self.peer { peer.graphs.bank.cache_clear_drained(); }
    }
}
impl Drop for Execution<'_> {
    fn drop(&mut self) { self.terminate(); }
}

#[cfg(test)]
mod tests {
    #[test]
    fn request_executor_reuses_shared_causal_units_and_terminal_transport_order() {
        let source = include_str!("engine.rs").split("#[cfg(test)]").next().unwrap();
        assert!(source.contains("Stateful {"));
        assert!(source.contains(".run(units)"));
        assert!(source.contains("self.link.exchange()?"));
        assert!(source.contains("bank.attach_device("));
        assert!(source.contains("spans_devices: false"));
        assert!(source.contains("device::observe_completed("));
        assert!(!source.split("pub fn decode(").nth(1).unwrap().contains("begin_capture("));
        let terminal = source.split("pub fn terminate(").nth(1).unwrap();
        assert!(terminal.find("terminal_quiesce").unwrap() < terminal.find("decode.stream.drain").unwrap());
        assert!(terminal.find("cache.drain").unwrap() < terminal.find("terminal_release").unwrap());
        assert!(!source.split("pub fn decode(").nth(1).unwrap().contains("for layer in"));
    }
}
