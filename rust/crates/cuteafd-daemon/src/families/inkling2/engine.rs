//! One whole-rank request executor; the shared scheduler owns request policy.
use super::{cache::Placement, experts::Dispatcher, head::Head, layers::Kernels,
    owners::RequestOwners, prefill::{Stateful, TokenIo}, weights::TargetWeights};
use crate::shared::{prefill_pipeline::Unit, spark_intake::SparkLink, token_io::DeviceLogits};
use anyhow::{ensure, Result};
use cuteafd_loader::families::inkling2::Inkling2Config;

/// Keep transport ahead of every consumer's retirement. On failure the owner
/// releases device waits, drains consumers, then releases registered buffers.
pub(crate) struct Execution<'a> {
    graphs: crate::shared::decode_graph::GraphBank<(usize, usize), crate::shared::decode_graph::GraphOwner<'a, ()>>,
    pub link: SparkLink<'a>,
    pub requests: RequestOwners<'a>,
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
    pub fn require_live(&self) -> Result<()> { ensure!(!self.terminal, "Inkling execution is terminal"); Ok(()) }
    pub fn new(link: SparkLink<'a>, requests: RequestOwners<'a>, head: Head<'a>,
        target: TargetWeights<'a>, kernels: Kernels<'a>, local: Vec<Box<dyn Dispatcher + 'a>>,
        cfg: Inkling2Config, graph_reserve: u64, ingress: Vec<cuteafd_core::SparkIngress>) -> Result<Self> {
        ensure!(target.rank == 0 && target.layers.iter().copied().eq(0..cfg.layers)
            && !requests.prefill.is_empty() && local.len() <= cfg.layers && ingress.len() == cfg.layers,
            "Inkling whole execution differs from admitted owners");
        use crate::shared::decode_graph::{GraphBank, device::{Basis, Cycle, Preference, Site}};
        let mut graphs = GraphBank::with_family("inkling2");
        graphs.attach_device(requests.decode.stream.device.library, requests.decode.stream.device.id, graph_reserve,
            Site { name: "inkling2.whole", cycle: Cycle::Other, preference: Preference::Eager,
                bootstrap: 0, basis: Basis::Bytes, spans_devices: false })?;
        tracing::warn!(graph_reserve, "Inkling unqualified graph layout starts eager with no borrowed prewarm or pins");
        Ok(Self { graphs, link, requests, head, target, kernels, local, cfg, ingress, next_identity: 1, terminal: false, draft:None, width_trace:None })
    }

    /// Tokens are submitted suffixes; placement frontiers define their origins
    /// while unit members retain absolute request-local causal spans. Every scratch head is consumed before another overwrites it.
    pub fn prefill(&mut self, placements: &mut [Placement], tokens: &[&[u32]], units: &[Unit],
        version: u64, on_logits: &mut dyn FnMut(usize, usize, DeviceLogits) -> Result<()>) -> Result<()> {
        ensure!(!self.terminal && placements.len() == tokens.len(), "Inkling execution terminal or request count differs");
        self.graphs.cache_cycle(crate::shared::decode_graph::device::Cycle::Prefill);
        let slots = placements.iter().map(|placement| placement.slot).collect::<Vec<_>>();
        let starts = placements.iter().map(|placement| placement.len).collect::<Vec<_>>();
        let result = (|| {
            let waves = self.link.exchange()?;
            let mut io = TokenIo { kernels: &self.kernels, head: &self.head, cache: &self.requests.cache,
                target: &self.target, cfg: &self.cfg, tokens, starts: &starts, slots: &slots, retained_rows: false, draft:self.draft.as_deref(),publish_draft:true,trace:None,on_logits };
            Stateful { kernels: &self.kernels, owners: &self.requests.prefill, cache: &self.requests.cache,
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
        self.graphs.cache_cycle(crate::shared::decode_graph::device::Cycle::Decode);
        let slots = placements.iter().map(|placement| placement.slot).collect::<Vec<_>>();
        let starts = placements.iter().map(|placement| placement.len).collect::<Vec<_>>();
        let result = (|| {
            let waves = self.link.exchange()?;
            let mut io = TokenIo { kernels: &self.kernels, head: &self.head, cache: &self.requests.cache,
                target: &self.target, cfg: &self.cfg, tokens, starts: &starts, slots: &slots, retained_rows: true, draft:self.draft.as_deref(),publish_draft:false,trace:self.width_trace.as_ref(),
                on_logits: &mut |_, _, _| Ok(()) };
            Stateful { kernels: &self.kernels, owners: std::slice::from_ref(&self.requests.decode),
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
    fn observe_completed(&mut self) -> Result<()> {
        let stream = &self.requests.decode.stream;
        // SAFETY: the sole issuing stream completed at this outer boundary;
        // every bank owner and persistent request/weight view is still retained.
        unsafe { crate::shared::decode_graph::device::observe_completed(stream.device.library, |visit|
            visit(stream.device.id, "inkling2.whole", false, &mut self.graphs)) }
    }
    pub fn terminate(&mut self) {
        if self.terminal { return; }
        self.terminal = true;
        let drained = (|| {
            self.link.terminal_quiesce()?;
            self.requests.decode.stream.drain()?;
            for owner in &self.requests.prefill { owner.stream.drain()?; }
            self.requests.cache.drain()?;
            self.link.terminal_drain_copies()?;
            self.link.terminal_release()
        })();
        crate::shared::decode_graph::fatal_drain(drained, "Inkling request executor retirement");
        self.graphs.cache_clear_drained();
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
        assert!(source.contains("graphs.attach_device("));
        assert!(source.contains("preference: Preference::Eager"));
        assert!(source.contains("bootstrap: 0"));
        assert!(source.contains("device::observe_completed("));
        assert!(!source.contains("begin_capture("));
        let terminal = source.split("pub fn terminate(").nth(1).unwrap();
        assert!(terminal.find("terminal_quiesce").unwrap() < terminal.find("decode.stream.drain").unwrap());
        assert!(terminal.find("cache.drain").unwrap() < terminal.find("terminal_release").unwrap());
        assert!(!source.contains("for layer in"));
    }
}
