//! One whole-rank request executor; the shared scheduler owns request policy.
use super::{cache::Placement, experts::Dispatcher, head::Head, layers::Kernels,
    owners::RequestOwners, prefill::{Stateful, TokenIo}, weights::TargetWeights};
use crate::shared::{prefill_pipeline::Unit, spark_intake::SparkLink, token_io::DeviceLogits};
use anyhow::{ensure, Result};
use cuteafd_loader::families::inkling2::Inkling2Config;

/// Keep transport ahead of every consumer's retirement. On failure the owner
/// releases device waits, drains consumers, then releases registered buffers.
pub(crate) struct Execution<'a> {
    pub link: SparkLink<'a>,
    pub requests: RequestOwners<'a>,
    pub head: Head<'a>,
    pub target: TargetWeights<'a>,
    pub kernels: Kernels<'a>,
    pub local: Vec<Box<dyn Dispatcher + 'a>>,
    pub cfg: Inkling2Config,
    next_identity: u64,
    terminal: bool,
}
impl<'a> Execution<'a> {
    pub fn new(link: SparkLink<'a>, requests: RequestOwners<'a>, head: Head<'a>,
        target: TargetWeights<'a>, kernels: Kernels<'a>, local: Vec<Box<dyn Dispatcher + 'a>>,
        cfg: Inkling2Config) -> Result<Self> {
        ensure!(target.rank == 0 && target.layers.iter().copied().eq(0..cfg.layers)
            && !requests.prefill.is_empty() && local.len() <= cfg.layers,
            "Inkling whole execution differs from admitted owners");
        Ok(Self { link, requests, head, target, kernels, local, cfg, next_identity: 1, terminal: false })
    }

    /// Tokens are complete request-local streams; unit members provide the
    /// causal spans. Every scratch head is consumed before another overwrites it.
    pub fn prefill(&mut self, placements: &mut [Placement], tokens: &[&[u32]], units: &[Unit],
        version: u64, on_logits: &mut dyn FnMut(usize, usize, DeviceLogits) -> Result<()>) -> Result<()> {
        ensure!(!self.terminal && placements.len() == tokens.len(), "Inkling execution terminal or request count differs");
        let slots = placements.iter().map(|placement| placement.slot).collect::<Vec<_>>();
        let result = (|| {
            let waves = self.link.exchange()?;
            let mut io = TokenIo { kernels: &self.kernels, head: &self.head, cache: &self.requests.cache,
                target: &self.target, cfg: &self.cfg, tokens, slots: &slots, on_logits };
            Stateful { kernels: &self.kernels, owners: &self.requests.prefill, cache: &self.requests.cache,
                target: &self.target, cfg: &self.cfg, placements, uploads: &mut self.requests.prefill_uploads,
                waves, onboarded: self.local.len(), local: &self.local, next_identity: &mut self.next_identity,
                placement_version: version, io: &mut io }.run(units)?;
            self.requests.decode.stream.drain()?;
            crate::shared::spark_waves::healthy()
        })();
        if result.is_err() { self.terminate(); }
        result
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
        let terminal = source.split("pub fn terminate(").nth(1).unwrap();
        assert!(terminal.find("terminal_quiesce").unwrap() < terminal.find("decode.stream.drain").unwrap());
        assert!(terminal.find("cache.drain").unwrap() < terminal.find("terminal_release").unwrap());
        assert!(!source.contains("for layer in"));
    }
}
