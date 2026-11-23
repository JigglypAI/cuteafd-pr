//! Graph capture-site classification (#2303). An unclassified site is
//! spanning: it stays prewarmed under `GRAPH_PREWARM=off` and sealed after
//! readiness (#1878). Evidence line numbers refer to dev 672a7e25f1.
use crate::shared::decode_graph::sites::{Scope, Table};

// (site, scope, capture and host-issue-order evidence).
pub(super) const SITES: &Table = &[
    ("glm5_flash.decode", Scope::Declared, "engine.rs:4128-4131 replay_on(1) captures peer_post: exchange_wait + recv at :4116-4117; spans with two ranks or a head-split peer"),
    ("glm5_flash.decode_local", Scope::Local, "engine/local_graph.rs:143 decode_graphed_local keeps every PeerExchange and ContextExchange push/wait outside replay_on; both ranks push before either waits"),
];

pub(super) fn spans(name: &str, declared: bool) -> bool {
    crate::shared::decode_graph::sites::spans(SITES, name, declared)
}

#[cfg(test)]
mod tests {
    #[test]
    fn every_site_is_classified_and_unknown_sites_span() {
        crate::shared::decode_graph::sites::testing::assert_classified("glm5_flash", super::SITES);
        assert!(super::spans("new-unreviewed-site", false));
    }

    #[test]
    fn graph_prewarm_off_warms_spanning_and_defers_local() {
        let off = |name, declared| crate::shared::decode_graph::sites::testing::assert_off_transitions(super::SITES, name, declared);
        assert_eq!(off("glm5_flash.decode", false), false, "glm5_flash one-GPU layout");
        assert_eq!(off("glm5_flash.decode", true), true, "glm5_flash two-GPU layout");
        assert_eq!(off("glm5_flash.decode_local", false), false, "glm5_flash exact-row local site (no peer dependency inside capture)");
    }
}
