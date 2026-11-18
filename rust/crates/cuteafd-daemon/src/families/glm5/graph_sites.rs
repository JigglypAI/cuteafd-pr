//! Graph capture-site classification (#2303). An unclassified site is
//! spanning: it stays prewarmed under `GRAPH_PREWARM=off` and sealed after
//! readiness (#1878). Evidence line numbers refer to dev 672a7e25f1.
use crate::shared::decode_graph::sites::{Scope, Table};

// (site, scope, capture and host-issue-order evidence).
pub(super) const SITES: &Table = &[
    ("glm5.segment", Scope::Declared, "engine.rs:2272,2286 exchange.wait(1, ..) inside the peer segment captured by replay_on(1) at :2306; rank 0 waits at :2183 inside replay_on(0); one rank has no peer"),
];

pub(super) fn spans(name: &str, declared: bool) -> bool {
    crate::shared::decode_graph::sites::spans(SITES, name, declared)
}

#[cfg(test)]
mod tests {
    #[test]
    fn every_site_is_classified_and_unknown_sites_span() {
        crate::shared::decode_graph::sites::testing::assert_classified("glm5", super::SITES);
        assert!(super::spans("new-unreviewed-site", false));
    }

    #[test]
    fn graph_prewarm_off_warms_spanning_and_defers_local() {
        let off = |name, declared| crate::shared::decode_graph::sites::testing::assert_off_transitions(super::SITES, name, declared);
        assert_eq!(off("glm5.segment", false), false, "glm5 one-GPU layout");
        assert_eq!(off("glm5.segment", true), true, "glm5 two-GPU layout");
    }
}
