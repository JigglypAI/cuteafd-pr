//! Graph capture-site classification (#2303). An unclassified site is
//! spanning: it stays prewarmed under `GRAPH_PREWARM=off` and sealed after
//! readiness (#1878). Evidence line numbers refer to dev 672a7e25f1.
use crate::shared::decode_graph::sites::{Scope, Table};

// (site, scope, capture and host-issue-order evidence).
pub(super) const SITES: &Table = &[
    ("mimo_v2.segment", Scope::Declared, "engine.rs:2228 graph(1, ..) captures wait(1, DIRECT) :2234 and push/wait(1, attended) :2249-2250; one rank has no peer"),
];

pub(super) fn spans(name: &str, declared: bool) -> bool {
    crate::shared::decode_graph::sites::spans(SITES, name, declared)
}

#[cfg(test)]
mod tests {
    #[test]
    fn every_site_is_classified_and_unknown_sites_span() {
        crate::shared::decode_graph::sites::testing::assert_classified("mimo_v2", super::SITES);
        assert!(super::spans("new-unreviewed-site", false));
    }

    #[test]
    fn graph_prewarm_off_warms_spanning_and_defers_local() {
        let off = |name, declared| crate::shared::decode_graph::sites::testing::assert_off_transitions(super::SITES, name, declared);
        assert_eq!(off("mimo_v2.segment", false), false, "mimo_v2 one-GPU layout");
        assert_eq!(off("mimo_v2.segment", true), true, "mimo_v2 two-GPU layout");
    }
}
