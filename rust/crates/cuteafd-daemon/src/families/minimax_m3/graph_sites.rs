//! Graph capture-site classification (#2303). An unclassified site is
//! spanning: it stays prewarmed under `GRAPH_PREWARM=off` and sealed after
//! readiness (#1878). Evidence line numbers refer to dev 672a7e25f1.
use crate::shared::decode_graph::sites::{Scope, Table};

// (site, scope, capture and host-issue-order evidence).
pub(super) const SITES: &Table = &[
    ("minimax_m3.section", Scope::Declared, "engine.rs:365-381 split Ffn captures a peer copy + peer_push_signal; :400-414 JoinRouted signals peers; Router (owners.rs:278) and heads sections (engine.rs:1142,1149,1208) are local; one bank per rank, declared by owners.rs:283 split()"),
];

pub(super) fn spans(name: &str, declared: bool) -> bool {
    crate::shared::decode_graph::sites::spans(SITES, name, declared)
}

#[cfg(test)]
mod tests {
    #[test]
    fn every_site_is_classified_and_unknown_sites_span() {
        crate::shared::decode_graph::sites::testing::assert_classified("minimax_m3", super::SITES);
        assert!(super::spans("new-unreviewed-site", false));
    }

    #[test]
    fn graph_prewarm_off_warms_spanning_and_defers_local() {
        let off = |name, declared| crate::shared::decode_graph::sites::testing::assert_off_transitions(super::SITES, name, declared);
        assert_eq!(off("minimax_m3.section", false), false, "minimax_m3 one-GPU layout");
        assert_eq!(off("minimax_m3.section", true), true, "minimax_m3 two-GPU layout");
    }
}
