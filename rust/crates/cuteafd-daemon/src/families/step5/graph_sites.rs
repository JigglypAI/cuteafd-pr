//! Graph capture-site classification (#2303). An unclassified site is
//! spanning: it stays prewarmed under `GRAPH_PREWARM=off` and sealed after
//! readiness (#1878). Evidence line numbers refer to dev 672a7e25f1.
use crate::shared::decode_graph::sites::{Scope, Table};

// (site, scope, capture and host-issue-order evidence).
pub(super) const SITES: &Table = &[
    ("step5.section", Scope::Local, "execution.rs:1343 Shared section per rank; TP2 push_to/wait at :1398,:1402 and replicate_result :1631 are issued outside capture; csa.rs:852 and swa.rs:838 enqueue on the owner stream only"),
];

pub(super) fn spans(name: &str, declared: bool) -> bool {
    crate::shared::decode_graph::sites::spans(SITES, name, declared)
}

#[cfg(test)]
mod tests {
    #[test]
    fn every_site_is_classified_and_unknown_sites_span() {
        crate::shared::decode_graph::sites::testing::assert_classified("step5", super::SITES);
        assert!(super::spans("new-unreviewed-site", false));
    }

    #[test]
    fn graph_prewarm_off_warms_spanning_and_defers_local() {
        let off = |name, declared| crate::shared::decode_graph::sites::testing::assert_off_transitions(super::SITES, name, declared);
        // Rank-local on one and two GPUs: no layout flag promotes this site.
        assert_eq!(off("step5.section", false), false, "step5 one-GPU layout");
    }
}
