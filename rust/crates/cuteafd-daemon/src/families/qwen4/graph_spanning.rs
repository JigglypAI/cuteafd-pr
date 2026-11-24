//! Capture-site classification at c64f8cb386, including host issue order.
//!
//! TP2 residual hops enqueue their push before their wait outside replay
//! (engine.rs:3250,3314; shared/peer_split/hop.rs:321,335). OwnerTp2 enqueues
//! both expert halves and the peer partial publication before returning
//! (shared/experts/rtx/owner_tp2.rs:226-255), also outside replay. Thus no
//! peer is waiting for a producer that this thread only submits after capture.

use crate::shared::decode_graph::sites::{Scope, Table};

// (visitor site, scope, capture/dependency evidence).
pub(super) const SITES: &Table = &[
    ("qwen4.segment", Scope::Local, "engine.rs:2917 local embedding/enter/attention/post_pre; :2961 local head; :3118 TP1 local segment; hops and experts outside replay at :2905,:2942,:2953"),
];

/// An unclassified site is spanning: it stays prewarmed and sealed (#2303).
pub(super) fn spans(name: &str) -> bool {
    crate::shared::decode_graph::sites::spans(SITES, name, false)
}

#[cfg(test)]
mod tests {
    #[test]
    fn every_qwen4_site_is_classified_and_unknown_sites_span() {
        crate::shared::decode_graph::sites::testing::assert_classified("qwen4", super::SITES);
        assert!(super::spans("new-unreviewed-site"));
    }

    #[test]
    fn graph_prewarm_off_warms_spanning_and_defers_local() {
        let off = |name, declared| crate::shared::decode_graph::sites::testing::assert_off_transitions(super::SITES, name, declared);
        // Rank-local on one and two GPUs: TP2 hops and experts run outside replay.
        assert!(!off("qwen4.segment", false));
    }
}
