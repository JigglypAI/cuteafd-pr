//! Capture-site classification at c64f8cb386, including host issue order.
//!
//! Heads issue both local attention ranks before either exchange publication
//! (layers.rs:931-937). Split FFNs likewise precede publish_ffn (:1049,1105,1119).
//! Route transfers publish before waiting (:880-884); RTX reduction finishes
//! enqueueing before local FFN (:1101-1105). Spark request copies precede the
//! local FFN, with reply waits only in finish (spark.rs:177-185,209). None of
//! these waits needs a producer issued after a local capture on this thread.
//! Context layouts retain their conservative seal even for ordinary steps:
//! context_pair_with queues both context protocols before producer joins
//! (layers.rs:326,1008); their peer flag waits must never be lazily captured.

use crate::shared::decode_graph::sites::{Scope, Table};

// (visitor site, intrinsic scope, capture/dependency evidence). The declared
// dependency is the context layout.
pub(super) const SITES: &Table = &[
    ("kimi_k3.step", Scope::Declared, "device.rs:1140-1185 local residual/attention; :2006-2021 local FFN; layers.rs:931,1049,1105 issue local steps before publication"),
    ("kimi_k3.context", Scope::Spanning, "device.rs:1409-1445 context_step captures peer flags via shared/peer_split/attention.rs context_decode; both rank protocols required"),
];

/// An unclassified site is spanning: it stays prewarmed and sealed (#2303).
pub(super) fn spans(name: &str, context: bool) -> bool {
    crate::shared::decode_graph::sites::spans(SITES, name, context)
}

#[cfg(test)]
mod tests {
    #[test]
    fn every_kimi_site_is_classified_and_context_stays_sealed() {
        crate::shared::decode_graph::sites::testing::assert_classified("kimi_k3", super::SITES);
        assert!(super::spans("kimi_k3.step", true) && !super::spans("kimi_k3.step", false));
        assert!(super::spans("kimi_k3.context", false));
        assert!(super::spans("new-unreviewed-site", false));
    }

    #[test]
    fn graph_prewarm_off_warms_spanning_and_defers_local() {
        let off = |name, declared| crate::shared::decode_graph::sites::testing::assert_off_transitions(super::SITES, name, declared);
        assert!(!off("kimi_k3.step", false), "Kimi heads/1-GPU step banks capture lazily");
        assert!(off("kimi_k3.step", true), "Kimi context layout seals its step banks");
        assert!(off("kimi_k3.context", false), "Kimi context banks always prewarm");
    }
}
