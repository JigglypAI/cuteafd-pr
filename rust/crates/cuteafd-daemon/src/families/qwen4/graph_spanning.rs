//! Capture-site classification at c64f8cb386, including host issue order.
//!
//! TP2 residual hops enqueue their push before their wait outside replay
//! (engine.rs:3250,3314; shared/peer_split/hop.rs:321,335). OwnerTp2 enqueues
//! both expert halves and the peer partial publication before returning
//! (shared/experts/rtx/owner_tp2.rs:226-255), also outside replay. Thus no
//! peer is waiting for a producer that this thread only submits after capture.

// (visitor site, spans devices, capture/dependency evidence).
pub(super) const SITES: &[(&str, bool, &str)] = &[
    ("qwen4.segment", false, "engine.rs:3263-3286 local embedding/enter/attention/post_pre; :3322 local head; :3484 TP1 local segment; hops and experts outside replay at :3250,:3299,:3314"),
];

pub(super) fn spans(name: &str) -> anyhow::Result<bool> {
    SITES.iter().find(|(site, _, _)| *site == name).map(|(_, spans, _)| *spans)
        .ok_or_else(|| anyhow::anyhow!("unclassified Qwen4 graph site {name}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_site_has_evidence_and_unknown_sites_fail_closed() {
        let mut names = std::collections::HashSet::new();
        for &(name, expected, evidence) in SITES {
            assert!(names.insert(name), "duplicate site {name}");
            assert!(evidence.contains(".rs:"), "missing source evidence {name}");
            assert_eq!(spans(name).unwrap(), expected);
        }
        assert!(spans("new-unreviewed-site").is_err());
    }
}
