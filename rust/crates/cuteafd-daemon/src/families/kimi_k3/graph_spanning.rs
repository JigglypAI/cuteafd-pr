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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Scope { ContextLayout, Spanning }

// (visitor site, intrinsic scope, capture/dependency evidence).
pub(super) const SITES: &[(&str, Scope, &str)] = &[
    ("kimi_k3.step", Scope::ContextLayout, "device.rs:1140-1185 local residual/attention; :2006-2021 local FFN; layers.rs:931,1049,1105 issue local steps before publication"),
    ("kimi_k3.context", Scope::Spanning, "device.rs:1409-1445 context_step captures peer flags via shared/peer_split/attention.rs context_decode; both rank protocols required"),
];

pub(super) fn spans(name: &str, context: bool) -> anyhow::Result<bool> {
    let (_, scope, _) = SITES.iter().find(|(site, _, _)| *site == name)
        .ok_or_else(|| anyhow::anyhow!("unclassified Kimi graph site {name}"))?;
    Ok(context || *scope == Scope::Spanning)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_classification_keeps_context_sealed() {
        let mut names = std::collections::HashSet::new();
        for &(name, scope, evidence) in SITES {
            assert!(names.insert(name), "duplicate site {name}");
            assert!(evidence.contains(".rs:"), "missing source evidence {name}");
            assert_eq!(spans(name, false).unwrap(), scope == Scope::Spanning);
            assert!(spans(name, true).unwrap());
        }
        assert!(spans("new-unreviewed-site", false).is_err());
    }
}
