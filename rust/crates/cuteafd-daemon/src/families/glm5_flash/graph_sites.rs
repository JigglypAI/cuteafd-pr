#[cfg(test)]
use crate::shared::decode_graph::sites::{Scope, Table};

#[cfg(test)]
const SITES: &Table = &[
    ("glm5_flash.decode_local", Scope::Local, "engine/local_graph.rs:137 decode_graphed_local keeps every PeerExchange and ContextExchange push/wait outside replay_on; both ranks push before either waits"),
];

#[cfg(test)]
fn spans(name: &str, declared: bool) -> bool {
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
        assert_eq!(off("glm5_flash.decode_local", false), false, "glm5_flash exact-row local site (no peer dependency inside capture)");
    }
}
