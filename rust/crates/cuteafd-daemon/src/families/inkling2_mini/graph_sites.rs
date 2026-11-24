//! Graph capture-site classification for Inkling 2 Mini (#2303).
use crate::shared::decode_graph::sites::{Scope, Table};

pub(super) const SITES: &Table = &[(
    "inkling2_mini.step",
    Scope::Local,
    "prefill.rs: monolithic 48-visit TP1 kernel-only decode/verify graph (attention, routes, local experts, draft taps); metadata uploads stay outside",
)];

pub(super) fn spans(name: &str, declared: bool) -> bool {
    crate::shared::decode_graph::sites::spans(SITES, name, declared)
}

#[cfg(test)]
mod tests {
    #[test]
    fn mini_site_is_classified_local_and_unknown_spans() {
        crate::shared::decode_graph::sites::testing::assert_classified("inkling2_mini", super::SITES);
        assert!(super::spans("new-unreviewed-site", false));
        assert!(!super::spans("inkling2_mini.step", false));
    }
}
