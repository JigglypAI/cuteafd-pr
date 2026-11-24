//! Graph capture-site classification for Inkling 2 Mini (#2303).
use crate::shared::decode_graph::sites::{Scope, Table};

pub(super) const SITES: &Table = &[(
    "inkling2_mini.section",
    Scope::Local,
    "prefill.rs: rank-local TP1 arithmetic per visit; packed metadata uploads stay outside captured sections",
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
        assert!(!super::spans("inkling2_mini.section", false));
    }
}
