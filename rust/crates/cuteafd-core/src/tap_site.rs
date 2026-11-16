//! Checkpoint-defined hidden-state sites, independent of layer numbering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TapSite {
    LayerExit(usize),
    LayerEntry(usize),
    FinalNorm,
}
impl TapSite {
    pub fn valid(self, layers: usize) -> bool {
        layers > 0 && match self { Self::LayerExit(layer) | Self::LayerEntry(layer) => layer < layers,
            Self::FinalNorm => true }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_sites_do_not_alias_the_last_layer_exit() {
        assert_ne!(TapSite::LayerExit(47), TapSite::LayerEntry(47));
        assert_ne!(TapSite::LayerExit(47), TapSite::FinalNorm);
        assert!(TapSite::FinalNorm.valid(48));
        assert!(!TapSite::LayerExit(48).valid(48));
        assert!(!TapSite::LayerEntry(48).valid(48));
        assert!(!TapSite::FinalNorm.valid(0));
    }
}
