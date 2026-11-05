//! Exact aligned regions used by the coordinator's physical weight arenas.
use anyhow::{Context, Result};
use std::ops::Range;

#[derive(Debug, Clone)]
pub struct PackedWeightRegions {
    pub regions: Vec<Range<usize>>,
    pub bytes: usize,
}

impl PackedWeightRegions {
    pub fn new(sizes: impl IntoIterator<Item = usize>) -> Result<Self> {
        let mut regions = Vec::new();
        let mut bytes = 0usize;
        for size in sizes {
            bytes = bytes.checked_add(255).context("weight region alignment overflow")? & !255;
            let end = bytes.checked_add(size).context("weight region extent overflow")?;
            regions.push(bytes..end);
            bytes = end;
        }
        Ok(Self { regions, bytes })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn preserves_plane_order_alignment_and_final_extent() {
        let layout = PackedWeightRegions::new([3, 0, 257, 16]).unwrap();
        assert_eq!(layout.regions, [0..3, 256..256, 256..513, 768..784]);
        assert_eq!(layout.bytes, 784);
        assert_eq!(PackedWeightRegions::new([]).unwrap().bytes, 0);
        assert!(PackedWeightRegions::new([usize::MAX, 1]).is_err());
        assert!(PackedWeightRegions::new([256, usize::MAX]).is_err());
    }
}
