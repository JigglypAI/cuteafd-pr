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
