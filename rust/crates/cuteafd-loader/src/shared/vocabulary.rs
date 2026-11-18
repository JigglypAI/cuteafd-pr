//! Contiguous vocabulary partition preserving native output blocks.
use anyhow::{ensure, Context, Result};
use std::ops::Range;

pub fn shard(outputs: usize, align: usize, rank: usize) -> Result<Range<usize>> {
    ensure!(align > 0 && outputs.div_ceil(align) > 1 && rank < 2, "invalid vocabulary partition");
    let split = outputs.div_ceil(align).div_ceil(2).checked_mul(align).context("vocabulary boundary overflow")?;
    ensure!(split < outputs, "empty vocabulary shard");
    Ok(if rank == 0 { 0..split } else { split..outputs })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shard_boundaries_cover_odd_vocab_and_preserve_aligned_padding() -> Result<()> {
        for (total, align, cut) in [(9, 1, 5), (201024, 128, 100608), (201088, 128, 100608)] {
            assert_eq!(shard(total, align, 0)?, 0..cut);
            assert_eq!(shard(total, align, 1)?, cut..total);
        }
        assert!(shard(128, 128, 0).is_err());
        assert!(shard(201088, 128, 2).is_err());
        Ok(())
    }
}
