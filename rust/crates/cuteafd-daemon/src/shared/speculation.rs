//! Shared speculative draft helpers; serving lifecycles stay with each family.

pub(crate) mod block_dspark;
pub(crate) mod dflash;
pub(crate) mod target;

pub(crate) mod weights;

/// Retain the original ingest chunk if any of it survives at `end`. Never trim
/// a straddling chunk: projection rounding can depend on its GEMM row count.
/// Context K/V are independent projections of each position's target taps, so
/// chunks overwritten before the first draft need neither capture nor update.
pub(crate) fn retain_context_chunk(window: Option<usize>, positions: std::ops::Range<usize>, end: usize) -> bool {
    !positions.is_empty() && window.is_none_or(|window| positions.end > end.saturating_sub(window))
}

#[cfg(test)]
mod tests {
    use super::retain_context_chunk;

    #[test]
    fn whole_chunk_ingest_matches_full_ring_and_preserves_shapes() {
        let window = 4096;
        for end in [17, window, window + 1, 8192, 65536] {
            for chunk in [128, 257, 2048] {
                let mut full = vec![[0u8; 16]; window];
                let mut skipped = full.clone();
                let mut kept = 0;
                for first in (0..end).step_by(chunk) {
                    let span = first..(first + chunk).min(end);
                    let retain = retain_context_chunk(Some(window), span.clone(), end);
                    for position in span.clone() {
                        // Include shape in the payload to detect accidental boundary trimming.
                        let mut bytes = [0; 16];
                        bytes[..8].copy_from_slice(&(position as u64).to_le_bytes());
                        bytes[8..].copy_from_slice(&(span.len() as u64).to_le_bytes());
                        full[position % window] = bytes;
                        if retain { skipped[position % window] = bytes; }
                    }
                    if retain { kept += span.len(); }
                }
                assert_eq!(full, skipped, "end={end} chunk={chunk}");
                assert!(kept <= end.min(window + chunk - 1));
            }
        }
        assert!(!retain_context_chunk(Some(window), 0..128, 8192));
        assert!(retain_context_chunk(Some(window), 4096..4224, 8193));
        assert!(retain_context_chunk(None, 0..128, 65536));
    }
}
