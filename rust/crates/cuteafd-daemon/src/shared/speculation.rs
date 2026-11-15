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

/// Planned marks split the discard horizon before the final prompt frontier.
pub(crate) fn next_context_frontier(prompt_end: usize, points: &[usize], start: usize) -> usize {
    points.iter().copied().filter(|&point| point > start).min().unwrap_or(prompt_end).min(prompt_end)
}

/// The latest skipped position is dead only once a full window follows it.
pub(crate) fn context_window_complete(window: usize, skipped_end: usize, frontier: usize) -> bool {
    skipped_end <= frontier.saturating_sub(window)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planned_marks_and_disconnects_publish_only_complete_windows() {
        let (window, end) = (4096, 16384);
        let points = [8192, 12288];
        let (mut full, mut candidate) = (vec![0usize; window], vec![0usize; window]);
        let mut skipped_end = 0;
        let mut withheld = 0;
        let mut retained = 0;
        for start in (0..end).step_by(128) {
            let frontier = next_context_frontier(end, &points, start);
            let keep = retain_context_chunk(Some(window), start..start + 128, frontier);
            if !keep { skipped_end = start + 128; }
            for position in start..start + 128 {
                full[position % window] = position + 1;
                if keep { candidate[position % window] = position + 1; }
            }
            // An unexpected disconnect may park only after all holes died.
            if context_window_complete(window, skipped_end, start + 128) {
                assert_eq!(full, candidate);
                retained += 1;
            } else { withheld += 1; }
            if points.contains(&(start + 128)) || start + 128 == end {
                assert!(context_window_complete(window, skipped_end, start + 128));
                assert_eq!(full, candidate, "planned prompt/intermediate mark");
                let restored = candidate.clone();
                assert_eq!(restored, full, "restored mark equals cold prefill");
            }
        }
        assert!(withheld > 0 && retained > 0);
        // A turn continues from the exact prompt ring without a new discard horizon.
        for position in end..end + 17 {
            full[position % window] = position + 1;
            candidate[position % window] = position + 1;
        }
        assert_eq!(full, candidate, "turn-end mark");
    }

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
