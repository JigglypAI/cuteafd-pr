//! History copies continue a neural draft the request's own history agrees with.
//!
//! The copy window is the continuation of the latest earlier occurrence of the
//! request's last 4-8 tokens. It extends a draft only when every drafted
//! position the policy kept matches it; the copied positions are then verified
//! on their own acceptance history (`cuteafd_core::DraftHistory`'s copy track).

/// Most copy positions beyond the neural draft.
pub(crate) const COPY_EXTENSION: usize = 32;
pub(crate) const MIN_DRAFT_AGREEMENT: usize = 3;
pub(crate) const MIN_ANCHORED_MATCH: usize = 7;

/// The copy tokens that follow `drafted`, when the window agrees with all of
/// it: at least three drafts and seven anchored tokens (suffix plus drafts).
pub(crate) fn continuation<'h>(history: &'h [u32], drafted: &[u32], greedy: bool) -> Option<&'h [u32]> {
    if !greedy || drafted.len() < MIN_DRAFT_AGREEMENT { return None; }
    let (_, suffix, copy) = window(history, drafted.len() + COPY_EXTENSION);
    if suffix + drafted.len() < MIN_ANCHORED_MATCH || copy.len() <= drafted.len()
        || drafted != &copy[..drafted.len()] { return None; }
    Some(&copy[drafted.len()..])
}

/// Trace-only agreement with the same selected history copy, independent of room.
pub(crate) fn agreement(history: &[u32], drafted: &[u32]) -> usize {
    let (_, _, copy) = window(history, drafted.len());
    copy.iter().zip(drafted).take_while(|(copied, neural)| copied == neural).count()
}

pub(crate) fn trace_window(history: &[u32], limit: usize) -> (Option<usize>, usize, Vec<u32>) {
    let (start, suffix, copy) = window(history, limit);
    (start, suffix, copy.to_vec())
}

/// Latest earlier occurrence of the longest matching history suffix (4-8 tokens).
fn window(history: &[u32], limit: usize) -> (Option<usize>, usize, &[u32]) {
    let len = history.len();
    for n in (4..=8).rev() {
        if len <= n { continue; }
        let tail = &history[len - n..];
        if let Some(start) = (0..len - n).rev().find(|&i| &history[i..i + n] == tail) {
            let from = start + n;
            return (Some(start), n, &history[from..from + limit.min(len - from)]);
        }
    }
    (None, 0, &[])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `copy` after an eight-token stretch that the history then repeats `suffix` tokens of.
    fn history(copy: &[u32], suffix: usize) -> Vec<u32> {
        let mut history: Vec<u32> = (1..=8).collect();
        history.extend_from_slice(copy);
        history.extend(900..910);
        history.extend((9 - suffix) as u32..=8);
        history
    }

    #[test]
    fn an_agreeing_draft_is_continued_by_the_rest_of_the_window() {
        let copy: Vec<u32> = (100..150).collect();
        let history = history(&copy, 8);
        for width in [3, 5, 7, 16] {
            let tail = continuation(&history, &copy[..width], true).unwrap();
            assert_eq!(tail, &copy[width..width + COPY_EXTENSION]);
        }
        assert_eq!(trace_window(&history, 4), (Some(0), 8, copy[..4].to_vec()));
        assert_eq!(agreement(&history, &[100, 101, 7, 103]), 2);
    }

    #[test]
    fn disagreement_short_drafts_weak_anchors_and_sampling_do_not_copy() {
        let copy: Vec<u32> = (100..150).collect();
        let anchored = history(&copy, 8);
        assert!(continuation(&anchored, &[100, 101, 7], true).is_none());
        assert!(continuation(&anchored, &copy[..2], true).is_none());
        assert!(continuation(&anchored, &copy[..5], false).is_none());
        // Four matched tokens need three drafts to reach seven anchored.
        let weak = history(&copy, 4);
        assert!(continuation(&weak, &copy[..3], true).is_some());
        assert!(continuation(&(0..40).collect::<Vec<u32>>(), &[1, 2, 3], true).is_none());
    }
}
