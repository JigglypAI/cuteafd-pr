//! Serving glue for the `history` draft policy (`cuteafd_core::DraftHistory`):
//! the launch configuration, the per-round trace record and the `/v1/stats`
//! snapshot. Every speculating family drives the same five calls; no family
//! holds policy code.
//!
//! A serve loop's round:
//! 1. [`Policy::plan`] before the draft pass: drafts to ask for per request;
//! 2. [`Policy::shape`] after it: each proposal cut to its plan (the head may
//!    skip the explore position), extended by an agreeing history copy, and
//!    fitted to the verify-row budget;
//! 3. [`Policy::observe`] after verification, with what the target accepted.
use super::copy;
use super::length::DraftLength;
use crate::shared::speculation_trace::Trace;
use cuteafd_core::{DraftControl, DraftHistory, DraftOutcome, DrafterSpec, HeadScore};
use std::sync::{Arc, Mutex, OnceLock};

/// One active request after the draft pass, in verifier order.
pub(crate) struct Proposal<'a> {
    pub id: u64,
    /// The drafter's tokens (empty: not drafted). Cut and extended in place.
    pub tokens: &'a mut Vec<u32>,
    /// Raw head score per drafted position, for a drafter with a trained head.
    pub head: Option<&'a [f32]>,
    /// Prompt and generated tokens with the round's anchor last, for a history copy.
    pub history: &'a [u32],
    /// Copies continue greedy requests only.
    pub greedy: bool,
}

pub(crate) struct Policy {
    history: DraftHistory,
    trace: Option<&'static Trace>,
    rounds: u64,
}

impl Policy {
    /// `block` is the drafter's native block; `head` what its confidence head
    /// reports; `copies` whether the family verifies history copies.
    pub fn new(length: DraftLength, block: usize, head: HeadScore, copies: bool) -> Self {
        let spec = DrafterSpec { block, head, copy: if copies { copy::COPY_EXTENSION } else { 0 } };
        let history = DraftHistory::new(spec, length.control());
        cuteafd_bench::context::set_resolved("draft-mode", &length.label(block));
        tracing::info!(policy = %history.control().label(), block, head = ?head, copy_positions = spec.copy,
            requested_length = ?length.draft_length, "draft policy");
        let policy = Self { history, trace: replay_trace(), rounds: 0 };
        policy.publish();
        policy
    }

    pub fn block(&self) -> usize { self.history.spec().block }
    /// The most drafted positions one request verifies in a round.
    pub fn max_positions(&self) -> usize { self.history.max_positions() }
    pub fn fixed(&self) -> bool { matches!(self.history.control(), DraftControl::Fixed(_)) }

    /// Drafts to ask the drafter for, per request. `requests` are (identity,
    /// room) in verifier order; `rows` is the round's verify-row budget.
    pub fn plan(&mut self, requests: &[(u64, usize)], rows: usize) -> Vec<usize> {
        self.history.plan(requests, rows)
    }

    /// Fit every proposal to its plan under `rows` verify rows. `proposals`
    /// cover every active request in verifier order. Returns how many of each
    /// request's tokens are the drafter's own; the rest are a history copy.
    pub fn shape(&mut self, proposals: &mut [Proposal<'_>], rows: usize) -> Vec<usize> {
        let mut neural = Vec::with_capacity(proposals.len());
        for proposal in proposals.iter_mut() {
            let kept = self.history.refine(proposal.id, proposal.tokens.len(), proposal.head);
            proposal.tokens.truncate(kept);
            if kept > 0 && self.history.spec().copy > 0 {
                if let Some(tail) = copy::continuation(proposal.history, proposal.tokens, proposal.greedy) {
                    let extension = self.history.extend_copy(proposal.id, tail.len());
                    proposal.tokens.extend_from_slice(&tail[..extension]);
                }
            }
            neural.push(kept);
        }
        let ids: Vec<u64> = proposals.iter().map(|proposal| proposal.id).collect();
        for ((proposal, length), neural) in proposals.iter_mut().zip(self.history.settle(&ids, rows)).zip(&mut neural) {
            proposal.tokens.truncate(length);
            *neural = (*neural).min(length);
        }
        neural
    }

    /// Learn from one verified round and record it.
    pub fn observe(&mut self, outcomes: &[DraftOutcome]) {
        let record = self.history.observe(outcomes, self.trace.is_some());
        if let (Some(trace), Some(record)) = (self.trace, record) {
            let round = self.rounds;
            if let Err(error) = trace.write(crate::shared::speculation_trace::record! {
                r#type: "draft_round", round: round, draft: record }) {
                tracing::warn!(%error, "draft round trace failed");
            }
        }
        self.rounds += 1;
        if self.rounds % PUBLISH_ROUNDS == 0 { self.publish(); }
    }

    pub fn release(&mut self, id: u64) { self.history.release(id); }

    fn publish(&self) {
        if let Ok(mut slot) = SNAPSHOT.lock() { *slot = Some(Arc::new(self.history.snapshot())); }
    }
}

/// The stats view may lag the policy by this many rounds.
const PUBLISH_ROUNDS: u64 = 32;
static SNAPSHOT: Mutex<Option<Arc<serde_json::Value>>> = Mutex::new(None);

/// The latest published policy state for the serving `stats` payload.
pub(crate) fn snapshot() -> serde_json::Value {
    trace_snapshot().0.as_deref().cloned().unwrap_or(serde_json::Value::Null)
}

/// Cheap owned view for asynchronous trace serialization.
pub(crate) struct TraceSnapshot(Option<Arc<serde_json::Value>>);
impl serde::Serialize for TraceSnapshot {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.as_deref().serialize(serializer)
    }
}
pub(crate) fn trace_snapshot() -> TraceSnapshot {
    TraceSnapshot(SNAPSHOT.lock().ok().and_then(|slot| slot.clone()))
}

/// The `SPECULATION_TRACE` file, shared with each family's own round records.
pub(crate) fn replay_trace() -> Option<&'static Trace> {
    static TRACE: OnceLock<Option<Trace>> = OnceLock::new();
    TRACE.get_or_init(|| match Trace::open() {
        Ok(trace) => trace,
        Err(error) => { tracing::warn!(%error, "draft round trace open failed"); None }
    }).as_ref()
}

/// One request's verified round. `rows` is the verifier rows executed and
/// `accepted` the accepted inputs, both with the anchor; `censored` when
/// verification stopped for a reason other than a miss.
pub(crate) fn outcome(id: u64, rows: usize, accepted: usize, censored: bool) -> DraftOutcome {
    DraftOutcome { id, proposed: rows.saturating_sub(1), accepted: accepted.saturating_sub(1), censored }
}

/// Whether a finished request stopped without the target judging its next
/// draft: it ended on its last verified row, or on a token the next draft
/// also proposed. A terminal token that disagrees with the draft is a miss.
pub(crate) fn finish_censored(finished: bool, committed: usize, rows: &[u32], last: Option<u32>) -> bool {
    finished && !rows.get(committed).is_some_and(|next| Some(*next) != last)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn length(fixed: Option<usize>) -> DraftLength { DraftLength { draft_length: fixed, ..Default::default() } }

    /// History whose last eight tokens repeat an earlier stretch followed by 100, 101, ...
    fn repeating(suffix: usize) -> Vec<u32> {
        let mut history: Vec<u32> = (0..8).collect();
        history.extend(100..140);
        history.extend(50..60);
        history.extend((8 - suffix) as u32..8);
        history
    }

    #[test]
    fn shape_cuts_to_the_plan_extends_agreeing_copies_and_fits_the_budget() {
        let mut policy = Policy::new(length(None), 4, HeadScore::None, true);
        let history = repeating(8);
        assert_eq!(policy.plan(&[(1, 40), (2, 40)], 64), [4, 4]);
        let (mut agreeing, mut differing) = (vec![100, 101, 102, 103], vec![100, 101, 7, 103]);
        let neural = policy.shape(&mut [
            Proposal { id: 1, tokens: &mut agreeing, head: None, history: &history, greedy: true },
            Proposal { id: 2, tokens: &mut differing, head: None, history: &history, greedy: true }], 64);
        // A cold copy track verifies one explored position.
        assert_eq!((neural, agreeing.as_slice(), differing.len()), (vec![4, 4], &[100, 101, 102, 103, 104][..], 4));
        policy.observe(&[outcome(1, 6, 6, false), outcome(2, 5, 3, false)]);
        // Sampled requests never copy. Five drafts fit: both requests' first two positions
        // (probability 1), then request 1's third (0.56) ahead of request 2's explore (0.44).
        assert_eq!(policy.plan(&[(1, 40), (2, 40)], 64), [4, 3]);
        let (mut first, mut second) = (vec![100, 101, 102, 103], vec![100, 101, 7]);
        let neural = policy.shape(&mut [
            Proposal { id: 1, tokens: &mut first, head: None, history: &history, greedy: false },
            Proposal { id: 2, tokens: &mut second, head: None, history: &history, greedy: true }], 7);
        assert_eq!((neural, first.len(), second.len()), (vec![3, 2], 3, 2));
    }

    #[test]
    fn fixed_length_and_off_come_from_the_launch_configuration() {
        let (mut fixed, settings) = cuteafd_bench::context::capture_resolved(||
            Policy::new(length(Some(9)), 7, HeadScore::Logit, true));
        assert!(fixed.fixed() && fixed.max_positions() == 7);
        assert!(settings.0.iter().any(|(name, value)| name == "draft-mode" && value == "fixed 7"));
        assert_eq!(fixed.plan(&[(1, 40)], 64), [7]);
        let mut off = Policy::new(DraftLength { draft_policy: super::super::length::DraftPolicyChoice::Off,
            ..Default::default() }, 7, HeadScore::None, true);
        assert_eq!((off.plan(&[(1, 40)], 64), off.max_positions()), (vec![0], 0));
    }

    #[test]
    fn outcomes_drop_the_anchor_and_terminal_agreement_is_censored() {
        assert_eq!(outcome(3, 6, 3, false), DraftOutcome { id: 3, proposed: 5, accepted: 2, censored: false });
        assert_eq!(outcome(3, 1, 1, true), DraftOutcome { id: 3, proposed: 0, accepted: 0, censored: true });
        let rows = [10, 11, 12, 13];
        assert!(!finish_censored(false, 2, &rows, Some(12)));
        assert!(!finish_censored(true, 2, &rows, Some(99)));
        assert!(finish_censored(true, 2, &rows, Some(12)));
        assert!(finish_censored(true, 4, &rows, Some(99)));
    }
}
