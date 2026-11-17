//! The `history` draft policy: how many drafted positions each request
//! verifies, decided from per-position acceptance history alone.
//!
//! A drafter declares its native block width and whether it has a trained
//! confidence head ([`DrafterSpec`]). Each round the serve loop:
//! 1. [`DraftHistory::plan`]s a length per request before the draft pass;
//! 2. [`DraftHistory::refine`]s it with what the drafter actually proposed
//!    (and its head scores, where it has a head);
//! 3. optionally [`DraftHistory::extend_copy`]s it with a history copy;
//! 4. [`DraftHistory::settle`]s the round under the verify-row budget;
//! 5. [`DraftHistory::observe`]s what the target accepted.
//!
//! **Probability.** For request `i` at position `k`,
//! `p_ik = (s_ik + m * g_k) / (t_ik + m)`: `s_ik / t_ik` are the accepted and
//! reached counts at `k` over the request's last [`REQUEST_WINDOW`] verify
//! rounds, `g_k` the server-wide acceptance rate at `k` over the last
//! [`SERVER_WINDOW`] rounds that reached `k`, and `m` = [`PRIOR_STRENGTH`].
//! A position is reached only when every position before it was accepted and
//! the target judged it; a round that stopped for another reason (stop token,
//! output limit, cancellation) is censored at its accepted prefix.
//!
//! **Length.** Positions are kept while their own `p_ik > 0.5`, each judged
//! alone (no running product), up to the first at or below one half.
//!
//! **Exploration.** One position past the stop is always verified, up to the
//! block, so neither the request's nor the server's estimate can lock low.
//!
//! **Head skip.** A drafter with a trained head keeps one online Platt fit per
//! position from raw head score to acceptance, trained on reached positions.
//! The explore position is skipped for a round when its calibrated probability
//! is below [`SKIP_CUT`], and the skip is booked as a miss at that position in
//! the request's and the server's history (when the kept prefix before it was
//! accepted, exactly as a rejected explore would be): otherwise the position's
//! history stands still and later rounds draft deeper. Drafters without a head
//! always explore.
//!
//! **Budget.** When the round's rows exceed the verify-row budget, positions
//! are granted across requests by the same probabilities, highest first, each
//! request keeping a prefix of its plan. There is no cost model.
//!
//! **Copies.** A history copy that continues past an agreeing neural draft is
//! a second track with identical math and its own windows; its server rate
//! starts at [`COPY_COLD_RATE`], so a cold copy verifies one explored position
//! and grows as copies are accepted.
//!
//! The request prior has strength two: after eight reached observations the
//! request owns 80% of the estimate. A code-conditioned server must not keep
//! a subsequent low-acceptance prose request at full width indefinitely.
use serde::Serialize;
use std::collections::{HashMap, VecDeque};

/// Verify rounds of a request's own history.
pub const REQUEST_WINDOW: usize = 8;
/// Rounds that reached a position, server wide.
pub const SERVER_WINDOW: usize = 256;
/// Weight `m` of the server rate against the request's own counts, in rounds.
/// V4.1 fixed7 card-order replay (code first, then prose; 20 prose requests):
/// m | prose rows/round | tokens/round | tokens/row | code-after tokens/round
/// 8 | 6.77             | 1.97         | 0.29       | 4.96
/// 4 | 5.54             | 1.93         | 0.35       | --
/// 2 | 4.12             | 1.83         | 0.45       | 4.89
/// 1 | 3.05             | 1.81         | 0.59       | --
/// Two cuts prose rows 39% for 7% fewer tokens/round; code changes -1.4%.
pub const PRIOR_STRENGTH: f64 = 2.;
/// A position is kept while its probability is above this.
pub const KEEP_CUT: f64 = 0.5;
/// A head-calibrated explore position below this is skipped for the round.
/// The one place this threshold lives: retune it here from card data.
pub const SKIP_CUT: f64 = 0.3;
/// Server rate of a neural position nobody has reached yet.
pub const NEURAL_COLD_RATE: f64 = 1.;
/// Server rate of a copy position nobody has reached yet.
pub const COPY_COLD_RATE: f64 = 0.5;

/// What a drafter's confidence head reports per drafted position.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum HeadScore {
    /// No trained head: the drafter always explores.
    None,
    /// The head's raw logit.
    Logit,
    /// The head's sigmoid.
    Probability,
}

/// Everything a drafter declares to the policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DrafterSpec {
    /// Native block: the most positions one draft pass proposes.
    pub block: usize,
    pub head: HeadScore,
    /// Most copy positions a history copy may add past the neural draft.
    pub copy: usize,
}

/// `DRAFT_POLICY` and the `DRAFT_LENGTH` measurement override.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DraftControl {
    History,
    /// Verify this many positions every round (capped by block, room and rows).
    Fixed(usize),
    /// Verify no drafts.
    Off,
}

impl DraftControl {
    pub fn label(self) -> String {
        match self {
            Self::History => "history".into(),
            Self::Fixed(n) => format!("fixed {n}"),
            Self::Off => "off".into(),
        }
    }
}

/// One request's verified round.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DraftOutcome {
    pub id: u64,
    /// Drafted positions the target verified (anchor excluded).
    pub proposed: usize,
    /// Leading drafted positions the target accepted.
    pub accepted: usize,
    /// Verification stopped for a reason other than a miss.
    pub censored: bool,
}

/// Which estimate a position belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Track {
    Neural,
    Copy,
}

/// One planned position as the policy saw it.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct PositionTrace {
    pub track: Track,
    /// Zero-based position within its track.
    pub position: usize,
    /// Probability the plan used.
    pub p: f64,
    /// The request's accepted and reached rounds at this position.
    pub accepted: u32,
    pub reached: u32,
    /// The server rate and how many rounds it rests on.
    pub server: f64,
    pub server_reached: u32,
    /// Verified past the stop.
    pub explore: bool,
    /// Raw head score and its calibrated probability, where the drafter has a head.
    pub head: Option<f32>,
    pub calibrated: Option<f64>,
    /// What the target said: accepted, rejected, or not reached.
    pub outcome: Option<bool>,
}

/// One request's round.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RequestTrace {
    pub id: u64,
    /// HTTP identity attached by the serving adapter, absent in offline replays.
    pub request_id: Option<String>,
    pub room: usize,
    /// Leading neural positions above the cut.
    pub keep: usize,
    /// The head skipped this round's explore: its calibrated probability.
    pub explore_skipped: Option<f64>,
    /// Neural and copy positions granted under the row budget.
    pub neural: usize,
    pub copy: usize,
    pub proposed: usize,
    pub accepted_drafts: usize,
    pub censored: bool,
    pub positions: Vec<PositionTrace>,
}

/// One observed round, for `SPECULATION_TRACE`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RoundTrace {
    pub policy: String,
    pub block: usize,
    /// Request rounds observed so far, and how many of them verified at most one
    /// draft although they had room for more: the share a card reads to tell a
    /// short-drafting workload from a locked estimate.
    pub rounds: u64,
    pub short_rounds: u64,
    pub short_share: f64,
    /// Explores verified, accepted and skipped by the head so far.
    pub explored: u64,
    pub explore_accepted: u64,
    pub head_skips: u64,
    pub requests: Vec<RequestTrace>,
}

/// Lifetime counters for `/v1/stats`.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct DraftHistoryStats {
    /// Request rounds observed.
    pub rounds: u64,
    pub verified_drafts: u64,
    pub accepted_drafts: u64,
    /// Request rounds that verified at most one draft although they had room for more.
    pub short_rounds: u64,
    /// Explore positions verified, and how many the target accepted.
    pub explored: u64,
    pub explore_accepted: u64,
    /// Explore positions the head skipped, and how many were booked as a miss.
    pub head_skips: u64,
    pub skip_misses: u64,
    /// Positions the row budget removed from plans.
    pub budget_cut: u64,
    /// Request rounds by verified draft count.
    pub length_histogram: Vec<u64>,
    pub position_reached: Vec<u64>,
    pub position_accepted: Vec<u64>,
    pub copy_reached: Vec<u64>,
    pub copy_accepted: Vec<u64>,
}

impl DraftHistoryStats {
    /// Share of request rounds that verified at most one draft with room for more.
    pub fn short_share(&self) -> f64 {
        if self.rounds == 0 { 0. } else { self.short_rounds as f64 / self.rounds as f64 }
    }
}

/// The last `SERVER_WINDOW` outcomes of one position.
#[derive(Clone, Debug)]
struct Window {
    bits: [u64; SERVER_WINDOW / 64],
    len: usize,
    next: usize,
    ones: usize,
}

impl Window {
    fn new() -> Self { Self { bits: [0; SERVER_WINDOW / 64], len: 0, next: 0, ones: 0 } }

    fn push(&mut self, accepted: bool) {
        let (word, bit) = (self.next / 64, 1u64 << (self.next % 64));
        if self.len == SERVER_WINDOW {
            self.ones -= usize::from(self.bits[word] & bit != 0);
        } else {
            self.len += 1;
        }
        if accepted {
            self.bits[word] |= bit;
            self.ones += 1;
        } else {
            self.bits[word] &= !bit;
        }
        self.next = (self.next + 1) % SERVER_WINDOW;
    }

    fn rate(&self, cold: f64) -> f64 {
        if self.len == 0 { cold } else { self.ones as f64 / self.len as f64 }
    }
}

/// Online Platt scaling `p = sigmoid(a * logit + b)` by a Newton step on the
/// log-loss with a decayed information matrix (a running fit over the last
/// few hundred reached positions).
#[derive(Clone, Copy, Debug)]
struct Platt {
    theta: [f64; 2],
    information: [[f64; 2]; 2],
}

const PLATT_DECAY: f64 = 0.998;
const PLATT_PRIOR: f64 = 2.;

impl Platt {
    fn new() -> Self { Self { theta: [1., 0.], information: [[PLATT_PRIOR, 0.], [0., PLATT_PRIOR]] } }

    fn apply(&self, logit: f64) -> f64 {
        let z = self.theta[0] * logit + self.theta[1];
        1. / (1. + (-z).exp())
    }

    fn observe(&mut self, logit: f64, accepted: bool) {
        let x = [logit, 1.];
        let p = self.apply(logit);
        let outcome = f64::from(u8::from(accepted));
        let curvature = (p * (1. - p)).max(0.02);
        for i in 0..2 {
            for j in 0..2 {
                self.information[i][j] = PLATT_DECAY * self.information[i][j] + curvature * x[i] * x[j];
            }
            // Stay well conditioned when one direction is unexcited (a constant score).
            self.information[i][i] = self.information[i][i].max(1e-3);
        }
        let [[a, b], [c, d]] = self.information;
        let determinant = a * d - b * c;
        if !(determinant.is_finite() && determinant > 1e-12) { return; }
        let gradient = [(outcome - p) * x[0], outcome - p];
        let step = [(d * gradient[0] - b * gradient[1]) / determinant,
            (a * gradient[1] - c * gradient[0]) / determinant];
        // Slope in [0, 3]: calibration may flatten a head, never invert it.
        self.theta[0] = (self.theta[0] + step[0].clamp(-1., 1.)).clamp(0., 3.);
        self.theta[1] = (self.theta[1] + step[1].clamp(-2., 2.)).clamp(-8., 8.);
    }
}

/// One verify round of a request: positions reached and accepted per track.
#[derive(Clone, Copy, Debug, Default)]
struct Round {
    reached: [u8; 2],
    accepted: [u8; 2],
}

/// A position's inputs at plan time.
#[derive(Clone, Copy, Debug)]
struct Planned {
    p: f64,
    accepted: u32,
    reached: u32,
    server: f64,
    server_reached: u32,
}

/// A request's plan between `plan` and `observe`.
#[derive(Clone, Debug, Default)]
struct Pending {
    room: usize,
    keep: usize,
    /// The skipped explore's calibrated probability.
    skipped: Option<f64>,
    skipped_position: Option<Planned>,
    /// Calibration used before this round's outcomes train the fit.
    calibrated: Vec<Option<f64>>,
    /// Planned positions per track, and how many of them the budget granted.
    planned: [Vec<Planned>; 2],
    granted: [usize; 2],
    head: Vec<f32>,
}

#[derive(Clone, Debug, Default)]
struct Request {
    rounds: VecDeque<Round>,
    pending: Option<Pending>,
}

impl Request {
    /// (accepted, reached) rounds of `position` in the request's window.
    fn counts(&self, track: usize, position: usize) -> (u32, u32) {
        let (mut accepted, mut reached) = (0, 0);
        for round in &self.rounds {
            reached += u32::from(usize::from(round.reached[track]) > position);
            accepted += u32::from(usize::from(round.accepted[track]) > position);
        }
        (accepted, reached)
    }
}

const NEURAL: usize = 0;
const COPY: usize = 1;

/// The policy and all of its state: one per serving process.
#[derive(Clone, Debug)]
pub struct DraftHistory {
    spec: DrafterSpec,
    control: DraftControl,
    server: [Vec<Window>; 2],
    platt: Vec<Platt>,
    requests: HashMap<u64, Request>,
    stats: DraftHistoryStats,
}

impl DraftHistory {
    pub fn new(spec: DrafterSpec, control: DraftControl) -> Self {
        let control = match control {
            DraftControl::Fixed(n) => DraftControl::Fixed(n.min(spec.block)),
            other => other,
        };
        Self {
            spec, control,
            server: [vec![Window::new(); spec.block], vec![Window::new(); spec.copy]],
            platt: vec![Platt::new(); spec.block],
            requests: HashMap::new(),
            stats: DraftHistoryStats {
                length_histogram: vec![0; spec.block + spec.copy + 1],
                position_reached: vec![0; spec.block], position_accepted: vec![0; spec.block],
                copy_reached: vec![0; spec.copy], copy_accepted: vec![0; spec.copy],
                ..Default::default()
            },
        }
    }

    pub fn spec(&self) -> DrafterSpec { self.spec }
    pub fn control(&self) -> DraftControl { self.control }
    pub fn stats(&self) -> &DraftHistoryStats { &self.stats }

    /// The most positions any request can verify in one round.
    pub fn max_positions(&self) -> usize {
        match self.control {
            DraftControl::History => self.spec.block + self.spec.copy,
            DraftControl::Fixed(n) => n,
            DraftControl::Off => 0,
        }
    }

    fn cold(track: usize) -> f64 { if track == NEURAL { NEURAL_COLD_RATE } else { COPY_COLD_RATE } }

    fn position(&self, request: &Request, track: usize, position: usize) -> Planned {
        let (accepted, reached) = request.counts(track, position);
        let window = &self.server[track][position];
        let server = window.rate(Self::cold(track));
        Planned { p: (f64::from(accepted) + PRIOR_STRENGTH * server) / (f64::from(reached) + PRIOR_STRENGTH),
            accepted, reached, server, server_reached: window.len as u32 }
    }

    /// The probability the policy would use for `id` at a neural position now.
    pub fn probability(&self, id: u64, position: usize) -> f64 {
        static NEW: std::sync::OnceLock<Request> = std::sync::OnceLock::new();
        let request = self.requests.get(&id).unwrap_or_else(|| NEW.get_or_init(Request::default));
        self.position(request, NEURAL, position).p
    }

    /// Positions of a track to verify: the leading ones above the cut plus one explore.
    fn track_plan(&self, request: &Request, track: usize, limit: usize) -> (Vec<Planned>, usize) {
        let mut planned = Vec::with_capacity(limit.min(8));
        let mut keep = 0;
        for position in 0..limit {
            let entry = self.position(request, track, position);
            planned.push(entry);
            if entry.p > KEEP_CUT { keep += 1; } else { break; }
        }
        (planned, keep)
    }

    /// Plan each request's neural draft before the draft pass. `requests` are
    /// (identity, room): the most drafts the request may verify this round.
    /// `max_rows` is the round's verify-row budget, anchors included. Returns
    /// the drafts to ask the drafter for, in request order.
    pub fn plan(&mut self, requests: &[(u64, usize)], max_rows: usize) -> Vec<usize> {
        for &(id, room) in requests {
            let request = self.requests.remove(&id).unwrap_or_default();
            let limit = room.min(self.spec.block);
            let (planned, keep) = match self.control {
                DraftControl::Off => (Vec::new(), 0),
                DraftControl::History => self.track_plan(&request, NEURAL, limit),
                DraftControl::Fixed(n) => {
                    let n = n.min(limit);
                    ((0..n).map(|position| self.position(&request, NEURAL, position)).collect(), n)
                }
            };
            let granted = planned.len();
            let pending = Pending { room, keep, skipped: None, planned: [planned, Vec::new()],
                granted: [granted, 0], ..Default::default() };
            self.requests.insert(id, Request { pending: Some(pending), ..request });
        }
        let ids: Vec<u64> = requests.iter().map(|&(id, _)| id).collect();
        self.settle(&ids, max_rows)
    }

    /// Report what the drafter proposed for `id`: `drafted` positions and, for
    /// a drafter with a head, their raw scores. Returns the neural positions
    /// still planned (the head may skip the explore).
    pub fn refine(&mut self, id: u64, drafted: usize, head: Option<&[f32]>) -> usize {
        let Some(pending) = self.requests.get_mut(&id).and_then(|request| request.pending.as_mut()) else { return 0 };
        pending.granted[NEURAL] = pending.granted[NEURAL].min(drafted);
        pending.planned[NEURAL].truncate(pending.granted[NEURAL]);
        let Some(head) = head.filter(|_| self.spec.head != HeadScore::None) else { return pending.granted[NEURAL] };
        pending.head = head[..head.len().min(self.spec.block)].to_vec();
        pending.calibrated = pending.head.iter().enumerate().map(|(position, &score)|
            score.is_finite().then(|| self.platt[position].apply(logit(self.spec.head, score)))).collect();
        let explore = pending.keep;
        if self.control == DraftControl::History && pending.granted[NEURAL] > explore {
            if let Some(&score) = pending.head.get(explore) {
                let calibrated = self.platt[explore].apply(logit(self.spec.head, score));
                if score.is_finite() && calibrated < SKIP_CUT {
                    pending.granted[NEURAL] = explore;
                    pending.skipped_position = pending.planned[NEURAL].get(explore).cloned();
                    pending.planned[NEURAL].truncate(explore);
                    pending.skipped = Some(calibrated);
                    self.stats.head_skips += 1;
                }
            }
        }
        pending.granted[NEURAL]
    }

    /// A history copy agrees with every neural position still planned for `id`
    /// and offers `available` more tokens. Returns how many of them to verify.
    pub fn extend_copy(&mut self, id: u64, available: usize) -> usize {
        if self.control != DraftControl::History { return 0; }
        let Some(mut request) = self.requests.remove(&id) else { return 0 };
        let mut count = 0;
        if let Some(room) = request.pending.as_ref().map(|pending| pending.room.saturating_sub(pending.granted[NEURAL])) {
            let (planned, _) = self.track_plan(&request, COPY, available.min(self.spec.copy).min(room));
            count = planned.len();
            let pending = request.pending.as_mut().expect("checked above");
            pending.granted[COPY] = count;
            pending.planned[COPY] = planned;
        }
        self.requests.insert(id, request);
        count
    }

    /// Fit the round into `max_rows` verify rows (one anchor per request plus
    /// its drafts): positions are granted across requests by probability,
    /// highest first, each request keeping a prefix of its plan; equal
    /// probabilities go to the request with fewer drafts so far, then to the
    /// older one. Returns each request's drafts.
    pub fn settle(&mut self, ids: &[u64], max_rows: usize) -> Vec<usize> {
        let wanted: Vec<[usize; 2]> = ids.iter().map(|id| self.requests.get(id)
            .and_then(|request| request.pending.as_ref()).map_or([0, 0], |pending| pending.granted)).collect();
        let total: usize = wanted.iter().map(|w| w[0] + w[1]).sum();
        let budget = max_rows.saturating_sub(ids.len());
        if total <= budget { return wanted.iter().map(|w| w[0] + w[1]).collect(); }
        let probability = |index: usize, taken: usize| {
            let pending = self.requests[&ids[index]].pending.as_ref().expect("wanted positions have a plan");
            if taken < wanted[index][NEURAL] { pending.planned[NEURAL][taken].p }
            else { pending.planned[COPY][taken - wanted[index][NEURAL]].p }
        };
        let mut granted = vec![0usize; ids.len()];
        for _ in 0..budget {
            let next = (0..ids.len()).filter(|&i| granted[i] < wanted[i][0] + wanted[i][1])
                .max_by(|&a, &b| probability(a, granted[a]).total_cmp(&probability(b, granted[b]))
                    .then_with(|| granted[b].cmp(&granted[a])).then_with(|| ids[b].cmp(&ids[a])));
            let Some(next) = next else { break };
            granted[next] += 1;
        }
        self.stats.budget_cut += (total - granted.iter().sum::<usize>()) as u64;
        for ((id, &granted), wanted) in ids.iter().zip(&granted).zip(&wanted) {
            let pending = self.requests.get_mut(id).and_then(|request| request.pending.as_mut())
                .expect("wanted positions have a plan");
            pending.granted = [granted.min(wanted[NEURAL]), granted.saturating_sub(wanted[NEURAL])];
            pending.planned[NEURAL].truncate(pending.granted[NEURAL]);
            pending.planned[COPY].truncate(pending.granted[COPY]);
        }
        granted
    }

    /// Learn from one verified round. With `trace`, also return what every
    /// planned position looked like and what became of it.
    pub fn observe(&mut self, outcomes: &[DraftOutcome], trace: bool) -> Option<RoundTrace> {
        let mut requests = Vec::with_capacity(if trace { outcomes.len() } else { 0 });
        for outcome in outcomes {
            let mut request = self.requests.remove(&outcome.id).unwrap_or_default();
            // A round the policy did not plan (a caller's own fixed shape) is all neural.
            let pending = request.pending.take().unwrap_or_else(|| Pending {
                room: outcome.proposed, keep: outcome.proposed.min(self.spec.block),
                granted: [outcome.proposed.min(self.spec.block), 0], ..Default::default() });
            let proposed = outcome.proposed.min(pending.granted[NEURAL] + pending.granted[COPY]);
            let accepted = outcome.accepted.min(proposed);
            // The position after the accepted prefix was judged unless the round was censored.
            let judged = (accepted + usize::from(!outcome.censored)).min(proposed);
            let neural = proposed.min(pending.granted[NEURAL]);
            let mut reached = [judged.min(neural), judged.saturating_sub(neural)];
            let kept = [accepted.min(neural), accepted.saturating_sub(neural)];
            // A skipped explore whose kept prefix was verified and accepted is a miss at
            // its position, as a rejected explore would have been. It trains no calibration.
            let skip_miss = pending.skipped.is_some() && !outcome.censored
                && neural == pending.keep && kept[NEURAL] == pending.keep;
            for track in [NEURAL, COPY] {
                for position in 0..reached[track] {
                    let hit = position < kept[track];
                    self.server[track][position].push(hit);
                    let (reached, accepted) = if track == NEURAL {
                        (&mut self.stats.position_reached, &mut self.stats.position_accepted)
                    } else { (&mut self.stats.copy_reached, &mut self.stats.copy_accepted) };
                    reached[position] += 1;
                    accepted[position] += u64::from(hit);
                    if track == NEURAL && self.spec.head != HeadScore::None {
                        if let Some(&score) = pending.head.get(position).filter(|score| score.is_finite()) {
                            self.platt[position].observe(logit(self.spec.head, score), hit);
                        }
                    }
                }
            }
            if skip_miss {
                self.server[NEURAL][pending.keep].push(false);
                reached[NEURAL] = pending.keep + 1;
                self.stats.skip_misses += 1;
            }
            let explores = self.control == DraftControl::History;
            if explores && !skip_miss && reached[NEURAL] > pending.keep {
                self.stats.explored += 1;
                self.stats.explore_accepted += u64::from(kept[NEURAL] > pending.keep);
            }
            self.stats.rounds += 1;
            self.stats.verified_drafts += proposed as u64;
            self.stats.accepted_drafts += accepted as u64;
            self.stats.short_rounds += u64::from(proposed <= 1 && pending.room.min(self.spec.block) > 1);
            if let Some(bucket) = self.stats.length_histogram.get_mut(proposed) { *bucket += 1; }
            if request.rounds.len() == REQUEST_WINDOW { request.rounds.pop_front(); }
            request.rounds.push_back(Round { reached: reached.map(|n| n as u8), accepted: kept.map(|n| n as u8) });
            if trace {
                let mut positions = Vec::with_capacity(proposed);
                for (track, label) in [(NEURAL, Track::Neural), (COPY, Track::Copy)] {
                    let skipped = if track == NEURAL { pending.skipped_position.as_ref() } else { None };
                    let planned = pending.planned[track].iter().enumerate()
                        .chain(skipped.map(|planned| (pending.keep, planned)));
                    for (position, planned) in planned {
                        let head = if track == NEURAL { pending.head.get(position).copied() } else { None };
                        positions.push(PositionTrace { track: label, position, p: planned.p,
                            accepted: planned.accepted, reached: planned.reached, server: planned.server,
                            server_reached: planned.server_reached,
                            explore: explores && if track == NEURAL { position >= pending.keep }
                                else { position + 1 == pending.planned[COPY].len() && planned.p <= KEEP_CUT },
                            head, calibrated: if track == NEURAL { pending.calibrated.get(position).copied().flatten() }
                                else { None },
                            outcome: (position < reached[track] && !(track == NEURAL
                                && pending.skipped.is_some() && position == pending.keep))
                                .then_some(position < kept[track]) });
                    }
                }
                requests.push(RequestTrace { id: outcome.id, request_id: None, room: pending.room, keep: pending.keep,
                    explore_skipped: pending.skipped, neural: pending.granted[NEURAL], copy: pending.granted[COPY],
                    proposed, accepted_drafts: accepted, censored: outcome.censored, positions });
            }
            self.requests.insert(outcome.id, request);
        }
        trace.then(|| RoundTrace { policy: self.control.label(), block: self.spec.block,
            rounds: self.stats.rounds, short_rounds: self.stats.short_rounds, short_share: self.stats.short_share(),
            explored: self.stats.explored, explore_accepted: self.stats.explore_accepted,
            head_skips: self.stats.head_skips, requests })
    }

    /// Forget a finished request.
    pub fn release(&mut self, id: u64) { self.requests.remove(&id); }

    /// Server-wide state for `/v1/stats`.
    pub fn snapshot(&self) -> serde_json::Value {
        let track = |track: usize, reached: &[u64], accepted: &[u64]| -> Vec<serde_json::Value> {
            self.server[track].iter().enumerate().map(|(position, window)| serde_json::json!({
                "position": position + 1, "window_rate": window.rate(Self::cold(track)), "window_rounds": window.len,
                "reached": reached[position], "accepted": accepted[position],
                "calibration": (track == NEURAL && self.spec.head != HeadScore::None)
                    .then(|| self.platt[position].theta),
            })).collect()
        };
        serde_json::json!({
            "policy": self.control.label(), "block": self.spec.block, "head": self.spec.head,
            "copy_positions": self.spec.copy, "request_window": REQUEST_WINDOW, "server_window": SERVER_WINDOW,
            "prior_strength": PRIOR_STRENGTH, "active_requests": self.requests.len(),
            "rounds": self.stats.rounds, "verified_drafts": self.stats.verified_drafts,
            "accepted_drafts": self.stats.accepted_drafts, "short_rounds": self.stats.short_rounds,
            "short_share": self.stats.short_share(),
            "explored": self.stats.explored, "explore_accepted": self.stats.explore_accepted,
            "head_skips": self.stats.head_skips, "skip_misses": self.stats.skip_misses, "budget_cut": self.stats.budget_cut,
            "length_histogram": self.stats.length_histogram,
            "positions": track(NEURAL, &self.stats.position_reached, &self.stats.position_accepted),
            "copy": track(COPY, &self.stats.copy_reached, &self.stats.copy_accepted),
        })
    }
}

fn logit(kind: HeadScore, score: f32) -> f64 {
    let score = f64::from(score);
    match kind {
        HeadScore::Probability => {
            let p = score.clamp(1e-6, 1. - 1e-6);
            (p / (1. - p)).ln()
        }
        HeadScore::Logit | HeadScore::None => score.clamp(-30., 30.),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy(block: usize) -> DraftHistory {
        DraftHistory::new(DrafterSpec { block, head: HeadScore::None, copy: 0 }, DraftControl::History)
    }

    fn outcome(id: u64, proposed: usize, accepted: usize) -> DraftOutcome {
        DraftOutcome { id, proposed, accepted, censored: false }
    }

    /// Deterministic uniform draws in [0, 1).
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
        /// Leading positions of `n` accepted at `rate` each.
        fn accepted(&mut self, n: usize, rate: f64) -> usize {
            (0..n).take_while(|_| self.next() < rate).count()
        }
    }

    #[test]
    fn a_cold_server_drafts_the_whole_block() {
        let mut policy = policy(7);
        assert_eq!(policy.plan(&[(1, 7), (2, 3), (3, 0)], 64), [7, 3, 0]);
        assert_eq!(policy.probability(9, 6), 1.);
    }

    #[test]
    fn probability_is_the_request_window_shrunk_to_the_server_rate() {
        let mut policy = policy(4);
        // Another request teaches the server: position 0 accepted 3 of 4, position 1 1 of 3.
        for accepted in [2, 1, 1, 0] {
            policy.plan(&[(9, 4)], 64);
            policy.observe(&[outcome(9, 4, accepted)], false);
        }
        assert_eq!(policy.probability(1, 0), 0.75);
        assert!((policy.probability(1, 1) - 1. / 3.).abs() < 1e-12);
        // Request 1 accepts position 0 twice and misses it once.
        for accepted in [1, 0, 1] {
            policy.plan(&[(1, 1)], 64);
            policy.observe(&[outcome(1, 1, accepted)], false);
        }
        let server = (3. + 2.) / (4. + 3.);
        assert!((policy.probability(1, 0) - (2. + PRIOR_STRENGTH * server) / (3. + PRIOR_STRENGTH)).abs() < 1e-12);
    }

    #[test]
    fn each_position_is_judged_alone_and_the_stop_is_explored() {
        let mut policy = policy(7);
        // Position 2 misses every time it is reached; positions 0 and 1 never do.
        for _ in 0..REQUEST_WINDOW {
            policy.plan(&[(1, 7)], 64);
            policy.observe(&[outcome(1, 7, 2)], false);
        }
        assert!(policy.probability(1, 0) > KEEP_CUT && policy.probability(1, 1) > KEEP_CUT);
        assert!(policy.probability(1, 2) <= KEEP_CUT);
        // Two kept positions plus the explored stop; nothing past it.
        assert_eq!(policy.plan(&[(1, 7)], 64), [3]);
        let trace = policy.observe(&[outcome(1, 3, 2)], true).unwrap();
        let positions = &trace.requests[0].positions;
        assert_eq!(positions.iter().map(|p| p.explore).collect::<Vec<_>>(), [false, false, true]);
        assert_eq!(positions.iter().map(|p| p.outcome).collect::<Vec<_>>(), [Some(true), Some(true), Some(false)]);
        // Position 2 was explored from the second round on: one miss already puts it under the cut.
        assert_eq!((policy.stats().explored, policy.stats().explore_accepted), (8, 0));
    }

    #[test]
    fn prose_request_overcomes_code_conditioned_server_within_its_window() {
        let mut policy = policy(7);
        for _ in 0..40 {
            policy.plan(&[(9, 7)], 64);
            policy.observe(&[outcome(9, 7, 7)], false);
        }
        // First position always agrees; each subsequent position agrees with
        // probability .3. This deterministic ten-round cycle reaches P2 on
        // every round and accepts it exactly three times.
        let accepted = [1, 1, 2, 1, 1, 2, 1, 1, 2, 1];
        let mut shortened = None;
        for round in 0..REQUEST_WINDOW {
            let depth = policy.plan(&[(1, 7)], 64)[0];
            policy.observe(&[outcome(1, depth, accepted[round].min(depth))], false);
            if policy.plan(&[(1, 7)], 64)[0] <= 3 { shortened = Some(round + 1); break; }
        }
        assert!(shortened.is_some(), "code-conditioned prior prevents request adaptation");
    }

    #[test]
    fn a_suppressed_position_recovers_within_the_request_window() {
        let mut policy = policy(5);
        for _ in 0..REQUEST_WINDOW {
            policy.plan(&[(1, 5)], 64);
            policy.observe(&[outcome(1, 5, 1)], false);
        }
        // Position 1 is suppressed for the request and for the server.
        assert_eq!(policy.plan(&[(1, 5)], 64), [2]);
        policy.observe(&[outcome(1, 2, 1)], false);
        // The drafter turns good: every explore of position 1 is now accepted.
        let mut rounds = 0;
        loop {
            let length = policy.plan(&[(1, 5)], 64)[0];
            if policy.requests[&1].pending.as_ref().unwrap().keep >= 2 { policy.observe(&[outcome(1, length, length)], false); break; }
            assert_eq!(length, 2);
            policy.observe(&[outcome(1, length, length)], false);
            rounds += 1;
            assert!(rounds <= REQUEST_WINDOW, "position 1 still suppressed after a full window");
        }
        // And the block refills one explored position at a time.
        for _ in 0..REQUEST_WINDOW {
            let length = policy.plan(&[(1, 5)], 64)[0];
            policy.observe(&[outcome(1, length, length)], false);
        }
        assert_eq!(policy.plan(&[(1, 5)], 64), [5]);
    }

    #[test]
    fn rounds_without_a_reach_age_out_of_the_request_window() {
        let mut policy = policy(4);
        for _ in 0..REQUEST_WINDOW {
            policy.plan(&[(1, 4)], 64);
            policy.observe(&[outcome(1, 4, 0)], false);
        }
        assert_eq!(policy.requests[&1].counts(NEURAL, 0), (0, 8));
        // Eight rounds with no room (no drafts) push the misses out.
        for _ in 0..REQUEST_WINDOW {
            assert_eq!(policy.plan(&[(1, 0)], 64), [0]);
            policy.observe(&[outcome(1, 0, 0)], false);
        }
        assert_eq!(policy.requests[&1].counts(NEURAL, 0), (0, 0));
    }

    #[test]
    fn no_request_stays_at_one_draft_when_real_acceptance_is_high() {
        for block in [3, 5, 7, 8, 16] {
            let mut policy = policy(block);
            let mut rng = Rng(block as u64);
            // Start from the worst state: every window says position 0 always misses.
            for _ in 0..SERVER_WINDOW {
                policy.plan(&[(1, block)], 64);
                policy.observe(&[outcome(1, block, 0)], false);
            }
            assert_eq!(policy.plan(&[(1, block)], 64), [1]);
            policy.observe(&[outcome(1, 1, 0)], false);
            let (mut short, mut drafts, mut fixed) = (0, 0, 0);
            let rounds = 2000;
            for round in 0..rounds {
                let length = policy.plan(&[(1, block)], 64)[0];
                let run = rng.accepted(block, 0.92);
                policy.observe(&[outcome(1, length, run.min(length))], false);
                if round >= 400 {
                    short += usize::from(length <= 1);
                    drafts += run.min(length);
                    fixed += run;
                }
            }
            assert!(short * 50 < rounds - 400, "block {block}: {short} short rounds");
            assert!(drafts as f64 >= 0.97 * fixed as f64, "block {block}: kept {drafts} of {fixed}");
        }
    }

    #[test]
    fn the_server_window_rolls() {
        let mut policy = policy(2);
        for _ in 0..SERVER_WINDOW {
            policy.plan(&[(1, 1)], 64);
            policy.observe(&[outcome(1, 1, 0)], false);
            policy.release(1);
        }
        assert_eq!(policy.probability(2, 0), 0.);
        for round in 1..=SERVER_WINDOW {
            policy.plan(&[(1, 1)], 64);
            policy.observe(&[outcome(1, 1, 1)], false);
            policy.release(1);
            assert!((policy.probability(2, 0) - round as f64 / SERVER_WINDOW as f64).abs() < 1e-12);
        }
        assert_eq!(policy.server[NEURAL][0].len, SERVER_WINDOW);
        // Position 1 was never reached: still cold.
        assert_eq!(policy.server[NEURAL][1].len, 0);
        assert_eq!(policy.stats().position_reached, [2 * SERVER_WINDOW as u64, 0]);
    }

    #[test]
    fn censored_rounds_and_unreached_positions_are_not_misses() {
        let mut policy = policy(5);
        policy.plan(&[(1, 5)], 64);
        // Accepted two, then a stop token: position 2 was never judged.
        policy.observe(&[DraftOutcome { id: 1, proposed: 5, accepted: 2, censored: true }], false);
        assert_eq!(policy.stats().position_reached, [1, 1, 0, 0, 0]);
        assert_eq!(policy.stats().position_accepted, [1, 1, 0, 0, 0]);
        policy.plan(&[(1, 5)], 64);
        // A real miss at position 2: positions 3 and 4 stay unreached.
        policy.observe(&[outcome(1, 5, 2)], false);
        assert_eq!(policy.stats().position_reached, [2, 2, 1, 0, 0]);
        assert_eq!(policy.stats().position_accepted, [2, 2, 0, 0, 0]);
        assert_eq!(policy.requests[&1].counts(NEURAL, 2), (0, 1));
        assert_eq!(policy.requests[&1].counts(NEURAL, 3), (0, 0));
        // Grammar truncation: only what was verified is booked.
        policy.plan(&[(1, 5)], 64);
        policy.observe(&[outcome(1, 1, 1)], false);
        assert_eq!(policy.stats().position_reached, [3, 2, 1, 0, 0]);
    }

    /// A head drafter whose position 1 has been explored and missed eight times. The
    /// rounds report no scores, so the calibration is still the identity.
    fn suppressed_head_policy() -> DraftHistory {
        let spec = DrafterSpec { block: 4, head: HeadScore::Probability, copy: 0 };
        let mut policy = DraftHistory::new(spec, DraftControl::History);
        for _ in 0..REQUEST_WINDOW {
            policy.plan(&[(1, 4)], 64);
            policy.refine(1, 4, None);
            policy.observe(&[outcome(1, 4, 1)], false);
        }
        assert_eq!(policy.stats().head_skips, 0);
        policy
    }

    #[test]
    fn the_head_skips_only_explores_calibrated_below_the_cut() {
        let mut policy = suppressed_head_policy();
        // Position 1 is the explore. The fit starts at the identity and learns from each
        // verified explore, so the cut is checked against the calibrated value in force.
        assert!(policy.platt[1].apply(logit(HeadScore::Probability, 0.29)) < SKIP_CUT);
        assert!(policy.platt[1].apply(logit(HeadScore::Probability, 0.31)) >= SKIP_CUT);
        let mut skipped = 0;
        for score in [0.02, 0.29, 0.31, 0.6, 0.9, 0.99] {
            assert_eq!(policy.plan(&[(1, 4)], 64), [2]);
            let calibrated = policy.platt[1].apply(logit(HeadScore::Probability, score));
            let kept = calibrated >= SKIP_CUT;
            let length = if kept { 2 } else { 1 };
            assert_eq!(policy.refine(1, 2, Some(&[0.9, score])), length, "score {score} calibrated {calibrated}");
            assert_eq!(policy.settle(&[1], 64), [length]);
            let trace = policy.observe(&[outcome(1, length, 1)], true).unwrap();
            assert_eq!(trace.requests[0].explore_skipped, (!kept).then_some(calibrated));
            skipped += u64::from(!kept);
        }
        assert!((1..6).contains(&skipped), "{skipped} of 6 skipped");
        assert_eq!(policy.stats().head_skips, skipped);
        // A kept position is never skipped, whatever the head says.
        let spec = DrafterSpec { block: 4, head: HeadScore::Probability, copy: 0 };
        let mut policy = DraftHistory::new(spec, DraftControl::History);
        assert_eq!(policy.plan(&[(1, 4)], 64), [4]);
        assert_eq!(policy.refine(1, 4, Some(&[0.01; 4])), 4);
        // Without a head the same scores change nothing.
        let mut policy = self::policy(4);
        for _ in 0..REQUEST_WINDOW {
            policy.plan(&[(1, 4)], 64);
            policy.observe(&[outcome(1, 4, 1)], false);
        }
        assert_eq!(policy.plan(&[(1, 4)], 64), [2]);
        assert_eq!(policy.refine(1, 2, Some(&[0.9, 0.01])), 2);
    }

    #[test]
    fn skipped_explore_trace_retains_the_decision_and_never_claims_a_verified_miss() {
        let mut policy = suppressed_head_policy();
        assert_eq!(policy.plan(&[(1, 4)], 64), [2]);
        let before = policy.platt[0].apply(logit(HeadScore::Probability, 0.9));
        assert_eq!(policy.refine(1, 2, Some(&[0.9, 0.02])), 1);
        let trace = policy.observe(&[outcome(1, 1, 1)], true).unwrap();
        let request = &trace.requests[0];
        assert_eq!(request.proposed, 1);
        assert_eq!(request.positions.len(), 2);
        assert_eq!(request.positions[0].calibrated, Some(before));
        let skipped = &request.positions[1];
        assert_eq!((skipped.position, skipped.head, skipped.outcome), (1, Some(0.02), None));
        assert!(skipped.explore && skipped.reached > 0);
        assert_eq!(skipped.calibrated, request.explore_skipped);
        assert_eq!(policy.stats().skip_misses, 1);
    }

    #[test]
    fn a_skipped_explore_is_booked_as_a_miss_at_its_position() {
        let mut policy = suppressed_head_policy();
        let skip = |policy: &mut DraftHistory, outcome: DraftOutcome| {
            assert_eq!(policy.plan(&[(1, 4)], 64), [2]);
            assert_eq!(policy.refine(1, 2, Some(&[0.9, 0.02])), 1);
            policy.observe(&[outcome], false);
        };
        let server = |policy: &DraftHistory| (policy.server[NEURAL][1].ones, policy.server[NEURAL][1].len);
        let (calibration, verified) = (policy.platt[1].theta, policy.stats().position_reached[1]);
        assert_eq!((policy.requests[&1].counts(NEURAL, 1), server(&policy)), ((0, 8), (0, 8)));
        // The kept position was accepted: the skipped explore counts as proposed 2, accepted 1.
        skip(&mut policy, outcome(1, 1, 1));
        assert_eq!((policy.requests[&1].counts(NEURAL, 1), server(&policy)), ((0, 8), (0, 9)));
        assert_eq!(policy.requests[&1].rounds.back().unwrap().reached[NEURAL], 2);
        assert_eq!((policy.stats().head_skips, policy.stats().skip_misses), (1, 1));
        // It is not a verified row and teaches the calibration nothing.
        assert_eq!((policy.platt[1].theta, policy.stats().position_reached[1]), (calibration, verified));
        assert_eq!(policy.stats().explored, 7);
        // The kept position itself missed: position 1 was never reached, skip or not.
        skip(&mut policy, outcome(1, 1, 0));
        assert_eq!(server(&policy), (0, 9));
        assert_eq!(policy.requests[&1].rounds.back().unwrap().reached[NEURAL], 1);
        // A censored round (stop token on the kept draft) books nothing either.
        skip(&mut policy, DraftOutcome { id: 1, proposed: 1, accepted: 1, censored: true });
        assert_eq!((server(&policy), policy.stats().skip_misses), ((0, 9), 1));
        // Without the booking the request's window would have aged the misses out and
        // drafted deeper: after eight skipped rounds position 1 is still suppressed.
        for _ in 0..REQUEST_WINDOW { skip(&mut policy, outcome(1, 1, 1)); }
        assert_eq!(policy.requests[&1].counts(NEURAL, 1), (0, 8));
        assert!(policy.probability(1, 1) <= KEEP_CUT);
    }

    #[test]
    fn head_calibration_trains_on_reached_positions_only() {
        let spec = DrafterSpec { block: 3, head: HeadScore::Logit, copy: 0 };
        let mut policy = DraftHistory::new(spec, DraftControl::Fixed(3));
        let before = policy.platt[2].theta;
        // An overconfident head at position 0: says 0.88, accepted half the time.
        for round in 0..400 {
            policy.plan(&[(1, 3)], 64);
            policy.refine(1, 3, Some(&[2., 2., 2.]));
            policy.observe(&[outcome(1, 3, round % 2)], false);
        }
        let calibrated = policy.platt[0].apply(2.);
        assert!((calibrated - 0.5).abs() < 0.1, "calibrated {calibrated}");
        // Position 1 was reached on accepted rounds only and always missed; position 2 never.
        assert!(policy.platt[1].apply(2.) < 0.2);
        assert_eq!(policy.platt[2].theta, before);
    }

    #[test]
    fn the_row_budget_goes_to_the_most_probable_positions() {
        let mut policy = policy(4);
        // Request 1 accepts everything, request 2 only position 0, request 3 nothing.
        for _ in 0..REQUEST_WINDOW {
            policy.plan(&[(1, 4), (2, 4), (3, 4)], 64);
            policy.observe(&[outcome(1, 4, 4), outcome(2, 4, 1), outcome(3, 4, 0)], false);
        }
        let requests = [(1, 4), (2, 4), (3, 4)];
        // Unbound: 4 kept, 1 kept + explore, explore alone.
        assert_eq!(policy.plan(&requests, 64), [4, 2, 1]);
        policy.observe(&[outcome(1, 4, 4), outcome(2, 2, 1), outcome(3, 1, 0)], false);
        // Three anchors plus five drafts: the explores (lowest probability) go first.
        assert_eq!(policy.plan(&requests, 8), [4, 1, 0]);
        policy.observe(&[outcome(1, 4, 4), outcome(2, 1, 1), outcome(3, 0, 0)], false);
        // Three drafts: request 2's first position outranks request 1's second.
        let p = |policy: &DraftHistory, id, position| policy.probability(id, position);
        assert!(p(&policy, 2, 0) > p(&policy, 1, 1) && p(&policy, 1, 1) > p(&policy, 2, 1));
        assert_eq!(policy.plan(&requests, 6), [2, 1, 0]);
        let cut = policy.stats().budget_cut;
        assert!(cut >= 4);
        policy.observe(&[outcome(1, 2, 2), outcome(2, 1, 1), outcome(3, 0, 0)], false);
        // Anchors alone.
        assert_eq!(policy.plan(&requests, 3), [0, 0, 0]);
        // Equal probabilities alternate, the older request first.
        let mut policy = self::policy(4);
        assert_eq!(policy.plan(&[(7, 4), (5, 4)], 5), [1, 2]);
    }

    #[test]
    fn fixed_and_off_ignore_history_but_keep_recording_it() {
        let spec = DrafterSpec { block: 7, head: HeadScore::None, copy: 32 };
        let mut fixed = DraftHistory::new(spec, DraftControl::Fixed(9));
        assert_eq!(fixed.control(), DraftControl::Fixed(7));
        for _ in 0..REQUEST_WINDOW {
            assert_eq!(fixed.plan(&[(1, 7), (2, 3)], 64), [7, 3]);
            assert_eq!(fixed.extend_copy(1, 32), 0);
            fixed.observe(&[outcome(1, 7, 0), outcome(2, 3, 0)], false);
        }
        assert_eq!(fixed.stats().position_reached[0], 16);
        assert_eq!(fixed.stats().explored, 0);
        // The row budget still caps a fixed width.
        assert_eq!(fixed.plan(&(0..16).map(|id| (id, 7)).collect::<Vec<_>>(), 24).iter().sum::<usize>(), 8);
        let mut off = DraftHistory::new(spec, DraftControl::Off);
        assert_eq!(off.plan(&[(1, 7)], 64), [0]);
        assert_eq!(off.max_positions(), 0);
    }

    #[test]
    fn copies_are_a_second_track_that_starts_at_one_explored_position() {
        let spec = DrafterSpec { block: 4, head: HeadScore::None, copy: 32 };
        let mut policy = DraftHistory::new(spec, DraftControl::History);
        let mut lengths = Vec::new();
        for _ in 0..40 {
            assert_eq!(policy.plan(&[(1, 64)], 64), [4]);
            assert_eq!(policy.refine(1, 4, None), 4);
            let copy = policy.extend_copy(1, 32);
            lengths.push(copy);
            assert_eq!(policy.settle(&[1], 64), [4 + copy]);
            policy.observe(&[outcome(1, 4 + copy, 4 + copy)], false);
        }
        assert_eq!(lengths[0], 1);
        assert!(lengths.windows(2).all(|pair| pair[1] >= pair[0] && pair[1] <= pair[0] + 1), "{lengths:?}");
        assert_eq!(*lengths.last().unwrap(), 32);
        assert_eq!(policy.stats().copy_reached[0], 40);
        // A neural miss leaves the copy track untouched.
        policy.plan(&[(1, 64)], 64);
        policy.extend_copy(1, 32);
        policy.settle(&[1], 64);
        let before = policy.stats().copy_reached.clone();
        policy.observe(&[outcome(1, 36, 2)], false);
        assert_eq!(policy.stats().copy_reached, before);
        // Copy rejections shorten the copy, never the neural draft.
        for _ in 0..REQUEST_WINDOW {
            policy.plan(&[(1, 64)], 64);
            let copy = policy.extend_copy(1, 32);
            policy.settle(&[1], 64);
            policy.observe(&[outcome(1, 4 + copy, 4)], false);
        }
        assert_eq!(policy.plan(&[(1, 64)], 64), [4]);
        assert_eq!(policy.extend_copy(1, 32), 1);
        // The copy shares the round's row budget and the request's room.
        assert_eq!(policy.settle(&[1], 5), [4]);
        policy.observe(&[outcome(1, 4, 4)], false);
        policy.plan(&[(1, 4)], 64);
        assert_eq!(policy.extend_copy(1, 32), 0);
    }

    #[test]
    fn interleaved_lanes_keep_their_own_plans() {
        let mut policy = policy(5);
        assert_eq!(policy.plan(&[(1, 5)], 64), [5]);
        assert_eq!(policy.plan(&[(2, 2)], 64), [2]);
        policy.observe(&[outcome(1, 5, 5)], false);
        let trace = policy.observe(&[outcome(2, 2, 0)], true).unwrap();
        assert_eq!((trace.requests[0].room, trace.requests[0].neural), (2, 2));
        policy.release(1);
        policy.release(2);
        assert!(policy.requests.is_empty());
    }

    #[test]
    fn trace_records_probability_counts_explore_and_head() {
        let spec = DrafterSpec { block: 3, head: HeadScore::Logit, copy: 0 };
        let mut policy = DraftHistory::new(spec, DraftControl::History);
        for _ in 0..4 {
            policy.plan(&[(1, 3)], 64);
            policy.refine(1, 3, Some(&[3., 1., 0.]));
            policy.observe(&[outcome(1, 3, 1)], false);
        }
        assert_eq!(policy.plan(&[(1, 3)], 64), [2]);
        policy.refine(1, 2, Some(&[3., 1.]));
        let trace = policy.observe(&[outcome(1, 2, 2)], true).unwrap();
        let value = serde_json::to_value(&trace).unwrap();
        assert_eq!(value["policy"], "history");
        let explore = &value["requests"][0]["positions"][1];
        assert_eq!(explore["track"], "neural");
        assert_eq!((explore["accepted"].as_u64(), explore["reached"].as_u64()), (Some(0), Some(4)));
        assert_eq!(explore["server"], 0.);
        assert_eq!(explore["server_reached"], 4);
        assert_eq!(explore["p"], 0.);
        assert_eq!(explore["explore"], true);
        assert_eq!(explore["head"], 1.);
        assert!(explore["calibrated"].as_f64().unwrap() > 0.);
        assert_eq!(explore["outcome"], true);
        let snapshot = policy.snapshot();
        assert_eq!(snapshot["positions"][0]["reached"], 5);
        assert_eq!(snapshot["explored"], 4);
        // One of five rounds drafted a single position with room for three.
        assert_eq!((value["rounds"].as_u64(), value["short_rounds"].as_u64()), (Some(5), Some(0)));
        assert_eq!(value["short_share"], 0.);
        assert_eq!(snapshot["short_share"], 0.);
    }
}
