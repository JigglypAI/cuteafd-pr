//! The `history` policy keeps the leading drafts whose individual token scores
//! clear the cut, plus one explore. Trained heads use an online per-position
//! Platt fit; DFlash2 uses raw candidate probability. Token-only drafters and
//! history copies retain the rolling per-position acceptance rule, with a
//! server prior fading over the first three request rounds.
//!
//! Plan before drafting, refine from the actual scores, optionally extend with
//! an agreeing copy, then settle the verify-row budget by highest next score
//! while preserving each request's prefix. Learn only from target-judged rows.
use serde::Serialize;
use std::collections::{HashMap, VecDeque};

/// Verify rounds of a request's own history.
pub const REQUEST_WINDOW: usize = 8;
/// Rounds that reached a position, server wide.
pub const SERVER_WINDOW: usize = 256;
/// Initial server pseudo-rounds before request-local evidence takes over.
/// V4.1 full-trace card-order replay (20 requests/content):
/// fade | prose tokens/rows | code-after-prose tokens/rows | mixed tokens/rows
/// none | 1.97/6.77         | 4.96/6.61                    | 2.84/6.93
/// 8    | 1.71/1.99         | 4.68/5.58                    | 2.50/3.20
/// 3    | 1.70/1.84         | 4.65/5.41                    | 2.47/2.96
/// 1    | 1.70/1.73         | 4.64/5.36                    | 2.43/2.77
pub const PRIOR_STRENGTH: f64 = 8.;
/// Completed verify rounds until server influence is exactly zero.
pub const FADE_ROUNDS: u64 = 3;
/// A position is kept while its probability is above this.
pub const KEEP_CUT: f64 = 0.3;
// #2460: fixed-width replay favors a shared cut for each kind of score.
const TRAINED_HEAD_KEEP_CUT: f64 = 0.3;
const CANDIDATE_KEEP_CUT: f64 = 0.5;
/// Server rate of a neural position nobody has reached yet.
pub const NEURAL_COLD_RATE: f64 = 1.;
/// Server rate of an unreached copy position: below KEEP_CUT=0.3 so a
/// cold agreeing copy verifies only its explored first row, never all 32.
pub const COPY_COLD_RATE: f64 = 0.25;

/// What a drafter reports per drafted position.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum HeadScore {
    /// No per-token score; use position history.
    None,
    /// The head's raw logit.
    Logit,
    /// The head's sigmoid.
    Probability,
    /// Raw, uncalibrated probability from the drafter's token selector.
    CandidateProbability,
}

impl HeadScore {
    fn trained(self) -> bool { matches!(self, Self::Logit | Self::Probability) }

    fn cut(self) -> Option<f64> {
        match self {
            Self::None => None,
            Self::Logit | Self::Probability => Some(TRAINED_HEAD_KEEP_CUT),
            Self::CandidateProbability => Some(CANDIDATE_KEEP_CUT),
        }
    }
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
    /// Per-position history estimate, whether used for planning or not.
    pub p: f64,
    /// Per-token score used by scored drafters.
    pub score: Option<f64>,
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
    /// Explores judged and accepted so far.
    pub explored: u64,
    pub explore_accepted: u64,
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
    score: Option<f64>,
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
    /// Planned positions per track, and how many of them the budget granted.
    planned: [Vec<Planned>; 2],
    granted: [usize; 2],
    head: Vec<f32>,
}

#[derive(Clone, Debug, Default)]
struct Request {
    completed_rounds: u64,
    rounds: VecDeque<Round>,
    pending: Option<Pending>,
}

impl Request {
    fn prior_strength(&self) -> f64 {
        PRIOR_STRENGTH * (FADE_ROUNDS.saturating_sub(self.completed_rounds) as f64 / FADE_ROUNDS as f64)
    }

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
            platt: if spec.head.trained() { vec![Platt::new(); spec.block] } else { Vec::new() },
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
        let prior = request.prior_strength();
        let denominator = f64::from(reached) + prior;
        // Unknown after the fade: stop here and explore this position.
        let p = if denominator == 0. { 0. } else { (f64::from(accepted) + prior * server) / denominator };
        Planned { p, score: None,
            accepted, reached, server, server_reached: window.len as u32 }
    }

    /// The history estimate for `id` at a neural position now.
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
    /// the drafts to ask the drafter for, in request order. With per-token
    /// scores the verify budget is applied only after the draft pass.
    pub fn plan(&mut self, requests: &[(u64, usize)], max_rows: usize) -> Vec<usize> {
        let scored = self.control == DraftControl::History && self.spec.head.cut().is_some();
        let mut depths = Vec::with_capacity(if scored { requests.len() } else { 0 });
        for &(id, room) in requests {
            let request = self.requests.remove(&id).unwrap_or_default();
            let limit = room.min(self.spec.block);
            let (planned, keep) = match self.control {
                DraftControl::Off => (Vec::new(), 0),
                DraftControl::History if scored => {
                    ((0..limit).map(|position| self.position(&request, NEURAL, position)).collect(), 0)
                }
                DraftControl::History => self.track_plan(&request, NEURAL, limit),
                DraftControl::Fixed(n) => {
                    let n = n.min(limit);
                    ((0..n).map(|position| self.position(&request, NEURAL, position)).collect(), n)
                }
            };
            let granted = planned.len();
            if scored { depths.push(granted); }
            let pending = Pending { room, keep, planned: [planned, Vec::new()],
                granted: [granted, 0], ..Default::default() };
            self.requests.insert(id, Request { pending: Some(pending), ..request });
        }
        // Score the candidates before spending any verify rows.
        if scored { return depths; }
        let ids: Vec<u64> = requests.iter().map(|&(id, _)| id).collect();
        self.settle(&ids, max_rows)
    }

    /// Cut the proposal to its score-based keep prefix and one explore.
    /// Token-only drafters keep their original count-based plan.
    pub fn refine(&mut self, id: u64, drafted: usize, scores: Option<&[f32]>) -> usize {
        let Some(pending) = self.requests.get_mut(&id).and_then(|request| request.pending.as_mut()) else { return 0 };
        let planned = &mut pending.planned[NEURAL];
        planned.truncate(drafted);
        if let Some(cut) = self.spec.head.cut() {
            let scores = scores.unwrap_or_default();
            if self.spec.head.trained() {
                pending.head = scores[..scores.len().min(planned.len())].to_vec();
            }
            for (position, (entry, &score)) in planned.iter_mut().zip(scores).enumerate() {
                entry.score = score.is_finite().then(|| {
                    if self.spec.head.trained() { self.platt[position].apply(logit(self.spec.head, score)) }
                    else { f64::from(score) }
                });
            }
            if self.control == DraftControl::History {
                pending.keep = planned.iter().take_while(|entry| entry.score.is_some_and(|p| p > cut)).count();
                planned.truncate(pending.keep + 1);
            }
        }
        pending.granted[NEURAL] = planned.len();
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
            if taken < wanted[index][NEURAL] {
                let entry = &pending.planned[NEURAL][taken];
                if self.control == DraftControl::History && self.spec.head.cut().is_some() {
                    entry.score.unwrap_or(0.)
                } else { entry.p }
            } else { pending.planned[COPY][taken - wanted[index][NEURAL]].p }
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
            let reached = [judged.min(neural), judged.saturating_sub(neural)];
            let kept = [accepted.min(neural), accepted.saturating_sub(neural)];
            for track in [NEURAL, COPY] {
                for position in 0..reached[track] {
                    let hit = position < kept[track];
                    self.server[track][position].push(hit);
                    let (reached, accepted) = if track == NEURAL {
                        (&mut self.stats.position_reached, &mut self.stats.position_accepted)
                    } else { (&mut self.stats.copy_reached, &mut self.stats.copy_accepted) };
                    reached[position] += 1;
                    accepted[position] += u64::from(hit);
                    if track == NEURAL && self.spec.head.trained() {
                        if let Some(&score) = pending.head.get(position).filter(|score| score.is_finite()) {
                            self.platt[position].observe(logit(self.spec.head, score), hit);
                        }
                    }
                }
            }
            let explores = self.control == DraftControl::History;
            if explores && reached[NEURAL] > pending.keep {
                self.stats.explored += 1;
                self.stats.explore_accepted += u64::from(kept[NEURAL] > pending.keep);
            }
            self.stats.rounds += 1;
            self.stats.verified_drafts += proposed as u64;
            self.stats.accepted_drafts += accepted as u64;
            self.stats.short_rounds += u64::from(proposed <= 1 && pending.room.min(self.spec.block) > 1);
            if let Some(bucket) = self.stats.length_histogram.get_mut(proposed) { *bucket += 1; }
            if request.rounds.len() == REQUEST_WINDOW { request.rounds.pop_front(); }
            request.completed_rounds = request.completed_rounds.saturating_add(1);
            request.rounds.push_back(Round { reached: reached.map(|n| n as u8), accepted: kept.map(|n| n as u8) });
            if trace {
                let mut positions = Vec::with_capacity(proposed);
                for (track, label) in [(NEURAL, Track::Neural), (COPY, Track::Copy)] {
                    for (position, planned) in pending.planned[track].iter().enumerate() {
                        let head = if track == NEURAL { pending.head.get(position).copied() } else { None };
                        positions.push(PositionTrace { track: label, position, p: planned.p, score: planned.score,
                            accepted: planned.accepted, reached: planned.reached, server: planned.server,
                            server_reached: planned.server_reached,
                            explore: explores && if track == NEURAL { position >= pending.keep }
                                else { position + 1 == pending.planned[COPY].len() && planned.p <= KEEP_CUT },
                            head, calibrated: head.and(planned.score),
                            outcome: (position < reached[track]).then_some(position < kept[track]) });
                    }
                }
                requests.push(RequestTrace { id: outcome.id, request_id: None, room: pending.room, keep: pending.keep,
                    neural: pending.granted[NEURAL], copy: pending.granted[COPY],
                    proposed, accepted_drafts: accepted, censored: outcome.censored, positions });
            }
            self.requests.insert(outcome.id, request);
        }
        trace.then(|| RoundTrace { policy: self.control.label(), block: self.spec.block,
            rounds: self.stats.rounds, short_rounds: self.stats.short_rounds, short_share: self.stats.short_share(),
            explored: self.stats.explored, explore_accepted: self.stats.explore_accepted,
            requests })
    }

    /// Forget a finished request.
    pub fn release(&mut self, id: u64) { self.requests.remove(&id); }

    /// Server-wide state for `/v1/stats`.
    pub fn snapshot(&self) -> serde_json::Value {
        let track = |track: usize, reached: &[u64], accepted: &[u64]| -> Vec<serde_json::Value> {
            self.server[track].iter().enumerate().map(|(position, window)| serde_json::json!({
                "position": position + 1, "window_rate": window.rate(Self::cold(track)), "window_rounds": window.len,
                "reached": reached[position], "accepted": accepted[position],
                "calibration": (track == NEURAL && self.spec.head.trained())
                    .then(|| self.platt[position].theta),
            })).collect()
        };
        serde_json::json!({
            "policy": self.control.label(), "block": self.spec.block, "head": self.spec.head,
            "copy_positions": self.spec.copy, "request_window": REQUEST_WINDOW, "server_window": SERVER_WINDOW,
            "prior_strength": PRIOR_STRENGTH, "fade_rounds": FADE_ROUNDS, "active_requests": self.requests.len(),
            "rounds": self.stats.rounds, "verified_drafts": self.stats.verified_drafts,
            "accepted_drafts": self.stats.accepted_drafts, "short_rounds": self.stats.short_rounds,
            "short_share": self.stats.short_share(),
            "explored": self.stats.explored, "explore_accepted": self.stats.explore_accepted,
            "budget_cut": self.stats.budget_cut,
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
        HeadScore::Logit => score.clamp(-30., 30.),
        HeadScore::None | HeadScore::CandidateProbability => unreachable!("not a trained head"),
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
        assert_eq!(policy.requests[&1].prior_strength(), 0.);
        assert!((policy.probability(1, 0) - 2. / 3.).abs() < 1e-12);
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
    fn server_prior_fades_exactly_to_zero_and_unknown_positions_explore() {
        let mut policy = policy(7);
        assert_eq!(Request::default().prior_strength(), 8.);
        for round in 0..12 {
            policy.plan(&[(1, 1)], 64);
            policy.observe(&[outcome(1, 1, 1)], false);
            let request = &policy.requests[&1];
            if round + 1 >= FADE_ROUNDS as usize { assert_eq!(request.prior_strength(), 0.); }
        }
        assert_eq!(policy.probability(1, 1), 0., "unreached suffix must not inherit the server after fade");
        assert_eq!(policy.plan(&[(1, 7)], 64), [2], "one known position plus unknown explore");
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

    #[test]
    fn trained_head_keeps_individual_calibrated_scores_and_explores_the_stop() {
        let spec = DrafterSpec { block: 4, head: HeadScore::Probability, copy: 0 };
        let mut policy = DraftHistory::new(spec, DraftControl::History);
        policy.platt[2].theta = [1., -4.];
        assert_eq!(policy.plan(&[(1, 4)], 64), [4]);
        // 0.45 * 0.45 < 0.3, yet both pass. Only calibration stops the third.
        assert_eq!(policy.refine(1, 4, Some(&[0.45, 0.45, 0.9, 0.99])), 3);
        let trace = policy.observe(&[outcome(1, 3, 2)], true).unwrap();
        let req = &trace.requests[0];
        assert_eq!((req.keep, req.proposed), (2, 3));
        let explore = &req.positions[2];
        assert!(explore.explore && explore.calibrated.unwrap() < TRAINED_HEAD_KEEP_CUT);
        assert_eq!(explore.outcome, Some(false));
        assert_eq!(policy.stats().explored, 1);
    }

    #[test]
    fn raw_probability_uses_a_strict_cut_without_calibration_or_history() {
        let spec = DrafterSpec { block: 4, head: HeadScore::CandidateProbability, copy: 0 };
        let mut policy = DraftHistory::new(spec, DraftControl::History);
        for _ in 0..REQUEST_WINDOW {
            assert_eq!(policy.plan(&[(1, 4)], 64), [4]);
            assert_eq!(policy.refine(1, 4, Some(&[0.1; 4])), 1);
            policy.observe(&[outcome(1, 1, 0)], false);
        }
        assert_eq!(policy.probability(1, 0), 0.);
        assert_eq!(policy.plan(&[(1, 4)], 64), [4]);
        assert_eq!(policy.refine(1, 4, Some(&[0.6, 0.6, 0.5, 0.99])), 3);
        let trace = policy.observe(&[outcome(1, 3, 3)], true).unwrap();
        assert_eq!(trace.requests[0].keep, 2);
        let explore = &trace.requests[0].positions[2];
        assert_eq!((explore.score, explore.calibrated, explore.outcome), (Some(0.5), None, Some(true)));
        assert!(policy.platt.is_empty());
    }

    #[test]
    fn settle_ranks_next_scored_token_and_preserves_prefixes() {
        for (head, a, b) in [(HeadScore::CandidateProbability, [0.6, 0.99], [0.9, 0.8]),
            (HeadScore::Logit, [0., 4.], [2., 1.])] {
            let mut policy = DraftHistory::new(DrafterSpec { block: 2, head, copy: 0 }, DraftControl::History);
            assert_eq!(policy.plan(&[(1, 2), (2, 2)], 4), [2, 2]);
            assert_eq!(policy.refine(1, 2, Some(&a)), 2);
            assert_eq!(policy.refine(2, 2, Some(&b)), 2);
            assert_eq!(policy.settle(&[1, 2], 4), [0, 2]);
        }
    }

    #[test]
    fn unscored_mtp_ignores_scores() {
        let mut policy = policy(4);
        for _ in 0..REQUEST_WINDOW {
            let proposed = policy.plan(&[(1, 4)], 64)[0];
            policy.observe(&[outcome(1, proposed, 1)], false);
        }
        assert_eq!(policy.plan(&[(1, 4)], 64), [2]);
        assert_eq!(policy.refine(1, 2, Some(&[0.9, 0.01])), 2);
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
        // Allocate by probability, not a fixture-specific cross-position rank.
        let mut unbounded = policy.clone();
        let wanted = unbounded.plan(&requests, 64);
        let granted = policy.plan(&requests, 6);
        assert_eq!(granted.iter().sum::<usize>(), 3);
        // Only each prefix's next row is eligible: a later p=1 cannot
        // outrank another request before its lower-p predecessor is granted.
        let mut replayed = vec![0usize; requests.len()];
        for _ in 0..granted.iter().sum::<usize>() {
            let eligible: Vec<_> = requests.iter().enumerate().filter(|(index, _)| replayed[*index] < wanted[*index])
                .map(|(index, &(id, _))| (index, unbounded.probability(id, replayed[index]))).collect();
            let best = eligible.iter().map(|(_, p)| *p).fold(f64::NEG_INFINITY, f64::max);
            let chosen = eligible.iter().filter(|(index, p)| *p == best && replayed[*index] < granted[*index])
                .min_by_key(|(index, _)| (replayed[*index], requests[*index].0))
                .expect("granted prefix must include a highest-probability eligible row").0;
            assert!(eligible.iter().all(|(_, p)| best >= *p));
            replayed[chosen] += 1;
        }
        assert_eq!(replayed, granted);
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
            assert_eq!(policy.refine(1, 3, Some(&[3., -2., 0.])), 2);
            policy.observe(&[outcome(1, 2, 1)], false);
        }
        assert_eq!(policy.plan(&[(1, 3)], 64), [3]);
        assert_eq!(policy.refine(1, 3, Some(&[3., -2., 0.])), 2);
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
        assert_eq!(explore["head"], -2.);
        assert_eq!(explore["score"], explore["calibrated"]);
        assert!(explore["calibrated"].as_f64().unwrap() < TRAINED_HEAD_KEEP_CUT);
        assert_eq!(explore["outcome"], true);
        let snapshot = policy.snapshot();
        assert_eq!(snapshot["positions"][0]["reached"], 5);
        assert_eq!(snapshot["explored"], 5);
        assert_eq!((value["rounds"].as_u64(), value["short_rounds"].as_u64()), (Some(5), Some(0)));
        assert_eq!(value["short_share"], 0.);
        assert_eq!(snapshot["short_share"], 0.);
    }
}
