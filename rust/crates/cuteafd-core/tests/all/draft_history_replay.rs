//! Replay of recorded fixed-width draft rounds through the `history` policy.
//!
//! The fixture is 1,758 neural rounds of a V4.1 Flash fixed-7 card trace
//! (every position verified, so any shorter plan's outcome is known exactly:
//! it accepts `min(length, accepted)` drafts): twelve sustained prose requests
//! (1,112 rounds) followed by ten code requests (646 rounds). Per round: the
//! request, its content, how many of its seven drafts the target accepted, and
//! the dSpark head's sigmoid per position. Conditional acceptance at positions
//! 1-7 is .91 .87 .83 .82 .80 .76 .73 on the code and .55 .47 .43 .44 .43 .43
//! .70 on the prose.
//!
//! Reference, over the full traces (tokens kept against fixed width / drafts
//! verified per round / rounds at one draft or fewer):
//! - code-heavy fixed-width cards: Kimi K6 fixed 5, 9,608 rounds, 99.91% /
//!   4.98 / 0.14%; V4.1 fixed 5, 4,314 rounds, 99.87% / 4.94 / 0.63%; MiniMax
//!   fixed 7, 2,399 rounds, 99.62% / 6.85 / 0.96%. (The rule as first replayed
//!   with a 3-in-4 cold server prior: 99.9 / 4.98 / 0.1, 99.9 / 4.95 / 0.6,
//!   99.5 / 6.79 / 1.0.)
//! - this fixture's trace by content (37k rounds): code 99.4% at 6.87 of 7
//!   drafts, none short; prose 91.2% at 2.19 of 7 drafts (rejected rows per
//!   round 1.42, against about 6 at fixed width), 26% of rounds short.
//! - the dSpark explore skip (calibrated head probability below `SKIP_CUT`,
//!   0.3), booked as a miss at the skipped position, replayed on that trace:
//!   18.5% fewer verify rows for 3.2% fewer tokens; tokens per verified row
//!   0.706 -> 0.839; of the explores skipped, 9,572 would have failed and
//!   1,478 passed. Prose rounds at one draft or fewer rise from 26% to 59%:
//!   the head keeps saying those explores would miss, it is not a lock (the
//!   explore returns as soon as the head scores it at 0.3 or above). At 0.2:
//!   8.5% fewer rows (13.8% on prose) for 1.1% fewer tokens. Without the
//!   booking the skip left the position's history unchanged, later rounds
//!   drafted deeper and rows rose 0.8%.
use cuteafd_core::{DraftControl, DraftHistory, DraftOutcome, DrafterSpec, HeadScore};

const BLOCK: usize = 7;

struct Round {
    id: u64,
    code: bool,
    accepted: usize,
    head: Vec<f32>,
}

fn rounds() -> Vec<Round> {
    include_str!("../fixtures/v41-fixed7-draft-rounds.jsonl").lines().map(|line| {
        let value: serde_json::Value = serde_json::from_str(line).unwrap();
        Round { id: value["id"].as_u64().unwrap(), code: value["content"] == "code",
            accepted: value["accepted"].as_u64().unwrap() as usize,
            head: value["head"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect() }
    }).collect()
}

/// Per content: rounds, tokens emitted, drafts verified, rounds at one draft or fewer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Tally {
    rounds: usize,
    tokens: usize,
    drafts: usize,
    short: usize,
}

/// (prose, code) tallies and the head's skips.
fn replay(head: HeadScore, control: DraftControl) -> ([Tally; 2], u64) {
    let mut policy = DraftHistory::new(DrafterSpec { block: BLOCK, head, copy: 0 }, control);
    let mut tallies = [Tally::default(); 2];
    for round in rounds() {
        policy.plan(&[(round.id, BLOCK)], 64);
        let length = policy.refine(round.id, BLOCK, Some(&round.head));
        assert_eq!(policy.settle(&[round.id], 64), [length]);
        let accepted = round.accepted.min(length);
        policy.observe(&[DraftOutcome { id: round.id, proposed: length, accepted, censored: false }], false);
        let tally = &mut tallies[usize::from(round.code)];
        tally.rounds += 1;
        tally.tokens += accepted + 1;
        tally.drafts += length;
        tally.short += usize::from(length <= 1);
    }
    assert_eq!(policy.stats().short_rounds as usize, tallies[0].short + tallies[1].short);
    (tallies, policy.stats().head_skips)
}

/// Independent headless mirror keeps the old prior-eight counterfactual
/// available without exposing a runtime policy knob or changing fixture data.
fn code_first_reference(fade: bool) -> [Tally; 2] {
    use std::collections::{HashMap, VecDeque};
    let mut rounds = rounds();
    rounds.sort_by_key(|round| !round.code); // Stable: preserve every request's round order.
    let mut server: [VecDeque<bool>; BLOCK] = std::array::from_fn(|_| VecDeque::new());
    let mut requests: HashMap<u64, VecDeque<(usize, usize)>> = HashMap::new();
    let mut completed: HashMap<u64, usize> = HashMap::new();
    let mut tallies = [Tally::default(); 2];
    for round in rounds {
        let n = completed.entry(round.id).or_default();
        let prior = if fade { 8. * (3usize.saturating_sub(*n) as f64 / 3.) } else { 8. };
        *n += 1;
        let request = requests.entry(round.id).or_default();
        let mut length = 0;
        for position in 0..BLOCK {
            let reached = request.iter().filter(|(reached, _)| position < *reached).count();
            let accepted = request.iter().filter(|(reached, accepted)| position < *reached && position < *accepted).count();
            let global = if server[position].is_empty() { 1. } else {
                server[position].iter().filter(|&&hit| hit).count() as f64 / server[position].len() as f64 };
            length += 1;
            let denominator = reached as f64 + prior;
            if denominator == 0. || (accepted as f64 + prior * global) / denominator <= if fade { 0.3 } else { 0.5 } { break; }
        }
        let accepted = round.accepted.min(length);
        let reached = length.min(accepted + 1);
        if request.len() == 8 { request.pop_front(); }
        request.push_back((reached, accepted));
        for position in 0..reached {
            if server[position].len() == 256 { server[position].pop_front(); }
            server[position].push_back(position < accepted);
        }
        {
            let tally = &mut tallies[usize::from(round.code)];
            tally.rounds += 1;
            tally.drafts += length;
            tally.tokens += accepted + 1;
            tally.short += usize::from(length <= 1);
        }
    }
    tallies
}

#[test]
fn code_first_requests_adapt_with_fewer_rows_without_losing_prose_tokens() {
    let mut rounds = rounds();
    rounds.sort_by_key(|round| !round.code);
    let mut policy = DraftHistory::new(DrafterSpec { block: BLOCK, head: HeadScore::None, copy: 0 }, DraftControl::History);
    let mut tallies = [Tally::default(); 2];
    for round in rounds {
        let length = policy.plan(&[(round.id, BLOCK)], 64)[0];
        let accepted = round.accepted.min(length);
        policy.observe(&[DraftOutcome { id: round.id, proposed: length, accepted, censored: false }], false);
        {
            let prose = &mut tallies[usize::from(round.code)];
            prose.rounds += 1;
            prose.drafts += length;
            prose.tokens += accepted + 1;
            prose.short += usize::from(length <= 1);
        }
    }
    assert_eq!(tallies, code_first_reference(true), "reference mirror must match production");
    let old = code_first_reference(false);
    let rows = |tally: Tally| (tally.drafts + tally.rounds) as f64 / tally.rounds as f64;
    let tokens = |tally: Tally| tally.tokens as f64 / tally.rounds as f64;
    // Stable unchanged sample: fade3/stop.3 prose rows4.0935 vs constant8 5.1124
    // (0.801x), code tokens3108/3138 (0.990x), prose2087/2030 (1.028x).
    eprintln!("code-first fade3-stop03 {tallies:?}; constant8 {old:?}");
    assert!(rows(tallies[0]) <= 0.90 * rows(old[0]), "prose rows must fall by at least ten percent");
    assert!(tokens(tallies[1]) >= 0.95 * tokens(old[1]), "code tokens must retain ninety-five percent");
    assert!(tokens(tallies[0]) >= 0.95 * tokens(old[0]), "prose tokens must retain ninety-five percent");
}

#[test]
fn history_drafts_long_on_code_and_short_on_prose() {
    let ([fixed_prose, fixed_code], _) = replay(HeadScore::None, DraftControl::Fixed(BLOCK));
    assert_eq!(fixed_prose, Tally { rounds: 1112, tokens: 2219, drafts: 7784, short: 0 });
    assert_eq!(fixed_code, Tally { rounds: 646, tokens: 3154, drafts: 4522, short: 0 });
    let ([prose, code], skips) = replay(HeadScore::None, DraftControl::History);
    assert_eq!(skips, 0);
    // Fade3/stop.3: code keeps 98.4% of fixed tokens at 6.51 drafts, never stuck short,
    // although it starts from a server window that 1,112 prose rounds taught.
    assert_eq!(code, Tally { rounds: 646, tokens: 3104, drafts: 4207, short: 0 });
    // Prose: 94.1% of fixed tokens for 43.6% of its verified drafts.
    assert_eq!(prose, Tally { rounds: 1112, tokens: 2087, drafts: 3395, short: 126 });
    // Measured prose token retention .941 with .436 of fixed drafted rows;
    // code retention .984. KEEP_CUT=.3 deliberately retains more useful prose.
    assert!(prose.tokens * 100 >= fixed_prose.tokens * 90);
    assert!(prose.drafts * 2 <= fixed_prose.drafts);
    assert!(code.tokens * 100 >= fixed_code.tokens * 97);
    assert!(code.drafts * prose.rounds > prose.drafts * code.rounds);
}

/// The dSpark head skip, booked as a miss: bounds, not exact counts (the fit
/// is floating point). It fires on prose, not on code, and spends fewer verify
/// rows on prose for nearly the same tokens. With fade3/stop.3, this
/// sample has 641 skips: prose drafts 3,395 -> 2,591 and tokens 2,087 ->
/// 2,039; code drafts 4,207 -> 4,189 and tokens 3,104 -> 3,104.
#[test]
fn the_head_skip_saves_prose_rows_and_leaves_code_alone() {
    let ([prose, code], _) = replay(HeadScore::None, DraftControl::History);
    let ([skip_prose, skip_code], skips) = replay(HeadScore::Probability, DraftControl::History);
    eprintln!("head skip: {skips} skips; prose {prose:?} -> {skip_prose:?}; code {code:?} -> {skip_code:?}");
    assert!(skips >= 50, "{skips} skips");
    assert!(skip_code.tokens.abs_diff(code.tokens) * 100 <= code.tokens);
    assert!(skip_code.drafts.abs_diff(code.drafts) * 50 <= code.drafts);
    assert!(skip_prose.drafts < prose.drafts, "prose drafts {} -> {}", prose.drafts, skip_prose.drafts);
    assert!(skip_prose.tokens * 100 >= prose.tokens * 90, "prose tokens {} -> {}", prose.tokens, skip_prose.tokens);
    // More prose rounds stop at one draft, and the explore still comes back: not a lock.
    assert!(skip_prose.short > prose.short && skip_prose.short < skip_prose.rounds);
    // Tokens per verified row (anchors included) do not fall.
    let efficiency = |tally: Tally| tally.tokens as f64 / (tally.drafts + tally.rounds) as f64;
    assert!(efficiency(skip_prose) >= efficiency(prose));
}
