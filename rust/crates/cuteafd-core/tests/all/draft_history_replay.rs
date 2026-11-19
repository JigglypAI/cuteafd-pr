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
//! The replay compares fixed width, the count-based fallback, and scored
//! history with online per-position calibration.
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

/// (prose, code) tallies.
fn replay(head: HeadScore, control: DraftControl) -> [Tally; 2] {
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
    assert_eq!(policy.stats().verified_drafts as usize, tallies[0].drafts + tallies[1].drafts);
    tallies
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
    let [fixed_prose, fixed_code] = replay(HeadScore::None, DraftControl::Fixed(BLOCK));
    assert_eq!(fixed_prose, Tally { rounds: 1112, tokens: 2219, drafts: 7784, short: 0 });
    assert_eq!(fixed_code, Tally { rounds: 646, tokens: 3154, drafts: 4522, short: 0 });
    let [prose, code] = replay(HeadScore::None, DraftControl::History);
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

#[test]
fn calibrated_confidence_replays_prose_and_code() {
    let [prose, code] = replay(HeadScore::Probability, DraftControl::History);
    assert_eq!(prose, Tally { rounds: 1112, tokens: 2158, drafts: 3142, short: 287 });
    assert_eq!(code, Tally { rounds: 646, tokens: 3144, drafts: 4291, short: 3 });
}
