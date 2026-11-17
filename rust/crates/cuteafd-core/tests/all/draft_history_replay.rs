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
//! - skipping the explore on dSpark's confidence head was replayed on that
//!   trace and rejected: at a calibrated cut of 0.2, 6,445 skips in 28,478
//!   prose rounds moved tokens 91.2% -> 91.0% and drafts 2.19 -> 2.18. A
//!   skipped explore also withholds the miss that keeps the position
//!   suppressed, so the position returns and is verified anyway. The head
//!   scores stay in the fixture as recorded data; the policy does not read them.
use cuteafd_core::{DraftControl, DraftHistory, DraftOutcome, DrafterSpec};

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

/// (prose, code) tallies, with or without reporting the recorded head scores.
fn replay(head: bool, control: DraftControl) -> [Tally; 2] {
    let mut policy = DraftHistory::new(DrafterSpec { block: BLOCK, copy: 0 }, control);
    let mut tallies = [Tally::default(); 2];
    for round in rounds() {
        policy.plan(&[(round.id, BLOCK)], 64);
        let length = policy.refine(round.id, BLOCK, head.then_some(round.head.as_slice()));
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
    tallies
}

#[test]
fn history_drafts_long_on_code_and_short_on_prose() {
    let [fixed_prose, fixed_code] = replay(false, DraftControl::Fixed(BLOCK));
    assert_eq!(fixed_prose, Tally { rounds: 1112, tokens: 2219, drafts: 7784, short: 0 });
    assert_eq!(fixed_code, Tally { rounds: 646, tokens: 3154, drafts: 4522, short: 0 });
    let [prose, code] = replay(false, DraftControl::History);
    // Code: 99.3% of fixed width's tokens at 6.78 of 7 drafts, never stuck short,
    // although it starts from a server window that 1,112 prose rounds taught.
    assert_eq!(code, Tally { rounds: 646, tokens: 3131, drafts: 4383, short: 0 });
    // Prose: 87.7% of fixed width's tokens for 28.7% of its verified drafts.
    assert_eq!(prose, Tally { rounds: 1112, tokens: 1946, drafts: 2232, short: 389 });
    assert!(code.drafts * prose.rounds > 3 * prose.drafts * code.rounds);
}

/// A drafter's head scores are data: reporting them changes no decision.
#[test]
fn recorded_head_scores_do_not_change_the_replay() {
    for control in [DraftControl::History, DraftControl::Fixed(BLOCK)] {
        assert_eq!(replay(true, control), replay(false, control));
    }
}
