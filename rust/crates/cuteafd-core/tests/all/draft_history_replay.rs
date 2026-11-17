//! Replay of recorded fixed-width draft rounds through the `history` policy.
//!
//! The fixture is the first 1,200 neural rounds (19 requests) of a V4.1 Flash
//! fixed-5 code-agent card trace: per round the request, how many of its five
//! verified drafts the target accepted, and the dSpark head's sigmoid per
//! position. Because every round verified the whole block, any shorter plan's
//! outcome is known exactly: it accepts `min(length, accepted)` drafts.
//!
//! Reference (the rule as first replayed over the full traces, with a 3-in-4
//! cold server prior; tokens kept against fixed width / verified drafts per
//! round / rounds at one draft or fewer): Kimi K6 fixed 5, 9,608 rounds, 99.9%
//! / 4.98 / 0.1%; V4.1 fixed 5, 4,314 rounds, 99.9% / 4.95 / 0.6%; MiniMax
//! fixed 7, 2,399 rounds, 99.5% / 6.79 / 1.0%. This implementation (cold
//! server rate 1) gives 99.91% / 4.98 / 0.14%, 99.87% / 4.94 / 0.63% and
//! 99.62% / 6.85 / 0.96% on the same traces. The sample below is their
//! hardest stretch: the first requests, before the server window fills.
use cuteafd_core::{DraftControl, DraftHistory, DraftOutcome, DrafterSpec, HeadScore};

const BLOCK: usize = 5;

struct Round {
    id: u64,
    accepted: usize,
    head: Vec<f32>,
}

fn rounds() -> Vec<Round> {
    include_str!("../fixtures/v41-fixed5-draft-rounds.jsonl").lines().map(|line| {
        let value: serde_json::Value = serde_json::from_str(line).unwrap();
        Round { id: value["id"].as_u64().unwrap(), accepted: value["accepted"].as_u64().unwrap() as usize,
            head: value["head"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect() }
    }).collect()
}

/// (tokens emitted, drafts verified, rounds at one draft or fewer, head skips).
fn replay(head: HeadScore, control: DraftControl) -> (usize, usize, usize, u64) {
    let mut policy = DraftHistory::new(DrafterSpec { block: BLOCK, head, copy: 0 }, control);
    let (mut tokens, mut drafts, mut short) = (0, 0, 0);
    for round in rounds() {
        policy.plan(&[(round.id, BLOCK)], 64);
        let length = policy.refine(round.id, BLOCK, Some(&round.head));
        assert_eq!(policy.settle(&[round.id], 64), [length]);
        let accepted = round.accepted.min(length);
        policy.observe(&[DraftOutcome { id: round.id, proposed: length, accepted, censored: false }], false);
        tokens += accepted + 1;
        drafts += length;
        short += usize::from(length <= 1);
    }
    assert_eq!(policy.stats().short_rounds as usize, short);
    (tokens, drafts, short, policy.stats().head_skips)
}

#[test]
fn history_keeps_fixed_width_tokens_on_recorded_rounds() {
    let (fixed, fixed_drafts, _, _) = replay(HeadScore::None, DraftControl::Fixed(BLOCK));
    assert_eq!((fixed, fixed_drafts), (4201, 6000));
    // 99.45% of fixed width's tokens at 4.78 drafts a round; 2.25% of rounds at one draft.
    assert_eq!(replay(HeadScore::None, DraftControl::History), (4178, 5739, 27, 0));
}

/// The dSpark head skip changes almost nothing on code traffic: on this sample
/// 14 skipped explores of 1,200 rounds, 4,177 tokens against 4,178 (a skipped
/// explore also withholds a miss, so later rounds draft slightly more: 5,798
/// drafts against 5,739). Bounds, not exact counts: the fit is floating point.
#[test]
fn the_head_skip_changes_little_on_recorded_rounds() {
    let (tokens, drafts, _, _) = replay(HeadScore::None, DraftControl::History);
    let (skip_tokens, skip_drafts, _, skips) = replay(HeadScore::Probability, DraftControl::History);
    eprintln!("head skip: {skips} skips, tokens {tokens} -> {skip_tokens}, drafts {drafts} -> {skip_drafts}");
    assert!((1..=60).contains(&skips), "{skips} skips");
    assert!(skip_tokens.abs_diff(tokens) * 200 <= tokens, "tokens {tokens} -> {skip_tokens}");
    assert!(skip_drafts.abs_diff(drafts) * 50 <= drafts, "drafts {drafts} -> {skip_drafts}");
}
