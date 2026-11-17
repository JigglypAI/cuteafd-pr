//! Host cost of the `history` draft policy per verify round, at a full lane.
//!
//! Each round plans, refines (recording head scores), extends a copy, settles
//! under the row budget and observes 16 requests of a seven-position drafter.
//! Run: `cargo test --release -p cuteafd-core --test all draft_history_host_cost_per_round
//! -- --ignored --nocapture`; `CUTEAFD_POLICY_BENCH_MAX_US` bounds the round median.
use cuteafd_core::{DraftControl, DraftHistory, DraftOutcome, DrafterSpec};
use std::time::Instant;

fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

#[test]
#[ignore = "release host-cost bench; run explicitly with --ignored --nocapture"]
fn draft_history_host_cost_per_round() {
    const REQUESTS: usize = 16;
    let spec = DrafterSpec { block: 7, copy: 32 };
    let mut policy = DraftHistory::new(spec, DraftControl::History);
    let requests: Vec<(u64, usize)> = (0..REQUESTS as u64).map(|id| (id, 40)).collect();
    let ids: Vec<u64> = requests.iter().map(|request| request.0).collect();
    let head = [2.5f32, 1.5, 0.5, 0., -0.5, -1., -2.];
    let (mut plan, mut shape, mut observe, mut round) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for step in 0..4000usize {
        let start = Instant::now();
        let depths = policy.plan(&requests, 64);
        let planned = Instant::now();
        for (&id, &depth) in ids.iter().zip(&depths) {
            let kept = policy.refine(id, depth, Some(&head));
            if kept >= 3 { policy.extend_copy(id, 32); }
        }
        let lengths = policy.settle(&ids, 64);
        let shaped = Instant::now();
        let outcomes: Vec<_> = ids.iter().zip(&lengths).map(|(&id, &length)| DraftOutcome { id, proposed: length,
            accepted: length.min((step + id as usize) % 9), censored: false }).collect();
        std::hint::black_box(policy.observe(&outcomes, step % 2 == 0));
        let done = Instant::now();
        if step >= 400 {
            let us = |from: Instant, to: Instant| to.duration_since(from).as_secs_f64() * 1e6;
            plan.push(us(start, planned));
            shape.push(us(planned, shaped));
            observe.push(us(shaped, done));
            round.push(us(start, done));
        }
    }
    let (plan, shape, observe, round) = (median(plan), median(shape), median(observe), median(round));
    eprintln!("C{REQUESTS} block 7 + copy 32, medians: round {round:.1} us, plan {plan:.1} us, shape {shape:.1} us, \
        observe {observe:.1} us");
    if let Some(bound) = std::env::var("CUTEAFD_POLICY_BENCH_MAX_US").ok().and_then(|value| value.parse::<f64>().ok()) {
        assert!(round <= bound, "round median {round:.1} us exceeds {bound} us");
    }
}
