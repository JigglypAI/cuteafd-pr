//! Mixed replay and boundary cost at the measured V4.1 inventory scale.
use super::*;
use super::super::device::{Basis, Cycle, Preference, Site};
use std::{hint::black_box, time::Instant};

#[test]
#[ignore = "optimized mixed cache microbenchmark; run explicitly with --nocapture"]
fn cache_mixed_v41_nanoseconds() {
    let mut banks: Vec<GraphBank<u32, u32>> = (0..90).map(|_| {
        let mut bank = GraphBank::new();
        bank.attach_device_for_test(-39050, 1 << 40, Site { name: "mixed", cycle: Cycle::Decode,
            preference: Preference::Capture, bootstrap: 4096, basis: Basis::Bytes,
            spans_devices: false }).unwrap();
        for key in 0..120 {
            let ticket = bank.cache_ticket(&key, false, None).unwrap();
            bank.cache_commit(key, key, ticket, 8, 1, false);
        }
        bank
    }).collect();
    for bank in &mut banks { bank.cache_tick(); }
    for (label, switch) in [("stable", false), ("switch_every_step", true)] {
        let steps = 2048;
        let start = Instant::now();
        for step in 0..steps {
            let lane = step % 2;
            let cycle = if switch && (step / 2) % 2 == 0 { Cycle::Verify } else { Cycle::Decode };
            // Each lane visits thirty banks and replays sixty-seven segments on
            // this device. The two-device model has about134 segments/cycle.
            for bank in &mut banks[lane * 30..lane * 30 + 30] { bank.cache_cycle(cycle); }
            for hit in 0..67 {
                let key = ((step + hit) % 120) as u32;
                black_box(banks[lane * 30 + hit % 30].cached_launch(&key));
            }
            for bank in &mut banks[lane * 30..lane * 30 + 30] { bank.cache_tick(); }
        }
        println!("cache_mixed inventory=10800 banks=90 banks_per_step=30 hits_per_step=67 mode={label} ns_per_step={:.2}",
            start.elapsed().as_nanos() as f64 / steps as f64);
    }
}
