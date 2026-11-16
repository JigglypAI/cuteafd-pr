//! Identical host-side workloads for baseline and candidate cache implementations.
use super::*;
use super::super::device::{Basis, Cycle, Preference, Site};
use std::{hint::black_box, sync::{Arc, Barrier}, time::Instant};

fn fixture(device: i32) -> Vec<GraphBank<u32, u32>> {
    (0..64).map(|_| {
        let mut bank = GraphBank::new();
        bank.attach_device_for_test(device, 1 << 40, Site { name: "perf", cycle: Cycle::Decode,
            preference: Preference::Capture, bootstrap: 4096, basis: Basis::Bytes,
            spans_devices: false }).unwrap();
        for key in 0..128 {
            let ticket = bank.cache_ticket(&key, false, None).unwrap();
            bank.cache_commit(key, key, ticket, 8, 1, false);
        }
        bank
    }).collect()
}

#[test]
#[ignore = "optimized CPU cache microbenchmark; run explicitly with --nocapture"]
fn cache_hot_paths_nanoseconds() {
    // Two workers own disjoint banks sharing one device controller. No CUDA work.
    for threads in [1, 2] {
        let banks = fixture(-39000 - threads as i32);
        let chunks = banks.len() / threads;
        let barrier = Arc::new(Barrier::new(threads));
        std::thread::scope(|scope| {
            let mut banks = banks;
            for (worker, banks) in banks.chunks_mut(chunks).enumerate() {
                let barrier = barrier.clone();
                scope.spawn(move || {
                    for operation in ["get", "contains", "cached_launch", "cycle_switch", "capture_cancel"] {
                        let iterations = if operation == "cycle_switch" { 2_048 } else { 131_072 };
                        barrier.wait();
                        let started = Instant::now();
                        for step in 0..iterations {
                            let key = black_box((step % 128) as u32);
                            let bank = &mut banks[step % chunks];
                            match operation {
                                "get" => { black_box(bank.get(&key)); }
                                "contains" => { black_box(bank.contains(&key)); }
                                "cached_launch" => { black_box(bank.cached_launch(&key)); }
                                "cycle_switch" => bank.cache_cycle(if (step / chunks) % 2 == 0 { Cycle::Verify } else { Cycle::Decode }),
                                "capture_cancel" => { drop(black_box(bank.cache_ticket(&1024, false, None))); }
                                _ => unreachable!(),
                            }
                        }
                        let ns = started.elapsed().as_nanos() as f64 / iterations as f64;
                        println!("cache_perf threads={threads} worker={worker} records=8192 banks=64 op={operation} ns={ns:.2}");
                        barrier.wait();
                    }
                });
            }
        });
    }
}
