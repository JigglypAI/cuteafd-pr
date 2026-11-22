//! A completed mark restore is a use of the host snapshot.
//!
//! When the device tier keeps an entry's KV pages but its mark lives in the host tier, every
//! hit on that entry is served by `HostCache::restore_mark`. That path is handed the key, so
//! it runs no lookup; before this file's fix it therefore neither counted a host hit nor
//! refreshed the snapshot's recency, and `LeastRecent` chose the snapshot that had just
//! served a request as its next victim.

use crate::common_hc_5::{
    default_cache, settle, snapshot, store_resident, target, tokens, write_snapshot, Device, DEVICE_BYTES,
};
use cuteafd_hostcache::cache::RestoreOutcome;
use cuteafd_hostcache::config::StoreMode;
use cuteafd_hostcache::pool::testing::CHUNK;
use cuteafd_hostcache::SnapshotKind;

/// A snapshot that just served a mark restore is the most recently used one, so the next
/// eviction takes another snapshot.
#[test]
fn a_snapshot_that_just_served_a_mark_restore_is_not_the_next_victim() {
    let mut cache = default_cache(3 * CHUNK as u64, StoreMode::OnRetain);
    let mut device = Device::new(DEVICE_BYTES);
    // Distinct first tokens: no sequence is a prefix of another, so the ancestor preference
    // cannot decide the victim; only recency can.
    let sequences: Vec<Vec<u32>> = (0..8u32).map(|i| (0..8).map(|j| 1000 * i + j).collect()).collect();
    let mut stored = Vec::new();
    for (index, tokens) in sequences.iter().enumerate() {
        let snapshot = snapshot(&mut device, SnapshotKind::Turn, tokens, 1, false, index as u32);
        write_snapshot(cache.engine_mut(), &snapshot);
        let _ = cache.store(&snapshot, index as u64);
        settle(&mut cache);
        cache.tick();
        stored.push(snapshot);
        if cache.metrics().host_evictions > 0 {
            break;
        }
    }
    assert!(cache.metrics().host_evictions > 0, "the quota forced an eviction");

    // The oldest resident snapshot serves a mark restore.
    let (oldest, key) = sequences
        .iter()
        .enumerate()
        .find_map(|(index, tokens)| cache.peek(tokens).map(|hit| (index, hit.key)))
        .expect("a resident snapshot");
    let tail = stored[oldest].tail.clone();
    assert!(matches!(cache.restore_mark(key, tail), RestoreOutcome::Done { .. }));

    // One more store forces another eviction; the snapshot that just served must survive it.
    let evictions = cache.metrics().host_evictions;
    let extra = snapshot(&mut device, SnapshotKind::Turn, &[9000, 9001, 9002, 9003], 1, false, 99);
    write_snapshot(cache.engine_mut(), &extra);
    let _ = cache.store(&extra, 99);
    settle(&mut cache);
    cache.tick();
    assert!(cache.metrics().host_evictions > evictions, "the extra store evicted a snapshot");
    assert!(cache.peek(&sequences[oldest]).is_some(), "the snapshot that just served a restore was evicted next");
}

/// The mark-restore path counts its hit: without it `host_hits` stays at the lookups'
/// count while the host tier serves every request.
#[test]
fn a_completed_mark_restore_counts_as_a_host_hit() {
    let mut cache = default_cache(4 * CHUNK as u64, StoreMode::OnRetain);
    let mut device = Device::new(DEVICE_BYTES);
    let snapshot = snapshot(&mut device, SnapshotKind::Turn, &tokens(8), 2, false, 1);
    write_snapshot(cache.engine_mut(), &snapshot);
    store_resident(&mut cache, &snapshot, 7);

    let key = cache.peek(&tokens(8)).expect("resident").key;
    assert_eq!(cache.metrics().host_hits, 0, "a peek is not a hit");
    let tail = snapshot.tail.iter().copied().collect();
    let outcome = cache.restore_mark(key, tail);
    assert!(matches!(outcome, RestoreOutcome::Done { .. }), "{outcome:?}");
    let metrics = cache.metrics();
    assert_eq!((metrics.restores, metrics.host_hits), (1, 1));
}

/// A mark restore that does not complete is not a hit.
#[test]
fn a_failed_mark_restore_is_not_a_host_hit() {
    let mut cache = default_cache(4 * CHUNK as u64, StoreMode::OnRetain);
    let mut device = Device::new(DEVICE_BYTES);
    let snapshot = snapshot(&mut device, SnapshotKind::Turn, &tokens(8), 2, false, 1);
    write_snapshot(cache.engine_mut(), &snapshot);
    store_resident(&mut cache, &snapshot, 3);

    let key = cache.peek(&tokens(8)).expect("resident").key;
    let outcome = cache.restore_mark(key.wrapping_add(1_000_000), Vec::new());
    assert_eq!(outcome, RestoreOutcome::Failed);
    let metrics = cache.metrics();
    assert_eq!((metrics.restores, metrics.host_hits), (0, 0));
}

/// A promotion is still one hit: its lookup counts it and the full restore adds none.
#[test]
fn a_full_restore_after_a_lookup_counts_one_hit() {
    let mut cache = default_cache(4 * CHUNK as u64, StoreMode::OnRetain);
    let mut device = Device::new(DEVICE_BYTES);
    let snapshot = snapshot(&mut device, SnapshotKind::Turn, &tokens(8), 2, true, 1);
    write_snapshot(cache.engine_mut(), &snapshot);
    store_resident(&mut cache, &snapshot, 5);

    let hit = cache.lookup(&tokens(8)).expect("host hit");
    let target = target(&mut device, &snapshot, 2);
    assert!(matches!(cache.restore(hit.key, &target), RestoreOutcome::Done { .. }));
    let metrics = cache.metrics();
    assert_eq!((metrics.restores, metrics.host_hits), (1, 1));
}
