//! Per-layer x per-expert route hit counters.
//!
//! Every MoE family already builds a [`RoundRoutes`] for its verify step: the
//! exact per-layer, per-row routed expert ids, at whatever top-k the family
//! uses (6 for DeepSeek V4/V4.1, 8 for GLM 5.3 / GLM 5.3 Flash / MiMo V2,
//! 10 for Qwen 3.8, 16 for Kimi K3). Under TP slicing every Spark sees every
//! route, so a coordinator-side count is the whole truth, including RTX-local
//! layers. This module keeps that matrix behind `EXPERT_USAGE` (off by
//! default: enabling it turns on route capture).
//!
//! Counters are monotonic; there is no reset endpoint (Prometheus diffs them).
//! The matrix is exported in `/v1/stats` as `expert_usage`.
//!
//! Accounting model: a token is counted once, when it commits. A decode round
//! commits its one row per request; a verify round commits only each request's
//! accepted prefix ([`record_committed`]), never the rejected draft rows (they
//! are routed again next round). A failed round is not counted (its routes
//! describe nothing). Prefill routes are not captured yet: the probe is armed
//! for verify rounds only, and a prefill chunk is wider than the probe's ring.
//!
//! Cost: the matrix is allocated once at [`enable`] (start-up only), so the
//! hot path takes a shared read lock, walks the round's ids and does one
//! relaxed atomic add per route. There is no per-round allocation, no
//! process-wide exclusive lock and no CUDA work. A 40-layer, 64-row, top-k 6
//! round is 15,360 relaxed adds; the read lock is shared, so concurrent lane
//! rounds do not serialize.
//!
//! The matrix is serialized only when `/v1/stats` is read: the daemon registers
//! [`snapshot`] as the endpoint's provider at serve start
//! ([`install_stats_provider`]), so no decode round builds or clones it.

use crate::shared::draft::routes::RoundRoutes;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

/// One enabled matrix: `experts` counters per routed layer. Every layer is
/// allocated at [`enable`] so recording never grows the table.
struct Table {
    experts: usize,
    rows: AtomicU64,
    layers: Box<[Box<[AtomicU64]>]>,
}

static USAGE: RwLock<Option<Arc<Table>>> = RwLock::new(None);
/// Mirrors `USAGE` being `Some` for the hot path (one check per lane round).
static ENABLED: AtomicBool = AtomicBool::new(false);

/// Enable the matrix for a model with `experts` routed experts per layer and
/// `layers` routed layers. Re-enabling replaces the table (start-up only).
pub(crate) fn enable(experts: u32, layers: u32) {
    let experts = experts as usize;
    let layers = layers as usize;
    let table = Table {
        experts,
        rows: AtomicU64::new(0),
        layers: (0..layers).map(|_| (0..experts).map(|_| AtomicU64::new(0))
            .collect::<Vec<_>>().into_boxed_slice()).collect::<Vec<_>>().into_boxed_slice(),
    };
    let mut slot = USAGE.write().unwrap_or_else(|error| error.into_inner());
    *slot = Some(Arc::new(table));
    ENABLED.store(true, Ordering::Release);
}

/// Whether the matrix is enabled (route capture must be on). Lock-free: this
/// runs once per lane round on the serving hot path.
pub(crate) fn enabled() -> bool {
    ENABLED.load(Ordering::Acquire)
}

/// Register the `/v1/stats` provider once at serve start. The matrix is built
/// only when the endpoint is read, never on the serving hot path.
pub(crate) fn install_stats_provider() {
    cuteafd_api::openai::set_expert_usage_provider(snapshot);
}

/// Enable the matrix from a serve command's `--expert-usage` flag, using the
/// process expert geometry fixed at startup.
pub(crate) fn enable_from_flag(flag: bool) {
    install_stats_provider();
    // The planner reads this process switch, so a bare `--expert-usage` launch
    // charges the probe it allocates (the launcher also exports the env var).
    cuteafd_loader::placement::set_expert_usage(flag);
    if !flag { return; }
    let geometry = cuteafd_core::expert_geometry();
    enable(geometry.experts, geometry.layers);
}

/// Count every routed expert of every real row of every routed layer.
///
/// `routes` is layer-major `[layer][row][slot]`; `topk` is whatever the family
/// uses. Dense and unfilled layers are skipped (their ids are zero and would
/// otherwise be booked to expert 0). Rows come from the round's own row count,
/// not from layer 0, so a partial layer 0 cannot undercount. Out-of-range ids
/// and layers are ignored (a different model's geometry must never panic
/// serving).
pub(crate) fn record(routes: &RoundRoutes) {
    let slot = USAGE.read().unwrap_or_else(|error| error.into_inner());
    let Some(table) = slot.as_ref() else { return };
    let (layers, rows, topk) = (routes.layers(), routes.rows(), routes.topk());
    if rows == 0 || topk == 0 { return; }
    table.rows.fetch_add(rows as u64, Ordering::Relaxed);
    let words = rows * topk;
    let flat = routes.flat();
    for layer in 0..layers {
        if routes.is_dense(layer) || !routes.is_filled(layer) { continue; }
        let Some(hits) = table.layers.get(layer) else { continue };
        let start = layer * words;
        let Some(ids) = flat.get(start..start + words) else { continue };
        for &id in ids {
            if let Some(cell) = hits.get(id as usize) {
                cell.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

/// One verified request's row accounting: executed rows and committed inputs.
pub(crate) trait CommittedRows {
    fn rows(&self) -> usize;
    fn accepted(&self) -> usize;
}

impl CommittedRows for crate::shared::draft::binding::RoundRequest<'_> {
    fn rows(&self) -> usize { self.rows }
    fn accepted(&self) -> usize { self.accepted }
}

impl CommittedRows for crate::shared::draft::resource::Verified<'_> {
    fn rows(&self) -> usize { self.rows }
    fn accepted(&self) -> usize { self.accepted }
}

impl CommittedRows for (usize, usize) {
    fn rows(&self) -> usize { self.0 }
    fn accepted(&self) -> usize { self.1 }
}

/// Count only the committed rows of a verify round: each request contributes
/// its accepted prefix (`accepted`, anchor included), not its executed rows.
/// A rejected draft token is routed again next round, so counting it here would
/// book it twice; the anchor and accepted drafts are the tokens that commit.
pub(crate) fn record_committed<T: CommittedRows>(routes: &RoundRoutes, requests: &[T]) {
    let slot = USAGE.read().unwrap_or_else(|error| error.into_inner());
    let Some(table) = slot.as_ref() else { return };
    let (layers, rows, topk) = (routes.layers(), routes.rows(), routes.topk());
    if rows == 0 || topk == 0 { return; }
    let committed: usize = requests.iter().map(|request| request.accepted().min(request.rows())).sum();
    table.rows.fetch_add(committed as u64, Ordering::Relaxed);
    let flat = routes.flat();
    for layer in 0..layers {
        if routes.is_dense(layer) || !routes.is_filled(layer) { continue; }
        let Some(hits) = table.layers.get(layer) else { continue };
        let start = layer * rows * topk;
        let Some(ids) = flat.get(start..start + rows * topk) else { continue };
        let mut offset = 0;
        for request in requests {
            let end = (offset + request.accepted().min(request.rows())).min(rows);
            for row in offset..end {
                for &id in &ids[row * topk..][..topk] {
                    if let Some(cell) = hits.get(id as usize) { cell.fetch_add(1, Ordering::Relaxed); }
                }
            }
            offset += request.rows();
        }
    }
}

/// As [`record_committed`](Self::record_committed), for a V4.1-style
/// `[layer][row][slot]` capture: `inputs` are the round's verifier rows per
/// request and `accepted` their committed prefixes (anchor included).
pub(crate) fn record_capture_committed(capture: &[Vec<[u32; 6]>], inputs: &[Vec<u32>], accepted: &[u32]) {
    let total: usize = inputs.iter().map(Vec::len).sum();
    if total == 0 { return; }
    let slot = USAGE.read().unwrap_or_else(|error| error.into_inner());
    let Some(table) = slot.as_ref() else { return };
    let committed: usize = inputs.iter().zip(accepted)
        .map(|(rows, &accepted)| (accepted as usize).min(rows.len())).sum();
    table.rows.fetch_add(committed as u64, Ordering::Relaxed);
    for (layer, routes) in capture.iter().enumerate() {
        if routes.is_empty() { continue; }
        let Some(hits) = table.layers.get(layer) else { continue };
        let mut offset = 0;
        for (rows, &accepted) in inputs.iter().zip(accepted) {
            let end = (offset + (accepted as usize).min(rows.len())).min(routes.len());
            for row in offset..end {
                for &id in &routes[row] {
                    if let Some(cell) = hits.get(id as usize) { cell.fetch_add(1, Ordering::Relaxed); }
                }
            }
            offset += rows.len();
        }
    }
}

/// Record a V4.1-style `[layer][row][slot]` capture (top-k 6) without building
/// a [`RoundRoutes`]: empty layers (not captured this round) are skipped.
pub(crate) fn record_capture(capture: &[Vec<[u32; 6]>]) {
    let rows = capture.iter().find(|rows| !rows.is_empty()).map_or(0, Vec::len);
    if rows == 0 { return; }
    let slot = USAGE.read().unwrap_or_else(|error| error.into_inner());
    let Some(table) = slot.as_ref() else { return };
    table.rows.fetch_add(rows as u64, Ordering::Relaxed);
    for (layer, routes) in capture.iter().enumerate() {
        if routes.is_empty() { continue; }
        let Some(hits) = table.layers.get(layer) else { continue };
        for row in routes {
            for &id in row {
                if let Some(cell) = hits.get(id as usize) {
                    cell.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    }
}

/// The `expert_usage` object exported in `/v1/stats`. Disabled is the small
/// `{"enabled": false}` shape; enabled carries the full matrix.
pub(crate) fn snapshot() -> serde_json::Value {
    let slot = USAGE.read().unwrap_or_else(|error| error.into_inner());
    let Some(table) = slot.as_ref() else {
        return serde_json::json!({ "enabled": false });
    };
    let hits: Vec<Vec<u64>> = table.layers.iter()
        .map(|layer| layer.iter().map(|cell| cell.load(Ordering::Relaxed)).collect())
        .collect();
    serde_json::json!({
        "enabled": true,
        "experts": table.experts,
        "layers": table.layers.len(),
        "rows": table.rows.load(Ordering::Relaxed),
        "hits": hits,
    })
}

/// `--expert-usage` on every family's serve command (and `serve-native`).
#[derive(Debug, Clone, clap::Args)]
pub(crate) struct ExpertUsageArgs {
    /// Count per-layer x per-expert route hits (turns on route capture).
    #[arg(long, env = "CUTEAFD_EXPERT_USAGE")]
    pub expert_usage: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The matrix is process-wide; serialize the tests that mutate it.
    static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn reset() {
        *USAGE.write().unwrap_or_else(|error| error.into_inner()) = None;
        ENABLED.store(false, Ordering::Release);
    }

    fn routes<const K: usize>(layers: usize, rows: &[Vec<[u32; K]>]) -> RoundRoutes {
        let mut routes = RoundRoutes::new(layers, K);
        routes.fill(rows, rows.first().map_or(0, Vec::len), u32::MAX).unwrap();
        routes
    }

    #[test]
    fn record_counts_each_route_once_per_row() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset();
        enable(8, 1);
        record(&routes(1, &[vec![[0, 1, 2, 3, 4, 5], [0, 0, 1, 1, 2, 2]]]));
        let snapshot = snapshot();
        assert_eq!(snapshot["rows"], 2);
        assert_eq!(snapshot["hits"][0][0], 3);
        assert_eq!(snapshot["hits"][0][1], 3);
        assert_eq!(snapshot["hits"][0][2], 3);
        assert_eq!(snapshot["hits"][0][3], 1);
        assert_eq!(snapshot["hits"][0][4], 1);
        assert_eq!(snapshot["hits"][0][5], 1);
        assert_eq!(snapshot["hits"][0][6], 0);
    }

    /// top-k is whatever the family uses: 6, 8, 10 and 16 all count.
    #[test]
    fn arbitrary_topk_is_counted() {
        let _guard = TEST_LOCK.lock().unwrap();
        for topk in [6usize, 8, 10, 16] {
            reset();
            enable(32, 1);
            let row: Vec<u32> = (0..topk as u32).collect();
            let mut routes = RoundRoutes::new(1, topk);
            routes.begin(1);
            routes.push(0, &row, u32::MAX).unwrap();
            record(&routes);
            let snapshot = snapshot();
            assert_eq!(snapshot["rows"], 1, "topk {topk}");
            for expert in 0..topk {
                assert_eq!(snapshot["hits"][0][expert], 1, "topk {topk} expert {expert}");
            }
            assert_eq!(snapshot["hits"][0][topk], 0, "topk {topk} next expert");
        }
    }

    /// Rows come from the round, not from layer 0: a partial layer 0 must not
    /// undercount the rows of the layers that did run.
    #[test]
    fn rows_come_from_the_round_not_layer_zero() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset();
        enable(8, 2);
        let mut routes = RoundRoutes::new(2, 6);
        routes.begin(3);
        // Layer 0 is dense (no routes); layer 1 has three real rows.
        routes.dense(0).unwrap();
        routes.push_rows(1, &[[1; 6], [2; 6], [3; 6]], u32::MAX).unwrap();
        record(&routes);
        let snapshot = snapshot();
        assert_eq!(snapshot["rows"], 3);
        assert_eq!(snapshot["hits"][0][0], 0, "dense layer 0 is not booked to expert 0");
        assert_eq!(snapshot["hits"][1][1], 6);
        assert_eq!(snapshot["hits"][1][2], 6);
        assert_eq!(snapshot["hits"][1][3], 6);
    }

    /// The V4.1 capture adapter skips empty (uncaptured) layers and counts the
    /// rows of the layers that did run.
    #[test]
    fn capture_adapter_skips_empty_layers() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset();
        enable(8, 2);
        record_capture(&[vec![], vec![[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]]]);
        let view = snapshot();
        assert_eq!(view["rows"], 2);
        assert_eq!(view["hits"][0][0], 0);
        assert_eq!(view["hits"][1][1], 2);
        // An all-empty capture records nothing.
        reset();
        enable(8, 2);
        record_capture(&[vec![], vec![]]);
        assert_eq!(snapshot(), serde_json::json!({ "enabled": true, "experts": 8, "layers": 2, "rows": 0, "hits": [[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]] }));
    }

    #[test]
    fn out_of_range_ids_are_ignored() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset();
        enable(4, 1);
        record(&routes(1, &[vec![[0, 9, 100, 3, 4, 5]]]));
        let snapshot = snapshot();
        assert_eq!(snapshot["experts"], 4);
        assert_eq!(snapshot["hits"][0][0], 1);
        assert_eq!(snapshot["hits"][0][3], 1);
        assert_eq!(snapshot["hits"][0].as_array().unwrap().len(), 4);
    }

    /// A round with more layers than the enabled geometry is ignored past the
    /// table's extent instead of panicking.
    #[test]
    fn out_of_range_layers_are_ignored() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset();
        enable(4, 1);
        let mut routes = RoundRoutes::new(2, 6);
        routes.begin(1);
        routes.push_rows(0, &[[1; 6]], u32::MAX).unwrap();
        routes.push_rows(1, &[[2; 6]], u32::MAX).unwrap();
        record(&routes);
        let snapshot = snapshot();
        assert_eq!(snapshot["layers"], 1);
        assert_eq!(snapshot["hits"][0][1], 6);
    }

    #[test]
    fn snapshot_shape_is_stable() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset();
        enable(384, 1);
        record(&routes(1, &[vec![[0; 6]]]));
        let snapshot = snapshot();
        assert_eq!(snapshot["enabled"], true);
        assert_eq!(snapshot["experts"], 384);
        assert_eq!(snapshot["layers"], 1);
        assert_eq!(snapshot["rows"], 1);
        let hits = snapshot["hits"].as_array().unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].as_array().unwrap().len(), 384);
    }

    /// `enabled()` is the per-lane-round hot-path check; it reads an atomic, so
    /// it neither takes the table lock nor reads a poisoned lock as "off" while
    /// the matrix is enabled.
    #[test]
    fn enabled_is_a_lock_free_flag() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset();
        assert!(!enabled());
        enable(4, 1);
        assert!(enabled());
        let _ = std::thread::spawn(|| {
            let _held = USAGE.write().unwrap_or_else(|error| error.into_inner());
            panic!("poison the table lock");
        }).join();
        assert!(USAGE.is_poisoned());
        assert!(enabled());
        USAGE.clear_poison();
        reset();
        assert!(!enabled());
    }

    #[test]
    fn disabled_snapshot_is_small() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset();
        let snapshot = snapshot();
        assert_eq!(snapshot, serde_json::json!({ "enabled": false }));
        assert!(!enabled());
    }

    fn request(id: u64, rows: usize, accepted: usize) -> crate::shared::draft::binding::RoundRequest<'static> {
        crate::shared::draft::binding::RoundRequest { id, rows, accepted,
            source: crate::shared::draft::evidence::DraftSource::Neural, evidence: None, censor: None }
    }

    /// A speculative round counts only the committed rows: a rejected draft
    /// token is routed again next round, so counting it here would book it
    /// twice. Request 0 executes 4 rows and commits 2; request 1 executes 1 and
    /// commits 1, so only rows 0,1 and 4 are counted.
    #[test]
    fn committed_rows_count_once_per_committed_token() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset();
        enable(8, 1);
        let mut routes = RoundRoutes::new(1, 6);
        routes.begin(5);
        // Row r routes to expert r, so the counts name exactly the committed rows.
        let rows: Vec<[u32; 6]> = (0..5).map(|row| [row as u32; 6]).collect();
        routes.push_rows(0, &rows, u32::MAX).unwrap();
        record_committed(&routes, &[request(1, 4, 2), request(2, 1, 1)]);
        let snapshot = snapshot();
        assert_eq!(snapshot["rows"], 3, "2 committed + 1 committed");
        assert_eq!(snapshot["hits"][0][0], 6, "row 0 committed");
        assert_eq!(snapshot["hits"][0][1], 6, "row 1 committed");
        assert_eq!(snapshot["hits"][0][2], 0, "row 2 rejected");
        assert_eq!(snapshot["hits"][0][3], 0, "row 3 rejected");
        assert_eq!(snapshot["hits"][0][4], 6, "row 4 committed");
    }

    /// The V4.1 capture adapter applies the same committed-row rule, per family
    /// top-k (6 here; the shared path covers 8/10/16).
    #[test]
    fn capture_adapter_counts_only_committed_rows() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset();
        enable(8, 1);
        let capture = vec![(0..5).map(|row| [row as u32; 6]).collect::<Vec<_>>()];
        let inputs = vec![vec![0u32; 4], vec![0u32; 1]];
        record_capture_committed(&capture, &inputs, &[2, 1]);
        let snapshot = snapshot();
        assert_eq!(snapshot["rows"], 3);
        assert_eq!(snapshot["hits"][0][0], 6);
        assert_eq!(snapshot["hits"][0][1], 6);
        assert_eq!(snapshot["hits"][0][2], 0);
        assert_eq!(snapshot["hits"][0][4], 6);
    }

    /// A fully accepted round counts every row, matching `record`.
    #[test]
    fn fully_accepted_round_matches_record() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset();
        enable(8, 1);
        let mut routes = RoundRoutes::new(1, 6);
        routes.begin(3);
        routes.push_rows(0, &[[1; 6], [2; 6], [3; 6]], u32::MAX).unwrap();
        record_committed(&routes, &[request(1, 2, 2), request(2, 1, 1)]);
        let committed = snapshot();
        reset();
        enable(8, 1);
        record(&routes);
        assert_eq!(committed, snapshot());
    }
}
