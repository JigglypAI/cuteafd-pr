//! The serving side of the verify-timing benchmark probe
//! ([`cuteafd_api::openai::probe::verify_timing`]): exact verify row shapes
//! through a family's production decode step (captured graphs and its own
//! bucket padding) on scratch sequences. Each scratch sequence is prefilled
//! once; every step then resets it to its prefilled length and commits
//! nothing (the speculative verify leaves recurrent state at the start and
//! later steps overwrite its records), so the server's state is unchanged.
//! The scheduler runs a probe only while it holds no other request, and the
//! family admits scratch pages only if they fit without evicting prefix-cache
//! entries.
use crate::shared::token_io::SelectBatch;
use anyhow::{ensure, Context, Result};
use cuteafd_api::openai::probe::verify_timing::{GraphDelta, VerifySample, VerifyTiming, VerifyTimingRecord};
use cuteafd_core::TargetSamplingParams;
use std::time::Instant;

/// What timing records: where the clock starts and stops.
pub(crate) const TIMING_SCOPE: &str = "verify: host wall from the serving verify call to a full device drain after it \
    (synchronized; serving overlaps the drain with token selection); select: greedy token selection of every real row \
    after that drain, to host results; scratch sequences reset after every step; one idle server";

/// One verify step's bucket: the one serving pads its rows to, and the one it ran.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Buckets {
    pub natural: usize,
    pub ran: usize,
}

/// A family's serving decode path as the probe drives it.
pub(crate) trait VerifyTarget {
    type Placement;
    /// Prefill `tokens` into a fresh scratch sequence (its context).
    fn prefill(&mut self, placement: &mut Self::Placement, tokens: &[u32]) -> Result<()>;
    /// One serving verify step of `rows` per sequence, padded to `forced` (else the
    /// natural bucket). A forced bucket must be one serving itself pads to (a fixed
    /// point of the family's bucket map) and at least the natural one.
    fn verify(&mut self, step: &mut [(&mut Self::Placement, usize)], tokens: &[u32], forced: Option<usize>)
        -> Result<Buckets>;
    /// Serving's token selection over the last verify's real rows; returns the rows selected.
    fn select(&mut self, batch: &SelectBatch) -> Result<usize>;
    fn synchronize(&mut self) -> Result<()>;
    fn len(placement: &Self::Placement) -> usize;
    fn reset(placement: &mut Self::Placement, len: usize);
    /// Startup graph prewarm is done and serving replays captured graphs.
    fn replay_ready(&self) -> bool;
}

/// Serving's bucket for `rows` and the requested one, checked: a forced bucket
/// is a fixed point of the bucket map at least the natural bucket.
pub(crate) fn checked_bucket(rows: usize, forced: Option<usize>, bucket_of: impl Fn(usize) -> Result<usize>)
    -> Result<Buckets> {
    let natural = bucket_of(rows)?;
    let Some(forced) = forced else { return Ok(Buckets { natural, ran: natural }) };
    ensure!(forced >= natural, "forced bucket {forced} is below the natural bucket {natural} of {rows} rows");
    ensure!(bucket_of(forced).ok() == Some(forced),
        "forced bucket {forced} is not a bucket serving pads to (its rows pad to {:?})", bucket_of(forced).ok());
    Ok(Buckets { natural, ran: forced })
}

/// The probe's NVTX ranges: armed only on this thread while a probe runs, so
/// ordinary serving pays one thread-local check where a family marks a phase.
mod nvtx {
    use std::cell::Cell;
    thread_local! { static ARMED: Cell<bool> = const { Cell::new(false) }; }

    pub(crate) struct Armed;
    impl Drop for Armed {
        fn drop(&mut self) { ARMED.with(|armed| armed.set(false)); }
    }
    pub(crate) fn arm() -> Armed {
        ARMED.with(|armed| armed.set(true));
        Armed
    }
    pub(crate) fn armed() -> bool { ARMED.with(Cell::get) }
}

/// An open NVTX range; it closes when dropped.
pub(crate) struct Range<'l>(Option<&'l cuteafd_ffi::NativeLibrary>);

impl Drop for Range<'_> {
    fn drop(&mut self) {
        if let Some(library) = self.0 {
            if let Err(error) = library.nvtx_range_pop() { tracing::debug!(%error, "NVTX range pop"); }
        }
    }
}

/// Opens range `name()` while a verify-timing probe runs on this thread; otherwise nothing.
#[inline]
pub(crate) fn range<'l>(library: Option<&'l cuteafd_ffi::NativeLibrary>, name: impl FnOnce() -> String) -> Range<'l> {
    let Some(library) = library.filter(|_| nvtx::armed()) else { return Range(None) };
    match std::ffi::CString::new(name()).map_err(anyhow::Error::from).and_then(|name| library.nvtx_range_push(&name)) {
        Ok(()) => Range(Some(library)),
        Err(error) => { tracing::debug!(%error, "NVTX range push"); Range(None) }
    }
}

fn graph_counters() -> GraphDelta {
    let mut total = GraphDelta { captures: cuteafd_ffi::graph_captures(), ..GraphDelta::default() };
    for snapshot in crate::shared::decode_graph::device::snapshots() {
        for counters in snapshot.cycles.values() {
            total.misses += counters.misses;
            total.eager_runs += counters.eager_runs;
        }
    }
    total
}

fn delta(before: GraphDelta, after: GraphDelta) -> GraphDelta {
    GraphDelta { captures: after.captures.saturating_sub(before.captures),
        misses: after.misses.saturating_sub(before.misses), eager_runs: after.eager_runs.saturating_sub(before.eager_runs) }
}

/// Scratch sequence `i` takes `context` tokens from offset `i` of `tokens`; its
/// verify rows are the tokens that follow.
pub(crate) fn run<T: VerifyTarget>(target: &mut T, library: Option<&cuteafd_ffi::NativeLibrary>, family: &str,
    timing: &VerifyTiming, tokens: &[u32], placements: &mut [T::Placement]) -> Result<VerifyTimingRecord> {
    timing.validate()?;
    ensure!(placements.len() == timing.sequences(), "{} scratch sequences for {} requests", placements.len(), timing.sequences());
    ensure!(tokens.len() >= timing.prompt_tokens(), "verify timing needs {} prompt tokens, got {}",
        timing.prompt_tokens(), tokens.len());
    let _armed = nvtx::arm();
    for (i, placement) in placements.iter_mut().enumerate() {
        let _range = range(library, || format!("verify-timing prefill {i}"));
        target.prefill(placement, &tokens[i..i + timing.context])?;
    }
    target.synchronize()?;
    let starts: Vec<usize> = placements.iter().map(T::len).collect();
    let mut samples = Vec::new();
    for shape in &timing.shapes {
        let n = shape.rows.len();
        let step_tokens: Vec<u32> = shape.rows.iter().enumerate()
            .flat_map(|(i, &rows)| tokens[i + timing.context..i + timing.context + rows].iter().copied()).collect();
        let mut batch = SelectBatch::default();
        for (i, &rows) in shape.rows.iter().enumerate() {
            batch.push_sequence(TargetSamplingParams::greedy(), None, &step_tokens[..rows], (starts[i] + 1) as u64)?;
        }
        let mut sample = VerifySample { shape: shape.to_string(), real_rows: shape.real_rows(), contexts: starts[..n].to_vec(),
            replay_ready: target.replay_ready(), warmups: timing.warmups, ..VerifySample::default() };
        let mut before = graph_counters();
        for round in 0..timing.warmups + timing.repeats {
            if round == timing.warmups { before = graph_counters(); }
            let label = || format!("verify {shape} R={} B={}", shape.real_rows(), shape.bucket.map_or("auto".into(), |b| b.to_string()));
            let outer = range(library, label);
            for (placement, &start) in placements.iter_mut().zip(&starts) { T::reset(placement, start); }
            let mut step: Vec<(&mut T::Placement, usize)> = placements.iter_mut().zip(&shape.rows)
                .map(|(placement, &rows)| (placement, rows)).collect();
            let timer = Instant::now();
            let buckets = target.verify(&mut step, &step_tokens, shape.bucket)
                .with_context(|| format!("verify shape {shape}"))?;
            target.synchronize()?;
            let verify_ms = timer.elapsed().as_secs_f64() * 1e3;
            let selected = {
                let _range = range(library, || "select".into());
                let timer = Instant::now();
                target.select(&batch).map(|rows| (rows, timer.elapsed().as_secs_f64() * 1e3))
            };
            drop(outer);
            let (rows, select_ms) = selected?;
            ensure!(rows == shape.real_rows(), "selection returned {rows} rows for {}", shape.real_rows());
            (sample.natural_bucket, sample.bucket) = (buckets.natural, buckets.ran);
            if round >= timing.warmups {
                sample.verify_ms.push(verify_ms);
                sample.select_ms.push(select_ms);
            }
        }
        for (placement, &start) in placements.iter_mut().zip(&starts) { T::reset(placement, start); }
        sample.graph = delta(before, graph_counters());
        sample.graphed = sample.replay_ready && sample.graph == GraphDelta::default();
        tracing::info!(family, shape = %sample.shape, real_rows = sample.real_rows, bucket = sample.bucket,
            natural_bucket = sample.natural_bucket, verify_ms = sample.verify_median_ms(), select_ms = sample.select_median_ms(),
            graphed = sample.graphed, "verify timing sample");
        samples.push(sample);
    }
    Ok(VerifyTimingRecord { family: family.into(), timing_scope: TIMING_SCOPE.into(), samples })
}

/// The scheduler's admission gate for a verify-timing probe: only an idle server runs one.
pub(crate) fn admit(probe: &super::ProbeRef, busy: bool) -> Option<Result<VerifyTiming>> {
    let timing = probe.as_ref()?.spec.verify_timing.clone()?;
    Some(if busy { Err(anyhow::anyhow!("verify timing runs only on an idle server; requests are in flight")) }
        else { timing.validate().map(|()| timing) })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forced_buckets_are_fixed_points_at_least_the_natural_bucket() {
        let ladder = |rows: usize| -> Result<usize> {
            [2, 4, 8, 16, 32, 64].into_iter().find(|&b| b >= rows).context("past the ladder")
        };
        assert_eq!(checked_bucket(33, None, ladder).unwrap(), Buckets { natural: 64, ran: 64 });
        assert_eq!(checked_bucket(32, Some(64), ladder).unwrap(), Buckets { natural: 32, ran: 64 });
        assert!(checked_bucket(33, Some(32), ladder).is_err(), "below natural");
        assert!(checked_bucket(32, Some(48), ladder).is_err(), "48 is not a bucket serving pads to");
        assert!(checked_bucket(32, Some(128), ladder).is_err(), "past the ladder");
        // Exact-row graphs: every row count is its own bucket.
        let exact = |rows: usize| -> Result<usize> { Ok(rows) };
        assert_eq!(checked_bucket(33, None, exact).unwrap(), Buckets { natural: 33, ran: 33 });
        assert_eq!(checked_bucket(33, Some(40), exact).unwrap(), Buckets { natural: 33, ran: 40 });
    }

    #[test]
    fn ranges_are_inert_unless_armed() {
        assert!(!nvtx::armed());
        {
            let _armed = nvtx::arm();
            assert!(nvtx::armed());
        }
        assert!(!nvtx::armed(), "disarmed when the probe ends, errors included");
    }

    #[test]
    fn only_an_idle_server_admits_and_ordinary_requests_are_untouched() {
        use cuteafd_api::openai::probe::{Probe, ProbeSpec};
        assert!(admit(&None, false).is_none());
        assert!(admit(&Some(Probe::new(ProbeSpec::default())), false).is_none());
        let timing: VerifyTiming = serde_json::from_value(serde_json::json!({"shapes": ["4x8"]})).unwrap();
        let probe = Some(Probe::new(ProbeSpec { verify_timing: Some(timing.clone()), ..ProbeSpec::default() }));
        assert_eq!(admit(&probe, false).unwrap().unwrap(), timing);
        assert!(admit(&probe, true).unwrap().is_err());
    }

    /// A model target whose state is each placement's length: every step must
    /// start from the prefilled length and the probe must leave every sequence there.
    struct Fake { seen: Vec<Vec<usize>>, rows: usize, fail_at: Option<usize> }
    impl VerifyTarget for Fake {
        type Placement = usize;
        fn prefill(&mut self, placement: &mut usize, tokens: &[u32]) -> Result<()> {
            *placement += tokens.len();
            Ok(())
        }
        fn verify(&mut self, step: &mut [(&mut usize, usize)], _tokens: &[u32], forced: Option<usize>) -> Result<Buckets> {
            self.seen.push(step.iter().map(|(p, _)| **p).collect());
            ensure!(self.fail_at != Some(self.seen.len()), "injected failure");
            self.rows = step.iter().map(|(_, rows)| rows).sum();
            for (placement, rows) in step.iter_mut() { **placement += *rows; }
            checked_bucket(self.rows, forced, |rows| Ok(rows.next_power_of_two()))
        }
        fn select(&mut self, batch: &SelectBatch) -> Result<usize> {
            ensure!(batch.rows.len() == self.rows, "selection batch differs from the verify rows");
            Ok(self.rows)
        }
        fn synchronize(&mut self) -> Result<()> { Ok(()) }
        fn len(placement: &usize) -> usize { *placement }
        fn reset(placement: &mut usize, len: usize) { *placement = len; }
        fn replay_ready(&self) -> bool { true }
    }

    #[test]
    fn every_step_starts_at_the_prefilled_length_and_the_probe_resets_after() {
        let timing: VerifyTiming = serde_json::from_value(serde_json::json!({"shapes": ["4x2", "5,4@16"],
            "context": 16, "warmups": 2, "repeats": 3})).unwrap();
        let tokens: Vec<u32> = (0..64).collect();
        let mut fake = Fake { seen: Vec::new(), rows: 0, fail_at: None };
        let mut placements = vec![0usize; 2];
        let record = run(&mut fake, None, "fake", &timing, &tokens, &mut placements).unwrap();
        assert_eq!(fake.seen.len(), 10);
        assert!(fake.seen.iter().all(|lengths| lengths == &[16, 16]), "{:?}", fake.seen);
        assert_eq!(placements, [16, 16], "scratch sequences end at their prefilled length");
        let [a, b] = &record.samples[..] else { panic!() };
        assert_eq!((a.real_rows, a.natural_bucket, a.bucket, a.verify_ms.len(), a.select_ms.len()), (8, 8, 8, 3, 3));
        assert_eq!((b.real_rows, b.natural_bucket, b.bucket, b.shape.as_str()), (9, 16, 16, "5,4@16"));
        assert_eq!(a.contexts, [16, 16]);
        assert!(a.replay_ready && a.graphed && a.graph == GraphDelta::default());
        assert!(run(&mut fake, None, "fake", &timing, &tokens[..22], &mut placements).is_err(), "short prompt");
        let forced: VerifyTiming = serde_json::from_value(serde_json::json!({"shapes": ["4,4@12"], "context": 16})).unwrap();
        assert!(run(&mut fake, None, "fake", &forced, &tokens, &mut vec![0; 2]).is_err(), "12 is not a bucket");
    }

    #[test]
    fn a_failed_step_fails_the_probe_without_partial_samples() {
        let timing: VerifyTiming = serde_json::from_value(serde_json::json!({"shapes": ["4x2"], "context": 16})).unwrap();
        let mut fake = Fake { seen: Vec::new(), rows: 0, fail_at: Some(2) };
        let error = run(&mut fake, None, "fake", &timing, &(0..64).collect::<Vec<_>>(), &mut vec![0; 2]).unwrap_err();
        assert!(format!("{error:#}").contains("injected failure"));
    }
}
