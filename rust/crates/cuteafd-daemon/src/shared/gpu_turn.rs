//! Turns between the scheduler and media encodes on the coordinator GPUs.
//!
//! A two-GPU head split issues each step as pushes and peer waits on both devices. A peer wait
//! spins on its GPU until the matching push lands, and the scheduler thread issues that push later
//! in the same step. The media owner (vision/audio tower) is a separate thread with its own CUDA
//! work. When cuBLAS does first-use work for a new GEMM shape (module load, residency), its driver
//! call can wait for the device to settle. If that happens while a step is half issued, the driver
//! call waits on the spinning peer wait, the peer wait waits on the scheduler's next push, and the
//! scheduler waits on the driver: nothing moves until the 60 s peer-wait timeout ends the engine
//! (seen on 2 x RTX PRO 6000, GLM Flash with the bundled vision tower on GPU1: the stall dump shows
//! `media-owner` running inside cuBLAS/libcuda while the scheduler sleeps).
//!
//! The fix keeps an encode from overlapping a half-issued step. A scheduler that issues two-GPU
//! steps holds the turn while it issues work and offers it only between complete engine calls,
//! where every push of the work issued so far has been issued (queued GPU work then completes on
//! its own). Encodes take the turn while they run. Encodes never exclude each other, and without a
//! two-GPU scheduler they never wait.

use std::sync::{Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

#[derive(Default)]
struct State {
    /// A scheduler that issues two-GPU steps exists.
    registered: bool,
    /// That scheduler holds the turn (it may be issuing a step).
    scheduler: bool,
    /// Encodes running.
    media: usize,
    /// Encodes waiting for the scheduler to offer the turn.
    media_waiting: usize,
    /// Encodes finished, for the per-encode stall limit.
    finished: u64,
}

pub(crate) struct GpuTurn {
    state: Mutex<State>,
    changed: Condvar,
}

/// The process-wide turn shared by the scheduler and every media owner.
pub(crate) static GPU_TURN: GpuTurn = GpuTurn::new();

/// How long the scheduler waits for one encode to finish before it declares the encoder stalled.
/// An encode of the largest admitted image takes well under a second.
pub(crate) const MEDIA_STALL: Duration = Duration::from_secs(60);

#[derive(Debug, thiserror::Error)]
pub(crate) enum TurnError {
    #[error("a media encode held the GPU turn for more than {0:?}; the engine cannot continue")]
    MediaStalled(Duration),
    #[error("a second two-GPU scheduler asked for the GPU turn")]
    SecondScheduler,
}

impl GpuTurn {
    pub(crate) const fn new() -> Self {
        Self {
            state: Mutex::new(State { registered: false, scheduler: false, media: 0, media_waiting: 0, finished: 0 }),
            changed: Condvar::new(),
        }
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        // A few counters updated under the lock: a panic elsewhere cannot leave them torn.
        self.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// The scheduler's turn, held for the life of its loop. `two_gpu` is false when the engine
    /// issues no peer waits (one GPU): the turn is then a no-op and encodes never wait.
    pub(crate) fn scheduler(&self, two_gpu: bool) -> Result<SchedulerTurn<'_>, TurnError> {
        self.scheduler_with_stall(two_gpu, MEDIA_STALL)
    }

    fn scheduler_with_stall(&self, two_gpu: bool, stall: Duration) -> Result<SchedulerTurn<'_>, TurnError> {
        if !two_gpu {
            return Ok(SchedulerTurn { turn: self, stall, active: false, held: false });
        }
        {
            let mut state = self.lock();
            if state.registered { return Err(TurnError::SecondScheduler); }
            state.registered = true;
        }
        let mut turn = SchedulerTurn { turn: self, stall, active: true, held: false };
        turn.take_back()?;
        Ok(turn)
    }

    /// Waits for a two-GPU scheduler to offer the turn, then holds it for one encode.
    pub(crate) fn media(&self) -> MediaTurn<'_> {
        let mut state = self.lock();
        state.media_waiting += 1;
        while state.scheduler {
            state = self.changed.wait(state).unwrap_or_else(|p| p.into_inner());
        }
        state.media_waiting -= 1;
        state.media += 1;
        MediaTurn { turn: self }
    }
}

/// Held by the scheduler while it may issue GPU work. Dropping it releases the turn.
pub(crate) struct SchedulerTurn<'a> {
    turn: &'a GpuTurn,
    stall: Duration,
    /// False on one GPU: every method returns at once.
    active: bool,
    held: bool,
}

impl SchedulerTurn<'_> {
    fn release(&mut self) {
        if self.held {
            self.turn.lock().scheduler = false;
            self.held = false;
            self.turn.changed.notify_all();
        }
    }

    /// Takes the turn once no encode runs or waits. Each finished encode restarts the stall limit.
    fn take_back(&mut self) -> Result<(), TurnError> {
        let mut deadline = Instant::now() + self.stall;
        let mut state = self.turn.lock();
        let mut finished = state.finished;
        while state.media > 0 || state.media_waiting > 0 {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                let error = TurnError::MediaStalled(self.stall);
                // The owner is stuck in a CUDA call, so the teardown that joins it would hang and
                // the scheduler's own failure report would never land: report it here.
                cuteafd_transport::health::record_failure(error.to_string());
                return Err(error);
            }
            state = self.turn.changed.wait_timeout(state, left).unwrap_or_else(|p| p.into_inner()).0;
            if state.finished != finished {
                finished = state.finished;
                deadline = Instant::now() + self.stall;
            }
        }
        state.scheduler = true;
        self.held = true;
        Ok(())
    }

    /// Call only between complete engine calls (no step half issued). Lets every waiting encode
    /// run, then takes the turn back. Returns at once when no encode waits.
    pub(crate) fn offer(&mut self) -> Result<(), TurnError> {
        if !self.active || self.turn.lock().media_waiting == 0 { return Ok(()); }
        self.release();
        self.take_back()
    }

    /// Releases the turn while the scheduler blocks with no GPU work to issue (waiting for a new
    /// request), so an idle scheduler never holds encodes back.
    pub(crate) fn idle<T>(&mut self, wait: impl FnOnce() -> T) -> Result<T, TurnError> {
        if !self.active { return Ok(wait()); }
        self.release();
        let value = wait();
        self.take_back()?;
        Ok(value)
    }
}

impl Drop for SchedulerTurn<'_> {
    fn drop(&mut self) {
        if self.active {
            self.release();
            self.turn.lock().registered = false;
        }
    }
}

/// Held for one encode. Dropping it releases the turn.
pub(crate) struct MediaTurn<'a> {
    turn: &'a GpuTurn,
}

impl Drop for MediaTurn<'_> {
    fn drop(&mut self) {
        {
            let mut state = self.turn.lock();
            state.media -= 1;
            state.finished += 1;
        }
        self.turn.changed.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{mpsc, Arc, Barrier};
    use std::thread;

    fn leak() -> &'static GpuTurn { Box::leak(Box::new(GpuTurn::new())) }

    fn wait_for(turn: &GpuTurn, ready: impl Fn(&State) -> bool) {
        while !ready(&turn.lock()) { thread::yield_now(); }
    }

    #[test]
    fn encodes_never_wait_without_a_two_gpu_scheduler() {
        let turn = leak();
        let _one_gpu = turn.scheduler(false).unwrap();
        let first = turn.media();
        let second = turn.media();
        assert_eq!(turn.lock().media, 2);
        drop((first, second));
    }

    #[test]
    fn encodes_run_together_when_the_scheduler_offers() {
        let turn = leak();
        let mut scheduler = turn.scheduler(true).unwrap();
        let barrier = Arc::new(Barrier::new(2));
        let workers: Vec<_> = (0..2).map(|_| {
            let barrier = barrier.clone();
            // Both encodes must hold the turn at once to pass the barrier.
            thread::spawn(move || { let _turn = turn.media(); barrier.wait(); })
        }).collect();
        wait_for(turn, |s| s.media_waiting == 2);
        scheduler.offer().unwrap();
        for worker in workers { worker.join().unwrap(); }
    }

    #[test]
    fn an_encode_never_starts_while_the_scheduler_issues() {
        let turn = leak();
        let mut scheduler = turn.scheduler(true).unwrap();
        let issuing = Arc::new(AtomicBool::new(true));
        let overlapped = Arc::new(AtomicBool::new(false));
        let media = {
            let (issuing, overlapped) = (issuing.clone(), overlapped.clone());
            thread::spawn(move || {
                let _turn = turn.media();
                if issuing.load(Ordering::SeqCst) { overlapped.store(true, Ordering::SeqCst); }
            })
        };
        wait_for(turn, |s| s.media_waiting == 1);
        // A step being issued: the encode stays queued however long it takes.
        thread::sleep(Duration::from_millis(50));
        assert_eq!(turn.lock().media, 0);
        issuing.store(false, Ordering::SeqCst);
        scheduler.offer().unwrap();
        media.join().unwrap();
        assert!(!overlapped.load(Ordering::SeqCst));
        let state = turn.lock();
        assert!(state.scheduler && state.media == 0, "the scheduler holds the turn again after the offer");
    }

    #[test]
    fn offer_without_waiting_encodes_keeps_the_turn() {
        let turn = leak();
        let mut scheduler = turn.scheduler(true).unwrap();
        scheduler.offer().unwrap();
        assert!(turn.lock().scheduler);
    }

    #[test]
    fn every_waiting_encode_runs_before_the_scheduler_resumes() {
        let turn = leak();
        let mut scheduler = turn.scheduler(true).unwrap();
        let done = Arc::new(AtomicUsize::new(0));
        let workers: Vec<_> = (0..3).map(|_| {
            let done = done.clone();
            thread::spawn(move || { let _turn = turn.media(); done.fetch_add(1, Ordering::SeqCst); })
        }).collect();
        wait_for(turn, |s| s.media_waiting == 3);
        scheduler.offer().unwrap();
        assert_eq!(done.load(Ordering::SeqCst), 3);
        for worker in workers { worker.join().unwrap(); }
    }

    #[test]
    fn an_idle_scheduler_lets_encodes_run() {
        let turn = leak();
        let mut scheduler = turn.scheduler(true).unwrap();
        let ran = scheduler.idle(|| thread::spawn(move || { let _turn = turn.media(); true }).join().unwrap()).unwrap();
        assert!(ran);
        assert!(turn.lock().scheduler);
    }

    #[test]
    fn a_stalled_encode_is_reported_not_waited_on_forever() {
        let turn = leak();
        let mut scheduler = turn.scheduler_with_stall(true, Duration::from_millis(50)).unwrap();
        let (release, released) = mpsc::channel::<()>();
        let media = thread::spawn(move || { let _turn = turn.media(); let _ = released.recv(); });
        wait_for(turn, |s| s.media_waiting == 1);
        assert!(matches!(scheduler.offer(), Err(TurnError::MediaStalled(_))));
        release.send(()).unwrap();
        media.join().unwrap();
    }

    #[test]
    fn idle_reports_a_stall_while_taking_the_turn_back() {
        let turn = leak();
        let mut scheduler = turn.scheduler_with_stall(true, Duration::from_millis(50)).unwrap();
        let (release, released) = mpsc::channel::<()>();
        let mut media = None;
        let result = scheduler.idle(|| {
            let (started, running) = mpsc::channel();
            media = Some(thread::spawn(move || { let _turn = turn.media(); started.send(()).unwrap(); let _ = released.recv(); }));
            // The encode holds the turn before the scheduler tries to take it back.
            running.recv().unwrap();
        });
        assert!(matches!(result, Err(TurnError::MediaStalled(_))));
        release.send(()).unwrap();
        media.unwrap().join().unwrap();
    }

    #[test]
    fn the_stall_limit_restarts_after_each_finished_encode() {
        let turn = leak();
        let mut scheduler = turn.scheduler_with_stall(true, Duration::from_millis(150)).unwrap();
        // A runs 0-100 ms; B starts at 60 ms (while the turn is offered) and runs to 180 ms. Both
        // finish under the limit, but together they hold the turn past 150 ms: the limit must
        // restart when A finishes, or the offer fails.
        let a = thread::spawn(move || { let _turn = turn.media(); thread::sleep(Duration::from_millis(100)); });
        let b = thread::spawn(move || {
            wait_for(turn, |s| !s.scheduler && s.media == 1);
            thread::sleep(Duration::from_millis(60));
            let _turn = turn.media();
            thread::sleep(Duration::from_millis(120));
        });
        wait_for(turn, |s| s.media_waiting == 1);
        scheduler.offer().unwrap();
        a.join().unwrap();
        b.join().unwrap();
        assert!(turn.lock().scheduler);
    }

    #[test]
    fn a_second_two_gpu_scheduler_is_refused() {
        let turn = leak();
        let first = turn.scheduler(true).unwrap();
        assert!(matches!(turn.scheduler(true), Err(TurnError::SecondScheduler)));
        drop(first);
        assert!(turn.scheduler(true).is_ok());
    }

    #[test]
    fn dropping_the_scheduler_turn_releases_it() {
        let turn = leak();
        drop(turn.scheduler(true).unwrap());
        let state = turn.lock();
        assert!(!state.scheduler && !state.registered);
    }
}
