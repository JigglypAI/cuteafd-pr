//! One paired RTX scheduler and its resident media owners take turns issuing CUDA work.
//!
//! A host driver call in the media owner can wait for a device to settle. If that device is
//! already spinning on a peer push that the scheduler has yet to issue, both hosts can stall.
//! The scheduler offers the turn only once all pushes of its current submission are queued.
//! Pending device work can then complete on its own; Spark exchange never takes this turn.

use std::cell::Cell;
use std::future::Future;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};
use tokio::sync::Notify;

#[derive(Default)]
struct State {
    scheduler: bool,
    media_waiting: usize,
    joining: usize,
    media_deadline: Option<Instant>,
    stalled: bool,
    closed: bool,
}

pub(crate) struct GpuTurn {
    stall: Duration,
    state: Mutex<State>,
    changed: Condvar,
    idle_changed: Notify,
}

#[derive(Clone, Copy, Debug, thiserror::Error)]
#[error("a media owner held the GPU turn for more than {0:?}; the engine cannot continue")]
pub(crate) struct MediaStalled(Duration);

impl GpuTurn {
    pub(crate) fn new() -> Self {
        Self::with_stall(Duration::from_secs(60))
    }

    fn with_stall(stall: Duration) -> Self {
        Self {
            stall,
            state: Mutex::new(State::default()),
            changed: Condvar::new(),
            idle_changed: Notify::new(),
        }
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn check(&self, state: &mut State) -> Result<(), MediaStalled> {
        state.stalled |= state
            .media_deadline
            .is_some_and(|deadline| Instant::now() >= deadline);
        if state.stalled {
            Err(MediaStalled(self.stall))
        } else {
            Ok(())
        }
    }

    /// Used by the single scheduler after startup, before serving requests.
    pub(crate) fn scheduler(&self) -> Result<SchedulerTurn<'_>, MediaStalled> {
        self.take()?;
        Ok(SchedulerTurn(self))
    }

    fn take(&self) -> Result<(), MediaStalled> {
        let mut state = self.lock();
        loop {
            self.check(&mut state)?;
            state = if let Some(deadline) = state.media_deadline {
                let left = deadline.saturating_duration_since(Instant::now());
                self.changed
                    .wait_timeout(state, left)
                    .unwrap_or_else(|p| p.into_inner())
                    .0
            } else if state.media_waiting != 0 || state.joining != 0 {
                self.changed.wait(state).unwrap_or_else(|p| p.into_inner())
            } else {
                state.scheduler = true;
                return Ok(());
            };
        }
    }

    fn release(&self) {
        self.lock().scheduler = false;
        self.changed.notify_all();
    }

    /// Used by each local owner for both encoding and native teardown.
    pub(crate) fn media(&self) -> MediaTurn<'_> {
        let mut state = self.lock();
        state.media_waiting += 1;
        while state.scheduler || state.media_deadline.is_some() {
            state = self.changed.wait(state).unwrap_or_else(|p| p.into_inner());
        }
        state.media_waiting -= 1;
        state.media_deadline = Some(Instant::now() + self.stall);
        let closed = state.stalled || state.closed;
        self.changed.notify_all();
        self.idle_changed.notify_one();
        MediaTurn { turn: self, closed }
    }

    /// Block new paired submissions through the thread join. This is not a
    /// media permit: the closing thread must still be free to drain and exit.
    pub(crate) fn joining(&self) -> JoinTurn<'_> {
        let mut state = self.lock();
        state.joining += 1;
        while state.scheduler {
            state = self.changed.wait(state).unwrap_or_else(|p| p.into_inner());
        }
        JoinTurn(self)
    }

    async fn watch_idle(&self) -> MediaStalled {
        loop {
            let deadline = {
                let mut state = self.lock();
                if let Err(error) = self.check(&mut state) {
                    return error;
                }
                state.media_deadline
            };
            // Notify stores a permit if the owner changed state after the snapshot, before
            // we started waiting. With no owner working, idle has neither a timer nor a poll.
            if let Some(deadline) = deadline {
                tokio::select! {
                    _ = self.idle_changed.notified() => {},
                    _ = tokio::time::sleep_until(deadline.into()) => {},
                }
            } else {
                self.idle_changed.notified().await;
            }
        }
    }
}

pub(crate) struct SchedulerTurn<'a>(&'a GpuTurn);

impl SchedulerTurn<'_> {
    pub(crate) fn check(&self) -> Result<(), MediaStalled> {
        self.0.check(&mut self.0.lock())
    }

    /// Call only between complete submissions. If the owner is not waiting, retain the turn.
    pub(crate) fn offer(&self) -> Result<Duration, MediaStalled> {
        {
            let mut state = self.0.lock();
            self.0.check(&mut state)?;
            if state.media_waiting == 0 && state.joining == 0 {
                return Ok(Duration::ZERO);
            }
            state.scheduler = false;
        }
        let started = Instant::now();
        self.0.changed.notify_all();
        self.0.take()?;
        Ok(started.elapsed())
    }

    /// Allow media while idle, but report a stall even if no request wakes the scheduler.
    pub(crate) async fn idle<T>(&self, wait: impl Future<Output = T>) -> Result<T, MediaStalled> {
        self.0.release();
        let value = tokio::select! {
            biased;
            error = self.0.watch_idle() => return Err(error),
            value = wait => value,
        };
        self.0.take()?;
        Ok(value)
    }
}

impl Drop for SchedulerTurn<'_> {
    fn drop(&mut self) {
        {
            let mut state = self.0.lock();
            state.closed = true;
            state.scheduler = false;
        }
        self.0.changed.notify_all();
    }
}

pub(crate) struct JoinTurn<'a>(&'a GpuTurn);

impl Drop for JoinTurn<'_> {
    fn drop(&mut self) {
        self.0.lock().joining -= 1;
        self.0.changed.notify_all();
    }
}

pub(crate) struct MediaTurn<'a> {
    turn: &'a GpuTurn,
    pub(crate) closed: bool,
}

impl Drop for MediaTurn<'_> {
    fn drop(&mut self) {
        {
            let mut state = self.turn.lock();
            let _ = self.turn.check(&mut state);
            state.media_deadline = None;
        }
        self.turn.changed.notify_all();
        self.turn.idle_changed.notify_one();
    }
}

/// Scheduler-side adapter; absent on a single GPU or without a local tower.
pub(crate) struct SchedulerMedia<'a> {
    turn: Option<SchedulerTurn<'a>>,
    excluded: Cell<Duration>,
}

impl<'a> SchedulerMedia<'a> {
    pub(crate) fn new(turn: Option<&'a GpuTurn>) -> Result<Self, MediaStalled> {
        Ok(Self {
            turn: turn.map(GpuTurn::scheduler).transpose()?,
            excluded: Cell::default(),
        })
    }

    pub(crate) fn offer(
        &self,
        require_live: impl FnOnce() -> anyhow::Result<()>,
    ) -> anyhow::Result<()> {
        if let Some(turn) = &self.turn {
            require_live()?;
            let waited = turn.offer()?;
            if !waited.is_zero() {
                self.excluded.set(self.excluded.get() + waited);
            }
        }
        Ok(())
    }

    pub(crate) fn prefill_excluded(&self) -> &Cell<Duration> {
        self.excluded.set(Duration::ZERO);
        &self.excluded
    }

    pub(crate) fn check(&self) -> Result<(), MediaStalled> {
        self.turn.as_ref().map_or(Ok(()), SchedulerTurn::check)
    }

    pub(crate) fn receive<T>(
        &self,
        runtime: &tokio::runtime::Runtime,
        receive: &mut tokio::sync::mpsc::Receiver<T>,
    ) -> Result<Option<T>, MediaStalled> {
        match &self.turn {
            Some(turn) => runtime.block_on(turn.idle(receive.recv())),
            None => Ok(receive.blocking_recv()),
        }
    }
}

pub(crate) fn paired_turn(paired: bool) -> Option<Arc<GpuTurn>> {
    paired.then(|| Arc::new(GpuTurn::new()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, mpsc};
    use std::thread;

    const TEST_WAIT: Duration = Duration::from_secs(5);

    fn until_waiting(turn: &GpuTurn) {
        let deadline = Instant::now() + TEST_WAIT;
        while turn.lock().media_waiting == 0 {
            assert!(Instant::now() < deadline, "media did not queue");
            thread::yield_now();
        }
    }

    #[test]
    fn completed_submission_offers_the_turn_through_media_teardown() {
        let turn = Arc::new(GpuTurn::new());
        let scheduler = turn.scheduler().unwrap();
        let (stages, observed) = mpsc::channel();
        let owner = {
            let turn = turn.clone();
            thread::spawn(move || {
                let _media = turn.media();
                assert!(!turn.lock().scheduler);
                stages.send("encode").unwrap();
                assert!(!turn.lock().scheduler);
                stages.send("teardown").unwrap();
            })
        };
        until_waiting(&turn);
        assert!(
            observed.try_recv().is_err(),
            "media entered during a submission"
        );
        scheduler.offer().unwrap();
        assert_eq!(observed.try_recv(), Ok("encode"));
        assert_eq!(observed.try_recv(), Ok("teardown"));
        owner.join().unwrap();
        assert!(turn.lock().scheduler);
    }

    #[test]
    fn scheduler_exit_allows_teardown_but_not_a_queued_encode() {
        let turn = Arc::new(GpuTurn::new());
        let scheduler = turn.scheduler().unwrap();
        let owner = {
            let turn = turn.clone();
            thread::spawn(move || {
                let media = turn.media();
                assert!(media.closed);
                assert!(!turn.lock().scheduler);
            })
        };
        until_waiting(&turn);
        drop(scheduler);
        owner.join().unwrap();
    }

    #[test]
    fn two_owners_and_join_share_a_completed_boundary() {
        let turn = Arc::new(GpuTurn::with_stall(TEST_WAIT));
        let scheduler = turn.scheduler().unwrap();
        let (done, finished) = mpsc::channel();
        let owners: Vec<_> = (0..2)
            .map(|id| {
                let (turn, done) = (turn.clone(), done.clone());
                thread::spawn(move || {
                    let _media = turn.media();
                    assert!(!turn.lock().scheduler);
                    done.send(id).unwrap();
                })
            })
            .collect();
        let joiner = {
            let turn = turn.clone();
            thread::spawn(move || {
                let _join = turn.joining();
                for owner in owners {
                    owner.join().unwrap();
                }
                assert!(!turn.lock().scheduler);
            })
        };
        let deadline = Instant::now() + TEST_WAIT;
        loop {
            let state = turn.lock();
            if state.joining == 1 && state.media_waiting == 2 {
                break;
            }
            drop(state);
            assert!(Instant::now() < deadline, "owner or join did not queue");
            thread::yield_now();
        }
        assert!(finished.try_recv().is_err());
        scheduler.offer().unwrap();
        joiner.join().unwrap();
        assert_eq!(finished.try_iter().count(), 2);
        assert!(turn.lock().scheduler);
    }

    #[test]
    fn idle_reports_stall_before_receiving_a_request_or_joining_media() {
        let turn = Arc::new(GpuTurn::with_stall(Duration::from_millis(50)));
        let scheduler = turn.scheduler().unwrap();
        let (release, released) = mpsc::channel::<()>();
        let owner = {
            let turn = turn.clone();
            thread::spawn(move || {
                let _media = turn.media();
                released.recv_timeout(TEST_WAIT).unwrap();
            })
        };
        until_waiting(&turn);
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .unwrap();
        let result = runtime.block_on(async {
            tokio::time::timeout(TEST_WAIT, scheduler.idle(std::future::pending::<()>()))
                .await
                .unwrap()
        });
        assert!(matches!(result, Err(MediaStalled(_))));
        release.send(()).unwrap();
        owner.join().unwrap();
        assert!(scheduler.check().is_err());
    }

    #[test]
    fn deadline_follows_owner_acquisition_not_the_offer_or_previous_encode() {
        let turn = GpuTurn::with_stall(TEST_WAIT);
        let media = turn.media();
        // Simulate expiry without timing the test around a narrow sleep.
        turn.lock().media_deadline = Some(Instant::now());
        drop(media);
        assert!(
            turn.scheduler().is_err(),
            "late completion must not hide a stall"
        );
        assert!(turn.media().closed, "a stalled turn stays terminal");
    }
}
