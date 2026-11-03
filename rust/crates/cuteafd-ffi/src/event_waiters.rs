//! Sleeping, independent completion waits for a bounded set of copy streams.
use crate::*;
use std::{sync::mpsc, thread::JoinHandle};

struct Wait {
    event: *mut c_void,
    tag: u32,
}
// SAFETY: submit's caller retains the recorded event until all workers join;
// CUDA permits event synchronization from a different host thread.
unsafe impl Send for Wait {}

pub struct EventWaiters {
    queues: Vec<mpsc::Sender<Wait>>,
    completed: mpsc::Receiver<(u32, Result<()>)>,
    workers: Vec<JoinHandle<()>>,
}

impl EventWaiters {
    pub fn new(library: &NativeLibrary, device: i32, streams: usize) -> Result<Self> {
        anyhow::ensure!(streams > 0, "empty event waiter set");
        let synchronize = library.cuteafd_cuda_event_synchronize()?;
        let set_device = library.cuteafd_cuda_set_device()?;
        let last_error = library.cuteafd_last_error()?;
        let (complete, completed) = mpsc::channel();
        let mut waiters = Self { queues: Vec::new(), completed, workers: Vec::new() };
        for index in 0..streams {
            let module = Arc::clone(&library.lib);
            let (queue, pending) = mpsc::channel::<Wait>();
            let complete = complete.clone();
            let worker = std::thread::Builder::new()
                .name(format!("push-copy-wait-{device}-{index}"))
                .spawn(move || {
                    let _module = module;
                    while let Ok(wait) = pending.recv() {
                        // SAFETY: the module is retained, output buffer is local,
                        // and submit's caller retains every recorded event.
                        let status = unsafe {
                            let status = set_device(device);
                            if status == 0 { synchronize(wait.event) } else { status }
                        };
                        let result = if status == 0 { Ok(()) } else {
                            let mut text = [0 as c_char; 512];
                            // SAFETY: valid writable buffer and retained function pointer.
                            let message = if unsafe { last_error(text.as_mut_ptr(), text.len()) } == 0 {
                                // Do not rely on an erroneous DSO to terminate its message.
                                text[text.len() - 1] = 0;
                                unsafe { CStr::from_ptr(text.as_ptr()) }.to_string_lossy().into_owned()
                            } else { "last error unavailable".into() };
                            Err(anyhow::anyhow!("push copy event wait returned status {status}: {message}"))
                        };
                        if complete.send((wait.tag, result)).is_err() { break; }
                    }
                })?;
            waiters.queues.push(queue);
            waiters.workers.push(worker);
        }
        Ok(waiters)
    }

    /// # Safety
    /// The event was recorded on this worker's stream using a blocking event.
    /// Retain it, its copy sources and destinations until completion or join.
    /// Do not record it again before receiving its completion.
    pub unsafe fn submit(&self, stream: usize, event: *mut c_void, tag: u32) -> Result<()> {
        self.queues.get(stream).context("invalid copy waiter index")?
            .send(Wait { event, tag }).map_err(|_| anyhow::anyhow!("copy waiter closed"))
    }

    /// Sleep for a completion, with a bounded wait so the receiver can also
    /// service newly arriving socket events when only a slow stream is pending.
    pub fn complete(&self, timeout: Option<std::time::Duration>) -> Result<Option<u32>> {
        let completion = match timeout {
            Some(timeout) => match self.completed.recv_timeout(timeout) {
                Ok(value) => value,
                Err(mpsc::RecvTimeoutError::Timeout) => return Ok(None),
                Err(mpsc::RecvTimeoutError::Disconnected) => anyhow::bail!("copy waiters closed"),
            },
            None => self.completed.recv().context("copy waiters closed")?,
        };
        completion.1?;
        Ok(Some(completion.0))
    }

    /// Close submissions and join every waiter, including on cancellation.
    pub fn join(&mut self) -> Result<()> {
        self.queues.clear();
        let mut failed = false;
        for worker in self.workers.drain(..) {
            failed |= worker.join().is_err();
        }
        anyhow::ensure!(!failed, "copy waiter panicked");
        Ok(())
    }
}

impl Drop for EventWaiters {
    fn drop(&mut self) {
        if let Err(error) = self.join() { tracing::error!(%error, "joining copy event waiters"); }
    }
}
