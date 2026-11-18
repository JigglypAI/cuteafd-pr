//! The device-ordered vocabulary handoff used by V4.1, without its shape or weights.
use super::memory::device::{Event, Stream};
use anyhow::{ensure, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer as Buffer, V41PeerCopy};
use std::{cell::RefCell, rc::Rc};

pub(crate) struct Handoff<'a> {
    lead: Rc<Stream<'a>>,
    peer: Rc<Stream<'a>>,
    ready: RefCell<Event<'a>>,
    finished: RefCell<Event<'a>>,
    acquired: RefCell<Event<'a>>,
    copy: V41PeerCopy<'a>,
}
impl<'a> Handoff<'a> {
    pub fn new(lead: Rc<Stream<'a>>, peer: Rc<Stream<'a>>) -> Result<Self> {
        let (a,b) = (lead.device,peer.device);
        ensure!(a.id != b.id && std::ptr::eq(a.library,b.library), "head handoff requires peer devices");
        for (d, other) in [(a,b), (b,a)] { d.run(|| d.library.cuda_enable_peer(other.id))?; }
        let copy = b.run(|| b.library.v41_peer_copy())?;
        Ok(Self { lead, peer, ready: RefCell::new(Event::new(a)?), finished: RefCell::new(Event::new(b)?),
            acquired: RefCell::new(Event::new(a)?), copy })
    }

    /// Queue both projections against the same input, followed by a gather on
    /// the lead stream. Its next event also protects peer scratch from reuse.
    pub fn project(&self, input: Buffer, remote_input: Buffer, local: impl FnOnce() -> Result<()>,
        remote: impl FnOnce() -> Result<()>, gather: impl FnOnce() -> Result<()>) -> Result<()> {
        let (lead,peer) = (&self.lead,&self.peer);
        let lib = lead.device.library;
        self.ready.borrow_mut().record(lead)?;
        peer.device.run(|| {
            // SAFETY: the input is published by the lead stream; all buffers
            // remain owned through the gather and the subsequent release edge.
            unsafe {
                lib.cuda_stream_wait_event(peer.raw, self.ready.borrow().raw)?;
                self.copy.launch(remote_input,input,input.bytes,peer.raw)?;
            }
            remote()
        })?;
        self.finished.borrow_mut().record(peer)?;
        lead.device.run(|| {
            local()?;
            // SAFETY: the peer records completion behind all shard writes.
            unsafe { lib.cuda_stream_wait_event(lead.raw,self.finished.borrow().raw)?; }
            gather()
        })?;
        self.acquired.borrow_mut().record(lead)?;
        peer.device.run(|| {
            // SAFETY: the next peer producer may only reuse scratch once the
            // lead's pitched read completes. The event wait is device-side.
            unsafe { lib.cuda_stream_wait_event(peer.raw,self.acquired.borrow().raw)?; }
            Ok(())
        })
    }

    pub fn copy_columns(&self, dst: Buffer, src: Buffer, rows: usize,
        dst_pitch: usize, src_pitch: usize, width: usize) -> Result<()> {
        ensure!(dst.device_id == self.lead.device.id, "head gather destination must be on lead");
        // SAFETY: caller invokes from the gather closure after the peer event;
        // checked views retain the source and destination through the release.
        unsafe { self.copy.launch_rows(dst,src,width,rows,dst_pitch,src_pitch,self.lead.raw) }?;
        Ok(())
    }
}
