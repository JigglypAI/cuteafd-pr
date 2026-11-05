//! Physical peer-exchange backings, shared with the native allocation path.
use anyhow::{ensure, Context, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PeerStorageLayout {
    pub receive_bytes: usize,
    pub control_bytes: usize,
    pub progress_bytes: usize,
}

impl PeerStorageLayout {
    pub fn new(slots: usize, receive_bytes: usize, diagnostics: bool) -> Result<Self> {
        ensure!(slots > 0 && receive_bytes > 0, "empty peer storage shape");
        let flags = slots.checked_add(1).context("peer control count overflow")?;
        Ok(Self {
            receive_bytes: receive_bytes.max(256),
            control_bytes: flags.checked_mul(16).context("peer control size overflow")?.max(256),
            progress_bytes: if diagnostics {
                flags.checked_mul(std::mem::size_of::<cuteafd_ffi::peer_exchange::PeerProgress>())
                    .context("peer progress size overflow")?.max(256)
            } else { 0 },
        })
    }
}
