//! Compact device distribution statistics; the acceptance decision stays on the host.
use crate::shared::memory::{DeviceAllocation, HostAllocation};
use anyhow::{ensure, Result};
use cuteafd_core::AcceptanceStats;
use cuteafd_ffi::{programs::{Programs, Scalar}, CuteafdDeviceBuffer, NativeLibrary};
use std::ffi::c_void;

const POINTERS: [&str; 6] = ["logits", "proposals", "stats", "argmax_ids", "argmax_scores", "status"];
const PROGRAM: &str = "shared_relaxed_acceptance_stats_f32";

pub(crate) struct AcceptanceOutput {
    pub greedy: Vec<(u32, f32)>,
    pub stats: Vec<AcceptanceStats>,
}

/// Allocated once at boot, outside graphs. Neither mode nor request identity
/// changes the vocabulary-head graph's pointers or shape.
pub(crate) struct AcceptanceWave<'a> {
    programs: Programs<'a>,
    device: DeviceAllocation<'a>,
    host: HostAllocation<'a>,
    capacity: usize,
}
impl<'a> AcceptanceWave<'a> {
    pub fn device_bytes(capacity: usize) -> usize {
        cuteafd_core::acceptance_wave_device_bytes(capacity)
    }

    pub fn new(library: &'a NativeLibrary, capacity: usize) -> Result<Self> {
        let programs = library.programs()?;
        programs.program(PROGRAM, &POINTERS)?;
        Ok(Self { programs, device: DeviceAllocation::new(library, Self::device_bytes(capacity))?,
            host: HostAllocation::new(library, Self::device_bytes(capacity))?, capacity })
    }

    /// # Safety
    /// Logits and stream belong to this device and remain live until drained.
    pub unsafe fn enqueue(&mut self, logits: CuteafdDeviceBuffer, proposals: &[u32],
        vocab: usize, stride: usize, stream: *mut c_void) -> Result<()> {
        let rows = proposals.len();
        ensure!(rows > 0 && rows <= self.capacity && vocab > 0 && stride >= vocab,
            "invalid acceptance terminal shape");
        ensure!(logits.device_id == self.device.buffer.device_id
            && logits.bytes >= ((rows - 1) * stride + vocab) * 4,
            "acceptance logits extent or device differs");
        let host = self.host.bytes_mut();
        for (word, proposal) in host[..rows * 4].chunks_exact_mut(4).zip(proposals) {
            word.copy_from_slice(&proposal.to_ne_bytes());
        }
        let slice = |offset, bytes| CuteafdDeviceBuffer {
            // SAFETY: each plane fits the persistent 28-byte-per-row allocation.
            ptr: unsafe { self.device.buffer.ptr.cast::<u8>().add(offset) }.cast(),
            bytes, ..self.device.buffer
        };
        let proposal = slice(0, rows * 4);
        let stats = slice(self.capacity * 4, rows * 12);
        let ids = slice(self.capacity * 16, rows * 4);
        let scores = slice(self.capacity * 20, rows * 4);
        let status = slice(self.capacity * 24, rows * 4);
        // SAFETY: all planes and the pinned staging remain owned until caller drains.
        unsafe {
            self.device.library.copy_h2d_async(proposal, &host[..rows * 4], stream)?;
            self.programs.program(PROGRAM, &POINTERS)?.launch(
                &[logits.ptr, proposal.ptr, stats.ptr, ids.ptr, scores.ptr, status.ptr],
                &[Scalar::I32(rows.try_into()?), Scalar::I32(vocab.try_into()?),
                    Scalar::I64(stride.try_into()?)], stream)?;
            self.device.library.copy_d2h_async(&mut host[self.capacity * 4..self.capacity * 4 + rows * 12], stats, stream)?;
            self.device.library.copy_d2h_async(&mut host[self.capacity * 16..self.capacity * 16 + rows * 4], ids, stream)?;
            self.device.library.copy_d2h_async(&mut host[self.capacity * 20..self.capacity * 20 + rows * 4], scores, stream)?;
            self.device.library.copy_d2h_async(&mut host[self.capacity * 24..self.capacity * 24 + rows * 4], status, stream)?;
        }
        Ok(())
    }

    /// Caller must first drain the stream that produced the compact landing.
    pub fn output(&self, proposals: &[u32], vocab: usize) -> Result<AcceptanceOutput> {
        decode(self.host.bytes(), self.capacity, proposals, vocab)
    }
}

fn decode(host: &[u8], capacity: usize, proposals: &[u32], vocab: usize) -> Result<AcceptanceOutput> {
    ensure!(host.len() >= capacity * 28 && proposals.len() <= capacity, "invalid acceptance landing extent");
    let word = |offset| u32::from_ne_bytes(host[offset..offset + 4].try_into().unwrap());
    let float = |offset| f32::from_bits(word(offset));
    let mut greedy = Vec::with_capacity(proposals.len());
    let mut stats = Vec::with_capacity(proposals.len());
    for (row, &proposal) in proposals.iter().enumerate() {
        let status = word(capacity * 24 + row * 4);
        ensure!(status == if proposal == u32::MAX { 1 } else { 0 },
            "acceptance terminal failed at row {row} with status {status}");
        let id = word(capacity * 16 + row * 4);
        let score = float(capacity * 20 + row * 4);
        ensure!((id as usize) < vocab && score.is_finite(), "invalid acceptance argmax at row {row}");
        greedy.push((id, score));
        stats.push(AcceptanceStats { draft_probability: float(capacity * 4 + row * 12),
            argmax_probability: float(capacity * 4 + row * 12 + 4),
            entropy: float(capacity * 4 + row * 12 + 8) });
    }
    Ok(AcceptanceOutput { greedy, stats })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_rows_keep_modes_separate_and_fail_closed() {
        let capacity = 3;
        let mut bytes = vec![0u8; capacity * 28];
        let mut put = |at: usize, value: u32| bytes[at..at + 4].copy_from_slice(&value.to_ne_bytes());
        for row in 0..capacity {
            put(capacity * 16 + row * 4, row as u32);
            put(capacity * 20 + row * 4, 1f32.to_bits());
            put(capacity * 24 + row * 4, u32::from(row == 0));
            put(capacity * 4 + row * 12, 0.2f32.to_bits());
            put(capacity * 4 + row * 12 + 4, 0.5f32.to_bits());
            put(capacity * 4 + row * 12 + 8, 1f32.to_bits());
        }
        let result = decode(&bytes, capacity, &[u32::MAX, 1, 2], 8).unwrap();
        assert_eq!(result.greedy, vec![(0, 1.0), (1, 1.0), (2, 1.0)]);
        let stats = result.stats[1];
        assert!(cuteafd_core::SpeculativeAcceptance::MinP { min_p: 0.3 }.accepts_draft(stats));
        assert!(!cuteafd_core::SpeculativeAcceptance::MinP { min_p: 0.5 }.accepts_draft(stats));
        assert!(decode(&bytes, capacity, &[0, 1, 2], 8).is_err());
        bytes[capacity * 24 + 4..capacity * 24 + 8].copy_from_slice(&2u32.to_ne_bytes());
        assert!(decode(&bytes, capacity, &[u32::MAX, 1, 2], 8).is_err());
    }
}
