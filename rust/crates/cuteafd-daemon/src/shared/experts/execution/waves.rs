//! Spark wave merging (#844), shared by every expert backend.
//!
//! Units for the same layer that arrive while the owner stream is busy are
//! copied from their retained RDMA frames into per-slot device slices on a
//! copy stream (GB10 has no GPUDirect RDMA; FC1 gathering mapped host rows in
//! place measured 4.9x slower). They flush as one launch of at most
//! [`MAX_MERGED_UNITS`] units when compute is idle, the layer or dtype changes,
//! intake is blocked, or the wave is full. A merged launch reads every member
//! row and writes every member reply through row-address tables, so output is
//! byte-identical to singleton launches.
//!
//! The component is dtype-generic: a staged slot holds `unit_rows` rows of at
//! most `row_bytes` each, and every request stages at its own ingress row
//! stride (FP8 K32, BF16, or a packed NVFP4 wire row). Backends decide whether
//! a request is eligible and supply only the launch of a [`Wave`]; staging
//! owns the copies, the address and route tables, the flush policy, cancel and
//! each member's completion fence on the backend's owner queue.
use crate::shared::experts::execution::timing::PendingUnits;
use crate::shared::experts::service::routes::RouteStaging;
use crate::shared::memory::{device, DeviceAllocation, HostAllocation, LoadStream, Tag};
use anyhow::{ensure, Context, Result};
use cuteafd_core::spark_executions::MAX_MERGED_UNITS;
use cuteafd_ffi::{CuteafdDeviceBuffer, CuteafdHostBuffer, NativeLibrary};
use cuteafd_transport::{expert::BackboneRequest, ExpertV2Dtype};
use std::ffi::c_void;
use std::rc::Rc;

/// (request id, layer id, rows, executor id): a member's reply identity.
pub(crate) type UnitIdentity = (u64, u32, u32, u64);

pub(crate) struct StagedUnit {
    pub(crate) slot: usize,
    pub(crate) layer: usize,
    pub(crate) identity: UnitIdentity,
    /// The retained mapped frame this unit was copied from (cancel key).
    pub(crate) source_address: usize,
    /// The unit's device slice (never the mapped frame).
    pub(crate) input: CuteafdDeviceBuffer,
    pub(crate) output: CuteafdDeviceBuffer,
    pub(crate) input_dtype: ExpertV2Dtype,
    pub(crate) output_dtype: ExpertV2Dtype,
}

/// One launch over staged members, in staging order.
pub(crate) struct Wave<'w> {
    pub(crate) layer: usize,
    pub(crate) rows: usize,
    pub(crate) members: &'w [StagedUnit],
    /// Row-address tables (`segmented`), else the lone member's contiguous
    /// device slice and reply rows.
    pub(crate) input: CuteafdDeviceBuffer,
    pub(crate) output: CuteafdDeviceBuffer,
    pub(crate) ids: CuteafdDeviceBuffer,
    pub(crate) weights: CuteafdDeviceBuffer,
    pub(crate) segmented: bool,
    pub(crate) input_dtype: ExpertV2Dtype,
    pub(crate) output_dtype: ExpertV2Dtype,
}

/// A staged request: the routed rows, their landed input and the reply rows.
pub(crate) struct Arrival<'r, 'b> {
    pub(crate) request: &'r BackboneRequest<'b>,
    /// The retained mapped request rows.
    pub(crate) input: CuteafdDeviceBuffer,
    /// The retained mapped reply rows.
    pub(crate) output: CuteafdDeviceBuffer,
    /// The backend's resident layer index for this request.
    pub(crate) layer: usize,
    pub(crate) executor_id: u64,
}

pub(crate) struct WaveStaging<'a> {
    library: &'a NativeLibrary,
    copy_stream: Rc<LoadStream<'a>>,
    copied: Vec<device::Event<'a>>,
    /// Per-slot device slices, `unit_rows * row_bytes` apart.
    plane: CuteafdDeviceBuffer,
    row_bytes: usize,
    unit_rows: usize,
    /// Rows of the widest merged wave (the address/route table extent).
    rows: usize,
    /// Members below this many rows launch alone (the package's recipe bound).
    minimum_rows: usize,
    topk: usize,
    /// Per slot: `unit_rows` input addresses, then `unit_rows` output addresses.
    descriptors: Vec<HostAllocation<'a>>,
    /// Input address table, route ids, route weights, output address table.
    device: [DeviceAllocation<'a>; 4],
    pub(crate) staged: Vec<StagedUnit>,
    pub(crate) staged_units: u64,
    pub(crate) flushes: u64,
    /// Flushes that launched two or more members through row addresses.
    pub(crate) merged: u64,
}

fn device_part(buffer: CuteafdDeviceBuffer, offset: usize, bytes: usize) -> Result<CuteafdDeviceBuffer> {
    ensure!(offset.checked_add(bytes).is_some_and(|end| end <= buffer.bytes), "wave staging device extent overflow");
    Ok(CuteafdDeviceBuffer { ptr: buffer.ptr.cast::<u8>().wrapping_add(offset).cast(), bytes, ..buffer })
}

impl<'a> WaveStaging<'a> {
    /// Pinned row-address descriptors one staged unit retains.
    pub(crate) fn pinned_bytes(extent: cuteafd_core::OwnerExtent) -> usize {
        extent.staging_rows as usize * 16
    }

    /// Device address and route tables at the widest merged wave.
    pub(crate) fn table_bytes(extent: cuteafd_core::OwnerExtent, topk: usize) -> Option<usize> {
        (extent.segment_rows()? as usize).checked_mul(16 + topk * 8)
    }

    /// Staging for a prefill owner whose package has a row-addressed program,
    /// or `None` when this owner never merges (decode, one buffered unit).
    /// `plane` holds every buffered unit's slice of `row_bytes`-wide rows.
    pub(crate) fn new(library: &'a NativeLibrary, extent: cuteafd_core::OwnerExtent, plane: CuteafdDeviceBuffer,
        row_bytes: usize, minimum_rows: usize) -> Result<Option<Self>> {
        let (Some(rows), Some(staged)) = (extent.segment_rows(), extent.staged_rows()) else { return Ok(None) };
        let (rows, unit_rows) = (rows as usize, extent.staging_rows as usize);
        let topk = cuteafd_core::expert_geometry().topk as usize;
        ensure!(row_bytes > 0 && row_bytes % 16 == 0, "staged rows must be a nonzero multiple of 16 bytes");
        ensure!((staged as usize).checked_mul(row_bytes).is_some_and(|bytes| bytes <= plane.bytes),
            "staged input plane is smaller than every buffered unit");
        let device = device::Device { library, id: library.cuda_get_device()? };
        let units = extent.buffered_units as usize;
        let descriptors = (0..units).map(|_| HostAllocation::new(library, Self::pinned_bytes(extent),
            crate::shared::memory::tag(crate::shared::memory::MemoryKind::Workspace, "experts/workspace/row_addresses")))
            .collect::<Result<Vec<_>>>()?;
        let table = |bytes| DeviceAllocation::new(library, bytes, Tag::workspace("experts/wave-tables"));
        Ok(Some(Self {
            library,
            copy_stream: Rc::new(LoadStream { library, raw: library.cuda_stream_create()? }),
            copied: (0..units).map(|_| device::Event::new(device)).collect::<Result<_>>()?,
            plane, row_bytes, unit_rows, rows, minimum_rows: minimum_rows.max(1), topk, descriptors,
            device: [table(rows * 8)?, table(rows * topk * 4)?, table(rows * topk * 4)?, table(rows * 8)?],
            staged: Vec::with_capacity(units), staged_units: 0, flushes: 0, merged: 0,
        }))
    }

    /// Groups on one owner stream share one copy stream.
    pub(crate) fn share_copy_stream(&mut self, other: &Self) {
        self.copy_stream = other.copy_stream.clone();
    }

    pub(crate) fn copy_stream(&self) -> *mut c_void { self.copy_stream.raw }

    /// Rows of the widest merged wave.
    pub(crate) fn rows(&self) -> usize { self.rows }

    /// The staged slices' device plane.
    pub(crate) fn plane(&self) -> CuteafdDeviceBuffer { self.plane }

    pub(crate) fn table_bytes_allocated(&self) -> usize {
        self.device.iter().map(|allocation| allocation.buffer.bytes).sum()
    }

    pub(crate) fn descriptor_bytes_allocated(&self) -> usize {
        self.descriptors.iter().map(|allocation| allocation.buffer.bytes).sum()
    }

    pub(crate) fn has_staged(&self) -> bool { !self.staged.is_empty() }

    /// The request's ingress row stride when it can stage here: its rows fit
    /// one unit slice at a 16-byte-aligned stride. The backend has already
    /// matched its dtype to a row-addressed program.
    fn stride(&self, request: &BackboneRequest<'_>) -> Option<usize> {
        let rows = request.rows() as usize;
        let stride = request.input_dtype().row_bytes(cuteafd_core::expert_geometry().hidden as usize).ok()?;
        (rows > 0 && rows <= self.unit_rows && stride <= self.row_bytes && stride % 16 == 0
            && request.hidden().len() >= rows * stride).then_some(stride)
    }

    /// Retire a staged unit whose peer is removed. Its copy completes first,
    /// so the caller may release the mapped frame and reuse the slot.
    pub(crate) fn cancel(&mut self, pending: &mut PendingUnits<'a>, input_address: usize) -> Result<bool> {
        let Some(index) = self.staged.iter().position(|unit| unit.source_address == input_address) else { return Ok(false) };
        // SAFETY: cancellation retires the already-issued copy before the
        // caller can release the mapped frame or reuse this device slot.
        unsafe { self.library.cuda_event_synchronize(self.copied[self.staged[index].slot].raw)?; }
        let unit = self.staged.remove(index);
        pending.cancel_reserved(unit.slot)?;
        Ok(true)
    }

    /// Launch every staged unit as one wave on `stream`; each member keeps
    /// its own completion fence. `launch` queues the expert math only.
    pub(crate) fn flush(&mut self, pending: &mut PendingUnits<'a>, routes: &RouteStaging<'a>, stream: *mut c_void,
        launch: &mut dyn FnMut(&Wave<'_>) -> Result<()>) -> Result<()> {
        if self.staged.is_empty() { return Ok(()) }
        let rows: usize = self.staged.iter().map(|unit| unit.identity.2 as usize).sum();
        ensure!(rows <= self.rows && self.staged.len() <= MAX_MERGED_UNITS as usize,
            "staged wave exceeds admitted rows");
        self.flushes += 1;
        tracing::debug!(target: "cuteafd::spark_coalesce", staged_units = self.staged_units,
            flushes = self.flushes, segments = self.staged.len(), rows, "Spark staged flush");
        let first_unit = &self.staged[0];
        let (layer, input_dtype, output_dtype) = (first_unit.layer, first_unit.input_dtype, first_unit.output_dtype);
        let segmented = self.staged.len() > 1;
        self.merged += u64::from(segmented);
        let topk = self.topk;
        let mut first = 0;
        for unit in &self.staged {
            ensure!(unit.layer == layer && unit.input_dtype == input_dtype && unit.output_dtype == output_dtype,
                "staged units have different layers or dtypes");
            let count = unit.identity.2 as usize;
            // SAFETY: every member's slice and mapped reply stay retained until
            // its fence below; the tables are overwritten only in stream order.
            unsafe {
                self.library.cuda_stream_wait_event(stream, self.copied[unit.slot].raw)?;
                routes.upload(unit.slot, 0, count * topk,
                    device_part(self.device[1].buffer, first * topk * 4, count * topk * 4)?,
                    device_part(self.device[2].buffer, first * topk * 4, count * topk * 4)?, stream)?;
                if segmented {
                    let source = self.descriptors[unit.slot].buffer;
                    let outputs = CuteafdHostBuffer { ptr: source.ptr.cast::<u8>()
                        .wrapping_add(self.unit_rows * 8).cast(), bytes: count * 8, ..source };
                    self.library.copy_host_buffers_h2d_batch_async(&[
                        device_part(self.device[0].buffer, first * 8, count * 8)?,
                        device_part(self.device[3].buffer, first * 8, count * 8)?],
                        &[source, outputs], &[count * 8, count * 8], stream)?;
                }
            }
            first += count;
        }
        let wave = Wave { layer, rows, members: &self.staged,
            input: if segmented { self.device[0].buffer } else { first_unit.input },
            output: if segmented { self.device[3].buffer } else { first_unit.output },
            ids: self.device[1].buffer, weights: self.device[2].buffer, segmented, input_dtype, output_dtype };
        launch(&wave)?;
        for unit in &self.staged {
            // SAFETY: the shared launch precedes every member fence on `stream`.
            unsafe { pending.record_reserved(unit.slot, unit.identity, stream)?; }
        }
        self.staged.clear();
        Ok(())
    }

    /// Stage `arrival` (copy it now, launch later) and return whether it was
    /// taken. Ineligible requests return `false` after nothing was queued.
    pub(crate) fn stage(&mut self, arrival: Arrival<'_, '_>, pending: &mut PendingUnits<'a>,
        routes: &mut RouteStaging<'a>, stream: *mut c_void, launch: &mut dyn FnMut(&Wave<'_>) -> Result<()>)
        -> Result<bool> {
        let request = arrival.request;
        let rows = request.rows() as usize;
        let Some(stride) = self.stride(request) else {
            tracing::debug!(target: "cuteafd::spark_coalesce", rows, reason = "incompatible_input_or_capacity", "Spark staging bypass");
            return Ok(false)
        };
        let (input_dtype, output_dtype) = (request.input_dtype(), request.output_dtype());
        let singleton = rows < self.minimum_rows;
        if singleton {
            tracing::debug!(target: "cuteafd::spark_coalesce", rows, reason = "singleton_recipe", "Spark staged boundary");
            self.flush(pending, routes, stream, launch)?;
        }
        if self.staged.first().is_some_and(|unit| unit.layer != arrival.layer
            || unit.input_dtype != input_dtype || unit.output_dtype != output_dtype) {
            tracing::debug!(target: "cuteafd::spark_coalesce", reason = "different_layer", "Spark staged boundary");
            self.flush(pending, routes, stream, launch)?;
        }
        let staged_rows: usize = self.staged.iter().map(|unit| unit.identity.2 as usize).sum();
        if staged_rows + rows > self.rows { self.flush(pending, routes, stream, launch)?; }
        let output_stride = request.output_row_bytes()? as usize;
        let (input, output) = (arrival.input, arrival.output);
        ensure!((input.ptr as usize) % 16 == 0 && (output.ptr as usize) % 16 == 0,
            "segmented mapped rows require 16-byte aligned input/output payloads");
        ensure!(input.bytes >= rows * stride && output.bytes >= rows * output_stride,
            "staged mapped row extent is too small");
        let slot = pending.next_slot()?;
        routes.pack(slot, request)?;
        let resident = device_part(self.plane, slot * self.unit_rows * self.row_bytes, rows * stride)?;
        let bytes = self.descriptors[slot].bytes_mut();
        for row in 0..rows {
            let input_address = (resident.ptr as usize).checked_add(row * stride).context("input row address overflow")?;
            let output_address = (output.ptr as usize).checked_add(row * output_stride)
                .context("output row address overflow")?;
            bytes[row * 8..row * 8 + 8].copy_from_slice(&(input_address as u64).to_le_bytes());
            let at = self.unit_rows * 8 + row * 8;
            bytes[at..at + 8].copy_from_slice(&(output_address as u64).to_le_bytes());
        }
        ensure!(pending.reserve()? == slot, "staged slot reservation changed");
        self.staged.push(StagedUnit { slot, layer: arrival.layer, source_address: input.ptr as usize,
            identity: (request.request_id(), request.layer(), request.rows(), arrival.executor_id),
            input: resident, output, input_dtype, output_dtype });
        let copy = self.copy_stream.raw;
        // SAFETY: the reserved unit retains its mapped source, device slice and
        // event. The copy is host-triggered now, with no wait on future compute.
        unsafe {
            pending.arrive_reserved(slot, copy)?;
            self.library.copy_d2d_async(resident, input, rows * stride, copy)?;
            self.library.cuda_event_record(self.copied[slot].raw, copy)?;
        }
        self.staged_units += 1;
        tracing::debug!(target: "cuteafd::spark_coalesce", staged_units = self.staged_units,
            staged = self.staged.len(), rows, slot, "Spark unit copied and staged");
        let staged_rows = staged_rows_after(&self.staged);
        let full = self.staged.len() == self.copied.len().min(MAX_MERGED_UNITS as usize) || staged_rows >= self.rows;
        if singleton || full { self.flush(pending, routes, stream, launch)?; }
        Ok(true)
    }
}

fn staged_rows_after(staged: &[StagedUnit]) -> usize {
    staged.iter().map(|unit| unit.identity.2 as usize).sum()
}

#[cfg(test)]
mod tests {
    #[test]
    fn staging_boundaries_flush_without_waiting_for_stream_idle() {
        let source = include_str!("waves.rs");
        let stage = source.split("    pub(crate) fn stage(").nth(1).unwrap().split("\nfn staged_rows_after").next().unwrap();
        assert!(!stage.contains("stream_idle") && !stage.contains("stream_query"));
        assert!(stage.contains("if singleton {\n            tracing::debug!"));
        assert!(stage.contains("unit.layer != arrival.layer"));
        assert!(stage.contains("if staged_rows + rows > self.rows { self.flush("));
        assert!(stage.contains("self.staged.len() == self.copied.len().min(MAX_MERGED_UNITS as usize)"));
        assert!(stage.contains("staged_rows >= self.rows"));
        assert!(stage.contains("if singleton || full { self.flush("));
    }

    #[test]
    fn staged_input_copy_is_host_triggered_and_compute_waits_on_recorded_event() {
        let source = include_str!("waves.rs");
        let stage = source.split("    pub(crate) fn stage(").nth(1).unwrap().split("\nfn staged_rows_after").next().unwrap();
        assert!(stage.find("copy_d2d_async(resident").unwrap() < stage.find("cuda_event_record(self.copied").unwrap());
        assert!(!stage.contains("cuda_stream_wait_event"));
        let flush = source.split("    pub(crate) fn flush(").nth(1).unwrap().split("    pub(crate) fn stage(").next().unwrap();
        assert!(flush.find("cuda_stream_wait_event").unwrap() < flush.find("launch(&wave)").unwrap());
        assert!(flush.contains("first_unit.input"));
        assert!(stage.contains("(resident.ptr as usize)"));
    }
}
