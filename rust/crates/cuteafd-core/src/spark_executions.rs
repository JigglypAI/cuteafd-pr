//! Typed Spark compute owners: one small decode execution per decode lane and
//! exactly one prefill execution per Spark. Planner, serve and golden build
//! the same value through [`SparkExecutions::admit`]; the worker allocates each
//! owner once and validates every request against its owner before launch.
use serde::{Deserialize, Serialize};

/// Most decode lanes a family runs (each owns one decode execution).
pub const MAX_SPARK_DECODE_LANES: usize = 2;
/// Most prefill units retained in flight on the single prefill owner.
pub const MAX_BUFFERED_PREFILL_UNITS: u8 = 4;

/// One independently scheduled decode lane. Capacities come from the family's
/// admitted decode extent and its covering package export, never prefill rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecodeExecution {
    pub capacity: u32,
    pub compiled_capacity: u32,
}

/// Exactly one scratch/stream owner per Spark. Network units retain independent
/// request/reply storage but enqueue on this stream without awaiting earlier replies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrefillExecution {
    /// Largest unit this owner can run. Static admission uses `unit_rows`;
    /// a future dynamic repartition must explicitly admit a larger extent.
    pub capacity: u32,
    /// Family package's covering export for that admitted extent.
    pub compiled_capacity: u32,
    /// Rows of one selected unit (the reserve divided by k). Per-unit input
    /// landing and reply staging are sized here, and prefill requests are
    /// bounded here.
    pub unit_rows: u32,
    /// Retained in-flight I/O units (k); one stream/scratch owner. Eligible
    /// segmented recipes size that arena at the admitted sum, not k arenas.
    pub buffered_units: u8,
}

/// Typed compute owners, independent of persistent endpoints and network rings.
/// The daemon allocates each decode entry once and the singleton prefill owner
/// once. Per-unit completion events retain all mapped I/O until its own reply.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SparkExecutions {
    /// Vector index is the request's execution lane for Decode and MtpVerify.
    pub decode: Vec<DecodeExecution>,
    /// Prefill and Benchmark rows use this owner, irrespective of endpoint lane.
    pub prefill: PrefillExecution,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionOwner { Decode(usize), Prefill }

/// One owner's extents for workspace accounting: singleton scratch at `compiled`,
/// stream-ordered device I/O at `capacity`, and pinned route staging of
/// `staging_rows` retained for each of `buffered_units` queued units.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OwnerExtent {
    pub owner: ExecutionOwner,
    pub capacity: u32,
    pub compiled: u32,
    pub staging_rows: u32,
    pub buffered_units: u8,
}

/// Units one merged Spark wave launches together. Staging may retain every
/// buffered unit, but one row-addressed launch holds at most this many.
pub const MAX_MERGED_UNITS: u32 = 2;

/// Code extent of a row-addressed (merged) program: two full 4096-row units.
/// Allocations use only the admitted merged rows, not this compile-time bound.
pub const SEGMENT_MAX_ROWS: u32 = MAX_MERGED_UNITS * 4096;

impl OwnerExtent {
    /// Rows of the widest merged wave (at most [`MAX_MERGED_UNITS`] units):
    /// the extent of segmented scratch and of the row-address tables.
    pub fn segment_rows(self) -> Option<u32> {
        match self.owner {
            ExecutionOwner::Prefill if self.buffered_units > 1 =>
                self.staging_rows.checked_mul(u32::from(self.buffered_units).min(MAX_MERGED_UNITS))
                    .filter(|&rows| rows <= SEGMENT_MAX_ROWS),
            _ => None,
        }
    }

    /// Rows of every retained staged unit: the extent of the per-slot device
    /// input slices (a unit can stage while an earlier pair is in flight).
    pub fn staged_rows(self) -> Option<u32> {
        self.segment_rows()?;
        self.staging_rows.checked_mul(u32::from(self.buffered_units))
    }
}

/// Device inputs for a segmented owner: per-slot staged rows of `row_bytes`
/// (the ingress row: FP8 K32, BF16, or a packed wire), sharing the ordinary
/// plane of `capacity` rows. A separate fallback plane (`sibling_row_bytes`,
/// a BF16-input sibling package) stays disjoint: an asynchronous staged copy
/// must never overwrite its pending reads.
pub fn segment_input_bytes(capacity: u32, staged_rows: u32, row_bytes: u32, sibling_row_bytes: Option<u32>) -> Option<u64> {
    let staged = u64::from(staged_rows).checked_mul(u64::from(row_bytes))?;
    match sibling_row_bytes {
        Some(sibling) => staged.checked_add(u64::from(capacity).checked_mul(u64::from(sibling))?),
        None => Some(staged.max(u64::from(capacity).checked_mul(u64::from(row_bytes))?)),
    }
}

/// The streamed arithmetic of a row-addressed program, which fixes its
/// intermediate scratch regions (installed package metadata).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentArithmetic {
    /// FP8/MXFP4/MXFP8 wire stream: one chunk-padded activation row of
    /// `act_row_bytes` (BF16 or MXFP8 down input) per routed pair.
    Stream { act_row_bytes: u32 },
    /// Calibrated NVFP4 W4A4 (fused gate/up): packed E2M1 input and act rows
    /// with their E4M3 block scales, `slice` the stored intermediate width.
    Nvfp4A4 { slice: u32 },
}

/// Exact streamed intermediate arena with the compiled tile-table offset.
/// This does not include row descriptors, wire frames or singleton scratch.
pub fn segment_scratch_bytes(hidden: u32, experts: u32, topk: u32, arithmetic: SegmentArithmetic,
    rows: u32, maximum: u32) -> Option<u64> {
    let width = match arithmetic { SegmentArithmetic::Stream { act_row_bytes } => act_row_bytes,
        SegmentArithmetic::Nvfp4A4 { slice } => slice };
    if hidden == 0 || experts == 0 || topk == 0 || width == 0
        || rows == 0 || rows > maximum || maximum > SEGMENT_MAX_ROWS { return None; }
    let align = |n: u64| n.checked_add(1023).map(|n| n / 1024 * 1024);
    let tiles = |p: u64| p.div_ceil(128) + u64::from(experts).min(p);
    let pairs = u64::from(rows) * u64::from(topk);
    let bound = u64::from(maximum) * u64::from(topk);
    let padded = tiles(pairs).checked_mul(128)?;
    let meta = align((4 + 3 * u64::from(experts) + tiles(bound)).checked_mul(4)?)?;
    let routes = align(pairs.checked_mul(4)?)?.checked_mul(2)?;
    let middle = match arithmetic {
        SegmentArithmetic::Stream { act_row_bytes } => align(padded.checked_mul(u64::from(act_row_bytes))?)?,
        SegmentArithmetic::Nvfp4A4 { slice } => {
            let (h, i) = (u64::from(hidden), u64::from(slice));
            align(padded.checked_mul(h / 2)?)?.checked_add(align(padded.checked_mul(h / 16)?)?)?
                .checked_add(align(padded.checked_mul(i / 2)?)?)?.checked_add(align(padded.checked_mul(i / 16)?)?)?
        }
    };
    let output = align(pairs.checked_mul(u64::from(hidden))?.checked_mul(2)?)?;
    meta.checked_add(routes)?.checked_add(middle)?.checked_add(output)
}

/// Bytes one owner holds at its extents, from the planner's census or the
/// worker's installed artifacts.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OwnerBytes {
    pub scratch: u64,
    pub io: u64,
    /// Per queued unit; multiplied by `buffered_units` only.
    pub pinned_staging: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ExecutionAdmissionError {
    #[error("Spark decode lane count {lanes} must be 1..={maximum}")]
    DecodeLanes { lanes: usize, maximum: usize },
    #[error("{owner} live capacity {capacity} requires a covering export, got {compiled}")]
    Capacity { owner: &'static str, capacity: u32, compiled: u32 },
    #[error("{owner} capacity {rows} has no covering export in the package ladder {exports:?}")]
    Uncovered { owner: &'static str, rows: u32, exports: Vec<u32> },
    #[error("Spark prefill unit rows {unit_rows} must be 1..={capacity}")]
    UnitRows { unit_rows: u32, capacity: u32 },
    #[error("Spark prefill buffered units {units} must be 1..={maximum}")]
    BufferedUnits { units: u8, maximum: u8 },
    #[error("Spark decode lane {lane} is outside {lanes} admitted lanes")]
    DecodeLane { lane: u32, lanes: usize },
    #[error("{owner} rows {rows} exceed admitted capacity {capacity}, or are empty")]
    Rows { owner: &'static str, rows: u32, capacity: u32 },
}

/// The smallest export in `exports` that covers `rows` live rows.
pub fn covering_export(exports: &[u32], rows: u32) -> Option<u32> {
    exports.iter().copied().filter(|&export| export >= rows).min()
}

impl SparkExecutions {
    /// The one admission constructor: `decode_rows` per decode lane, the
    /// admitted prefill extent (`prefill_rows`), the selected unit and the
    /// retained unit count, with every compiled capacity taken from
    /// the package's export ladder. An uncovered capacity is a typed refusal,
    /// never a silently larger or smaller owner.
    pub fn admit(decode_rows: &[u32], prefill_rows: u32, unit_rows: u32, buffered_units: u8, exports: &[u32])
        -> Result<Self, ExecutionAdmissionError> {
        let cover = |owner, rows| covering_export(exports, rows).filter(|_| rows > 0)
            .ok_or_else(|| ExecutionAdmissionError::Uncovered { owner, rows, exports: exports.to_vec() });
        let decode = decode_rows.iter().map(|&rows| Ok(DecodeExecution { capacity: rows,
            compiled_capacity: cover("decode", rows)? })).collect::<Result<Vec<_>, _>>()?;
        let plan = Self { decode, prefill: PrefillExecution { capacity: prefill_rows,
            compiled_capacity: cover("prefill", prefill_rows)?, unit_rows, buffered_units } };
        plan.validate()?;
        Ok(plan)
    }

    pub fn validate(&self) -> Result<(), ExecutionAdmissionError> {
        if self.decode.is_empty() || self.decode.len() > MAX_SPARK_DECODE_LANES {
            return Err(ExecutionAdmissionError::DecodeLanes { lanes: self.decode.len(), maximum: MAX_SPARK_DECODE_LANES });
        }
        let check = |owner, capacity, compiled| {
            if capacity == 0 || compiled < capacity || compiled > 4096 {
                Err(ExecutionAdmissionError::Capacity { owner, capacity, compiled })
            } else { Ok(()) }
        };
        for lane in &self.decode { check("decode", lane.capacity, lane.compiled_capacity)?; }
        check("prefill", self.prefill.capacity, self.prefill.compiled_capacity)?;
        if !(1..=self.prefill.capacity).contains(&self.prefill.unit_rows) {
            return Err(ExecutionAdmissionError::UnitRows { unit_rows: self.prefill.unit_rows,
                capacity: self.prefill.capacity });
        }
        if !(1..=MAX_BUFFERED_PREFILL_UNITS).contains(&self.prefill.buffered_units) {
            return Err(ExecutionAdmissionError::BufferedUnits { units: self.prefill.buffered_units,
                maximum: MAX_BUFFERED_PREFILL_UNITS });
        }
        Ok(())
    }

    /// Every owner's extents: each decode owner holds one unit of its own
    /// capacity; the prefill owner holds `buffered_units` units of `unit_rows`.
    pub fn owners(&self) -> impl Iterator<Item = OwnerExtent> + '_ {
        self.decode.iter().enumerate().map(|(lane, owner)| OwnerExtent { owner: ExecutionOwner::Decode(lane),
            capacity: owner.capacity, compiled: owner.compiled_capacity, staging_rows: owner.capacity, buffered_units: 1 })
            .chain(std::iter::once(OwnerExtent { owner: ExecutionOwner::Prefill, capacity: self.prefill.capacity,
                compiled: self.prefill.compiled_capacity, staging_rows: self.prefill.unit_rows,
                buffered_units: self.prefill.buffered_units }))
    }

    /// The one workspace sum shared by planner and worker: scratch and device
    /// I/O once per owner, pinned staging times its queued units. `None` on
    /// overflow; `bytes` errors propagate.
    pub fn workspace_bytes<E>(&self, mut bytes: impl FnMut(OwnerExtent) -> Result<OwnerBytes, E>)
        -> Result<Option<crate::expert_geometry::SparkWorkspaceBytes>, E> {
        let owners = self.owners().map(|extent| bytes(extent).map(|owner| crate::expert_geometry::SparkOwnerWorkspace {
            scratch_bytes: owner.scratch, io_bytes: owner.io, pinned_staging_bytes: owner.pinned_staging,
            buffered_units: extent.buffered_units })).collect::<Result<Vec<_>, E>>()?;
        Ok(crate::expert_geometry::spark_owner_workspace_bytes(owners))
    }

    /// Largest live rows any owner admits.
    pub fn capacity(&self) -> u32 {
        self.decode.iter().map(|lane| lane.capacity).fold(self.prefill.capacity, u32::max)
    }

    /// Largest compiled export any owner loads.
    pub fn compiled_capacity(&self) -> u32 {
        self.decode.iter().map(|lane| lane.compiled_capacity).fold(self.prefill.compiled_capacity, u32::max)
    }

    /// Validate a request against its typed owner before any GPU launch:
    /// `decode_lane` is `Some` for Decode/MtpVerify rows and `None` for
    /// Prefill/Benchmark rows. An endpoint lane never selects prefill scratch.
    pub fn owner(&self, decode_lane: Option<u32>, rows: u32) -> Result<ExecutionOwner, ExecutionAdmissionError> {
        self.validate()?;
        let (owner, capacity, name) = match decode_lane {
            None => (ExecutionOwner::Prefill, self.prefill.unit_rows, "prefill"),
            Some(lane) => {
                let entry = self.decode.get(lane as usize)
                    .ok_or(ExecutionAdmissionError::DecodeLane { lane, lanes: self.decode.len() })?;
                (ExecutionOwner::Decode(lane as usize), entry.capacity, "decode")
            }
        };
        if rows == 0 || rows > capacity {
            return Err(ExecutionAdmissionError::Rows { owner: name, rows, capacity });
        }
        Ok(owner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LADDER: [u32; 8] = crate::EXL3_CAPACITIES;

    #[test]
    fn segmented_allocation_uses_live_sum_and_keeps_decode_unchanged() {
        let plan = SparkExecutions::admit(&[64], 4096, 2752, 3, &LADDER).unwrap();
        let owners = plan.owners().collect::<Vec<_>>();
        assert_eq!(owners[0].segment_rows(), None);
        // One merged launch holds two units; all three stage.
        assert_eq!(owners[1].segment_rows(), Some(5504));
        assert_eq!(owners[1].staged_rows(), Some(8256));
        assert_eq!(owners[0].staged_rows(), None);
        let stream = SegmentArithmetic::Stream { act_row_bytes: 768 };
        let size = |rows| segment_scratch_bytes(4096, 256, 8, stream, rows, SEGMENT_MAX_ROWS).unwrap();
        assert!(size(5504) < size(SEGMENT_MAX_ROWS));
        assert_eq!(size(5504), 420_051_968);
        assert!(segment_scratch_bytes(4096, 256, 8, stream, SEGMENT_MAX_ROWS + 1, SEGMENT_MAX_ROWS).is_none());
        assert!(segment_scratch_bytes(4096, 256, 8, stream, 0, SEGMENT_MAX_ROWS).is_none());
        assert!(segment_scratch_bytes(4096, 256, 8, stream, 4096, 16384).is_none(), "code bound above 2 units");
    }

    #[test]
    fn nvfp4_a4_segment_scratch_matches_the_exporter_region_list() {
        // MiniMax TP3 (H 6144, 128 experts, top-4, slice 1024) at the 2 x 2752
        // merged wave: the fork's `_regions(stream, a4, fused)` with the
        // compiled 8192-row tile table (test_cuteafd_expert_dispatch_policy).
        let (h, e, k, i, rows) = (6144u64, 128u64, 4u64, 1024u64, 5504u64);
        let align = |n: u64| n.div_ceil(1024) * 1024;
        let pairs = rows * k;
        let padded = (pairs.div_ceil(128) + e.min(pairs)) * 128;
        let expected = align(4 * (4 + 3 * e + 8192 * k / 128 + e)) + 2 * align(pairs * 4)
            + align(padded * h / 2) + align(padded * h / 16) + align(padded * i / 2) + align(padded * i / 16)
            + align(pairs * h * 2);
        let a4 = SegmentArithmetic::Nvfp4A4 { slice: 1024 };
        assert_eq!(segment_scratch_bytes(6144, 128, 4, a4, 5504, SEGMENT_MAX_ROWS), Some(expected));
        assert!(segment_scratch_bytes(6144, 128, 4, a4, 5504, SEGMENT_MAX_ROWS).unwrap()
            < segment_scratch_bytes(6144, 128, 4, a4, 8192, SEGMENT_MAX_ROWS).unwrap());
    }

    #[test]
    fn admission_takes_covering_exports_from_the_package_ladder() {
        let plan = SparkExecutions::admit(&[64, 80], 4096, 2752, 3, &LADDER).unwrap();
        assert_eq!(plan.decode, [DecodeExecution { capacity: 64, compiled_capacity: 80 },
            DecodeExecution { capacity: 80, compiled_capacity: 80 }]);
        assert_eq!(plan.prefill, PrefillExecution { capacity: 4096, compiled_capacity: 4096,
            unit_rows: 2752, buffered_units: 3 });
        assert_eq!((plan.capacity(), plan.compiled_capacity()), (4096, 4096));
        let encoded = serde_json::to_vec(&plan).unwrap();
        assert_eq!(serde_json::from_slice::<SparkExecutions>(&encoded).unwrap(), plan);
        // Native packages export only m1/m80/main.
        let native = SparkExecutions::admit(&[64], 2752, 2752, 1, &[1, 80, 4096]).unwrap();
        assert_eq!((native.decode[0].compiled_capacity, native.prefill.compiled_capacity), (80, 4096));
    }

    #[test]
    fn uncovered_decode_or_prefill_capacity_is_a_typed_refusal() {
        assert_eq!(SparkExecutions::admit(&[128], 4096, 4096, 1, &[1, 80, 4096]).unwrap().decode[0].compiled_capacity, 4096);
        assert!(matches!(SparkExecutions::admit(&[96], 1024, 1024, 1, &[1, 80, 1024]).unwrap().decode[0],
            DecodeExecution { compiled_capacity: 1024, .. }));
        assert_eq!(SparkExecutions::admit(&[81], 64, 64, 1, &[1, 64, 80]).unwrap_err(),
            ExecutionAdmissionError::Uncovered { owner: "decode", rows: 81, exports: vec![1, 64, 80] });
        assert!(matches!(SparkExecutions::admit(&[64], 4097, 4097, 1, &LADDER),
            Err(ExecutionAdmissionError::Uncovered { owner: "prefill", rows: 4097, .. })));
        assert!(matches!(SparkExecutions::admit(&[0], 4096, 4096, 1, &LADDER),
            Err(ExecutionAdmissionError::Uncovered { owner: "decode", rows: 0, .. })));
        assert!(matches!(SparkExecutions::admit(&[], 4096, 4096, 1, &LADDER), Err(ExecutionAdmissionError::DecodeLanes { .. })));
        assert!(matches!(SparkExecutions::admit(&[64; 3], 4096, 4096, 1, &LADDER), Err(ExecutionAdmissionError::DecodeLanes { .. })));
        for units in [0, 5] {
            assert!(matches!(SparkExecutions::admit(&[64], 4096, 2048, units, &LADDER),
                Err(ExecutionAdmissionError::BufferedUnits { .. })));
        }
        assert_eq!(SparkExecutions::admit(&[64], 2048, 2752, 3, &LADDER).unwrap_err(),
            ExecutionAdmissionError::UnitRows { unit_rows: 2752, capacity: 2048 });
    }

    #[test]
    fn workspace_charges_one_prefill_scratch_and_divides_the_reserve_into_units() {
        let bytes = |plan: &SparkExecutions| plan.workspace_bytes(|extent| Ok::<_, ()>(OwnerBytes {
            scratch: u64::from(extent.compiled) * 1000, io: u64::from(extent.capacity) * 10,
            pinned_staging: u64::from(extent.staging_rows) })).unwrap().unwrap();
        let mut totals = Vec::new();
        for (unit, units) in [(4096, 1), (4096, 2), (2752, 3), (2048, 4)] {
            let plan = SparkExecutions::admit(&[64, 64], 4096, unit, units, &LADDER).unwrap();
            let total = bytes(&plan);
            assert_eq!(total.device_bytes, 2 * (80_000 + 640) + 4_096_000 + 40_960);
            assert_eq!(total.pinned_bytes, 2 * 64 + u64::from(unit) * u64::from(units));
            totals.push(total.device_bytes);
        }
        // More waves never add scratch: the reserve is divided, not multiplied.
        assert!(totals.windows(2).all(|pair| pair[0] == pair[1]));
    }

    #[test]
    fn typed_owners_keep_decode_prefill_and_buffered_io_independent() {
        let plan = SparkExecutions { decode: vec![
            DecodeExecution { capacity: 80, compiled_capacity: 80 },
            DecodeExecution { capacity: 64, compiled_capacity: 256 }],
            prefill: PrefillExecution { capacity: 4096, compiled_capacity: 4096, unit_rows: 2752, buffered_units: 3 } };
        plan.validate().unwrap();
        assert_eq!(plan.owner(Some(0), 80).unwrap(), ExecutionOwner::Decode(0));
        assert_eq!(plan.owner(Some(1), 64).unwrap(), ExecutionOwner::Decode(1));
        assert_eq!(plan.owner(None, 2752).unwrap(), ExecutionOwner::Prefill);
        assert!(matches!(plan.owner(Some(1), 65), Err(ExecutionAdmissionError::Rows { .. })));
        assert!(matches!(plan.owner(Some(2), 1), Err(ExecutionAdmissionError::DecodeLane { .. })));
        assert!(matches!(plan.owner(None, 2753), Err(ExecutionAdmissionError::Rows { capacity: 2752, .. })));
        assert!(plan.owner(None, 0).is_err());
        let mut invalid = plan.clone(); invalid.decode[0].compiled_capacity = 64;
        assert!(matches!(invalid.validate(), Err(ExecutionAdmissionError::Capacity { owner: "decode", .. })));
        invalid = plan.clone(); invalid.prefill.compiled_capacity = 2048;
        assert!(matches!(invalid.validate(), Err(ExecutionAdmissionError::Capacity { owner: "prefill", .. })));
        invalid = plan; invalid.decode.clear();
        assert!(matches!(invalid.validate(), Err(ExecutionAdmissionError::DecodeLanes { .. })));
    }
}

#[cfg(test)]
mod segment_input_tests {
    use super::*;
    #[test]
    fn input_planes_charge_staged_ingress_rows_and_keep_bf16_readers_disjoint() {
        for (capacity, unit, count) in [(4096, 2752, 3), (4096, 769, 3), (1024, 1024, 2)] {
            let rows = unit * count;
            // FP8 K32 wire rows, BF16 rows (NVFP4 A4) and a packed NVFP4 wire row.
            for row_bytes in [4224u32, 8192, 4096 * 9 / 16 + 16] {
                let staged = u64::from(rows) * u64::from(row_bytes);
                assert_eq!(segment_input_bytes(capacity, rows, row_bytes, None),
                    Some(staged.max(u64::from(capacity) * u64::from(row_bytes))));
                let sibling = u64::from(capacity) * 8192;
                assert_eq!(segment_input_bytes(capacity, rows, row_bytes, Some(8192)), Some(sibling + staged));
                assert!(sibling + staged > sibling.max(staged), "aliasing the BF16 plane would undercharge");
            }
        }
    }
}

/// Capacity contract declared by the installed expert artifact. No family or
/// canonical row ladder participates in native decode-state admission.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExpertExportCapacities {
    pub capacities: Vec<u32>,
    pub decode: Vec<u32>,
}
impl ExpertExportCapacities {
    pub fn from_manifest(manifest: &serde_json::Value) -> Result<Self, String> {
        let variants = manifest["variants"].as_array().ok_or("missing expert variants")?;
        let capacities = variants.iter().map(|variant| {
            variant.get("requested_rows").unwrap_or(&variant["capacity_rows"])
                .as_u64().and_then(|n| u32::try_from(n).ok())
                .filter(|&n| n > 0 && n <= 4096).ok_or("invalid expert export capacity")
        }).collect::<Result<Vec<_>, _>>()?;
        let decode = manifest["decode_capacities"].as_array().ok_or("missing expert decode_capacities")?
            .iter().map(|v| v.as_u64().and_then(|n| u32::try_from(n).ok())
                .ok_or("invalid expert decode capacity")).collect::<Result<Vec<_>, _>>()?;
        let mut capacities = capacities;
        capacities.sort_unstable();
        let exports = Self { capacities, decode };
        exports.validate()?;
        Ok(exports)
    }

    /// Revalidate contracts received through the installed worker report.
    pub fn validate(&self) -> Result<(), String> {
        if self.capacities.is_empty()
            || self.capacities.iter().any(|&n| n == 0 || n > 4096)
            || !self.capacities.windows(2).all(|p| p[0] < p[1])
            || self.decode.first() != Some(&1) || !self.decode.windows(2).all(|p| p[0] < p[1])
            || self.decode.iter().any(|n| !self.capacities.contains(n)) {
            return Err("expert decode capacities must be ordered, unique compiled exports starting at m1".into());
        }
        Ok(())
    }

    pub fn covering(&self, rows: usize) -> Result<Vec<u32>, String> {
        let compiled = self.capacities.iter().copied().find(|&n| n as usize >= rows && rows > 0)
            .ok_or_else(|| format!("expert package does not cover {rows} rows"))?;
        Ok(self.capacities.iter().copied().filter(|&n| n <= compiled).collect())
    }

    pub fn grouped_decode(&self, compiled: u32) -> Result<u32, String> {
        if !self.capacities.contains(&compiled) { return Err(format!("missing compiled expert m{compiled}")); }
        self.decode.iter().copied().filter(|&n| n <= compiled).next_back()
            .ok_or_else(|| "expert package lacks admitted decode cover".into())
    }
}

#[cfg(test)]
mod expert_capacity_tests {
    use super::ExpertExportCapacities;
    #[test]
    fn shared_recipe_matches_target_and_draft_export_contracts() {
        for family in ["v41", "dsv4f", "dsv4p"] {
            for role in ["spark", "spark_tp2", "spark_tp3", "spark_tp6", "spark_tp8", "spark_w320", "rtx_backbone", "rtx_tp2", "coordinator", "dspark_tp2"] {
                let recipe = crate::mxfp4_layouts::native_export_capacities(family, role);
                let value = serde_json::json!({"decode_capacities":recipe.decode,
                    "variants":recipe.capacities.iter().map(|n| serde_json::json!({"capacity_rows":n})).collect::<Vec<_>>()});
                assert_eq!(ExpertExportCapacities::from_manifest(&value).unwrap(), recipe);
                let expected = if family == "v41" && !matches!(role, "coordinator" | "dspark_tp2") { crate::V41_VERIFY_ROWS as u32 } else { 80 };
                assert_eq!(recipe.grouped_decode(4096).unwrap(), expected);
            }
        }
    }

    #[test]
    fn received_capacity_contracts_require_ordered_declared_exports() {
        let exports = ExpertExportCapacities { capacities: vec![1,16,64,4096], decode: vec![1,16,64] };
        let encoded = serde_json::to_value(&exports).unwrap();
        let restored: ExpertExportCapacities = serde_json::from_value(encoded.clone()).unwrap();
        assert_eq!(restored, exports);
        restored.validate().unwrap();
        for capacities in [vec![], vec![0,1,64], vec![1,64,16], vec![1,16,16,64], vec![1,4097]] {
            let invalid = ExpertExportCapacities { capacities, ..exports.clone() };
            assert!(invalid.validate().is_err());
        }
        for decode in [vec![], vec![16,64], vec![1,64,16], vec![1,16,16], vec![1,80]] {
            let invalid = ExpertExportCapacities { decode, ..exports.clone() };
            assert!(invalid.validate().is_err());
        }
    }

    #[test]
    fn manifests_admit_their_own_decode_ladders() {
        for ceiling in [64, 80] {
            let value = serde_json::json!({"decode_capacities":[1,16,ceiling],
                "variants":[{"capacity_rows":1},{"capacity_rows":16},
                    {"capacity_rows":ceiling},{"capacity_rows":4096}]});
            let exports = ExpertExportCapacities::from_manifest(&value).unwrap();
            assert_eq!(exports.grouped_decode(4096).unwrap(), ceiling);
            assert_eq!(exports.covering(17).unwrap(), [1,16,ceiling]);
            let mut invalid = value;
            invalid["decode_capacities"] = serde_json::json!([1,16,128]);
            assert!(ExpertExportCapacities::from_manifest(&invalid).is_err());
        }
    }
}
