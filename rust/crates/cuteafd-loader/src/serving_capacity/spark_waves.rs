//! One manifest-backed merged-wave census for planner and worker admission.
use anyhow::{ensure, Context, Result};
use cuteafd_core::{ExpertGeometry, OwnerExtent};
use cuteafd_ffi::programs::ProgramSpec;

pub const NATIVE_PLANES: [&str; 7] = ["meta", "pair_row", "pair_pos", "act", "grouped_routing", "grouped_route", "partial"];

pub fn native_segment_layout(spec: &ProgramSpec, geometry: ExpertGeometry, width: usize)
    -> Result<(Vec<(String, u64, u64)>, u64)> {
    ensure!(spec.capacity_rows == 8192 && spec.geometry["row_addresses"] == true
        && spec.geometry["hidden"].as_u64() == Some(u64::from(geometry.hidden))
        && spec.geometry["intermediate"].as_u64() == Some(width as u64)
        && spec.geometry["experts"].as_u64() == Some(u64::from(geometry.experts))
        && spec.geometry["topk"].as_u64() == Some(u64::from(geometry.topk)), "native segment geometry mismatch");
    cuteafd_core::spark_executions::wave_scratch_layout(NATIVE_PLANES.iter().map(|&name|
        spec.scratch.get(name).copied().map(|bytes| (name, bytes))).collect::<Option<Vec<_>>>()
            .context("native segment scratch plane missing")?).context("native segment scratch layout overflow")
}

/// Plane/table/descriptor bytes do not depend on backend; scratch is the
/// maximum native width arena, or the one EXLR8 manifest-sized arena.
pub fn staging(extent: OwnerExtent, geometry: ExpertGeometry) -> Result<cuteafd_core::spark_executions::WaveStagingLayout> {
    staging_for_input(extent, geometry, geometry.hidden * 33 / 32)
}

/// Spark native/EXLR8 staging has one wire contract. Validate the actual
/// installed input stride before allocating, rather than undercharging BF16.
pub fn staging_for_input(extent: OwnerExtent, geometry: ExpertGeometry, row_bytes: u32)
    -> Result<cuteafd_core::spark_executions::WaveStagingLayout> {
    if extent.segment_rows().is_none() { return Ok(Default::default()); }
    ensure!(row_bytes == geometry.hidden * 33 / 32,
        "Spark wave staging requires FP8 K32 ingress, not {row_bytes}-byte rows");
    cuteafd_core::spark_executions::wave_staging_layout(extent, row_bytes, geometry.topk,
        cuteafd_core::spark_executions::MAX_MERGED_UNITS).context("Spark wave staging extent overflow")
}

pub fn native_scratch(programs: &serde_json::Value, extent: OwnerExtent, geometry: ExpertGeometry,
    widths: impl IntoIterator<Item = usize>) -> Result<u64> {
    if extent.segment_rows().is_none() { return Ok(0); }
    let family = geometry.family().context("unknown native family")?;
    widths.into_iter().try_fold(0, |maximum, width| {
        let name = format!("native_experts_{family}_w{width}_segments");
        ensure!(programs["programs"].as_array().is_some_and(|entries|
            entries.iter().filter(|entry| entry["name"] == name).count() == 1),
            "missing {name} in PROGRAMS.json; republish placement inputs");
        let spec = crate::plan::exlr8::workspace::program_spec(programs, &name)?;
        Ok(maximum.max(native_segment_layout(&spec, geometry, width)?.1))
    })
}

pub fn exlr8_scratch(programs: &serde_json::Value, extent: OwnerExtent, geometry: ExpertGeometry,
    unit_limit: usize, activations: crate::plan::exlr8::PrefillActivations) -> Result<u64> {
    let Some(rows) = extent.segment_rows().filter(|_| extent.staging_rows > 64) else { return Ok(0); };
    let family = cuteafd_ffi::exlr8::KernelFamily::from_geometry(geometry.hidden as usize,
        geometry.intermediate as usize / 128, geometry.topk as usize, geometry.experts as usize)?;
    let layout = crate::plan::exlr8::execution_workspace_from_manifest(programs, family, rows as usize,
        crate::plan::exlr8::ExecutionVariant::Segments, unit_limit, activations)?;
    Ok(layout.bytes.try_into()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn staging_skips_single_flight_and_rejects_noncanonical_native_ingress() {
        let geometry = ExpertGeometry::DEEPSEEK_V41;
        let mut extent = OwnerExtent { owner: cuteafd_core::ExecutionOwner::Prefill,
            capacity: 2752, compiled: 4096, staging_rows: 2752, buffered_units: 3 };
        assert_eq!(staging(extent, geometry).unwrap(), staging_for_input(extent, geometry, 5280).unwrap());
        assert!(staging_for_input(extent, geometry, geometry.hidden * 2).unwrap_err().to_string().contains("FP8 K32 ingress"));
        extent.buffered_units = 1;
        assert_eq!(staging_for_input(extent, geometry, geometry.hidden * 2).unwrap(), Default::default());
    }

    #[test]
    fn missing_native_segments_name_the_export_and_republication() {
        let extent = OwnerExtent { owner: cuteafd_core::ExecutionOwner::Prefill, capacity: 2752,
            compiled: 4096, staging_rows: 2752, buffered_units: 3 };
        let error = native_scratch(&serde_json::json!({"programs":[]}), extent,
            ExpertGeometry::DEEPSEEK_V4_FLASH, [1024]).unwrap_err();
        assert_eq!(error.to_string(), "missing native_experts_dsv4f_w1024_segments in PROGRAMS.json; republish placement inputs");
    }

    #[test]
    fn native_widths_share_one_maximum_arena_and_planner_layout() -> Result<()> {
        let programs = serde_json::json!({"programs":crate::plan::testing::native_segment_programs()});
        let geometry = ExpertGeometry::DEEPSEEK_V41;
        let extent = OwnerExtent { owner: cuteafd_core::ExecutionOwner::Prefill, capacity: 2752,
            compiled: 4096, staging_rows: 2752, buffered_units: 3 };
        let first = native_scratch(&programs, extent, geometry, [448])?;
        let second = native_scratch(&programs, extent, geometry, [512])?;
        assert_eq!(native_scratch(&programs, extent, geometry, [448, 512, 448])?, first.max(second));
        assert_eq!(native_scratch(&programs, OwnerExtent { owner: cuteafd_core::ExecutionOwner::Decode(0), ..extent },
            geometry, [448,512])?, 0);
        Ok(())
    }
}

/// All retained EXLR8 execution owners, including the old K1 arenas. Spark
/// dispatch is SM121: compact decode plus grouped decode at its unit threshold.
pub fn exlr8_owner_bytes(programs: &serde_json::Value, extent: OwnerExtent, geometry: ExpertGeometry,
    units: usize, activations: crate::plan::exlr8::PrefillActivations) -> Result<cuteafd_core::OwnerBytes> {
    exlr8_owner_from_specs(|name| crate::plan::exlr8::workspace::program_spec(programs, name),
        extent, geometry, units, activations)
}

pub fn exlr8_owner_from_specs(mut spec: impl FnMut(&str) -> Result<ProgramSpec>, extent: OwnerExtent,
    geometry: ExpertGeometry, units: usize, activations: crate::plan::exlr8::PrefillActivations)
    -> Result<cuteafd_core::OwnerBytes> {
    use crate::plan::exlr8::{execution_workspace, residency::ExpertSet, ExecutionVariant, PrefillActivations};
    let family = cuteafd_ffi::exlr8::KernelFamily::from_geometry(geometry.hidden as usize,
        geometry.intermediate as usize / 128, geometry.topk as usize, geometry.experts as usize)?;
    let mut arena = |variant: ExecutionVariant, rows: usize, activations| -> Result<u64> {
        Ok(execution_workspace(&spec(variant.program(family, activations))?, rows,
            ExpertSet::Routed, variant, units, activations)?.bytes as u64)
    };
    let decode_rows = extent.capacity.min(64) as usize;
    let mut scratch = arena(ExecutionVariant::DecodeCompact, decode_rows, PrefillActivations::A16)?;
    if decode_rows >= if units <= 3 { 16 } else { 8 } {
        scratch += arena(ExecutionVariant::DecodeGrouped, decode_rows, PrefillActivations::A16)?;
    }
    if extent.capacity > 64 { scratch += arena(ExecutionVariant::Prefill, extent.capacity as usize, activations)?; }
    if let Some(rows) = extent.segment_rows().filter(|_| extent.staging_rows > 64) {
        scratch += arena(ExecutionVariant::Segments, rows as usize, activations)?;
    }
    let stage = staging(extent, geometry)?;
    Ok(cuteafd_core::OwnerBytes { scratch, io: stage.plane_bytes + stage.table_bytes,
        pinned_staging: u64::from(extent.staging_rows) * u64::from(geometry.topk) * 8 + stage.pinned_per_unit })
}

pub fn exlr8_worker_bytes(programs: &serde_json::Value, executions: &cuteafd_core::SparkExecutions,
    geometry: ExpertGeometry, units: usize, activations: crate::plan::exlr8::PrefillActivations) -> Result<u64> {
    let owners = executions.workspace_bytes(|extent| exlr8_owner_bytes(programs, extent, geometry, units, activations))?
        .context("EXLR8 worker owner overflow")?;
    owners.total().and_then(|bytes| bytes.checked_add(u64::from(executions.capacity()) * u64::from(geometry.hidden) * 33 / 32))
        .context("EXLR8 worker fallback staging overflow")
}
