//! Manifest-sized row-addressed native expert programs on one owner stream.
use super::*;
use cuteafd_ffi::programs::{BoundProgram, Scalar};
use std::collections::BTreeMap;

const POINTERS: [&str; 15] = ["input_rows", "w13", "s13", "w2", "s2", "ids", "routing",
    "meta", "pair_row", "pair_pos", "act", "grouped_routing", "grouped_route", "partial", "output_rows"];

pub(super) struct NativeSegments<'a> {
    programs: BTreeMap<usize, (BoundProgram<'a>, Vec<(String, u64, u64)>)>,
    arena: DeviceAllocation<'a>,
}

impl<'a> NativeSegments<'a> {
    pub(super) fn load(library: &'a NativeLibrary, path: &std::path::Path, widths: impl Iterator<Item = usize>) -> Result<Self> {
        let programs = library.programs()?.with_manifest(path)?;
        let geometry = cuteafd_core::expert_geometry();
        let family = geometry.family().context("unknown native expert geometry")?;
        let mut loaded = BTreeMap::new();
        let mut bytes = 0;
        for width in widths {
            if loaded.contains_key(&width) { continue; }
            let name = format!("native_experts_{family}_w{width}_segments");
            let spec = programs.spec(&name)?;
            let (layout, size) = cuteafd_loader::serving_capacity::spark_waves::native_segment_layout(spec, geometry, width)?;
            bytes = bytes.max(size);
            loaded.insert(width, (programs.bound(&name, &POINTERS)?, layout));
        }
        Ok(Self { programs: loaded, arena: DeviceAllocation::new(library, bytes.try_into()?,
            crate::shared::memory::Tag::workspace("experts/workspace/native_segments"))? })
    }

    pub(super) fn launch(&self, width: usize, weights: [*mut c_void; 4], wave: &waves::Wave<'_>, stream: *mut c_void) -> Result<()> {
        let (program, layout) = self.programs.get(&width).context("native segment width not admitted")?;
        let mut pointers = vec![wave.input.ptr];
        pointers.extend(weights);
        pointers.extend([wave.ids.ptr, wave.weights.ptr]);
        pointers.extend(layout.iter().map(|(_, offset, _)| self.arena.buffer.ptr.cast::<u8>().wrapping_add(*offset as usize).cast()));
        pointers.push(wave.output.ptr);
        // SAFETY: the owner retains its manifest-sized arena, weights and every wave view through all member fences.
        unsafe { program.launch(&pointers, &[Scalar::I32(wave.rows.try_into()?)], stream) }
    }
}


pub(super) struct NativeWaveLaunch<'s, 'a> {
    pub kernel: &'s V41ExpertKernel<'a>,
    pub slots: &'s [*mut c_void; 44],
    pub scratch: &'s DeviceAllocation<'a>,
    pub decode: &'s Option<DecodeExecution<'a>>,
    pub small: &'s Option<DecodeExecution<'a>>,
    pub reduction: &'s Option<super::super::ExpertReductionPlan>,
    pub reducer: &'s V41CompactReducer<'a>,
    pub segments: &'s Option<NativeSegments<'a>>,
    pub timing: &'s Option<timing::WorkerTiming<'a>>,
    pub timing_units: &'s mut std::collections::VecDeque<bool>,
    pub stream: *mut c_void,
    pub width: usize,
    pub weights: [*mut c_void; 4],
}

impl NativeWaveLaunch<'_, '_> {
    pub(super) fn launch(&mut self, wave: &waves::Wave<'_>) -> Result<()> {
        // SAFETY: the owner retains all staged views, weights, scratch and reply frames through the member fences.
        unsafe {
            if let Some(timing) = self.timing { timing.record(0, self.stream)?; }
            if wave.segmented {
                self.segments.as_ref().context("native merged program not admitted")?
                    .launch(self.width, self.weights, wave, self.stream)?;
            } else {
                let (kernel, slots, _) = ExpertExecution::select_state(wave.rows.try_into()?, self.kernel,
                    self.slots, self.scratch, self.decode, self.small, self.reduction)?;
                let mut pointers = *slots;
                pointers[0] = wave.input.ptr;
                pointers[1] = wave.ids.ptr;
                pointers[2] = wave.weights.ptr;
                kernel.launch(&V41ExpertLaunchArgs::new(kernel.info(), pointers, wave.rows as u32, self.stream)?)?;
                match kernel.output_kind() {
                    cuteafd_ffi::V41ExpertOutputKind::Fp32Tokens =>
                        self.reducer.compact_tokens(slots[41].cast(), wave.output.ptr.cast(), wave.rows as u32, self.stream)?,
                    cuteafd_ffi::V41ExpertOutputKind::Bf16Routes =>
                        self.reducer.compact_bf16_routes(slots[41].cast(), wave.output.ptr.cast(), wave.rows as u32, self.stream)?,
                    cuteafd_ffi::V41ExpertOutputKind::Fp32Routes =>
                        self.reducer.compact(slots[41].cast(), wave.output.ptr.cast(), wave.rows as u32, self.stream)?,
                }
            }
            if let Some(timing) = self.timing { timing.record(1, self.stream)?; }
        }
        self.timing_units.extend((0..wave.members.len()).map(|index| index == 0));
        Ok(())
    }
}


pub(super) fn same_recipe(left: Option<(u32, cuteafd_ffi::V41ExpertReduction)>,
    right: Option<(u32, cuteafd_ffi::V41ExpertReduction)>) -> bool {
    left.is_some() && left == right
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn merged_members_require_equal_capacity_and_full_recipe() {
        let recipe = cuteafd_ffi::V41ExpertReduction { implementation: 2, plane_fold: 3,
            slice_width: 128, plane_count: 1, ..Default::default() };
        assert!(same_recipe(Some((4096, recipe)), Some((4096, recipe))));
        assert!(!same_recipe(Some((4096, recipe)), Some((1024, recipe))));
        for other in [cuteafd_ffi::V41ExpertReduction { slice_width: 64, ..recipe },
            cuteafd_ffi::V41ExpertReduction { plane_fold: 1, ..recipe },
            cuteafd_ffi::V41ExpertReduction { plane_count: 2, ..recipe }] {
            assert!(!same_recipe(Some((4096, recipe)), Some((4096, other))));
        }
        assert!(!same_recipe(None, None));
    }
}
