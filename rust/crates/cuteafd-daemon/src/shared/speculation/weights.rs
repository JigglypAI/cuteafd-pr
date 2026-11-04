//! Header-only drafter sources and bounded dense load records, shared by block drafters.
use crate::shared::{experts::rtx::dense::{DenseLoad, Job, Operand, Rect, Step, STEP_BYTES},
    fp8_linear::{Fp8Scales, Fp8Weight}, memory::DeviceAllocation};
use anyhow::{ensure, Context, Result};
use cuteafd_core::DType;
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use cuteafd_loader::{read_safetensors_metadata, SafetensorsTensorMetadata};
use std::{collections::HashMap, ffi::c_void, path::{Path, PathBuf}};

pub(crate) struct Source {
    path: PathBuf,
    tensors: HashMap<String, SafetensorsTensorMetadata>,
    file_bytes: u64,
}

impl Source {
    pub fn open(snapshot: &Path) -> Result<Self> {
        let path = snapshot.join("model.safetensors");
        let tensors = read_safetensors_metadata(&path)?.into_iter().map(|t| (t.name.clone(), t)).collect();
        let file_bytes = std::fs::metadata(&path)?.len();
        Ok(Self { path, tensors, file_bytes })
    }

    pub fn rect(&self, name: &str, shape: &[usize]) -> Result<Rect> {
        let tensor = self.tensors.get(name).with_context(|| format!("drafter checkpoint has no {name}"))?;
        ensure!(tensor.dtype == DType::Bf16, "{name}: drafter weights must be BF16, found {:?}", tensor.dtype);
        let bytes = shape.iter().try_fold(2usize, |n, &d| n.checked_mul(d))
            .with_context(|| format!("{name}: BF16 tensor byte count overflow"))?;
        ensure!(tensor.shape == shape && tensor.byte_length == bytes as u64,
            "{name}: shape {:?}/{} bytes, expected BF16 {shape:?}/{bytes} bytes", tensor.shape, tensor.byte_length);
        ensure!(tensor.byte_offset.checked_add(tensor.byte_length).is_some_and(|end| end <= self.file_bytes),
            "{name} lies past the drafter checkpoint file");
        let rows = if shape.len() >= 2 { shape[0] } else { 1 };
        ensure!(rows > 0 && bytes > 0 && bytes % rows == 0, "{name}: invalid tensor row extent");
        Ok(Rect { name: name.into(), path: self.path.clone(), base: tensor.byte_offset,
            row_bytes: bytes / rows, rows: 0..rows, cols: 0..bytes / rows })
    }
}

/// One resident representation; FP8 conversion stages bounded source windows,
/// never a second, full-sized BF16 device owner.
pub(crate) enum Weight<'a> {
    Bf16(DeviceAllocation<'a>),
    Fp8(Fp8Weight<'a>),
}

pub(crate) struct Load<'a> {
    library: &'a NativeLibrary,
    source: Source,
    dense: DenseLoad<'a>,
    prefetched: Option<Vec<u8>>,
}

impl<'a> Load<'a> {
    pub fn open(library: &'a NativeLibrary, snapshot: &Path) -> Result<Self> {
        Ok(Self { library, source: Source::open(snapshot)?, dense: DenseLoad::mount(library), prefetched: None })
    }

    /// Preserve the existing overlapped mount read when no Recorder is active.
    pub fn with_prefetched(library: &'a NativeLibrary, snapshot: &Path, bytes: Vec<u8>) -> Result<Self> {
        let mut load = Self::open(library, snapshot)?;
        if !crate::shared::experts::rtx::dense::is_recording() { load.prefetched = Some(bytes); }
        Ok(load)
    }

    pub fn validate(&self, shapes: impl IntoIterator<Item = (String, Vec<usize>)>) -> Result<()> {
        for (name, shape) in shapes { self.source.rect(&name, &shape)?; }
        Ok(())
    }

    pub fn tensor(&self, name: &str, shape: &[usize]) -> Result<DeviceAllocation<'a>> {
        self.operand(vec![self.source.rect(name, shape)?])
    }

    fn operand(&self, rects: Vec<Rect>) -> Result<DeviceAllocation<'a>> {
        let bytes = rects.iter().try_fold(0usize, |n, r| n.checked_add(r.bytes())).context("drafter operand overflow")?;
        let allocation = DeviceAllocation::new(self.library, bytes.max(256))?;
        let mut at = 0;
        let lands: Vec<_> = rects.into_iter().map(|rect| { let offset = at; at += rect.bytes(); (rect, offset) }).collect();
        if let Some(prefetched) = &self.prefetched {
            for (rect, offset) in &lands {
                let start = usize::try_from(rect.base)?;
                let end = start.checked_add(rect.bytes()).context("drafter prefetched extent overflow")?;
                let data = prefetched.get(start..end).context("drafter tensor lies past prefetched file")?;
                self.library.copy_h2d(view(allocation.buffer, *offset, data.len())?, data)?;
            }
        } else {
            self.dense.operand(Operand { device: allocation.buffer.device_id, buffer: allocation.buffer, lands, bytes })?;
        }
        Ok(allocation)
    }

    pub fn concat(&self, parts: &[(&str, usize)], cols: usize) -> Result<DeviceAllocation<'a>> {
        self.operand(self.parts(parts, cols)?)
    }

    fn parts(&self, parts: &[(&str, usize)], cols: usize) -> Result<Vec<Rect>> {
        parts.iter().map(|(name, rows)| self.source.rect(name, &[*rows, cols])).collect()
    }

    pub fn matrix(&self, parts: &[(&str, usize)], cols: usize, fp8: bool, scales: Fp8Scales,
        stream: *mut c_void) -> Result<Weight<'a>> {
        if !fp8 { return self.concat(parts, cols).map(Weight::Bf16); }
        let rects = self.parts(parts, cols)?;
        let rows = parts.iter().try_fold(0usize, |n, (_, rows)| n.checked_add(*rows)).context("drafter matrix overflow")?;
        if self.prefetched.is_some() {
            let source = self.operand(rects)?;
            let packed = Fp8Weight::pack(self.library, source.buffer.ptr, rows, cols, scales, stream);
            // SAFETY: source and pack destinations remain owned until this load stream drains.
            if let Err(error) = unsafe { self.library.cuda_stream_synchronize(stream) } {
                self.library.quarantine_module_after_failed_drain();
                std::mem::forget(source);
                if let Ok(packed) = packed { packed.quarantine(); }
                return Err(error.context("drafter pack owners quarantined after failed drain"));
            }
            return packed.map(Weight::Fp8);
        }
        let packed = Fp8Weight::allocate(self.library, rows, cols)?;
        let (values, scale) = packed.buffers();
        for step in pack_steps(rects, cols, scales, values, scale)? { self.dense.step(step, stream)?; }
        Ok(Weight::Fp8(packed))
    }
}

fn view(buffer: CuteafdDeviceBuffer, at: usize, bytes: usize) -> Result<CuteafdDeviceBuffer> {
    ensure!(at.checked_add(bytes).is_some_and(|end| end <= buffer.bytes), "drafter pack destination out of bounds");
    // SAFETY: the checked byte range stays in the loader-owned allocation.
    Ok(CuteafdDeviceBuffer { ptr: unsafe { buffer.ptr.cast::<u8>().add(at) }.cast(), bytes, ..buffer })
}

fn pack_steps(rects: Vec<Rect>, cols: usize, rule: Fp8Scales, packed: CuteafdDeviceBuffer,
    scale: CuteafdDeviceBuffer) -> Result<Vec<Step>> {
    let mut first = 0usize;
    let mut steps = Vec::new();
    ensure!(cols > 0 && cols % 128 == 0, "drafter FP8 input width must be a multiple of 128");
    let row_bytes = cols.checked_mul(2).context("drafter FP8 row extent overflow")?;
    let scale_cols = cols / 128 * 4;
    let window = (STEP_BYTES / row_bytes / 16) * 16;
    ensure!(window > 0, "drafter FP8 row tile exceeds bounded staging");
    for rect in rects {
        ensure!(rect.rows.len() % 16 == 0 && rect.width() == row_bytes, "drafter FP8 source must hold whole 16-row tiles");
        for start in (0..rect.rows.len()).step_by(window) {
            let rows = window.min(rect.rows.len() - start);
            let values_at = first.checked_mul(cols).context("drafter FP8 value offset overflow")?;
            let scales_at = first.checked_mul(scale_cols).context("drafter FP8 scale offset overflow")?;
            let job = Job::W8a16Pack { n: rows, k: cols, rule,
                packed: view(packed, values_at, rows * cols)?,
                scale: view(scale, scales_at, rows * scale_cols)? };
            steps.push(Step { device: packed.device_id, rects: vec![rect.sub(start..start + rows)], jobs: vec![job] });
            first = first.checked_add(rows).context("drafter FP8 row count overflow")?;
        }
    }
    ensure!(first.checked_mul(cols) == Some(packed.bytes) && first.checked_mul(scale_cols) == Some(scale.bytes),
        "drafter FP8 packing footprint differs from its resident allocation");
    Ok(steps)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::experts::rtx::dense::{self, RankLayout};
    use cuteafd_loader::plan::testing::{t, write_safetensors};
    use std::os::unix::fs::FileExt;

    fn config() -> super::super::dflash::DflashConfig {
        super::super::dflash::DflashConfig { hidden: 128, intermediate: 128, layers: 2,
            heads: 1, kv_heads: 1, head_dim: 128, eps: 1e-6, theta: 10000., block: 8,
            group: 16, mask_token: 0, rank: 256, taps: vec![0], vocab: 16,
            window: 2048, row_window: true, sample_from_anchor: false }
    }

    fn write_drafter(root: &Path) -> Result<()> {
        std::fs::create_dir_all(root)?;
        let tensors = config().tensor_shapes().into_iter().map(|(name, shape)| t(name, "BF16", &shape)).collect::<Vec<_>>();
        let path = root.join("model.safetensors");
        write_safetensors(&path, &tensors);
        let file = std::fs::OpenOptions::new().write(true).open(&path)?;
        for tensor in read_safetensors_metadata(&path)? {
            let bytes = (0..tensor.byte_length).map(|i| (i as u8).wrapping_mul(29) ^ tensor.name.len() as u8).collect::<Vec<_>>();
            file.write_all_at(&bytes, tensor.byte_offset)?;
        }
        Ok(())
    }

    #[test]
    fn drafter_and_target_repositories_share_one_namespace_plan() -> Result<()> {
        let root = tempfile::tempdir()?;
        let target = root.path().join("hf-home/hub/models--target/snapshots/a");
        let draft = root.path().join("hf-home/hub/models--draft/snapshots/b");
        write_drafter(&target)?;
        write_drafter(&draft)?;
        let mut operands = Vec::new();
        for snapshot in [&target, &draft] {
            let source = Source::open(snapshot)?;
            for (name, shape) in config().tensor_shapes() {
                let rect = source.rect(&name, &shape)?;
                let bytes = rect.bytes();
                operands.push(Operand { device: 1, bytes, lands: vec![(rect, 0)],
                    buffer: CuteafdDeviceBuffer { bytes, device_id: 1, ..Default::default() } });
            }
        }
        let layout = RankLayout::new(operands, Vec::new(), 1)?;
        let plan = layout.checked(root.path(), 8, 7)?;
        assert_eq!(plan.plan.files.len(), 2);
        assert!(plan.plan.files.iter().any(|file| file.contains("models--target/")));
        assert!(plan.plan.files.iter().any(|file| file.contains("models--draft/")));
        assert!(plan.plan.entries.iter().all(|entry| entry.rank == 7));
        let bytes = config().tensor_shapes().iter().map(|(_, s)| s.iter().product::<usize>() * 2).sum::<usize>();
        assert_eq!(plan.totals.rank_bytes[7], (2 * bytes) as u64);
        assert!(layout.checked(&target, 8, 7).is_err(), "snapshot root must not masquerade as namespace root");
        Ok(())
    }

    #[test]
    fn fp8_concat_steps_cover_disjoint_tiles_without_bf16_residency() -> Result<()> {
        let cols = 128;
        let per = STEP_BYTES / (cols * 2);
        let rect = |name: &str, rows| Rect { name: name.into(), path: "draft".into(), base: 0,
            row_bytes: cols * 2, rows: 0..rows, cols: 0..cols * 2 };
        let rows = per + 32;
        let mut values = vec![0u8; 2 * rows * cols];
        let mut scales = vec![0u8; 2 * rows * cols / 128 * 4];
        let buffer = |data: &mut [u8]| CuteafdDeviceBuffer { ptr: data.as_mut_ptr().cast(),
            bytes: data.len(), device_id: 1, ..Default::default() };
        let steps = pack_steps(vec![rect("gate", rows), rect("up", rows)], cols, Fp8Scales::Amax,
            buffer(&mut values), buffer(&mut scales))?;
        assert_eq!(steps.len(), 4);
        assert!(steps.iter().all(|step| step.device == 1 && step.bytes() <= STEP_BYTES));
        let mut packed_bytes = 0;
        for step in steps {
            let Job::W8a16Pack { n, k, packed, scale, .. } = step.jobs[0] else { panic!("pack job") };
            assert_eq!(n % 16, 0);
            assert_eq!(packed.ptr as usize - values.as_ptr() as usize, packed_bytes);
            assert_eq!(scale.ptr as usize - scales.as_ptr() as usize, packed_bytes / 128 * 4);
            packed_bytes += n * k;
        }
        assert_eq!(packed_bytes, 2 * rows * cols);
        Ok(())
    }

    #[test]
    #[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
    fn muse_drafter_all_tensor_bytes_match_mount_after_push_commit() -> Result<()> {
        use crate::shared::memory::device::Device;
        let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build()?;
        let library = fixture.load()?;
        fixture.configure_memory(&library, 1, 1 << 30)?;
        fixture.immediate_copies(&library)?;
        let root = tempfile::tempdir_in(fixture.directory())?;
        let snapshot = root.path().join("models--muse-draft/snapshots/test");
        write_drafter(&snapshot)?;
        let device = Device { library: &library, id: 1 };
        let shapes = config().tensor_shapes();
        // Fuse the production GQA destinations: Q/K/V and gate/up share owners;
        // every other source keeps its native shape, including the 3-D bases.
        let mut groups = shapes.iter().map(|(name, shape)| vec![(name.clone(), shape.clone())]).collect::<Vec<_>>();
        for layer in 0..config().layers {
            for suffixes in [vec!["self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"],
                vec!["mlp.gate_proj.weight", "mlp.up_proj.weight"]] {
                let mut fused = Vec::new();
                for suffix in suffixes {
                    let name = format!("layers.{layer}.{suffix}");
                    let at = groups.iter().position(|group| group[0].0 == name).context("fused source")?;
                    fused.extend(groups.remove(at));
                }
                groups.push(fused);
            }
        }
        let load = || device.run(|| {
            // Mount keeps its existing full-file prefetch; recording must ignore it.
            let bytes = if dense::is_recording() { Vec::new() } else { std::fs::read(snapshot.join("model.safetensors"))? };
            let load = Load::with_prefetched(&library, &snapshot, bytes)?;
            load.validate(shapes.clone())?;
            groups.iter().map(|group| {
                if group.len() == 1 { return load.tensor(&group[0].0, &group[0].1); }
                let cols = group[0].1[1];
                let parts = group.iter().map(|(name, shape)| (name.as_str(), shape[0])).collect::<Vec<_>>();
                let Weight::Bf16(owner) = load.matrix(&parts, cols, false, Fp8Scales::Amax, std::ptr::null_mut())?
                    else { unreachable!("BF16 fixture") };
                Ok(owner)
            }).collect::<Result<Vec<_>>>()
        });
        let mounted = load()?;
        let (pushed, records) = dense::recording(load)?;
        assert_eq!(records.operands.len(), shapes.len() - 3 * config().layers);
        assert_eq!(records.operands.iter().map(|operand| operand.lands.len()).sum::<usize>(), shapes.len());
        let layout = RankLayout::new(records.operands.clone(), Vec::new(), 1)?;
        let plan = layout.checked(root.path(), 8, 7)?;
        let source_bytes = shapes.iter().map(|(_, shape)| shape.iter().product::<usize>() * 2).sum::<usize>();
        assert_eq!(plan.totals.rank_bytes[7], source_bytes as u64);
        assert_eq!(plan.plan.files.len(), 1);
        assert!(plan.plan.files[0].contains("models--muse-draft/"));
        assert!(records.steps.is_empty() && records.finishes.is_empty());
        assert!(records.operands.iter().all(|operand| operand.device == 1));
        let sockets = tempfile::tempdir_in(fixture.directory())?;
        // The same receive_rank completion/commit protocol as serving, no GPU.
        crate::shared::experts::rtx::push::tests::receive_pushed(device, None, root.path(), sockets.path(),
            &records, None, 6, 1)?;
        for (((group, expected), actual), operand) in groups.iter().zip(&mounted).zip(&pushed).zip(&records.operands) {
            let bytes = operand.bytes;
            let read = |owner: &DeviceAllocation<'_>| device.run(|| {
                let mut out = vec![0u8; bytes];
                library.copy_d2h(&mut out, owner.buffer)?;
                Ok(out)
            });
            let expected_bytes = read(expected)?;
            assert_eq!(expected_bytes, read(actual)?, "{group:?}");
            let source = Source::open(&snapshot)?;
            let native = group.iter().map(|(name, shape)| {
                let rect = source.rect(name, shape)?;
                let mut bytes = vec![0; rect.bytes()];
                rect.read(&mut bytes)?;
                Ok(bytes)
            }).collect::<Result<Vec<_>>>()?.into_iter().flatten().collect::<Vec<_>>();
            assert_eq!(expected_bytes, native, "mount concatenation {group:?}");
        }
        Ok(())
    }
}
