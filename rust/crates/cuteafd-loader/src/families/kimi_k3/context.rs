//! Exact context-only allocations shared by the Kimi planner and native owner.
use anyhow::{ensure, Context, Result};
use cuteafd_core::MlaCacheFormat;
use crate::placement::ContextBuffers;

const ROWS: usize = 64;
const HEADS: usize = 48;
const METADATA: [&str; 5] = ["context slots", "compact page tables", "shard lengths", "gather tables", "commit blocks"];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextLayout {
    buffers: ContextBuffers,
    workspace: Vec<(&'static str, usize)>,
    metadata_banks: usize,
}

/// Required names are format-specific: a BF16 package cannot admit FP8 context.
pub fn program_names(format: MlaCacheFormat) -> Vec<String> {
    let tag = if format == MlaCacheFormat::Fp8 { "_fp8" } else { "" };
    let mut names = vec!["kimik3_lse_combine2".into(), format!("kimik3_paged_staging_gather{tag}")];
    for head in [0, HEADS] {
        names.push(format!("kimik3_mla_partial{tag}_h{head}_m64"));
    }
    names
}

impl ContextLayout {
    pub fn from_manifest(manifest: &serde_json::Value, format: MlaCacheFormat,
        max_context: usize, unit_rows: usize, metadata_banks: usize) -> Result<Self> {
        ensure!((1..=super::COMPILED_CONTEXT).contains(&max_context)
            && (1..=super::allocation::UNIT_ROWS_MAX).contains(&unit_rows)
            && (1..=super::allocation::LANE_ROWS_MAX).contains(&metadata_banks),
            "Kimi context allocation capacity outside supported geometry");
        let programs = manifest["programs"].as_array().context("Kimi context program inventory missing")?;
        let mut scratch = 0;
        for name in program_names(format) {
            let entries = programs.iter().filter(|p| p["name"].as_str() == Some(&name)).collect::<Vec<_>>();
            ensure!(entries.len() == 1, "Kimi context requires one {name} program");
            if name.contains("_partial") {
                let bytes = entries[0]["scratch_bytes_at_capacity"]["scratch"].as_u64()
                    .with_context(|| format!("Kimi context missing {name} scratch extent"))?;
                let bytes = usize::try_from(bytes).context("Kimi context scratch extent overflow")?;
                let geometry = &entries[0]["geometry"];
                ensure!(geometry["num_splits"] == 16 && geometry["runtime_splits"] == true
                    && geometry["lse_base2"] == true,
                    "Kimi context {name} requires runtime splits (capacity 16) and base-2 LSE");
                let expected = (ROWS * HEADS * 16 * 512 * 2).next_multiple_of(1024)
                    + ROWS * HEADS * 16 * 4;
                ensure!(bytes == expected, "Kimi context {name} scratch extent {bytes} differs from native {expected}");
                scratch = scratch.max(bytes);
            }
        }
        let half_pages = max_context.div_ceil(ROWS).div_ceil(2);
        let partial = ROWS * HEADS * (512 * 2 + 4);
        let mut workspace = vec![("own-head partial", partial), ("peer-head partial", partial), ("partial scratch", scratch)];
        for (name, bytes) in METADATA.into_iter().zip([unit_rows * 8,
            ROWS * half_pages * 4, ROWS * 4, 2 * half_pages * 4,
            3 * (unit_rows.div_ceil(ROWS) + 1) * 8]) {
            let stride = bytes.max(256).next_multiple_of(256);
            workspace.push((name, stride.checked_mul(metadata_banks).context("Kimi context metadata banks overflow")?));
        }
        let workspace_bytes = workspace.iter().try_fold(0u64, |sum, (_, bytes)|
            sum.checked_add(*bytes as u64).context("Kimi context workspace overflow"))?;
        let buffers = ContextBuffers { staging_unit_bytes: (ROWS * format.record_bytes()) as u64,
            staging_unit_rows: ROWS as u64, compiled_extent: max_context as u64,
            query_row_bytes: (HEADS * 576 * 2) as u64, partial_row_bytes: (HEADS * (512 * 2 + 4)) as u64,
            decode_rows: ROWS as u64, lanes: 1, workspace_bytes, ..Default::default() };
        buffers.demands()?;
        Ok(Self { buffers, workspace, metadata_banks })
    }

    pub fn buffers(&self) -> ContextBuffers { self.buffers }
    pub fn workspace(&self) -> Vec<(&'static str, usize)> { self.workspace.clone() }

    /// Each unit keeps its compact/gather/commit metadata until the wave drains.
    pub fn metadata_extent(&self, name: &str, bank: usize) -> Result<(usize, usize)> {
        ensure!(METADATA.contains(&name) && bank < self.metadata_banks, "Kimi context metadata bank outside admission");
        let bytes = self.workspace.iter().find(|(key, _)| *key == name).context("Kimi context metadata absent")?.1;
        let stride = bytes / self.metadata_banks;
        Ok((bank * stride, stride))
    }
}

#[cfg(test)]
pub(crate) fn add_manifest_programs(manifest: &mut serde_json::Value) {
    for format in [MlaCacheFormat::Bf16, MlaCacheFormat::Fp8] {
        for name in program_names(format) {
            let splits = 16;
            let bytes = if name.contains("_partial") {
                (ROWS * HEADS * splits * 512 * 2).next_multiple_of(1024) + ROWS * HEADS * splits * 4
            } else { 0 };
            // Combine is shared by the two record formats.
            let programs = manifest["programs"].as_array_mut().unwrap();
            if !programs.iter().any(|p| p["name"] == name) {
                programs.push(serde_json::json!({"name":name,"scratch_bytes_at_capacity":{"scratch":bytes},
                    "geometry":{"num_splits":16,"runtime_splits":true,"lse_base2":true}}));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn named_runtime_allocations_equal_planner_context_charges() {
        let mut manifest = serde_json::json!({"programs":[]});
        add_manifest_programs(&mut manifest);
        for format in [MlaCacheFormat::Bf16, MlaCacheFormat::Fp8] {
            for extent in [262_144, 524_288, 1_048_576] {
                let layout = ContextLayout::from_manifest(&manifest, format, extent, 2752, 3).unwrap();
                let buffers = layout.buffers();
                let demands = buffers.demands().unwrap();
                for gpu in 0..2 {
                    let charge = |name: &str| demands.iter().find(|d| d.gpu == gpu && d.group == name).unwrap().bytes;
                    assert_eq!(charge("context workspace"), layout.workspace().iter().map(|(_, n)| *n as u64).sum::<u64>());
                    assert_eq!(charge("context staging"), 2 * extent as u64 * format.record_bytes() as u64);
                    assert_eq!(charge("context exchange"), crate::placement::ContextExchangeLayout::new(&buffers).bytes().unwrap());
                }
                for name in METADATA {
                    let bytes = layout.workspace().into_iter().find(|(key, _)| *key == name).unwrap().1;
                    for bank in 0..3 {
                        let (offset, stride) = layout.metadata_extent(name, bank).unwrap();
                        assert_eq!(offset, bank * stride);
                        assert_eq!(stride % 256, 0);
                        assert_eq!(3 * stride, bytes);
                    }
                    assert!(layout.metadata_extent(name, 3).is_err());
                }
            }
        }
    }

    #[test]
    fn context_manifest_is_format_specific_and_fails_closed() {
        let mut manifest = serde_json::json!({"programs":[]});
        add_manifest_programs(&mut manifest);
        for name in program_names(MlaCacheFormat::Fp8) {
            let mut bad = manifest.clone();
            bad["programs"].as_array_mut().unwrap().retain(|p| p["name"] != name);
            assert!(ContextLayout::from_manifest(&bad, MlaCacheFormat::Fp8, 1024, 64, 1)
                .unwrap_err().to_string().contains(&name));
        }
        let mut bad = manifest.clone();
        bad["programs"].as_array_mut().unwrap().iter_mut().find(|p| p["name"] == "kimik3_mla_partial_fp8_h0_m64")
            .unwrap()["scratch_bytes_at_capacity"]["scratch"] = 1.into();
        assert!(ContextLayout::from_manifest(&bad, MlaCacheFormat::Fp8, 1024, 64, 1).is_err());
        assert!(ContextLayout::from_manifest(&manifest, MlaCacheFormat::Fp8, usize::MAX, 64, 1).is_err());
    }
}
