//! Header-native MoonViT3d builder. Execution remains gated until encoder oracles.
use crate::shared::vision::{ConstantRegion, TensorRead, TowerSpec};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::vision::{
    TowerExtension, TowerNorm, TowerProjector, VisionBlock, VisionSpec, NO_VISION_OFFSET as NONE,
};
use cuteafd_loader::families::kimi_k3::vision::{VisionContract, PROJECTOR_RMS_EPS, TOWER_RMS_EPS};
use std::{collections::BTreeMap, path::Path};

/// Official GPU arange/pow bytes, identical on SM120 and SM121 in the pinned
/// reference image. CPU pow differs at indices11/25; never regenerate on host.
const INV_FREQ_BITS: [u32; 32] = [
    0x3f800000, 0x3f3ff911, 0x3f0ff59a, 0x3ed7e89b, 0x3ea1e89b, 0x3e72d423, 0x3e361887, 0x3e088d77,
    0x3dcccccd, 0x3d99940d, 0x3d6655c2, 0x3d2cba16, 0x3d0186e3, 0x3cc2434f, 0x3c91ad39, 0x3c5a7bf2,
    0x3c23d70a, 0x3bf5b9b0, 0x3bb8449c, 0x3b8a2e77, 0x3b4f3e38, 0x3b1b690d, 0x3ae91528, 0x3aaec98e,
    0x3a83126f, 0x3a44948d, 0x3a136a16, 0x39dd1725, 0x39a5cb60, 0x3978a815, 0x393a7753, 0x390bd472,
];

pub(crate) fn tower(snapshot: &Path, max_output_rows: usize) -> Result<TowerSpec> {
    ensure!(
        (1..=16384).contains(&max_output_rows),
        "Kimi vision output capacity must be1..=16384"
    );
    let cfg: serde_json::Value = cuteafd_loader::read_json_file(&snapshot.join("config.json"))?;
    let geometry = VisionContract::from_config(&cfg)?;
    let mut catalog = TowerSpec::catalog(snapshot, "vision_tower.")?;
    catalog.extend(TowerSpec::catalog(snapshot, "mm_projector.")?);
    let packed = geometry.pack(catalog.values().map(|(_, m)| m))?;
    let mut reads = Vec::new();
    let mut offsets = BTreeMap::new();
    for tensor in packed.tensors {
        let (path, metadata) = catalog
            .get(&tensor.metadata.name)
            .context("Kimi tower catalog changed")?;
        offsets.insert(metadata.name.clone(), tensor.destination);
        reads.push(TensorRead {
            path: path.clone(),
            metadata: metadata.clone(),
            destination: usize::try_from(tensor.destination)?,
            vector: false,
        });
    }
    let at = |name: &str| {
        offsets
            .get(name)
            .copied()
            .with_context(|| format!("missing Kimi vision tensor {name}"))
    };
    let frequency_offset = packed.arena_bytes;
    let constant=ConstantRegion::fp32("rotary.spatial.inv_freq",frequency_offset,
        INV_FREQ_BITS.into_iter().map(f32::from_bits),
        "GPU FP32 arange(0,128,4)/128; reciprocal(10000**exponent); shared H/W",
        "moonshotai/Kimi-K3@f831ab66814297da540d832a5235f8e904f29d06 modeling_kimi_k3.py sha256=b9171c96726eda55234c92ac8dfae7e24c512fda68968ae8f2c3782b42665ea2; torch2.12.0a0+5aff3928d8.nv26.05 CUDA13.2; SM120/SM121 sha256=8f9f46a7ce1bf44ab866c40009fd40efc81fd340f2690db7969d9eff247bf650");
    let final_norm = TowerNorm {
        kind: 2,
        epsilon: TOWER_RMS_EPS,
        weight: at("vision_tower.encoder.final_layernorm.weight")?,
        bias: NONE,
    };
    let projector = TowerProjector {
        kind: 1,
        activation: 1,
        input_width: 4096,
        hidden_width: 4096,
        output_width: 7168,
        fc1: at("mm_projector.proj.0.weight")?,
        fc2: at("mm_projector.proj.2.weight")?,
        ..Default::default()
    };
    let extension = TowerExtension {
        block_norm_kind: 2,
        activation: 2,
        block_norm_epsilon: TOWER_RMS_EPS,
        attention_width: 1536,
        temporal_patch: 1,
        pixel_layout: 1,
        temporal_input: 0,
        rotary_kind: 2,
        rotary_axis_dims: [0, 64, 64],
        rotated_dims: 128,
        passthrough_dims: 0,
        inv_freq: [NONE, frequency_offset, frequency_offset],
        absolute_kind: 1,
        position_height: 64,
        position_width: 64,
        interpolation: 1,
        absolute_position: at("vision_tower.patch_embed.pos_emb.weight")?,
        final_norm,
        spatial_merge: 2,
        projection_order: 1,
        projectors: [projector, Default::default()],
        post_norm: TowerNorm {
            kind: 2,
            epsilon: PROJECTOR_RMS_EPS,
            weight: at("mm_projector.post_norm.weight")?,
            bias: NONE,
        },
        ..Default::default()
    };
    let mut native = VisionSpec {
        abi_version: 4,
        max_tokens: u32::try_from(max_output_rows)?,
        output_width: 7168,
        weight_bytes: frequency_offset + 128,
        patch: at("vision_tower.patch_embed.proj.weight")?,
        inv_freq: NONE,
        hidden: 1024,
        heads: 12,
        kv_heads: 12,
        head_dim: 128,
        intermediate: 4096,
        patch_size: 14,
        merger_width: 4096,
        norm_eps: TOWER_RMS_EPS,
        patch_bias: NONE,
        pos_embed: NONE,
        merger_norm: NONE,
        merger_norm_bias: NONE,
        merger_fc1: NONE,
        merger_fc1_bias: NONE,
        merger_fc2: NONE,
        merger_fc2_bias: NONE,
        merger_extra: [NONE; 8],
        tower: Some(extension),
        ..Default::default()
    }
    .with_depth(geometry.depth);
    for layer in 0..geometry.depth {
        let stem = format!("vision_tower.encoder.blocks.{layer}");
        let field = |suffix: &str| at(&format!("{stem}.{suffix}"));
        native.blocks[layer] = VisionBlock {
            qkv: field("wqkv.weight")?,
            qkv_bias: NONE,
            proj: field("wo.weight")?,
            proj_bias: NONE,
            gate_up: field("mlp.fc0.weight")?,
            gate_up_bias: NONE,
            down: field("mlp.fc1.weight")?,
            down_bias: NONE,
            norm1: field("norm0.weight")?,
            norm2: field("norm1.weight")?,
            key0_bias: NONE,
            window: 0,
            column_order: 0,
        };
        native.norm1_bias[layer] = NONE;
        native.norm2_bias[layer] = NONE;
        native.q_norm[layer] = NONE;
        native.k_norm[layer] = NONE;
    }
    TowerSpec::from_parts_with_constants(native, reads, vec![constant]).map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    #[test]
    fn frequencies_are_gpu_reference_bytes_not_cpu_pow() {
        let bytes = INV_FREQ_BITS
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        assert_eq!(
            format!("{:x}", Sha256::digest(bytes)),
            "8f9f46a7ce1bf44ab866c40009fd40efc81fd340f2690db7969d9eff247bf650"
        );
        assert_eq!(INV_FREQ_BITS[11], 0x3d2cba16);
        assert_eq!(INV_FREQ_BITS[25], 0x3a44948d);
    }
    #[test]
    fn capacity_is_bounded_before_touching_checkpoint() {
        for capacity in [0, 16385] {
            assert!(tower(Path::new("absent"), capacity).is_err());
        }
    }
}
