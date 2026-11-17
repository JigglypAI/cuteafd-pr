//! Header-native MoonViT3d builder. Execution remains gated until encoder oracles.
use crate::shared::vision::{ConstantRegion, ReadTransform, TensorRead, TowerSpec};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::vision::{
    TowerExtension, TowerNorm, TowerProjector, VisionBlock, VisionSpec, NO_VISION_OFFSET as NONE,
};
use cuteafd_loader::families::kimi_k3::vision::{VisionContract, MAX_IMAGE_TOKENS, PROJECTOR_RMS_EPS, TOWER_RMS_EPS};
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
        (1..=MAX_IMAGE_TOKENS as usize).contains(&max_output_rows),
        "Kimi vision output capacity must be1..={MAX_IMAGE_TOKENS}"
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
            transform: ReadTransform::Direct,
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
    fn header_fixture_builds_complete_one_arena_descriptor() -> Result<()> {
        use std::io::Write;
        let cfg = serde_json::json!({"model_type":"kimi_k3","text_config":{"hidden_size":7168},"vision_config":{
            "vt_hidden_size":1024,"vt_num_hidden_layers":27,"vt_num_attention_heads":12,"qkv_hidden_size":1536,
            "vt_intermediate_size":4096,"patch_size":14,"init_pos_emb_height":64,"init_pos_emb_width":64,
            "init_pos_emb_time":4,"mm_hidden_size":1024,"text_hidden_size":7168,"activation_func":"gelu_pytorch_tanh",
            "norm_type":"rmsnorm","mlp_type":"mlp2","pos_emb_type":"divided_fixed","pos_emb_interpolation_mode":"bilinear",
            "merge_type":"sd2_tpool","mm_projector_type":"patchmergerv2","projector_hidden_act":"gelu",
            "attn_bias":false,"linear_bias":false,"patch_embed_proj_bias":false,"merge_kernel_size":[2,2],"projector_ln_eps":1e-5}});
        let g = VisionContract::from_config(&cfg)?;
        let dir = tempfile::tempdir()?;
        std::fs::write(dir.path().join("config.json"), serde_json::to_vec(&cfg)?)?;
        let mut header = serde_json::Map::new();
        let mut end = 0u64;
        for (name, shape) in g.tensor_shapes() {
            let bytes = shape.iter().product::<usize>() as u64 * 2;
            header.insert(
                name,
                serde_json::json!({"dtype":"BF16","shape":shape,"data_offsets":[end,end+bytes]}),
            );
            end += bytes;
        }
        let header = serde_json::to_vec(&header)?;
        let mut file = std::fs::File::create(dir.path().join("model.safetensors"))?;
        file.write_all(&(header.len() as u64).to_le_bytes())?;
        file.write_all(&header)?;
        file.set_len(8 + header.len() as u64 + end)?;
        let spec = tower(dir.path(), MAX_IMAGE_TOKENS as usize)?;
        assert_eq!(spec.native.weight_bytes, 894_718_080);
        assert_eq!(spec.native.blocks.len(), 27);
        let extension = spec.native.tower.as_ref().unwrap();
        assert_eq!(extension.inv_freq[1], extension.inv_freq[2]);
        assert_eq!(extension.attention_width, 1536);
        assert_eq!(extension.final_norm.epsilon, TOWER_RMS_EPS);
        assert_eq!(spec.constant_regions().len(), 1);
        assert_eq!(spec.constant_regions()[0].name, "rotary.spatial.inv_freq");
        assert_eq!(spec.constant_regions()[0].bytes.len(), 128);
        extension.validate(&spec.native)?;
        crate::shared::vision::push::tests::assert_vision_plan(dir.path(), &spec);
        assert!(spec.staged_rects().is_empty());
        assert_eq!(spec.constant_regions()[0].offset + 128, spec.native.weight_bytes);
        let mut k31 = cfg.clone();
        k31["model_type"] = "kimi_k31".into();
        std::fs::write(dir.path().join("config.json"), serde_json::to_vec(&k31)?)?;
        let sibling = tower(dir.path(), MAX_IMAGE_TOKENS as usize)?;
        assert_eq!(sibling.native.weight_bytes, spec.native.weight_bytes);
        assert_eq!(sibling.native.blocks, spec.native.blocks);
        assert_eq!(sibling.native.tower, spec.native.tower);
        assert_eq!(sibling.constant_regions(), spec.constant_regions());
        Ok(())
    }
    #[test]
    fn capacity_is_bounded_before_touching_checkpoint() {
        for capacity in [0, MAX_IMAGE_TOKENS as usize + 1] {
            assert!(tower(Path::new("absent"), capacity).is_err());
        }
    }
}
