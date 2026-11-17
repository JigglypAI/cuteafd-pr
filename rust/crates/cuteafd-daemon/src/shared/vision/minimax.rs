//! Header-only MiniMax CLIP tower. Native dispatch remains oracle-gated.
use super::*;
use anyhow::{ensure, Context};
use cuteafd_ffi::vision::{TowerExtension, TowerNorm, TowerProjector};
use cuteafd_loader::families::minimax_m3::vision::{VisionContract, MAX_IMAGE_TOKENS};

// Provisional CPU capture only: vendor construction uses a CUDA default device.
// Replace from both-architecture real-module buffers before enabling dispatch.
const INV_FREQ_BITS: [u32; 13] = [
    0x3f800000, 0x3efc1a50, 0x3e7843d0, 0x3df47c46, 0x3d70c376, 0x3ced1928, 0x3c697d22, 0x3be5ef2c,
    0x3b626f12, 0x3adefc9d, 0x3a5b9797, 0x39d83fcc, 0x3954f507,
];

impl TowerSpec {
    pub fn minimax(snapshot: &Path, max_output_rows: usize) -> anyhow::Result<Self> {
        ensure!(
            (1..=MAX_IMAGE_TOKENS).contains(&max_output_rows),
            "MiniMax vision output capacity must be 1..={MAX_IMAGE_TOKENS}"
        );
        let cfg = cuteafd_loader::read_json_file(&snapshot.join("config.json"))?;
        let contract = VisionContract::from_config(&cfg)?;
        let g = &contract.geometry;
        let mut catalog = tower_catalog(snapshot, "vision_tower.")?;
        catalog.extend(tower_catalog(snapshot, "multi_modal_projector.")?);
        catalog.extend(tower_catalog(snapshot, "patch_merge_mlp.")?);
        let packed = contract.pack(catalog.values().map(|(_, metadata)| metadata))?;
        let mut reads = Vec::new();
        let mut offsets = BTreeMap::new();
        for tensor in packed.tensors {
            let (path, metadata) = catalog
                .get(&tensor.metadata.name)
                .context("MiniMax tower catalog changed")?;
            offsets.insert(metadata.name.clone(), tensor.destination);
            // Only CLIPVisionEmbeddings.forward casts this FP32 checkpoint tensor.
            let transform =
                if metadata.name == "vision_tower.vision_model.embeddings.patch_embedding.weight" {
                    ensure!(
                        metadata.dtype == DType::F32,
                        "MiniMax patch source must be FP32"
                    );
                    ReadTransform::F32ToBf16
                } else {
                    ensure!(
                        metadata.dtype == DType::Bf16,
                        "MiniMax non-patch source must be BF16"
                    );
                    ReadTransform::Direct
                };
            ensure!(
                transform.resident_bytes(metadata)? as u64 == tensor.resident_bytes,
                "MiniMax tower resident extent mismatch"
            );
            reads.push(TensorRead {
                path: path.clone(),
                metadata: metadata.clone(),
                destination: usize::try_from(tensor.destination)?,
                transform,
            });
        }
        let at = |name: &str| {
            offsets
                .get(name)
                .copied()
                .with_context(|| format!("missing MiniMax vision tensor {name}"))
        };
        let frequency_offset = packed.arena_bytes;
        let constant = ConstantRegion::fp32("rotary.thw.inv_freq", frequency_offset,
            INV_FREQ_BITS.into_iter().map(f32::from_bits),
            "CPU FP32 arange(0,26,2)/26; reciprocal(10000**exponent); shared T/H/W; torch2.14.1+cpu",
            "sglang@a9bf8c76a0115f5e2672bfdfc61d70a2b7936744 minimax_vl_common.py sha256=89ceb45627033fa95530d60939d060d4747abe99b3b85a022cbef31a8259e18f");
        let projector = |prefix: &str, input_width| -> anyhow::Result<TowerProjector> {
            Ok(TowerProjector {
                kind: 1,
                activation: 1,
                input_width,
                hidden_width: g.projector_hidden as u32,
                output_width: g.projection as u32,
                fc1: at(&format!("{prefix}.linear_1.weight"))?,
                fc1_bias: at(&format!("{prefix}.linear_1.bias"))?,
                fc2: at(&format!("{prefix}.linear_2.weight"))?,
                fc2_bias: at(&format!("{prefix}.linear_2.bias"))?,
                ..Default::default()
            })
        };
        let extension = TowerExtension {
            block_norm_kind: 1,
            activation: 1,
            block_norm_epsilon: g.layer_norm_eps as f32,
            attention_width: g.hidden as u32,
            temporal_patch: 2,
            pixel_layout: 2,
            temporal_input: 1,
            rotary_kind: 1,
            rotary_axis_dims: [26; 3],
            rotated_dims: 78,
            passthrough_dims: 2,
            inv_freq: [frequency_offset; 3],
            pre_norm: TowerNorm {
                kind: 1,
                epsilon: g.layer_norm_eps as f32,
                weight: at("vision_tower.vision_model.pre_layrnorm.weight")?,
                bias: at("vision_tower.vision_model.pre_layrnorm.bias")?,
            },
            spatial_merge: 2,
            projection_order: 2,
            projectors: [
                projector("multi_modal_projector", g.hidden as u32)?,
                projector("patch_merge_mlp", (4 * g.projection) as u32)?,
            ],
            ..Default::default()
        };
        let mut native = VisionSpec {
            abi_version: 4,
            max_tokens: u32::try_from(max_output_rows)?,
            output_width: g.projection as u32,
            weight_bytes: frequency_offset + 52,
            patch: at("vision_tower.vision_model.embeddings.patch_embedding.weight")?,
            hidden: g.hidden as u32,
            heads: g.heads as u32,
            kv_heads: g.heads as u32,
            head_dim: (g.hidden / g.heads) as u32,
            intermediate: g.intermediate as u32,
            patch_size: g.patch as u32,
            merger_width: (4 * g.projection) as u32,
            norm_eps: g.layer_norm_eps as f32,
            patch_bias: NO_VISION_OFFSET,
            pos_embed: NO_VISION_OFFSET,
            inv_freq: NO_VISION_OFFSET,
            merger_norm: NO_VISION_OFFSET,
            merger_norm_bias: NO_VISION_OFFSET,
            merger_fc1: NO_VISION_OFFSET,
            merger_fc2: NO_VISION_OFFSET,
            merger_fc1_bias: NO_VISION_OFFSET,
            merger_fc2_bias: NO_VISION_OFFSET,
            merger_extra: [NO_VISION_OFFSET; 8],
            tower: Some(extension),
            ..Default::default()
        }
        .with_depth(g.layers);
        for layer in 0..g.layers {
            let stem = format!("vision_tower.vision_model.encoder.layers.{layer}");
            let field = |suffix: &str| at(&format!("{stem}.{suffix}"));
            let qkv = field("self_attn.q_proj.weight")?;
            let qkv_bias = field("self_attn.q_proj.bias")?;
            for (index, part) in [(1, "k"), (2, "v")] {
                ensure!(
                    field(&format!("self_attn.{part}_proj.weight"))?
                        == qkv + (index * g.hidden * g.hidden * 2) as u64
                        && field(&format!("self_attn.{part}_proj.bias"))?
                            == qkv_bias + (index * g.hidden * 2) as u64,
                    "MiniMax QKV concatenation extent mismatch"
                );
            }
            native.blocks[layer] = VisionBlock {
                qkv,
                qkv_bias,
                proj: field("self_attn.out_proj.weight")?,
                proj_bias: field("self_attn.out_proj.bias")?,
                gate_up: field("mlp.fc1.weight")?,
                gate_up_bias: field("mlp.fc1.bias")?,
                down: field("mlp.fc2.weight")?,
                down_bias: field("mlp.fc2.bias")?,
                norm1: field("layer_norm1.weight")?,
                norm2: field("layer_norm2.weight")?,
                key0_bias: NO_VISION_OFFSET,
                window: 0,
                column_order: 0,
            };
            native.norm1_bias[layer] = field("layer_norm1.bias")?;
            native.norm2_bias[layer] = field("layer_norm2.bias")?;
            native.q_norm[layer] = NO_VISION_OFFSET;
            native.k_norm[layer] = NO_VISION_OFFSET;
        }
        Self::from_parts_with_constants(native, reads, vec![constant]).map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn snapshot() -> tempfile::TempDir {
        let cfg = serde_json::json!({"model_type":"minimax_m3_vl",
            "text_config":{"hidden_size":6144}, "multimodal_projector_bias":true,
            "projector_hidden_act":"gelu", "projector_hidden_size":6144,
            "vision_feature_layer":-1,"vision_feature_select_strategy":"full",
            "vision_config":{"hidden_size":1280,"num_hidden_layers":32,
                "num_attention_heads":16,"intermediate_size":5120,"patch_size":14,
                "projection_dim":6144,"num_channels":3,"hidden_act":"gelu",
                "position_embedding_type":"rope","rope_mode":"3d","rope_theta":10000.0,
                "layer_norm_eps":1e-5,"vision_segment_max_frames":4,
                "img_token_compression_config":{"image_token_compression_method":"patch_merge",
                    "spatial_merge_size":2,"temporal_patch_size":2}}});
        let contract = VisionContract::from_config(&cfg).unwrap();
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(
            directory.path().join("config.json"),
            serde_json::to_vec(&cfg).unwrap(),
        )
        .unwrap();
        let mut header = serde_json::Map::new();
        let mut end = 0u64;
        for tensor in contract.tensor_contracts() {
            let dtype = if tensor.source_dtype == DType::F32 {
                "F32"
            } else {
                "BF16"
            };
            let bytes = tensor.shape.iter().product::<usize>() as u64
                * if tensor.source_dtype == DType::F32 {
                    4
                } else {
                    2
                };
            header.insert(
                tensor.name,
                serde_json::json!({"dtype":dtype,"shape":tensor.shape,
                "data_offsets":[end,end+bytes]}),
            );
            end += bytes;
        }
        let bytes = serde_json::to_vec(&header).unwrap();
        let mut file = File::create(directory.path().join("model.safetensors")).unwrap();
        file.write_all(&(bytes.len() as u64).to_le_bytes()).unwrap();
        file.write_all(&bytes).unwrap();
        file.set_len(8 + bytes.len() as u64 + end).unwrap();
        directory
    }

    #[test]
    fn header_builder_and_push_inventory_have_one_bf16_backing() {
        let directory = snapshot();
        let spec = TowerSpec::minimax(directory.path(), MAX_IMAGE_TOKENS).unwrap();
        assert_eq!(spec.reads.len(), 523);
        assert_eq!(spec.native.depth, 32);
        assert_eq!(spec.native.weight_bytes, 1_731_134_516);
        assert_eq!(spec.constants.len(), 1);
        assert_eq!(spec.constants[0].name, "rotary.thw.inv_freq");
        assert_eq!(spec.constants[0].offset + 52, spec.native.weight_bytes);
        assert_eq!(
            spec.reads
                .iter()
                .filter(|r| r.metadata.dtype == DType::F32)
                .count(),
            1
        );
        assert_eq!(
            spec.reads
                .iter()
                .filter(|r| r.metadata.name.starts_with("vision_tower."))
                .count(),
            515
        );
        assert_eq!(
            spec.reads
                .iter()
                .filter(|r| r.metadata.name.starts_with("multi_modal_projector."))
                .count(),
            4
        );
        assert_eq!(
            spec.reads
                .iter()
                .filter(|r| r.metadata.name.starts_with("patch_merge_mlp."))
                .count(),
            4
        );
        let extension = spec.native.tower.as_ref().unwrap();
        assert_eq!(extension.rotary_axis_dims, [26; 3]);
        assert_eq!(extension.passthrough_dims, 2);
        assert_eq!(extension.final_norm, TowerNorm::default());
        assert_eq!(extension.projectors[1].input_width, 24576);
        extension.validate(&spec.native).unwrap();
        let patch = &spec.reads[0];
        assert_eq!(patch.transform, ReadTransform::F32ToBf16);
        assert_eq!(patch.metadata.byte_length, 6_021_120);
        assert_eq!(
            patch.transform.resident_bytes(&patch.metadata).unwrap(),
            3_010_560
        );
        assert!(spec.reads[1..]
            .iter()
            .all(|read| read.transform == ReadTransform::Direct));
        assert!(spec.reads.iter().all(|read| read.destination % 256 == 0));
        assert_eq!(
            spec.staged_rects().iter().map(|r| r.bytes()).sum::<usize>(),
            6_021_120
        );
        let contribution =
            push::Contribution::prepare(0, Some((&spec, 0x100000)), None, None, None).unwrap();
        let records = contribution.records();
        assert_eq!(records.operands.len(), 522);
        assert_eq!(records.steps.len(), spec.staged_rects().len());
        assert!(records.steps.iter().all(|step| step.jobs.len() == 1));
        push::tests::assert_vision_plan(directory.path(), &spec);
        let resident: usize = spec
            .reads
            .iter()
            .map(|r| r.transform.resident_bytes(&r.metadata).unwrap())
            .sum();
        assert_eq!(
            resident as u64 + spec.constants[0].bytes.len() as u64,
            spec.native.weight_bytes
        );
        assert!(TowerSpec::from_snapshot(directory.path(), MAX_IMAGE_TOKENS).is_err());
    }

    #[test]
    fn frequency_bytes_and_capacity_are_explicit() {
        use sha2::{Digest, Sha256};
        let bytes: Vec<_> = INV_FREQ_BITS
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect();
        assert_eq!(
            format!("{:x}", Sha256::digest(bytes)),
            "12f61d6b811cf2908575af2029d2271c3f6ab08c3a2c9810ebafa2d175f7ccdf"
        );
        for capacity in [0, MAX_IMAGE_TOKENS + 1] {
            assert!(TowerSpec::minimax(Path::new("absent"), capacity).is_err());
        }
    }
}
