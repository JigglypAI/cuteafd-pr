//! Header-native Hy4-flash ViT on the generic still tower: biased patch-16
//! raster embedding, scale-factor learned positions, separate biased Q/K/V,
//! exact GELU, RMSNorm + conv merger and checkpoint newline rows.
use super::*;
use anyhow::Context;
use cuteafd_ffi::vision::{TowerExtension, TowerNorm, TowerProjector};
use cuteafd_loader::families::hy4_flash::vision::{VisionContract, MAX_MERGED_ROWS, POSITION_EDGE};

impl TowerSpec {
    pub fn hy4_flash(snapshot: &Path, max_output_rows: usize) -> anyhow::Result<Self> {
        anyhow::ensure!((1..=MAX_MERGED_ROWS).contains(&max_output_rows),
            "Hy4 vision merged capacity must be 1..={MAX_MERGED_ROWS}");
        let cfg = cuteafd_loader::read_json_file(&snapshot.join("config.json"))?;
        let contract = VisionContract::from_config(&cfg)?;
        let catalog = tower_catalog(snapshot, "vit.")?;
        let packed = contract.pack(catalog.values().map(|(_, metadata)| metadata))?;
        let mut reads = Vec::with_capacity(packed.tensors.len());
        for tensor in &packed.tensors {
            let (path, metadata) = catalog.get(&tensor.metadata.name).context("Hy4 tower catalog changed")?;
            reads.push(TensorRead { path: path.clone(), metadata: metadata.clone(),
                destination: usize::try_from(tensor.destination)?, transform: ReadTransform::Direct });
        }
        let at = |name: &str| packed.offset(name);
        let (h, o, m) = (contract.hidden as u32, contract.output as u32, contract.merger_hidden as u32);
        let extension = TowerExtension {
            block_norm_kind: 1,
            activation: 1,
            block_norm_epsilon: contract.eps,
            attention_width: h,
            temporal_patch: 1,
            pixel_layout: 1,
            temporal_input: 0,
            rotary_kind: 0,
            absolute_kind: 2,
            position_height: POSITION_EDGE as u32,
            position_width: POSITION_EDGE as u32,
            interpolation: 2,
            absolute_position: at("vit.embeddings.position_embedding.weight")?,
            final_norm: TowerNorm { kind: 2, epsilon: contract.eps, weight: at("vit.perceive.norm.weight")?, bias: NO_VISION_OFFSET },
            spatial_merge: 2,
            projection_order: 3,
            projectors: [TowerProjector {
                kind: 1, activation: 1, input_width: 4 * h, hidden_width: m, output_width: o,
                fc1: at("vit.perceive.proj.0.weight")?, fc1_bias: at("vit.perceive.proj.0.bias")?,
                fc2: at("vit.perceive.proj.2.weight")?, fc2_bias: at("vit.perceive.proj.2.bias")?,
                ..Default::default()
            }, Default::default()],
            ..Default::default()
        };
        let mut merger_extra = [NO_VISION_OFFSET; 8];
        merger_extra[0] = at("vit.perceive.image_newline")?;
        let mut native = VisionSpec {
            abi_version: 4,
            max_tokens: u32::try_from(max_output_rows)?,
            output_width: o,
            weight_bytes: packed.arena_bytes,
            patch: at("vit.embeddings.patch_embedding.weight")?,
            patch_bias: at("vit.embeddings.patch_embedding.bias")?,
            hidden: h,
            heads: contract.heads as u32,
            kv_heads: contract.heads as u32,
            head_dim: (contract.hidden / contract.heads) as u32,
            intermediate: contract.intermediate as u32,
            patch_size: 16,
            merger_width: 4 * h,
            norm_eps: contract.eps,
            pos_embed: NO_VISION_OFFSET,
            inv_freq: NO_VISION_OFFSET,
            merger_norm: NO_VISION_OFFSET,
            merger_norm_bias: NO_VISION_OFFSET,
            merger_fc1: NO_VISION_OFFSET,
            merger_fc1_bias: NO_VISION_OFFSET,
            merger_fc2: NO_VISION_OFFSET,
            merger_fc2_bias: NO_VISION_OFFSET,
            merger_extra,
            tower: Some(extension),
            ..Default::default()
        }.with_depth(contract.layers);
        for layer in 0..contract.layers {
            let field = |suffix: &str| at(&format!("vit.layers.{layer}.{suffix}"));
            // Q/K/V weights and biases are adjacent (checked by the pack); the
            // native tower projects them separately from these base offsets.
            native.blocks[layer] = VisionBlock {
                qkv: field("self_attn.q_proj.weight")?,
                qkv_bias: field("self_attn.q_proj.bias")?,
                proj: field("self_attn.o_proj.weight")?,
                proj_bias: field("self_attn.o_proj.bias")?,
                gate_up: field("mlp.dense_h_to_4h.weight")?,
                gate_up_bias: field("mlp.dense_h_to_4h.bias")?,
                down: field("mlp.dense_4h_to_h.weight")?,
                down_bias: field("mlp.dense_4h_to_h.bias")?,
                norm1: field("input_layernorm.weight")?,
                norm2: field("post_attention_layernorm.weight")?,
                key0_bias: NO_VISION_OFFSET,
                window: 0,
                column_order: 0,
            };
            native.norm1_bias[layer] = field("input_layernorm.bias")?;
            native.norm2_bias[layer] = field("post_attention_layernorm.bias")?;
            native.q_norm[layer] = NO_VISION_OFFSET;
            native.k_norm[layer] = NO_VISION_OFFSET;
        }
        Ok(Self::from_parts(native, reads)?.with_family(cuteafd_loader::media::ImageFamily::Hy4Flash))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::io::Write;

    /// Header-only Hy4 snapshot: real config, synthetic sparse `vit.*` shard.
    pub(crate) fn snapshot() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let cfg: serde_json::Value = serde_json::from_str(include_str!(concat!(env!("CARGO_MANIFEST_DIR"),
            "/../cuteafd-loader/src/families/hy4_flash/fixtures/vision_config.json"))).unwrap();
        std::fs::write(dir.path().join("config.json"), serde_json::to_vec(&cfg).unwrap()).unwrap();
        let contract = VisionContract::from_config(&cfg).unwrap();
        let mut header = serde_json::Map::new();
        let mut end = 0u64;
        for (name, shape) in contract.tensor_shapes() {
            let start = end;
            end += shape.iter().product::<usize>() as u64 * 2;
            header.insert(name, serde_json::json!({"dtype":"BF16","shape":shape,"data_offsets":[start,end]}));
        }
        let bytes = serde_json::to_vec(&header).unwrap();
        let mut f = File::create(dir.path().join("vit.safetensors")).unwrap();
        f.write_all(&(bytes.len() as u64).to_le_bytes()).unwrap();
        f.write_all(&bytes).unwrap();
        f.set_len(8 + bytes.len() as u64 + end).unwrap();
        let mut map = serde_json::Map::new();
        for name in header.keys() { map.insert(name.clone(), "vit.safetensors".into()); }
        map.insert("model.layers.0.unread.weight".into(), "absent-lm.safetensors".into());
        std::fs::write(dir.path().join("model.safetensors.index.json"),
            serde_json::to_vec(&serde_json::json!({"weight_map": map})).unwrap()).unwrap();
        dir
    }

    #[test]
    fn hy4_headers_build_a_valid_generic_tower_without_lm_reads() {
        let dir = snapshot();
        let spec = TowerSpec::from_snapshot(dir.path(), MAX_MERGED_ROWS).unwrap();
        assert_eq!(spec.image_family(), cuteafd_loader::media::ImageFamily::Hy4Flash);
        assert_eq!(spec.reads.len(), 441);
        assert!(spec.reads.iter().all(|r| r.destination % 256 == 0 && r.transform == ReadTransform::Direct));
        let tower = spec.native.tower.unwrap();
        assert_eq!((tower.absolute_kind, tower.projection_order, spec.native.head_dim), (2, 3, 72));
        assert_ne!(spec.native.merger_extra[0], NO_VISION_OFFSET);
        assert_eq!(spec.row_markers(), 1);
        assert!(TowerSpec::hy4_flash(dir.path(), MAX_MERGED_ROWS + 1).is_err());
        let mut wrong = spec.native.clone();
        wrong.tower.as_mut().unwrap().projection_order = 1;
        assert!(wrong.tower.unwrap().validate(&wrong).is_err());
        let mut missing = spec.native.clone();
        missing.merger_extra[0] = NO_VISION_OFFSET;
        assert!(missing.tower.unwrap().validate(&missing).is_err());
        assert_ne!(spec.encoder_id("r", 120), spec.encoder_id("r", 121));
    }
}
