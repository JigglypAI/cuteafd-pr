//! Checkpoint-native Perception-Encoder layouts, shared by Inkling 2 and Step-5.
//! This is a header-only contract; accepting a layout does not qualify serving.
use crate::SafetensorsTensorMetadata;
use cuteafd_core::DType;
use serde_json::Value;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PerceptionKind {
    Inkling2,
    Step5,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PerceptionGeometry {
    pub kind: PerceptionKind,
    pub hidden: usize,
    pub depth: usize,
    pub heads: usize,
    pub intermediate: usize,
    pub patch: usize,
    pub position_grid: usize,
    pub output: usize,
}

impl PerceptionGeometry {
    pub fn from_config(cfg: &Value) -> Result<Self, String> {
        let v = &cfg["vision_config"];
        let (kind, fields, geometry): (_, Vec<(&str, u64)>, _) = match cfg["model_type"].as_str() {
            Some("inkling2") => (
                PerceptionKind::Inkling2,
                vec![
                    ("hidden_size", 1024),
                    ("num_hidden_layers", 23),
                    ("num_attention_heads", 16),
                    ("native_patch_size", 20),
                    ("temporal_patch_size", 2),
                    ("adapter_merge_size", 2),
                    ("projection_hidden_dim", 4096),
                    ("posemb_grid_size", 32),
                ],
                (1024, 23, 16, 4096, 20, 32, 4608),
            ),
            Some("step5_vl") => (
                PerceptionKind::Step5,
                vec![
                    ("width", 1536),
                    ("layers", 47),
                    ("heads", 16),
                    ("image_size", 728),
                    ("patch_size", 14),
                ],
                (1536, 47, 16, 8960, 14, 52, 4096),
            ),
            other => {
                return Err(format!(
                    "unsupported Perception-Encoder model_type {other:?}"
                ))
            }
        };
        for (field, expected) in fields {
            if v[field].as_u64() != Some(expected) {
                return Err(format!("Perception-Encoder vision_config.{field} needs {expected}; add an exporter/kernel"));
            }
        }
        let (hidden, depth, heads, intermediate, patch, position_grid, output) = geometry;
        match kind {
            PerceptionKind::Inkling2 => {
                if v["mlp_ratio"].as_f64() != Some(4.0)
                    || v["projector_bias"] != false
                    || v["layer_norm_eps"].as_f64() != Some(1e-5)
                    || cfg["text_config"]["hidden_size"].as_u64() != Some(output as u64)
                {
                    return Err("unsupported Inkling 2 PE normalization/projector geometry".into());
                }
            }
            PerceptionKind::Step5 => {
                // Absent keys use the vendor configuration defaults, not a
                // different interpretation of the checkpoint's learned positions.
                let option =
                    |name: &str, default| v.get(name).map_or(Some(default), Value::as_bool);
                if v["model_type"] != "perception_encoder"
                    || v["hidden_act"] != "quick_gelu"
                    || option("use_cls_token", false) != Some(false)
                    || option("use_ln_post", false) != Some(false)
                    || option("use_abs_posemb", true) != Some(true)
                    || option("use_rope2d", true) != Some(true)
                    || option("use_ln_pre", true) != Some(true)
                    || v["pool_type"] != "none"
                    || v.get("mlp_ratio")
                        .is_some_and(|x| x.as_f64() != Some(8960.0 / 1536.0))
                    || cfg["text_config"]["hidden_size"].as_u64() != Some(output as u64)
                    || cfg["image_token_len"] != 169
                    || cfg["image_token_id"] != 128001
                {
                    return Err(
                        "unsupported Step-5 PE normalization/positions/projector geometry".into(),
                    );
                }
            }
        }
        Ok(Self {
            kind,
            hidden,
            depth,
            heads,
            intermediate,
            patch,
            position_grid,
            output,
        })
    }

    pub fn is_tensor(self, name: &str) -> bool {
        match self.kind {
            PerceptionKind::Inkling2 => name.starts_with("model.visual."),
            PerceptionKind::Step5 => {
                name.starts_with("vision_model.") || name.starts_with("vit_large_projector.")
            }
        }
    }

    /// Packing order concatenates separate Q/K/V with no converted residency.
    pub fn tensor_shapes(self) -> Vec<(String, Vec<usize>)> {
        let mut out = Vec::new();
        let mut put = |name: &str, shape: &[usize]| out.push((name.to_owned(), shape.to_vec()));
        let h = self.hidden;
        let p = match self.kind {
            PerceptionKind::Inkling2 => {
                put(
                    "model.visual.backbone.embeddings.patch_embedding.weight",
                    &[h, 3 * self.patch * self.patch],
                );
                put(
                    "model.visual.backbone.embeddings.position_embedding",
                    &[self.position_grid * self.position_grid, h],
                );
                put("model.visual.backbone.embeddings.class_embedding", &[h]);
                put(
                    "model.visual.backbone.embeddings.cls_position_embedding",
                    &[h],
                );
                "model.visual.backbone"
            }
            PerceptionKind::Step5 => {
                put("vision_model.conv1.weight", &[h, 3, self.patch, self.patch]);
                put(
                    "vision_model.positional_embedding",
                    &[self.position_grid * self.position_grid, h],
                );
                "vision_model"
            }
        };
        put(&format!("{p}.ln_pre.weight"), &[h]);
        put(&format!("{p}.ln_pre.bias"), &[h]);
        for layer in 0..self.depth {
            let block = match self.kind {
                PerceptionKind::Inkling2 => format!("{p}.layers.{layer}"),
                PerceptionKind::Step5 => format!("{p}.transformer.resblocks.{layer}"),
            };
            let mut field =
                |suffix: &str, shape: &[usize]| put(&format!("{block}.{suffix}"), shape);
            let attention = match self.kind {
                PerceptionKind::Inkling2 => {
                    for qkv in ["q", "k", "v"] {
                        field(&format!("self_attn.{qkv}_proj.weight"), &[h, h]);
                    }
                    for qkv in ["q", "k", "v"] {
                        field(&format!("self_attn.{qkv}_proj.bias"), &[h]);
                    }
                    "self_attn"
                }
                PerceptionKind::Step5 => {
                    field("attn.in_proj_weight", &[3 * h, h]);
                    field("attn.in_proj_bias", &[3 * h]);
                    "attn"
                }
            };
            field(&format!("{attention}.out_proj.weight"), &[h, h]);
            field(&format!("{attention}.out_proj.bias"), &[h]);
            field("mlp.c_fc.weight", &[self.intermediate, h]);
            field("mlp.c_fc.bias", &[self.intermediate]);
            field("mlp.c_proj.weight", &[h, self.intermediate]);
            field("mlp.c_proj.bias", &[h]);
            for name in [
                "ln_1.weight",
                "ln_1.bias",
                "ln_2.weight",
                "ln_2.bias",
                "ls_1.gamma",
                "ls_2.gamma",
            ] {
                field(name, &[h]);
            }
        }
        match self.kind {
            PerceptionKind::Inkling2 => {
                put("model.visual.proj_in.weight", &[4096, 4 * h]);
                put("model.visual.proj_out.weight", &[self.output, 4096]);
            }
            PerceptionKind::Step5 => {
                put("vision_model.vit_downsampler1.weight", &[2 * h, h, 3, 3]);
                put("vision_model.vit_downsampler1.bias", &[2 * h]);
                put(
                    "vision_model.vit_downsampler2.weight",
                    &[4 * h, 2 * h, 3, 3],
                );
                put("vision_model.vit_downsampler2.bias", &[4 * h]);
                put("vit_large_projector.weight", &[self.output, 4 * h]);
            }
        }
        out
    }

    /// Exact native scratch_perception backing, with runtime image extents
    /// bounded by the admitted merged-row capacity and official Step grids.
    pub fn scratch_bytes(self, tokens: u64) -> Result<u64, String> {
        let inkling = self.kind == PerceptionKind::Inkling2;
        if !(1..=4096).contains(&tokens) || (!inkling && tokens < 81) {
            return Err("PE capacity must be 1..=4096, or 81..=4096 for Step".into());
        }
        let n = if inkling {
            (tokens * 4 + 1).min(16384)
        } else if tokens >= 169 {
            2704
        } else {
            1296
        };
        let h = self.hidden as u64;
        let mut sizes = vec![
            n * h * 2,
            n * h * 2,
            n * h * 2,
            n * 3 * (self.patch as u64).pow(2) * 2,
            n * 3 * h * 2,
        ];
        sizes.extend([16384 * h * 2; 4]);
        sizes.extend([
            n * h * 2,
            n * self.intermediate as u64 * 2,
            tokens * 4096 * 2,
            tokens * self.output as u64 * 2,
            tokens * 4 * h * 2,
            2 * 4,
            16384 * 16 * 4,
            768 * 4,
        ]);
        sizes.push(if inkling {
            n * 20 * 20 * 3
        } else {
            3024 * 3024 * 3
        });
        if !inkling {
            sizes.extend([
                3024 * 3024 * 3 * 4,
                3024 * 728 * 3 * 4,
                728 * 728 * 3 * 4,
                728 * 4,
                728 * 4,
                728 * 4,
                728 * 4,
                728 * 64 * 4,
                728 * 64 * 4,
                (n / 4 * 1536 * 9 * 2).max(n / 16 * 3072 * 9 * 2),
                n / 4 * 3072 * 2,
                n / 16 * 6144 * 2,
            ]);
        }
        Ok(sizes
            .into_iter()
            .map(|bytes| bytes.div_ceil(256) * 256)
            .sum())
    }

    /// Reference constants: Inkling uses Python float64 power then F32, while
    /// Step builds CPU Torch F32 1/pow(10000, arange(0,48,2)/48). These immutable
    /// bit patterns are checked against the pinned vendor builders on both GPUs.
    pub fn inv_freq(self) -> Vec<f32> {
        self.kind.inv_freq()
    }

    /// One aligned BF16 arena plus its immutable FP32 frequencies; return
    /// physical bytes, never payload plus backing.
    pub fn resident_weights<'a>(
        self,
        tensors: impl Iterator<Item = &'a SafetensorsTensorMetadata>,
    ) -> Result<u64, String> {
        let mut catalog = BTreeMap::new();
        for tensor in tensors.filter(|t| self.is_tensor(&t.name)) {
            if catalog.insert(tensor.name.as_str(), tensor).is_some() {
                return Err(format!("duplicate PE tensor {}", tensor.name));
            }
        }
        let expected = self.tensor_shapes();
        if catalog.len() != expected.len() {
            return Err(format!(
                "PE tower needs {} tensors, found {}",
                expected.len(),
                catalog.len()
            ));
        }
        let mut bytes = 0u64;
        for (name, shape) in expected {
            let tensor = catalog
                .get(name.as_str())
                .ok_or_else(|| format!("missing {name}"))?;
            let extent = shape.iter().product::<usize>() as u64 * 2;
            if tensor.dtype != DType::Bf16 || tensor.shape != shape || tensor.byte_length != extent
            {
                return Err(format!(
                    "{name} needs BF16 {shape:?}; add an exporter/kernel"
                ));
            }
            bytes = bytes.div_ceil(256) * 256 + extent;
        }
        Ok(bytes.div_ceil(256) * 256 + self.inv_freq().len() as u64 * 4)
    }
}

impl PerceptionKind {
    pub fn inv_freq(self) -> Vec<f32> {
        let bits: &[u32] = match self {
            PerceptionKind::Inkling2 => &[
                0x3f800000, 0x3f0ff59a, 0x3ea1e89b, 0x3e361887, 0x3dcccccd, 0x3d6655c3, 0x3d0186e2,
                0x3c91ad39, 0x3c23d70a, 0x3bb8449c, 0x3b4f3e37, 0x3ae91528, 0x3a83126f, 0x3a136a16,
                0x39a5cb5f, 0x393a7753,
            ],
            PerceptionKind::Step5 => &[
                0x3f800000, 0x3f2e6928, 0x3eeda63c, 0x3ea1e89b, 0x3e5c9d35, 0x3e164d7a, 0x3dcccccd,
                0x3d8b8754, 0x3d3e1e95, 0x3d0186e3, 0x3cb07dc7, 0x3c707bf6, 0x3c23d70a, 0x3bdf3ee9,
                0x3b98187a, 0x3b4f3e38, 0x3b0d3169, 0x3ac0632e, 0x3a83126f, 0x3a3298ba, 0x39f35a5c,
                0x39a5cb60, 0x3961e8a7, 0x3919e8f2,
            ],
        };
        bits.iter().copied().map(f32::from_bits).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn geometry(kind: PerceptionKind) -> PerceptionGeometry {
        match kind {
            PerceptionKind::Inkling2 => PerceptionGeometry {
                kind,
                hidden: 1024,
                depth: 23,
                heads: 16,
                intermediate: 4096,
                patch: 20,
                position_grid: 32,
                output: 4608,
            },
            PerceptionKind::Step5 => PerceptionGeometry {
                kind,
                hidden: 1536,
                depth: 47,
                heads: 16,
                intermediate: 8960,
                patch: 14,
                position_grid: 52,
                output: 4096,
            },
        }
    }
    fn headers(g: PerceptionGeometry) -> Vec<SafetensorsTensorMetadata> {
        g.tensor_shapes()
            .into_iter()
            .map(|(name, shape)| SafetensorsTensorMetadata {
                byte_length: shape.iter().product::<usize>() as u64 * 2,
                name,
                shape,
                dtype: DType::Bf16,
                byte_offset: 0,
            })
            .collect()
    }
    #[test]
    fn checkpoint_native_layouts_have_complete_unique_inventory() {
        for (kind, count) in [
            (PerceptionKind::Inkling2, 422),
            (PerceptionKind::Step5, 667),
        ] {
            let g = geometry(kind);
            let tensors = headers(g);
            assert_eq!(tensors.len(), count);
            let checkpoint_bytes = match kind {
                PerceptionKind::Inkling2 => 655_386_624,
                PerceptionKind::Step5 => 3_962_029_568,
            };
            assert_eq!(
                g.resident_weights(tensors.iter()).unwrap(),
                checkpoint_bytes + g.inv_freq().len() as u64 * 4
            );
            assert_eq!(
                checkpoint_bytes,
                tensors.iter().map(|t| t.byte_length).sum::<u64>()
            );
            assert!(g.resident_weights(tensors.iter().skip(1)).is_err());
            let mut wrong = tensors.clone();
            wrong[0].dtype = DType::F32;
            assert!(g.resident_weights(wrong.iter()).is_err());
            let mut wrong = tensors.clone();
            wrong[0].shape[0] += 1;
            assert!(g.resident_weights(wrong.iter()).is_err());
            let mut wrong = tensors.clone();
            wrong.push(wrong[0].clone());
            assert!(g.resident_weights(wrong.iter()).is_err());
            let mut wrong = tensors.clone();
            let mut extra = wrong[0].clone();
            extra.name = match kind {
                PerceptionKind::Inkling2 => "model.visual.extra",
                PerceptionKind::Step5 => "vision_model.extra",
            }
            .into();
            wrong.push(extra);
            assert!(g.resident_weights(wrong.iter()).is_err());
        }
    }
    #[test]
    fn scratch_capacity_includes_cls_and_both_official_step_grids() {
        let inkling = geometry(PerceptionKind::Inkling2);
        let step = geometry(PerceptionKind::Step5);
        for g in [inkling, step] {
            assert!(g.scratch_bytes(0).is_err());
            assert!(g.scratch_bytes(4097).is_err());
            assert_eq!(g.scratch_bytes(4096).unwrap() % 256, 0);
        }
        assert!(step.scratch_bytes(80).is_err());
        assert!(step.scratch_bytes(82).unwrap() > step.scratch_bytes(81).unwrap());
        assert!(step.scratch_bytes(169).unwrap() > step.scratch_bytes(168).unwrap());
        assert!(inkling.scratch_bytes(2).unwrap() > inkling.scratch_bytes(1).unwrap());
    }
    #[test]
    fn step5_has_learned_positions_and_rotary_without_class_token() {
        let mut cfg = serde_json::json!({"model_type":"step5_vl", "text_config":{"hidden_size":4096},
            "image_token_len":169,"image_token_id":128001, "vision_config":{"model_type":"perception_encoder",
            "width":1536,"layers":47,"heads":16,"image_size":728,"patch_size":14,"pool_type":"none",
            "use_cls_token":false,"use_ln_post":false,"hidden_act":"quick_gelu"}});
        assert_eq!(
            PerceptionGeometry::from_config(&cfg).unwrap(),
            geometry(PerceptionKind::Step5)
        );
        for key in ["use_rope2d", "use_abs_posemb", "use_ln_pre"] {
            cfg["vision_config"][key] = false.into();
            assert!(PerceptionGeometry::from_config(&cfg).is_err());
            cfg["vision_config"].as_object_mut().unwrap().remove(key);
        }
    }
    #[test]
    fn inkling_preserves_class_position_and_bias_free_projector() {
        let mut cfg = serde_json::json!({"model_type":"inkling2","text_config":{"hidden_size":4608},"vision_config":{
            "hidden_size":1024,"num_hidden_layers":23,"num_attention_heads":16,"native_patch_size":20,
            "temporal_patch_size":2,"adapter_merge_size":2,"projection_hidden_dim":4096,"posemb_grid_size":32,
            "mlp_ratio":4.0,"projector_bias":false,"layer_norm_eps":1e-5}});
        assert_eq!(
            PerceptionGeometry::from_config(&cfg).unwrap(),
            geometry(PerceptionKind::Inkling2)
        );
        cfg["vision_config"]["projector_bias"] = true.into();
        assert!(PerceptionGeometry::from_config(&cfg).is_err());
    }
}
