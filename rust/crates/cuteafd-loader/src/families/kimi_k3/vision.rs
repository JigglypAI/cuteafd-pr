//! Official MoonViT3d header/numerics contract shared by K3 and K3.1.
//! Header acceptance alone does not qualify native image execution.
use crate::SafetensorsTensorMetadata;
use anyhow::{ensure, Context, Result};
use cuteafd_core::DType;
use serde_json::Value;
use std::collections::BTreeMap;

pub const ORACLE_TORCH: &str = "2.13.0";
pub const ORACLE_TORCH_COMMIT: &str = "cf30153c4c131c8164ee7798e5022d810682e2cb";
/// Torch 2.13 uses FP32 opmath epsilon for BF16 RMSNorm(eps=None).
pub const TOWER_RMS_EPS: f32 = 1.1920928955078125e-7;
pub const PROJECTOR_RMS_EPS: f32 = 1e-5;
pub const SOURCE_BYTES: u64 = 894_717_952;
/// Processor limits floor-to14 before its final28px padding. The executed
/// 7069x1833 boundary reaches132x506 patches; full official encoder+projector
/// and dynamic attention passed this shape on SM120 and SM121.
pub const MAX_IMAGE_PATCHES: u32 = 66_792;
pub const MAX_IMAGE_TOKENS: u32 = MAX_IMAGE_PATCHES / 4;
pub const MAX_IMAGE_CANVAS_PIXELS: u64 = MAX_IMAGE_PATCHES as u64 * 14 * 14;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VisionContract {
    pub hidden: usize,
    pub depth: usize,
    pub heads: usize,
    pub attention_width: usize,
    pub intermediate: usize,
    pub patch: usize,
    pub position_grid: usize,
    pub time_positions: usize,
    pub merge: usize,
    pub output: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedTensor {
    pub metadata: SafetensorsTensorMetadata,
    pub destination: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedWeights {
    pub tensors: Vec<PackedTensor>,
    /// Physical aligned backing; excludes scratch, constants and allocation tail.
    pub arena_bytes: u64,
}

impl VisionContract {
    pub fn from_config(config: &Value) -> Result<Self> {
        ensure!(
            matches!(config["model_type"].as_str(), Some("kimi_k3" | "kimi_k31")),
            "MoonViT3d requires Kimi K3 or K3.1"
        );
        let vision = &config["vision_config"];
        for (name, expected) in [
            ("vt_hidden_size", 1024),
            ("vt_num_hidden_layers", 27),
            ("vt_num_attention_heads", 12),
            ("qkv_hidden_size", 1536),
            ("vt_intermediate_size", 4096),
            ("patch_size", 14),
            ("init_pos_emb_height", 64),
            ("init_pos_emb_width", 64),
            ("init_pos_emb_time", 4),
            ("mm_hidden_size", 1024),
            ("text_hidden_size", 7168),
        ] {
            ensure!(
                vision[name].as_u64() == Some(expected),
                "unsupported Kimi vision {name}; needs {expected}; add an exporter/kernel"
            );
        }
        for (name, expected) in [
            ("activation_func", "gelu_pytorch_tanh"),
            ("norm_type", "rmsnorm"),
            ("mlp_type", "mlp2"),
            ("pos_emb_type", "divided_fixed"),
            ("pos_emb_interpolation_mode", "bilinear"),
            ("merge_type", "sd2_tpool"),
            ("mm_projector_type", "patchmergerv2"),
            ("projector_hidden_act", "gelu"),
        ] {
            ensure!(
                vision[name].as_str() == Some(expected),
                "unsupported Kimi vision {name}; needs {expected}"
            );
        }
        for name in ["attn_bias", "linear_bias", "patch_embed_proj_bias"] {
            ensure!(vision[name] == false, "Kimi vision {name} must be false");
        }
        ensure!(
            vision["merge_kernel_size"] == serde_json::json!([2, 2])
                && vision["projector_ln_eps"].as_f64() == Some(1e-5)
                && config["text_config"]["hidden_size"] == 7168,
            "unsupported Kimi projector/pooling geometry or numerics"
        );
        Ok(Self {
            hidden: 1024,
            depth: 27,
            heads: 12,
            attention_width: 1536,
            intermediate: 4096,
            patch: 14,
            position_grid: 64,
            time_positions: 4,
            merge: 2,
            output: 7168,
        })
    }

    pub fn is_tensor(name: &str) -> bool {
        name.starts_with("vision_tower.") || name.starts_with("mm_projector.")
    }

    pub fn tensor_shapes(self) -> Vec<(String, Vec<usize>)> {
        let mut tensors = Vec::new();
        let mut put = |name: String, shape: Vec<usize>| tensors.push((name, shape));
        put(
            "vision_tower.patch_embed.proj.weight".into(),
            vec![self.hidden, 3, self.patch, self.patch],
        );
        put(
            "vision_tower.patch_embed.pos_emb.weight".into(),
            vec![self.position_grid, self.position_grid, self.hidden],
        );
        for layer in 0..self.depth {
            let stem = format!("vision_tower.encoder.blocks.{layer}");
            for norm in ["norm0", "norm1"] {
                put(format!("{stem}.{norm}.weight"), vec![self.hidden]);
            }
            put(
                format!("{stem}.wqkv.weight"),
                vec![3 * self.attention_width, self.hidden],
            );
            put(
                format!("{stem}.wo.weight"),
                vec![self.hidden, self.attention_width],
            );
            put(
                format!("{stem}.mlp.fc0.weight"),
                vec![self.intermediate, self.hidden],
            );
            put(
                format!("{stem}.mlp.fc1.weight"),
                vec![self.hidden, self.intermediate],
            );
        }
        put(
            "vision_tower.encoder.final_layernorm.weight".into(),
            vec![self.hidden],
        );
        let grouped = self.merge * self.merge * self.hidden;
        put("mm_projector.proj.0.weight".into(), vec![grouped, grouped]);
        put(
            "mm_projector.proj.2.weight".into(),
            vec![self.output, grouped],
        );
        put("mm_projector.post_norm.weight".into(), vec![self.output]);
        tensors
    }

    pub fn pack<'a>(
        self,
        tensors: impl Iterator<Item = &'a SafetensorsTensorMetadata>,
    ) -> Result<PackedWeights> {
        let mut catalog = BTreeMap::new();
        for tensor in tensors.filter(|t| Self::is_tensor(&t.name)) {
            ensure!(
                catalog.insert(tensor.name.as_str(), tensor).is_none(),
                "duplicate Kimi vision tensor {}",
                tensor.name
            );
        }
        let expected = self.tensor_shapes();
        ensure!(
            catalog.len() == expected.len(),
            "Kimi vision requires {} tensors, found {}",
            expected.len(),
            catalog.len()
        );
        let mut packed = PackedWeights {
            tensors: Vec::new(),
            arena_bytes: 0,
        };
        for (name, shape) in expected {
            let tensor = *catalog
                .get(name.as_str())
                .with_context(|| format!("missing {name}"))?;
            let bytes = shape
                .iter()
                .try_fold(2u64, |n, &dim| n.checked_mul(dim as u64))
                .context("Kimi vision tensor extent overflow")?;
            ensure!(
                tensor.dtype == DType::Bf16 && tensor.shape == shape && tensor.byte_length == bytes,
                "{name} requires BF16 {shape:?}; add an exporter/kernel"
            );
            ensure!(
                tensor.byte_offset.checked_add(bytes).is_some(),
                "{name} source extent overflows"
            );
            let destination = packed
                .arena_bytes
                .checked_add(255)
                .context("Kimi vision alignment overflow")?
                / 256
                * 256;
            packed.arena_bytes = destination
                .checked_add(bytes)
                .context("Kimi vision arena overflow")?;
            packed.tensors.push(PackedTensor {
                metadata: tensor.clone(),
                destination,
            });
        }
        packed.arena_bytes = packed
            .arena_bytes
            .checked_add(255)
            .context("Kimi vision alignment overflow")?
            / 256
            * 256;
        Ok(packed)
    }

    pub fn output_rows(self, frames: usize, height: usize, width: usize) -> Result<usize> {
        ensure!(
            (1..=self.time_positions).contains(&frames)
                && height > 0
                && width > 0
                && height <= 512
                && width <= 512
                && height % self.merge == 0
                && width % self.merge == 0,
            "invalid Kimi vision grid"
        );
        let rows = height
            .checked_mul(width)
            .context("Kimi vision grid overflow")?;
        ensure!(
            frames.checked_mul(rows).is_some_and(|n| n <= MAX_IMAGE_PATCHES as usize),
            "Kimi vision grid exceeds native capacity"
        );
        Ok(rows / (self.merge * self.merge))
    }

    /// Rows are contiguous image raster patches; merger groups spatially, then means over time.
    pub fn pooled_patch_indices(
        self,
        frames: usize,
        height: usize,
        width: usize,
    ) -> Result<Vec<Vec<[usize; 4]>>> {
        let output_rows = self.output_rows(frames, height, width)?;
        let mut groups = Vec::with_capacity(output_rows);
        for y in (0..height).step_by(self.merge) {
            for x in (0..width).step_by(self.merge) {
                groups.push(
                    (0..frames)
                        .map(|t| {
                            let first = t * height * width + y * width + x;
                            [first, first + 1, first + width, first + width + 1]
                        })
                        .collect(),
                );
            }
        }
        Ok(groups)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn config(model: &str) -> Value {
        serde_json::json!({"model_type":model,"text_config":{"hidden_size":7168},"vision_config":{
            "vt_hidden_size":1024,"vt_num_hidden_layers":27,"vt_num_attention_heads":12,
            "qkv_hidden_size":1536,"vt_intermediate_size":4096,"patch_size":14,
            "init_pos_emb_height":64,"init_pos_emb_width":64,"init_pos_emb_time":4,
            "mm_hidden_size":1024,"text_hidden_size":7168,"activation_func":"gelu_pytorch_tanh",
            "norm_type":"rmsnorm","mlp_type":"mlp2","pos_emb_type":"divided_fixed",
            "pos_emb_interpolation_mode":"bilinear","merge_type":"sd2_tpool",
            "mm_projector_type":"patchmergerv2","projector_hidden_act":"gelu",
            "attn_bias":false,"linear_bias":false,"patch_embed_proj_bias":false,
            "merge_kernel_size":[2,2],"projector_ln_eps":1e-5}})
    }
    fn headers(contract: VisionContract) -> Vec<SafetensorsTensorMetadata> {
        contract
            .tensor_shapes()
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
    fn both_checkpoints_share_the_complete_native_tower_contract() {
        let k3 = VisionContract::from_config(&config("kimi_k3")).unwrap();
        let k31 = VisionContract::from_config(&config("kimi_k31")).unwrap();
        assert_eq!(k3, k31);
        let tensors = headers(k3);
        let packed = k3.pack(tensors.iter()).unwrap();
        assert_eq!(tensors.len(), 168);
        assert_eq!(packed.arena_bytes, SOURCE_BYTES);
        assert_eq!(
            packed
                .tensors
                .iter()
                .map(|t| t.metadata.byte_length)
                .sum::<u64>(),
            SOURCE_BYTES
        );
        assert!(packed.tensors.iter().all(|t| t.destination % 256 == 0));
        assert!(packed
            .tensors
            .windows(2)
            .all(|w| w[0].destination + w[0].metadata.byte_length <= w[1].destination));
        assert_eq!(
            tensors
                .iter()
                .find(|t| t.name.ends_with("blocks.0.wqkv.weight"))
                .unwrap()
                .shape,
            [4608, 1024]
        );
        assert!(k3
            .pack(
                tensors
                    .iter()
                    .filter(|t| !t.name.ends_with("post_norm.weight"))
            )
            .is_err());
    }
    #[test]
    fn header_validation_rejects_missing_biases_unknown_fields_and_wrong_qkv_width() {
        let contract = VisionContract::from_config(&config("kimi_k3")).unwrap();
        let tensors = headers(contract);
        for fault in 0..7 {
            let mut bad = tensors.clone();
            match fault {
                0 => bad[0].dtype = DType::F32,
                1 => {
                    bad.iter_mut()
                        .find(|t| t.name.ends_with("blocks.0.wqkv.weight"))
                        .unwrap()
                        .shape[0] = 1536
                }
                2 => bad[0].byte_length -= 1,
                3 => bad[0].byte_offset = u64::MAX,
                4 => bad.push(bad[0].clone()),
                5 => bad[1].name = "vision_tower.unexpected".into(),
                _ => {
                    let mut extra = bad[0].clone();
                    extra.name = "vision_tower.patch_embed.proj.bias".into();
                    bad.push(extra);
                }
            }
            assert!(contract.pack(bad.iter()).is_err(), "fault {fault}");
        }
    }
    #[test]
    fn temporal_mean_reduces_frames_not_spatial_patch_order() {
        let contract = VisionContract::from_config(&config("kimi_k3")).unwrap();
        assert_eq!(contract.output_rows(1, 2, 4).unwrap(), 2);
        assert_eq!(contract.output_rows(4, 2, 4).unwrap(), 2);
        assert_eq!(
            contract.pooled_patch_indices(2, 2, 4).unwrap(),
            vec![
                vec![[0, 1, 4, 5], [8, 9, 12, 13]],
                vec![[2, 3, 6, 7], [10, 11, 14, 15]]
            ]
        );
        for (t, h, w) in [
            (0, 2, 2),
            (5, 2, 2),
            (1, 3, 2),
            (1, 2, 0),
            (1, 514, 2),
            (4, 256, 256),
        ] {
            assert!(contract.output_rows(t, h, w).is_err());
        }
    }
    #[test]
    fn explicit_rms_epsilon_is_opmath_fp32_not_bf16_or_projector() {
        assert_eq!(TOWER_RMS_EPS.to_bits(), f32::EPSILON.to_bits());
        assert_eq!(TOWER_RMS_EPS, 2f32.powi(-23));
        assert_ne!(TOWER_RMS_EPS, PROJECTOR_RMS_EPS);
        assert_ne!(TOWER_RMS_EPS, 0.0078125);
        assert_eq!(ORACLE_TORCH, "2.13.0");
    }
    #[test]
    fn unsupported_position_activation_and_norm_contracts_are_refused() {
        for (pointer, value) in [
            ("/vision_config/qkv_hidden_size", Value::from(1024)),
            (
                "/vision_config/projector_hidden_act",
                Value::from("gelu_pytorch_tanh"),
            ),
            ("/vision_config/activation_func", Value::from("gelu")),
            (
                "/vision_config/pos_emb_interpolation_mode",
                Value::from("bicubic"),
            ),
            ("/vision_config/projector_ln_eps", Value::from(1e-6)),
            ("/vision_config/linear_bias", Value::from(true)),
        ] {
            let mut cfg = config("kimi_k3");
            *cfg.pointer_mut(pointer).unwrap() = value;
            assert!(VisionContract::from_config(&cfg).is_err(), "{pointer}");
        }
    }
}
