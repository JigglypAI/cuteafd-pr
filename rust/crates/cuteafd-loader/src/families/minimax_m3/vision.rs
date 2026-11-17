//! Bundled MiniMax vision facts from native headers; not an execution qualification.
use super::MiniMaxM3VisionConfig;
use crate::SafetensorsTensorMetadata;
use anyhow::{ensure, Context, Result};
use cuteafd_core::DType;
use serde_json::Value;
use std::collections::BTreeMap;

pub const REFERENCE_REVISION: &str = "a9bf8c76a0115f5e2672bfdfc61d70a2b7936744";
pub const SOURCE_BYTES: u64 = 1_734_145_024;
/// Supported SGLang M3.1 default image rectangle, not the older area policy.
pub const MAX_IMAGE_EDGE: usize = 672;
pub const MAX_IMAGE_PATCHES: usize = 48 * 48;
pub const MAX_IMAGE_TOKENS: usize = MAX_IMAGE_PATCHES / 4;

#[derive(Clone, Debug, PartialEq)]
pub struct VisionContract {
    pub geometry: MiniMaxM3VisionConfig,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorContract {
    pub name: String,
    pub shape: Vec<usize>,
    pub source_dtype: DType,
    pub resident_dtype: DType,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedTensor {
    pub metadata: SafetensorsTensorMetadata,
    pub destination: u64,
    pub resident_bytes: u64,
    pub resident_dtype: DType,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedWeights {
    pub tensors: Vec<PackedTensor>,
    pub source_bytes: u64,
    /// The physical aligned backing, without scratch or the allocation tail.
    pub arena_bytes: u64,
}

impl VisionContract {
    pub fn from_config(config: &Value) -> Result<Self> {
        let geometry = MiniMaxM3VisionConfig::from_hf(config)?
            .context("MiniMax checkpoint has no bundled vision tower")?;
        ensure!(
            (
                geometry.hidden,
                geometry.layers,
                geometry.heads,
                geometry.intermediate,
                geometry.patch,
                geometry.spatial_merge,
                geometry.temporal_patch,
                geometry.projection,
                geometry.projector_hidden,
                geometry.max_segment_frames
            ) == (1280, 32, 16, 5120, 14, 2, 2, 6144, 6144, 4),
            "unsupported MiniMax vision geometry; add an exporter/kernel"
        );
        ensure!(
            geometry.rope_theta == 10000.0 && geometry.layer_norm_eps == 1e-5,
            "unsupported MiniMax vision rotary/normalization numerics"
        );
        ensure!(
            config["text_config"]["hidden_size"] == 6144
                && config["multimodal_projector_bias"] == true
                && config["vision_config"]["num_channels"] == 3,
            "MiniMax vision requires biased 6144-wide projectors and RGB input"
        );
        if let Some(bias) = config.get("patch_merge_bias") {
            ensure!(bias == true, "MiniMax patch merger must retain its biases");
        }
        if let Some(compression) = config.get("img_token_compression_config") {
            ensure!(
                compression == &config["vision_config"]["img_token_compression_config"],
                "conflicting MiniMax image compression configurations"
            );
        }
        Ok(Self { geometry })
    }

    pub fn is_tensor(name: &str) -> bool {
        [
            "vision_tower.",
            "multi_modal_projector.",
            "patch_merge_mlp.",
        ]
        .iter()
        .any(|prefix| name.starts_with(prefix))
    }

    /// Deterministic packing keeps separate Q/K/V adjacent without repacking weights.
    pub fn tensor_contracts(&self) -> Vec<TensorContract> {
        let g = &self.geometry;
        let h = g.hidden;
        let mut tensors = Vec::new();
        let mut put = |name: String, shape: Vec<usize>, source_dtype, resident_dtype| {
            tensors.push(TensorContract {
                name,
                shape,
                source_dtype,
                resident_dtype,
            });
        };
        put(
            "vision_tower.vision_model.embeddings.patch_embedding.weight".into(),
            vec![h, 3, g.temporal_patch, g.patch, g.patch],
            DType::F32,
            DType::Bf16,
        );
        for suffix in ["weight", "bias"] {
            put(
                format!("vision_tower.vision_model.pre_layrnorm.{suffix}"),
                vec![h],
                DType::Bf16,
                DType::Bf16,
            );
        }
        for layer in 0..g.layers {
            let stem = format!("vision_tower.vision_model.encoder.layers.{layer}");
            for suffix in ["weight", "bias"] {
                for part in ["q", "k", "v"] {
                    put(
                        format!("{stem}.self_attn.{part}_proj.{suffix}"),
                        if suffix == "weight" {
                            vec![h, h]
                        } else {
                            vec![h]
                        },
                        DType::Bf16,
                        DType::Bf16,
                    );
                }
            }
            for suffix in ["weight", "bias"] {
                for (name, rows, cols) in [
                    ("self_attn.out_proj", h, h),
                    ("mlp.fc1", g.intermediate, h),
                    ("mlp.fc2", h, g.intermediate),
                ] {
                    put(
                        format!("{stem}.{name}.{suffix}"),
                        if suffix == "weight" {
                            vec![rows, cols]
                        } else {
                            vec![rows]
                        },
                        DType::Bf16,
                        DType::Bf16,
                    );
                }
                for norm in ["layer_norm1", "layer_norm2"] {
                    put(
                        format!("{stem}.{norm}.{suffix}"),
                        vec![h],
                        DType::Bf16,
                        DType::Bf16,
                    );
                }
            }
        }
        for (prefix, input) in [
            ("multi_modal_projector", h),
            ("patch_merge_mlp", 4 * g.projection),
        ] {
            for (layer, rows, cols) in [
                (1, g.projector_hidden, input),
                (2, g.projection, g.projector_hidden),
            ] {
                put(
                    format!("{prefix}.linear_{layer}.weight"),
                    vec![rows, cols],
                    DType::Bf16,
                    DType::Bf16,
                );
                put(
                    format!("{prefix}.linear_{layer}.bias"),
                    vec![rows],
                    DType::Bf16,
                    DType::Bf16,
                );
            }
        }
        tensors
    }

    pub fn pack<'a>(
        &self,
        tensors: impl Iterator<Item = &'a SafetensorsTensorMetadata>,
    ) -> Result<PackedWeights> {
        let mut catalog = BTreeMap::new();
        for tensor in tensors.filter(|t| Self::is_tensor(&t.name)) {
            ensure!(
                catalog.insert(tensor.name.as_str(), tensor).is_none(),
                "duplicate MiniMax vision tensor {}",
                tensor.name
            );
        }
        let expected = self.tensor_contracts();
        ensure!(
            catalog.len() == expected.len(),
            "MiniMax vision requires {} tensors, found {}",
            expected.len(),
            catalog.len()
        );
        let mut packed = PackedWeights {
            tensors: Vec::new(),
            source_bytes: 0,
            arena_bytes: 0,
        };
        for contract in expected {
            let tensor = *catalog
                .get(contract.name.as_str())
                .with_context(|| format!("missing {}", contract.name))?;
            let elements = contract
                .shape
                .iter()
                .try_fold(1u64, |n, &dim| n.checked_mul(dim as u64))
                .context("MiniMax vision tensor shape overflow")?;
            let source_bytes = elements
                .checked_mul(if contract.source_dtype == DType::F32 {
                    4
                } else {
                    2
                })
                .context("MiniMax vision source extent overflow")?;
            ensure!(
                tensor.dtype == contract.source_dtype
                    && tensor.shape == contract.shape
                    && tensor.byte_length == source_bytes,
                "{} requires {:?} {:?}; add an exporter/kernel",
                contract.name,
                contract.source_dtype,
                contract.shape
            );
            ensure!(
                tensor.byte_offset.checked_add(tensor.byte_length).is_some(),
                "{} source extent overflows",
                contract.name
            );
            let destination = packed
                .arena_bytes
                .checked_add(255)
                .context("MiniMax vision alignment overflow")?
                / 256
                * 256;
            let resident_bytes = elements
                .checked_mul(2)
                .context("MiniMax vision resident extent overflow")?;
            packed.arena_bytes = destination
                .checked_add(resident_bytes)
                .context("MiniMax vision arena overflow")?;
            packed.source_bytes = packed
                .source_bytes
                .checked_add(source_bytes)
                .context("MiniMax vision source bytes overflow")?;
            packed.tensors.push(PackedTensor {
                metadata: tensor.clone(),
                destination,
                resident_bytes,
                resident_dtype: contract.resident_dtype,
            });
        }
        packed.arena_bytes = packed
            .arena_bytes
            .checked_add(255)
            .context("MiniMax vision alignment overflow")?
            / 256
            * 256;
        Ok(packed)
    }

    /// SGLang MiniMaxM31 _resize_shape: ceil to28, fit672 rectangle using
    /// integer ratios, then ceil again. Even tiny images have one merge cell.
    pub fn resize_shape(&self, height: u32, width: u32) -> Result<(u32, u32)> {
        resize_shape(height, width)
    }

    /// Vendor 3D rotary uses 26 dimensions per axis, leaving two head dimensions untouched.
    pub fn rotary_axis_dims(&self) -> [usize; 3] {
        let head = self.geometry.hidden / self.geometry.heads;
        [2 * ((head / 3) / 2); 3]
    }

    pub fn rotary_frequencies(&self) -> [Vec<f64>; 3] {
        self.rotary_axis_dims().map(|dim| {
            (0..dim)
                .step_by(2)
                .map(|i| self.geometry.rope_theta.powf(-(i as f64) / dim as f64))
                .collect()
        })
    }

    /// Coordinates follow the processor's 2x2-block-major row order.
    pub fn patch_coordinates(
        &self,
        frames: usize,
        height: usize,
        width: usize,
    ) -> Result<Vec<[usize; 3]>> {
        let merge = self.geometry.spatial_merge;
        ensure!(
            frames == 1
                && height > 0
                && width > 0
                && height <= MAX_IMAGE_EDGE / self.geometry.patch
                && width <= MAX_IMAGE_EDGE / self.geometry.patch
                && height % merge == 0
                && width % merge == 0,
            "invalid MiniMax vision grid"
        );
        let rows = frames
            .checked_mul(height)
            .and_then(|n| n.checked_mul(width))
            .context("MiniMax vision grid overflow")?;
        ensure!(
            rows <= MAX_IMAGE_PATCHES,
            "MiniMax vision grid exceeds native capacity"
        );
        let mut coordinates = Vec::with_capacity(rows);
        for t in 0..frames {
            for y in (0..height).step_by(merge) {
                for x in (0..width).step_by(merge) {
                    for dy in 0..merge {
                        for dx in 0..merge {
                            coordinates.push([t, y + dy, x + dx]);
                        }
                    }
                }
            }
        }
        Ok(coordinates)
    }
}

/// Supported vendor still-image geometry, shared by the tensor contract and processor.
pub fn resize_shape(height: u32, width: u32) -> Result<(u32, u32)> {
    ensure!(height > 0 && width > 0, "MiniMax image dimensions must be positive");
    let factor = 28u64;
    let align = |n: u64| n.div_ceil(factor) * factor;
    let mut h = align(u64::from(height));
    let mut w = align(u64::from(width));
    let max = MAX_IMAGE_EDGE as u64;
    if h > max || w > max {
        let nh = (h * max / h).min(h * max / w);
        let nw = (w * max / h).min(w * max / w);
        h = align(nh.max(factor));
        w = align(nw.max(factor));
    }
    Ok((u32::try_from(h)?, u32::try_from(w)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn config() -> Value {
        super::super::config::tests::release_config()
    }
    fn headers(contract: &VisionContract) -> Vec<SafetensorsTensorMetadata> {
        contract
            .tensor_contracts()
            .into_iter()
            .map(|tensor| SafetensorsTensorMetadata {
                byte_length: tensor.shape.iter().product::<usize>() as u64
                    * if tensor.source_dtype == DType::F32 {
                        4
                    } else {
                        2
                    },
                name: tensor.name,
                shape: tensor.shape,
                dtype: tensor.source_dtype,
                byte_offset: 0,
            })
            .collect()
    }
    #[test]
    fn complete_native_inventory_includes_both_projectors() {
        let contract = VisionContract::from_config(&config()).unwrap();
        let tensors = headers(&contract);
        let packed = contract.pack(tensors.iter()).unwrap();
        assert_eq!(tensors.len(), 523);
        assert_eq!(packed.source_bytes, SOURCE_BYTES);
        assert_eq!(packed.arena_bytes, SOURCE_BYTES - 3_010_560);
        assert_eq!(packed.tensors[0].metadata.dtype, DType::F32);
        assert_eq!(packed.tensors[0].resident_dtype, DType::Bf16);
        assert!(packed
            .tensors
            .windows(2)
            .all(|w| w[0].destination + w[0].resident_bytes <= w[1].destination));
        assert!(packed.tensors.iter().all(|t| t.destination % 256 == 0));
        let qkv = &packed.tensors[3..6];
        assert_eq!(
            qkv[1].destination,
            qkv[0].destination + qkv[0].resident_bytes
        );
        assert_eq!(
            qkv[2].destination,
            qkv[1].destination + qkv[1].resident_bytes
        );
        assert!(contract
            .pack(
                tensors
                    .iter()
                    .filter(|t| !t.name.starts_with("patch_merge_mlp."))
            )
            .is_err());
    }
    #[test]
    fn malformed_native_inventory_is_not_accepted() {
        let contract = VisionContract::from_config(&config()).unwrap();
        let tensors = headers(&contract);
        for fault in 0..6 {
            let mut bad = tensors.clone();
            match fault {
                0 => bad[0].dtype = DType::Bf16,
                1 => bad[0].shape[2] = 1,
                2 => bad[0].byte_length -= 1,
                3 => bad[0].byte_offset = u64::MAX,
                4 => bad.push(bad[0].clone()),
                _ => bad[1].name = "vision_tower.unexpected".into(),
            }
            assert!(contract.pack(bad.iter()).is_err(), "fault {fault}");
        }
    }
    #[test]
    fn rotary_preserves_temporal_axes_and_two_passthrough_channels() {
        let contract = VisionContract::from_config(&config()).unwrap();
        assert_eq!(contract.rotary_axis_dims(), [26; 3]);
        let frequencies = contract.rotary_frequencies();
        assert!(frequencies
            .iter()
            .all(|axis| axis.len() == 13 && axis[0] == 1.0));
        assert!((frequencies[1][1] - 10000f64.powf(-2.0 / 26.0)).abs() < 1e-15);
        assert_eq!(
            contract.patch_coordinates(1, 2, 4).unwrap(),
            vec![
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [0, 0, 2],
                [0, 0, 3],
                [0, 1, 2],
                [0, 1, 3]
            ]
        );
        assert!(contract.patch_coordinates(1, 3, 4).is_err());
        assert!(contract.patch_coordinates(0, 2, 4).is_err());
        assert!(contract.patch_coordinates(5, 2, 4).is_err());
    }
    #[test]
    fn sglang_default_rectangle_is_not_smart_resize() {
        let contract = VisionContract::from_config(&config()).unwrap();
        for (input, expected) in [
            ((900, 500), (672, 392)),
            ((13, 17), (28, 28)),
            ((432, 640), (448, 644)),
            ((1009, 731), (672, 504)),
            ((672, 672), (672, 672)),
        ] {
            assert_eq!(contract.resize_shape(input.0, input.1).unwrap(), expected);
        }
        assert_eq!(MAX_IMAGE_TOKENS, 576);
        assert!(contract.patch_coordinates(1, 50, 48).is_err());
        assert!(contract.patch_coordinates(2, 2, 2).is_err());
        assert!(contract.resize_shape(0, 28).is_err());
    }
    #[test]
    fn incompatible_math_is_refused_before_allocating() {
        for (pointer, value) in [
            ("/vision_config/hidden_act", Value::from("quick_gelu")),
            ("/vision_config/rope_mode", Value::from("2d")),
            ("/multimodal_projector_bias", Value::from(false)),
            ("/text_config/hidden_size", Value::from(7168)),
            ("/vision_config/layer_norm_eps", Value::from(1e-6)),
        ] {
            let mut cfg = config();
            *cfg.pointer_mut(pointer).unwrap() = value;
            assert!(VisionContract::from_config(&cfg).is_err(), "{pointer}");
        }
    }
}
