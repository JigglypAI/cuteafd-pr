//! Bundled Hy4-flash ViT (`vit.*`) facts from the checkpoint's own config and
//! headers, plus the text-side interleaved M-RoPE coordinates of image spans.
//!
//! Reference: Tencent's serving tower, sglang@ccb8d9cc `models/hunyuan_v4_vl.py`
//! (cross-checked against Transformers PR 50903 `HYV4FlashVisionTransformer`):
//! SigLIP2 body with biased conv patch-16 embedding, a learned 128x128 position
//! grid resized by `scale_factor=(h+0.1)/128` bilinear, 27 pre-LN blocks
//! (LayerNorm + bias, separate biased Q/K/V/O, 16 heads x 72, exact GELU MLP),
//! then RMSNorm, 2x2/2 conv (1152->4608) + GELU + 1x1 conv (4608->3072), and the
//! checkpoint's `image_newline` appended after every merged row.
use crate::SafetensorsTensorMetadata;
use anyhow::{bail, ensure, Context, Result};
use cuteafd_core::DType;
use serde_json::Value;
use std::collections::BTreeMap;

pub const REFERENCE_REVISION: &str = "ccb8d9ccaa309f71dda42e37517a7b0ee6afae8d";
/// SGLang `smart_resize`: 32-pixel merged cells, 64x64..2048x2048-pixel budgets.
pub const MIN_PIXELS: u64 = 4096;
pub const MAX_PIXELS: u64 = 4_194_304;
/// At most 128x128 patches: 64x64 merged rows plus 64 newline rows.
pub const MAX_MERGED_ROWS: usize = 4096;
pub const MAX_IMAGE_TOKENS: usize = 64 * 65;
pub const PATCH: usize = 16;
pub const POSITION_EDGE: usize = 128;
pub const HEAD_DIM: usize = 72;

#[derive(Clone, Debug, PartialEq)]
pub struct VisionContract {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub intermediate: usize,
    pub output: usize,
    pub merger_hidden: usize,
    pub eps: f32,
    pub image_token: u32,
    pub image_start: u32,
    pub image_end: u32,
    /// Interleaved M-RoPE sections (T, H, W) over the 32 text rope pairs.
    pub mrope_section: [usize; 3],
}

impl VisionContract {
    pub fn from_config(config: &Value) -> Result<Self> {
        let v = config.get("vision_config").filter(|v| v.is_object())
            .context("Hy4-flash checkpoint has no bundled vision_config")?;
        let int = |key: &str| v[key].as_u64().with_context(|| format!("Hy4 vision_config lacks {key}"));
        let fixed = [
            ("hidden_size", 1152), ("num_hidden_layers", 27), ("num_attention_heads", 16),
            ("num_key_value_heads", 16), ("intermediate_size", 4304), ("out_hidden_size", 3072),
            ("patch_size", 16), ("spatial_merge_size", 2), ("temporal_patch_size", 1),
            ("spatial_patch_size", 1), ("num_channels", 3), ("max_image_size", 2048),
            ("cat_extra_token", 0), ("learnable_mlp_pooling_size", 0),
        ];
        for (key, want) in fixed {
            ensure!(int(key)? == want, "Hy4 vision_config.{key}={}, need {want}; add a tower exporter/kernel", v[key]);
        }
        let flag = |key: &str, want: bool| {
            // Absent keys take the serving config defaults (SGLang HYV4VLVisionConfig).
            let value = v.get(key).and_then(Value::as_bool);
            ensure!(value.is_none_or(|value| value == want), "Hy4 vision_config.{key} must be {want}");
            Ok(())
        };
        flag("use_2d_rope", false)?;
        flag("use_image_newline_token", true)?;
        flag("perceive_post_norm", false)?;
        flag("vision_full_attention", false)?;
        ensure!(v["hidden_act"] == "gelu" && v["interpolate_mode"] == "bilinear"
            && v["projector_type"] == "rmsnorm_mlp_adapter",
            "unsupported Hy4 vision activation/interpolation/projector");
        ensure!(v.get("perceive_intermediate_size").is_none_or(Value::is_null),
            "Hy4 vision merger hidden width must be 4 x hidden");
        let eps = v["rms_norm_eps"].as_f64().context("Hy4 vision rms_norm_eps")?;
        ensure!(eps == 1e-5, "Hy4 vision LayerNorm/RMSNorm epsilon {eps}, need 1e-5");
        let text = &config["text_config"];
        ensure!(text["hidden_size"] == 3072, "Hy4 vision output width differs from text hidden_size");
        let id = |key: &str| config[key].as_u64().and_then(|v| u32::try_from(v).ok())
            .with_context(|| format!("Hy4 config lacks {key}"));
        let (image_token, image_start, image_end) = (id("image_token_id")?, id("image_start_token_id")?, id("image_end_token_id")?);
        ensure!(id("vision_start_token_id")? == image_start, "Hy4 vision/image start ids differ");
        let mut sections = None;
        for (kind, rope) in text["rope_parameters"].as_object().context("Hy4 text rope_parameters")? {
            ensure!(rope["mrope_interleaved"] == true, "Hy4 {kind} rope must be interleaved M-RoPE");
            let s: Vec<usize> = serde_json::from_value(rope["mrope_section"].clone())
                .with_context(|| format!("Hy4 {kind} mrope_section"))?;
            let s: [usize; 3] = s.try_into().map_err(|_| anyhow::anyhow!("Hy4 mrope_section needs three axes"))?;
            ensure!(sections.is_none_or(|prior| prior == s), "Hy4 rope tables use different mrope sections");
            sections = Some(s);
        }
        let mrope_section = sections.context("Hy4 text rope has no mrope_section")?;
        ensure!(mrope_section == [12, 10, 10] && text["qk_rope_head_dim"] == 64 && text["swa_qk_rope_head_dim"] == 64,
            "unsupported Hy4 mrope_section {mrope_section:?}; export an interleaved M-RoPE producer for it");
        Ok(Self { hidden: 1152, layers: 27, heads: 16, intermediate: 4304, output: 3072, merger_hidden: 4608,
            eps: eps as f32, image_token, image_start, image_end, mrope_section })
    }

    pub fn is_tensor(name: &str) -> bool { name.starts_with("vit.") }

    /// Header order, also the packed arena order. All tensors stay BF16 resident.
    pub fn tensor_shapes(&self) -> Vec<(String, Vec<usize>)> {
        let (h, i, o, m) = (self.hidden, self.intermediate, self.output, self.merger_hidden);
        let mut out = vec![
            ("vit.embeddings.patch_embedding.weight".to_string(), vec![h, 3, PATCH, PATCH]),
            ("vit.embeddings.patch_embedding.bias".to_string(), vec![h]),
            ("vit.embeddings.position_embedding.weight".to_string(), vec![POSITION_EDGE * POSITION_EDGE + 1, h]),
        ];
        for layer in 0..self.layers {
            let stem = format!("vit.layers.{layer}");
            // Q, K and V stay adjacent so one fused-QKV projection reads them in place.
            for part in ["q", "k", "v"] {
                out.push((format!("{stem}.self_attn.{part}_proj.weight"), vec![h, h]));
            }
            for part in ["q", "k", "v"] {
                out.push((format!("{stem}.self_attn.{part}_proj.bias"), vec![h]));
            }
            out.push((format!("{stem}.self_attn.o_proj.weight"), vec![h, h]));
            out.push((format!("{stem}.self_attn.o_proj.bias"), vec![h]));
            out.push((format!("{stem}.mlp.dense_h_to_4h.weight"), vec![i, h]));
            out.push((format!("{stem}.mlp.dense_h_to_4h.bias"), vec![i]));
            out.push((format!("{stem}.mlp.dense_4h_to_h.weight"), vec![h, i]));
            out.push((format!("{stem}.mlp.dense_4h_to_h.bias"), vec![h]));
            for norm in ["input_layernorm", "post_attention_layernorm"] {
                out.push((format!("{stem}.{norm}.weight"), vec![h]));
                out.push((format!("{stem}.{norm}.bias"), vec![h]));
            }
        }
        out.push(("vit.perceive.norm.weight".into(), vec![h]));
        out.push(("vit.perceive.proj.0.weight".into(), vec![m, h, 2, 2]));
        out.push(("vit.perceive.proj.0.bias".into(), vec![m]));
        out.push(("vit.perceive.proj.2.weight".into(), vec![o, m, 1, 1]));
        out.push(("vit.perceive.proj.2.bias".into(), vec![o]));
        out.push(("vit.perceive.image_newline".into(), vec![o]));
        out
    }

    /// Validate every `vit.*` header and lay them out 256-byte aligned. The
    /// separate Q/K/V weights (and biases) must land contiguously; the tightly
    /// aligned 1152x1152 BF16 matrices make that hold without any repacking.
    pub fn pack<'a>(&self, tensors: impl Iterator<Item = &'a SafetensorsTensorMetadata>) -> Result<PackedVision> {
        let mut catalog = BTreeMap::new();
        for tensor in tensors.filter(|t| Self::is_tensor(&t.name)) {
            ensure!(catalog.insert(tensor.name.as_str(), tensor).is_none(), "duplicate Hy4 vision tensor {}", tensor.name);
        }
        let expected = self.tensor_shapes();
        if catalog.len() != expected.len() {
            let known: std::collections::BTreeSet<_> = expected.iter().map(|(n, _)| n.as_str()).collect();
            let extra: Vec<_> = catalog.keys().filter(|n| !known.contains(*n)).collect();
            bail!("Hy4 vision needs {} vit.* tensors, found {} (unexpected {extra:?}); add a tower exporter/kernel",
                expected.len(), catalog.len());
        }
        let mut packed = PackedVision { tensors: Vec::with_capacity(expected.len()), arena_bytes: 0 };
        for (name, shape) in expected {
            let tensor = *catalog.get(name.as_str()).with_context(|| format!("missing Hy4 vision tensor {name}"))?;
            let bytes = shape.iter().try_fold(2u64, |n, &d| n.checked_mul(d as u64)).context("Hy4 vision extent overflow")?;
            ensure!(tensor.dtype == DType::Bf16 && tensor.shape == shape && tensor.byte_length == bytes
                && tensor.byte_offset.checked_add(bytes).is_some(),
                "{name}: {:?} {:?}; need BF16 {shape:?}; add a tower exporter/kernel", tensor.dtype, tensor.shape);
            let destination = packed.arena_bytes.checked_add(255).context("Hy4 vision alignment overflow")? / 256 * 256;
            packed.arena_bytes = destination.checked_add(bytes).context("Hy4 vision arena overflow")?;
            packed.tensors.push(PackedTensor { metadata: tensor.clone(), destination });
        }
        for layer in 0..self.layers {
            for kind in ["weight", "bias"] {
                let at = |part: &str| packed.offset(&format!("vit.layers.{layer}.self_attn.{part}_proj.{kind}"));
                let width = if kind == "weight" { (self.hidden * self.hidden * 2) as u64 } else { (self.hidden * 2) as u64 };
                let (q, k, v) = (at("q")?, at("k")?, at("v")?);
                ensure!(k == q + width && v == k + width, "Hy4 layer {layer} Q/K/V {kind}s are not contiguous");
            }
        }
        Ok(packed)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedTensor {
    pub metadata: SafetensorsTensorMetadata,
    pub destination: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedVision {
    pub tensors: Vec<PackedTensor>,
    pub arena_bytes: u64,
}
impl PackedVision {
    pub fn offset(&self, name: &str) -> Result<u64> {
        self.tensors.iter().find(|t| t.metadata.name == name).map(|t| t.destination)
            .with_context(|| format!("missing packed Hy4 vision tensor {name}"))
    }
}

/// Merged LM span of a patch grid: `h/2` rows of `w/2` features plus one newline.
pub fn span_rows(grid: [usize; 2]) -> Option<usize> {
    let [h, w] = grid;
    (h >= 2 && w >= 2 && h % 2 == 0 && w % 2 == 0 && h <= 128 && w <= 128)
        .then(|| (h / 2) * (w / 2 + 1))
}

/// SGLang `smart_resize` (factor 32, 4096..4194304 pixels; Python round = ties-even).
pub fn smart_resize(height: u32, width: u32) -> Result<(u32, u32)> {
    let (h, w, f) = (f64::from(height), f64::from(width), 32.0f64);
    ensure!(height > 0 && width > 0, "image dimensions must be positive");
    ensure!(h.max(w) / h.min(w) <= 200.0, "Image aspect ratio must not exceed 200");
    let (mut rh, mut rw) = ((h / f).round_ties_even() * f, (w / f).round_ties_even() * f);
    if rh * rw > MAX_PIXELS as f64 {
        let scale = (h * w / MAX_PIXELS as f64).sqrt();
        rh = f.max((h / scale / f).floor() * f);
        rw = f.max((w / scale / f).floor() * f);
    } else if rh * rw < MIN_PIXELS as f64 {
        let scale = (MIN_PIXELS as f64 / (h * w)).sqrt();
        rh = (h * scale / f).ceil() * f;
        rw = (w * scale / f).ceil() * f;
    }
    let (rh, rw) = (rh.max(f), rw.max(f));
    ensure!(rh * rw <= MAX_PIXELS as f64, "Image aspect ratio and patch rounding exceed max_pixels");
    Ok((rh as u32, rw as u32))
}

/// One image span in native token rows: `start` is its first placeholder row,
/// `grid` its patch grid (h, w) before the 2x2 merge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageSpan {
    pub start: usize,
    pub grid: [usize; 2],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Span {
    start: usize,
    end: usize,
    base: i64,
    columns: usize,
    delta_after: i64,
}

/// Interleaved 3D rope coordinates (SGLang `get_hy_v4_vl_mrope_positions`):
/// text rows carry equal axes; image row `r`, column `c` (including the newline
/// column) carries `[s, s+r, s+c]`, and the following text resumes at
/// `s + max(rows, cols)`. Slot positions (KV, SWA ring, indexer limits and
/// group ids) never use these values.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RopePositions {
    spans: Vec<Span>,
}

impl RopePositions {
    pub fn new(tokens: &[u32], images: &[ImageSpan], contract: &VisionContract) -> Result<Self> {
        let mut spans = Vec::with_capacity(images.len());
        let (mut end, mut delta) = (0usize, 0i64);
        for (i, image) in images.iter().enumerate() {
            let [h, w] = image.grid;
            let len = span_rows(image.grid).with_context(|| format!("Hy4 image {i} grid {h}x{w} is invalid"))?;
            let (rows, columns) = (h / 2, w / 2 + 1);
            let next = image.start.checked_add(len).context("Hy4 rope span overflow")?;
            ensure!(image.start >= end.max(1) && next < tokens.len()
                && tokens[image.start - 1] == contract.image_start && tokens[next] == contract.image_end
                && tokens[image.start..next].iter().all(|&t| t == contract.image_token),
                "Hy4 image {i} rows/markers differ from its grid");
            ensure!(!tokens[end..image.start].contains(&contract.image_token), "Hy4 image placeholder outside its span");
            let base = i64::try_from(image.start)? + delta;
            let extent = i64::try_from(rows.max(columns))?;
            delta += extent - i64::try_from(len)?;
            spans.push(Span { start: image.start, end: next, base, columns, delta_after: delta });
            end = next;
        }
        ensure!(!tokens[end..].contains(&contract.image_token), "Hy4 image placeholder outside its span");
        let positions = Self { spans };
        if let Some(last) = tokens.len().checked_sub(1) { positions.at(last)?; }
        Ok(positions)
    }

    /// Coordinates of a prompt, decode or verify row (any row beyond the prompt
    /// continues the final text offset).
    pub fn at(&self, row: usize) -> Result<[i64; 3]> {
        let n = self.spans.partition_point(|span| span.start <= row);
        if let Some(span) = n.checked_sub(1).map(|i| &self.spans[i]) {
            if row < span.end {
                let offset = row - span.start;
                let (r, c) = (i64::try_from(offset / span.columns)?, i64::try_from(offset % span.columns)?);
                return Ok([span.base, span.base + r, span.base + c]);
            }
        }
        let delta = n.checked_sub(1).map_or(0, |i| self.spans[i].delta_after);
        let position = i64::try_from(row)?.checked_add(delta).context("Hy4 rope position overflow")?;
        ensure!(position >= 0 && position < i64::from(i32::MAX), "Hy4 rope position out of range");
        Ok([position; 3])
    }

    pub fn is_text(&self) -> bool { self.spans.is_empty() }
    pub fn delta(&self) -> i64 { self.spans.last().map_or(0, |s| s.delta_after) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contract() -> VisionContract {
        VisionContract { hidden: 1152, layers: 27, heads: 16, intermediate: 4304, output: 3072, merger_hidden: 4608,
            eps: 1e-5, image_token: 9, image_start: 7, image_end: 8, mrope_section: [12, 10, 10] }
    }

    #[test]
    fn transformers_docstring_layout_and_text_resume() {
        // One 4x4-patch image (2x2 merged + newline column): 2 rows x 3 cols.
        let c = contract();
        let mut tokens = vec![1, 2, 7];
        tokens.extend([9; 6]);
        tokens.extend([8, 3, 4]);
        let rope = RopePositions::new(&tokens, &[ImageSpan { start: 3, grid: [4, 4] }], &c).unwrap();
        let rows: Vec<_> = (0..tokens.len()).map(|r| rope.at(r).unwrap()).collect();
        assert_eq!(rows[..3], [[0; 3], [1; 3], [2; 3]]);
        assert_eq!(rows[3..9], [[3, 3, 3], [3, 3, 4], [3, 3, 5], [3, 4, 3], [3, 4, 4], [3, 4, 5]]);
        // Next text continues at base + max(2, 3) = 6.
        assert_eq!(rows[9], [6; 3]);
        assert_eq!(rope.at(40).unwrap(), [37; 3]);
        assert_eq!(rope.delta(), -3);
        let text = RopePositions::new(&[1, 2, 3], &[], &c).unwrap();
        assert!(text.is_text());
        assert_eq!(text.at(5).unwrap(), [5; 3]);
    }

    #[test]
    fn rejects_misaligned_spans_and_stray_placeholders() {
        let c = contract();
        let tokens = [7, 9, 9, 9, 9, 9, 9, 8];
        assert!(RopePositions::new(&tokens, &[ImageSpan { start: 1, grid: [4, 4] }], &c).is_ok());
        assert!(RopePositions::new(&tokens, &[ImageSpan { start: 1, grid: [2, 4] }], &c).is_err());
        assert!(RopePositions::new(&tokens, &[], &c).is_err());
        assert!(RopePositions::new(&tokens, &[ImageSpan { start: 1, grid: [3, 4] }], &c).is_err());
    }

    #[test]
    fn smart_resize_matches_sglang() {
        assert_eq!(smart_resize(512, 512).unwrap(), (512, 512));
        assert_eq!(smart_resize(480, 640).unwrap(), (480, 640));
        assert_eq!(smart_resize(4000, 3000).unwrap(), (2336, 1760));
        assert_eq!(smart_resize(10, 10).unwrap(), (64, 64));
        assert_eq!(smart_resize(48, 48).unwrap(), (64, 64));
        assert!(smart_resize(1, 400).is_err());
        assert_eq!(span_rows([128, 128]), Some(MAX_IMAGE_TOKENS));
        assert_eq!(span_rows([130, 2]), None);
    }

    #[test]
    fn real_config_and_header_pack() {
        let config: Value = serde_json::from_str(include_str!("fixtures/vision_config.json")).unwrap();
        let c = VisionContract::from_config(&config).unwrap();
        assert_eq!((c.image_token, c.image_start, c.image_end), (120120, 120118, 120119));
        let mut offset = 0u64;
        let tensors: Vec<_> = c.tensor_shapes().into_iter().map(|(name, shape)| {
            let bytes = shape.iter().product::<usize>() as u64 * 2;
            let meta = SafetensorsTensorMetadata { name, dtype: DType::Bf16, shape, byte_offset: offset, byte_length: bytes };
            offset += bytes;
            meta
        }).collect();
        assert_eq!(tensors.len(), 441);
        let packed = c.pack(tensors.iter()).unwrap();
        assert_eq!(packed.tensors.len(), 441);
        assert!(packed.tensors.iter().all(|t| t.destination % 256 == 0));
        // Header payload 0.869 GiB, plus alignment padding.
        assert!(packed.arena_bytes >= offset && packed.arena_bytes < offset + 441 * 256);
        let mut wrong = tensors.clone();
        wrong[0].dtype = DType::F32;
        assert!(c.pack(wrong.iter()).unwrap_err().to_string().contains("exporter/kernel"));
        let mut extra = tensors.clone();
        extra.push(SafetensorsTensorMetadata { name: "vit.extra".into(), ..tensors[1].clone() });
        assert!(c.pack(extra.iter()).is_err());
        let mut bad = config.clone();
        bad["vision_config"]["use_2d_rope"] = true.into();
        assert!(VisionContract::from_config(&bad).is_err());
        let mut sections = config.clone();
        sections["text_config"]["rope_parameters"]["sliding_attention"]["mrope_section"] = serde_json::json!([16, 8, 8]);
        assert!(VisionContract::from_config(&sections).is_err());
    }
}
