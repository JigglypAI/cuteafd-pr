//! Official Kimi still processor: downscale only, Pillow RGBA resize, checkerboard,
//! black right/bottom padding to28px. Native encoder receives one raster frame.
use super::{
    invalid, BlendRounding, Checkerboard, DecodedPixels, EncoderId, ImageGrid, MediaError,
    PreparedImage, PreprocessId, Resample, Result,
};
use sha2::{Digest, Sha256};
use std::path::Path;

#[derive(Clone, Debug)]
pub struct KimiProcessor {
    patch: u32,
    merge: u32,
    patch_limit: u64,
    side_limit: u32,
    checkerboard: Checkerboard,
    identity: PreprocessId,
}
impl KimiProcessor {
    pub fn from_snapshot(snapshot: &Path) -> Result<Self> {
        let raw = std::fs::read(snapshot.join("preprocessor_config.json"))
            .map_err(|e| MediaError::Config(e.to_string()))?;
        let config: serde_json::Value =
            serde_json::from_slice(&raw).map_err(|e| MediaError::Config(e.to_string()))?;
        Self::from_config(&config)
    }
    pub fn from_config(config: &serde_json::Value) -> Result<Self> {
        let v = &config["media_proc_cfg"];
        if v["patch_size"] != 14
            || v["merge_kernel_size"] != 2
            || v["in_patch_limit"] != 65536
            || v["patch_limit_on_one_side"] != 512
            || !v["fixed_output_tokens"].is_null()
            || v["transparent_bg_fill_stage"] != "after_resize"
            || v["image_mean"] != serde_json::json!([0.5, 0.5, 0.5])
            || v["image_std"] != serde_json::json!([0.5, 0.5, 0.5])
        {
            return Err(MediaError::Config(
                "unsupported Kimi still-image processor geometry/numerics".into(),
            ));
        }
        let checkerboard =
            Checkerboard::from_config(&v["transparent_bg_config"], BlendRounding::Float32Truncate)?;
        let identity = serde_json::json!({"recipe":"kimi-still-v1:scale-down-truncate:pillow-bicubic-rgba:checkerboard-f32-truncate:black-pad:row-major",
            "patch":v["patch_size"],"merge":v["merge_kernel_size"],"patch_limit":v["in_patch_limit"],
            "side_limit":v["patch_limit_on_one_side"],"mean":v["image_mean"],"std":v["image_std"],
            "alpha":v["transparent_bg_config"],"alpha_stage":v["transparent_bg_fill_stage"]});
        let identity = PreprocessId(
            Sha256::digest(
                serde_json::to_vec(&identity).map_err(|e| MediaError::Config(e.to_string()))?,
            )
            .into(),
        );
        Ok(Self {
            patch: 14,
            merge: 2,
            patch_limit: 65536,
            side_limit: 512,
            checkerboard,
            identity,
        })
    }

    /// Width,height returned separately for valid content and padded canvas.
    pub fn geometry(&self, width: u32, height: u32) -> Result<([u32; 2], [u32; 2])> {
        if width == 0 || height == 0 {
            return Err(invalid("Kimi image dimensions must be positive"));
        }
        let w = f64::from(width);
        let h = f64::from(height);
        let patch = f64::from(self.patch);
        let count =
            f64::from((width / self.patch).max(1)) * f64::from((height / self.patch).max(1));
        let scale = 1f64
            .min((self.patch_limit as f64 / count).sqrt())
            .min(f64::from(self.side_limit) * patch / w)
            .min(f64::from(self.side_limit) * patch / h);
        let max_side = self.side_limit * self.patch;
        let resized = [
            ((w * scale) as u32).max(1).min(max_side),
            ((h * scale) as u32).max(1).min(max_side),
        ];
        let factor = self.patch * self.merge;
        let canvas = [
            resized[0].div_ceil(factor) * factor,
            resized[1].div_ceil(factor) * factor,
        ];
        Ok((resized, canvas))
    }

    pub fn prepare(
        &self,
        pixels: DecodedPixels,
        encoder: EncoderId,
        max_tokens: usize,
    ) -> Result<PreparedImage> {
        let source = pixels.dimensions();
        let (content, canvas) = self.geometry(source[0], source[1])?;
        let patches = u64::from(canvas[0] / self.patch) * u64::from(canvas[1] / self.patch);
        let tokens = usize::try_from(patches / u64::from(self.merge * self.merge))
            .map_err(|_| invalid("Kimi token count overflow"))?;
        if patches > self.patch_limit || tokens > max_tokens || max_tokens == 0 {
            return Err(invalid(
                "Kimi padded image exceeds admitted encoder capacity",
            ));
        }
        let resized = pixels.resize(content[0], content[1], Resample::Bicubic)?;
        let rgb = self.checkerboard.composite(resized)?;
        let bytes = u64::from(canvas[0]) * u64::from(canvas[1]) * 3;
        let mut padded =
            vec![0; usize::try_from(bytes).map_err(|_| invalid("Kimi canvas extent overflow"))?];
        for row in 0..content[1] as usize {
            let src = row * content[0] as usize * 3;
            let dst = row * canvas[0] as usize * 3;
            padded[dst..dst + content[0] as usize * 3]
                .copy_from_slice(&rgb.data[src..src + content[0] as usize * 3]);
        }
        let geometry = cuteafd_core::media::ImagePixels {
            source,
            canvas,
            content: [0, 0, content[0], content[1]],
        };
        let grid = ImageGrid {
            t: 1,
            h: canvas[1] / self.patch,
            w: canvas[0] / self.patch,
        };
        let key = PreparedImage::key_for_pixels(encoder, self.identity, grid, &padded, geometry)?;
        Ok(PreparedImage {
            key,
            encoder,
            processor: self.identity,
            grid,
            rgb8: padded.into(),
            pixels: geometry,
            tokens,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::media::RgbImage;
    fn config() -> serde_json::Value {
        serde_json::json!({"media_proc_cfg":{"patch_size":14,"merge_kernel_size":2,"in_patch_limit":65536,
            "patch_limit_on_one_side":512,"fixed_output_tokens":null,"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],
            "transparent_bg_fill_stage":"after_resize","transparent_bg_config":{"pattern":"chessboard",
                "chessboard_square_size":8,"chessboard_square_on_top_left":true,"chessboard_white_value":255,"chessboard_gray_value":180}}})
    }
    #[test]
    fn original_dimensions_content_and_black_padding_are_preserved() {
        let p = KimiProcessor::from_config(&config()).unwrap();
        let prepared = p
            .prepare(
                DecodedPixels::Rgb(RgbImage {
                    width: 17,
                    height: 13,
                    data: vec![42; 17 * 13 * 3],
                }),
                EncoderId([1; 32]),
                16384,
            )
            .unwrap();
        assert_eq!(prepared.pixels.source, [17, 13]);
        assert_eq!(prepared.pixels.canvas, [28, 28]);
        assert_eq!(prepared.pixels.content, [0, 0, 17, 13]);
        assert_eq!(prepared.tokens, 1);
        assert_eq!(prepared.grid, ImageGrid { t: 1, h: 2, w: 2 });
        for y in 0..28 {
            for x in 0..28 {
                assert_eq!(
                    prepared.rgb8[(y * 28 + x) * 3],
                    if y < 13 && x < 17 { 42 } else { 0 }
                );
            }
        }
    }
    #[test]
    fn rgba_checkerboard_uses_checkpoint_colors_after_resize() {
        let p = KimiProcessor::from_config(&config()).unwrap();
        let prepared = p
            .prepare(
                DecodedPixels::Rgba {
                    width: 2,
                    height: 1,
                    data: vec![10, 20, 30, 0, 100, 150, 200, 128],
                },
                EncoderId([1; 32]),
                16384,
            )
            .unwrap();
        assert_eq!(&prepared.rgb8[..6], &[255, 255, 255, 177, 202, 227]);
        assert!(p
            .prepare(
                DecodedPixels::Rgb(RgbImage {
                    width: 29,
                    height: 29,
                    data: vec![0; 29 * 29 * 3]
                }),
                EncoderId([1; 32]),
                1
            )
            .is_err());
    }
    #[test]
    fn official_processor_rgb8_hwc_oracle_matches_alpha_resize_and_padding() {
        let p = KimiProcessor::from_config(&config()).unwrap();
        for (h, w, alpha, expected) in [
            (
                13,
                17,
                false,
                "f3dcf2fff5f6813dbfbc645814faaf8b9cdfb709e718346ee19b1b98d28df36e",
            ),
            (
                432,
                640,
                false,
                "10e80e03d9abe308eb9638068d5344dd100e8c786179e8711fa5915ab7012a6b",
            ),
            (
                13,
                17,
                true,
                "97465b5504df6b0a661b9f2ce9394dcc281190b5abcb90e99930d4d9dd6e1741",
            ),
            (
                129,
                7201,
                true,
                "88ea6153c07f3c0640c1fc18c4546911d356e911236b7ad94bb621fd309e294f",
            ),
            (
                28,
                28,
                false,
                "33bb432d0a5684cce3434961207fa233e0000443017af202ce066ddb071b60db",
            ),
        ] {
            let channels = if alpha { 4 } else { 3 };
            let data = (0..h * w * channels)
                .map(|index| ((index * 37 + 11) % 256) as u8)
                .collect();
            let decoded = if alpha {
                DecodedPixels::Rgba {
                    width: w,
                    height: h,
                    data,
                }
            } else {
                DecodedPixels::Rgb(RgbImage {
                    width: w,
                    height: h,
                    data,
                })
            };
            let prepared = p.prepare(decoded, EncoderId([1; 32]), 16384).unwrap();
            assert_eq!(
                format!("{:x}", Sha256::digest(&prepared.rgb8)),
                expected,
                "HWC source{h}x{w} alpha{alpha}"
            );
        }
    }
    #[test]
    fn official_padding_boundaries_are_preserved_before_admission() {
        let p = KimiProcessor::from_config(&config()).unwrap();
        assert_eq!(
            p.geometry(3584, 3584).unwrap(),
            ([3584, 3584], [3584, 3584])
        );
        assert_eq!(
            p.geometry(3585, 3585).unwrap(),
            ([3585, 3585], [3612, 3612])
        );
        assert_eq!(
            p.geometry(3598, 3598).unwrap(),
            ([3584, 3584], [3584, 3584])
        );
        assert_eq!(
            p.geometry(7069, 1833).unwrap(),
            ([7069, 1833], [7084, 1848])
        );
        // Shared ceilings remain nominal until the full boundary encoder runs.
        assert_eq!(7084u64 / 14 * (1848 / 14), 66792);
    }
    #[test]
    fn config_numerics_and_identity_are_explicit() {
        let a = KimiProcessor::from_config(&config()).unwrap();
        let mut cfg = config();
        cfg["media_proc_cfg"]["transparent_bg_config"]["chessboard_gray_value"] = 179.into();
        let b = KimiProcessor::from_config(&cfg).unwrap();
        assert_ne!(a.identity, b.identity);
        cfg["media_proc_cfg"]["transparent_bg_fill_stage"] = "before_resize".into();
        assert!(KimiProcessor::from_config(&cfg).is_err());
        assert!(a.geometry(0, 1).is_err());
    }
}
