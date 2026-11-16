//! Processor-owned checkerboard colors and explicit alpha rounding.
use super::{DecodedPixels, MediaError, Result, RgbImage};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlendRounding {
    /// Historical integer alpha compositing, nearest with half rounded up.
    Nearest,
    /// Official NumPy float32 alpha operations, then uint8 truncation.
    Float32Truncate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Checkerboard {
    square: u32,
    white: u8,
    gray: u8,
    white_top_left: bool,
    rounding: BlendRounding,
}
impl Checkerboard {
    pub fn new(
        square: u32,
        white: u8,
        gray: u8,
        white_top_left: bool,
        rounding: BlendRounding,
    ) -> Result<Self> {
        if square == 0 {
            return Err(MediaError::Config(
                "checkerboard square size must be positive".into(),
            ));
        }
        Ok(Self {
            square,
            white,
            gray,
            white_top_left,
            rounding,
        })
    }

    /// Read the checkpoint's transparent_bg_config object. Color values are
    /// inputs, never inferred from the model family or a global decoder default.
    pub fn from_config(value: &serde_json::Value, rounding: BlendRounding) -> Result<Self> {
        let bad = |field: &str| MediaError::Config(format!("invalid checkerboard {field}"));
        if value["pattern"] != "chessboard" {
            return Err(bad("pattern"));
        }
        let square = value["chessboard_square_size"]
            .as_u64()
            .and_then(|n| u32::try_from(n).ok())
            .ok_or_else(|| bad("square size"))?;
        let color = |field: &str| {
            value[field]
                .as_u64()
                .and_then(|n| u8::try_from(n).ok())
                .ok_or_else(|| bad(field))
        };
        let top = value["chessboard_square_on_top_left"]
            .as_bool()
            .ok_or_else(|| bad("top-left color"))?;
        Self::new(
            square,
            color("chessboard_white_value")?,
            color("chessboard_gray_value")?,
            top,
            rounding,
        )
    }

    pub fn composite(self, pixels: DecodedPixels) -> Result<RgbImage> {
        match pixels {
            DecodedPixels::Rgb(image) => Ok(image),
            DecodedPixels::Rgba {
                width,
                height,
                data,
            } => {
                let expected = u64::from(width)
                    .checked_mul(u64::from(height))
                    .and_then(|n| n.checked_mul(4));
                if width == 0 || height == 0 || expected != Some(data.len() as u64) {
                    return Err(MediaError::Invalid(
                        "RGBA extent differs from dimensions".into(),
                    ));
                }
                let data = data
                    .chunks_exact(4)
                    .enumerate()
                    .flat_map(|(index, pixel)| {
                        let x = index % width as usize;
                        let y = index / width as usize;
                        let white = (x / self.square as usize + y / self.square as usize) % 2 == 0;
                        let background = if white == self.white_top_left {
                            self.white
                        } else {
                            self.gray
                        };
                        (0..3).map(move |channel| match self.rounding {
                            BlendRounding::Nearest => {
                                let alpha = u32::from(pixel[3]);
                                ((u32::from(pixel[channel]) * alpha
                                    + u32::from(background) * (255 - alpha)
                                    + 127)
                                    / 255) as u8
                            }
                            BlendRounding::Float32Truncate => {
                                // NumPy promotes uint8 RGB/background to float32 here.
                                // Separate operations preserve its non-FMA rounding.
                                let alpha = f32::from(pixel[3]) / 255.0;
                                let foreground = alpha * f32::from(pixel[channel]);
                                let remainder = 1.0f32 - alpha;
                                let back = remainder * f32::from(background);
                                (foreground + back) as u8
                            }
                        })
                    })
                    .collect();
                Ok(RgbImage {
                    width,
                    height,
                    data,
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn config() -> serde_json::Value {
        serde_json::json!({"pattern":"chessboard","chessboard_square_size":8,
            "chessboard_square_on_top_left":true,"chessboard_white_value":255,"chessboard_gray_value":180})
    }
    #[test]
    fn parameters_come_from_processor_and_rounding_is_explicit() {
        let p = Checkerboard::from_config(&config(), BlendRounding::Float32Truncate).unwrap();
        let image = DecodedPixels::Rgba {
            width: 2,
            height: 1,
            data: vec![10, 20, 30, 0, 100, 150, 200, 128],
        };
        assert_eq!(
            p.composite(image.clone()).unwrap().data,
            [255, 255, 255, 177, 202, 227]
        );
        let nearest = Checkerboard::new(8, 255, 180, true, BlendRounding::Nearest).unwrap();
        assert_eq!(
            nearest.composite(image).unwrap().data,
            [255, 255, 255, 177, 202, 227]
        );
        let mut cfg = config();
        cfg["chessboard_white_value"] = 204.into();
        let p = Checkerboard::from_config(&cfg, BlendRounding::Float32Truncate).unwrap();
        assert_eq!(
            p.composite(DecodedPixels::Rgba {
                width: 1,
                height: 1,
                data: vec![1, 2, 3, 128]
            })
            .unwrap()
            .data,
            [102, 102, 103]
        );
    }
    #[test]
    fn squares_and_top_left_color_are_not_fixed() {
        let p = Checkerboard::new(1, 250, 170, false, BlendRounding::Float32Truncate).unwrap();
        let image = DecodedPixels::Rgba {
            width: 2,
            height: 2,
            data: vec![0; 16],
        };
        assert_eq!(
            p.composite(image).unwrap().data,
            [170, 170, 170, 250, 250, 250, 250, 250, 250, 170, 170, 170]
        );
    }
    #[test]
    fn invalid_policy_or_storage_is_refused() {
        for (field, value) in [
            ("chessboard_square_size", serde_json::json!(0)),
            ("chessboard_gray_value", serde_json::json!(256)),
            ("chessboard_square_on_top_left", serde_json::json!(1)),
            ("pattern", serde_json::json!("white")),
        ] {
            let mut cfg = config();
            cfg[field] = value;
            assert!(Checkerboard::from_config(&cfg, BlendRounding::Float32Truncate).is_err());
        }
        let p = Checkerboard::new(8, 255, 180, true, BlendRounding::Float32Truncate).unwrap();
        assert!(p
            .composite(DecodedPixels::Rgba {
                width: 1,
                height: 1,
                data: vec![0; 3]
            })
            .is_err());
    }
}

#[cfg(test)]
mod oracle_tests {
    use super::*;
    use sha2::{Digest, Sha256};
    #[test]
    fn numpy_float32_alpha_all_values_and_colors_match_exact_bytes() {
        // NumPy2.1.0 official Kimi blend on every alpha/color pair, RGB repeated.
        for (color, expected) in [
            (
                255,
                "af1f7a5c38ec1113b47a968bd93444a02070873e46118f86db886cecb8d9067d",
            ),
            (
                180,
                "df6968f6419e5ab05a28208b0f8f368ea3bca4c9396960f4c7a97d06799c37fd",
            ),
        ] {
            let mut rgba = Vec::with_capacity(256 * 256 * 4);
            for alpha in 0..=255u8 {
                for value in 0..=255u8 {
                    rgba.extend_from_slice(&[value, value, value, alpha]);
                }
            }
            let p =
                Checkerboard::new(512, color, color, true, BlendRounding::Float32Truncate).unwrap();
            let rgb = p
                .composite(DecodedPixels::Rgba {
                    width: 256,
                    height: 256,
                    data: rgba,
                })
                .unwrap();
            assert_eq!(
                format!("{:x}", Sha256::digest(rgb.data)),
                expected,
                "background {color}"
            );
        }
    }
    #[test]
    fn pillow_rgba_resize_then_numpy_composite_matches_reference_digest() {
        // Pillow12.2.0 RGBA17x13 ->11x19 BICUBIC, checker255/180 at8px,
        // NumPy2.1.0 truncation. Source pattern spans alpha and every channel.
        let data = (0..17 * 13 * 4)
            .map(|i| ((i * 37 + 11) % 256) as u8)
            .collect();
        let resized = DecodedPixels::Rgba {
            width: 17,
            height: 13,
            data,
        }
        .resize(11, 19, super::super::Resample::Bicubic)
        .unwrap();
        let p = Checkerboard::new(8, 255, 180, true, BlendRounding::Float32Truncate).unwrap();
        let out = p.composite(resized).unwrap();
        assert_eq!(
            format!("{:x}", Sha256::digest(out.data)),
            "bce47f7470ae1165d3cd921552946e836bab2078c122c4e41c31c2f88c4cf8b8"
        );
    }
}
