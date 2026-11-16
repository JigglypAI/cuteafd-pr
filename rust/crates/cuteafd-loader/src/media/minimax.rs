//! Supported SGLang M3.1 default: Pillow RGB8 bicubic, ceil28/max672 rectangle.
use super::{DecodedPixels, Resample, Result, RgbImage};
use crate::families::minimax_m3::vision::VisionContract;

/// No request-specific max_pixels policy: the API rejects that override before
/// reaching this processor. Preserve source dimensions separately for job metadata.
pub fn resize_default(pixels: DecodedPixels, contract: &VisionContract) -> Result<RgbImage> {
    let [width, height] = pixels.dimensions();
    let (out_h, out_w) = contract
        .resize_shape(height, width)
        .map_err(|error| super::MediaError::Config(error.to_string()))?;
    // Supported SGLang default converts RGB before its PIL resize, no alpha
    // checkerboard. This reuses the shared Pillow byte-exact CPU implementation.
    let rgb = pixels.into_rgb(super::AlphaPolicy::Drop);
    match DecodedPixels::Rgb(rgb).resize(out_w, out_h, Resample::Bicubic)? {
        DecodedPixels::Rgb(image) => Ok(image),
        _ => unreachable!("RGB resize remains RGB"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    #[test]
    fn supported_sglang_default_five_images_match_pillow_chw_byte_hashes() {
        let cfg = crate::families::minimax_m3::config::tests::release_config();
        let contract = VisionContract::from_config(&cfg).unwrap();
        for (h, w, expected) in [
            (
                13,
                17,
                "7280ba6f6dd56dace7427fc091941cc77f30da4b84e473b34825060762e29e85",
            ),
            (
                432,
                640,
                "4560c80aee2672c72f425373181e0c9276e309f0f50b21c0033e9d092eeb3eb1",
            ),
            (
                640,
                432,
                "d7878c49f62fd4c6fdca825414b8bd26b11ed114eda5f4cb8310143da3e7009b",
            ),
            (
                672,
                672,
                "5e53d6b887eee9c588571b3fdd38e77f2090e0656c2e6d803f75a816fe7d26b3",
            ),
            (
                1009,
                731,
                "0716f2252afcefd968e944ee24e660322424e321bb01f92e8af28a18c3c0f71e",
            ),
        ] {
            let data = (0..h * w * 3)
                .map(|index| ((index * 37 + 11) % 256) as u8)
                .collect();
            let out = resize_default(
                DecodedPixels::Rgb(RgbImage {
                    width: w,
                    height: h,
                    data,
                }),
                &contract,
            )
            .unwrap();
            // SGLang's resize_frames returns CHW uint8; the Rust payload is HWC.
            // Hash an explicit CHW view on both sides, not differently laid-out bytes.
            let chw = (0..3)
                .flat_map(|channel| out.data.chunks_exact(3).map(move |pixel| pixel[channel]))
                .collect::<Vec<_>>();
            assert_eq!(
                format!("{:x}", Sha256::digest(chw)),
                expected,
                "source{h}x{w}"
            );
        }
    }
}
