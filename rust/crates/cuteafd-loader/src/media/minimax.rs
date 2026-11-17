//! Supported SGLang M3.1 default: Pillow RGB8 bicubic, ceil28/max672 rectangle.
use super::{DecodedPixels, ImageFamily, MediaError, ProcessorConfig, Resample, Result, RgbImage};
use std::path::Path;

pub(super) fn processor_config(snapshot: &Path) -> Result<ProcessorConfig> {
    let config = crate::read_json_file(&snapshot.join("config.json")).map_err(|e| MediaError::Config(e.to_string()))?;
    VisionContract::from_config(&config).map_err(|e| MediaError::Config(e.to_string()))?;
    let raw: serde_json::Value = crate::read_json_file(&snapshot.join("preprocessor_config.json"))
        .map_err(|e| MediaError::Config(e.to_string()))?;
    let expected = ProcessorConfig::for_family(ImageFamily::MiniMax);
    if raw["patch_size"] != 14 || raw["size"] != serde_json::json!([672,672])
        || raw["image_mean"] != serde_json::json!(expected.mean)
        || raw["image_std"] != serde_json::json!(expected.std)
        || raw["img_token_compression_config"]["spatial_merge_size"] != 2
        || raw["img_token_compression_config"]["temporal_patch_size"] != 2 {
        return Err(MediaError::Config("unsupported MiniMax SGLang processor geometry/numerics".into()));
    }
    expected.validate()?;
    Ok(expected)
}

pub(super) fn resize_shape(height: u32, width: u32) -> Result<(u32,u32)> {
    crate::families::minimax_m3::vision::resize_shape(height,width)
        .map_err(|e| MediaError::Config(e.to_string()))
}
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
    fn registered_groups_keep_supported_geometry_and_checked_feature_spans() {
        use super::super::{EncoderId,ImageProcessor,ImagePromptLayout,PreparedImageGroup};
        let config=ProcessorConfig::for_family(ImageFamily::MiniMax);
        let encoder=EncoderId([9;32]);
        for (height,width) in [(13,17),(432,640),(640,432),(672,672),(1009,731)] {
            let data=(0..height*width*3).map(|i| ((i*37+11)%256) as u8).collect();
            let mut group=config.prepare_group(DecodedPixels::Rgb(RgbImage {width,height,data}),encoder).unwrap();
            assert!(group.fragments.is_empty());
            assert_eq!(group.source,[width,height]);
            let expected=resize_shape(height,width).unwrap();
            let image=&group.images[0];
            assert_eq!([image.grid.h*14,image.grid.w*14],[expected.0,expected.1]);
            assert!(image.tokens<=576);
            assert_eq!(image.processor,config.id());
            config.assemble_group_prompt(&mut group,&ImagePromptLayout::Single {placeholder:200025}).unwrap();
            let rows=group.feature_rows().unwrap();
            let expanded=PreparedImageGroup::expand_prompt(&[1,200025,2],200025,&[group],200064,600).unwrap();
            assert_eq!(expanded.tokens.len(),rows+2);
            assert_eq!(expanded.media.len(),1);
        }
        let mut bounded=config.clone(); bounded.max_image_tokens=1;
        assert!(bounded.prepare_rgb(RgbImage {width:56,height:56,data:vec![0;56*56*3]},encoder).is_err());
        let mut wrong=config;wrong.std[0]=0.5;assert!(wrong.validate().is_err());
    }
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
            let hwc_expected = match (h, w) {
                (13, 17) => "5600054d347417aad6031db4ef68dab5c6539dcfb95e19a5f37fc171765940cf",
                (432, 640) => "095b9344ce7046df713d388dfed0af6202bc7458a03fec7541c8a46ff1dfa8ff",
                (640, 432) => "11d4bb0a9afa3cbd7477075dd705ae46ec9bcc6297655e6c0bcc5d9527ce3c4e",
                (672, 672) => "97dac130616a513af1612a23731fffea294709a21eacec6dcca72686b2f50ac5",
                (1009, 731) => "99f7fccbaa0f266f284a8eb9671159e532505a27ad467cfd88eb66e876aaf3e7",
                _ => unreachable!("pinned oracle cases"),
            };
            assert_eq!(
                format!("{:x}", Sha256::digest(&out.data)),
                hwc_expected,
                "PillowHWC source{h}x{w}"
            );
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
