use super::*;
use cuteafd_loader::media::{EncoderId, ImageFamily, ImageProcessor, ProcessorConfig, RgbImage};

#[test]
fn fixture_input_is_exact_across_encoder_architectures_for_mistral_qwen_and_mimo() {
    for family in [ImageFamily::Mistral, ImageFamily::Qwen, ImageFamily::Mimo] {
        let processor = ProcessorConfig::for_family(family);
        let rgb = RgbImage { width: 64, height: 64, data: vec![37; 64 * 64 * 3] };
        let resident = processor.prepare_rgb(rgb.clone(), EncoderId([120; 32])).unwrap();
        let mut remote = processor.prepare_rgb(rgb, EncoderId([121; 32])).unwrap();
        assert_ne!(resident.key, remote.key);
        assert_eq!(resident.grid, remote.grid);
        assert_eq!(resident.rgb8, remote.rgb8);
        if family == ImageFamily::Mistral {
            assert_eq!(resident.tokens, 12);
            assert_eq!([resident.grid.t, resident.grid.h, resident.grid.w], [1, 6, 6]);
        }
        let span = ProbeMedia { start: 1, len: resident.tokens, kind: "image".into(),
            key: hex(&resident.key.0), grid: [resident.grid.t, resident.grid.h, resident.grid.w],
            identity: Some(ProbeImageIdentity::prepared(&resident)), input_key: None,
            fixture: None, image_url: None };
        span.verify_image(&resident).unwrap();
        span.verify_image(&remote).unwrap();
        assert_ne!(ProbeImageIdentity::prepared(&resident), ProbeImageIdentity::prepared(&remote));
        let mut changed = remote.rgb8.to_vec(); changed[0] ^= 1; remote.rgb8 = changed.into();
        assert!(span.verify_image(&remote).is_err());
        remote.rgb8 = resident.rgb8.clone(); remote.processor.0[0] ^= 1;
        assert!(span.verify_image(&remote).is_err());
        let mut legacy = span.clone(); legacy.identity = None;
        legacy.verify_image(&resident).unwrap();
        assert!(legacy.verify_image(&remote).is_err());
    }
}

#[test]
fn padded_kimi_identity_rebinds_encoder_without_losing_content_geometry() {
    use cuteafd_loader::media::{ImageGrid, PreparedImage, PreprocessId};
    let pixels = cuteafd_core::media::ImagePixels { source:[17,13], canvas:[28,28], content:[0,0,17,13] };
    let mut image = PreparedImage { key:cuteafd_loader::media::ImageKey([0;32]), encoder:EncoderId([120;32]),
        processor:PreprocessId([1;32]), grid:ImageGrid {t:1,h:2,w:2}, rgb8:vec![0;28*28*3].into(), pixels,tokens:1 };
    image.key=PreparedImage::key_for_pixels(image.encoder,image.processor,image.grid,&image.rgb8,pixels).unwrap();
    let span=ProbeMedia {start:1,len:1,kind:"image".into(),key:hex(&image.key.0),grid:[1,2,2],
        identity:Some(ProbeImageIdentity::prepared(&image)),input_key:None,fixture:None,image_url:None};
    span.verify_image(&image).unwrap();
    image.encoder=EncoderId([121;32]);
    image.key=PreparedImage::key_for_pixels(image.encoder,image.processor,image.grid,&image.rgb8,pixels).unwrap();
    span.verify_image(&image).unwrap();
    image.pixels.content[2]-=1;
    assert!(span.verify_image(&image).is_err());
}
