//! Inkling's official still-image processor preserves original pixels, composites
//! alpha on checkerboard, then pads raw sentinel-1 cells to40px without resize.
//! RGB8 carries an arbitrary pad value; valid-content metadata selects sentinel
//! normalization in the native owner, and participates in the cache identity.
use super::{invalid,DecodedPixels,EncoderId,ImageGrid,PreparedImage,PreprocessId,Result};
use sha2::{Digest,Sha256};

pub fn prepare(pixels:DecodedPixels,encoder:EncoderId,max_tokens:usize) -> Result<PreparedImage> {
    let source=pixels.dimensions();
    if source.contains(&0) || u64::from(source[0])*u64::from(source[1])>64<<20 || max_tokens==0 {
        return Err(invalid("Inkling image source/capacity invalid"));
    }
    let width=source[0].div_ceil(40).checked_mul(40).ok_or_else(|| invalid("Inkling padded width overflow"))?;
    let height=source[1].div_ceil(40).checked_mul(40).ok_or_else(|| invalid("Inkling padded height overflow"))?;
    let tokens=(width as usize/40).checked_mul(height as usize/40).ok_or_else(|| invalid("Inkling token count overflow"))?;
    if tokens>max_tokens || tokens*4+1>16384 {return Err(invalid("Inkling original image exceeds admitted native tower capacity; no resize is permitted"));}
    let rgb=pixels.checkerboard();
    if rgb.data.len()!=source[0] as usize*source[1] as usize*3 {return Err(invalid("Inkling RGB extent differs from source"));}
    let mut canvas=vec![0;width as usize*height as usize*3];
    for row in 0..source[1] as usize {
        let destination=row*width as usize*3;
        canvas[destination..destination+source[0] as usize*3].copy_from_slice(&rgb.data[row*source[0] as usize*3..(row+1)*source[0] as usize*3]);
    }
    let pixels=cuteafd_core::media::ImagePixels {source,canvas:[width,height],content:[0,0,source[0],source[1]]};
    let grid=ImageGrid {t:1,h:height/20,w:width/20};
    let mut hash=Sha256::new();
    hash.update(b"inkling2-pe-still-v1:original-pixels:checkerboard8:pad-1:merge2:clip-bf16-pe-remap");
    hash.update(40u32.to_le_bytes());let processor=PreprocessId(hash.finalize().into());
    let key=PreparedImage::key_for_pixels(encoder,processor,grid,&canvas,pixels)?;
    Ok(PreparedImage {key,encoder,processor,grid,rgb8:canvas.into(),pixels,tokens})
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::media::RgbImage;
    #[test]
    fn preserves_source_pixels_and_distinguishes_padding_from_black() {
        let rgb=RgbImage {width:41,height:39,data:(0..41*39*3).map(|i|(i%256) as u8).collect()};
        let prepared=prepare(DecodedPixels::Rgb(rgb.clone()),EncoderId([1;32]),4096).unwrap();
        assert_eq!(prepared.pixels.source,[41,39]);assert_eq!(prepared.pixels.canvas,[80,40]);
        assert_eq!(prepared.pixels.content,[0,0,41,39]);assert_eq!(prepared.tokens,2);
        for row in 0..39 {assert_eq!(&prepared.rgb8[row*80*3..row*80*3+41*3],&rgb.data[row*41*3..(row+1)*41*3]);}
        let black=PreparedImage::key_for(prepared.encoder,prepared.processor,prepared.grid,&prepared.rgb8);
        assert_ne!(black,prepared.key);
        assert!(prepare(DecodedPixels::Rgb(rgb),EncoderId([1;32]),1).is_err());
    }
    #[test]
    fn alpha_composite_precedes_sentinel_padding_without_resize() {
        let image=DecodedPixels::Rgba {width:2,height:1,data:vec![10,20,30,0,100,150,200,128]};
        let prepared=prepare(image,EncoderId([1;32]),4096).unwrap();
        assert_eq!(&prepared.rgb8[..6],&[204,204,204,152,177,202]);
        assert_eq!(prepared.pixels.content,[0,0,2,1]);
    }
}
