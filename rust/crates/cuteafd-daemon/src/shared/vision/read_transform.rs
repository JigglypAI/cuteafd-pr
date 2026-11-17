//! Explicit checkpoint-to-resident conversions; temporary source bytes never
//! become a second weight backing. Push and mount share these exact boundaries.
use super::{DType, Result, SafetensorsTensorMetadata, VisionError};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReadTransform {
    Direct,
    Bf16ToF32,
    F32ToBf16,
}
impl ReadTransform {
    pub(crate) fn resident_bytes(self, metadata: &SafetensorsTensorMetadata) -> Result<usize> {
        let bytes = usize::try_from(metadata.byte_length)
            .map_err(|_| VisionError::Unsupported("media source extent overflow".into()))?;
        let invalid =
            || VisionError::Unsupported("media read conversion dtype/extent mismatch".into());
        if bytes == 0 {
            return Err(invalid());
        }
        match self {
            Self::Direct => Ok(bytes),
            Self::Bf16ToF32 if metadata.dtype == DType::Bf16 && bytes % 2 == 0 => {
                bytes.checked_mul(2).ok_or_else(invalid)
            }
            Self::F32ToBf16 if metadata.dtype == DType::F32 && bytes % 4 == 0 => Ok(bytes / 2),
            _ => Err(invalid()),
        }
    }
    pub(crate) fn source_alignment(self) -> usize {
        match self {
            Self::F32ToBf16 => 4,
            _ => 2,
        }
    }
    pub(crate) fn convert(self, source: &[u8], out: &mut [u8]) -> Result<()> {
        let invalid = || {
            VisionError::Unsupported("media conversion source/destination extent mismatch".into())
        };
        match self {
            Self::Direct if source.len() == out.len() => out.copy_from_slice(source),
            Self::Bf16ToF32
                if source.len() % 2 == 0 && source.len().checked_mul(2) == Some(out.len()) =>
            {
                for (input, output) in source.chunks_exact(2).zip(out.chunks_exact_mut(4)) {
                    output.copy_from_slice(
                        &(u32::from(u16::from_le_bytes([input[0], input[1]])) << 16).to_le_bytes(),
                    );
                }
            }
            Self::F32ToBf16 if source.len() % 4 == 0 && source.len() / 2 == out.len() => {
                for (input, output) in source.chunks_exact(4).zip(out.chunks_exact_mut(2)) {
                    let bits = u32::from_le_bytes(input.try_into().unwrap());
                    // Match the existing push load_quant::narrow_f32 exactly:
                    // RNE for finite values, signed quiet top-payload NaNs.
                    let narrowed = if bits & 0x7fff_ffff > 0x7f80_0000 {
                        ((bits >> 16) | 0x40) as u16
                    } else {
                        ((bits.wrapping_add(0x7fff + ((bits >> 16) & 1))) >> 16) as u16
                    };
                    output.copy_from_slice(&narrowed.to_le_bytes());
                }
            }
            _ => return Err(invalid()),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn legacy_bf16_widen_preserves_all65536_bit_patterns() {
        let source = (0..=u16::MAX)
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>();
        let mut actual = vec![0; source.len() * 2];
        ReadTransform::Bf16ToF32
            .convert(&source, &mut actual)
            .unwrap();
        let expected = (0..=u16::MAX)
            .flat_map(|bits| f32::from_bits(u32::from(bits) << 16).to_le_bytes())
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }
    #[test]
    fn fp32_narrow_rounds_ties_even_and_has_no_resident_source_copy() {
        let bits = [
            0x3f808000u32,
            0x3f818000,
            0xbf808000,
            0x7f800000,
            0xff800000,
            0x7fc12345,
        ];
        let source = bits
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        let mut actual = vec![0; 12];
        ReadTransform::F32ToBf16
            .convert(&source, &mut actual)
            .unwrap();
        assert_eq!(
            actual,
            [0x3f80u16, 0x3f82, 0xbf80, 0x7f80, 0xff80, 0x7fc1]
                .into_iter()
                .flat_map(u16::to_le_bytes)
                .collect::<Vec<_>>()
        );
        assert!(ReadTransform::F32ToBf16
            .convert(&source[..3], &mut actual)
            .is_err());
    }
}
