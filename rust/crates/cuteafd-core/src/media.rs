//! Shared media identity types. Native model tokens never contain these keys.

use serde::{Deserialize, Serialize};

/// The full SHA-256 identity of an image, including encoder and preprocessing.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct ImageKey(pub [u8; 32]);

/// The full SHA-256 identity of canonical PCM, preprocessing and its audio tower.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct AudioKey(pub [u8; 32]);

/// Modality is part of cache and prefix identity, even when digest bytes coincide.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum MediaKey {
    Image(ImageKey),
    Audio(AudioKey),
}
impl MediaKey {
    pub fn bytes(&self) -> &[u8; 32] {
        match self {
            Self::Image(key) => &key.0,
            Self::Audio(key) => &key.0,
        }
    }
    pub fn is_audio(self) -> bool {
        matches!(self, Self::Audio(_))
    }
}
impl From<ImageKey> for MediaKey {
    fn from(key: ImageKey) -> Self {
        Self::Image(key)
    }
}
impl From<AudioKey> for MediaKey {
    fn from(key: AudioKey) -> Self {
        Self::Audio(key)
    }
}
impl std::hash::Hash for MediaKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Preserve existing image page identities; audio alone adds a domain tag.
        match self {
            Self::Image(key) => key.hash(state),
            Self::Audio(key) => {
                b"cuteafd.audio.v1".hash(state);
                key.hash(state);
            }
        }
    }
}
#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum SerializedMediaKey {
    Image(ImageKey),
    Audio { audio: AudioKey },
}
impl Serialize for MediaKey {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Image(key) => SerializedMediaKey::Image(*key),
            Self::Audio(key) => SerializedMediaKey::Audio { audio: *key },
        }
        .serialize(serializer)
    }
}
impl<'de> Deserialize<'de> for MediaKey {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(match SerializedMediaKey::deserialize(deserializer)? {
            SerializedMediaKey::Image(key) => Self::Image(key),
            SerializedMediaKey::Audio { audio } => Self::Audio(audio),
        })
    }
}

/// Pixel geometry independent of the encoder's patch grid. Source dimensions
/// remain available to prompt templates after resize/crop; content marks valid
/// RGB inside a canvas whose padding can have a distinct normalized value.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ImagePixels {
    pub source: [u32; 2], // width, height before preprocessing
    pub canvas: [u32; 2],
    pub content: [u32; 4], // x, y, width, height in canvas pixels
}
impl ImagePixels {
    pub fn full(width: u32, height: u32) -> Self {
        Self {
            source: [width, height],
            canvas: [width, height],
            content: [0, 0, width, height],
        }
    }
    pub fn valid(self) -> bool {
        let [width, height] = self.canvas;
        let [x, y, w, h] = self.content;
        !self.source.contains(&0)
            && width > 0
            && height > 0
            && w > 0
            && h > 0
            && x.checked_add(w).is_some_and(|end| end <= width)
            && y.checked_add(h).is_some_and(|end| end <= height)
            && u64::from(width) * u64::from(height) <= 64 << 20
    }
    pub fn padded(self) -> bool {
        self.content != [0, 0, self.canvas[0], self.canvas[1]]
    }
}

/// Merge-2 image rows shared by processors, encoders and probe admission.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ImageTokenGeometry {
    pub merge: u32,
    pub row_markers: u32,
}
impl ImageTokenGeometry {
    pub const MERGED: Self = Self {
        merge: 2,
        row_markers: 0,
    };
    pub const ROW_MARKERS: Self = Self {
        merge: 2,
        row_markers: 1,
    };

    pub fn extent(self, grid: [u32; 3]) -> Option<(usize, usize, usize)> {
        let [t, h, w] = grid;
        if t != 1
            || self.merge == 0
            || self.row_markers > 1
            || h == 0
            || w == 0
            || h % self.merge != 0
            || w % self.merge != 0
        {
            return None;
        }
        let (h, w) = ((h / self.merge) as usize, (w / self.merge) as usize);
        Some((
            h,
            w,
            h.checked_mul(w.checked_add(self.row_markers as usize)?)?,
        ))
    }
}

/// One image owner's declared limits, shared by local and remote admission.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ImageEncodeLimits {
    pub max_patches: u32,
    pub max_output_rows: u32,
    pub max_canvas_pixels: u64,
    pub geometry: ImageTokenGeometry,
}
impl ImageEncodeLimits {
    pub fn validate(
        self,
        grid: [u32; 3],
        pixels: ImagePixels,
        rows: u64,
        bytes: u64,
    ) -> Result<(), &'static str> {
        if self.max_patches == 0
            || self.max_patches > 65536
            || self.max_output_rows == 0
            || self.max_output_rows > 16384
            || self.max_canvas_pixels == 0
            || self.max_canvas_pixels > 64 << 20
        {
            return Err("invalid image owner capacity");
        }
        let patches = u64::from(grid[1]) * u64::from(grid[2]);
        let canvas = u64::from(pixels.canvas[0]) * u64::from(pixels.canvas[1]);
        let expected = self
            .geometry
            .extent(grid)
            .ok_or("invalid still image patch grid")?
            .2 as u64;
        if !pixels.valid()
            || patches > u64::from(self.max_patches)
            || rows == 0
            || rows != expected
            || rows > u64::from(self.max_output_rows)
            || canvas > self.max_canvas_pixels
            || canvas.checked_mul(3) != Some(bytes)
        {
            return Err("image source/grid/output exceeds declared capacity");
        }
        Ok(())
    }
}

/// The encoder's PCM-to-feature geometry, independent of request batching.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub enum AudioTokenGeometry {
    #[default]
    MimoRvq,
    /// Causal patch stages produce one feature per product of patch strides.
    PcmPatches { samples_per_token: usize },
}
impl AudioTokenGeometry {
    pub fn is_mimo(&self) -> bool {
        *self == Self::MimoRvq
    }
    pub fn tokens(self, samples: usize) -> Option<usize> {
        if samples == 0 || samples > 7_200_000 {
            return None;
        }
        match self {
            Self::MimoRvq => {
                if samples <= 480 {
                    return None;
                }
                let frames = samples / 240 + 1;
                let codes = (frames / 6000) * 1500 + (frames % 6000).div_ceil(2).div_ceil(2);
                Some(codes.div_ceil(4))
            }
            Self::PcmPatches {
                samples_per_token: 0,
            } => None,
            Self::PcmPatches { samples_per_token } => Some(samples.div_ceil(samples_per_token)),
        }
    }
}

/// A contiguous span of media feature rows in the expanded native prompt.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct MediaSpan {
    pub start: usize,
    pub len: usize,
    pub key: MediaKey,
}

impl MediaSpan {
    pub fn checked_end(&self) -> Option<usize> {
        self.start.checked_add(self.len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_content_extent_preserves_source_and_rejects_overflow() {
        let mut pixels = ImagePixels::full(640, 480);
        assert!(pixels.valid());
        assert!(!pixels.padded());
        pixels.source = [1234, 777];
        pixels.content = [0, 0, 639, 479];
        assert!(pixels.valid());
        assert!(pixels.padded());
        for content in [
            [0, 0, 0, 1],
            [0, 0, 641, 480],
            [u32::MAX, 0, 2, 1],
            [0, u32::MAX, 1, 2],
        ] {
            pixels.content = content;
            assert!(!pixels.valid());
        }
    }

    #[test]
    fn image_geometry_preserves_merged_families_and_row_markers() {
        for h in (2..=110).step_by(2) {
            for w in (2..=110).step_by(2) {
                assert_eq!(
                    ImageTokenGeometry::MERGED.extent([1, h, w]),
                    Some(((h / 2) as usize, (w / 2) as usize, (h * w / 4) as usize))
                );
                assert_eq!(
                    ImageTokenGeometry::ROW_MARKERS.extent([1, h, w]),
                    Some((
                        (h / 2) as usize,
                        (w / 2) as usize,
                        (h / 2 * (w / 2 + 1)) as usize
                    ))
                );
            }
        }
        for grid in [[0, 4, 4], [2, 4, 4], [1, 0, 4], [1, 3, 4], [1, 4, 3]] {
            assert_eq!(ImageTokenGeometry::MERGED.extent(grid), None);
            assert_eq!(ImageTokenGeometry::ROW_MARKERS.extent(grid), None);
        }
    }

    #[test]
    fn audio_patch_geometry_matches_all_stage_stride_boundaries() {
        let patch = AudioTokenGeometry::PcmPatches {
            samples_per_token: 16 * 6 * 5 * 4,
        };
        for samples in [1, 16, 96, 480, 481, 1919, 1920, 1921, 24000, 7_200_000] {
            assert_eq!(patch.tokens(samples), Some(samples.div_ceil(1920)));
        }
        for invalid in [0, 7_200_001, usize::MAX] {
            assert_eq!(patch.tokens(invalid), None);
        }
        assert_eq!(
            AudioTokenGeometry::PcmPatches {
                samples_per_token: 0
            }
            .tokens(1),
            None
        );
        assert_eq!(AudioTokenGeometry::MimoRvq.tokens(480), None);
        assert_eq!(AudioTokenGeometry::MimoRvq.tokens(24000), Some(7));
    }

    #[test]
    fn full_key_and_span_round_trip() {
        let span = MediaSpan {
            start: 7,
            len: 4,
            key: ImageKey([0xab; 32]).into(),
        };
        let json = serde_json::to_vec(&span).unwrap();
        assert_eq!(serde_json::from_slice::<MediaSpan>(&json).unwrap(), span);
        assert_eq!(span.checked_end(), Some(11));
    }

    #[test]
    fn image_wire_identity_stays_legacy_and_audio_is_domain_separated() {
        use std::hash::{Hash, Hasher};
        let image = ImageKey([0xab; 32]);
        let audio = MediaKey::Audio(AudioKey(image.0));
        assert_eq!(
            serde_json::to_vec(&MediaKey::Image(image)).unwrap(),
            serde_json::to_vec(&image).unwrap()
        );
        assert_eq!(
            serde_json::from_slice::<MediaKey>(&serde_json::to_vec(&audio).unwrap()).unwrap(),
            audio
        );
        let hash = |value: &dyn Fn(&mut std::collections::hash_map::DefaultHasher)| {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            value(&mut h);
            h.finish()
        };
        assert_eq!(
            hash(&|h| image.hash(h)),
            hash(&|h| MediaKey::Image(image).hash(h))
        );
        assert_ne!(hash(&|h| audio.hash(h)), hash(&|h| image.hash(h)));
    }

    #[test]
    fn overflowing_span_is_not_a_valid_frontier() {
        let span = MediaSpan {
            start: usize::MAX,
            len: 1,
            key: ImageKey([0; 32]).into(),
        };
        assert_eq!(span.checked_end(), None);
    }
}
