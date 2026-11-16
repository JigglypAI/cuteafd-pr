//! Immutable constants share the tower arena and the coordinator push lifetime.
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, sync::Arc};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ConstantProvenance {
    pub source: String,
    pub recipe: String,
    pub content_sha256: [u8; 32],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ConstantRegion {
    pub name: String,
    pub offset: u64,
    pub alignment: u64,
    pub bytes: Arc<[u8]>,
    pub provenance: ConstantProvenance,
}

impl ConstantRegion {
    pub(crate) fn new(
        name: impl Into<String>,
        offset: u64,
        alignment: u64,
        bytes: impl Into<Arc<[u8]>>,
        source: impl Into<String>,
        recipe: impl Into<String>,
    ) -> Self {
        let bytes = bytes.into();
        let content_sha256 = Sha256::digest(&bytes).into();
        Self {
            name: name.into(),
            offset,
            alignment,
            bytes,
            provenance: ConstantProvenance {
                source: source.into(),
                recipe: recipe.into(),
                content_sha256,
            },
        }
    }

    pub(crate) fn fp32(
        name: &str,
        offset: u64,
        values: impl IntoIterator<Item = f32>,
        recipe: &str,
        source: &str,
    ) -> Self {
        let bytes = values
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        Self::new(name, offset, 4, bytes, source, recipe)
    }

    pub(crate) fn validate(&self, arena_bytes: u64) -> Result<(), String> {
        if self.name.is_empty()
            || self.alignment == 0
            || !self.alignment.is_power_of_two()
            || self.offset % self.alignment != 0
            || self.bytes.is_empty()
            || self
                .offset
                .checked_add(self.bytes.len() as u64)
                .is_none_or(|end| end > arena_bytes)
            || self.provenance.source.is_empty()
            || self.provenance.recipe.is_empty()
            || <[u8; 32]>::from(Sha256::digest(&self.bytes)) != self.provenance.content_sha256
        {
            return Err(format!(
                "invalid or corrupted tower constant region {}",
                self.name
            ));
        }
        Ok(())
    }
}

/// Validate one physical destination set, including external tensor writes.
pub(crate) fn validate_regions(
    regions: &[ConstantRegion],
    arena_bytes: u64,
    reads: impl IntoIterator<Item = (u64, u64)>,
) -> Result<(), String> {
    let mut names = BTreeSet::new();
    let mut ranges = Vec::new();
    for region in regions {
        region.validate(arena_bytes)?;
        if !names.insert(region.name.as_str()) {
            return Err(format!("duplicate tower constant name {}", region.name));
        }
        ranges.push((region.offset, region.offset + region.bytes.len() as u64));
    }
    for (start, bytes) in reads {
        let end = start
            .checked_add(bytes)
            .ok_or("tower tensor destination overflow")?;
        if bytes == 0 || end > arena_bytes {
            return Err("tower tensor outside admitted arena".into());
        }
        ranges.push((start, end));
    }
    ranges.sort_unstable();
    if ranges.windows(2).any(|w| w[0].1 > w[1].0) {
        return Err("overlapping tower constants or tensor destinations".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn region() -> ConstantRegion {
        ConstantRegion::new(
            "rotary.h",
            256,
            256,
            vec![1, 2, 3, 4],
            "vendor-sha",
            "FP32 arange/pow",
        )
    }
    #[test]
    fn immutable_regions_validate_and_clone_without_copying_payload() {
        let region = region();
        let copy = region.clone();
        assert!(Arc::ptr_eq(&region.bytes, &copy.bytes));
        assert_eq!(
            region.provenance.content_sha256,
            <[u8; 32]>::from(Sha256::digest([1, 2, 3, 4]))
        );
        assert!(validate_regions(&[region], 512, [(0, 256)]).is_ok());
    }
    #[test]
    fn malformed_constants_names_hashes_and_destination_sets_are_refused() {
        for fault in 0..9 {
            let mut r = region();
            match fault {
                0 => r.name.clear(),
                1 => r.alignment = 3,
                2 => r.offset = 257,
                3 => r.offset = u64::MAX,
                4 => r.bytes = Arc::from([]),
                5 => r.provenance.content_sha256[0] ^= 1,
                6 => r.provenance.source.clear(),
                7 => r.provenance.recipe.clear(),
                _ => r.alignment = 0,
            }
            assert!(r.validate(512).is_err(), "fault {fault}");
        }
        let first = region();
        assert!(validate_regions(&[first.clone(), first.clone()], 512, []).is_err());
        let mut second = first.clone();
        second.name = "rotary.w".into();
        assert!(validate_regions(&[first.clone(), second], 512, []).is_err());
        assert!(validate_regions(&[first], 512, [(258, 4)]).is_err());
    }
}

#[cfg(test)]
pub(super) fn assert_original_table(spec: &super::TowerSpec) {
    let n = &spec.native;
    let expected: Vec<f32> = if n.perception.is_some() {
        use cuteafd_loader::media::perception_vision::PerceptionKind;
        match n.reserved {
            6 => PerceptionKind::Inkling2.inv_freq(),
            7 => PerceptionKind::Step5.inv_freq(),
            _ => panic!("PE kind"),
        }
    } else if n.reserved == super::mistral::RESERVED {
        let base = (0..64)
            .map(|i| 1.0 / 10000f32.powf((2 * i) as f32 / 128f32))
            .collect::<Vec<_>>();
        (0..64)
            .map(|i| {
                if i < 32 {
                    base[2 * i]
                } else {
                    base[2 * (i - 32) + 1]
                }
            })
            .collect()
    } else {
        let count = if n.abi_version == 2 {
            n.head_dim as usize / 4
        } else {
            16
        };
        (0..count)
            .map(|i| 1.0f32 / 10000f32.powf(i as f32 / count as f32))
            .collect()
    };
    let expected = expected
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    assert_eq!(spec.constant_regions().len(), 1);
    let region = &spec.constant_regions()[0];
    assert_eq!(region.offset, n.inv_freq);
    assert_eq!(&*region.bytes, expected.as_slice());
    assert_eq!(region.offset + region.bytes.len() as u64, n.weight_bytes);
    region.validate(n.weight_bytes).unwrap();
}
