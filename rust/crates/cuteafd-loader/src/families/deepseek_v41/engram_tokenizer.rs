//! The checkpoint's tokenizer compression, using the same Rust normalizers as upstream.
use anyhow::{ensure, Context, Result};
use std::collections::HashMap;
use tokenizers::{NormalizedString, Normalizer, Tokenizer};

pub struct EngramTokenMap {
    ids: Vec<u32>,
    pad_id: u32,
}

impl EngramTokenMap {
    pub fn from_file(path: &std::path::Path) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(path)
            .map_err(|error| anyhow::anyhow!("loading engram tokenizer: {error}"))?;
        Self::from_tokenizer(&tokenizer)
    }

    pub fn from_tokenizer(tokenizer: &Tokenizer) -> Result<Self> {
        let ids = compressed_ids(tokenizer)?;
        ensure!(
            ids.len() == 129_280,
            "official V4.1 tokenizer must contain 129280 tokens"
        );
        ensure!(
            ids.iter().max() == Some(&99_091),
            "engram compressed vocabulary must contain 99092 tokens"
        );
        Ok(Self {
            pad_id: ids[2],
            ids,
        })
    }

    /// Prepare hashes immediately when token IDs are known, before model execution.
    pub fn prepare_batch(
        &self,
        history: &cuteafd_core::EngramHistory,
        start: u64,
        token_ids: &[u32],
        image_mask: Option<&[bool]>,
        max_rows: usize,
    ) -> Result<cuteafd_core::EngramBatch> {
        ensure!(
            token_ids.len() <= max_rows,
            "engram token batch exceeds row capacity"
        );
        ensure!(
            image_mask.is_none_or(|mask| mask.len() == token_ids.len()),
            "engram image mask length mismatch"
        );
        ensure!(
            history.pad_id() == self.pad_id,
            "engram history uses a different compressed padding ID"
        );
        let compressed: Result<Vec<_>> = token_ids
            .iter()
            .enumerate()
            .map(|(row, &token)| self.compress(token, image_mask.is_some_and(|mask| mask[row])))
            .collect();
        Ok(history.prepare(start, &compressed?, max_rows)?)
    }

    /// Rebuild bounded native-token history at the replay start, preserving image barriers.
    pub(crate) fn history_at(&self, position: u64, recent: &[u32], image_mask: Option<&[u8]>) -> Result<cuteafd_core::EngramHistory> {
        ensure!(recent.len() == position.min(3) as usize
            && image_mask.is_none_or(|mask| mask.len() == recent.len()),
            "engram resume token or image lookback differs");
        let compressed = recent.iter().enumerate().map(|(i, &token)|
            self.compress(token, image_mask.is_some_and(|mask| mask[i] != 0)))
            .collect::<Result<Vec<_>>>()?;
        Ok(cuteafd_core::EngramHistory::from_recent(self.pad_id(), position, &compressed)?)
    }

    pub fn pad_id(&self) -> u32 {
        self.pad_id
    }

    pub fn compressed_ids(&self) -> &[u32] {
        &self.ids
    }

    /// None denotes an image-span token, including image delimiters and newlines.
    pub fn compress(&self, token: u32, is_image: bool) -> Result<Option<u32>> {
        let id = *self
            .ids
            .get(token as usize)
            .context("engram tokenizer ID out of range")?;
        Ok((!is_image).then_some(id))
    }
}

fn compressed_ids(tokenizer: &Tokenizer) -> Result<Vec<u32>> {
    let normalizer: tokenizers::normalizers::NormalizerWrapper =
        serde_json::from_value(serde_json::json!({
            "type": "Sequence", "normalizers": [
                {"type":"NFKC"}, {"type":"NFD"}, {"type":"StripAccents"}, {"type":"Lowercase"},
                {"type":"Replace", "pattern":{"Regex":"[ \\t\\r\\n]+"}, "content":" "},
                {"type":"Replace", "pattern":{"Regex":"^ $"}, "content":"\u{e000}"},
                {"type":"Strip", "strip_left":true, "strip_right":true},
                {"type":"Replace", "pattern":{"String":"\u{e000}"}, "content":" "}
            ]
        }))?;
    let mut keys = HashMap::<String, u32>::new();
    let mut ids = Vec::with_capacity(tokenizer.get_vocab_size(true));
    for token in 0..tokenizer.get_vocab_size(true) {
        let raw = tokenizer
            .id_to_token(token as u32)
            .context("tokenizer vocabulary has an ID hole")?;
        let text = tokenizer
            .decode(&[token as u32], false)
            .map_err(|error| anyhow::anyhow!("{error}"))?;
        let key = if text.contains('\u{fffd}') {
            raw
        } else {
            let mut normalized = NormalizedString::from(text.as_str());
            normalizer
                .normalize(&mut normalized)
                .map_err(|error| anyhow::anyhow!("{error}"))?;
            if normalized.get().is_empty() {
                text
            } else {
                normalized.get().to_owned()
            }
        };
        let next = keys.len() as u32;
        ids.push(*keys.entry(key).or_insert(next));
    }
    Ok(ids)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn replay_lookback_rebuilds_full_engram_hashes_at_image_boundaries() -> Result<()> {
        let map = EngramTokenMap { ids: (0..100).map(|id| id / 2).collect(), pad_id: 1 };
        let tokens: Vec<_> = (0..600).map(|i| (i % 100) as u32).collect();
        for image_start in [101, 102] {
            let mask: Vec<_> = (0..600).map(|i| (image_start..image_start + 184).contains(&i)).collect();
            let mut full = cuteafd_core::EngramHistory::new(map.pad_id())?;
            for start in 0..tokens.len() {
                let recent = start.saturating_sub(3);
                let resumed = map.history_at(start as u64, &tokens[recent..start],
                    Some(&mask[recent..start].iter().copied().map(u8::from).collect::<Vec<_>>()))?;
                let end = (start + 128).min(tokens.len());
                let expected = map.prepare_batch(&full, start as u64, &tokens[start..end], Some(&mask[start..end]), 128)?;
                let actual = map.prepare_batch(&resumed, start as u64, &tokens[start..end], Some(&mask[start..end]), 128)?;
                assert_eq!(actual.hashes(), expected.hashes(), "image {image_start}, replay {start}");
                assert_eq!(resumed.lookback(), full.lookback());
                full.commit(&expected, 1)?;
            }
        }
        assert!(map.history_at(100, &[1, 2], None).is_err());
        assert!(map.history_at(100, &[1, 2, 3], Some(&[0])).is_err());
        Ok(())
    }

    #[test]
    fn compression_preserves_spaces_and_normalizes_accents() -> Result<()> {
        let vocab: tokenizers::models::wordlevel::WordLevel = serde_json::from_str(&serde_json::json!({
            "type":"WordLevel", "vocab":{"[UNK]":0," The":1,"THE":2,"thé":3," ":4,"":5,"\t":6}, "unk_token":"[UNK]"
        }).to_string())?;
        let tokenizer = Tokenizer::new(vocab);
        let ids = compressed_ids(&tokenizer)?;
        assert_eq!(ids[1], ids[2]);
        assert_eq!(ids[2], ids[3]);
        assert_ne!(ids[4], ids[5]);
        assert_eq!(ids[4], ids[6]);
        assert!(EngramTokenMap::from_tokenizer(&tokenizer).is_err());
        Ok(())
    }
}
