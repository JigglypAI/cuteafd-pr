//! Exact scored-selector scratch for one immutable deployment capacity.
use serde_json::Value;
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScoredDecodeTile {
    prefix: u64,
    rows: u64,
    pages: u64,
}

#[derive(Debug, Error)]
#[error("invalid scored decode tile contract or admitted capacity")]
pub struct InvalidScoredDecodeTile;

/// Each rank owns one parity of index pages; partial final pages still count.
pub fn admitted_pages(context_tokens: u64, tokens_per_index_page: u64) -> u64 {
    assert!(context_tokens > 0 && tokens_per_index_page > 0);
    context_tokens.div_ceil(tokens_per_index_page).div_ceil(2)
}

impl ScoredDecodeTile {
    pub fn from_geometry(geometry: &Value) -> Result<Option<Self>, InvalidScoredDecodeTile> {
        let Some(tile) = geometry.get("admitted_tile") else { return Ok(None) };
        let number = |key: &str| tile[key].as_u64().filter(|&n| n > 0).ok_or(InvalidScoredDecodeTile);
        if number("page_rows")? != 64 || number("tile_rows")? != 32
            || number("tile_columns")? != 512 || number("element_bytes")? != 4 {
            return Err(InvalidScoredDecodeTile);
        }
        let out = Self { prefix: number("prefix_bytes")?, rows: number("capacity_rows")?,
            pages: number("capacity_pages")? };
        if out.prefix % 1024 != 0 { return Err(InvalidScoredDecodeTile) }
        out.bytes(out.pages)?;
        Ok(Some(out))
    }

    pub fn bytes(self, pages: u64) -> Result<u64, InvalidScoredDecodeTile> {
        if pages == 0 || pages > self.pages { return Err(InvalidScoredDecodeTile) }
        self.rows.div_ceil(32).checked_mul(32)
            .and_then(|rows| pages.checked_mul(64).map(|tokens| (rows, tokens)))
            .and_then(|(rows, tokens)| rows.checked_mul(tokens.div_ceil(512)))
            .and_then(|n| n.checked_mul(512 * 4))
            .and_then(|n| n.checked_add(self.prefix)).ok_or(InvalidScoredDecodeTile)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_deployment_tile_and_parity_rounding() {
        let geometry = serde_json::json!({"admitted_tile": {"prefix_bytes": 1051648,
            "capacity_rows": 64, "capacity_pages": 16384, "page_rows": 64,
            "tile_rows": 32, "tile_columns": 512, "element_bytes": 4}});
        let tile = ScoredDecodeTile::from_geometry(&geometry).unwrap().unwrap();
        assert_eq!(tile.bytes(16384).unwrap(), 269487104);
        assert_eq!(tile.bytes(admitted_pages(1048576, 64)).unwrap(), 135269376);
        assert_eq!(admitted_pages(129, 64), 2);
        assert_eq!(admitted_pages(1048576, 256), 2048);
        assert_eq!(tile.bytes(1).unwrap(), tile.bytes(8).unwrap());
        assert!(tile.bytes(0).is_err());
        assert!(tile.bytes(16385).is_err());
        let mut malformed = geometry.clone();
        malformed["admitted_tile"]["tile_columns"] = 256.into();
        assert!(ScoredDecodeTile::from_geometry(&malformed).is_err());
    }
}
