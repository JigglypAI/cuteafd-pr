//! Hy4-flash (`hy_v4_flash`, `HYV4FlashForConditionalGeneration`) checkpoint facts,
//! FP8-only Engram mapped-table reader and multi-layer look-ahead, packed
//! coordinator weight arenas, and shared-contract execution layouts.
pub mod config;
pub mod engram;
pub mod execution;
pub mod weight_layout;

pub use config::{Hy4DflyConfig, Hy4FlashConfig, Hy4VisionConfig};
pub use engram::{
    advance_tail, decode_fp8_row, decode_fp8_rows_to_bf16_bytes, Hy4EngramHasher,
    Hy4EngramHistory, Hy4EngramLookahead, Hy4EngramShape, Hy4EngramTable,
    ENGRAM_DECODED_TOKEN_BYTES, ENGRAM_EMBED_WIDTH, ENGRAM_FP8_ROW_BYTES, ENGRAM_HEADS,
};

/// Paged KV cache precision for the 12 DSA layers (576-dim absorbed MLA record:
/// 512 normalized latent + 64 RoPE key). The 30 SWA-512 layers keep fixed
/// per-sequence circular rings of the same record width.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Hy4KvCache {
    #[default]
    Bf16,
    Fp8,
}

impl Hy4KvCache {
    pub const fn record_bytes(self) -> usize {
        match self {
            // 576 BF16 elements (512 latent + 64 RoPE).
            Self::Bf16 => 1152,
            // 512 E4M3 latent + 4x F32 block-128 scales (16 B) + 64 BF16 RoPE (128 B).
            Self::Fp8 => 656,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Fp8 => "fp8",
        }
    }
}

impl std::str::FromStr for Hy4KvCache {
    type Err = String;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "bf16" => Ok(Self::Bf16),
            "fp8" => Ok(Self::Fp8),
            _ => Err("Hy4-flash KV format must be bf16 or fp8".into()),
        }
    }
}
