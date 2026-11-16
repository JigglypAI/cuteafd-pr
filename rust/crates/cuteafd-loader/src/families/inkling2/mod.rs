//! Inkling 2 Small checkpoint facts. Execution uses the shared engine contracts.
pub mod admission;
pub mod config;
pub mod experts;
pub mod execution;
pub mod weight_layout;
pub mod verify;
pub mod conv;
pub use config::{Inkling2Config, Inkling2MtpConfig, Inkling2VisionConfig};

/// Global KV only; local rings and convolution marks retain their native precision.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InklingKvCache { Bf16, #[default] Fp8 }
impl InklingKvCache {
    pub fn record_bytes(self, heads: usize, dim: usize) -> usize {
        2 * heads * dim * if self == Self::Bf16 { 2 } else { 1 }
            + if self == Self::Fp8 { 8 * heads } else { 0 }
    }
    pub fn name(self) -> &'static str { match self { Self::Bf16 => "bf16", Self::Fp8 => "fp8" } }
    pub fn record_format(self) -> &'static str { match self {
        Self::Bf16 => "inkling2_bf16_kv_f32_conv_whole_v1",
        Self::Fp8 => "inkling2_fp8_head_scales_kv_f32_conv_whole_v2",
    } }
}
impl std::str::FromStr for InklingKvCache {
    type Err = String;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value { "bf16" => Ok(Self::Bf16), "fp8" => Ok(Self::Fp8),
            _ => Err("Inkling KV format must be bf16 or fp8".into()) }
    }
}
