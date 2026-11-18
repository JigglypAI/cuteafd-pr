//! Attention operand precision, independent of weights and KV storage.

/// Operand precision of both attention QK and PV products (FP32 accumulation).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum AttentionPrecision {
    #[default]
    Bf16,
    Fp8,
}

impl AttentionPrecision {
    pub const fn name(self) -> &'static str {
        match self { Self::Bf16 => "bf16", Self::Fp8 => "fp8" }
    }
}

impl std::fmt::Display for AttentionPrecision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str(self.name()) }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0:?}: expected bf16 or fp8")]
pub struct AttentionPrecisionError(String);

impl std::str::FromStr for AttentionPrecision {
    type Err = AttentionPrecisionError;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "bf16" => Ok(Self::Bf16),
            "fp8" => Ok(Self::Fp8),
            other => Err(AttentionPrecisionError(other.into())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_default_never_depends_on_checkpoint_format() {
        assert_eq!(AttentionPrecision::default(), AttentionPrecision::Bf16);
        for mode in [AttentionPrecision::Bf16, AttentionPrecision::Fp8] {
            assert_eq!(mode.to_string().parse(), Ok(mode));
            assert_eq!(serde_json::to_string(&mode).unwrap(), format!("\"{mode}\""));
        }
        for removed in ["auto", "", "BF16", "bf16-qk/fp8-pv"] {
            assert!(removed.parse::<AttentionPrecision>().is_err(), "{removed}");
        }
    }
}
