//! Shared attention QK/PV precision, independent of core and KV storage.
use super::PlanError;
pub use cuteafd_core::attention::AttentionPrecision;

pub const OPTION: &str = "--attention-precision";

/// The attention operand modes provided by a family's programs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrecisionSupport {
    pub modes: &'static [AttentionPrecision],
    pub source: &'static str,
}

pub const BF16_ONLY: PrecisionSupport = PrecisionSupport {
    modes: &[AttentionPrecision::Bf16],
    source: "BF16 attention; no FP8 attention program is exported (#2146)",
};
pub const MLA: PrecisionSupport = PrecisionSupport {
    modes: &[AttentionPrecision::Bf16, AttentionPrecision::Fp8],
    source: "shared MLA BF16 QK/PV default; FP8 attention is opt-in (#2146)",
};

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct Resolved {
    pub mode: AttentionPrecision,
    pub reason: String,
}

pub fn resolve(family: &dyn super::Family, requested: AttentionPrecision) -> Result<Resolved, PlanError> {
    let support = family.attention_precision();
    if support.modes.contains(&requested) {
        Ok(Resolved { mode: requested, reason: support.source.into() })
    } else {
        Err(PlanError::UnsupportedValue {
            option: OPTION, value: requested.name().into(), family: family.id().into(),
            reason: format!("no {} attention program is exported (available: {}); {}", requested.name(),
                support.modes.iter().map(|mode| mode.name()).collect::<Vec<_>>().join(", "), support.source),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_family_defaults_to_bf16_regardless_of_checkpoint_view() {
        for family in crate::plan::family::registry() {
            let resolved = resolve(*family, AttentionPrecision::default()).unwrap();
            assert_eq!(resolved.mode, AttentionPrecision::Bf16, "{}", family.id());
        }
    }

    #[test]
    fn unsupported_fp8_is_refused_not_silently_replaced() {
        for family in crate::plan::family::registry() {
            let result = resolve(*family, AttentionPrecision::Fp8);
            if family.attention_precision().modes.contains(&AttentionPrecision::Fp8) {
                assert_eq!(result.unwrap().mode, AttentionPrecision::Fp8);
            } else {
                let error = result.unwrap_err().to_string();
                assert!(error.contains(OPTION) && error.contains("fp8") && error.contains(family.id()), "{error}");
            }
        }
    }
}
